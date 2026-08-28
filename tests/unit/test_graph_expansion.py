"""Behaviour tests for GraphIndex retrieval expansion.

``_expand_graph_neighbors`` used to pull every direct child of every
referenced node, unconditionally and after the top_k cut, so a single hit
on a well-cross-referenced document could return dozens of full section
bodies. These tests pin the bounded replacement:

- children are opt-in (``include_children``) and capped per parent
- the child bodies are stood in for by one document outline per document
  where a reference was actually followed
- the synthetic scores expansion assigns stay strictly ordered at the
  score scale hybrid search actually produces (RRF, ~0.008-0.016), not
  just at cosine scale

The store is faked at the class boundary — these are unit tests for the
expansion logic, not for the SQL, which is exercised by the integration
suite.
"""

from __future__ import annotations

import pytest
from agentic.knowledge.models import RetrievedItem

from agentic_project_service.services import knowledge_search

KB_ID = "kb-1"
TOC_A = "toc-a"
TOC_B = "toc-b"


# ---------------------------------------------------------------------------
# Fakes + builders
# ---------------------------------------------------------------------------


def _hit(node_id: str, *, toc_id: str = TOC_A, refs: list[str] | None = None, score: float = 0.9):
    """A search result as the retrieval pipeline hands it to expansion."""
    return RetrievedItem(
        item_id=f"row-{toc_id}-{node_id}",
        text=f"body of {node_id}",
        score=score,
        source_id="src-A",
        knowledge_base_id=KB_ID,
        meta={"toc_id": toc_id, "node_id": node_id, "referenced_nodes": refs or []},
    )


def _node_row(node_id: str, *, toc_id: str = TOC_A, parent: str | None = None, refs=None):
    """A graph_index_nodes row as the store returns it."""
    return {
        "id": f"row-{toc_id}-{node_id}",
        "toc_id": toc_id,
        "node_id": node_id,
        "title": f"Section {node_id}",
        "text": f"body of {node_id}",
        "depth": 0 if parent is None else 1,
        "parent_node_id": parent,
        "meta": {"referenced_nodes": refs or []},
        "source_id": "src-A",
    }


class FakeGraphIndexStore:
    """Stands in for GraphIndexStore, recording what expansion asked for."""

    def __init__(self, nodes=None, children=None, outlines=None):
        self._nodes = nodes or {}
        self._children = children or {}
        self._outlines = outlines or {}
        self.children_calls: list[list] = []
        self.outline_calls: list[list] = []

    def get_nodes_by_ids(self, selections):
        return {key: self._nodes[key] for key in selections if key in self._nodes}

    def get_children_by_parent_ids(self, parent_selections):
        self.children_calls.append(list(parent_selections))
        return {key: self._children[key] for key in parent_selections if key in self._children}

    def get_toc_outline(self, toc_ids):
        self.outline_calls.append(list(toc_ids))
        return {tid: self._outlines[tid] for tid in toc_ids if tid in self._outlines}


@pytest.fixture
def install_store(monkeypatch):
    """Install a FakeGraphIndexStore for the duration of one test."""

    def _install(store):
        monkeypatch.setattr(
            knowledge_search,
            "GraphIndexStore",
            lambda db_session, knowledge_base_id: store,
        )
        return store

    return _install


def _outline(toc_id=TOC_A, nodes=None):
    return {
        "doc_name": "Master Services Agreement",
        "source_id": "src-A",
        "nodes": nodes
        or [
            {"node_id": "0001", "title": "Definitions", "depth": 0},
            {"node_id": "0002", "title": "Indemnification", "depth": 0},
            {"node_id": "0003", "title": "Scope of Indemnity", "depth": 1},
        ],
    }


def _expand(results, store, retrieval_config=None):
    return knowledge_search._expand_graph_neighbors(
        db_session=None,
        results=results,
        knowledge_base_id=KB_ID,
        retrieval_config=retrieval_config,
    )


def _by_method(items, method):
    return [it for it in items if (it.meta or {}).get("retrieval_method") == method]


# ---------------------------------------------------------------------------
# Children fan-out
# ---------------------------------------------------------------------------


class TestChildrenFanOut:
    def test_children_are_not_expanded_by_default(self, install_store):
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                children={(TOC_A, "0002"): [_node_row("0003", parent="0002")]},
                outlines={TOC_A: _outline()},
            )
        )

        out = _expand([_hit("0001", refs=["0002"])], store)

        assert _by_method(out, "graph_expansion"), "referenced node should still be pulled in"
        assert _by_method(out, "graph_expansion_child") == []

    def test_children_are_expanded_when_enabled(self, install_store):
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                children={(TOC_A, "0002"): [_node_row("0003", parent="0002")]},
                outlines={TOC_A: _outline()},
            )
        )

        out = _expand(
            [_hit("0001", refs=["0002"])],
            store,
            {"graph_expansion": {"include_children": True}},
        )

        children = _by_method(out, "graph_expansion_child")
        assert [c.meta["node_id"] for c in children] == ["0003"]

    def test_children_are_capped_per_parent_in_document_order(self, install_store):
        five = [_node_row(f"000{i}", parent="0002") for i in range(3, 8)]
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                children={(TOC_A, "0002"): five},
                outlines={TOC_A: _outline()},
            )
        )

        out = _expand(
            [_hit("0001", refs=["0002"])],
            store,
            {"graph_expansion": {"include_children": True, "max_children_per_parent": 2}},
        )

        children = _by_method(out, "graph_expansion_child")
        assert [c.meta["node_id"] for c in children] == ["0003", "0004"]

    def test_children_are_not_queried_when_disabled(self, install_store):
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                children={(TOC_A, "0002"): [_node_row("0003", parent="0002")]},
                outlines={TOC_A: _outline()},
            )
        )

        _expand([_hit("0001", refs=["0002"])], store)

        assert store.children_calls == []


# ---------------------------------------------------------------------------
# Document outline
# ---------------------------------------------------------------------------


class TestDocumentOutline:
    def test_outline_emitted_once_per_document_with_a_followed_reference(self, install_store):
        store = install_store(
            FakeGraphIndexStore(
                nodes={
                    (TOC_A, "0002"): _node_row("0002"),
                    (TOC_A, "0003"): _node_row("0003"),
                },
                outlines={TOC_A: _outline()},
            )
        )

        out = _expand(
            [_hit("0001", refs=["0002"]), _hit("0004", refs=["0003"])],
            store,
        )

        outlines = _by_method(out, "graph_toc")
        assert len(outlines) == 1
        assert outlines[0].meta["toc_id"] == TOC_A

    def test_no_outline_when_no_reference_was_followed(self, install_store):
        store = install_store(FakeGraphIndexStore(outlines={TOC_A: _outline()}))

        out = _expand([_hit("0001", refs=[])], store)

        assert _by_method(out, "graph_toc") == []

    def test_outline_only_for_documents_where_a_reference_was_followed(self, install_store):
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                outlines={TOC_A: _outline(), TOC_B: _outline(TOC_B)},
            )
        )

        out = _expand(
            [_hit("0001", refs=["0002"]), _hit("0009", toc_id=TOC_B, refs=[])],
            store,
        )

        assert [o.meta["toc_id"] for o in _by_method(out, "graph_toc")] == [TOC_A]

    def test_outline_can_be_disabled(self, install_store):
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                outlines={TOC_A: _outline()},
            )
        )

        out = _expand(
            [_hit("0001", refs=["0002"])],
            store,
            {"graph_expansion": {"include_doc_toc": False}},
        )

        assert _by_method(out, "graph_toc") == []

    def test_outline_renders_ids_and_indents_by_depth(self, install_store):
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                outlines={TOC_A: _outline()},
            )
        )

        out = _expand([_hit("0001", refs=["0002"])], store)

        text = _by_method(out, "graph_toc")[0].text
        assert "[0001] Definitions" in text
        assert "  [0003] Scope of Indemnity" in text

    def test_long_outline_is_truncated_with_a_marker(self, install_store):
        limit = knowledge_search.GRAPH_TOC_MAX_NODES
        many = [
            {"node_id": f"{i:04d}", "title": f"Section {i}", "depth": 0}
            for i in range(1, limit + 6)
        ]
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                outlines={TOC_A: _outline(nodes=many)},
            )
        )

        out = _expand([_hit("0001", refs=["0002"])], store)

        text = _by_method(out, "graph_toc")[0].text
        assert len(text.splitlines()) == limit + 1
        assert "5 more sections" in text


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestExpansionScores:
    def test_tiers_stay_ordered_at_rrf_score_scale(self, install_store):
        """Hybrid search fuses with RRF, whose scores run ~1/(60+rank) — an order
        of magnitude below cosine. A fixed 0.01 decrement swallows that scale
        whole and collapses every tier onto the same clamped value."""
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                children={(TOC_A, "0002"): [_node_row("0003", parent="0002")]},
                outlines={TOC_A: _outline()},
            )
        )
        pool = [_hit("0001", refs=["0002"], score=0.0164), _hit("0005", score=0.0077)]

        out = _expand(pool, store, {"graph_expansion": {"include_children": True}})

        toc = _by_method(out, "graph_toc")[0].score
        parent = _by_method(out, "graph_expansion")[0].score
        child = _by_method(out, "graph_expansion_child")[0].score
        assert child < parent < toc < 0.0077
        assert child > 0, "tiers must stay on the pool's scale, not clamp to zero"
