"""Behaviour tests for GraphIndex retrieval expansion.

``_expand_graph_neighbors`` used to pull every direct child of every
referenced node, unconditionally and after the top_k cut, so a single hit
on a well-cross-referenced document could return dozens of full section
bodies. These tests pin the bounded replacement:

- children are opt-in (``include_children``) and capped per parent
- the child bodies are stood in for by one document outline per document
  where a reference was actually followed
- the config block is read defensively: it is unvalidated JSONB, so a
  malformed value must degrade to defaults rather than take the search down
- the synthetic scores stay strictly ordered even when the pool's own
  minimum is 0.0, which hybrid and full-text searches both reach

The store is faked at the class boundary — these are unit tests for the
expansion logic. ``get_toc_outline``'s SQL is covered separately, against a
real database, in tests/test_graph_index_store.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

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

    def __init__(self, nodes=None, children=None, outlines=None, outline_error=None):
        self._nodes = nodes or {}
        self._children = children or {}
        self._outlines = outlines or {}
        self._outline_error = outline_error
        self.children_calls: list[list] = []
        self.outline_calls: list[list] = []

    def get_nodes_by_ids(self, selections):
        return {key: self._nodes[key] for key in selections if key in self._nodes}

    def get_children_by_parent_ids(self, parent_selections):
        self.children_calls.append(list(parent_selections))
        return {key: self._children[key] for key in parent_selections if key in self._children}

    def get_toc_outline(self, toc_ids, limit):
        self.outline_calls.append(list(toc_ids))
        if self._outline_error:
            raise self._outline_error
        return {
            tid: {**self._outlines[tid], "nodes": self._outlines[tid]["nodes"][:limit]}
            for tid in toc_ids
            if tid in self._outlines
        }


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


def _outline(toc_id=TOC_A, nodes=None, total_nodes=None):
    resolved = (
        nodes
        if nodes is not None
        else [
            {"node_id": "0001", "title": "Definitions", "depth": 0},
            {"node_id": "0002", "title": "Indemnification", "depth": 0},
            {"node_id": "0003", "title": "Scope of Indemnity", "depth": 1},
        ]
    )
    return {
        "doc_name": "Master Services Agreement",
        "source_id": "src-A",
        "nodes": resolved,
        "total_nodes": total_nodes if total_nodes is not None else len(resolved),
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


def _store_with_ref(**kwargs):
    """The common shape: one hit referencing 0002, which has children."""
    return FakeGraphIndexStore(
        nodes={(TOC_A, "0002"): _node_row("0002")},
        children={(TOC_A, "0002"): [_node_row("0003", parent="0002")]},
        outlines={TOC_A: _outline()},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Config parsing — retrieval_config is unvalidated JSONB
# ---------------------------------------------------------------------------


class TestMalformedConfig:
    """A typo in retrieval_config must not take retrieval down. context_handler
    catches bare Exception and reports an empty knowledge base, so an
    AttributeError here silently discards a search that already succeeded."""

    @pytest.mark.parametrize("bad", ["off", 5, True, ["children"]])
    def test_non_object_block_falls_back_to_defaults(self, install_store, bad):
        store = install_store(_store_with_ref())

        out = _expand([_hit("0001", refs=["0002"])], store, {"graph_expansion": bad})

        assert _by_method(out, "graph_expansion"), "search must still return"
        assert _by_method(out, "graph_expansion_child") == []

    def test_string_false_does_not_enable_children(self, install_store):
        """bool("false") is True, and include_children defaults off — so a
        loose truthiness check fails in the expensive direction. The registry
        this PR edits already uses string booleans ("if_add_node_summary":
        "yes"), so a caller writing "no" here is a live possibility."""
        store = install_store(_store_with_ref())

        out = _expand(
            [_hit("0001", refs=["0002"])],
            store,
            {"graph_expansion": {"include_children": "false"}},
        )

        assert _by_method(out, "graph_expansion_child") == []

    def test_non_boolean_does_not_disable_the_outline(self, install_store):
        store = install_store(_store_with_ref())

        out = _expand(
            [_hit("0001", refs=["0002"])],
            store,
            {"graph_expansion": {"include_doc_toc": 0}},
        )

        assert len(_by_method(out, "graph_toc")) == 1

    def test_numeric_string_cap_falls_back_to_default(self, install_store):
        """int("5") would succeed, but Studio reads a string cap as the default
        — the two must not disagree about what is stored."""
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
            {"graph_expansion": {"include_children": True, "max_children_per_parent": "5"}},
        )

        assert (
            len(_by_method(out, "graph_expansion_child"))
            == knowledge_search.GRAPH_DEFAULT_MAX_CHILDREN
        )

    def test_cap_is_bounded_above(self, install_store):
        """A floor without a ceiling restores the unbounded fan-out this
        bounding exists to remove."""
        many = [_node_row(f"{i:04d}", parent="0002") for i in range(3, 60)]
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                children={(TOC_A, "0002"): many},
                outlines={TOC_A: _outline()},
            )
        )

        out = _expand(
            [_hit("0001", refs=["0002"])],
            store,
            {"graph_expansion": {"include_children": True, "max_children_per_parent": 1_000_000}},
        )

        children = _by_method(out, "graph_expansion_child")
        assert len(children) == knowledge_search.GRAPH_MAX_CHILDREN_CEILING


# ---------------------------------------------------------------------------
# Children fan-out
# ---------------------------------------------------------------------------


class TestChildrenFanOut:
    def test_children_are_not_expanded_by_default(self, install_store):
        store = install_store(_store_with_ref())

        out = _expand([_hit("0001", refs=["0002"])], store)

        assert _by_method(out, "graph_expansion"), "referenced node should still be pulled in"
        assert _by_method(out, "graph_expansion_child") == []

    def test_children_are_expanded_when_enabled(self, install_store):
        store = install_store(_store_with_ref())

        out = _expand(
            [_hit("0001", refs=["0002"])],
            store,
            {"graph_expansion": {"include_children": True}},
        )

        children = _by_method(out, "graph_expansion_child")
        assert [c.meta["node_id"] for c in children] == ["0003"]

    def test_children_are_capped_in_document_order_not_row_order(self, install_store):
        """get_children_by_parent_ids has no ORDER BY, so the cap must not be
        applied to whatever order the rows arrive in."""
        shuffled = [
            _node_row("0006", parent="0002"),
            _node_row("0004", parent="0002"),
            _node_row("0007", parent="0002"),
            _node_row("0003", parent="0002"),
            _node_row("0005", parent="0002"),
        ]
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                children={(TOC_A, "0002"): shuffled},
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

    def test_a_child_already_in_the_pool_does_not_consume_cap_budget(self, install_store):
        """Deduplication happens before the cap is charged, so the cap bounds
        newly-added items rather than total context for that parent."""
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                children={
                    (TOC_A, "0002"): [
                        _node_row("0003", parent="0002"),
                        _node_row("0004", parent="0002"),
                        _node_row("0005", parent="0002"),
                    ]
                },
                outlines={TOC_A: _outline()},
            )
        )
        pool = [_hit("0001", refs=["0002"]), _hit("0003")]

        out = _expand(
            pool,
            store,
            {"graph_expansion": {"include_children": True, "max_children_per_parent": 2}},
        )

        children = _by_method(out, "graph_expansion_child")
        assert [c.meta["node_id"] for c in children] == ["0004", "0005"]

    def test_children_are_not_queried_when_disabled(self, install_store):
        store = install_store(_store_with_ref())

        _expand([_hit("0001", refs=["0002"])], store)

        assert store.children_calls == []


# ---------------------------------------------------------------------------
# Document outline
# ---------------------------------------------------------------------------


class TestDocumentOutline:
    """The outline's SQL (paging, the toc join, the uuid[] cast) is covered by
    tests/test_graph_index_store.py, which runs against Postgres. The fakes
    here cover only the surrounding logic."""

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
        store = install_store(_store_with_ref())

        out = _expand(
            [_hit("0001", refs=["0002"])],
            store,
            {"graph_expansion": {"include_doc_toc": False}},
        )

        assert _by_method(out, "graph_toc") == []

    def test_no_outline_item_for_a_document_with_no_nodes(self, install_store):
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                outlines={TOC_A: _outline(nodes=[])},
            )
        )

        out = _expand([_hit("0001", refs=["0002"])], store)

        assert _by_method(out, "graph_toc") == []

    def test_outline_failure_does_not_discard_the_search(self, install_store):
        """The outline decorates results that already exist — a statement
        timeout on it must not throw away a search that succeeded."""
        store = install_store(_store_with_ref(outline_error=RuntimeError("statement timeout")))

        out = _expand([_hit("0001", refs=["0002"])], store)

        assert _by_method(out, "graph_toc") == []
        assert [r.meta["node_id"] for r in _by_method(out, "graph_expansion")] == ["0002"]

    def test_outline_renders_ids_and_indents_by_depth(self, install_store):
        store = install_store(_store_with_ref())

        out = _expand([_hit("0001", refs=["0002"])], store)

        text = _by_method(out, "graph_toc")[0].text
        assert "[0001] Definitions" in text
        assert "  [0003] Scope of Indemnity" in text

    def test_truncated_outline_reports_how_many_sections_were_omitted(self, install_store):
        """The store pages the query, so the count of omitted sections comes
        from the total rather than from what was fetched."""
        limit = knowledge_search.GRAPH_TOC_MAX_NODES
        fetched = [
            {"node_id": f"{i:04d}", "title": f"Section {i}", "depth": 0}
            for i in range(1, limit + 1)
        ]
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                outlines={TOC_A: _outline(nodes=fetched, total_nodes=limit + 5)},
            )
        )

        out = _expand([_hit("0001", refs=["0002"])], store)

        text = _by_method(out, "graph_toc")[0].text
        assert len(text.splitlines()) == limit + 1
        assert "5 more sections" in text

    def test_outline_is_structural_and_never_triggers_image_fetching(self, install_store):
        """An item with no page info makes context_handler fall back to
        fetching every image for that source — and graph references are
        intra-document, so the outline's source always names a document that
        already has precise page coverage. Structural items must be skipped
        before that fallback is reached."""
        store = install_store(_store_with_ref())

        out = _expand([_hit("0001", refs=["0002"])], store)

        outline = _by_method(out, "graph_toc")[0]
        assert knowledge_search._is_structural_item(outline) is True
        assert knowledge_search._is_structural_item(_by_method(out, "graph_expansion")[0]) is False

    def test_outline_is_labelled_in_formatted_context(self, install_store):
        """Without an annotation the outline reaches the model as a bare [N]
        with nothing marking it as structure rather than source text."""
        store = install_store(_store_with_ref())

        out = _expand([_hit("0001", refs=["0002"])], store)

        annotation = knowledge_search._format_chunk_annotation(_by_method(out, "graph_toc")[0])
        assert "outline" in annotation.lower()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestExpansionScores:
    def test_tiers_stay_ordered_at_small_score_scale(self):
        """A fixed decrement has to assume a score scale. Cosine, normalized
        RRF and reranker scores are all different, so the step is taken
        relative to the pool's own minimum instead."""
        min_score = 0.02
        toc, parent, child = knowledge_search._expansion_tier_scores(min_score)

        assert child < parent < toc < min_score
        assert child > 0, "tiers must stay on the pool's scale, not clamp to zero"

    def test_tiers_stay_ordered_when_the_pool_minimum_is_zero(self):
        """Reachable: bm25_score returns 0.0 for an empty tsvector, and the
        vector store substitutes 0.0 for a NULL similarity. A step derived by
        multiplying the minimum would collapse every tier onto 0.0."""
        toc, parent, child = knowledge_search._expansion_tier_scores(0.0)

        assert child < parent < toc < 0.0

    def test_expansion_items_carry_the_tiers(self, install_store):
        store = install_store(_store_with_ref())
        pool = [_hit("0001", refs=["0002"], score=0.7), _hit("0005", score=0.4)]

        out = _expand(pool, store, {"graph_expansion": {"include_children": True}})

        toc = _by_method(out, "graph_toc")[0].score
        parent = _by_method(out, "graph_expansion")[0].score
        child = _by_method(out, "graph_expansion_child")[0].score
        assert child < parent < toc < 0.4


# ---------------------------------------------------------------------------
# Wiring — the defaults that ship, and the call sites that read them
# ---------------------------------------------------------------------------


class TestWiring:
    def test_registry_ships_the_documented_defaults(self):
        from agentic_project_service.strategies.registry import STRATEGY_REGISTRY

        cfg = STRATEGY_REGISTRY["graph_index"]["default_retrieval_config"]["graph_expansion"]
        assert cfg == {
            "include_children": False,
            "max_children_per_parent": knowledge_search.GRAPH_DEFAULT_MAX_CHILDREN,
            "include_doc_toc": True,
        }

    def test_both_call_sites_pass_retrieval_config_to_expansion(self):
        """Dropping the argument at either call site silently reverts every
        search to the defaults. Exercising the call sites for real needs a
        database, so this asserts the wiring structurally instead."""
        source = Path(knowledge_search.__file__).read_text()
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_expand_graph_neighbors"
        ]

        assert len(calls) == 2, "expected the sync and async graph_index call sites"
        for call in calls:
            passed = [a.id for a in call.args if isinstance(a, ast.Name)]
            passed += [kw.value.id for kw in call.keywords if isinstance(kw.value, ast.Name)]
            assert "retrieval_config" in passed
