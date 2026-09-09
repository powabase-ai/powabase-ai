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
  minimum is 0.0, which the BM25 leg reaches on a row admitted by a negated
  term (hybrid's fused scores are strictly positive)

The store is faked at the class boundary — these are unit tests for the
expansion logic. ``get_toc_outline``'s SQL is covered separately, against a
real database, in tests/test_graph_index_store.py.
"""

from __future__ import annotations

import logging

import pytest
from agentic.knowledge.models import RetrievedItem
from sqlalchemy.exc import SQLAlchemyError

from agentic_project_service.services import knowledge_search

KB_ID = "kb-1"
TOC_A = "toc-a"
TOC_B = "toc-b"


# ---------------------------------------------------------------------------
# Fakes + builders
# ---------------------------------------------------------------------------


def _source_for(toc_id: str) -> str:
    """One document per toc_id, the way indexing writes them."""
    return "src-A" if toc_id == TOC_A else "src-B"


def _hit(node_id: str, *, toc_id: str = TOC_A, refs: list[str] | None = None, score: float = 0.9):
    """A search result as the retrieval pipeline hands it to expansion."""
    return RetrievedItem(
        item_id=f"row-{toc_id}-{node_id}",
        text=f"body of {node_id}",
        score=score,
        source_id=_source_for(toc_id),
        knowledge_base_id=KB_ID,
        meta={"toc_id": toc_id, "node_id": node_id, "referenced_nodes": refs or []},
    )


def _node_row(
    node_id: str,
    *,
    toc_id: str = TOC_A,
    parent: str | None = None,
    refs=None,
    source_id: str | None = None,
):
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
        "source_id": source_id or _source_for(toc_id),
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
        # Reversed on purpose. The real store ORs the pairs into one WHERE
        # with no ORDER BY, so the dict comes back in DB row order and the
        # caller must not inherit its ranking from this mapping. Returning
        # `selections` order here would hide exactly that bug.
        return {key: self._nodes[key] for key in reversed(list(selections)) if key in self._nodes}

    def get_children_by_parent_ids(self, parent_selections):
        self.children_calls.append(list(parent_selections))
        # Reversed for the same reason as get_nodes_by_ids.
        return {
            key: self._children[key]
            for key in reversed(list(parent_selections))
            if key in self._children
        }

    def get_toc_outline(self, toc_ids, limit):
        self.outline_calls.append(list(toc_ids))
        if self._outline_error:
            raise self._outline_error
        return {
            tid: {**self._outlines[tid], "nodes": self._outlines[tid]["nodes"][:limit]}
            for tid in toc_ids
            if tid in self._outlines
        }


class _CountingSession:
    """Answers the one ``SELECT COUNT(*)`` the graph_index branch runs."""

    def __init__(self, node_count: int):
        self._node_count = node_count

    def execute(self, *args, **kwargs):
        count = self._node_count

        class _Result:
            def scalar(self):
                return count

        return _Result()


class _RecordingSession:
    """Session stub that records how a failed statement was cleaned up."""

    def __init__(self):
        self.savepoints: list[str] = []
        self.rollback_calls = 0

    def begin_nested(self):
        session = self

        class _Savepoint:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                session.savepoints.append("rolled-back" if exc_type else "committed")
                return False

        return _Savepoint()

    def rollback(self):
        self.rollback_calls += 1


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
        "source_id": _source_for(toc_id),
        "nodes": resolved,
        "total_nodes": total_nodes if total_nodes is not None else len(resolved),
    }


def _expand(results, store, retrieval_config=None, db_session=None):
    return knowledge_search._expand_graph_neighbors(
        db_session=db_session if db_session is not None else _RecordingSession(),
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

    def test_a_boolean_cap_is_rejected_not_read_as_one(self, install_store):
        """isinstance(True, int) is True in Python, so without the explicit
        bool exclusion this would cap children at 1 rather than defaulting."""
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
            {"graph_expansion": {"include_children": True, "max_children_per_parent": True}},
        )

        assert (
            len(_by_method(out, "graph_expansion_child"))
            == knowledge_search.GRAPH_DEFAULT_MAX_CHILDREN
        )

    def test_malformed_values_are_warned_about_with_the_kb_id(self, install_store, caplog):
        """Degrading silently is what makes a misconfiguration permanent, and
        a warning with no KB in it cannot be acted on."""
        store = install_store(_store_with_ref())

        with caplog.at_level(logging.WARNING):
            _expand(
                [_hit("0001", refs=["0002"])],
                store,
                {"graph_expansion": {"include_children": "yes", "maxChildrenPerParent": 10}},
            )

        text = caplog.text
        assert "include_children" in text
        assert "maxChildrenPerParent" in text, "an unknown key must not pass silently"
        assert KB_ID in text

    def test_out_of_range_cap_is_warned_about_not_just_clamped(self, install_store, caplog):
        """The INFO summary reports the effective cap, which reads as though
        the configured value had been honoured."""
        store = install_store(_store_with_ref())

        with caplog.at_level(logging.WARNING):
            _expand(
                [_hit("0001", refs=["0002"])],
                store,
                {"graph_expansion": {"include_children": True, "max_children_per_parent": 999}},
            )

        assert "clamped" in caplog.text
        assert KB_ID in caplog.text

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

    def test_referenced_nodes_are_capped(self, install_store):
        """A hit's references are its largest cost: measured against a real
        corpus, one node with 12 references pulled ~27k tokens of section
        bodies — against the 16k KB context budget in force at the time."""
        many = {(TOC_A, f"{i:04d}"): _node_row(f"{i:04d}") for i in range(10, 30)}
        store = install_store(FakeGraphIndexStore(nodes=many, outlines={TOC_A: _outline()}))
        refs = [f"{i:04d}" for i in range(10, 30)]

        out = _expand(
            [_hit("0001", refs=refs)],
            store,
            {"graph_expansion": {"max_referenced_nodes": 4}},
        )

        assert len(_by_method(out, "graph_expansion")) == 4

    def test_referenced_node_cap_is_bounded_above(self, install_store):
        """A configured million restores exactly the unbounded fan-out the cap
        removes. The pool deliberately holds more references than the ceiling
        and the assertion is an equality: against a smaller pool, or written
        as ``<= ceiling``, it would hold for any cap at all — including a
        million read straight from the config."""
        refs = [f"{i:04d}" for i in range(10, 160)]
        many = {(TOC_A, ref): _node_row(ref) for ref in refs}
        store = install_store(FakeGraphIndexStore(nodes=many, outlines={TOC_A: _outline()}))

        out = _expand(
            [_hit("0001", refs=refs)],
            store,
            {"graph_expansion": {"max_referenced_nodes": 10_000}},
        )

        assert len(refs) > knowledge_search.GRAPH_MAX_REFERENCED_CEILING, "fixture too small"
        assert (
            len(_by_method(out, "graph_expansion")) == knowledge_search.GRAPH_MAX_REFERENCED_CEILING
        )

    def test_the_cap_keeps_references_the_most_hits_agree_on(self, install_store):
        """Once the cap bites, which references survive has to be decided by
        something. Consensus first — a section two hits both point at is a
        better bet than one only the top hit mentions."""
        nodes = {(TOC_A, n): _node_row(n) for n in ("0010", "0011", "0012")}
        store = install_store(FakeGraphIndexStore(nodes=nodes, outlines={TOC_A: _outline()}))
        pool = [
            _hit("0001", refs=["0010", "0011"], score=0.9),
            _hit("0002", refs=["0011", "0012"], score=0.5),
        ]

        out = _expand(pool, store, {"graph_expansion": {"max_referenced_nodes": 1}})

        kept = [r.meta["node_id"] for r in _by_method(out, "graph_expansion")]
        assert kept == ["0011"], "the reference both hits share should win"

    def test_ties_on_consensus_fall_back_to_the_best_referring_hit(self, install_store):
        """The stronger hit points at the *higher* node_id on purpose: with the
        scores keyed the other way round, node_id order alone would produce
        the same answer and the score key would be untested."""
        nodes = {(TOC_A, n): _node_row(n) for n in ("0010", "0012")}
        store = install_store(FakeGraphIndexStore(nodes=nodes, outlines={TOC_A: _outline()}))
        pool = [
            _hit("0001", refs=["0012"], score=0.9),
            _hit("0002", refs=["0010"], score=0.5),
        ]

        out = _expand(pool, store, {"graph_expansion": {"max_referenced_nodes": 1}})

        assert [r.meta["node_id"] for r in _by_method(out, "graph_expansion")] == ["0012"]

    def test_the_score_key_is_the_best_referring_hit_not_the_worst_or_the_first(
        self, install_store
    ):
        """Both references have two hits, so consensus ties and the score key
        alone decides. The scores are arranged so `max` picks 0021 while
        `min` and first-seen both pick 0012 — a fixture where each reference
        is named once cannot separate those three, because they are the same
        number."""
        nodes = {(TOC_A, n): _node_row(n) for n in ("0012", "0021")}
        store = install_store(FakeGraphIndexStore(nodes=nodes, outlines={TOC_A: _outline()}))
        pool = [
            _hit("0001", refs=["0021"], score=0.1),
            _hit("0002", refs=["0012"], score=0.5),
            _hit("0003", refs=["0021"], score=0.9),
            _hit("0004", refs=["0012"], score=0.4),
        ]

        out = _expand(pool, store, {"graph_expansion": {"max_referenced_nodes": 1}})

        kept = [r.meta["node_id"] for r in _by_method(out, "graph_expansion")]
        assert kept == ["0021"], "0021's best hit is 0.9; its first and worst are 0.1"

    def test_the_final_tiebreak_is_the_key_not_the_order_references_were_seen(self, install_store):
        """Consensus and score both tie, so only the key separates them. It is
        seen in descending order, so insertion order — what a dropped or
        reversed tiebreak falls back to — gives the opposite answer."""
        nodes = {(TOC_A, n): _node_row(n) for n in ("0020", "0030")}
        store = install_store(FakeGraphIndexStore(nodes=nodes, outlines={TOC_A: _outline()}))
        pool = [_hit("0001", refs=["0030", "0020"], score=0.5)]

        out = _expand(pool, store, {"graph_expansion": {"max_referenced_nodes": 2}})

        kept = [r.meta["node_id"] for r in _by_method(out, "graph_expansion")]
        assert kept == ["0020", "0030"]

    def test_a_hit_naming_a_reference_twice_does_not_outrank_consensus(self, install_store):
        """ref_hits counts how many *hits* point at a section. Nothing
        guarantees one hit's referenced_nodes is deduplicated, and counting
        occurrences instead lets a single hit manufacture its own consensus."""
        nodes = {(TOC_A, n): _node_row(n) for n in ("0010", "0011")}
        store = install_store(FakeGraphIndexStore(nodes=nodes, outlines={TOC_A: _outline()}))
        pool = [
            _hit("0001", refs=["0010", "0010"], score=0.5),
            _hit("0002", refs=["0011"], score=0.9),
        ]

        out = _expand(pool, store, {"graph_expansion": {"max_referenced_nodes": 1}})

        kept = [r.meta["node_id"] for r in _by_method(out, "graph_expansion")]
        assert kept == ["0011"], "a doubled reference in one hit is still one hit"

    def test_emitted_order_follows_the_ranking_not_the_store(self, install_store):
        """The ranking has to reach the emitted list, not just the fetch.
        Every referenced node carries the same score and context formatting
        truncates positionally, so emission order decides which references
        survive a tight token budget."""
        nodes = {(TOC_A, n): _node_row(n) for n in ("0010", "0011", "0019")}
        store = install_store(FakeGraphIndexStore(nodes=nodes, outlines={TOC_A: _outline()}))
        pool = [
            _hit("0001", refs=["0019", "0011"], score=0.9),
            _hit("0002", refs=["0011", "0010"], score=0.2),
        ]

        out = _expand(pool, store, {"graph_expansion": {"max_referenced_nodes": 3}})

        kept = [r.meta["node_id"] for r in _by_method(out, "graph_expansion")]
        # 0011 has two hits; 0019 and 0010 have one each, ordered by their
        # referring hit's score — which is the reverse of node_id order, so
        # neither the store's order nor a tiebreak-only sort produces this.
        assert kept == ["0011", "0019", "0010"]

    def test_children_follow_their_parents_ranking(self, install_store):
        """Children inherit their parent's standing, so a positional cut
        should reach the least-agreed-on parent's subtree first."""
        nodes = {(TOC_A, n): _node_row(n) for n in ("0010", "0019")}
        children = {
            (TOC_A, "0010"): [_node_row("0010.1", parent="0010")],
            (TOC_A, "0019"): [_node_row("0019.1", parent="0019")],
        }
        store = install_store(
            FakeGraphIndexStore(nodes=nodes, children=children, outlines={TOC_A: _outline()})
        )
        pool = [
            _hit("0001", refs=["0010"], score=0.9),
            _hit("0002", refs=["0019"], score=0.2),
        ]

        out = _expand(
            pool,
            store,
            {"graph_expansion": {"include_children": True, "max_referenced_nodes": 2}},
        )

        kept = [r.meta["node_id"] for r in _by_method(out, "graph_expansion_child")]
        assert kept == ["0010.1", "0019.1"]

    def test_capping_references_is_reported(self, install_store, caplog):
        nodes = {(TOC_A, f"{i:04d}"): _node_row(f"{i:04d}") for i in range(10, 20)}
        store = install_store(FakeGraphIndexStore(nodes=nodes, outlines={TOC_A: _outline()}))

        with caplog.at_level(logging.INFO):
            _expand(
                [_hit("0001", refs=[f"{i:04d}" for i in range(10, 20)])],
                store,
                {"graph_expansion": {"max_referenced_nodes": 3}},
            )

        # Asserted on the cap line's own wording: the summary line that always
        # logs already contains both numbers, so a bare "10"/"3" check passes
        # with the statement under test deleted.
        assert "exceed the cap of" in caplog.text
        assert "10 referenced nodes" in caplog.text

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
        store = install_store(_store_with_ref(outline_error=SQLAlchemyError("statement timeout")))

        out = _expand([_hit("0001", refs=["0002"])], store)

        assert _by_method(out, "graph_toc") == []
        assert [r.meta["node_id"] for r in _by_method(out, "graph_expansion")] == ["0002"]

    def test_a_non_database_failure_still_propagates(self, install_store):
        """Only database errors are treated as best-effort. A bug in the
        outline path is not transient, and swallowing it would hide the
        feature disappearing behind a single log line."""
        store = install_store(_store_with_ref(outline_error=KeyError("total_nodes")))

        with pytest.raises(KeyError):
            _expand([_hit("0001", refs=["0002"])], store)

    def test_outline_failure_rolls_the_session_back_to_a_savepoint(self, install_store):
        """Swallowing a DBAPI error without rolling back leaves the session in
        a deactivated transaction, so the *next* statement raises
        PendingRollbackError — which context_handler turns into an empty
        knowledge base, or which detonates outside every try block later. The
        outline must undo only its own statement, hence a savepoint rather
        than a full rollback: whatever the caller already has pending on this
        session stays pending."""
        session = _RecordingSession()
        store = _store_with_ref(outline_error=SQLAlchemyError("statement timeout"))
        install_store(store)

        out = knowledge_search._expand_graph_neighbors(
            db_session=session,
            results=[_hit("0001", refs=["0002"])],
            knowledge_base_id=KB_ID,
            retrieval_config={"graph_expansion": {"include_children": True}},
        )

        assert session.savepoints == ["rolled-back"]
        assert session.rollback_calls == 0, (
            "a full rollback would discard the caller's pending work"
        )
        # The search survives, and the statement after the failure still runs.
        assert [r.meta["node_id"] for r in _by_method(out, "graph_expansion")] == ["0002"]
        assert store.children_calls, "expansion must keep using the session afterwards"

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

    def test_a_complete_outline_carries_no_truncation_marker(self, install_store):
        """`remaining > 0` rather than `>= 0`: a "... (0 more sections)" line
        would ship on every complete outline the model reads."""
        store = install_store(_store_with_ref())

        out = _expand([_hit("0001", refs=["0002"])], store)

        assert "more sections" not in _by_method(out, "graph_toc")[0].text

    def test_a_single_omitted_section_is_reported(self, install_store):
        limit = knowledge_search.GRAPH_TOC_MAX_NODES
        fetched = [
            {"node_id": f"{i:04d}", "title": f"Section {i}", "depth": 0}
            for i in range(1, limit + 1)
        ]
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                outlines={TOC_A: _outline(nodes=fetched, total_nodes=limit + 1)},
            )
        )

        out = _expand([_hit("0001", refs=["0002"])], store)

        assert "1 more sections" in _by_method(out, "graph_toc")[0].text

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

    def test_tiers_stay_on_scale_for_a_negative_pool_minimum(self):
        """Rerankers are supported for this strategy and cross-encoders emit
        negative scores routinely.

        Ordering alone does not catch a missing ``abs()``: the step would fall
        back to the 1e-6 floor and the tiers would still come out ordered,
        just six orders of magnitude too close together to survive a float
        comparison. The size of the step is the thing to assert."""
        min_score = -2.0
        toc, parent, child = knowledge_search._expansion_tier_scores(min_score)

        assert child < parent < toc < min_score
        assert toc == pytest.approx(min_score - abs(min_score) * 0.01)

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
    def test_the_shipped_bounds_are_the_ones_that_were_chosen(self):
        """Literals, not the constants themselves. Every other assertion here
        compares the registry to the constant, which pins the link and leaves
        the value free — the bound this whole feature is named for was not
        pinned by anything until this test."""
        assert knowledge_search.GRAPH_DEFAULT_MAX_REFERENCED_NODES == 10
        assert knowledge_search.GRAPH_DEFAULT_MAX_CHILDREN == 3
        assert knowledge_search.GRAPH_MAX_REFERENCED_CEILING == 100
        assert knowledge_search.GRAPH_MAX_CHILDREN_CEILING == 20
        assert knowledge_search.GRAPH_DEFAULT_INCLUDE_CHILDREN is False
        assert knowledge_search.GRAPH_DEFAULT_INCLUDE_DOC_TOC is True

    def test_registry_ships_the_documented_defaults(self):
        from agentic_project_service.strategies.registry import STRATEGY_REGISTRY

        cfg = STRATEGY_REGISTRY["graph_index"]["default_retrieval_config"]["graph_expansion"]
        assert cfg == {
            "include_children": False,
            "max_children_per_parent": knowledge_search.GRAPH_DEFAULT_MAX_CHILDREN,
            "max_referenced_nodes": knowledge_search.GRAPH_DEFAULT_MAX_REFERENCED_NODES,
            "include_doc_toc": True,
        }
        # Adding a knob to the registry but not to the frozenset makes the
        # read path log "unknown key" on every query for every KB created
        # with the defaults.
        assert set(cfg) == knowledge_search.GRAPH_EXPANSION_KEYS

    def test_sync_search_hands_expansion_the_kb_config_and_id(self, monkeypatch):
        """Dropping — or transposing — the arguments at the call site silently
        reverts every search to the defaults, or stamps expansion items with a
        config dict where a knowledge_base_id belongs. Drives the real
        ``search_knowledge_base`` graph_index branch with a session that only
        has to answer the node COUNT(*)."""
        recorded = {}

        def _recording_expand(db_session, results, knowledge_base_id, retrieval_config=None):
            recorded["kb_id"] = knowledge_base_id
            recorded["retrieval_config"] = retrieval_config
            return results

        monkeypatch.setattr(knowledge_search, "_expand_graph_neighbors", _recording_expand)
        monkeypatch.setattr(
            knowledge_search,
            "GraphIndexNodeStore",
            lambda db_session, knowledge_base_id: object(),
        )
        monkeypatch.setattr(
            knowledge_search,
            "_execute_retrieval_pipeline",
            lambda **kwargs: [],
        )

        retrieval_config = {
            "method": "vector_search",
            "graph_expansion": {"include_children": True},
        }
        knowledge_search.search_knowledge_base(
            db_session=_CountingSession(node_count=1),
            knowledge_base_id=KB_ID,
            query="q",
            retrieval_method="vector_search",
            indexing_config={"strategy": "graph_index"},
            retrieval_config=retrieval_config,
        )

        assert recorded["kb_id"] == KB_ID, "arguments transposed"
        assert recorded["retrieval_config"] is retrieval_config

    @pytest.mark.asyncio
    async def test_async_search_hands_expansion_the_kb_config_and_id(self, monkeypatch):
        """The agent path runs the async variant, and nothing in tests/
        referenced ``search_knowledge_base_async`` at all — so it could revert
        to the defaults, or transpose the same two arguments, with CI green."""
        recorded = {}

        def _recording_expand(db_session, results, knowledge_base_id, retrieval_config=None):
            recorded["kb_id"] = knowledge_base_id
            recorded["retrieval_config"] = retrieval_config
            return results

        async def _empty_pipeline(**kwargs):
            return []

        monkeypatch.setattr(knowledge_search, "_expand_graph_neighbors", _recording_expand)
        monkeypatch.setattr(
            knowledge_search,
            "GraphIndexNodeStore",
            lambda db_session, knowledge_base_id: object(),
        )
        monkeypatch.setattr(knowledge_search, "_aexecute_retrieval_pipeline", _empty_pipeline)

        retrieval_config = {
            "method": "vector_search",
            "graph_expansion": {"include_children": True},
        }
        await knowledge_search.search_knowledge_base_async(
            db_session=_CountingSession(node_count=1),
            knowledge_base_id=KB_ID,
            query="q",
            retrieval_method="vector_search",
            indexing_config={"strategy": "graph_index"},
            retrieval_config=retrieval_config,
        )

        assert recorded["kb_id"] == KB_ID, "arguments transposed"
        assert recorded["retrieval_config"] is retrieval_config


# ---------------------------------------------------------------------------
# What expansion reports about itself
# ---------------------------------------------------------------------------


class TestExpansionReporting:
    """Expansion drops things for three different reasons, and a drop nobody
    can see reads as a corpus that had nothing to give."""

    def test_a_reference_with_no_node_row_is_reported(self, install_store, caplog):
        """`referenced_nodes` and `graph_index_nodes` are written and deleted
        separately, so a reference can outlive the section it names. The old
        loop walked the fetched map and never noticed; the ranked loop
        computes the miss explicitly, which makes discarding it a choice."""
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                outlines={TOC_A: _outline()},
            )
        )

        with caplog.at_level(logging.WARNING):
            out = _expand([_hit("0001", refs=["0002", "0404"])], store)

        assert len(_by_method(out, "graph_expansion")) == 1
        assert "1 of 2" in caplog.text
        assert KB_ID in caplog.text

    def test_the_summary_says_how_many_references_were_selected(self, install_store, caplog):
        """Whenever the store's keys spell their toc_ids the way the pool
        does, the map is a subset of the selections and its size is already
        the emitted count — so renaming it changes nothing on its own. What
        was missing is the other number: without ``selected``, 10 chosen and
        6 resolved reads exactly like 6 chosen.

        Asserted against the summary record alone. ``caplog.text`` also holds
        the missing-row warning, which contains "of N selected references" —
        so a whole-text assertion passes with the summary's count mutated
        away, which is the very confound this test was added to remove.

        The fixture makes the three counts differ: 5 references, a cap of 4,
        and 2 of the 4 selected resolvable. Equal numbers are substitutable
        ones — with 3 refs under a cap of 10, ``candidates``, ``selected``
        and each other are the same integer and any of them can stand in."""
        refs = ["0002", "0003", "0404", "0405", "0406"]
        store = install_store(
            FakeGraphIndexStore(
                nodes={
                    (TOC_A, "0002"): _node_row("0002"),
                    (TOC_A, "0003"): _node_row("0003"),
                },
                outlines={TOC_A: _outline()},
            )
        )

        with caplog.at_level(logging.INFO):
            _expand(
                [_hit("0001", refs=refs)],
                store,
                {"graph_expansion": {"max_referenced_nodes": 4}},
            )

        summaries = [r.getMessage() for r in caplog.records if "emitted" in r.getMessage()]
        assert len(summaries) == 1
        assert "emitted 2 neighbors" in summaries[0]
        assert "4 selected" in summaries[0]
        assert "from 5 candidate refs" in summaries[0]

    def test_the_child_cap_reports_what_it_dropped(self, install_store, caplog):
        """The low child default is justified by the outline naming the
        sections it omits — but ``include_doc_toc`` can be off, and then the
        stated mitigation is absent with nothing recording the loss.

        Read off the child-cap record, not ``caplog.text``: the summary line
        emits ``outline=%s`` too, so a whole-text assertion is satisfied by
        the wrong line and the cap record's own fields go unpinned."""
        children = [_node_row(f"000{i}", parent="0002") for i in range(3, 7)]
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                children={(TOC_A, "0002"): children},
                outlines={TOC_A: _outline()},
            )
        )

        with caplog.at_level(logging.INFO):
            out = _expand(
                [_hit("0001", refs=["0002"])],
                store,
                {
                    "graph_expansion": {
                        "include_children": True,
                        "max_children_per_parent": 2,
                        "include_doc_toc": False,
                    }
                },
            )

        assert len(_by_method(out, "graph_expansion_child")) == 2
        cap_lines = [r.getMessage() for r in caplog.records if "withheld" in r.getMessage()]
        assert len(cap_lines) == 1
        assert "withheld 2 of 4 children" in cap_lines[0]
        assert "cap=2" in cap_lines[0]
        assert "outline=False" in cap_lines[0], "the cap line has to carry it, not the summary"

    def test_children_already_in_the_pool_are_not_counted_as_dropped(self, install_store, caplog):
        """A child the search already found is skipped without consuming cap
        budget, so it is not something the cap withheld either.

        The pooled child sits *after* the cut on purpose: anywhere before it,
        a naive "everything left in the list" count gives the same answer as
        the dedup-aware one and the distinction goes untested."""
        children = [_node_row(f"000{i}", parent="0002") for i in range(3, 7)]
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                children={(TOC_A, "0002"): children},
                outlines={TOC_A: _outline()},
            )
        )

        with caplog.at_level(logging.INFO):
            out = _expand(
                [_hit("0001", refs=["0002"]), _hit("0006")],
                store,
                {"graph_expansion": {"include_children": True, "max_children_per_parent": 2}},
            )

        kept = [c.meta["node_id"] for c in _by_method(out, "graph_expansion_child")]
        assert kept == ["0003", "0004"]
        # 0005 was withheld; 0006 was already in the pool, so it was not.
        assert "withheld 1 of 3 children" in caplog.text


# ---------------------------------------------------------------------------
# Behaviour that is correct today and was pinned by nothing
# ---------------------------------------------------------------------------


class TestUnpinnedInvariants:
    """None of these was failing. Each is a way a later edit could spend the
    budget this feature exists to bound, without any test objecting."""

    def test_a_reference_already_in_the_pool_is_not_re_emitted(self, install_store):
        """The search already returned 0002 and paid for it once. Re-adding it
        spends cap budget on a duplicate — invisible in the response, which
        just looks like a shorter answer."""
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, "0002"): _node_row("0002")},
                outlines={TOC_A: _outline()},
            )
        )
        pool = [_hit("0001", refs=["0002"], score=0.9), _hit("0002", score=0.4)]

        out = _expand(pool, store)

        assert _by_method(out, "graph_expansion") == []

    def test_a_negative_cap_from_a_legacy_row_means_none_not_all_but_one(self, install_store):
        """The route rejects a negative cap on write; the read-path clamp is
        for rows written before it. Unclamped, ``ranked[:-1]`` drops one
        reference from the *end* and keeps the rest — a cap that expands."""
        refs = [f"{i:04d}" for i in range(10, 15)]
        store = install_store(
            FakeGraphIndexStore(
                nodes={(TOC_A, r): _node_row(r) for r in refs},
                outlines={TOC_A: _outline()},
            )
        )

        out = _expand(
            [_hit("0001", refs=refs)],
            store,
            {"graph_expansion": {"max_referenced_nodes": -1}},
        )

        assert _by_method(out, "graph_expansion") == []

    def test_a_scoreless_item_in_the_pool_does_not_break_the_tiers(self, install_store):
        """``score`` is Optional on RetrievedItem, and the tier floor is
        ``min()`` over the pool — over a list containing None it raises, and
        expansion fails the whole search rather than a scored subset."""
        store = install_store(_store_with_ref())
        scoreless = _hit("0009")
        scoreless.score = None
        pool = [_hit("0001", refs=["0002"], score=0.7), scoreless]

        out = _expand(pool, store)

        assert [r.meta["node_id"] for r in _by_method(out, "graph_expansion")] == ["0002"]

    def test_a_child_cap_of_zero_asks_the_store_for_nothing(self, install_store, caplog):
        """0 means off, and off should cost no query. It also must not report
        a withholding: everything it withheld is what was asked for."""
        store = install_store(_store_with_ref())

        with caplog.at_level(logging.INFO):
            out = _expand(
                [_hit("0001", refs=["0002"])],
                store,
                {"graph_expansion": {"include_children": True, "max_children_per_parent": 0}},
            )

        assert store.children_calls == []
        assert _by_method(out, "graph_expansion_child") == []
        assert "withheld" not in caplog.text


# ---------------------------------------------------------------------------
# Page spans on expansion items
# ---------------------------------------------------------------------------


class TestExpansionPages:
    """An expansion item with no pages makes the image path fall back to
    attaching every image of its source — the amplification the structural
    guard exists to prevent, reached through a different door. ``page and``
    is falsy for page 0 and for a section whose end page was never recorded.
    """

    def _pages_of(self, install_store, node_meta):
        row = _node_row("0002")
        row["meta"] = {**row["meta"], **node_meta}
        store = install_store(
            FakeGraphIndexStore(nodes={(TOC_A, "0002"): row}, outlines={TOC_A: _outline()})
        )

        out = _expand([_hit("0001", refs=["0002"])], store)
        return _by_method(out, "graph_expansion")[0].meta["pages"]

    def test_a_span_becomes_an_inclusive_range(self, install_store):
        assert self._pages_of(install_store, {"start_page": 3, "end_page": 5}) == [3, 4, 5]

    def test_page_zero_is_a_page(self, install_store):
        assert self._pages_of(install_store, {"start_page": 0, "end_page": 0}) == [0]

    def test_a_start_only_span_still_yields_its_page(self, install_store):
        """``_pages_for_item`` has had this fallback all along; expansion
        stamped `pages: []` for the same node."""
        assert self._pages_of(install_store, {"start_page": 7}) == [7]

    def test_no_page_metadata_yields_no_pages(self, install_store):
        assert self._pages_of(install_store, {}) == []

    def test_unparseable_pages_yield_no_pages(self, install_store):
        """meta is JSONB written by the indexer; a string here must not raise
        in the middle of a successful search."""
        assert self._pages_of(install_store, {"start_page": "seven"}) == []

    def test_an_unparseable_end_page_falls_back_to_the_start(self, install_store):
        """The start-only fallback has to be reachable when it is the *end*
        that is unreadable — which is the only way this differs from
        ``_pages_for_item``, and the direction that matters: returning ``[]``
        for a node that knows its start page is what sends the image path
        into the fetch-everything fallback."""
        assert self._pages_of(install_store, {"start_page": 3, "end_page": "x"}) == [3]


# ---------------------------------------------------------------------------
# More than one document
# ---------------------------------------------------------------------------


class TestAcrossDocuments:
    """Every other fixture here lives in one document. The ranking is global
    across the search, so a fixture that cannot tell two documents apart
    cannot show it working."""

    def test_the_ranking_spans_documents(self, install_store):
        nodes = {
            (TOC_A, "0010"): _node_row("0010"),
            (TOC_B, "0020"): _node_row("0020", toc_id=TOC_B, source_id="src-B"),
        }
        store = install_store(
            FakeGraphIndexStore(
                nodes=nodes,
                outlines={TOC_A: _outline(), TOC_B: _outline(toc_id=TOC_B)},
            )
        )
        pool = [
            _hit("0001", refs=["0020"], toc_id=TOC_B, score=0.9),
            _hit("0002", refs=["0010"], score=0.5),
        ]

        out = _expand(pool, store, {"graph_expansion": {"max_referenced_nodes": 1}})

        kept = _by_method(out, "graph_expansion")
        assert [r.meta["node_id"] for r in kept] == ["0020"]
        assert kept[0].source_id == "src-B", "expansion items carry their own document"

    def test_one_outline_per_document_a_reference_was_followed_into(self, install_store):
        nodes = {
            (TOC_A, "0010"): _node_row("0010"),
            (TOC_B, "0020"): _node_row("0020", toc_id=TOC_B, source_id="src-B"),
        }
        store = install_store(
            FakeGraphIndexStore(
                nodes=nodes,
                outlines={TOC_A: _outline(), TOC_B: _outline(toc_id=TOC_B)},
            )
        )
        pool = [
            _hit("0001", refs=["0020"], toc_id=TOC_B, score=0.9),
            _hit("0002", refs=["0010"], score=0.5),
        ]

        out = _expand(pool, store)

        assert sorted(r.item_id for r in _by_method(out, "graph_toc")) == [TOC_A, TOC_B]
