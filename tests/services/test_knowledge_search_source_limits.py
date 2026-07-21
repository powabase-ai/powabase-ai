"""
Service-level tests for per-source retrieval limits in the chunk-retrieval
pipeline (sync + async _execute_retrieval_pipeline, _read_source_limits,
_merge_floor_items).

These exercise the over-fetch, floor back-fill merge (incl. the cosine-vs-RRF
re-scaling guard and the reranker reorder-not-truncate branch) and final
per-source selection without a DB by patching the retriever maps, the floor
fetch, billing hooks, and enrichment lookup. The window-query SQL itself
(vector_search_per_source) needs Postgres and is covered by manual/integration
verification.
"""

from unittest.mock import MagicMock

import pytest
from agentic.knowledge.models import RetrievedItem

from agentic_project_service.services import knowledge_search as ks


def _item(item_id: str, source_id: str, score: float) -> RetrievedItem:
    return RetrievedItem(item_id=item_id, text=f"t-{item_id}", score=score, source_id=source_id)


def _skewed_pool() -> list[RetrievedItem]:
    """20 candidates, heavily skewed toward source 'A', score-desc."""
    pool: list[RetrievedItem] = []
    # source A dominates the top of the ranking
    for i in range(12):
        pool.append(_item(f"a{i}", "A", 0.99 - i * 0.01))
    for i in range(4):
        pool.append(_item(f"b{i}", "B", 0.60 - i * 0.01))
    for i in range(4):
        pool.append(_item(f"c{i}", "C", 0.40 - i * 0.01))
    return pool


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Patch billing/enrichment to no-ops and capture the retriever's fetch_count."""
    captured = {}

    def _fake_retriever(*, store, query, fetch_count, **kwargs):
        captured["fetch_count"] = fetch_count
        return _skewed_pool()[:fetch_count]

    monkeypatch.setitem(ks.CHUNK_RETRIEVER_MAP, "vector_search", _fake_retriever)
    monkeypatch.setattr(ks, "_bill_retrieval_action", lambda *a, **k: None)
    monkeypatch.setattr(ks, "_post_retrieval_charge", lambda *a, **k: None)
    monkeypatch.setattr(ks, "_attach_enrichment_metadata", lambda *a, **k: False)
    # Floor back-fill hits the DB/embedder; stub it off by default. Individual
    # tests override it to inject specific small-source items.
    monkeypatch.setattr(ks, "_run_source_floor", lambda *a, **k: [])
    return captured


def _run(retrieval_config, top_k=5, method="vector_search"):
    return ks._execute_retrieval_pipeline(
        db_session=MagicMock(),
        store=MagicMock(),
        query="q",
        top_k=top_k,
        retrieval_method=method,
        indexing_config={},
        retrieval_config=retrieval_config,
        filter_metadata=None,
        knowledge_base_id="kb-1",
        request_id="test-rid",
    )


class TestReadSourceLimits:
    @pytest.mark.parametrize(
        "cfg,expected",
        [
            (None, (None, None)),
            ({}, (None, None)),
            ({"min_per_source": 0, "max_per_source": 0}, (None, None)),
            ({"max_per_source": 2}, (None, 2)),
            ({"min_per_source": 1}, (1, None)),
            ({"min_per_source": "2", "max_per_source": "3"}, (2, 3)),
            ({"max_per_source": -1}, (None, None)),
            ({"max_per_source": "bad"}, (None, None)),
        ],
    )
    def test_read(self, cfg, expected):
        assert ks._read_source_limits(cfg) == expected


class TestPipelineSourceLimits:
    def test_no_limits_preserves_top_k(self, patched_pipeline):
        results = _run({"method": "vector_search"}, top_k=5)
        # No over-fetch, plain top_k of the skewed pool (all source A).
        assert patched_pipeline["fetch_count"] == 5
        assert [r.source_id for r in results] == ["A"] * 5

    def test_max_per_source_caps_and_overfetches(self, patched_pipeline):
        results = _run({"method": "vector_search", "max_per_source": 2}, top_k=5)
        # Over-fetched a candidate pool larger than top_k.
        assert patched_pipeline["fetch_count"] == 5 * ks.PER_SOURCE_CANDIDATE_FACTOR
        counts = {s: [r.source_id for r in results].count(s) for s in {"A", "B", "C"}}
        assert counts["A"] <= 2
        assert len(results) == 5

    def test_min_per_source_enforces_diversity(self, patched_pipeline):
        results = _run({"method": "vector_search", "min_per_source": 1}, top_k=5)
        sources = {r.source_id for r in results}
        # Without the floor, top-5 would be all 'A'; the floor pulls in B and C.
        assert {"A", "B", "C"}.issubset(sources)

    def test_overfetch_capped(self, patched_pipeline):
        # A pathological top_k must not trigger an unbounded scan.
        _run({"method": "vector_search", "max_per_source": 2}, top_k=1000)
        assert patched_pipeline["fetch_count"] == ks.PER_SOURCE_CANDIDATE_CAP


class TestMergeFloorItems:
    def test_dedup_keeps_main_and_tags_floor(self):
        main = [_item("a1", "A", 0.9), _item("a2", "A", 0.8)]
        floor = [_item("a1", "A", 0.5), _item("b1", "B", 0.4)]
        merged = ks._merge_floor_items(main, floor)
        ids = [m.item_id for m in merged]
        assert ids == ["a1", "a2", "b1"]  # a1 not duplicated
        # the main copy of a1 is kept (untagged); only new b1 is floor-tagged
        by_id = {m.item_id: m for m in merged}
        assert by_id["a1"].meta.get("source_floor") is None
        assert by_id["b1"].meta.get("source_floor") is True

    def test_no_rescore_default_preserves_cosine_scores(self):
        # vector_search path: scores are commensurable, leave them untouched.
        main = [_item("m1", "A", 0.02)]
        floor = [_item("f1", "B", 0.90)]
        merged = ks._merge_floor_items(main, floor)
        assert {m.item_id: m.score for m in merged}["f1"] == 0.90

    def test_rescore_below_main_for_noncosine_pool(self):
        # hybrid/full_text pool: cosine floor scores must be pushed below the
        # main pool's minimum (preserving floor order) so they can't outrank it.
        main = [_item("m1", "A", 0.030), _item("m2", "A", 0.010)]
        floor = [_item("f1", "B", 0.95), _item("f2", "C", 0.80)]
        merged = ks._merge_floor_items(main, floor, rescore_below_main=True)
        by_id = {m.item_id: m for m in merged}
        main_min = 0.010
        assert by_id["f1"].score < main_min
        assert by_id["f2"].score < main_min
        assert by_id["f1"].score > by_id["f2"].score  # relative order preserved
        # sorted by score, both main items lead the floor items
        ordered = sorted(merged, key=lambda x: x.score, reverse=True)
        assert [o.item_id for o in ordered[:2]] == ["m1", "m2"]


class TestFloorBackfill:
    def test_floor_pulls_in_sources_missing_from_global_pool(self, patched_pipeline, monkeypatch):
        # Main retrieval returns ONLY the dominant source 'A' (mimics a 58-chunk
        # doc burying the small ones). The floor back-fills B and C.
        monkeypatch.setitem(
            ks.CHUNK_RETRIEVER_MAP,
            "vector_search",
            lambda **k: [_item(f"a{i}", "A", 0.99 - i * 0.01) for i in range(20)],
        )
        monkeypatch.setattr(
            ks,
            "_run_source_floor",
            lambda *a, **k: [_item("b1", "B", 0.30), _item("c1", "C", 0.20)],
        )
        results = _run(
            {"method": "vector_search", "min_per_source": 1, "max_per_source": 3}, top_k=5
        )
        sources = {r.source_id for r in results}
        # Without the floor, top-5 would be all 'A'; the back-fill guarantees B & C.
        assert {"A", "B", "C"}.issubset(sources)
        assert sum(1 for r in results if r.source_id == "A") <= 3  # max cap holds

    def test_hybrid_floor_does_not_outrank_main(self, patched_pipeline, monkeypatch):
        # Regression for the score-scale bug: hybrid main pool is RRF-scored
        # (~0.02), floor is cosine (~0.5). Without rescaling, cosine floor items
        # would sort to the TOP. The pipeline must keep main matches on top while
        # still surfacing the floor sources.
        monkeypatch.setitem(
            ks.CHUNK_RETRIEVER_MAP,
            "hybrid",
            lambda **k: [_item(f"a{i}", "A", 0.03 - i * 0.001) for i in range(10)],
        )
        monkeypatch.setattr(
            ks,
            "_run_source_floor",
            lambda *a, **k: [_item("b1", "B", 0.55), _item("c1", "C", 0.50)],
        )
        results = _run(
            {"method": "hybrid", "min_per_source": 1, "max_per_source": 3},
            top_k=5,
            method="hybrid",
        )
        # top result is a genuine main match, NOT a floor item
        assert results[0].source_id == "A"
        assert not (results[0].meta or {}).get("source_floor")
        # diversity still achieved
        assert {"A", "B", "C"}.issubset({r.source_id for r in results})

    def test_reranker_with_limits_reorders_without_pretruncating(
        self, patched_pipeline, monkeypatch
    ):
        # The reranker must see the WHOLE merged pool (main + floor), not be
        # pre-truncated to top_k — else floor items get dropped before selection.
        captured = {}

        def fake_rerank(*, query, results, reranker_config, final_top_k):
            captured["final_top_k"] = final_top_k
            return results  # passthrough, no truncation

        monkeypatch.setattr(ks, "_apply_reranking", fake_rerank)
        monkeypatch.setitem(
            ks.CHUNK_RETRIEVER_MAP,
            "vector_search",
            lambda **k: [_item(f"a{i}", "A", 0.9 - i * 0.01) for i in range(20)],
        )
        monkeypatch.setattr(
            ks, "_run_source_floor", lambda *a, **k: [_item("b1", "B", 0.5), _item("c1", "C", 0.4)]
        )
        _run(
            {
                "method": "vector_search",
                "min_per_source": 1,
                "max_per_source": 3,
                "reranker": {"model": "x", "candidate_count": 20},
            },
            top_k=5,
        )
        # 20 main + 2 floor = 22; reranker reorders all of them
        assert captured["final_top_k"] == 22


@pytest.mark.asyncio
async def test_async_pipeline_applies_floor(monkeypatch):
    # The async pipeline (workflow-engine path) must wire the floor too.
    async def fake_aretriever(**kwargs):
        return [_item(f"a{i}", "A", 0.9 - i * 0.01) for i in range(10)]

    async def fake_afloor(*a, **k):
        return [_item("b1", "B", 0.30), _item("c1", "C", 0.20)]

    monkeypatch.setitem(ks.ASYNC_CHUNK_RETRIEVER_MAP, "vector_search", fake_aretriever)
    monkeypatch.setattr(ks, "_arun_source_floor", fake_afloor)
    monkeypatch.setattr(ks, "_bill_retrieval_action", lambda *a, **k: None)
    monkeypatch.setattr(ks, "_post_retrieval_charge", lambda *a, **k: None)
    monkeypatch.setattr(ks, "_attach_enrichment_metadata", lambda *a, **k: False)

    results = await ks._aexecute_retrieval_pipeline(
        db_session=MagicMock(),
        store=MagicMock(),
        query="q",
        top_k=5,
        retrieval_method="vector_search",
        indexing_config={},
        retrieval_config={"method": "vector_search", "min_per_source": 1, "max_per_source": 3},
        filter_metadata=None,
        knowledge_base_id="kb-1",
        request_id="test-rid",
    )
    assert {"A", "B", "C"}.issubset({r.source_id for r in results})
