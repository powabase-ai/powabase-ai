"""Unit specs for the pg_search keyword-search path and how it is routed.

The store must prefer the pg_search index when the extension is installed AND
this KB's index is ready, and otherwise keep today's behaviour (bm25s file
index, else the bounded tsvector fallback). Detection must never raise on the
search path.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest

from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services import pg_bm25_index as pgb

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


class _ChunkStore(bvs.BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


class _NodeStore(bvs.BasePgVectorStore):
    TABLE = "graph_index_nodes"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


@pytest.fixture(autouse=True)
def _clear_caches():
    pgb.reset_pg_bm25_caches()
    yield
    pgb.reset_pg_bm25_caches()


def _spy_session(rows=()):
    """Session recording (sql, params) and returning `rows` for every execute."""
    session = MagicMock()
    calls: list[tuple[str, dict]] = []

    def execute(statement, params=None):
        calls.append((getattr(statement, "text", str(statement)), params or {}))
        result = MagicMock()
        result.fetchall.return_value = list(rows)
        result.__iter__ = lambda self: iter(rows)
        return result

    session.execute = execute
    session.calls = calls
    return session


def _row(i: int, score: float):
    return (str(uuid.uuid4()), f"text {i}", score, str(uuid.uuid4()), {"k": "v"})


# ---------------------------------------------------------------------------
# The emitted search SQL
# ---------------------------------------------------------------------------


def _capture(store_cls=_ChunkStore, rows=(), **kwargs):
    session = _spy_session(rows)
    store = store_cls(db_session=session, knowledge_base_id=KB)
    items = asyncio.run(store.pg_bm25_search("Beschwerde", top_k=7, **kwargs))
    return session.calls, items


def test_search_sql_uses_the_kb_literal_so_the_partial_index_matches():
    calls, _ = _capture()
    sql, params = calls[-1]
    assert f"knowledge_base_id = '{KB}'" in sql
    assert "kb_id" not in params


def test_search_sql_uses_the_match_operator_and_score_ordering():
    calls, _ = _capture()
    sql, params = calls[-1]
    assert "c.text ||| :bm25_query" in sql
    assert "pdb.score(c.id)" in sql
    assert "ORDER BY pdb.score(c.id) DESC" in sql
    assert "LIMIT :top_k" in sql
    assert params["top_k"] == 7
    assert params["bm25_query"] == "Beschwerde"


def test_search_sql_for_an_expression_indexed_table():
    calls, _ = _capture(store_cls=_NodeStore)
    sql, _ = calls[-1]
    assert "(COALESCE(c.title, '') || ' ' || COALESCE(c.text, '')) ||| :bm25_query" in sql


def test_source_ids_are_bound_as_a_uuid_array():
    sid = str(uuid.uuid4())
    calls, _ = _capture(source_ids=[sid])
    sql, params = calls[-1]
    assert "c.source_id = ANY(CAST(:source_ids AS uuid[]))" in sql
    assert params["source_ids"] == "{" + sid + "}"


def test_item_ids_are_bound_as_a_uuid_array():
    iid = str(uuid.uuid4())
    calls, _ = _capture(item_ids={iid})
    sql, params = calls[-1]
    assert "c.id = ANY(CAST(:item_ids AS uuid[]))" in sql
    assert params["item_ids"] == "{" + iid + "}"


def test_filter_metadata_is_bound_as_jsonb_containment():
    calls, _ = _capture(filter_metadata={"lang": "de"})
    sql, params = calls[-1]
    assert "c.meta @> CAST(:filter_lang AS jsonb)" in sql
    assert params["filter_lang"] == '{"lang": "de"}'


def test_scores_land_on_the_retrieved_items():
    _, items = _capture(rows=[_row(1, 3.5), _row(2, 1.25)])
    assert [i.score for i in items] == [3.5, 1.25]
    assert items[0].knowledge_base_id == KB


def test_a_non_uuid_kb_id_is_refused_rather_than_interpolated():
    session = _spy_session()
    store = _ChunkStore(db_session=session, knowledge_base_id="'; DROP TABLE ai.chunks; --")
    with pytest.raises(ValueError):
        asyncio.run(store.pg_bm25_search("q", top_k=5))
    assert session.calls == []


@pytest.mark.parametrize(
    "adversarial",
    [
        "it's",
        'a "quoted" phrase',
        "field:value",
        "back\\slash",
        "' OR 1=1; DROP TABLE ai.chunks; --",
        "Besch\x00werde",
        "a" * 40_000,
        "AND OR NOT ~ ^ + - ( ) [ ] { } /",
    ],
)
def test_adversarial_queries_are_bound_never_interpolated(adversarial):
    session = _spy_session()
    store = _ChunkStore(db_session=session, knowledge_base_id=KB)
    asyncio.run(store.pg_bm25_search(adversarial, top_k=5))
    for sql, params in session.calls:
        assert adversarial[:40] not in sql
        assert "\x00" not in params.get("bm25_query", "")


def test_a_blank_query_returns_empty_without_touching_the_database():
    session = _spy_session()
    store = _ChunkStore(db_session=session, knowledge_base_id=KB)
    assert asyncio.run(store.pg_bm25_search("   ", top_k=5)) == []
    assert session.calls == []


# ---------------------------------------------------------------------------
# Routing: extension present/absent x index ready/not
# ---------------------------------------------------------------------------


def _routed_store(installed: bool, ready: bool):
    store = _ChunkStore(db_session=_spy_session(), knowledge_base_id=KB)
    calls: list[str] = []

    async def pg(*a, **k):
        calls.append("pg")
        return []

    async def fallback(*a, **k):
        calls.append("tsvector")
        return []

    store.pg_bm25_search = pg
    store.full_text_search = fallback
    patches = [
        patch.object(bvs.pg_bm25_index, "pg_search_installed", return_value=installed),
        patch.object(bvs.pg_bm25_index, "bm25_index_ready", return_value=ready),
    ]
    return store, calls, patches


def _run_bm25s(store, patches):
    for p in patches:
        p.start()
    try:
        return asyncio.run(store.bm25s_search("Beschwerde", top_k=5))
    finally:
        for p in patches:
            p.stop()


def test_pg_path_is_used_when_extension_installed_and_index_ready():
    store, calls, patches = _routed_store(installed=True, ready=True)
    _run_bm25s(store, patches)
    assert calls == ["pg"]


def test_pg_path_is_skipped_when_the_index_is_not_ready():
    store, calls, patches = _routed_store(installed=True, ready=False)
    _run_bm25s(store, patches)
    assert calls == ["tsvector"]


def test_pg_path_is_skipped_when_the_extension_is_absent():
    store, calls, patches = _routed_store(installed=False, ready=True)
    _run_bm25s(store, patches)
    assert calls == ["tsvector"]


def test_index_readiness_is_not_probed_when_the_extension_is_absent():
    store, _calls, _p = _routed_store(installed=False, ready=True)
    with (
        patch.object(bvs.pg_bm25_index, "pg_search_installed", return_value=False),
        patch.object(bvs.pg_bm25_index, "bm25_index_ready") as ready,
    ):
        asyncio.run(store.bm25s_search("q", top_k=5))
    ready.assert_not_called()


def test_detection_failure_falls_back_instead_of_raising():
    store, calls, _p = _routed_store(installed=True, ready=True)
    with patch.object(
        bvs.pg_bm25_index, "pg_search_installed", side_effect=RuntimeError("catalog gone")
    ):
        asyncio.run(store.bm25s_search("q", top_k=5))
    assert calls == ["tsvector"]


def test_a_failing_pg_query_falls_back_to_todays_behaviour():
    store = _ChunkStore(db_session=_spy_session(), knowledge_base_id=KB)
    calls: list[str] = []

    async def pg(*a, **k):
        raise RuntimeError("`chunks` does not contain a `USING bm25` index")

    async def fallback(*a, **k):
        calls.append("tsvector")
        return []

    store.pg_bm25_search = pg
    store.full_text_search = fallback
    with (
        patch.object(bvs.pg_bm25_index, "pg_search_installed", return_value=True),
        patch.object(bvs.pg_bm25_index, "bm25_index_ready", return_value=True),
    ):
        asyncio.run(store.bm25s_search("q", top_k=5))
    assert calls == ["tsvector"]


def test_a_keyword_timeout_from_the_fallback_still_propagates():
    """full_text retrieval must keep answering 503 when the fallback times out."""
    store = _ChunkStore(db_session=_spy_session(), knowledge_base_id=KB)

    async def fallback(*a, **k):
        raise bvs.KeywordSearchTimeout(KB, 10_000)

    store.full_text_search = fallback
    with patch.object(bvs.pg_bm25_index, "pg_search_installed", return_value=False):
        with pytest.raises(bvs.KeywordSearchTimeout):
            asyncio.run(store.bm25s_search("q", top_k=5))


def test_hybrid_keyword_leg_uses_the_pg_path_when_available():
    store = _ChunkStore(db_session=_spy_session(), knowledge_base_id=KB)
    calls: list[str] = []

    async def pg(*a, **k):
        calls.append("pg")
        return []

    store.pg_bm25_search = pg
    with (
        patch.object(bvs.pg_bm25_index, "pg_search_installed", return_value=True),
        patch.object(bvs.pg_bm25_index, "bm25_index_ready", return_value=True),
    ):
        asyncio.run(
            store.keyword_search_for_hybrid(
                "q", top_k=4, filter_metadata=None, item_ids=None, source_ids=None
            )
        )
    assert calls == ["pg"]
