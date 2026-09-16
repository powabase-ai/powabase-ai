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


class _FullDocumentStore(bvs.BasePgVectorStore):
    TABLE = "full_documents"
    TEXT_COL = "summary"
    SEARCH_TEXT_COL = "summary"


class _Doc2JsonStore(bvs.BasePgVectorStore):
    TABLE = "doc2json_documents"
    TEXT_COL = "summary"
    SEARCH_TEXT_COL = "summary"


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
    items = asyncio.run(store.pg_bm25_search("Wanderung", top_k=7, **kwargs))
    return session.calls, items


def test_search_sql_names_the_kbs_partition_and_nothing_else():
    """The partition bound is the knowledge-base filter.

    A scored query has to name a relation that carries a bm25 index, and the
    parent of a partitioned table never does -- pg_search refuses it outright,
    predicate or no predicate. So the KB is expressed as the relation, and no
    ``knowledge_base_id`` predicate is needed or wanted.
    """
    calls, _ = _capture()
    sql, params = calls[-1]
    assert f'FROM "ai".chunks_kb_{uuid.UUID(KB).hex} c' in sql
    assert "knowledge_base_id" not in sql
    assert "kb_id" not in params


def test_search_sql_derives_the_partition_from_a_validated_uuid():
    """Nothing a caller supplies reaches the relation name unvalidated."""
    calls, _ = _capture()
    sql, _ = calls[-1]
    assert pgb.partition_name(KB, "chunks") in sql
    assert "-" not in sql.split(" c\n")[0].split("chunks_kb_")[-1][:32]


def test_search_sql_for_full_documents_names_its_own_partition():
    calls, _ = _capture(store_cls=_FullDocumentStore)
    sql, _ = calls[-1]
    assert f'FROM "ai".full_documents_kb_{uuid.UUID(KB).hex} c' in sql
    assert "c.summary ||| :bm25_query" in sql


def test_search_sql_uses_the_match_operator_and_score_ordering():
    calls, _ = _capture()
    sql, params = calls[-1]
    assert "c.text ||| :bm25_query" in sql
    assert "pdb.score(c.id)" in sql
    assert "ORDER BY pdb.score(c.id) DESC" in sql
    assert "LIMIT :top_k" in sql
    assert params["top_k"] == 7
    assert params["bm25_query"] == "Wanderung"


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


def test_a_table_with_no_partition_never_reaches_the_database():
    """``doc2json_documents`` has no partition, so it has no scored query.

    Raising here rather than emitting a query against the bare table is what
    keeps the routing honest: ``bm25s_search`` catches it and falls back, where
    a parent-table query would have raised inside Postgres instead.
    """
    session = _spy_session()
    store = _Doc2JsonStore(db_session=session, knowledge_base_id=KB)
    with pytest.raises(ValueError):
        asyncio.run(store.pg_bm25_search("Wanderung", top_k=5))
    assert session.calls == []


def test_a_kb_without_a_partition_falls_back_instead_of_erroring():
    store = _Doc2JsonStore(db_session=_spy_session(), knowledge_base_id=KB)
    calls: list[str] = []

    async def fallback(*a, **k):
        calls.append("tsvector")
        return []

    store.full_text_search = fallback
    with (
        patch.object(bvs.pg_bm25_index, "pg_search_installed", return_value=True),
        patch.object(bvs.pg_bm25_index, "bm25_index_ready", return_value=False),
    ):
        asyncio.run(store.bm25s_search("Wanderung", top_k=5))
    assert calls == ["tsvector"]


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
        return asyncio.run(store.bm25s_search("Wanderung", top_k=5))
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


def test_a_failing_pg_query_warns_once_with_its_cause_and_no_traceback(caplog, monkeypatch):
    """The fallback runs on every search while the pg path keeps failing, so a
    traceback per search would flood the log. One WARNING names the cause; the
    repeats go to DEBUG."""
    monkeypatch.setattr(bvs, "_WARNED_TIMEOUT_OVERRIDES", set())
    store = _ChunkStore(db_session=_spy_session(), knowledge_base_id=KB)

    async def pg(*a, **k):
        raise RuntimeError("`chunks_kb_x` does not contain a `USING bm25` index")

    async def fallback(*a, **k):
        return []

    store.pg_bm25_search = pg
    store.full_text_search = fallback
    with (
        caplog.at_level("DEBUG", logger=bvs.logger.name),
        patch.object(bvs.pg_bm25_index, "pg_search_installed", return_value=True),
        patch.object(bvs.pg_bm25_index, "bm25_index_ready", return_value=True),
    ):
        for _ in range(3):
            asyncio.run(store.bm25s_search("q", top_k=5))

    records = [r for r in caplog.records if "pg_search keyword search failed" in r.getMessage()]
    warnings = [r for r in records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "does not contain a `USING bm25` index" in warnings[0].getMessage()
    assert warnings[0].exc_info is None
    assert len(records) == 3
