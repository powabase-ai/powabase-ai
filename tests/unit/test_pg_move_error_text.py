"""Reading errors without tripping over them.

Handlers on the move and indexing paths promise never to raise, and a few
decide what an error *is* from its text. Neither may depend on a message being
there, or being in English.
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import MagicMock, patch

import psycopg
import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services import pg_bm25_index as pgb
from agentic_project_service.tasks import indexing

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
HEX = "3f2504e04f8911d39a0c0305e82c3301"


class _Diag:
    def __init__(self, constraint_name=None, table_name=None, schema_name=None):
        self.constraint_name = constraint_name
        self.table_name = table_name
        self.schema_name = schema_name


def _check_violation(message, diag=None):
    orig = psycopg.errors.CheckViolation(message)
    wrapped = IntegrityError("INSERT INTO ai.chunks ...", {}, orig)
    if diag is not None:
        # psycopg's diag is read-only on real errors; the handlers only read it.
        wrapped.orig = MagicMock(sqlstate="23514", diag=diag, __str__=lambda self: message)
    return wrapped


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError(""),
        OperationalError("stmt", {}, Exception("")),
        OperationalError("stmt", {}, Exception("\n")),
    ],
)
def test_the_first_line_of_an_empty_message_is_the_error_type(exc):
    assert pgb.first_error_line(exc)


def test_a_move_race_is_recognised_by_its_constraint_name_in_any_language():
    localised = (
        "la nouvelle ligne de la relation « chunks_default » viole la contrainte de vérification"
    )
    race = _check_violation(
        localised,
        _Diag(f"bm25_move_{HEX}", "chunks_default", "ai"),
    )
    assert pgb.is_partition_move_race(race) is True


def test_a_partition_constraint_violation_carries_no_constraint_name():
    race = _check_violation(
        "die neue Zeile verletzt die Partitionsbedingung", _Diag(None, "chunks_default", "ai")
    )
    assert pgb.is_partition_move_race(race) is True


def test_an_ordinary_check_is_not_a_race_even_if_its_message_looks_like_one():
    ordinary = _check_violation(
        'new row for relation "chunks" violates check constraint "bm25_move_look_alike"',
        _Diag("tokens_positive", "chunks", "ai"),
    )
    assert pgb.is_partition_move_race(ordinary) is False


def test_refusal_tracing_and_clearing_never_raise_on_an_empty_message():
    empty = _check_violation("")
    assert pgb.move_check_refusal_item_table(empty) is None
    assert pgb.clear_move_check_after_refusal(MagicMock(), empty) == []


def test_a_failed_fence_drop_with_an_empty_error_is_logged_not_raised(monkeypatch):
    monkeypatch.setattr(pgb, "MOVE_CHECK_CLEANUP_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(
        pgb, "_drop_move_checks", MagicMock(side_effect=OperationalError("x", {}, Exception("")))
    )
    pgb._drop_failed_move_check(MagicMock(), KB, "chunks")


def test_the_indexing_requeue_handler_survives_an_empty_lock_error(monkeypatch):
    monkeypatch.setattr(indexing, "db", MagicMock())
    monkeypatch.setattr(indexing, "get_knowledge_base", lambda _id: {"indexing_config": {}})
    monkeypatch.setattr(indexing, "get_source", lambda _id: {"extraction_status": "extracted"})
    monkeypatch.setattr(indexing, "_claim_indexed_source", lambda *_a: 1)
    requeue = MagicMock()
    monkeypatch.setattr(indexing, "_handle_storage_error", requeue)

    class _Orig(Exception):
        sqlstate = "55P03"

    def _raise(**_kwargs):
        raise OperationalError("stmt", {}, _Orig(""))

    monkeypatch.setattr(indexing, "_run_index_body", _raise)
    indexing.index_source.run(KB, "src", indexed_source_id="is-1")
    requeue.assert_called_once()


def test_the_bm25_rebuild_scan_names_the_kb_on_the_item_table_itself(monkeypatch):
    """The join to indexed_sources alone cannot prune the partitions."""
    seen: list[str] = []
    session = MagicMock()
    session.execute.side_effect = lambda sql, params=None: seen.append(sql.text) or iter([])
    monkeypatch.setattr(indexing, "db", MagicMock(session=session))
    for item_table, alias in (("chunks", "c"), ("full_documents", "d"), ("graph_index_nodes", "n")):
        seen.clear()
        list(indexing._iter_items_for_kb_bm25(KB, item_table))
        assert re.search(rf"\b{alias}\.knowledge_base_id = :kb\b", seen[0]), seen[0]


def test_pg_search_failure_warnings_are_bounded_and_keyed_by_cause(monkeypatch, caplog):
    monkeypatch.setattr(bvs, "_WARNED_PG_BM25_FAILURES", set())
    monkeypatch.setattr(bvs, "_WARNED_PG_BM25_FAILURES_MAX", 4)

    class _Store(bvs.BasePgVectorStore):
        TABLE = "chunks"
        TEXT_COL = "text"
        SEARCH_TEXT_COL = "text"

    async def fallback(*a, **k):
        return []

    def search(kb_id, cause):
        store = _Store(db_session=MagicMock(), knowledge_base_id=kb_id)

        async def pg(*a, **k):
            raise RuntimeError(cause)

        store.pg_bm25_search = pg
        store.full_text_search = fallback
        store._file_index_retired = lambda: True
        asyncio.run(store.bm25s_search("q", top_k=1))

    with (
        caplog.at_level("WARNING", logger=bvs.logger.name),
        patch.object(bvs.pg_bm25_index, "pg_search_installed", return_value=True),
        patch.object(bvs.pg_bm25_index, "bm25_index_ready", return_value=True),
    ):
        search(KB, "first cause")
        search(KB, "second cause")
        for i in range(20):
            search(f"00000000-0000-4000-8000-{i:012d}", "first cause")

    warned = [r.getMessage() for r in caplog.records if "keyword search failed" in r.getMessage()]
    assert any("first cause" in m for m in warned) and any("second cause" in m for m in warned)
    assert len(bvs._WARNED_PG_BM25_FAILURES) <= 4
