"""Small things the pre-push review of the round-3 fixes found.

- A strategy change must not drop a working bm25 index on a server that could
  not build it again.
- A connection refused for bad credentials is not a transient failure.
- The search path says, once per knowledge base, when a KB with its own
  partition answers keyword search from the slow tsvector fallback, and
  neither file-index check swallows an error without a word.
"""

from __future__ import annotations


import logging
from unittest.mock import MagicMock, patch

import psycopg
import pytest
from sqlalchemy.exc import OperationalError

from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services import pg_bm25_index as pgb
from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY
from agentic_project_service.tasks import indexing
from tests.unit.test_pg_bm25_lifecycle import KB, _FakeConn, _FakeEngine, _with_partition
from tests.unit.test_pg_bm25_search_routing import _file_index_store, _run_bm25s


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(pgb, "_unsafe_build_warning_logged", False)
    monkeypatch.setattr(pgb, "_concurrent_build_override", lambda: False)
    bvs._WARNED_PG_BM25_FAILURES.clear()
    pgb.reset_pg_bm25_caches()
    yield
    bvs._WARNED_PG_BM25_FAILURES.clear()


# --- M-1 --------------------------------------------------------------------


def _drops(monkeypatch) -> list:
    dropped: list = []
    monkeypatch.setattr(
        pgb,
        "_drop_indexes_on_other_item_tables",
        lambda conn, kb_id, item_table: dropped.append(item_table) or ["bm25_chunks_x"],
    )
    return dropped


def test_an_unsafe_server_keeps_the_index_a_strategy_change_left_behind(monkeypatch):
    dropped = _drops(monkeypatch)
    conn = _FakeConn(
        kb_row=("full_document", "hybrid", "german"),
        relkinds=_with_partition(item_table="full_documents"),
        build_safe=False,
    )
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["status"] == "unavailable"
    assert dropped == []
    assert "dropped_indexes" not in out


def test_a_safe_server_still_drops_it(monkeypatch):
    dropped = _drops(monkeypatch)
    conn = _FakeConn(
        kb_row=("full_document", "hybrid", "german"),
        relkinds=_with_partition(item_table="full_documents"),
    )
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert dropped == ["full_documents"]
    assert out["dropped_indexes"] == ["bm25_chunks_x"]


# --- M-7 --------------------------------------------------------------------


def _connect_failure(message):
    return OperationalError(None, None, psycopg.OperationalError(message))


@pytest.mark.parametrize(
    "message",
    [
        'connection failed: FATAL:  password authentication failed for user "service"',
        'connection failed: FATAL:  role "service" does not exist',
        'connection failed: FATAL:  database "project" does not exist',
        'connection failed: FATAL:  no pg_hba.conf entry for host "10.0.0.1"',
        "connection failed: fe_sendauth: no password supplied",
        'connection failed: FATAL:  permission denied for database "project"',
    ],
)
def test_a_connection_refused_for_its_credentials_or_target_is_not_transient(message):
    assert pgb.is_connect_failure(_connect_failure(message)) is False
    assert pgb.is_transient_db_error(_connect_failure(message)) is False


@pytest.mark.parametrize(
    "message",
    [
        "connection failed: FATAL:  the database system is starting up",
        "connection failed: FATAL:  the database system is in recovery mode",
        'connection failed: connection to server at "10.0.0.1", port 5432 failed: Connection refused',
        "connection failed: could not translate host name",
    ],
)
def test_a_server_that_is_down_or_recovering_is_still_transient(message):
    assert pgb.is_transient_db_error(_connect_failure(message)) is True


# --- round-2 leftovers --------------------------------------------------------


def test_the_auto_indexing_setting_says_what_it_does_with_pg_search():
    description = SETTINGS_REGISTRY["BM25_AUTO_INDEXING"].description
    assert "bm25s" in description
    assert "pg_search" in description
    assert "KB detail page" not in description


def test_a_moved_kb_answering_from_the_tsvector_fallback_warns_once(monkeypatch, caplog):
    with caplog.at_level(logging.DEBUG, logger=bvs.logger.name):
        for _ in range(3):
            store, calls, patches = _file_index_store(monkeypatch, partition=True)
            _run_bm25s(store, patches)
    warnings = [
        r for r in caplog.records if r.levelno == logging.WARNING and "tsvector" in r.getMessage()
    ]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert KB in warnings[0].getMessage()


def test_the_stores_file_index_check_logs_the_error_it_hides(caplog):
    store = bvs.BasePgVectorStore.__new__(bvs.BasePgVectorStore)
    store.TABLE = "chunks"
    store.kb_id = KB
    store.session = MagicMock()
    with (
        caplog.at_level(logging.WARNING, logger=bvs.logger.name),
        patch.object(pgb, "pg_search_installed", side_effect=RuntimeError("catalog gone")),
    ):
        assert store._file_index_retired() is False
    assert [r for r in caplog.records if "catalog gone" in r.getMessage()]


def test_the_indexing_file_index_check_logs_the_error_it_hides(monkeypatch, caplog):
    monkeypatch.setattr(
        indexing, "_get_kb_indexing_strategy", MagicMock(side_effect=RuntimeError("no row"))
    )
    with caplog.at_level(logging.WARNING, logger=indexing.logger.name):
        assert indexing._kb_file_index_retired(KB) is False
    assert [r for r in caplog.records if "no row" in r.getMessage()]
