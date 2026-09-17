"""Unit tests for bm25_build_outcome's never-raises paths (fake binds only).

Live upsert/read ordering and savepoint isolation against a real Postgres
are covered separately in ``tests/pg_search``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

from agentic_project_service.services.bm25_build_outcome import (
    STATUSES,
    read_bm25_build_outcome,
    record_bm25_build_outcome,
)

KB_ID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


def _fake_engine():
    engine = MagicMock(spec=Engine)
    conn = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    engine.begin.return_value.__exit__.return_value = False
    return engine, conn


def _fake_connection():
    conn = MagicMock(spec=Connection)
    conn.begin_nested.return_value.__enter__.return_value = None
    conn.begin_nested.return_value.__exit__.return_value = False
    return conn


def _fake_session():
    session = MagicMock(spec=Session)
    session.begin_nested.return_value.__enter__.return_value = None
    session.begin_nested.return_value.__exit__.return_value = False
    return session


def test_statuses_are_exactly_the_documented_ones():
    assert STATUSES == frozenset(
        {
            "queued",
            "moving",
            "building",
            "completing",
            "ready",
            "retrying",
            "failed",
            "needs_build",
            "unavailable",
        }
    )


# ---------------------------------------------------------------------------
# record_bm25_build_outcome
# ---------------------------------------------------------------------------


def test_record_uses_the_engines_own_transaction():
    engine, conn = _fake_engine()

    record_bm25_build_outcome(engine, KB_ID, "chunks", "ready")

    engine.begin.assert_called_once()
    conn.execute.assert_called_once()


def test_record_uses_a_savepoint_on_a_connection():
    conn = _fake_connection()

    record_bm25_build_outcome(conn, KB_ID, "chunks", "ready")

    conn.begin_nested.assert_called_once()
    conn.execute.assert_called_once()


def test_record_uses_a_savepoint_on_a_session():
    session = _fake_session()

    record_bm25_build_outcome(session, KB_ID, "chunks", "ready")

    session.begin_nested.assert_called_once()
    session.execute.assert_called_once()


def test_record_never_raises_and_logs_a_warning_when_the_engine_write_fails(caplog):
    engine, conn = _fake_engine()
    conn.execute.side_effect = RuntimeError("relation ai.bm25_index_builds does not exist")

    with caplog.at_level("WARNING"):
        record_bm25_build_outcome(engine, KB_ID, "chunks", "ready")

    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_record_never_raises_and_logs_a_warning_when_the_savepoint_write_fails(caplog):
    conn = _fake_connection()
    conn.execute.side_effect = RuntimeError("boom")

    with caplog.at_level("WARNING"):
        record_bm25_build_outcome(conn, KB_ID, "chunks", "ready")

    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_record_rejects_an_unknown_status_without_touching_the_bind(caplog):
    engine, conn = _fake_engine()

    with caplog.at_level("WARNING"):
        record_bm25_build_outcome(engine, KB_ID, "chunks", "not-a-real-status")

    engine.begin.assert_not_called()
    conn.execute.assert_not_called()
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_record_rejects_an_invalid_kb_id_without_touching_the_bind(caplog):
    engine, conn = _fake_engine()

    with caplog.at_level("WARNING"):
        record_bm25_build_outcome(engine, "not-a-uuid", "chunks", "ready")

    engine.begin.assert_not_called()
    conn.execute.assert_not_called()
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_record_passes_reason_and_attempts_through():
    engine, conn = _fake_engine()

    record_bm25_build_outcome(
        engine, KB_ID, "chunks", "retrying", reason="lock conflict", attempts=3
    )

    _, params = conn.execute.call_args.args
    assert params["reason"] == "lock conflict"
    assert params["attempts"] == 3
    assert params["status"] == "retrying"
    assert params["item_table"] == "chunks"
    assert params["kb_id"] == KB_ID


# ---------------------------------------------------------------------------
# read_bm25_build_outcome
# ---------------------------------------------------------------------------


def test_read_returns_none_for_an_invalid_kb_id_without_touching_the_bind():
    engine, conn = _fake_engine()

    result = read_bm25_build_outcome(engine, "not-a-uuid")

    assert result is None
    engine.begin.assert_not_called()
    conn.execute.assert_not_called()


def test_read_never_raises_and_returns_none_when_the_query_fails(caplog):
    engine, conn = _fake_engine()
    conn.execute.side_effect = RuntimeError("relation ai.bm25_index_builds does not exist")

    with caplog.at_level("WARNING"):
        result = read_bm25_build_outcome(engine, KB_ID)

    assert result is None
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_read_returns_none_when_there_is_no_row():
    conn = _fake_connection()
    conn.execute.return_value.mappings.return_value.first.return_value = None

    result = read_bm25_build_outcome(conn, KB_ID)

    assert result is None


def test_read_maps_the_row_to_the_documented_keys():
    conn = _fake_connection()
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "status": "ready",
        "reason": None,
        "item_table": "chunks",
        "attempts": 2,
        "updated_at": "2026-09-16T00:00:00+00:00",
    }

    result = read_bm25_build_outcome(conn, KB_ID)

    assert result == {
        "status": "ready",
        "reason": None,
        "item_table": "chunks",
        "attempts": 2,
        "updated_at": "2026-09-16T00:00:00+00:00",
    }
