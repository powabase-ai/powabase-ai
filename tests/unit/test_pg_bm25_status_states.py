"""What a knowledge base's recorded BM25 build status says, in every state.

A status nobody updates, a ``ready`` that hides unfinished work, an ordinary
skip that reads as a failure, and a reason that names an internal step
instead of what to do: each misleads whoever reads ``bm25_status``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg
import pytest
from celery.exceptions import Retry
from sqlalchemy.exc import IntegrityError, OperationalError

from agentic_project_service.routes import knowledge_bases as kb_route
from agentic_project_service.services import bm25_build_outcome
from agentic_project_service.services import pg_bm25_index as pgb
from agentic_project_service.tasks import indexing
from tests.unit.test_pg_bm25_lifecycle import (
    KB,
    _FakeConn,
    _FakeEngine,
    _with_partition,
)

SERVICE = "agentic_project_service.services.pg_bm25_index"
R = "agentic_project_service.routes.knowledge_bases"

_EXISTING_GERMAN = (
    "CREATE INDEX bm25_chunks_x ON ai.chunks_kb_x USING bm25 "
    "(id, ((text)::pdb.simple('stemmer=german')), source_id, meta) WITH (key_field=id)"
)


def _connect_failure(message="connection failed: FATAL:  the database system is starting up"):
    """What SQLAlchemy raises when the pool cannot open a connection: no statement, no SQLSTATE."""
    return OperationalError(None, None, psycopg.OperationalError(message))


@pytest.fixture
def task(monkeypatch):
    recorded: list[tuple] = []

    def record(bind, kb_id, item_table, status, reason=None, attempts=None):
        recorded.append((status, reason, item_table))

    monkeypatch.setattr(indexing, "record_bm25_build_outcome", record)
    monkeypatch.setattr(indexing, "db", MagicMock())
    monkeypatch.setattr(pgb, "keyword_item_table", lambda bind, kb_id: "chunks")
    retry = MagicMock(side_effect=Retry("retry"))
    monkeypatch.setattr(indexing.ensure_pg_bm25_index, "retry", retry)
    return indexing.ensure_pg_bm25_index, retry, recorded


# ---------------------------------------------------------------------------
# A failure to reconnect is transient
# ---------------------------------------------------------------------------


def test_a_failure_to_open_a_connection_is_transient():
    assert pgb.is_transient_db_error(_connect_failure()) is True


def test_an_operational_error_from_a_statement_still_needs_a_sqlstate_or_a_lost_connection():
    error = OperationalError("SELECT 1", {}, psycopg.OperationalError("something odd"))
    assert pgb.is_transient_db_error(error) is False


def test_a_retry_that_cannot_reconnect_is_retried_again(task):
    ensure, retry, recorded = task
    with patch(f"{SERVICE}.ensure_bm25_index", side_effect=_connect_failure()):
        with pytest.raises(Retry):
            ensure.run(KB)
    retry.assert_called_once()
    status, reason, _ = recorded[-1]
    assert status == "retrying"
    assert "could not connect to the database" in reason


def test_the_item_table_is_resolved_again_once_the_database_answers(task, monkeypatch):
    """A run that starts while the server recovers cannot read the KB's item
    table; its outcome must still be recorded once it can."""
    ensure, _retry, recorded = task
    answers = iter([None, "chunks", "chunks"])
    monkeypatch.setattr(pgb, "keyword_item_table", lambda bind, kb_id: next(answers))
    with patch(
        f"{SERVICE}.ensure_bm25_index", return_value={"status": "ready", "item_table": "chunks"}
    ):
        ensure.run(KB)
    assert [status for status, *_ in recorded] == ["ready"]


def test_an_unreadable_keyword_item_table_is_logged_as_a_warning(caplog):
    session = MagicMock()
    session.execute.side_effect = _connect_failure()
    with caplog.at_level(logging.WARNING, logger=pgb.logger.name):
        assert pgb.keyword_item_table(session, KB) is None
    assert [r for r in caplog.records if r.levelno == logging.WARNING and KB in r.getMessage()]


# ---------------------------------------------------------------------------
# Every waiting status ages
# ---------------------------------------------------------------------------


def _aged(status, seconds):
    return {
        "status": status,
        "reason": "why",
        "item_table": "chunks",
        "attempts": 1,
        "updated_at": datetime.now(timezone.utc) - timedelta(seconds=seconds),
    }


@pytest.mark.parametrize("status", ["queued", "retrying", "moving", "building", "completing"])
def test_a_waiting_status_silent_for_too_long_is_stale(status):
    got, reason = kb_route._stale_or_recorded_status(
        _aged(status, kb_route.BM25_BUILD_OUTCOME_STALE_SECONDS + 60)
    )
    assert got == "stale"
    assert repr(status) in reason


@pytest.mark.parametrize("status", ["failed", "unavailable", "needs_build", "ready"])
def test_a_settled_status_does_not_age(status):
    got, _ = kb_route._stale_or_recorded_status(
        _aged(status, kb_route.BM25_BUILD_OUTCOME_STALE_SECONDS + 60)
    )
    assert got == status


# ---------------------------------------------------------------------------
# ``ready`` with the partition's post-commit work unfinished
# ---------------------------------------------------------------------------


def _detail(*, pg_state="ready", pending=False, outcome=None):
    with (
        patch(f"{R}.db"),
        patch(f"{R}._keyword_index_backend", return_value="pg_search"),
        patch(f"{R}.pg_bm25_status", return_value=pg_state),
        patch(f"{SERVICE}.partition_completion_pending", return_value=pending),
        patch(f"{R}.read_bm25_build_outcome", return_value=outcome),
        patch(f"{R}.get_setting", return_value=True),
    ):
        return kb_route._bm25_status_detail({"id": KB, **_KB})


_KB = {
    "indexing_config": {"strategy": "chunk_embed"},
    "retrieval_config": {"method": "hybrid"},
}


def _recorded(status, reason="why"):
    return {
        "status": status,
        "reason": reason,
        "item_table": "chunks",
        "attempts": 1,
        "updated_at": datetime.now(timezone.utc),
    }


def test_a_ready_index_on_an_unfinished_partition_reports_completing():
    status, reason = _detail(pending=True)
    assert status == "completing"
    assert "keyword search is served" in reason
    assert "deleting" in reason


@pytest.mark.parametrize("recorded", ["retrying", "failed", "completing"])
def test_an_unfinished_partition_reports_what_its_last_run_recorded(recorded):
    assert _detail(pending=True, outcome=_recorded(recorded)) == (recorded, "why")


def test_a_finished_partition_with_a_ready_index_is_ready_whatever_was_recorded():
    assert _detail(pending=False, outcome=_recorded("failed")) == ("ready", None)


def test_a_ready_index_the_server_cannot_rebuild_for_a_new_language_reports_unavailable():
    got = _detail(pending=False, outcome=_recorded("unavailable", "not safe here"))
    assert got == ("unavailable", "not safe here")


def test_the_ensure_says_completing_before_it_completes_a_partition(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(pgb, "_partition_completion_pending", lambda conn, kb, table: True)
    monkeypatch.setattr(
        pgb, "_complete_partition", lambda conn, kb, table: seen.append("complete") or {}
    )
    conn = _FakeConn(relkinds=_with_partition(), indexdef=_EXISTING_GERMAN)
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn), on_progress=seen.append)
    assert out["status"] == "ready"
    assert seen == ["completing", "complete"]


def test_nothing_to_complete_records_no_completing(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(pgb, "_partition_completion_pending", lambda conn, kb, table: False)
    conn = _FakeConn(relkinds=_with_partition(), indexdef=_EXISTING_GERMAN)
    pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn), on_progress=seen.append)
    assert "completing" not in seen


def test_start_up_dispatches_an_ensure_for_every_unfinished_partition(monkeypatch):
    other = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
    monkeypatch.setattr(
        pgb,
        "partitions_needing_completion",
        lambda engine: [(KB, "chunks"), (other, "chunks"), (KB, "graph_index_nodes")],
    )
    delay = MagicMock()
    monkeypatch.setattr(indexing.ensure_pg_bm25_index, "delay", delay)
    assert indexing.dispatch_partition_completion_at_start(MagicMock()) == [KB, other]
    assert [c.args for c in delay.call_args_list] == [(KB,), (other,)]


def test_the_start_up_dispatch_never_raises(monkeypatch):
    monkeypatch.setattr(
        pgb, "partitions_needing_completion", MagicMock(side_effect=RuntimeError("no db"))
    )
    assert indexing.dispatch_partition_completion_at_start(MagicMock()) == []


def test_partition_completion_pending_never_raises():
    session = MagicMock()
    session.execute.side_effect = RuntimeError("gone")
    assert pgb.partition_completion_pending(session, KB, "chunks") is False


# ---------------------------------------------------------------------------
# A deliberate skip is not a failure
# ---------------------------------------------------------------------------


def test_needs_build_and_completing_are_statuses_the_table_accepts():
    for status in ("needs_build", "completing"):
        assert status in bm25_build_outcome.STATUSES
    migration = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "0032_add_bm25_index_builds_table.py"
    ).read_text()
    assert "'needs_build'" in migration and "'completing'" in migration


def test_rows_found_by_an_automatic_run_record_needs_build_at_info(task, caplog):
    ensure, retry, recorded = task
    with (
        caplog.at_level(logging.INFO),
        patch(
            f"{SERVICE}.ensure_bm25_index",
            return_value={"status": "skipped", "reason": "row_move_not_allowed"},
        ),
    ):
        ensure.run(KB)
    status, reason, _ = recorded[-1]
    assert status == "needs_build"
    assert "POST /build-bm25" in reason
    retry.assert_not_called()
    assert "POST /build-bm25" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_needs_build_record_is_reported():
    with (
        patch(f"{R}.db"),
        patch(f"{R}._keyword_index_backend", return_value="pg_search"),
        patch(f"{R}.pg_bm25_status", return_value="absent"),
        patch(f"{R}.read_bm25_build_outcome", return_value=_recorded("needs_build", "move it")),
    ):
        assert kb_route._bm25_status_detail({"id": KB, **_KB}) == ("needs_build", "move it")


# ---------------------------------------------------------------------------
# Reasons say what happened and what to do, in words
# ---------------------------------------------------------------------------


_STEPS = [
    "prepare",
    "move gate",
    "pre-flight check of DEFAULT",
    "parent lock",
    "check on DEFAULT",
    "DEFAULT lock",
    "copy",
    "delete from DEFAULT",
    "validate the check on DEFAULT",
    "key, unique indexes and foreign keys",
    "attach",
    "mirror settings",
    "commit",
    "fence",
    "validate",
    "indexes, keys and attach",
    "detach",
]


@pytest.mark.parametrize("step", _STEPS)
def test_no_reason_quotes_an_internal_step_name(step):
    error = RuntimeError("boom")
    error.bm25_move_step = step
    reason = indexing._bm25_failure_reason(error)
    assert f"'{step}'" not in reason
    assert "step" not in reason
    assert "worker log" in reason


def test_every_step_the_service_names_has_a_description():
    import inspect

    source = inspect.getsource(pgb)
    import re

    named = set(re.findall(r'step = "([^"]+)"', source)) | set(
        re.findall(r'step="([^"]+)"', source)
    )
    assert named <= set(indexing.MOVE_STEP_DESCRIPTIONS), named - set(
        indexing.MOVE_STEP_DESCRIPTIONS
    )


def test_an_actionable_refusal_keeps_its_instructions():
    error = pgb.PartitionMoveRefused(
        "ai.chunks_kb_x is an unattached clone that no longer matches ai.chunks_default, and "
        "it holds rows; not dropping it. Move its rows back or drop it, then retry"
    )
    error.bm25_move_step = "prepare"
    reason = indexing._bm25_failure_reason(error)
    assert "Move its rows back or drop it, then retry" in reason


def test_a_database_error_is_named_not_just_numbered():
    error = IntegrityError("INSERT", {}, psycopg.errors.UniqueViolation("duplicate key"))
    error.orig = MagicMock(sqlstate="23505")
    error.bm25_move_step = "copy"
    reason = indexing._bm25_failure_reason(error)
    assert "23505" in reason and "unique_violation" in reason
    assert "copying" in reason


@pytest.mark.parametrize(
    "code",
    [
        "extension_absent",
        "kb_not_found",
        "retrieval_method",
        "strategy",
        "table_not_partitioned",
        "default_partition_absent",
    ],
)
def test_a_skip_is_recorded_in_words(task, code):
    ensure, _retry, recorded = task
    with patch(f"{SERVICE}.ensure_bm25_index", return_value={"status": "skipped", "reason": code}):
        ensure.run(KB)
    status, reason, _ = recorded[-1]
    assert status == "failed"
    assert reason.startswith("not built: ")
    assert reason != f"not built: {code}"
    assert reason == indexing.skip_reason_text(code)
