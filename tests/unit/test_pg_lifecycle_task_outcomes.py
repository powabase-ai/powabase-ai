"""The pg_search index tasks persist what happened, and say so in order.

A move can retry for many minutes before it succeeds or gives up. Each run
records its progress in ``ai.bm25_index_builds`` (``queued`` when it starts,
``moving``/``building`` as the service reaches those steps, ``retrying`` with
the reason when it schedules another attempt, ``failed`` when it gives up,
``ready`` at the end) so the knowledge base can report it; the log says
"retrying" only once a retry is decided, and the give-up is its last line, at
ERROR, naming the sessions that were in the way.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import Retry

from agentic_project_service.services import pg_bm25_index as pgb
from agentic_project_service.tasks import indexing

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
SERVICE = "agentic_project_service.services.pg_bm25_index"


def _lock_timeout(holders=None, step="attach"):
    from sqlalchemy.exc import OperationalError

    class _Orig(Exception):
        sqlstate = "55P03"

    exc = OperationalError("LOCK TABLE", {}, _Orig("canceling statement due to lock timeout"))
    exc.bm25_lock_holders = holders or []
    exc.bm25_move_step = step
    return exc


@pytest.fixture
def task(monkeypatch):
    recorded: list[tuple] = []

    def record(bind, kb_id, item_table, status, reason=None, attempts=None):
        recorded.append((status, reason, attempts, item_table))

    monkeypatch.setattr(indexing, "record_bm25_build_outcome", record)
    monkeypatch.setattr(indexing, "db", MagicMock())
    monkeypatch.setattr(pgb, "keyword_item_table", lambda bind, kb_id: "chunks")
    spy = MagicMock(side_effect=Retry("retry"))
    monkeypatch.setattr(indexing.ensure_pg_bm25_index, "retry", spy)
    return indexing.ensure_pg_bm25_index, spy, recorded


def _statuses(recorded):
    return [status for status, *_ in recorded]


def test_a_run_records_queued_first_then_the_services_progress_then_ready(task):
    ensure, _retry, recorded = task

    def service(kb_id, engine=None, on_progress=None, allow_row_move=True):
        on_progress("moving")
        on_progress("building")
        return {"status": "ready", "item_table": "chunks"}

    with patch(f"{SERVICE}.ensure_bm25_index", side_effect=service):
        ensure.run(KB)

    assert _statuses(recorded) == ["queued", "moving", "building", "ready"]
    assert {item_table for *_, item_table in recorded} == {"chunks"}


def test_a_transient_failure_records_retrying_with_a_reason_and_logs_after_deciding(task, caplog):
    ensure, retry, recorded = task
    error = _lock_timeout(
        holders=[{"pid": 4242, "mode": "AccessShareLock", "granted": True, "xact_seconds": 9.5}]
    )
    order: list[str] = []
    retry.side_effect = lambda **kw: order.append(("retry", kw.get("throw"))) or Retry("retry")

    with caplog.at_level(logging.INFO), patch(f"{SERVICE}.ensure_bm25_index", side_effect=error):
        with pytest.raises(Retry):
            ensure.run(KB)

    assert _statuses(recorded) == ["queued", "retrying"]
    status, reason, attempts, _ = recorded[-1]
    assert attempts == 1
    assert "4242" in reason and "attach" in reason
    retrying = [r for r in caplog.records if "Retrying" in r.getMessage()]
    assert len(retrying) == 1
    assert order == [("retry", False)]


def test_exhausting_the_retries_records_failed_and_ends_with_an_error_naming_the_holders(
    task, caplog, monkeypatch
):
    ensure, retry, recorded = task
    monkeypatch.setattr(ensure, "max_retries", 0)
    error = _lock_timeout(
        holders=[{"pid": 4242, "mode": "AccessShareLock", "granted": True, "xact_seconds": 9.5}]
    )

    with caplog.at_level(logging.INFO), patch(f"{SERVICE}.ensure_bm25_index", side_effect=error):
        with pytest.raises(Exception, match="lock timeout"):
            ensure.run(KB)

    retry.assert_not_called()
    assert _statuses(recorded) == ["queued", "failed"]
    assert not [r for r in caplog.records if "Retrying" in r.getMessage()]
    last = caplog.records[-1]
    assert last.levelno == logging.ERROR
    assert "4242" in last.getMessage()


def test_a_real_failure_records_failed_without_retrying(task, caplog):
    ensure, retry, recorded = task
    from sqlalchemy.exc import ProgrammingError

    class _Orig(Exception):
        sqlstate = "42601"

    error = ProgrammingError("INSERT", {}, _Orig("syntax error at or near"))
    error.bm25_move_step = "copy"

    with caplog.at_level(logging.INFO), patch(f"{SERVICE}.ensure_bm25_index", side_effect=error):
        with pytest.raises(ProgrammingError):
            ensure.run(KB)

    retry.assert_not_called()
    assert _statuses(recorded) == ["queued", "failed"]
    reason = recorded[-1][1]
    assert "42601" in reason and "copy" in reason
    assert caplog.records[-1].levelno == logging.ERROR


def test_another_build_in_progress_is_retried_and_recorded(task):
    ensure, retry, recorded = task
    with patch(
        f"{SERVICE}.ensure_bm25_index",
        return_value={"status": "skipped", "reason": "partition_build_in_progress"},
    ):
        with pytest.raises(Retry):
            ensure.run(KB)
    assert _statuses(recorded) == ["queued", "retrying"]
    assert "another" in recorded[-1][1]


def test_a_kb_without_a_keyword_index_records_nothing(task, monkeypatch):
    ensure, _retry, recorded = task
    monkeypatch.setattr(pgb, "keyword_item_table", lambda bind, kb_id: None)
    with patch(
        f"{SERVICE}.ensure_bm25_index",
        return_value={"status": "skipped", "reason": "retrieval_method"},
    ):
        ensure.run(KB)
    assert recorded == []


def test_reasons_carry_no_sql_or_query_text(task):
    ensure, _retry, recorded = task
    error = _lock_timeout(
        holders=[
            {
                "pid": 7,
                "mode": "AccessShareLock",
                "granted": True,
                "xact_seconds": 1.0,
                "query": "SELECT secret FROM ai.chunks",
            }
        ]
    )
    with patch(f"{SERVICE}.ensure_bm25_index", side_effect=error):
        with pytest.raises(Retry):
            ensure.run(KB)
    reason = recorded[-1][1]
    assert "SELECT" not in reason and "secret" not in reason and "LOCK TABLE" not in reason


def test_retry_countdowns_are_jittered_within_bounds(monkeypatch):
    countdowns = {indexing._pg_bm25_retry_countdown(2) for _ in range(50)}
    assert len(countdowns) > 1
    assert all(120 <= c <= 150 for c in countdowns), countdowns
    assert all(c <= 750 for c in (indexing._pg_bm25_retry_countdown(10) for _ in range(20)))


def test_the_drop_task_logs_its_give_up_at_error(monkeypatch, caplog):
    drop = indexing.drop_pg_bm25_index
    monkeypatch.setattr(drop, "max_retries", 0)
    retry = MagicMock(side_effect=Retry("retry"))
    monkeypatch.setattr(drop, "retry", retry)
    with (
        caplog.at_level(logging.INFO),
        patch(f"{SERVICE}.drop_bm25_index", side_effect=pgb.PartitionBuildInProgress("busy")),
    ):
        with pytest.raises(pgb.PartitionBuildInProgress):
            drop.run(KB)
    retry.assert_not_called()
    assert caplog.records[-1].levelno == logging.ERROR
    assert KB in caplog.records[-1].getMessage()


def test_a_ready_index_retires_the_kbs_frozen_file_index(task, monkeypatch):
    ensure, _retry, _recorded = task
    store = MagicMock()
    monkeypatch.setattr(indexing, "SparseIndexStore", MagicMock(return_value=store))
    with patch(
        f"{SERVICE}.ensure_bm25_index", return_value={"status": "ready", "item_table": "chunks"}
    ):
        ensure.run(KB)
    indexing.SparseIndexStore.assert_called_once_with(knowledge_base_id=KB)
    store.delete_index.assert_called_once_with(item_table="chunks")


@pytest.mark.parametrize("status", ["building", "skipped"])
def test_the_file_index_stays_until_the_index_is_ready(task, monkeypatch, status):
    ensure, _retry, _recorded = task
    monkeypatch.setattr(indexing, "SparseIndexStore", MagicMock())
    with patch(
        f"{SERVICE}.ensure_bm25_index", return_value={"status": status, "item_table": "chunks"}
    ):
        ensure.run(KB)
    indexing.SparseIndexStore.assert_not_called()


def test_a_failure_to_delete_the_file_index_does_not_fail_the_build(task, monkeypatch):
    ensure, _retry, recorded = task
    store = MagicMock()
    store.delete_index.side_effect = OSError("read-only file system")
    monkeypatch.setattr(indexing, "SparseIndexStore", MagicMock(return_value=store))
    with patch(
        f"{SERVICE}.ensure_bm25_index", return_value={"status": "ready", "item_table": "chunks"}
    ):
        assert ensure.run(KB)["status"] == "ready"
    assert recorded[-1][0] == "ready"


def test_an_automatic_run_does_not_allow_a_row_move_and_the_operator_run_does(task):
    ensure, _retry, _recorded = task
    seen: list[bool] = []

    def service(kb_id, engine=None, on_progress=None, allow_row_move=True):
        seen.append(allow_row_move)
        return {"status": "ready", "item_table": "chunks"}

    with patch(f"{SERVICE}.ensure_bm25_index", side_effect=service):
        ensure.run(KB)
        ensure.run(KB, allow_row_move=True)

    assert seen == [False, True]


def test_rows_found_by_an_automatic_run_record_failed_pointing_at_build_bm25(task, caplog):
    ensure, retry, recorded = task

    with (
        caplog.at_level(logging.WARNING),
        patch(
            f"{SERVICE}.ensure_bm25_index",
            return_value={
                "status": "skipped",
                "reason": "row_move_not_allowed",
                "item_table": "chunks",
            },
        ),
    ):
        ensure.run(KB)

    status, reason, _, _ = recorded[-1]
    assert status == "failed"
    assert "POST /build-bm25" in reason and "DEFAULT" in reason
    retry.assert_not_called()
    assert "POST /build-bm25" in caplog.text


def test_a_graph_nodes_move_that_gives_up_on_its_gate_says_graph_indexing_holds_it():
    error = _lock_timeout(step="move gate")
    error.bm25_item_table = "graph_index_nodes"
    assert "graph_index source is indexing" in indexing._bm25_failure_reason(error)
    error.bm25_item_table = "chunks"
    assert "graph_index" not in indexing._bm25_failure_reason(error)
