"""A write that races a partition move must leave the source claimable.

A write of a knowledge base's rows that Postgres routed to the DEFAULT
partition just as that knowledge base's partition was attached fails with
SQLSTATE 23514 -- "violates partition constraint", or the move's temporary
``bm25_move_<kb>`` check. Nothing about the source is wrong, so it must be
re-queued within the attempts bound, not marked ``failed`` with nothing to
re-queue it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import psycopg
import pytest
from sqlalchemy.exc import IntegrityError

from agentic_project_service.services import pg_bm25_index as pgb
from agentic_project_service.tasks import indexing

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
SRC = "11111111-1111-4111-8111-111111111111"
IS_ID = "22222222-2222-4222-8222-222222222222"


def _integrity(message: str) -> IntegrityError:
    return IntegrityError("INSERT INTO ai.chunks ...", {}, psycopg.errors.CheckViolation(message))


RACES = [
    'new row for relation "chunks_default" violates partition constraint',
    'new row for relation "chunks_default" violates check constraint '
    '"bm25_move_3f2504e04f8911d39a0c0305e82c3301"',
]


@pytest.mark.parametrize("message", RACES)
def test_a_partition_move_race_is_recognised(message):
    assert pgb.is_partition_move_race(_integrity(message)) is True


def test_an_ordinary_check_violation_is_not_a_race():
    assert (
        pgb.is_partition_move_race(
            _integrity('new row for relation "chunks" violates check constraint "tokens_positive"')
        )
        is False
    )
    assert pgb.is_partition_move_race(RuntimeError("violates partition constraint")) is False


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(indexing, "db", MagicMock())
    monkeypatch.setattr(indexing, "get_knowledge_base", lambda _id: {"indexing_config": {}})
    monkeypatch.setattr(indexing, "get_source", lambda _id: {"extraction_status": "extracted"})
    monkeypatch.setattr(indexing, "_claim_indexed_source", lambda *_a: 1)
    requeue = MagicMock()
    failed = MagicMock()
    monkeypatch.setattr(indexing, "_handle_storage_error", requeue)
    monkeypatch.setattr(indexing, "_fenced_mark_failed", failed)
    return monkeypatch, requeue, failed


@pytest.mark.parametrize("message", RACES)
def test_index_source_requeues_a_source_whose_write_raced_the_attach(harness, message):
    monkeypatch, requeue, failed = harness

    def _raise(**_kwargs):
        raise _integrity(message)

    monkeypatch.setattr(indexing, "_run_index_body", _raise)

    out = indexing.index_source.run(KB, SRC, indexed_source_id=IS_ID)

    failed.assert_not_called()
    requeue.assert_called_once()
    assert requeue.call_args.kwargs["indexed_source_id"] == IS_ID
    assert "partition" in requeue.call_args.kwargs["cause"]
    assert out["status"] == "retrying_or_failed"


def test_index_source_still_fails_a_real_check_violation(harness):
    monkeypatch, requeue, failed = harness

    def _raise(**_kwargs):
        raise _integrity(
            'new row for relation "chunks" violates check constraint "tokens_positive"'
        )

    monkeypatch.setattr(indexing, "_run_index_body", _raise)

    indexing.index_source.run(KB, SRC, indexed_source_id=IS_ID)

    requeue.assert_not_called()
    failed.assert_called_once()


def _operational(sqlstate: str, message: str):
    from sqlalchemy.exc import OperationalError

    class _Orig(Exception):
        pass

    orig = _Orig(message)
    orig.sqlstate = sqlstate
    return OperationalError("INSERT INTO ai.chunks ...", {}, orig)


LOCK_CONFLICTS = [
    ("40P01", "deadlock detected"),
    ("55P03", "canceling statement due to lock timeout"),
]


@pytest.mark.parametrize(("sqlstate", "message"), LOCK_CONFLICTS)
def test_a_lock_conflict_is_recognised(sqlstate, message):
    assert pgb.is_lock_conflict(_operational(sqlstate, message)) is True


def test_other_operational_errors_are_not_lock_conflicts():
    assert pgb.is_lock_conflict(_operational("57014", "canceling statement due to user")) is False
    assert pgb.is_lock_conflict(RuntimeError("deadlock detected")) is False


@pytest.mark.parametrize(("sqlstate", "message"), LOCK_CONFLICTS)
def test_index_source_requeues_a_source_that_lost_a_lock_conflict(harness, sqlstate, message):
    """A deadlock or lock timeout says another transaction was in the way -- a
    partition move holding the item table, typically -- not that the source is
    bad, so it is re-queued within the attempts bound."""
    monkeypatch, requeue, failed = harness

    def _raise(**_kwargs):
        raise _operational(sqlstate, message)

    monkeypatch.setattr(indexing, "_run_index_body", _raise)

    out = indexing.index_source.run(KB, SRC, indexed_source_id=IS_ID)

    failed.assert_not_called()
    requeue.assert_called_once()
    assert requeue.call_args.kwargs["indexed_source_id"] == IS_ID
    assert out["status"] == "retrying_or_failed"


def test_a_lock_conflict_before_the_claim_is_not_requeued(harness):
    """Only the owner of the row may re-queue it."""
    monkeypatch, requeue, failed = harness
    monkeypatch.setattr(
        indexing,
        "_claim_indexed_source",
        MagicMock(side_effect=_operational("40P01", "deadlock detected")),
    )

    indexing.index_source.run(KB, SRC, indexed_source_id=IS_ID)

    requeue.assert_not_called()
    failed.assert_not_called()


# ---------------------------------------------------------------------------
# A move check left behind on DEFAULT, met by an indexing write
# ---------------------------------------------------------------------------

FENCE = pgb.default_move_check_name(KB)


def _check_refusal(constraint=FENCE, table="chunks_default", schema="ai", sqlstate="23514"):
    """A refused write as psycopg reports it: the names come from the diagnostics."""
    from types import SimpleNamespace

    class _Orig(Exception):
        pass

    orig = _Orig(f'new row for relation "{table}" violates check constraint "{constraint}"')
    orig.sqlstate = sqlstate
    orig.diag = SimpleNamespace(constraint_name=constraint, table_name=table, schema_name=schema)
    return IntegrityError("INSERT INTO ai.chunks ...", {}, orig)


def test_a_move_check_refusal_is_traced_to_its_item_table_by_the_diagnostics():
    assert pgb.move_check_refusal_item_table(_check_refusal()) == "chunks"
    assert (
        pgb.move_check_refusal_item_table(_check_refusal(table="graph_index_nodes_default"))
        == "graph_index_nodes"
    )


@pytest.mark.parametrize(
    "exc",
    [
        # Postgres' own partition constraint: nothing on DEFAULT to clear.
        _check_refusal(constraint=None),
        # The message names a move check, but the diagnostics name another one.
        _check_refusal(constraint="tokens_positive"),
        _check_refusal(constraint="bm25_move_not_a_uuid"),
        _check_refusal(table="chunks"),
        _check_refusal(table="doc2json_documents_default"),
        _check_refusal(schema="elsewhere"),
        _check_refusal(sqlstate="23505"),
        RuntimeError(f'violates check constraint "{FENCE}"'),
    ],
)
def test_anything_else_is_not_a_move_check_refusal(exc):
    assert pgb.move_check_refusal_item_table(exc) is None


def test_clearing_after_a_refusal_makes_one_non_blocking_try(monkeypatch):
    engine = object()
    calls = []
    monkeypatch.setattr(
        pgb,
        "clear_leftover_move_checks",
        lambda eng, item_table, wait_seconds: (
            calls.append((eng, item_table, wait_seconds)) or [FENCE]
        ),
    )

    assert pgb.clear_move_check_after_refusal(engine, _check_refusal()) == [FENCE]
    assert calls == [(engine, "chunks", 0.0)]

    calls.clear()
    assert pgb.clear_move_check_after_refusal(engine, _check_refusal(constraint=None)) == []
    assert calls == []


@pytest.mark.parametrize(
    "error",
    [_operational("55P03", "could not obtain lock"), RuntimeError("connection refused")],
)
def test_clearing_after_a_refusal_never_raises(monkeypatch, error):
    def _raise(*_args):
        raise error

    monkeypatch.setattr(pgb, "clear_leftover_move_checks", _raise)

    assert pgb.clear_move_check_after_refusal(object(), _check_refusal()) == []


def test_index_source_clears_the_check_that_refused_its_write_before_requeueing(harness):
    """Otherwise a check whose move could not drop it refuses every attempt, and
    the source fails once the attempts run out."""
    monkeypatch, requeue, failed = harness
    order: list[str] = []
    refusal = _check_refusal()
    indexing.db.session.rollback.side_effect = lambda: order.append("rollback")
    clear = MagicMock(side_effect=lambda *_a: order.append("clear") or [FENCE])
    monkeypatch.setattr(pgb, "clear_move_check_after_refusal", clear)
    requeue.side_effect = lambda **_kw: order.append("requeue")

    def _raise(**_kwargs):
        raise refusal

    monkeypatch.setattr(indexing, "_run_index_body", _raise)

    indexing.index_source.run(KB, SRC, indexed_source_id=IS_ID)

    clear.assert_called_once_with(indexing.db.engine, refusal)
    # Its own failed transaction still holds a lock on DEFAULT until rolled back.
    assert order.index("rollback") < order.index("clear") < order.index("requeue")
    failed.assert_not_called()


@pytest.mark.parametrize(
    "exc",
    [_check_refusal(), _operational("40P01", "deadlock detected")],
)
def test_index_source_requeues_a_lock_conflict_or_move_race_after_a_short_delay(harness, exc):
    monkeypatch, requeue, _failed = harness
    monkeypatch.setattr(pgb, "clear_move_check_after_refusal", MagicMock(return_value=[]))

    def _raise(**_kwargs):
        raise exc

    monkeypatch.setattr(indexing, "_run_index_body", _raise)

    indexing.index_source.run(KB, SRC, indexed_source_id=IS_ID)

    assert requeue.call_args.kwargs["countdown"] == indexing.MOVE_CONFLICT_REQUEUE_COUNTDOWN_SECONDS


def test_the_requeue_delay_outlasts_a_moves_own_cleanup_but_not_an_index_build_retry():
    """Long enough that a failed move has finished trying to drop its check, so
    a retry is not spent on a check about to go; short next to the first
    retry of the index build task, so indexing is not held back for long."""
    countdown = indexing.MOVE_CONFLICT_REQUEUE_COUNTDOWN_SECONDS
    assert countdown > pgb.MOVE_CHECK_CLEANUP_WAIT_SECONDS + pgb.DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS
    assert countdown < indexing._pg_bm25_retry_countdown(0)


@pytest.mark.parametrize(
    ("exc", "sqlstate"),
    [
        (_operational("40P01", "deadlock detected"), "40P01"),
        (_operational("55P03", "canceling statement due to lock timeout"), "55P03"),
        (_check_refusal(), "23514"),
    ],
)
def test_the_requeue_warning_names_the_sqlstate_without_blaming_a_move(
    harness, caplog, exc, sqlstate
):
    """A deadlock or lock timeout can come from any other transaction, not only
    a partition move, so the warning says what happened rather than guessing."""
    monkeypatch, _requeue, _failed = harness
    monkeypatch.setattr(pgb, "clear_move_check_after_refusal", MagicMock(return_value=[]))

    def _raise(**_kwargs):
        raise exc

    monkeypatch.setattr(indexing, "_run_index_body", _raise)

    with caplog.at_level("WARNING"):
        indexing.index_source.run(KB, SRC, indexed_source_id=IS_ID)

    (message,) = [r.getMessage() for r in caplog.records if SRC in r.getMessage()]
    assert f"SQLSTATE {sqlstate}" in message
    assert "raced a partition move" not in message
