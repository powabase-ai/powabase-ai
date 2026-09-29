"""One build or drop at a time on ``ai.embeddings``, gated before Postgres sees it.

Every per-knowledge-base index lives on the one shared table, and every
``CREATE INDEX CONCURRENTLY`` / ``DROP INDEX CONCURRENTLY`` on it takes the
table's self-conflicting ``ShareUpdateExclusiveLock``. A second knowledge base's
DDL used to queue *inside Postgres* on that lock, holding a snapshot, and the
build it queued behind waited on that snapshot at its very end: a cycle, and the
deadlock detector killed the build after all of its work (issue #95). These
specs pin the gate that replaces the queue:

* a table-level advisory lock is taken before every DDL path and released on
  every exit, the exceptional ones included;
* a busy table yields ``table_busy`` with no DDL issued, no session setting
  lifted and no build count written;
* the task reschedules on it with a long fixed countdown that does not spend the
  counted retry budget while a build on the table is demonstrably alive, is
  bounded all the same, and falls back to the counted retry when the lock is
  held and nothing is running;
* the start-up sweep and the post-indexing dispatch are unchanged -- they
  dispatch, and the gate decides.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from celery.exceptions import Retry

from agentic_project_service.services import pg_vector_index as pvi
from agentic_project_service.tasks import indexing as idx
from tests.unit.test_per_kb_vector_index import (
    _FAILURE_COMMENT_DDL,
    KB,
    _ensure,
    _ensure_conn,
    _FakeConn,
    _FakeEngine,
    _index_row,
    _older_definition_row,
    _transient_exc,
)

KB_OTHER = "3f2504e0-4f89-11d3-9a0c-0305e82c3302"

_LOCK_TRY = "pg_try_advisory_lock"
_LOCK_RELEASE = "pg_advisory_unlock"
_HOLDERS_QUERY = "pg_locks"

# One row of ``table_ddl_holders``' evidence: what a live build on the table looks
# like from another backend.
_LIVE_BUILD = (
    4242,
    "client backend",
    "ShareUpdateExclusiveLock",
    "active",
    3_600,
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS hnsw_kb_x_1536 ON ai.embeddings",
)


class _GatedConn(_FakeConn):
    """A ``_FakeConn`` whose advisory locks are told apart by subject.

    ``_ensure_conn`` answers every ``pg_try_advisory_lock`` with True, which was
    enough while there was one lock per index. The gate is a second lock on a
    different subject, and these specs are about the two disagreeing -- this
    index free, the table taken -- so the answer is read from the bind.
    """

    def __init__(self, *args, table_free=True, holders=(), index_free=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_free = table_free
        self.index_free = index_free
        self.holders = list(holders)

    def execute(self, clause, params=None):
        sql = clause.text if hasattr(clause, "text") else str(clause)
        relation = (params or {}).get("relation")
        if _LOCK_TRY in sql:
            self.statements.append(" ".join(sql.split()))
            self.params.append(params)
            if self._fail_on is not None and self._fail_on in sql:
                raise self._exc
            free = self.table_free if relation == pvi.table_lock_relation() else self.index_free
            return _Rows([(free,)])
        if _HOLDERS_QUERY in sql:
            self.statements.append(" ".join(sql.split()))
            self.params.append(params)
            return _Rows(list(self.holders))
        return super().execute(clause, params)

    # -- what the specs read ------------------------------------------------

    def _positions(self, fragment: str, relation: str | None = None) -> list[int]:
        return [
            i
            for i, (st, p) in enumerate(zip(self.statements, self.params))
            if fragment in st and (relation is None or (p or {}).get("relation") == relation)
        ]

    def table_tries(self) -> list[int]:
        return self._positions(_LOCK_TRY, pvi.table_lock_relation())

    def table_releases(self) -> list[int]:
        return self._positions(_LOCK_RELEASE, pvi.table_lock_relation())

    def index_releases(self, dims: int = 1536) -> list[int]:
        return self._positions(_LOCK_RELEASE, pvi.index_lock_relation(KB, dims))

    def ddl(self) -> list[int]:
        return self._positions("INDEX CONCURRENTLY")


class _Rows:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._rows[0][0] if self._rows else None


# The four ways a reconcile reaches DDL on the shared table. Each is the
# ``_ensure_conn`` shape that drives the real loop down that path.
_DDL_PATHS = {
    # No index, over the build threshold: CREATE INDEX CONCURRENTLY.
    "build": dict(existing=(), rows_by_dims={1536: 20_000}),
    # An INVALID index: the repair's DROP INDEX CONCURRENTLY, then the build.
    "repair": dict(existing=(_index_row(KB, 1536, valid=False),), rows_by_dims={1536: 20_000}),
    # A valid index at or below the drop threshold: DROP INDEX CONCURRENTLY.
    "threshold_drop": dict(existing=(_index_row(KB, 1536),), rows_by_dims={1536: 1_000}),
    # A valid index built from an older definition: count, drop, rebuild.
    "definition_rebuild": dict(existing=(_older_definition_row(),), rows_by_dims={1536: 20_000}),
}


def _gated(path: str, **kwargs) -> _GatedConn:
    return _ensure_conn(cls=_GatedConn, **_DDL_PATHS[path], **kwargs)


# ---------------------------------------------------------------------------
# The subject
# ---------------------------------------------------------------------------


def test_the_table_lock_is_one_subject_per_table_and_no_index_shares_it():
    subject = pvi.table_lock_relation()
    assert subject == pvi.table_lock_relation(), "one subject, whoever asks"
    assert "embeddings" in subject
    for dims in (384, 1536):
        assert subject != pvi.index_lock_relation(KB, dims)
    # Not the bare table name either: that is the shape the BM25 path's per-item-
    # table build lock uses (``ai.chunks``), and a future partitioned embeddings
    # table must not silently share a key with this gate.
    assert subject != f"{pvi.AI_SCHEMA}.embeddings"


# ---------------------------------------------------------------------------
# Taken before every DDL path, released on every exit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(_DDL_PATHS))
def test_every_ddl_path_takes_the_table_lock_before_its_first_ddl(monkeypatch, path):
    conn = _gated(path)
    outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "ready", outcome
    assert conn.ddl(), f"the {path} path issued no DDL: {conn.statements}"
    tries, releases = conn.table_tries(), conn.table_releases()
    assert len(tries) == 1, f"one table lock per dimension, not per statement: {conn.statements}"
    assert tries[0] < conn.ddl()[0], conn.statements
    assert releases and releases[-1] > conn.ddl()[-1], "released only after the last DDL"
    # Taken after the per-index lock, and given back before it.
    index_try = conn._positions(_LOCK_TRY, pvi.index_lock_relation(KB, 1536))
    assert index_try and index_try[0] < tries[0], conn.statements
    assert releases[-1] < conn.index_releases()[-1], conn.statements


def test_the_definition_rebuild_takes_the_table_before_it_counts_the_rebuild(monkeypatch):
    """The rebuild count is written before the drop, so the gate has to be earlier still.

    Otherwise a busy table would spend one of ``MAX_CONSECUTIVE_DEFINITION_REBUILDS``
    on a rebuild that never started -- three busy boots and the index is frozen on its
    old definition.
    """
    conn = _gated("definition_rebuild")
    _ensure(monkeypatch, conn)
    first_comment = conn._positions(_FAILURE_COMMENT_DDL)[0]
    assert conn.table_tries()[0] < first_comment, conn.statements


@pytest.mark.parametrize("path", sorted(_DDL_PATHS))
def test_the_table_lock_is_released_when_the_ddl_raises(monkeypatch, path):
    fragment = "DROP INDEX CONCURRENTLY" if path != "build" else "CREATE INDEX CONCURRENTLY"
    conn = _gated(path, fail_on=fragment, exc=_transient_exc())
    with pytest.raises(Exception):
        _ensure(monkeypatch, conn)
    assert conn.table_tries(), conn.statements
    assert conn.table_releases(), "a failed DDL must not keep the whole table gated"
    assert conn.table_releases()[-1] > conn.ddl()[-1]
    assert conn.index_releases(), conn.statements


def test_a_release_that_fails_discards_the_connection_so_the_gate_cannot_leak(monkeypatch):
    """Session-scoped, so a pooled connection still holding it would gate the table for ever."""
    conn = _gated("build", fail_on=_LOCK_RELEASE)
    _ensure(monkeypatch, conn)
    assert conn.invalidated, conn.statements


def test_a_reconcile_with_nothing_to_do_never_asks_for_the_table(monkeypatch):
    """A 10-hour build elsewhere must not hold up a knowledge base that needs no DDL."""
    conn = _ensure_conn(
        cls=_GatedConn,
        existing=(_index_row(KB, 1536),),
        rows_by_dims={1536: 20_000},
        table_free=False,
    )
    outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "ready", outcome
    assert conn.table_tries() == [], conn.statements


# ---------------------------------------------------------------------------
# A busy table: no DDL, no transaction left waiting, nothing counted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(_DDL_PATHS))
def test_a_busy_table_yields_table_busy_with_no_ddl_and_nothing_counted(monkeypatch, path):
    conn = _gated(path, table_free=False, holders=[_LIVE_BUILD])
    outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "table_busy", outcome
    assert outcome["reason"] == "table_ddl_in_progress", outcome
    assert outcome["build_alive"] is True, outcome
    assert outcome["index"] == pvi.per_kb_index_name(KB, 1536), outcome
    assert outcome["table_busy"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    assert outcome["table_holders"][0]["pid"] == 4242, outcome
    assert "CREATE INDEX CONCURRENTLY" in outcome["table_holders"][0]["query"]
    assert conn.ddl() == [], f"DDL issued while the table was busy: {conn.statements}"
    assert conn.issued("SET statement_timeout") == [], "nothing lifted, nothing to wait with"
    assert conn.issued("SET lock_timeout") == []
    assert conn.issued(_FAILURE_COMMENT_DDL) == [], (
        "a table_busy outcome is neither a failed nor an interrupted attempt"
    )
    assert conn.table_releases() == [], "a lock that was never taken is not given back"
    assert conn.index_releases(), "the per-index lock is still given back"
    assert pvi.outcome_waits_for_the_table(outcome) is True
    assert pvi.outcome_needs_another_attempt(outcome) is False, (
        "the counted reschedule is for an INVALID index with a live build of its own"
    )


def test_a_held_table_lock_with_nothing_running_is_reported_as_a_dead_holder(monkeypatch):
    """The evidence is what the task spends its wait budget on, so its absence is reported."""
    conn = _gated("build", table_free=False, holders=[])
    outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "table_busy", outcome
    assert outcome["reason"] == "table_lock_held", outcome
    assert outcome["build_alive"] is False, outcome
    assert outcome["table_holders"] == [], outcome
    assert conn.ddl() == []


def test_an_ungated_ddl_on_the_table_is_waited_for_like_a_gated_one(monkeypatch):
    """The advisory lock only stops callers of this module.

    A manual ``DROP INDEX CONCURRENTLY``, or a worker still running a version from
    before the gate during a rolling deploy, holds the table's own lock and not the
    advisory one. Queueing behind it is the exact cycle the gate exists to avoid,
    so the lock is given straight back and the outcome is the same ``table_busy``.
    """
    conn = _gated("build", table_free=True, holders=[_LIVE_BUILD])
    outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "table_busy", outcome
    assert outcome["build_alive"] is True, outcome
    assert conn.ddl() == [], conn.statements
    assert conn.table_releases(), "an acquired gate that is not used must be given back"


def test_the_holders_check_is_run_on_the_table_and_not_on_one_index(monkeypatch):
    """Per table: the issue's "no build is running" was true of one index and false of the table."""
    conn = _gated("build", table_free=False, holders=[_LIVE_BUILD])
    _ensure(monkeypatch, conn)
    holder_params = [p for st, p in zip(conn.statements, conn.params) if _HOLDERS_QUERY in st]
    assert holder_params, conn.statements
    assert holder_params[0]["table"] == f'"{pvi.AI_SCHEMA}".embeddings'
    sql = next(st for st in conn.statements if _HOLDERS_QUERY in st)
    assert "pg_backend_pid()" in sql, "its own backend is never evidence of someone else"
    assert "granted" in sql, "a queued waiter is not a build that is alive"


def test_table_busy_outranks_every_other_status(monkeypatch):
    """The one outcome that has to come back, and on a clock of its own."""
    conn = _ensure_conn(
        cls=_GatedConn,
        existing=(),
        dims_present=(768, 1536),
        rows_by_dims={768: 20_000, 1536: 20_000},
        table_free=False,
        holders=[_LIVE_BUILD],
    )
    outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "table_busy", outcome
    assert outcome["table_busy"] == [
        pvi.per_kb_index_name(KB, 768),
        pvi.per_kb_index_name(KB, 1536),
    ], outcome


def test_a_busy_table_is_reported_to_the_progress_hook(monkeypatch):
    events: list[tuple[str, dict]] = []
    conn = _gated("build", table_free=False, holders=[_LIVE_BUILD])
    _ensure(monkeypatch, conn, on_progress=lambda status, **f: events.append((status, f)))
    assert [status for status, _ in events] == ["waiting_for_table"], events
    assert events[0][1]["dims"] == 1536


# ---------------------------------------------------------------------------
# The deleted knowledge base's drop is DDL on the same table
# ---------------------------------------------------------------------------


def _drop_conn(**kwargs) -> _GatedConn:
    return _GatedConn(
        answers=[("i.indisvalid", [_index_row(KB, 1536)])],
        **kwargs,
    )


def test_the_deleted_kbs_drop_takes_the_table_lock_around_its_ddl():
    conn = _drop_conn()
    outcome = pvi.drop_per_kb_vector_indexes(KB, engine=_FakeEngine(conn))
    assert outcome["indexes"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    assert conn.table_tries()[0] < conn.ddl()[0] < conn.table_releases()[-1], conn.statements


def test_the_deleted_kbs_drop_raises_table_busy_with_no_ddl():
    conn = _drop_conn(table_free=False, holders=[_LIVE_BUILD])
    with pytest.raises(pvi.PerKbVectorIndexTableBusy) as raised:
        pvi.drop_per_kb_vector_indexes(KB, engine=_FakeEngine(conn))
    assert raised.value.build_alive is True
    assert raised.value.holders[0]["pid"] == 4242
    # A subclass, so every caller that already retries a held lock retries this too.
    assert isinstance(raised.value, pvi.PerKbVectorIndexBuildInProgress)
    assert conn.ddl() == [], conn.statements
    assert conn.index_releases(), conn.statements


# ---------------------------------------------------------------------------
# The task: wait on its own clock, without spending the counted budget
# ---------------------------------------------------------------------------


def _busy(alive: bool = True) -> dict:
    return {
        "status": "table_busy",
        "reason": "table_ddl_in_progress" if alive else "table_lock_held",
        "index": "hnsw_kb_abc_1536",
        "table_busy": ["hnsw_kb_abc_1536"],
        "table_holders": [pvi._holder_evidence(_LIVE_BUILD)] if alive else [],
        "build_alive": alive,
        "built": [],
        "dropped": [],
    }


@pytest.fixture
def ensure_task(monkeypatch):
    task = idx.ensure_per_kb_vector_index
    spy = MagicMock(side_effect=lambda *a, **k: Retry("retry"))
    monkeypatch.setattr(task, "retry", spy)

    def run(outcome_or_exc, *, retries: int = 0, table_waits: int = 0):
        if isinstance(outcome_or_exc, BaseException):
            service = MagicMock(side_effect=outcome_or_exc)
        else:
            service = MagicMock(return_value=outcome_or_exc)
        monkeypatch.setattr(pvi, "ensure_per_kb_vector_index", service)
        task.push_request(retries=retries, args=[KB], kwargs={"table_waits": table_waits})
        try:
            return task.run(KB, table_waits=table_waits)
        finally:
            task.pop_request()

    run.spy = spy
    run.task = task
    return run


def test_a_live_build_on_the_table_reschedules_on_the_long_fixed_countdown(ensure_task):
    with pytest.raises(Retry):
        ensure_task(_busy(alive=True))
    ensure_task.spy.assert_called_once()
    call = ensure_task.spy.call_args.kwargs
    assert call["countdown"] == idx.PER_KB_TABLE_WAIT_COUNTDOWN_S
    assert call["kwargs"]["table_waits"] == 1
    assert call["throw"] is False


def test_waiting_for_the_table_does_not_spend_the_counted_budget(ensure_task):
    """Far past ``max_retries`` in Celery's own counter, and still waiting.

    A queued knowledge base may legitimately wait eight to ten hours for one large
    build, which is 50-60 of these waits against a counted budget of seven.
    """
    waits = 60
    with pytest.raises(Retry):
        ensure_task(_busy(alive=True), retries=waits, table_waits=waits)
    call = ensure_task.spy.call_args.kwargs
    assert call["kwargs"]["table_waits"] == waits + 1
    # Celery refuses a retry once ``request.retries + 1 > max_retries``, so the
    # override has to move the ceiling by exactly the waits it does not count.
    assert waits + 1 <= call["max_retries"], call


def test_a_transient_failure_after_many_waits_still_gets_its_counted_retries(ensure_task):
    waits = 60
    with pytest.raises(Retry):
        ensure_task(_transient_exc(), retries=waits, table_waits=waits)
    call = ensure_task.spy.call_args.kwargs
    # The first counted attempt's backoff (30 s, jittered by up to 25 %), not the
    # capped one Celery's own counter would ask for.
    assert 30 <= call["countdown"] <= 38, call
    assert waits + 1 <= call["max_retries"], call
    assert "kwargs" not in call or call["kwargs"]["table_waits"] == waits


def test_the_counted_budget_is_still_spent_by_counted_attempts(ensure_task):
    """Waits are subtracted, not forgiven: six counted retries after any number of waits."""
    waits = 40
    counted = idx.PG_BM25_TASK_MAX_RETRIES
    with pytest.raises(Exception) as raised:
        ensure_task(_transient_exc(), retries=waits + counted, table_waits=waits)
    assert not isinstance(raised.value, Retry)
    ensure_task.spy.assert_not_called()


def test_waiting_for_the_table_is_bounded_and_says_what_it_waited_for(ensure_task, caplog):
    with caplog.at_level(logging.ERROR):
        outcome = ensure_task(
            _busy(alive=True),
            retries=idx.PER_KB_TABLE_MAX_WAITS,
            table_waits=idx.PER_KB_TABLE_MAX_WAITS,
        )
    ensure_task.spy.assert_not_called()
    assert outcome["status"] == "table_busy"
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert "hnsw_kb_abc_1536" in message, message
    assert "4242" in message, message


def test_a_dead_holder_falls_back_to_the_counted_retry(ensure_task):
    """Held, and nothing on the table running: the jittered, counted backoff."""
    with pytest.raises(Retry):
        ensure_task(_busy(alive=False), retries=2, table_waits=0)
    call = ensure_task.spy.call_args.kwargs
    assert 120 <= call["countdown"] <= 150, call
    assert call.get("kwargs", {"table_waits": 0})["table_waits"] == 0, (
        "a counted retry is not a wait"
    )


def test_a_dead_holder_gives_up_at_the_counted_bound(ensure_task, caplog):
    with caplog.at_level(logging.ERROR):
        outcome = ensure_task(
            _busy(alive=False), retries=idx.PG_BM25_TASK_MAX_RETRIES + 3, table_waits=3
        )
    ensure_task.spy.assert_not_called()
    assert outcome["status"] == "table_busy"
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert "hnsw_kb_abc_1536" in message, message


def test_the_wait_budget_outlasts_the_longest_build_measured():
    """9.6 h for the 2.85M-row build alone, and a queue can hold several of them."""
    assert idx.PER_KB_TABLE_WAIT_COUNTDOWN_S >= 5 * 60, "a poll, not a spin"
    total_s = idx.PER_KB_TABLE_WAIT_COUNTDOWN_S * idx.PER_KB_TABLE_MAX_WAITS
    assert total_s >= 24 * 3600, total_s
    assert total_s <= 7 * 24 * 3600, "and still a bound"


def test_the_real_celery_retry_accepts_a_wait_past_max_retries(monkeypatch):
    """Not the spy: Celery's own ceiling check, against a request far past the budget.

    ``is_eager`` makes ``retry`` hand the signature back instead of publishing it,
    which leaves only the arithmetic this depends on -- ``request.retries + 1`` against
    the ``max_retries`` the task passes.
    """
    task = idx.ensure_per_kb_vector_index
    monkeypatch.setattr(pvi, "ensure_per_kb_vector_index", MagicMock(return_value=_busy()))
    waits = 60
    task.push_request(
        retries=waits,
        args=[KB],
        kwargs={"table_waits": waits},
        called_directly=False,
        is_eager=True,
        id="task-1",
    )
    try:
        with pytest.raises(Retry) as raised:
            task.run(KB, table_waits=waits)
    finally:
        task.pop_request()
    assert raised.value.sig.kwargs["table_waits"] == waits + 1
    assert raised.value.sig.args == [KB] or tuple(raised.value.sig.args) == (KB,)
    assert raised.value.when == idx.PER_KB_TABLE_WAIT_COUNTDOWN_S


def test_the_drop_task_waits_for_a_live_build_without_counting_it(monkeypatch):
    task = idx.drop_per_kb_vector_index
    spy = MagicMock(side_effect=lambda *a, **k: Retry("retry"))
    monkeypatch.setattr(task, "retry", spy)
    busy = pvi.PerKbVectorIndexTableBusy(
        "busy", holders=[pvi._holder_evidence(_LIVE_BUILD)], lock_held=True
    )
    monkeypatch.setattr(pvi, "drop_per_kb_vector_indexes", MagicMock(side_effect=busy))
    waits = 30
    task.push_request(retries=waits, args=[KB], kwargs={"table_waits": waits})
    try:
        with pytest.raises(Retry):
            task.run(KB, table_waits=waits)
    finally:
        task.pop_request()
    call = spy.call_args.kwargs
    assert call["countdown"] == idx.PER_KB_TABLE_WAIT_COUNTDOWN_S
    assert call["kwargs"]["table_waits"] == waits + 1
    assert waits + 1 <= call["max_retries"]


def test_the_drop_task_counts_a_dead_holder(monkeypatch):
    task = idx.drop_per_kb_vector_index
    spy = MagicMock(side_effect=lambda *a, **k: Retry("retry"))
    monkeypatch.setattr(task, "retry", spy)
    busy = pvi.PerKbVectorIndexTableBusy("busy", holders=[], lock_held=True)
    monkeypatch.setattr(pvi, "drop_per_kb_vector_indexes", MagicMock(side_effect=busy))
    task.push_request(retries=1, args=[KB], kwargs={})
    try:
        with pytest.raises(Retry):
            task.run(KB)
    finally:
        task.pop_request()
    call = spy.call_args.kwargs
    assert 60 <= call["countdown"] <= 75, call


# ---------------------------------------------------------------------------
# Dispatch is unchanged: it dispatches, and the gate decides
# ---------------------------------------------------------------------------


def test_the_start_up_sweep_dispatches_every_pending_kb_plainly(monkeypatch):
    """Ten at once is fine now: nine of them reschedule instead of queueing in Postgres."""
    pending = [f"3f2504e0-4f89-11d3-9a0c-0305e82c33{n:02d}" for n in range(pvi.MAX_SWEEP_DISPATCH)]
    monkeypatch.setattr(pvi, "kbs_needing_a_per_kb_index", lambda engine: list(pending))
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        idx.ensure_per_kb_vector_index, "delay", lambda *a, **k: calls.append((a, k))
    )
    assert idx.dispatch_per_kb_vector_indexes_at_start(object()) == pending
    assert calls == [((kb,), {}) for kb in pending], (
        "no countdown, no staggering, no table_waits: the gate is the only serialiser"
    )


def test_the_post_indexing_dispatch_is_plain_too(monkeypatch):
    monkeypatch.setattr(idx, "_per_kb_vector_index_action", lambda kb_id: "build")
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        idx.ensure_per_kb_vector_index, "delay", lambda *a, **k: calls.append((a, k))
    )
    assert idx.dispatch_per_kb_vector_index(KB_OTHER) is True
    assert calls == [((KB_OTHER,), {})]
