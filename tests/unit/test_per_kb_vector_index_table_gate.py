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

import fakeredis
import pytest
from celery.exceptions import Reject, Retry

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
# like from another backend. The columns are the holder query's: pid, backend type,
# which lock (the table's own, or this module's advisory gate), its mode, the
# backend's state, and three ages in seconds -- its current statement, its
# transaction, and its last state change -- then the statement text.
_LIVE_BUILD = (
    4242,
    "client backend",
    "relation",
    "ShareUpdateExclusiveLock",
    "active",
    3_600,
    3_600,
    3_600,
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS hnsw_kb_x_1536 ON ai.embeddings",
)

# A session that took a conflicting lock and went idle holding it: not a build,
# and nothing that ends on its own.
_IDLE_IN_TRANSACTION = (
    5151,
    "client backend",
    "relation",
    "ShareLock",
    "idle in transaction",
    7,
    14_400,
    14_393,
    "CREATE INDEX idx_x ON ai.embeddings (item_id)",
)

# What another role's backend looks like to a role without pg_read_all_stats.
_INVISIBLE = (
    6161,
    None,
    "relation",
    "ShareUpdateExclusiveLock",
    None,
    None,
    None,
    None,
    "<insufficient privilege>",
)


def _advisory_holder(idle_s: int, state: str = "idle", pid: int = 7171):
    """The backend holding this module's table gate, seen between two statements."""
    return (pid, "client backend", "advisory", "ExclusiveLock", state, 0, idle_s, idle_s, "RESET x")


class _GatedConn(_FakeConn):
    """A ``_FakeConn`` whose advisory locks are told apart by subject.

    ``_ensure_conn`` answers every ``pg_try_advisory_lock`` with True, which was
    enough while there was one lock per index. The gate is a second lock on a
    different subject, and these specs are about the two disagreeing -- this
    index free, the table taken -- so the answer is read from the bind.

    It also models the backend behind the handle: ``pg_backend_pid()`` answers
    ``pid``, an ``invalidate()`` (``_discard_connection``) moves to a new backend,
    and an advisory lock taken on the old backend is gone -- its unlock answers
    False, the way Postgres does for a lock this session does not hold.
    """

    def __init__(
        self,
        *args,
        table_free=True,
        holders=(),
        holders_seq=None,
        advisory_holders=(),
        holders_error=None,
        index_free=True,
        busy_index_dims=(),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.table_free = table_free
        self.index_free = index_free
        self.busy_index_relations = {pvi.index_lock_relation(KB, d) for d in busy_index_dims}
        self.holders = list(holders)
        self.holders_seq = [list(h) for h in holders_seq] if holders_seq is not None else None
        self.advisory_holders = list(advisory_holders)
        self.holders_error = holders_error
        self.pid = 100
        self.taken_on: dict[str, int] = {}

    def invalidate(self):
        super().invalidate()
        self.pid += 1

    def _record(self, sql, params):
        self.statements.append(" ".join(sql.split()))
        self.params.append(params)

    def execute(self, clause, params=None):
        sql = clause.text if hasattr(clause, "text") else str(clause)
        relation = (params or {}).get("relation")
        if _LOCK_TRY in sql:
            self._record(sql, params)
            if self._fail_on is not None and self._fail_on in sql:
                raise self._exc
            if relation == pvi.table_lock_relation():
                free = self.table_free
            else:
                free = self.index_free and relation not in self.busy_index_relations
            if free:
                self.taken_on[relation] = self.pid
            return _Rows([(free,)])
        if _LOCK_RELEASE in sql:
            self._record(sql, params)
            if self._fail_on is not None and self._fail_on in sql:
                raise self._exc
            held = self.taken_on.pop(relation, None) == self.pid
            return _Rows([(held,)])
        if _HOLDERS_QUERY in sql:
            self._record(sql, params)
            if self.holders_error is not None:
                raise self.holders_error
            if "locktype = 'advisory'" in sql:
                return _Rows(list(self.advisory_holders))
            if self.holders_seq is not None:
                return _Rows(self.holders_seq.pop(0) if self.holders_seq else [])
            return _Rows(list(self.holders))
        if sql.strip() == "SELECT pg_backend_pid()":
            self._record(sql, params)
            return _Rows([(self.pid,)])
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


@pytest.fixture(autouse=True)
def waiter_store(monkeypatch):
    """The tasks' waiter marker, in a fake Redis of this test's own."""
    store = fakeredis.FakeStrictRedis()
    monkeypatch.setattr(idx, "_waiter_redis", lambda: store)
    return store


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
    """A give-up is a failure the task's own state says, not a SUCCESS with a dict."""
    with caplog.at_level(logging.ERROR):
        with pytest.raises(pvi.PerKbVectorIndexTableWaitExhausted):
            ensure_task(
                _busy(alive=True),
                retries=idx.PER_KB_TABLE_MAX_WAITS,
                table_waits=idx.PER_KB_TABLE_MAX_WAITS,
            )
    ensure_task.spy.assert_not_called()
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
        with pytest.raises(pvi.PerKbVectorIndexTableWaitExhausted):
            ensure_task(_busy(alive=False), retries=idx.PG_BM25_TASK_MAX_RETRIES + 3, table_waits=3)
    ensure_task.spy.assert_not_called()
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert "hnsw_kb_abc_1536" in message, message


def test_the_wait_budget_outlasts_the_longest_build_measured():
    """9.6 h for the 2.85M-row build alone, and a queue can hold several of them."""
    assert idx.PER_KB_TABLE_WAIT_COUNTDOWN_S >= 5 * 60, "a poll, not a spin"
    total_s = idx.PER_KB_TABLE_WAIT_COUNTDOWN_S * idx.PER_KB_TABLE_MAX_WAITS
    assert total_s >= 24 * 3600, total_s
    assert total_s <= 7 * 24 * 3600, "and still a bound"


def _invalid_left_behind() -> dict:
    return {
        "status": "building",
        "reason": "invalid_index_build_in_progress",
        "reschedule": True,
        "index": "hnsw_kb_abc_1536",
        "built": [],
        "dropped": [],
    }


# Every retry site in both tasks, each driven to the point where it reschedules.
# ``(task, service to stub, what the stub does)``.
_RETRY_SITES = {
    "ensure_transient_failure": (
        "ensure_per_kb_vector_index",
        "ensure_per_kb_vector_index",
        lambda: MagicMock(side_effect=_transient_exc()),
    ),
    "ensure_table_wait": (
        "ensure_per_kb_vector_index",
        "ensure_per_kb_vector_index",
        lambda: MagicMock(return_value=_busy(alive=True)),
    ),
    "ensure_table_held_counted": (
        "ensure_per_kb_vector_index",
        "ensure_per_kb_vector_index",
        lambda: MagicMock(return_value=_busy(alive=False)),
    ),
    "ensure_invalid_reschedule": (
        "ensure_per_kb_vector_index",
        "ensure_per_kb_vector_index",
        lambda: MagicMock(return_value=_invalid_left_behind()),
    ),
    "drop_counted": (
        "drop_per_kb_vector_index",
        "drop_per_kb_vector_indexes",
        lambda: MagicMock(side_effect=pvi.PerKbVectorIndexBuildInProgress("held")),
    ),
    "drop_table_wait": (
        "drop_per_kb_vector_index",
        "drop_per_kb_vector_indexes",
        lambda: MagicMock(
            side_effect=pvi.PerKbVectorIndexTableBusy(
                "busy", holders=[pvi._holder_evidence(_LIVE_BUILD)], lock_held=True
            )
        ),
    ),
}


@pytest.mark.parametrize("site", sorted(_RETRY_SITES))
def test_every_retry_site_clears_celerys_own_ceiling_after_many_waits(monkeypatch, site):
    """Not the spy: Celery's own ceiling check, at every site, far past the budget.

    ``is_eager`` makes ``retry`` hand the signature back instead of publishing it,
    which leaves only the arithmetic this depends on -- ``request.retries + 1``
    against the ``max_retries`` the task passes. A site that forgot the override
    raises ``MaxRetriesExceededError`` (or re-raises its exception) here instead of
    rescheduling; on the drop task's counted path that is a deleted knowledge
    base's index orphaned for good.
    """
    task_name, service_name, stub = _RETRY_SITES[site]
    task = getattr(idx, task_name)
    monkeypatch.setattr(pvi, service_name, stub())
    waits = 60
    task.push_request(
        retries=waits,
        args=[KB],
        kwargs={"table_waits": waits},
        called_directly=False,
        is_eager=True,
        id=f"task-{site}",
    )
    try:
        with pytest.raises(Retry) as raised:
            task.run(KB, table_waits=waits)
    finally:
        task.pop_request()
    assert tuple(raised.value.sig.args) == (KB,)
    expected_waits = waits + 1 if site.endswith("table_wait") else waits
    assert raised.value.sig.kwargs["table_waits"] == expected_waits
    if site.endswith("table_wait"):
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


# ---------------------------------------------------------------------------
# Round 1 review: a gate whose connection was replaced
# ---------------------------------------------------------------------------


def test_a_gate_whose_connection_was_replaced_is_taken_again_before_the_build(monkeypatch):
    """A RESET that fails after the repair DROP discards the backend, and the gate with it.

    The session lock died with the old backend, so a ``held`` flag that survives the
    reconnect would let the build that follows -- up to nine hours of it -- run with
    no gate at all, and its release would unlock nothing.
    """
    conn = _gated("repair", fail_on="RESET lock_timeout")
    _ensure(monkeypatch, conn)
    drop = conn._positions("DROP INDEX CONCURRENTLY")[0]
    create = conn._positions("CREATE INDEX CONCURRENTLY")[0]
    tries = conn.table_tries()
    assert len(tries) == 2, f"the gate must be taken again: {conn.statements}"
    assert tries[0] < drop < tries[1] < create, conn.statements


def test_a_gate_that_cannot_be_taken_again_after_a_reconnect_stops_before_the_build(
    monkeypatch,
):
    conn = _gated("repair", fail_on="RESET lock_timeout")

    original = conn.execute

    def table_taken_by_someone_else_after_the_discard(clause, params=None):
        if conn.pid > 100:
            conn.table_free = False
        return original(clause, params)

    conn.execute = table_taken_by_someone_else_after_the_discard
    outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "table_busy", outcome
    assert conn._positions("CREATE INDEX CONCURRENTLY") == [], conn.statements


def test_releasing_a_lock_this_session_no_longer_holds_is_a_warning(caplog):
    conn = _GatedConn()
    relation = pvi.table_lock_relation()
    conn.taken_on[relation] = 99  # taken on a backend that is gone
    with caplog.at_level(logging.WARNING):
        pvi._release_lock(conn, relation)
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert relation in message, message
    assert "replaced" in message, message


def test_a_connection_that_cannot_be_discarded_while_holding_a_lock_is_an_error(caplog):
    """A pooled connection still holding the table gate gates the whole project."""
    conn = _GatedConn(fail_on=_LOCK_RELEASE)

    def cannot(*_a, **_k):
        raise RuntimeError("invalidate failed")

    conn.invalidate = cannot
    with caplog.at_level(logging.ERROR):
        pvi._release_lock(conn, pvi.table_lock_relation())
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, caplog.records
    assert pvi.table_lock_relation() in errors[0].getMessage()


# ---------------------------------------------------------------------------
# Round 1 review: which holders are a build
# ---------------------------------------------------------------------------


def test_an_idle_in_transaction_holder_refuses_the_gate_but_is_not_a_build(monkeypatch):
    """A stray LOCK TABLE or plain CREATE INDEX left open is not something that finishes.

    It still refuses the gate -- queueing behind it would hold a snapshot too -- but it
    must not buy 48 hours of uncounted waits.
    """
    conn = _gated("build", table_free=False, holders=[_IDLE_IN_TRANSACTION])
    outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "table_busy", outcome
    assert outcome["build_alive"] is False, outcome
    assert outcome["reason"] == "table_held_without_a_build", outcome
    holder = outcome["table_holders"][0]
    assert holder["kind"] == "stalled", holder
    assert holder["xact_s"] == 14_400 and holder["idle_s"] == 14_393, holder
    assert conn.ddl() == []


def test_a_holder_this_role_cannot_see_is_unknown_and_named_once(monkeypatch, caplog):
    """Without pg_read_all_stats another role's backend is all NULLs -- autovacuum included."""
    monkeypatch.setattr(pvi, "_warned_invisible_holders", False)
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            outcome = _ensure(monkeypatch, _gated("build", table_free=False, holders=[_INVISIBLE]))
    assert outcome["build_alive"] is False, outcome
    assert outcome["table_holders"][0]["kind"] == "unknown", outcome
    warnings = [r.getMessage() for r in caplog.records if "pg_read_all_stats" in r.getMessage()]
    assert len(warnings) == 1, warnings


def test_describing_an_invisible_holder_says_so_and_prints_no_none():
    text = pvi.describe_table_holders([pvi._holder_evidence(_INVISIBLE)])
    assert "query not visible" in text, text
    assert "None" not in text, text


def test_a_failed_holder_query_refuses_the_gate_and_is_reported(monkeypatch, caplog):
    """The evidence is unreadable: refuse, say why, and let the counted path decide."""
    conn = _gated("build", table_free=True, holders_error=RuntimeError("pg_locks unreadable"))
    with caplog.at_level(logging.WARNING):
        outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "table_busy", outcome
    assert outcome["build_alive"] is False, outcome
    assert outcome["reason"] == "table_holders_unreadable", outcome
    assert "pg_locks unreadable" in outcome["evidence_error"], outcome
    assert conn.ddl() == []
    assert conn.table_releases(), "a gate taken before the failed check is given back"
    assert any("pg_locks unreadable" in r.getMessage() for r in caplog.records)


def test_a_failed_holder_query_does_not_orphan_a_deleted_kbs_index():
    """It used to surface as PerKbVectorIndexDropFailed -- "drop it by hand" -- with no DROP issued."""
    conn = _drop_conn(table_free=False, holders_error=RuntimeError("pg_locks unreadable"))
    with pytest.raises(pvi.PerKbVectorIndexTableBusy) as raised:
        pvi.drop_per_kb_vector_indexes(KB, engine=_FakeEngine(conn))
    assert raised.value.evidence_error
    assert conn.ddl() == []


def test_the_gate_holder_between_two_statements_is_a_live_build(monkeypatch):
    """Between try-lock and the CIC's grant, or DROP and CREATE, only the advisory lock shows."""
    conn = _gated("build", table_free=False, holders=[], advisory_holders=[_advisory_holder(2)])
    outcome = _ensure(monkeypatch, conn)
    assert outcome["build_alive"] is True, outcome
    assert outcome["table_holders"][0]["lock"] == "advisory", outcome
    assert outcome["table_holders"][0]["pid"] == 7171


def test_a_gate_holder_idle_for_an_hour_is_not_a_build(monkeypatch):
    conn = _gated("build", table_free=False, holders=[], advisory_holders=[_advisory_holder(3_600)])
    outcome = _ensure(monkeypatch, conn)
    assert outcome["build_alive"] is False, outcome
    assert outcome["table_holders"][0]["kind"] == "stalled", outcome


def test_the_advisory_holder_is_only_looked_up_when_the_gate_is_refused(monkeypatch):
    conn = _gated("build")
    _ensure(monkeypatch, conn)
    assert not [st for st in conn.statements if "locktype = 'advisory'" in st]


def test_evidence_from_every_refused_dimension_is_kept(monkeypatch):
    """A later dimension that saw nothing must not turn a live build into a dead holder."""
    conn = _ensure_conn(
        cls=_GatedConn,
        existing=(),
        dims_present=(768, 1536),
        rows_by_dims={768: 20_000, 1536: 20_000},
        table_free=False,
        holders_seq=[[_LIVE_BUILD], []],
    )
    outcome = _ensure(monkeypatch, conn)
    assert outcome["build_alive"] is True, outcome
    assert [h["pid"] for h in outcome["table_holders"]] == [4242], outcome


# ---------------------------------------------------------------------------
# Round 1 review: table_busy outranks every other status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("other", ["build_lock_held", "build_repeatedly_failed", "cap"])
def test_table_busy_outranks_what_another_dimension_reports(monkeypatch, other):
    kwargs: dict = dict(
        cls=_GatedConn,
        dims_present=(768, 1536),
        rows_by_dims={768: 20_000, 1536: 20_000},
        table_free=False,
        holders=[_LIVE_BUILD],
    )
    if other == "build_lock_held":
        kwargs.update(existing=(), busy_index_dims=(768,))
    elif other == "build_repeatedly_failed":
        kwargs.update(
            existing=(
                _index_row(KB, 768, valid=False, failures=pvi.MAX_CONSECUTIVE_BUILD_FAILURES),
            )
        )
    else:
        kwargs.update(existing=(), index_count=pvi.MAX_PER_KB_INDEXES)
    conn = _ensure_conn(**kwargs)
    if other == "cap":
        # The cap refuses 768 before any DDL; 1536 is its own dimension's place --
        # make the project exactly full for the first look and room for the second.
        counts = iter([pvi.MAX_PER_KB_INDEXES, 0])
        monkeypatch.setattr(pvi, "per_kb_index_count", lambda _conn: next(counts))
    outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "table_busy", outcome
    assert pvi.per_kb_index_name(KB, 1536) in outcome["table_busy"], outcome


# ---------------------------------------------------------------------------
# Round 1 review: the task's side of it
# ---------------------------------------------------------------------------


def _busy_with(*rows) -> dict:
    holders = [pvi._holder_evidence(row) for row in rows]
    alive = any(h["kind"] == "running" for h in holders)
    return {
        **_busy(alive=alive),
        "table_holders": holders,
        "reason": "table_ddl_in_progress" if alive else "table_held_without_a_build",
    }


def test_a_table_held_by_a_non_build_takes_the_counted_path_with_a_warning(ensure_task, caplog):
    with caplog.at_level(logging.WARNING):
        with pytest.raises(Retry):
            ensure_task(_busy_with(_IDLE_IN_TRANSACTION), retries=0)
    call = ensure_task.spy.call_args.kwargs
    assert call.get("kwargs", {"table_waits": 0})["table_waits"] == 0, "counted, not a wait"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("5151" in w for w in warnings), warnings


def test_a_build_older_than_half_a_day_is_waited_for_at_warning(ensure_task, caplog):
    old = list(_LIVE_BUILD)
    old[5] = idx.PER_KB_TABLE_WAIT_WARN_AFTER_S + 60
    with caplog.at_level(logging.INFO):
        with pytest.raises(Retry):
            ensure_task(_busy_with(tuple(old)))
    waits = [r for r in caplog.records if "Waiting to" in r.getMessage()]
    assert waits and waits[0].levelno == logging.WARNING, waits


def test_a_young_build_is_waited_for_at_info(ensure_task, caplog):
    with caplog.at_level(logging.INFO):
        with pytest.raises(Retry):
            ensure_task(_busy_with(_LIVE_BUILD))
    waits = [r for r in caplog.records if "Waiting to" in r.getMessage()]
    assert waits and waits[0].levelno == logging.INFO, waits


def test_the_summary_keeps_the_skips_table_busy_overrode(ensure_task, caplog):
    outcome = {**_busy(alive=True), "build_repeatedly_failed": [768]}
    with caplog.at_level(logging.INFO):
        with pytest.raises(Retry):
            ensure_task(outcome)
    summary = [r.getMessage() for r in caplog.records if "outcome=" in r.getMessage()]
    assert summary and "build_repeatedly_failed=768" in summary[0], summary


def test_a_wait_the_broker_refuses_is_logged_and_raised(monkeypatch, caplog):
    """Celery wraps a failed publish in Reject even with throw=False."""
    task = idx.ensure_per_kb_vector_index
    monkeypatch.setattr(task, "retry", MagicMock(side_effect=Reject("broker down")))
    monkeypatch.setattr(pvi, "ensure_per_kb_vector_index", MagicMock(return_value=_busy()))
    task.push_request(retries=0, args=[KB], kwargs={})
    try:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Reject):
                task.run(KB)
    finally:
        task.pop_request()
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(KB in e for e in errors), errors


def test_a_drop_the_broker_cannot_reschedule_names_the_orphans(monkeypatch, caplog):
    task = idx.drop_per_kb_vector_index
    monkeypatch.setattr(task, "retry", MagicMock(side_effect=Reject("broker down")))
    busy = pvi.PerKbVectorIndexTableBusy(
        "busy", holders=[pvi._holder_evidence(_LIVE_BUILD)], lock_held=True
    )
    monkeypatch.setattr(pvi, "drop_per_kb_vector_indexes", MagicMock(side_effect=busy))
    monkeypatch.setattr(idx, "_orphaned_vector_index_names", lambda kb: ["ai.hnsw_kb_abc_1536"])
    task.push_request(retries=0, args=[KB], kwargs={})
    try:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Reject):
                task.run(KB)
    finally:
        task.pop_request()
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("ai.hnsw_kb_abc_1536" in e for e in errors), errors


def test_the_drop_task_gives_up_waiting_at_the_bound_and_names_the_orphans(monkeypatch, caplog):
    task = idx.drop_per_kb_vector_index
    spy = MagicMock(side_effect=lambda *a, **k: Retry("retry"))
    monkeypatch.setattr(task, "retry", spy)
    busy = pvi.PerKbVectorIndexTableBusy(
        "busy", holders=[pvi._holder_evidence(_LIVE_BUILD)], lock_held=True
    )
    monkeypatch.setattr(pvi, "drop_per_kb_vector_indexes", MagicMock(side_effect=busy))
    monkeypatch.setattr(idx, "_orphaned_vector_index_names", lambda kb: ["ai.hnsw_kb_abc_1536"])
    bound = idx.PER_KB_TABLE_MAX_WAITS
    task.push_request(retries=bound, args=[KB], kwargs={"table_waits": bound})
    try:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(pvi.PerKbVectorIndexTableWaitExhausted):
                task.run(KB, table_waits=bound)
    finally:
        task.pop_request()
    spy.assert_not_called()
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("ai.hnsw_kb_abc_1536" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Round 1 review: one waiter per knowledge base
# ---------------------------------------------------------------------------


def _run_as(task, task_id, *, table_waits=0, retries=0):
    task.push_request(retries=retries, args=[KB], kwargs={"table_waits": table_waits}, id=task_id)
    try:
        return task.run(KB, table_waits=table_waits)
    finally:
        task.pop_request()


def test_a_dispatch_behind_a_waiting_task_is_superseded_without_a_survey(monkeypatch, waiter_store):
    task = idx.ensure_per_kb_vector_index
    monkeypatch.setattr(task, "retry", MagicMock(side_effect=lambda *a, **k: Retry("retry")))
    service = MagicMock(return_value=_busy())
    monkeypatch.setattr(pvi, "ensure_per_kb_vector_index", service)

    with pytest.raises(Retry):
        _run_as(task, "waiter-1")
    assert service.call_count == 1
    assert waiter_store.ttl(idx._waiter_key(KB)) >= idx.PER_KB_TABLE_WAIT_COUNTDOWN_S

    outcome = _run_as(task, "dispatch-2")
    assert outcome["status"] == "superseded", outcome
    assert outcome["waiter"] == "waiter-1", outcome
    assert service.call_count == 1, "a superseded dispatch runs no survey at all"


def test_the_waiter_itself_is_not_superseded_and_gives_the_marker_back(monkeypatch, waiter_store):
    task = idx.ensure_per_kb_vector_index
    monkeypatch.setattr(task, "retry", MagicMock(side_effect=lambda *a, **k: Retry("retry")))
    service = MagicMock(return_value=_busy())
    monkeypatch.setattr(pvi, "ensure_per_kb_vector_index", service)
    with pytest.raises(Retry):
        _run_as(task, "waiter-1")

    service.return_value = {"status": "ready", "built": ["x"], "dropped": []}
    outcome = _run_as(task, "waiter-1", table_waits=1, retries=1)
    assert outcome["status"] == "ready", outcome
    assert waiter_store.get(idx._waiter_key(KB)) is None, "the marker goes with the wait"


def test_a_second_task_that_finds_the_table_busy_leaves_the_wait_to_the_first(
    monkeypatch, waiter_store
):
    """Both surveyed before either became the waiter: only one of them keeps polling."""
    task = idx.ensure_per_kb_vector_index
    spy = MagicMock(side_effect=lambda *a, **k: Retry("retry"))
    monkeypatch.setattr(task, "retry", spy)
    service = MagicMock(return_value=_busy())
    monkeypatch.setattr(pvi, "ensure_per_kb_vector_index", service)
    real_current = idx._current_waiter
    monkeypatch.setattr(idx, "_current_waiter", lambda kb_id: None)  # both pass the entry check
    with pytest.raises(Retry):
        _run_as(task, "waiter-1")
    outcome = _run_as(task, "dispatch-2")
    monkeypatch.setattr(idx, "_current_waiter", real_current)
    assert outcome["status"] == "superseded", outcome
    assert spy.call_count == 1


def test_a_counted_retry_gives_the_marker_back(monkeypatch, waiter_store):
    task = idx.ensure_per_kb_vector_index
    monkeypatch.setattr(task, "retry", MagicMock(side_effect=lambda *a, **k: Retry("retry")))
    service = MagicMock(return_value=_busy())
    monkeypatch.setattr(pvi, "ensure_per_kb_vector_index", service)
    with pytest.raises(Retry):
        _run_as(task, "waiter-1")
    service.side_effect = _transient_exc()
    with pytest.raises(Retry):
        _run_as(task, "waiter-1", table_waits=1, retries=1)
    assert waiter_store.get(idx._waiter_key(KB)) is None


def test_an_unreachable_waiter_store_costs_the_dedup_and_nothing_else(monkeypatch):
    class _Down:
        def __getattr__(self, name):
            def boom(*a, **k):
                raise ConnectionError("redis down")

            return boom

    monkeypatch.setattr(idx, "_waiter_redis", lambda: _Down())
    task = idx.ensure_per_kb_vector_index
    monkeypatch.setattr(task, "retry", MagicMock(side_effect=lambda *a, **k: Retry("retry")))
    monkeypatch.setattr(pvi, "ensure_per_kb_vector_index", MagicMock(return_value=_busy()))
    with pytest.raises(Retry):
        _run_as(task, "waiter-1")


def test_a_task_with_no_id_skips_the_marker(monkeypatch, waiter_store):
    """Called directly -- no Celery request, nothing to tell two tasks apart by."""
    monkeypatch.setattr(
        pvi,
        "ensure_per_kb_vector_index",
        MagicMock(return_value={"status": "ready", "built": [], "dropped": []}),
    )
    idx.ensure_per_kb_vector_index.run(KB)
    assert waiter_store.keys() == []


_REAL_REDIS_URL = __import__("os").getenv("TEST_REDIS_URL")


@pytest.fixture(params=["fakeredis", "redis"])
def marker_client(request, monkeypatch):
    if request.param == "fakeredis":
        client = fakeredis.FakeStrictRedis()
    else:
        if not _REAL_REDIS_URL:
            pytest.skip("TEST_REDIS_URL not set")
        import redis

        client = redis.from_url(_REAL_REDIS_URL)
        client.flushdb()
    monkeypatch.setattr(idx, "_waiter_redis", lambda: client)
    yield client
    if request.param == "redis":
        client.flushdb()


def test_the_marker_is_claimed_once_renewed_by_its_owner_and_released_only_by_it(
    marker_client,
):
    assert idx._claim_waiter(KB, "a") == "a"
    assert idx._claim_waiter(KB, "b") == "a", "a second claimant learns who waits"
    marker_client.expire(idx._waiter_key(KB), 5)
    assert idx._claim_waiter(KB, "a") == "a"
    assert marker_client.ttl(idx._waiter_key(KB)) > 5, "the owner's claim renews it"
    idx._release_waiter(KB, "b")
    assert idx._current_waiter(KB) == "a", "only the owner gives it back"
    idx._release_waiter(KB, "a")
    assert idx._current_waiter(KB) is None
