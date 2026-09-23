"""Dispatch wiring for the per-knowledge-base vector index.

The service decides *whether* to reconcile an index in the indexing path, so a
project whose knowledge bases are all small never enqueues a task at all. These
pin that decision, that a failure to dispatch cannot fail indexing, and that
the start-up sweep and the knowledge-base delete are wired to the same tasks the
pg_search path uses at those points.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from agentic_project_service.services import pg_vector_index as pvi
from agentic_project_service.tasks import indexing as idx

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


@pytest.fixture
def dispatched(monkeypatch):
    """Record the ensure task's dispatches instead of enqueueing them."""
    calls: list[str] = []
    monkeypatch.setattr(
        idx.ensure_per_kb_vector_index, "delay", lambda kb_id, *a, **k: calls.append(kb_id)
    )
    return calls


def _action(monkeypatch, value):
    monkeypatch.setattr(idx, "_per_kb_vector_index_action", lambda kb_id: value)


# ---------------------------------------------------------------------------
# After a source finishes indexing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["build", "drop"])
def test_a_source_that_crosses_a_threshold_dispatches_a_reconcile(monkeypatch, dispatched, action):
    _action(monkeypatch, action)
    assert idx.dispatch_per_kb_vector_index(KB) is True
    assert dispatched == [KB]


def test_a_source_that_changes_nothing_dispatches_nothing(monkeypatch, dispatched):
    """The point of checking the row count here: no task per indexed source."""
    _action(monkeypatch, None)
    assert idx.dispatch_per_kb_vector_index(KB) is False
    assert dispatched == []


def test_a_broker_failure_does_not_fail_the_indexing_run(monkeypatch):
    """Indexing has already committed; the start-up sweep is the backstop."""
    _action(monkeypatch, "build")

    def boom(kb_id, *a, **k):
        raise RuntimeError("broker down")

    monkeypatch.setattr(idx.ensure_per_kb_vector_index, "delay", boom)
    assert idx.dispatch_per_kb_vector_index(KB) is False


def test_an_unreadable_row_count_is_not_an_error(monkeypatch):
    """A knowledge base whose state cannot be read keeps the index it has."""

    class _Session:
        def connection(self):
            raise RuntimeError("no connection")

        def rollback(self):
            pass

    monkeypatch.setattr(idx.db, "session", _Session(), raising=False)
    assert idx._per_kb_vector_index_action(KB) is None


def test_the_dispatch_is_wired_into_the_post_commit_side_effects():
    """It has to run after the commit, next to the shared index's ensure.

    CREATE INDEX CONCURRENTLY cannot run in a transaction at all, and the rows
    that may have crossed the threshold are only visible once committed.
    """
    body = inspect.getsource(idx._run_index_body)
    assert "dispatch_per_kb_vector_index(knowledge_base_id)" in body
    ensure_at = body.index("ensure_embedding_index(db.session")
    dispatch_at = body.index("dispatch_per_kb_vector_index(knowledge_base_id)")
    assert ensure_at < dispatch_at, "the dispatch must follow the committing ensure"


# ---------------------------------------------------------------------------
# Start-up sweep
# ---------------------------------------------------------------------------


def test_the_start_up_sweep_dispatches_one_reconcile_per_knowledge_base(monkeypatch, dispatched):
    other = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
    monkeypatch.setattr(pvi, "kbs_needing_a_per_kb_index", lambda engine: [KB, other])
    assert idx.dispatch_per_kb_vector_indexes_at_start(object()) == [KB, other]
    assert dispatched == [KB, other]


def test_the_start_up_sweep_never_raises(monkeypatch, dispatched):
    """A start-up must not be stopped by a sweep; a source finishing retries it."""

    def boom(engine):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(pvi, "kbs_needing_a_per_kb_index", boom)
    assert idx.dispatch_per_kb_vector_indexes_at_start(object()) == []
    assert dispatched == []


def test_one_failed_dispatch_does_not_stop_the_others(monkeypatch):
    other = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
    monkeypatch.setattr(pvi, "kbs_needing_a_per_kb_index", lambda engine: [KB, other])
    sent: list[str] = []

    def flaky(kb_id, *a, **k):
        if kb_id == KB:
            raise RuntimeError("broker down")
        sent.append(kb_id)

    monkeypatch.setattr(idx.ensure_per_kb_vector_index, "delay", flaky)
    assert idx.dispatch_per_kb_vector_indexes_at_start(object()) == [other]
    assert sent == [other]


def test_the_sweep_reads_its_settings_without_touching_the_shared_session():
    """It runs inside the boot's migration transaction.

    ``get_setting`` reads ``ai.project_settings`` through ``db.session``, and on
    a database where that table does not exist yet -- the state at a first boot
    -- the failed read leaves that transaction aborted and takes the boot's own
    next statement with it (observed: "Database migration failed ... current
    transaction is aborted"). So the sweep reads its thresholds on a connection
    of its own.
    """
    source = inspect.getsource(pvi.kbs_needing_a_per_kb_index)
    assert "read_overrides(conn" in source
    assert "get_setting" not in source
    assert "thresholds()" not in source, "the no-argument form goes through db.session"


def test_the_sweep_runs_at_the_same_point_as_the_bm25_one():
    """Not a second mechanism: the same start-up hook, beside the BM25 sweep."""
    from agentic_project_service import main

    source = inspect.getsource(main)
    assert "dispatch_per_kb_vector_indexes_at_start(db.engine)" in source
    bm25_at = source.index("dispatch_partition_completion_at_start(db.engine)")
    vector_at = source.index("dispatch_per_kb_vector_indexes_at_start(db.engine)")
    assert abs(vector_at - bm25_at) < 2000, "the two sweeps must sit together"


# ---------------------------------------------------------------------------
# Knowledge-base delete
# ---------------------------------------------------------------------------


def test_deleting_a_knowledge_base_drops_its_vector_index():
    """The KB row is gone, so nothing else will ever reconcile the index."""
    from agentic_project_service.routes import knowledge_bases as routes

    source = inspect.getsource(routes)
    assert "drop_per_kb_vector_index.delay(kb_id)" in source
    bm25_at = source.index("drop_pg_bm25_index.delay(kb_id)")
    vector_at = source.index("drop_per_kb_vector_index.delay(kb_id)")
    assert bm25_at < vector_at, "both drops belong to the delete path"


# ---------------------------------------------------------------------------
# The tasks' own retry decisions
# ---------------------------------------------------------------------------


@pytest.fixture
def retry_spy(monkeypatch):
    """Replace both tasks' self.retry with a spy that hands back a Retry."""
    from celery.exceptions import Retry

    spies = {}
    for task in (idx.ensure_per_kb_vector_index, idx.drop_per_kb_vector_index):
        spy = MagicMock(side_effect=Retry("retry"))
        monkeypatch.setattr(task, "retry", spy)
        spies[task.name] = spy
    return spies


def _lost_connection():
    """A transient database failure the pg_search path already classifies as one."""
    from sqlalchemy.exc import OperationalError

    class _Orig(Exception):
        sqlstate = "57P01"

    return OperationalError("SELECT 1", {}, _Orig("terminating connection"))


def test_the_drop_task_retries_a_build_that_is_in_progress(monkeypatch, retry_spy):
    """A contended drop must requeue, not report a drop that did not happen."""
    from celery.exceptions import Retry

    monkeypatch.setattr(
        pvi,
        "drop_per_kb_vector_indexes",
        MagicMock(side_effect=pvi.PerKbVectorIndexBuildInProgress("held")),
    )
    with pytest.raises(Retry):
        idx.drop_per_kb_vector_index.run(KB)
    spy = retry_spy[idx.drop_per_kb_vector_index.name]
    spy.assert_called_once()
    assert spy.call_args.kwargs["countdown"] > 0


def test_the_ensure_task_retries_a_transient_database_failure(monkeypatch, retry_spy):
    from celery.exceptions import Retry

    monkeypatch.setattr(
        pvi, "ensure_per_kb_vector_index", MagicMock(side_effect=_lost_connection())
    )
    with pytest.raises(Retry):
        idx.ensure_per_kb_vector_index.run(KB)
    retry_spy[idx.ensure_per_kb_vector_index.name].assert_called_once()


def test_the_ensure_task_does_not_retry_a_real_failure(monkeypatch, retry_spy):
    """A bad knowledge base id is not going to get better; fail it loudly."""
    monkeypatch.setattr(
        pvi, "ensure_per_kb_vector_index", MagicMock(side_effect=ValueError("not a UUID"))
    )
    with pytest.raises(ValueError):
        idx.ensure_per_kb_vector_index.run(KB)
    retry_spy[idx.ensure_per_kb_vector_index.name].assert_not_called()


def test_the_drop_task_names_the_orphans_when_it_gives_up(monkeypatch, caplog):
    """Nothing comes back to an index whose knowledge base row is already gone.

    So the last retry has to name it: the route's dispatch-failure path says "has
    to be dropped by hand" and the task's exhaustion path must too, or an
    operator has no way to find what is left behind.
    """
    import logging

    from celery.exceptions import Retry

    monkeypatch.setattr(
        pvi,
        "drop_per_kb_vector_indexes",
        MagicMock(side_effect=pvi.PerKbVectorIndexBuildInProgress("held")),
    )
    monkeypatch.setattr(idx, "_orphaned_vector_index_names", lambda kb: ["ai.hnsw_kb_abc_1536"])
    task = idx.drop_per_kb_vector_index
    monkeypatch.setattr(task, "retry", MagicMock(side_effect=Retry("retry")))

    # push_request is how Celery itself supplies a Context; `request` is a
    # read-only property, so it cannot be monkeypatched.
    task.push_request(retries=task.max_retries)
    try:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(pvi.PerKbVectorIndexBuildInProgress):
                task.run(KB)
    finally:
        task.pop_request()
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert "dropped by hand" in message, message
    assert "ai.hnsw_kb_abc_1536" in message, message


def test_the_boot_sweeps_count_is_bounded_short_enough_for_a_start_up():
    """It runs on the boot path, after the migrations, before the lock is released.

    Measured 14 ms over 66,000 embeddings -- about 1.1 s extrapolated to 5.3
    million -- so the ceiling is for a pathological case. This codebase's other
    boot-path bound is 5 s; anything much larger is a start-up that looks hung.
    """
    assert 0 < pvi.SWEEP_TIMEOUT_MS <= 10_000


def test_the_ensure_task_returns_the_services_outcome(monkeypatch, retry_spy):
    outcome = {"status": "ready", "built": ["hnsw_kb_x_1536"], "dropped": []}
    monkeypatch.setattr(pvi, "ensure_per_kb_vector_index", MagicMock(return_value=outcome))
    assert idx.ensure_per_kb_vector_index.run(KB) == outcome
