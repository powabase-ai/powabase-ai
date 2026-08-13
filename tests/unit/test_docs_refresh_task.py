"""Tests for the self-rescheduling docs-refresh Celery task (tasks/docs_refresh.py).

These guard the three fleet-safety behaviours that have no other coverage:
  - the ``DOCS_KB_REFRESH_ENABLED`` gate (the single guard stopping EVERY
    project's worker from git-cloning docs repos + indexing — a broken gate is a
    fleet-wide cost event),
  - the Redis SET-NX lock (only one worker replica runs the refresh per tick),
  - the ``finally`` re-arm (if broken, periodic refresh silently stops after one
    run — there is no Celery beat to restart it).

Redis, the refresh service, and the self-reschedule are all mocked, so these run
without a broker, DB, or embeddings.
"""

from unittest.mock import MagicMock

from agentic_project_service.tasks import docs_refresh as task_mod


def _patch_common(mocker, *, lock_acquired=True):
    """Mock redis + the refresh service + the self-reschedule. Returns the mocks."""
    redis_mock = MagicMock()
    redis_mock.set.return_value = lock_acquired
    mocker.patch.object(task_mod, "_get_redis", return_value=(redis_mock, "ref:docs_refresh_lock"))
    refresh_mock = mocker.patch.object(task_mod, "refresh_docs_kb", return_value={"docs": 3})
    rearm_mock = mocker.patch.object(task_mod.refresh_docs_kb_task, "apply_async")
    return redis_mock, refresh_mock, rearm_mock


def test_gate_off_does_not_run_or_rearm(mocker, monkeypatch):
    """Not the docs project (gate unset/false): don't run the refresh AND don't
    self-perpetuate — otherwise every project worker would loop forever."""
    monkeypatch.delenv("DOCS_KB_REFRESH_ENABLED", raising=False)
    redis_mock, refresh_mock, rearm_mock = _patch_common(mocker)

    task_mod.refresh_docs_kb_task()

    refresh_mock.assert_not_called()
    rearm_mock.assert_not_called()  # gate returns BEFORE the re-arm finally
    redis_mock.set.assert_not_called()


def test_gate_on_lock_acquired_runs_releases_and_rearms(mocker, monkeypatch):
    monkeypatch.setenv("DOCS_KB_REFRESH_ENABLED", "true")
    redis_mock, refresh_mock, rearm_mock = _patch_common(mocker, lock_acquired=True)

    task_mod.refresh_docs_kb_task()

    refresh_mock.assert_called_once()
    redis_mock.eval.assert_called_once()  # lock released via compare-and-delete
    rearm_mock.assert_called_once()
    # re-arm schedules the next tick with a countdown
    assert "countdown" in rearm_mock.call_args.kwargs


def test_gate_on_lock_held_skips_refresh_but_rearms(mocker, monkeypatch):
    """Another replica holds the lock: skip the refresh this tick but STILL re-arm
    (each replica's chain must keep the schedule alive)."""
    monkeypatch.setenv("DOCS_KB_REFRESH_ENABLED", "true")
    redis_mock, refresh_mock, rearm_mock = _patch_common(mocker, lock_acquired=False)

    task_mod.refresh_docs_kb_task()

    refresh_mock.assert_not_called()
    redis_mock.eval.assert_not_called()  # we never held the lock
    rearm_mock.assert_called_once()


def test_rearm_happens_even_when_refresh_raises(mocker, monkeypatch):
    """A refresh that raises must not kill the schedule (the finally re-arms) and
    must still release the lock."""
    monkeypatch.setenv("DOCS_KB_REFRESH_ENABLED", "true")
    redis_mock, refresh_mock, rearm_mock = _patch_common(mocker, lock_acquired=True)
    refresh_mock.side_effect = RuntimeError("boom")

    task_mod.refresh_docs_kb_task()  # must not raise

    redis_mock.eval.assert_called_once()  # lock released (compare-and-delete) despite the error
    rearm_mock.assert_called_once()  # schedule survives


def test_get_redis_client_is_cached(mocker):
    """The Redis client is built once and reused across ticks — no fresh
    ConnectionPool per call (see tasks/docs_refresh.py's _get_redis)."""
    mocker.patch.object(task_mod, "_redis_client", None)
    fake_client = object()
    from_url_mock = mocker.patch(
        "agentic_project_service.tasks.docs_refresh.redis.from_url",
        return_value=fake_client,
    )

    first, _ = task_mod._get_redis()
    second, _ = task_mod._get_redis()

    assert first is second is fake_client
    from_url_mock.assert_called_once()


def test_next_interval_floor(monkeypatch):
    """A mis-set interval env can never tight-loop below the floor."""
    monkeypatch.setenv("DOCS_KB_REFRESH_INTERVAL_SECONDS", "1")
    assert task_mod._next_interval() == task_mod._MIN_INTERVAL_SECONDS
    monkeypatch.setenv("DOCS_KB_REFRESH_INTERVAL_SECONDS", "not-a-number")
    assert task_mod._next_interval() == task_mod._DEFAULT_INTERVAL_SECONDS
