"""The stale-run cutoff is a project setting, bound into both UPDATEs, and the
cleanup runs from a self-rescheduling task (the project stack runs no beat)."""

from unittest.mock import MagicMock, patch

import fakeredis
import redis

from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY
from agentic_project_service.tasks import cleanup as cleanup_mod


def test_setting_registered():
    d = SETTINGS_REGISTRY["RUN_STALE_AFTER_SECONDS"]
    assert (d.type, d.default, d.min, d.max, d.category) == ("int", 1800, 600, 86400, "agents")


def _fake_db(monkeypatch, rowcounts=(0, 0)):
    fake_db = MagicMock()
    results = [MagicMock(rowcount=n) for n in rowcounts]
    fake_db.session.execute.side_effect = results
    monkeypatch.setattr(cleanup_mod, "db", fake_db)
    return fake_db


def test_cutoff_is_bound_into_both_updates(monkeypatch):
    fake_db = _fake_db(monkeypatch)
    monkeypatch.setattr(
        cleanup_mod, "get_setting", lambda key: {"RUN_STALE_AFTER_SECONDS": 2400}[key]
    )

    assert cleanup_mod.run_stale_cleanup() == {"agent_runs": 0, "orchestration_runs": 0}

    calls = fake_db.session.execute.call_args_list
    assert len(calls) == 2
    tables = []
    for call in calls:
        sql = str(call.args[0])
        assert "INTERVAL '10 minutes'" not in sql
        assert "(:stale_after * INTERVAL '1 second')" in sql
        assert call.args[1] == {"stale_after": 2400}
        tables.append("agent_runs" if ".agent_runs" in sql else "orchestration_runs")
    assert tables == ["agent_runs", "orchestration_runs"]
    fake_db.session.commit.assert_called_once()


def test_counts_are_reported(monkeypatch, caplog):
    _fake_db(monkeypatch, rowcounts=(3, 1))
    monkeypatch.setattr(cleanup_mod, "get_setting", lambda key: 1800)
    with caplog.at_level("INFO", logger=cleanup_mod.logger.name):
        assert cleanup_mod.run_stale_cleanup() == {"agent_runs": 3, "orchestration_runs": 1}
    assert "Cleaned up 3 stale agent runs and 1 stale orchestration runs" in caplog.text


def test_the_task_runs_the_cleanup(monkeypatch):
    monkeypatch.setattr(
        cleanup_mod, "run_stale_cleanup", lambda: {"agent_runs": 2, "orchestration_runs": 0}
    )
    assert cleanup_mod.cleanup_stale_runs.run() == {"agent_runs": 2, "orchestration_runs": 0}


# --- the self-rescheduling tick (the project stack runs no Celery beat) ---


def test_tick_runs_cleanup_under_the_lock_and_reschedules():
    fake_redis = MagicMock()
    fake_redis.set.return_value = True
    with (
        patch.object(cleanup_mod, "_get_redis", return_value=(fake_redis, "ref-a:stale_runs_lock")),
        patch.object(
            cleanup_mod,
            "run_stale_cleanup",
            return_value={"agent_runs": 1, "orchestration_runs": 0},
        ) as run,
        patch.object(cleanup_mod.stale_runs_tick, "apply_async") as reschedule,
    ):
        cleanup_mod.stale_runs_tick.run()
    fake_redis.set.assert_called_once_with("ref-a:stale_runs_lock", "1", nx=True, ex=60)
    run.assert_called_once()
    fake_redis.delete.assert_called_once_with("ref-a:stale_runs_lock")
    assert reschedule.call_args.kwargs["countdown"] == 300


def test_tick_skips_work_when_another_worker_holds_the_lock():
    fake_redis = MagicMock()
    fake_redis.set.return_value = False
    with (
        patch.object(cleanup_mod, "_get_redis", return_value=(fake_redis, "k")),
        patch.object(cleanup_mod, "run_stale_cleanup") as run,
        patch.object(cleanup_mod.stale_runs_tick, "apply_async") as reschedule,
    ):
        cleanup_mod.stale_runs_tick.run()
    run.assert_not_called()
    fake_redis.delete.assert_not_called()
    reschedule.assert_called_once()


def test_tick_reschedules_even_when_cleanup_fails():
    fake_redis = MagicMock()
    fake_redis.set.return_value = True
    with (
        patch.object(cleanup_mod, "_get_redis", return_value=(fake_redis, "k")),
        patch.object(cleanup_mod, "run_stale_cleanup", side_effect=RuntimeError("db down")),
        patch.object(cleanup_mod.stale_runs_tick, "apply_async") as reschedule,
    ):
        cleanup_mod.stale_runs_tick.run()
    fake_redis.delete.assert_called_once_with("k")
    reschedule.assert_called_once()


def test_tick_reschedules_when_redis_is_down():
    broken = MagicMock()
    broken.set.side_effect = redis.ConnectionError("down")
    with (
        patch.object(cleanup_mod, "_get_redis", return_value=(broken, "k")),
        patch.object(cleanup_mod, "run_stale_cleanup") as run,
        patch.object(cleanup_mod.stale_runs_tick, "apply_async") as reschedule,
    ):
        cleanup_mod.stale_runs_tick.run()
    run.assert_not_called()
    reschedule.assert_called_once()


def test_tick_reschedules_when_no_redis_client_can_be_built(monkeypatch):
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    with (
        patch.object(cleanup_mod, "run_stale_cleanup") as run,
        patch.object(cleanup_mod.stale_runs_tick, "apply_async") as reschedule,
    ):
        cleanup_mod.stale_runs_tick.run()
    run.assert_not_called()
    reschedule.assert_called_once()


def test_two_ticks_at_once_clean_up_once():
    """Every worker seeds a chain; the lock keeps the work to one per tick."""
    server = fakeredis.FakeServer()
    held = fakeredis.FakeStrictRedis(server=server)
    held.set("ref-a:stale_runs_lock", "1", nx=True, ex=60)  # another worker is mid-tick
    with (
        patch.object(
            cleanup_mod,
            "_get_redis",
            return_value=(fakeredis.FakeStrictRedis(server=server), "ref-a:stale_runs_lock"),
        ),
        patch.object(cleanup_mod, "run_stale_cleanup") as run,
        patch.object(cleanup_mod.stale_runs_tick, "apply_async"),
    ):
        cleanup_mod.stale_runs_tick.run()
    run.assert_not_called()
    assert 0 < held.ttl("ref-a:stale_runs_lock") <= 60


def test_the_lock_key_carries_the_project_ref(monkeypatch):
    monkeypatch.setenv("PROJECT_REF", "ref-a")
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://127.0.0.1:1/0")
    _, key = cleanup_mod._get_redis()
    assert key == "ref-a:stale_runs_lock"


def test_worker_ready_seeds_the_tick():
    from agentic_project_service import celery as celery_mod

    with patch.object(cleanup_mod.stale_runs_tick, "delay") as delay:
        celery_mod.seed_stale_runs_tick()
    delay.assert_called_once()


def test_the_beat_entry_stays_for_deployments_that_run_beat():
    from agentic_project_service.celery import celery_app

    assert celery_app.conf.beat_schedule["cleanup-stale-runs"]["task"] == "cleanup_stale_runs"
