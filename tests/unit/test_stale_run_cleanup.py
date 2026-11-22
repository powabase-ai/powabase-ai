"""The stale-run cutoff is a project setting, bound into both UPDATEs, and the
cleanup runs from a self-rescheduling task (the project stack runs no beat)."""

from unittest.mock import MagicMock, patch

import fakeredis
import pytest
import redis

from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY
from agentic_project_service.tasks import cleanup as cleanup_mod


def test_setting_registered():
    d = SETTINGS_REGISTRY["RUN_STALE_AFTER_SECONDS"]
    assert (d.type, d.default, d.min, d.max, d.category) == ("int", 1800, 600, 86400, "agents")


AGENT_LIVE = ("11111111-1111-4111-8111-111111111111", "run_live")
AGENT_DEAD = ("22222222-2222-4222-8222-222222222222", "run_dead")
ORCH_LIVE = ("33333333-3333-4333-8333-333333333333", "orch_run_live")
ORCH_DEAD = ("44444444-4444-4444-8444-444444444444", "orch_run_dead")


class _FakeDb:
    """Answers the two candidate SELECTs, then counts the UPDATEs."""

    def __init__(self, agent_rows, orch_rows):
        self.agent_rows, self.orch_rows = agent_rows, orch_rows
        self.calls = []
        self.session = MagicMock()
        self.session.execute.side_effect = self._execute

    def _execute(self, statement, params=None):
        sql = str(statement)
        self.calls.append((sql, params))
        result = MagicMock()
        if sql.lstrip().startswith("SELECT"):
            result.fetchall.return_value = (
                self.agent_rows if ".agent_runs" in sql else self.orch_rows
            )
        else:
            result.rowcount = len(params["ids"])
        return result

    def updates(self):
        return [(sql, params) for sql, params in self.calls if "UPDATE" in sql]


@pytest.fixture
def owners(monkeypatch):
    """A fake Redis holding the owner records of live runs."""
    monkeypatch.setenv("PROJECT_REF", "ref-a")
    client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr(cleanup_mod, "broker_client", lambda: client)
    monkeypatch.setattr(
        cleanup_mod, "get_setting", lambda key: {"RUN_STALE_AFTER_SECONDS": 2400}[key]
    )
    return client


def _install(monkeypatch, agent_rows=(), orch_rows=()):
    fake = _FakeDb(list(agent_rows), list(orch_rows))
    monkeypatch.setattr(cleanup_mod, "db", fake)
    return fake


def test_the_cutoff_is_bound_into_both_candidate_queries(owners, monkeypatch):
    fake = _install(monkeypatch)
    assert cleanup_mod.run_stale_cleanup() == {"agent_runs": 0, "orchestration_runs": 0}
    selects = [(sql, params) for sql, params in fake.calls if "SELECT" in sql]
    assert [".agent_runs" in sql for sql, _ in selects] == [True, False]
    for sql, params in selects:
        assert "INTERVAL '10 minutes'" not in sql
        assert "(:stale_after * INTERVAL '1 second')" in sql
        assert "status = 'running'" in sql
        assert params["stale_after"] == 2400
    assert fake.updates() == []  # nothing old enough: no UPDATE


def test_a_long_running_run_with_a_live_owner_survives(owners, monkeypatch):
    owners.hset("runs:owner:ref-a:run_live", mapping={"instance": "pod-a", "owner": ""})
    owners.hset(
        "runs:owner:ref-a:orch_run_live",
        mapping={"instance": "pod-a", "owner": "", "approvable": "0"},
    )
    fake = _install(monkeypatch, [AGENT_LIVE, AGENT_DEAD], [ORCH_LIVE, ORCH_DEAD])

    assert cleanup_mod.run_stale_cleanup() == {"agent_runs": 1, "orchestration_runs": 1}

    (agent_sql, agent_params), (orch_sql, orch_params) = fake.updates()
    assert ".agent_runs" in agent_sql and agent_params["ids"] == [AGENT_DEAD[0]]
    assert ".orchestration_runs" in orch_sql and orch_params["ids"] == [ORCH_DEAD[0]]
    for sql in (agent_sql, orch_sql):
        assert "id = ANY(CAST(:ids AS uuid[]))" in sql
        assert "status = 'running'" in sql  # still running when it is failed
        assert cleanup_mod.INTERRUPTED_ERROR.replace("'", "''") not in sql  # bound, not inlined
    assert agent_params["error"] == cleanup_mod.INTERRUPTED_ERROR
    fake.session.commit.assert_called_once()


def test_an_ownerless_old_run_is_failed(owners, monkeypatch):
    fake = _install(monkeypatch, [AGENT_DEAD])
    assert cleanup_mod.run_stale_cleanup() == {"agent_runs": 1, "orchestration_runs": 0}
    assert [params["ids"] for _, params in fake.updates()] == [[AGENT_DEAD[0]]]


def test_redis_down_fails_no_run(monkeypatch, caplog):
    monkeypatch.setattr(cleanup_mod, "get_setting", lambda key: 1800)
    broken = MagicMock()
    broken.pipeline.side_effect = redis.ConnectionError("down")
    monkeypatch.setattr(cleanup_mod, "broker_client", lambda: broken)
    fake = _install(monkeypatch, [AGENT_DEAD], [ORCH_DEAD])
    with caplog.at_level("WARNING", logger=cleanup_mod.logger.name):
        assert cleanup_mod.run_stale_cleanup() == {"agent_runs": 0, "orchestration_runs": 0}
    assert fake.updates() == []
    assert "skipping this sweep" in caplog.text


def test_no_usable_redis_url_fails_no_run(monkeypatch):
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    monkeypatch.setattr(cleanup_mod, "get_setting", lambda key: 1800)
    fake = _install(monkeypatch, [AGENT_DEAD])
    assert cleanup_mod.run_stale_cleanup() == {"agent_runs": 0, "orchestration_runs": 0}
    assert fake.updates() == []


def test_without_candidates_redis_is_not_asked(monkeypatch):
    monkeypatch.setattr(cleanup_mod, "get_setting", lambda key: 1800)
    client = MagicMock()
    monkeypatch.setattr(cleanup_mod, "broker_client", lambda: client)
    _install(monkeypatch)
    cleanup_mod.run_stale_cleanup()
    client.pipeline.assert_not_called()


def test_counts_are_logged(owners, monkeypatch, caplog):
    _install(monkeypatch, [AGENT_DEAD], [ORCH_DEAD])
    with caplog.at_level("INFO", logger=cleanup_mod.logger.name):
        assert cleanup_mod.run_stale_cleanup() == {"agent_runs": 1, "orchestration_runs": 1}
    assert "Cleaned up 1 stale agent runs and 1 stale orchestration runs" in caplog.text


def test_a_completed_agent_run_drops_a_stale_interrupt_error():
    """A run the cleanup failed while it was in fact alive must not stay
    'completed' with the interrupt error once it finishes."""
    from agentic_project_service.models.tenant import AgentRunStatus
    from agentic_project_service.services.session import update_agent_run

    session = MagicMock()
    update_agent_run(session, "run_1", status=AgentRunStatus.COMPLETED, content="done")
    assert "error = NULL" in str(session.execute.call_args_list[0].args[0])

    session = MagicMock()
    update_agent_run(session, "run_1", status=AgentRunStatus.FAILED, error="boom")
    sql = str(session.execute.call_args_list[0].args[0])
    assert "error = NULL" not in sql and "error = :error" in sql


def test_a_completed_orchestration_run_drops_a_stale_interrupt_error(monkeypatch):
    from agentic_project_service.services import orchestration as orch_service

    fake_db = MagicMock()
    monkeypatch.setattr(orch_service, "db", fake_db)
    orch_service.update_orchestration_run("orch_run_1", status="completed", content="done")
    assert "error = NULL" in str(fake_db.session.execute.call_args_list[0].args[0])

    fake_db.reset_mock()
    orch_service.update_orchestration_run("orch_run_1", status="failed", error="boom")
    assert "error = NULL" not in str(fake_db.session.execute.call_args_list[0].args[0])


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
