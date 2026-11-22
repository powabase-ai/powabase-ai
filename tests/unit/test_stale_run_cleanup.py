"""The stale-run cleanup fails only runs that are old and have no live owner, and
cleanup runs from a self-rescheduling task (the project stack runs no beat)."""

import os
from types import SimpleNamespace
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


_REAL_REDIS_URL = os.getenv("TEST_REDIS_URL")


@pytest.fixture(params=["fakeredis", "redis"])
def chain(request, monkeypatch):
    """One Redis for the tick and its seed (fakeredis, and the real one when
    TEST_REDIS_URL is set); enqueues are recorded, not sent."""
    monkeypatch.setenv("PROJECT_REF", "ref-a")
    if request.param == "fakeredis":
        client = fakeredis.FakeStrictRedis()
    else:
        if not _REAL_REDIS_URL:
            pytest.skip("TEST_REDIS_URL not set")
        client = redis.from_url(_REAL_REDIS_URL)
        client.flushdb()
        request.addfinalizer(client.flushdb)
    monkeypatch.setattr(cleanup_mod, "broker_client", lambda: client)
    seeded, enqueued = [], []
    monkeypatch.setattr(
        cleanup_mod.stale_runs_tick, "delay", lambda generation: seeded.append(generation)
    )
    monkeypatch.setattr(
        cleanup_mod.stale_runs_tick,
        "apply_async",
        lambda args, countdown: enqueued.append((args[0], countdown)),
    )
    return SimpleNamespace(client=client, seeded=seeded, enqueued=enqueued)


def _tick(generation):
    cleanup_mod.stale_runs_tick.run(generation)


def test_n_worker_starts_leave_exactly_one_live_chain(chain, monkeypatch):
    monkeypatch.setattr(cleanup_mod, "run_stale_cleanup", dict)
    for _ in range(5):
        cleanup_mod.seed_stale_runs_chain()
    assert len(set(chain.seeded)) == 5
    for generation in chain.seeded:  # every seeded chain ticks once
        _tick(generation)
    assert chain.enqueued == [(chain.seeded[-1], 300)]
    # ... and the survivor keeps going.
    _tick(chain.seeded[-1])
    assert chain.enqueued[-1] == (chain.seeded[-1], 300)


def test_the_current_chain_runs_the_cleanup_under_the_lock_and_reschedules(chain, monkeypatch):
    seen = {}

    def cleanup():
        seen["lock_ttl"] = chain.client.ttl("ref-a:stale_runs_lock")
        return {"agent_runs": 1, "orchestration_runs": 0}

    monkeypatch.setattr(cleanup_mod, "run_stale_cleanup", cleanup)
    cleanup_mod.seed_stale_runs_chain()
    _tick(chain.seeded[0])
    assert 0 < seen["lock_ttl"] <= 60
    assert not chain.client.exists("ref-a:stale_runs_lock")
    assert chain.enqueued == [(chain.seeded[0], 300)]
    assert 0 < chain.client.ttl("ref-a:stale_runs_chain") <= 900


def test_a_lost_chain_record_is_adopted_by_the_first_tick(chain, monkeypatch):
    monkeypatch.setattr(cleanup_mod, "run_stale_cleanup", dict)
    cleanup_mod.seed_stale_runs_chain()
    cleanup_mod.seed_stale_runs_chain()
    chain.client.flushdb()  # Redis restarted without persistence
    first, second = chain.seeded
    _tick(second)
    _tick(first)
    assert chain.enqueued == [(second, 300)]


def test_a_tick_without_a_generation_never_displaces_a_chain(chain, monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(cleanup_mod, "run_stale_cleanup", run)
    cleanup_mod.seed_stale_runs_chain()
    _tick(None)
    run.assert_not_called()
    assert chain.enqueued == []


def test_the_lock_is_released_only_by_its_holder(chain, monkeypatch):
    """A cleanup that outlived its lock must not delete the next holder's."""

    def slow_cleanup():
        chain.client.set("ref-a:stale_runs_lock", "another-worker", ex=60)

    monkeypatch.setattr(cleanup_mod, "run_stale_cleanup", slow_cleanup)
    cleanup_mod.seed_stale_runs_chain()
    _tick(chain.seeded[0])
    assert chain.client.get("ref-a:stale_runs_lock") == b"another-worker"


def test_tick_skips_work_while_another_worker_holds_the_lock(chain, monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(cleanup_mod, "run_stale_cleanup", run)
    cleanup_mod.seed_stale_runs_chain()
    chain.client.set("ref-a:stale_runs_lock", "other", ex=60)
    _tick(chain.seeded[0])
    run.assert_not_called()
    assert chain.client.get("ref-a:stale_runs_lock") == b"other"
    assert chain.enqueued == [(chain.seeded[0], 300)]


def test_tick_reschedules_even_when_cleanup_fails(chain, monkeypatch):
    monkeypatch.setattr(
        cleanup_mod, "run_stale_cleanup", MagicMock(side_effect=RuntimeError("db down"))
    )
    cleanup_mod.seed_stale_runs_chain()
    _tick(chain.seeded[0])
    assert not chain.client.exists("ref-a:stale_runs_lock")
    assert chain.enqueued == [(chain.seeded[0], 300)]


def test_redis_down_keeps_the_chain_going_without_work(monkeypatch):
    broken = MagicMock()
    broken.get.side_effect = redis.ConnectionError("down")
    broken.set.side_effect = redis.ConnectionError("down")
    monkeypatch.setattr(cleanup_mod, "broker_client", lambda: broken)
    run = MagicMock()
    monkeypatch.setattr(cleanup_mod, "run_stale_cleanup", run)
    with patch.object(cleanup_mod.stale_runs_tick, "apply_async") as reschedule:
        _tick("gen-1")
    run.assert_not_called()
    assert reschedule.call_args.kwargs == {"args": ["gen-1"], "countdown": 300}


def test_no_usable_redis_url_keeps_the_chain_going(monkeypatch):
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    run = MagicMock()
    monkeypatch.setattr(cleanup_mod, "run_stale_cleanup", run)
    with patch.object(cleanup_mod.stale_runs_tick, "apply_async") as reschedule:
        _tick("gen-1")
    run.assert_not_called()
    reschedule.assert_called_once()


def test_the_seed_still_starts_a_chain_when_redis_is_down(monkeypatch):
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    with patch.object(cleanup_mod.stale_runs_tick, "delay") as delay:
        cleanup_mod.seed_stale_runs_chain()
    delay.assert_called_once()


def test_worker_ready_seeds_a_chain():
    from agentic_project_service import celery as celery_mod

    with patch.object(cleanup_mod, "seed_stale_runs_chain") as seed:
        celery_mod.seed_stale_runs_tick()
    seed.assert_called_once()


# --- the chain never drops to zero ---


def test_a_failed_seed_enqueue_restores_the_previous_generation(chain, monkeypatch):
    chain.client.set("ref-a:stale_runs_chain", "old-generation", ex=900)

    def broken_delay(generation):
        raise ConnectionError("broker down")

    monkeypatch.setattr(cleanup_mod.stale_runs_tick, "delay", broken_delay)
    cleanup_mod.seed_stale_runs_chain()  # does not raise
    assert chain.client.get("ref-a:stale_runs_chain") == b"old-generation"
    assert 0 < chain.client.ttl("ref-a:stale_runs_chain") <= 900


def test_a_failed_first_seed_enqueue_leaves_no_generation(chain, monkeypatch):
    monkeypatch.setattr(
        cleanup_mod.stale_runs_tick, "delay", MagicMock(side_effect=ConnectionError("down"))
    )
    cleanup_mod.seed_stale_runs_chain()
    assert not chain.client.exists("ref-a:stale_runs_chain")


def test_a_chain_that_cannot_re_enqueue_gives_up_its_generation(chain, monkeypatch):
    monkeypatch.setattr(cleanup_mod, "run_stale_cleanup", dict)
    monkeypatch.setattr(cleanup_mod.time, "sleep", lambda s: None)
    cleanup_mod.seed_stale_runs_chain()
    monkeypatch.setattr(
        cleanup_mod.stale_runs_tick,
        "apply_async",
        MagicMock(side_effect=ConnectionError("broker down")),
    )
    _tick(chain.seeded[0])
    assert cleanup_mod.stale_runs_tick.apply_async.call_count == 3
    assert not chain.client.exists("ref-a:stale_runs_chain")


@pytest.fixture
def fresh_check(monkeypatch):
    monkeypatch.setattr(cleanup_mod, "_last_chain_check", float("-inf"))


def test_a_run_start_re_seeds_a_dead_chain(chain, fresh_check):
    cleanup_mod.ensure_stale_runs_chain()
    assert len(chain.seeded) == 1
    assert chain.client.get("ref-a:stale_runs_chain").decode() == chain.seeded[0]


def test_a_live_chain_is_left_alone(chain, fresh_check):
    chain.client.set("ref-a:stale_runs_chain", "live", ex=900)
    cleanup_mod.ensure_stale_runs_chain()
    assert chain.seeded == []
    assert chain.client.get("ref-a:stale_runs_chain") == b"live"


def test_the_chain_check_runs_at_most_once_a_minute(chain, fresh_check, monkeypatch):
    calls = []
    real = chain.client.exists
    monkeypatch.setattr(chain.client, "exists", lambda *k: calls.append(k) or real(*k))
    chain.client.set("ref-a:stale_runs_chain", "live", ex=900)
    for _ in range(5):
        cleanup_mod.ensure_stale_runs_chain()
    assert len(calls) == 1


def test_the_chain_check_with_redis_down_does_nothing(monkeypatch, fresh_check):
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    with patch.object(cleanup_mod, "seed_stale_runs_chain") as seed:
        cleanup_mod.ensure_stale_runs_chain()  # never raises
    seed.assert_not_called()


def test_every_run_start_asks_for_the_chain_check_off_the_request_thread(monkeypatch):
    from flask import Flask

    from agentic_project_service.services import run_gate as gate_mod

    asked = []
    monkeypatch.setattr(cleanup_mod, "ensure_stale_runs_chain_soon", lambda: asked.append(1))
    monkeypatch.setattr(gate_mod, "get_setting", lambda key: 0)  # gate off
    app = Flask(__name__)

    @app.route("/run", methods=["POST"])
    @gate_mod.admit_run
    def run():
        return {"ok": True}

    with app.test_client() as c:
        assert c.post("/run").status_code == 200
    assert asked == [1]


def test_ensure_soon_spawns_at_most_one_check_per_interval(monkeypatch, fresh_check):
    started = []

    class _Thread:
        def __init__(self, target, name, daemon):
            started.append(name)

        def start(self):
            pass

    monkeypatch.setattr(cleanup_mod.threading, "Thread", _Thread)
    for _ in range(3):
        cleanup_mod.ensure_stale_runs_chain_soon()
    assert started == ["stale-runs-chain-check"]


def test_the_beat_entry_stays_for_deployments_that_run_beat():
    from agentic_project_service.celery import celery_app

    assert celery_app.conf.beat_schedule["cleanup-stale-runs"]["task"] == "cleanup_stale_runs"
