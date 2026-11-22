"""MAX_CONCURRENT_RUNS: a leased Redis semaphore over in-flight run requests.

Runs against fakeredis with Lua. Set ``TEST_REDIS_URL`` (a database this suite
may flush, e.g. ``redis://localhost:6379/15``) to run every test here against a
real Redis as well: the lease scores come from the Redis clock.
"""

import os
import time
from unittest.mock import MagicMock

import fakeredis
import pytest
import redis

from agentic_project_service.services import broker_redis
from agentic_project_service.services import run_gate as gate_mod
from agentic_project_service.services.run_gate import RunGate
from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY

_REAL_REDIS_URL = os.getenv("TEST_REDIS_URL")


@pytest.fixture(autouse=True)
def _ref(monkeypatch):
    monkeypatch.setenv("PROJECT_REF", "ref-a")


@pytest.fixture(params=["fakeredis", "redis"])
def client(request):
    if request.param == "fakeredis":
        yield fakeredis.FakeStrictRedis()
        return
    if not _REAL_REDIS_URL:
        pytest.skip("TEST_REDIS_URL not set")
    real = redis.from_url(_REAL_REDIS_URL)
    real.flushdb()
    yield real
    real.flushdb()


@pytest.fixture
def gate(client):
    return RunGate(redis_client=client, renew_in_background=False)


def test_setting_is_registered_and_off_by_default():
    d = SETTINGS_REGISTRY["MAX_CONCURRENT_RUNS"]
    assert (d.type, d.default, d.category, d.min, d.max) == ("int", 0, "agents", 0, 10000)


def test_zero_means_no_gate_and_no_redis_call():
    client = MagicMock()
    a = RunGate(redis_client=client, renew_in_background=False).try_acquire(0)
    assert a.admitted and a.token is None
    client.register_script.assert_not_called()


def test_admits_up_to_the_limit_then_reports_active(gate):
    first, second, third = gate.try_acquire(2), gate.try_acquire(2), gate.try_acquire(2)
    assert first.admitted and second.admitted
    assert first.token and second.token and first.token != second.token
    assert not third.admitted and third.active_runs == 2


def test_release_frees_a_slot(gate):
    held = gate.try_acquire(1)
    gate.release(held.token)
    assert gate.try_acquire(1).admitted


def test_two_processes_share_the_limit(client):
    one = RunGate(redis_client=client, renew_in_background=False)
    two = RunGate(redis_client=client, renew_in_background=False)
    assert one.try_acquire(1).admitted
    refused = two.try_acquire(1)
    assert not refused.admitted and refused.active_runs == 1


def test_an_expired_lease_frees_its_slot(gate, monkeypatch):
    monkeypatch.setattr(gate_mod, "LEASE_SECONDS", 0.2)
    gate.try_acquire(1)
    time.sleep(0.3)
    assert gate.try_acquire(1).admitted


def test_renew_all_keeps_held_leases_alive(gate, monkeypatch):
    monkeypatch.setattr(gate_mod, "LEASE_SECONDS", 0.4)
    gate.try_acquire(1)
    time.sleep(0.25)
    assert gate.renew_all() == 1
    time.sleep(0.25)
    assert not gate.try_acquire(1).admitted


def test_a_released_slot_is_not_renewed(gate):
    held = gate.try_acquire(1)
    gate.release(held.token)
    assert gate.renew_all() == 0


def test_a_lost_lease_is_dropped_from_renewal(gate, client, monkeypatch, caplog):
    gate.try_acquire(1)
    client.delete(gate.key())
    with caplog.at_level("WARNING", logger=gate_mod.logger.name):
        assert gate.renew_all() == 0
    assert "expired before renewal" in caplog.text
    assert gate.renew_all() == 0
    assert not gate._held


def test_active_runs_counts_live_leases(gate):
    gate.try_acquire(5)
    gate.try_acquire(5)
    assert gate.active_runs() == 2


def test_active_runs_is_zero_on_an_empty_project(gate):
    assert gate.active_runs() == 0


def test_key_carries_the_project_ref_and_a_ttl(gate, client):
    gate.try_acquire(3)
    assert gate.key() == "runs:active:ref-a"
    assert client.exists("runs:active:ref-a")
    assert 0 < client.ttl("runs:active:ref-a") <= 2 * gate_mod.LEASE_SECONDS


def test_the_project_ref_is_read_at_call_time(gate, client, monkeypatch):
    monkeypatch.setenv("PROJECT_REF", "ref-b")
    gate.try_acquire(3)
    assert client.exists("runs:active:ref-b")
    assert not client.exists("runs:active:ref-a")


def test_redis_down_fails_open():
    client = MagicMock()
    client.register_script.side_effect = redis.ConnectionError("down")
    g = RunGate(redis_client=client, health_client=client, renew_in_background=False)
    a = g.try_acquire(1)
    assert a.admitted and a.token is None
    assert g.active_runs() is None
    g.release(None)  # nothing to release
    assert g.renew_all() == 0


def test_a_failing_release_does_not_raise():
    client = MagicMock()
    client.zrem.side_effect = redis.ConnectionError("down")
    g = RunGate(redis_client=client, renew_in_background=False)
    g._held["tok"] = (time.monotonic(), gate_mod.MAX_HOLD_SECONDS)
    g.release("tok")
    assert not g._held


def test_an_unusable_broker_url_warns_once_and_fails_open(monkeypatch, caplog):
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    g = RunGate(renew_in_background=False)
    with caplog.at_level("WARNING", logger=gate_mod.logger.name):
        assert all(g.try_acquire(1).admitted for _ in range(3))
        assert g.active_runs() is None
    warnings = [r for r in caplog.records if r.name == gate_mod.logger.name]
    assert len(warnings) == 1


def test_clients_use_the_bounded_broker_client(monkeypatch):
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://127.0.0.1:1/0")
    g = RunGate(renew_in_background=False)
    request_kwargs = g._redis().connection_pool.connection_kwargs
    health_kwargs = g._health_redis().connection_pool.connection_kwargs
    assert request_kwargs["socket_timeout"] == broker_redis.REDIS_SOCKET_TIMEOUT_SECONDS
    assert health_kwargs["socket_timeout"] == gate_mod.HEALTH_SOCKET_TIMEOUT_SECONDS == 0.25
    assert health_kwargs["socket_connect_timeout"] == 0.25
    assert g._redis().get_retry().get_retries() == 0
    assert g._health_redis().get_retry().get_retries() == 0


def test_the_renewer_thread_starts_on_the_first_admission(client, monkeypatch):
    g = RunGate(redis_client=client, renew_in_background=True)
    assert g._renewer is None
    g.try_acquire(0)
    assert g._renewer is None  # gate off: no thread
    held = g.try_acquire(2)
    assert g._renewer is not None and g._renewer.is_alive() and g._renewer.daemon
    first = g._renewer
    other = g.try_acquire(2)
    assert g._renewer is first
    # Hold nothing once the test ends: the daemon thread outlives it, and a
    # held token would be "renewed" against a database later tests flush.
    g.release(held.token)
    g.release(other.token)


def test_a_slot_held_past_its_ceiling_is_no_longer_renewed(gate, client, caplog):
    """A slot whose release never ran (a leak inside a live process) stops
    being renewed after its hold ceiling, so it frees itself one lease later."""
    held = gate.try_acquire(1, max_hold_seconds=0)
    kept = gate.try_acquire(2, max_hold_seconds=3600)
    with caplog.at_level("WARNING", logger=gate_mod.logger.name):
        assert gate.renew_all() == 1
    assert "held longer than" in caplog.text
    assert list(gate._held) == [kept.token]
    # The lease is left to expire, not removed: the run may still be going.
    assert client.zscore(gate.key(), held.token) is not None


def test_the_default_hold_ceiling_is_the_default_stale_run_cutoff():
    """Not a day: a leaked slot must not lock a small project out for 24 hours."""
    assert gate_mod.MAX_HOLD_SECONDS == SETTINGS_REGISTRY["RUN_STALE_AFTER_SECONDS"].default
    assert gate_mod.MAX_HOLD_SECONDS <= 1800


def test_admit_run_holds_a_slot_no_longer_than_the_stale_run_cutoff(gate, monkeypatch):
    from flask import Flask

    settings = {"MAX_CONCURRENT_RUNS": 1, "RUN_STALE_AFTER_SECONDS": 900}
    monkeypatch.setattr(gate_mod, "get_setting", settings.__getitem__)
    monkeypatch.setattr(gate_mod, "default_gate", gate)
    seen = {}
    app = Flask(__name__)

    @app.route("/run", methods=["POST"])
    @gate_mod.admit_run
    def run():
        ((_, ceiling),) = gate._held.values()
        seen["ceiling"] = ceiling
        return {"ok": True}

    with app.test_client() as c:
        assert c.post("/run").status_code == 200
    assert seen == {"ceiling": 900}


def test_an_unreadable_cutoff_falls_back_to_the_default_ceiling(gate, monkeypatch):
    from flask import Flask

    monkeypatch.setattr(gate_mod, "get_setting", {"MAX_CONCURRENT_RUNS": 1}.__getitem__)
    monkeypatch.setattr(gate_mod, "default_gate", gate)
    seen = {}
    app = Flask(__name__)

    @app.route("/run", methods=["POST"])
    @gate_mod.admit_run
    def run():
        ((_, ceiling),) = gate._held.values()
        seen["ceiling"] = ceiling
        return {"ok": True}

    with app.test_client() as c:
        assert c.post("/run").status_code == 200
    assert seen == {"ceiling": gate_mod.MAX_HOLD_SECONDS}
