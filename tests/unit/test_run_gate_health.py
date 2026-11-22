"""Readiness detail and the active-runs gauge."""

from unittest.mock import MagicMock

import fakeredis
import pytest
import redis

from agentic_project_service import main
from agentic_project_service.services import run_gate as gate_mod
from agentic_project_service.services.run_gate import ActiveRunsCollector, RunGate


def _gate_with(n, monkeypatch):
    monkeypatch.setenv("PROJECT_REF", "ref-a")
    g = RunGate(redis_client=fakeredis.FakeStrictRedis(), renew_in_background=False)
    for _ in range(n):
        g.try_acquire(100)
    monkeypatch.setattr(gate_mod, "default_gate", g)
    return g


def _broken_gate():
    broken = MagicMock()
    broken.register_script.side_effect = redis.ConnectionError("down")
    return RunGate(redis_client=broken, renew_in_background=False)


def test_plain_health_has_no_run_fields():
    assert main.health_body(False) == {"status": "healthy", "service": "project-service"}


def test_health_with_runs(monkeypatch):
    _gate_with(3, monkeypatch)
    monkeypatch.setattr(gate_mod, "get_setting", lambda key: {"MAX_CONCURRENT_RUNS": 8}[key])
    assert main.health_body(True) == {
        "status": "healthy",
        "service": "project-service",
        "active_runs": 3,
        "max_concurrent_runs": 8,
    }


def test_health_with_runs_is_null_on_errors(monkeypatch):
    monkeypatch.setattr(gate_mod, "default_gate", _broken_gate())

    def unreadable(key):
        raise RuntimeError("db down")

    monkeypatch.setattr(gate_mod, "get_setting", unreadable)
    body = main.health_body(True)
    assert body["active_runs"] is None and body["max_concurrent_runs"] is None


@pytest.fixture
def app():
    return main.create_app(testing=True)


def test_the_probe_touches_neither_redis_nor_settings(app, monkeypatch):
    gate = MagicMock()
    settings = MagicMock()
    monkeypatch.setattr(gate_mod, "default_gate", gate)
    monkeypatch.setattr(gate_mod, "get_setting", settings)
    with app.test_client() as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "healthy", "service": "project-service"}
    gate.active_runs.assert_not_called()
    settings.assert_not_called()


def test_health_route_with_runs(app, monkeypatch):
    _gate_with(2, monkeypatch)
    monkeypatch.setattr(gate_mod, "get_setting", lambda key: 4)
    with app.test_client() as c:
        resp = c.get("/health?runs=1")
    assert resp.status_code == 200
    assert resp.get_json()["active_runs"] == 2
    assert resp.get_json()["max_concurrent_runs"] == 4


def test_health_route_with_runs_is_200_when_redis_is_down(app, monkeypatch):
    monkeypatch.setattr(gate_mod, "default_gate", _broken_gate())
    monkeypatch.setattr(gate_mod, "get_setting", lambda key: 0)
    with app.test_client() as c:
        resp = c.get("/health?runs=1")
    assert resp.status_code == 200
    assert resp.get_json()["active_runs"] is None


def test_collector_reports_the_project_wide_count(monkeypatch):
    g = _gate_with(2, monkeypatch)
    families = list(ActiveRunsCollector(g).collect())
    assert [f.name for f in families] == ["project_api_active_runs"]
    assert [s.value for s in families[0].samples] == [2]


def test_collector_follows_the_default_gate(monkeypatch):
    _gate_with(1, monkeypatch)
    families = list(ActiveRunsCollector().collect())
    assert [s.value for s in families[0].samples] == [1]


def test_collector_omits_the_sample_when_unknown():
    families = list(ActiveRunsCollector(_broken_gate()).collect())
    assert families[0].samples == []


def test_metrics_exposes_the_gauge(app, monkeypatch):
    _gate_with(3, monkeypatch)
    with app.test_client() as c:
        resp = c.get("/metrics")
    assert resp.status_code == 200
    assert "project_api_active_runs 3.0" in resp.get_data(as_text=True)


def test_a_second_app_does_not_register_the_collector_twice():
    main.create_app(testing=True)
    main.create_app(testing=True)  # raises if the second registration is not tolerated


def test_registering_the_collector_does_not_read_redis():
    """The default registry calls collect() at register time unless the
    collector describes itself; app startup must not wait on Redis."""
    from prometheus_client import CollectorRegistry

    gate = MagicMock()
    CollectorRegistry(auto_describe=True).register(ActiveRunsCollector(gate))
    gate.active_runs.assert_not_called()
