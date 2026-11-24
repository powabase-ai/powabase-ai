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


def test_the_collector_describes_itself_without_reading_redis():
    """describe() must name the gauge, or prometheus_client cannot see that a
    second create_app() registers it twice, and the scrape then carries the
    gauge twice and is rejected."""
    from prometheus_client import CollectorRegistry

    gate = MagicMock()
    registry = CollectorRegistry()
    registry.register(ActiveRunsCollector(gate))
    with pytest.raises(ValueError):
        registry.register(ActiveRunsCollector(gate))
    gate.active_runs.assert_not_called()
    assert [m.name for m in ActiveRunsCollector(gate).describe()] == ["project_api_active_runs"]


def test_two_apps_in_one_process_expose_the_gauge_once(monkeypatch):
    from prometheus_client import REGISTRY, generate_latest

    monkeypatch.setattr(gate_mod, "default_gate", _broken_gate())
    main.create_app(testing=True)
    main.create_app(testing=True)
    exposition = generate_latest(REGISTRY).decode()
    assert exposition.count("# TYPE project_api_active_runs gauge") == 1


def test_plain_health_has_no_run_fields():
    assert main.health_body(False) == {"status": "healthy", "service": "project-service"}


@pytest.fixture
def usable_broker(monkeypatch):
    """Every Redis-backed component can build its client."""
    from agentic_project_service.services import rate_limit, run_registry

    monkeypatch.setattr(
        run_registry,
        "_default",
        run_registry.RunRegistry(redis_client=fakeredis.FakeStrictRedis(), start_thread=False),
    )
    monkeypatch.setattr(
        rate_limit, "external_limiter", rate_limit.RedisRateLimiter(fakeredis.FakeStrictRedis())
    )


def test_health_with_runs(monkeypatch, usable_broker):
    _gate_with(3, monkeypatch)
    monkeypatch.setattr(gate_mod, "get_setting", lambda key: {"MAX_CONCURRENT_RUNS": 8}[key])
    assert main.health_body(True) == {
        "status": "healthy",
        "service": "project-service",
        "active_runs": 3,
        "max_concurrent_runs": 8,
        "coordination_disabled": [],
    }


def test_a_bad_broker_url_shows_on_health(monkeypatch):
    """An unusable CELERY_BROKER_URL turns the gate, the limiter and remote
    approvals off for the life of the process; /health?runs=1 says so."""
    from agentic_project_service.services import rate_limit, run_registry

    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    monkeypatch.setattr(gate_mod, "default_gate", RunGate(renew_in_background=False))
    monkeypatch.setattr(run_registry, "_default", run_registry.RunRegistry(start_thread=False))
    monkeypatch.setattr(rate_limit, "external_limiter", rate_limit.RedisRateLimiter())
    monkeypatch.setattr(gate_mod, "get_setting", lambda key: 0)
    body = main.health_body(True)
    assert body["coordination_disabled"] == ["approvals", "rate_limit", "run_gate"]


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
