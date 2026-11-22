"""The workflow execution limit is shared by every process of the project.

End users (a JWT with a user id) get 20 per minute each; the service role, which
is what the execute routes accept today, gets one per-project bucket whose size
is the WORKFLOW_EXECUTIONS_PER_MINUTE setting (default 120).

Runs against fakeredis with Lua. Set ``TEST_REDIS_URL`` (a database this suite
may flush, e.g. ``redis://localhost:6379/15``) to run the Redis-backed tests
against a real Redis as well.
"""

import os
from unittest.mock import MagicMock

import fakeredis
import pytest
import redis
from flask import Flask, g

from agentic_project_service.services import rate_limit as rl

_REAL_REDIS_URL = os.getenv("TEST_REDIS_URL")


@pytest.fixture(autouse=True)
def _ref(monkeypatch):
    monkeypatch.setenv("PROJECT_REF", "ref-a")


@pytest.fixture(params=["fakeredis", "redis"])
def connect(request):
    """A factory of clients that all talk to the same Redis."""
    if request.param == "fakeredis":
        server = fakeredis.FakeServer()
        yield lambda: fakeredis.FakeStrictRedis(server=server)
        return
    if not _REAL_REDIS_URL:
        pytest.skip("TEST_REDIS_URL not set")
    real = redis.from_url(_REAL_REDIS_URL)
    real.flushdb()
    yield lambda: redis.from_url(_REAL_REDIS_URL)
    real.flushdb()


def _app(user="user-1"):
    app = Flask(__name__)

    @app.before_request
    def _who():
        g.user_id = user

    @app.route("/exec", methods=["POST"])
    @rl.rate_limit_executions
    def exec_():
        return {"ok": True}

    return app


def test_twenty_then_429_with_retry_after(connect, monkeypatch):
    monkeypatch.setattr(rl, "external_limiter", rl.RedisRateLimiter(connect()))
    with _app().test_client() as c:
        for _ in range(20):
            assert c.post("/exec").status_code == 200
        resp = c.post("/exec")
    assert resp.status_code == 429
    assert 1 <= int(resp.headers["Retry-After"]) <= 60
    assert resp.get_json() == {"error": "Rate limit exceeded. Max 20 executions per minute."}


def test_the_window_is_shared_across_processes(connect, monkeypatch):
    """Executions another process counted in Redis count against this one."""
    other_process = rl.RedisRateLimiter(connect())
    for _ in range(15):
        assert other_process.try_acquire("workflow-exec:user-1", 20) is None
    monkeypatch.setattr(rl, "external_limiter", rl.RedisRateLimiter(connect()))
    with _app().test_client() as c:
        for _ in range(5):
            assert c.post("/exec").status_code == 200
        assert c.post("/exec").status_code == 429


def test_users_are_independent(connect, monkeypatch):
    monkeypatch.setattr(rl, "external_limiter", rl.RedisRateLimiter(connect()))
    with _app("user-1").test_client() as c:
        for _ in range(20):
            c.post("/exec")
        assert c.post("/exec").status_code == 429
    with _app("user-2").test_client() as c:
        assert c.post("/exec").status_code == 200


def test_the_key_carries_user_and_project_and_expires(connect, monkeypatch):
    monkeypatch.setattr(rl, "external_limiter", rl.RedisRateLimiter(connect()))
    with _app("user-1").test_client() as c:
        assert c.post("/exec").status_code == 200
    ttl = connect().ttl("ratelimit:workflow-exec:user-1:ref-a")
    assert 0 < ttl <= 120


def test_projects_are_independent(connect, monkeypatch):
    monkeypatch.setattr(rl, "external_limiter", rl.RedisRateLimiter(connect()))
    with _app().test_client() as c:
        for _ in range(20):
            c.post("/exec")
        assert c.post("/exec").status_code == 429
        monkeypatch.setenv("PROJECT_REF", "ref-b")
        assert c.post("/exec").status_code == 200


def test_redis_down_fails_open(monkeypatch):
    broken = MagicMock()
    broken.register_script.side_effect = redis.ConnectionError("down")
    monkeypatch.setattr(rl, "external_limiter", rl.RedisRateLimiter(broken))
    with _app().test_client() as c:
        assert all(c.post("/exec").status_code == 200 for _ in range(25))


def test_an_unusable_broker_url_fails_open(monkeypatch):
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    monkeypatch.setattr(rl, "external_limiter", rl.RedisRateLimiter())
    with _app().test_client() as c:
        assert all(c.post("/exec").status_code == 200 for _ in range(25))


# ---------------------------------------------------------------------------
# The real routes: both execute endpoints take the service role key, whose
# token carries no end-user id, so they share one per-project service bucket.
# ---------------------------------------------------------------------------


def _workflows_app():
    from agentic_project_service.routes import workflows as workflows_route

    app = Flask(__name__)
    app.register_blueprint(workflows_route.workflows_bp)
    return app, workflows_route


def _service_role_calls(monkeypatch, limiter, n, paths):
    """POST ``n`` times, cycling over ``paths``; the workflow itself is never found."""
    from unittest.mock import patch

    app, workflows_route = _workflows_app()
    monkeypatch.setattr(rl, "external_limiter", limiter)
    codes = []
    with (
        patch.object(workflows_route, "_workflow_pre_check"),
        patch.object(workflows_route, "build_workflow_from_db", return_value=None),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"role": "service_role", "is_service_role": True},
        ),
        app.test_client() as c,
    ):
        for i in range(n):
            resp = c.post(paths[i % len(paths)], json={}, headers={"Authorization": "Bearer x"})
            codes.append((resp.status_code, resp.headers.get("Retry-After")))
    return codes


_BOTH = ["/api/workflows/wf-1/execute", "/api/workflows/wf-1/execute/stream"]


def test_the_service_role_gets_its_own_project_bucket(connect, monkeypatch):
    limiter = rl.RedisRateLimiter(connect())
    codes = _service_role_calls(
        monkeypatch, limiter, rl.DEFAULT_SERVICE_EXECUTIONS_PER_MINUTE + 1, _BOTH
    )
    assert all(code == 404 for code, _ in codes[:-1])  # past the limiter, workflow missing
    assert codes[-1][0] == 429
    assert 1 <= int(codes[-1][1]) <= 60
    assert connect().exists("ratelimit:workflow-exec:service:ref-a")
    assert not connect().exists("ratelimit:workflow-exec:anonymous:ref-a")


def test_both_execute_routes_share_the_service_bucket(connect, monkeypatch):
    monkeypatch.setattr(rl, "get_setting", lambda key: 4)
    limiter = rl.RedisRateLimiter(connect())
    assert [c for c, _ in _service_role_calls(monkeypatch, limiter, 2, _BOTH[:1])] == [404, 404]
    codes = [c for c, _ in _service_role_calls(monkeypatch, limiter, 3, _BOTH[1:])]
    assert codes == [404, 404, 429]


def test_the_configured_service_limit_is_honoured(connect, monkeypatch):
    seen = []

    def fake_get_setting(key):
        seen.append(key)
        return 3

    monkeypatch.setattr(rl, "get_setting", fake_get_setting)
    limiter = rl.RedisRateLimiter(connect())
    codes = [c for c, _ in _service_role_calls(monkeypatch, limiter, 4, _BOTH)]
    assert codes == [404, 404, 404, 429]
    assert set(seen) == {"WORKFLOW_EXECUTIONS_PER_MINUTE"}


def test_the_service_429_names_the_configured_limit(connect, monkeypatch):
    monkeypatch.setattr(rl, "get_setting", lambda key: 1)
    monkeypatch.setattr(rl, "external_limiter", rl.RedisRateLimiter(connect()))
    app, workflows_route = _workflows_app()
    from unittest.mock import patch

    with (
        patch.object(workflows_route, "_workflow_pre_check"),
        patch.object(workflows_route, "build_workflow_from_db", return_value=None),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"role": "service_role", "is_service_role": True},
        ),
        app.test_client() as c,
    ):
        c.post(_BOTH[0], json={}, headers={"Authorization": "Bearer x"})
        resp = c.post(_BOTH[0], json={}, headers={"Authorization": "Bearer x"})
    assert resp.status_code == 429
    assert resp.get_json() == {"error": "Rate limit exceeded. Max 1 executions per minute."}
