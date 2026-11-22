"""The workflow execution limit is shared by every process of the project.

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
