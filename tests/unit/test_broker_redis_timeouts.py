"""Redis on the request path must never hang a request.

The approvals registry and the rate limiter build their clients with short
socket timeouts and no retries, so a Redis that accepts connections but never
answers (a blackhole) costs about a second and then fails open, like a refused
connection does.
"""

import socket
import time
from unittest.mock import MagicMock

import pytest
import redis
from flask import Flask, g

from agentic_project_service.services import broker_redis
from agentic_project_service.services import rate_limit as rl
from agentic_project_service.services.run_registry import RunRegistry


@pytest.fixture
def blackhole(monkeypatch):
    """A TCP port that completes the handshake and then never answers."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    monkeypatch.setenv("CELERY_BROKER_URL", f"redis://127.0.0.1:{server.getsockname()[1]}/0")
    yield
    server.close()


def _kwargs(client):
    return client.connection_pool.connection_kwargs


def test_broker_clients_have_short_timeouts_and_no_retries(monkeypatch):
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://127.0.0.1:1/0")
    client = broker_redis.broker_client()
    kwargs = _kwargs(client)
    assert kwargs["socket_connect_timeout"] == broker_redis.REDIS_CONNECT_TIMEOUT_SECONDS == 0.5
    assert kwargs["socket_timeout"] == broker_redis.REDIS_SOCKET_TIMEOUT_SECONDS == 1.0
    assert kwargs["socket_keepalive"] is True
    assert client.get_retry().get_retries() == 0


def test_the_limiter_and_the_registry_use_the_broker_client(monkeypatch):
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://127.0.0.1:1/0")
    limiter_client = rl.RedisRateLimiter()._redis()
    registry_client = RunRegistry(start_thread=False)._redis()
    for client in (limiter_client, registry_client):
        assert _kwargs(client)["socket_timeout"] == broker_redis.REDIS_SOCKET_TIMEOUT_SECONDS
        assert client.get_retry().get_retries() == 0


def test_a_blackholed_redis_fails_the_limiter_open_within_its_timeout(blackhole):
    limiter = rl.RedisRateLimiter()
    started = time.monotonic()
    assert limiter.try_acquire("workflow-exec:service", 1) is None
    assert time.monotonic() - started < 1.6


def test_a_blackholed_redis_does_not_hang_a_workflow_execute(blackhole, monkeypatch):
    monkeypatch.setattr(rl, "external_limiter", rl.RedisRateLimiter())
    app = Flask(__name__)

    @app.before_request
    def _who():
        g.user_id = "user-1"

    @app.route("/exec", methods=["POST"])
    @rl.rate_limit_executions
    def exec_():
        return {"ok": True}

    started = time.monotonic()
    with app.test_client() as c:
        assert c.post("/exec").status_code == 200
    assert time.monotonic() - started < 1.6


def test_a_redis_timeout_fails_the_limiter_open():
    slow = MagicMock()
    slow.register_script.return_value = MagicMock(side_effect=redis.TimeoutError("timed out"))
    assert rl.RedisRateLimiter(slow).try_acquire("workflow-exec:service", 1) is None
