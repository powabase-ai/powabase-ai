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


def test_an_unusable_broker_url_warns_once_and_fails_open(monkeypatch, caplog):
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    limiter = rl.RedisRateLimiter()
    with caplog.at_level("WARNING", logger=rl.logger.name):
        assert all(limiter.try_acquire("workflow-exec:service", 1) is None for _ in range(3))
    warnings = [r for r in caplog.records if r.name == rl.logger.name]
    assert len(warnings) == 1


def test_one_client_per_process_for_each_url_and_timeouts(monkeypatch):
    """The broker Redis is shared by every project: one connection pool per
    process, not one per component and per sweep."""
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://127.0.0.1:1/0")
    first = broker_redis.broker_client()
    assert broker_redis.broker_client() is first
    assert broker_redis.broker_client(socket_timeout=0.25, connect_timeout=0.25) is not first
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://127.0.0.1:1/1")
    assert broker_redis.broker_client() is not first


def test_the_registry_gate_limiter_and_sweep_share_that_client(monkeypatch):
    from agentic_project_service.services.run_gate import RunGate

    monkeypatch.setenv("CELERY_BROKER_URL", "redis://127.0.0.1:1/0")
    shared = broker_redis.broker_client()
    assert RunRegistry(start_thread=False)._redis() is shared
    assert RunGate(renew_in_background=False)._redis() is shared
    assert rl.RedisRateLimiter()._redis() is shared


def test_an_unusable_url_is_not_cached(monkeypatch):
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    for _ in range(2):
        with pytest.raises(ValueError):
            broker_redis.broker_client()
