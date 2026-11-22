"""POST /api/agents/runs/<id>/approve when the run lives in another process."""

import os
import time
from unittest.mock import MagicMock, patch

import fakeredis
import pytest
import redis
from agentic.execution.context import ExecutionContext
from flask import Flask

from agentic_project_service.routes import agents as agents_route
from agentic_project_service.services import run_registry
from agentic_project_service.services.run_registry import (
    Delivery,
    RegistryUnavailable,
    RemoteRun,
    RunRegistry,
)

_REAL_REDIS_URL = os.getenv("TEST_REDIS_URL")

USER = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"
HEADERS = {"Authorization": "Bearer fake"}


def _app():
    app = Flask(__name__)
    app.register_blueprint(agents_route.agents_bp)
    return app


def _as_user(uid=USER):
    return patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": uid, "role": "authenticated", "aud": "authenticated"},
    )


def _as_service():
    return patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"role": "service_role", "is_service_role": True},
    )


def _post(run_id="run_1", body=None):
    with _app().test_client() as c:
        return c.post(
            f"/api/agents/runs/{run_id}/approve",
            json=body if body is not None else {"approved": True},
            headers=HEADERS,
        )


def _remote(owner):
    return patch.object(agents_route, "lookup_remote_run", return_value=RemoteRun("pod-a", owner))


def _not_local():
    return patch.object(agents_route, "get_active_run_context", return_value=None)


def test_forwards_to_the_owning_process():
    with (
        _not_local(),
        _remote(USER),
        patch.object(agents_route, "deliver_approval", return_value=Delivery.RESUMED) as deliver,
        _as_user(),
    ):
        resp = _post()
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "resumed"}
    deliver.assert_called_once_with("run_1", RemoteRun("pod-a", USER), {"approved": True})


def test_unknown_everywhere_is_404():
    with (
        _not_local(),
        patch.object(agents_route, "lookup_remote_run", return_value=None),
        patch.object(agents_route, "deliver_approval") as deliver,
        _as_user(),
    ):
        assert _post().status_code == 404
    deliver.assert_not_called()


def test_another_users_run_is_404_and_not_forwarded():
    with (
        _not_local(),
        _remote(OTHER),
        patch.object(agents_route, "deliver_approval") as deliver,
        _as_user(),
    ):
        assert _post().status_code == 404
    deliver.assert_not_called()


def test_a_service_started_run_needs_the_service_role():
    with (
        _not_local(),
        _remote(None),
        patch.object(agents_route, "deliver_approval", return_value=Delivery.RESUMED) as deliver,
    ):
        with _as_user():
            assert _post().status_code == 404
        with _as_service():
            assert _post().status_code == 200
    deliver.assert_called_once()


def test_owner_process_gone_is_404():
    with (
        _not_local(),
        _remote(USER),
        patch.object(agents_route, "deliver_approval", return_value=Delivery.GONE),
        _as_user(),
    ):
        assert _post().status_code == 404


UNCONFIRMED_BODY = {
    "status": "unconfirmed",
    "message": "The decision was sent but the run has not confirmed it yet; "
    "check the run before retrying.",
}


def test_an_unconfirmed_delivery_is_202_not_404():
    """The run may have resumed: a 404 would tell a client it is gone."""
    with (
        _not_local(),
        _remote(USER),
        patch.object(agents_route, "deliver_approval", return_value=Delivery.UNCONFIRMED),
        _as_user(),
    ):
        resp = _post()
    assert resp.status_code == 202
    assert resp.get_json() == UNCONFIRMED_BODY


def _assert_unavailable(resp):
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "2"
    assert resp.get_json() == {"error": "approvals_unavailable"}


def test_a_redis_error_on_the_owner_lookup_is_503_with_retry_after():
    with (
        _not_local(),
        patch.object(
            agents_route, "lookup_remote_run", side_effect=RegistryUnavailable("hgetall failed")
        ),
        patch.object(agents_route, "deliver_approval") as deliver,
        _as_user(),
    ):
        _assert_unavailable(_post())
    deliver.assert_not_called()


def test_a_redis_error_on_the_publish_is_503_with_retry_after():
    with (
        _not_local(),
        _remote(USER),
        patch.object(agents_route, "deliver_approval", return_value=Delivery.UNAVAILABLE),
        _as_user(),
    ):
        _assert_unavailable(_post())


def test_the_ownership_check_still_comes_before_the_publish():
    """Another user's run is 404 even though the publish would fail."""
    with (
        _not_local(),
        _remote(OTHER),
        patch.object(agents_route, "deliver_approval", return_value=Delivery.UNAVAILABLE),
        _as_user(),
    ):
        assert _post().status_code == 404


def test_a_broken_redis_end_to_end_is_503_not_404():
    """The registry itself, on a Redis that refuses every command."""
    broken = MagicMock()
    broken.hgetall.side_effect = redis.ConnectionError("down")
    registry = RunRegistry(redis_client=broken, instance="pod-b", start_thread=False)
    with (
        patch.object(agents_route, "get_active_run_context", registry.context),
        patch.object(agents_route, "lookup_remote_run", registry.lookup_remote),
        patch.object(agents_route, "deliver_approval", registry.deliver_remote),
        _as_user(),
    ):
        _assert_unavailable(_post())

    found = MagicMock()
    found.hgetall.return_value = {b"instance": b"pod-a", b"owner": USER.encode()}
    found.publish.side_effect = redis.ConnectionError("down")
    registry = RunRegistry(redis_client=found, instance="pod-b", start_thread=False)
    with (
        patch.object(agents_route, "get_active_run_context", registry.context),
        patch.object(agents_route, "lookup_remote_run", registry.lookup_remote),
        patch.object(agents_route, "deliver_approval", registry.deliver_remote),
        _as_user(),
    ):
        _assert_unavailable(_post())


def test_a_gone_owner_keeps_the_plain_404_body():
    with (
        _not_local(),
        _remote(USER),
        patch.object(agents_route, "deliver_approval", return_value=Delivery.GONE),
        _as_user(),
    ):
        resp = _post()
    assert resp.get_json() == {"error": "Run not found or not waiting for approval"}


def test_an_invalid_body_is_400_before_any_lookup():
    with (
        _not_local(),
        patch.object(agents_route, "lookup_remote_run") as lookup,
        _as_user(),
    ):
        assert _post(body={"approved": "yes"}).status_code == 400
    lookup.assert_not_called()


def test_a_local_context_still_wins():
    ctx = ExecutionContext(execution_id="run_1")
    ctx.reset_approval()
    with (
        patch.object(agents_route, "get_active_run_context", return_value=ctx),
        patch.object(agents_route, "get_active_run_owner", return_value=USER),
        patch.object(agents_route, "lookup_remote_run") as lookup,
        _as_user(),
    ):
        assert _post().status_code == 200
    lookup.assert_not_called()
    assert ctx.get_approval_decision() == {"approved": True}


@pytest.fixture(params=["fakeredis", "redis"])
def connect(request):
    """A factory of clients that all talk to the same Redis (real one with TEST_REDIS_URL)."""
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


@pytest.fixture
def two_processes(connect):
    """Process A runs the listener and holds the run; the route runs on process B."""
    a = RunRegistry(connect(), instance="pod-a:1:aaaa")
    b = RunRegistry(connect(), instance="pod-b:2:bbbb", start_thread=False)
    ctx = ExecutionContext(execution_id="run_1")
    ctx.reset_approval()
    a.register("run_1", ctx, owner_user_id=USER)
    assert _wait_until(lambda: b.lookup_remote("run_1") is not None)
    with (
        patch.object(agents_route, "get_active_run_context", b.context),
        patch.object(agents_route, "get_active_run_owner", b.owner),
        patch.object(agents_route, "lookup_remote_run", b.lookup_remote),
        patch.object(agents_route, "deliver_approval", b.deliver_remote),
        _as_user(),
    ):
        yield a, ctx
    a.close()
    b.close()


def test_end_to_end_across_two_registries(two_processes):
    """The route on process B resumes a run that waits on process A."""
    _, ctx = two_processes
    resp = _post(body={"approved": False, "reason": "no"})
    assert resp.status_code == 200
    assert ctx.get_approval_event().is_set()
    assert ctx.get_approval_decision() == {"approved": False, "reason": "no"}


def test_a_dead_owner_listener_is_202_unconfirmed_until_it_reconnects(two_processes, monkeypatch):
    for name, value in (
        ("HEARTBEAT_SECONDS", 0.2),
        ("PONG_TIMEOUT_SECONDS", 0.3),
        ("ACK_TIMEOUT_SECONDS", 0.5),
        ("RESUBSCRIBE_BACKOFF_SECONDS", 0.05),
    ):
        monkeypatch.setattr(run_registry, name, value)
    a, ctx = two_processes
    deaf = _Deaf(a._pubsub)
    a._pubsub = deaf  # the connection died without Redis or the listener noticing
    time.sleep(0.15)
    resp = _post()
    assert resp.status_code == 202
    assert resp.get_json() == UNCONFIRMED_BODY
    assert ctx.get_approval_decision() is None
    assert _wait_until(lambda: a._pubsub not in (None, deaf) and a._subscribed.is_set())
    assert _post().status_code == 200
    assert ctx.get_approval_decision() == {"approved": True}


class _Deaf:
    """A subscription whose connection died silently: nothing arrives, no PONG."""

    def __init__(self, inner):
        self.inner = inner

    def get_message(self, timeout=0.0):
        time.sleep(min(timeout or 0.0, 0.05))

    def ping(self):
        pass

    def close(self):
        self.inner.close()


def _wait_until(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()
