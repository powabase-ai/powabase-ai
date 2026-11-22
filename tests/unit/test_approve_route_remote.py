"""POST /api/agents/runs/<id>/approve when the run lives in another process."""

import time
from unittest.mock import patch

import fakeredis
from agentic.execution.context import ExecutionContext
from flask import Flask

from agentic_project_service.routes import agents as agents_route
from agentic_project_service.services.run_registry import RemoteRun, RunRegistry

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
        patch.object(agents_route, "deliver_approval", return_value=True) as deliver,
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
        patch.object(agents_route, "deliver_approval", return_value=True) as deliver,
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
        patch.object(agents_route, "deliver_approval", return_value=False),
        _as_user(),
    ):
        assert _post().status_code == 404


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


def test_end_to_end_across_two_registries():
    """The route on process B resumes a run that waits on process A."""
    server = fakeredis.FakeServer()
    a = RunRegistry(fakeredis.FakeStrictRedis(server=server), instance="pod-a")
    b = RunRegistry(fakeredis.FakeStrictRedis(server=server), instance="pod-b", start_thread=False)
    ctx = ExecutionContext(execution_id="run_1")
    ctx.reset_approval()
    a.register("run_1", ctx, owner_user_id=USER)
    try:
        assert _wait_until(lambda: b.lookup_remote("run_1") is not None)
        with (
            patch.object(agents_route, "get_active_run_context", b.context),
            patch.object(agents_route, "get_active_run_owner", b.owner),
            patch.object(agents_route, "lookup_remote_run", b.lookup_remote),
            patch.object(agents_route, "deliver_approval", b.deliver_remote),
            _as_user(),
        ):
            resp = _post(body={"approved": False, "reason": "no"})
        assert resp.status_code == 200
        assert ctx.get_approval_event().is_set()
        assert ctx.get_approval_decision() == {"approved": False, "reason": "no"}
    finally:
        a.close()
        b.close()


def _wait_until(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()
