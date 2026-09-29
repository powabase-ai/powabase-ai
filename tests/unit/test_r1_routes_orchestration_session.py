"""POST /api/orchestrations/<id>/run/stream: an end user may only continue a
session they own, in the orchestration they are running.

Without this, a caller who learns another user's ``session_id`` runs the
orchestration with that user's history in the prompt, and the run is saved
under that user. DB-free: the session lookup is stubbed.
"""

import asyncio
import inspect
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from agentic_project_service.routes import _workflow_helpers
from agentic_project_service.routes import orchestrations as orchestrations_route
from agentic_project_service.services import orchestration as orchestration_service
from agentic_project_service.services.tool_caller import ToolCaller

USER_ID = "11111111-1111-4111-8111-111111111111"
OTHER_USER_ID = "22222222-2222-4222-8222-222222222222"
ORCH_ID = "4a8b2d3f-6c8e-4f22-8b4d-9e3a7c5f2b81"
OTHER_ORCH_ID = "6d0e4f5a-8b1c-4d44-8e6f-b05c9e7d4a03"
HEADERS = {"Authorization": "Bearer fake"}


def _as_user():
    return patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": USER_ID, "role": "authenticated", "aud": "authenticated"},
    )


def _as_service_role():
    return patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"role": "service_role", "is_service_role": True},
    )


def _run(existing_session, *, service=False, session_id="orch_sess_abc"):
    """POST a run; return (response, check_balance mock, session model mock).

    ``check_balance`` raises a sentinel, so a request that gets past the
    ownership gate surfaces as that exception rather than doing real work.
    """
    app = Flask(__name__)
    app.testing = True  # let the sentinel propagate
    app.register_blueprint(orchestrations_route.orchestrations_bp)
    body = {"message": "hi"}
    if session_id is not None:
        body["session_id"] = session_id
    with (
        _as_service_role() if service else _as_user(),
        patch.object(orchestrations_route.db, "session", MagicMock()),
        patch.object(orchestrations_route, "OrchestrationSessionModel") as session_model,
        patch.object(
            orchestrations_route.billing,
            "check_balance",
            side_effect=RuntimeError("got past the session check"),
        ) as check_balance,
        app.test_client() as c,
    ):
        session_model.query.filter_by.return_value.first.return_value = existing_session
        try:
            resp = c.post(f"/api/orchestrations/{ORCH_ID}/run/stream", headers=HEADERS, json=body)
        except RuntimeError as exc:
            assert "got past the session check" in str(exc)
            resp = None
    return resp, check_balance, session_model


def _session(user_id, orchestration_id=ORCH_ID):
    return MagicMock(user_id=user_id, orchestration_id=orchestration_id)


class TestEndUserSessionGate:
    @pytest.mark.parametrize(
        "existing",
        [
            pytest.param(_session(OTHER_USER_ID), id="another-users-session"),
            pytest.param(_session(None), id="ownerless-session"),
            pytest.param(_session(USER_ID, OTHER_ORCH_ID), id="own-session-other-orchestration"),
        ],
    )
    def test_refused_with_404_before_any_work(self, existing):
        resp, check_balance, session_model = _run(existing)
        assert resp is not None and resp.status_code == 404
        assert resp.get_json() == {"error": "Session not found"}
        check_balance.assert_not_called()
        session_model.query.filter_by.assert_called_once_with(session_id="orch_sess_abc")

    def test_own_session_in_this_orchestration_proceeds(self):
        resp, check_balance, _ = _run(_session(USER_ID))
        assert resp is None
        check_balance.assert_called_once()

    def test_own_session_matches_an_upper_case_orchestration_id(self):
        """The path segment is compared as a uuid, not as a string."""
        existing = _session(USER_ID, ORCH_ID.upper())
        resp, check_balance, _ = _run(existing)
        assert resp is None
        check_balance.assert_called_once()

    def test_unknown_session_id_proceeds_and_becomes_the_callers(self):
        resp, check_balance, _ = _run(None)
        assert resp is None
        check_balance.assert_called_once()

    def test_no_session_id_skips_the_lookup(self):
        resp, check_balance, session_model = _run(None, session_id=None)
        assert resp is None
        session_model.query.filter_by.assert_not_called()


class TestServiceRoleUnchanged:
    def test_service_role_may_continue_any_session(self):
        resp, check_balance, session_model = _run(
            _session(OTHER_USER_ID, OTHER_ORCH_ID), service=True
        )
        assert resp is None
        check_balance.assert_called_once()
        session_model.query.filter_by.assert_not_called()


# ---------------------------------------------------------------------------
# Workflow orchestration blocks run as the service role, and say so
# ---------------------------------------------------------------------------


class TestWorkflowOrchestrationCaller:
    def test_build_orchestration_requires_a_caller(self):
        """No default: a new call site that forgets it fails loudly, not tool-less."""
        param = inspect.signature(orchestration_service.build_orchestration).parameters["caller"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty

    def test_workflow_block_passes_the_service_role(self):
        output = MagicMock(content="done", steps=[], usage={})
        output.status.is_success.return_value = True
        orchestration = MagicMock()
        orchestration.run.return_value = output
        with patch.object(
            orchestration_service,
            "build_orchestration",
            return_value=(MagicMock(), orchestration),
        ) as build:
            services = _workflow_helpers.make_services()
            result = asyncio.run(services["run_orchestration"](ORCH_ID, "hi"))

        assert result["status"] == "completed"
        caller = build.call_args.kwargs["caller"]
        assert isinstance(caller, ToolCaller)
        assert caller.is_end_user is False


# ---------------------------------------------------------------------------
# GET /api/orchestrations/<id>/sessions — an end user lists only their own
# ---------------------------------------------------------------------------


def _list_sessions(*, service=False, user_id=USER_ID):
    app = Flask(__name__)
    app.register_blueprint(orchestrations_route.orchestrations_bp)
    with (
        _as_service_role() if service else _as_user(),
        patch.object(orchestrations_route, "get_current_user_id", return_value=user_id),
        patch.object(orchestrations_route, "OrchestrationSessionModel") as session_model,
        app.test_client() as c,
    ):
        query = session_model.query.filter_by.return_value
        query.filter_by.return_value = query
        query.order_by.return_value.limit.return_value.all.return_value = []
        resp = c.get(f"/api/orchestrations/{ORCH_ID}/sessions", headers=HEADERS)
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp, session_model


class TestListOrchestrationSessionsScoping:
    def test_end_user_is_filtered_to_their_own_sessions(self):
        _, session_model = _list_sessions()
        session_model.query.filter_by.assert_called_once_with(orchestration_id=ORCH_ID)
        session_model.query.filter_by.return_value.filter_by.assert_called_once_with(
            user_id=USER_ID
        )

    def test_end_user_without_an_id_sees_nothing_rather_than_ownerless_sessions(self):
        resp, session_model = _list_sessions(user_id=None)
        assert resp.get_json() == {"sessions": [], "total": 0}
        session_model.query.filter_by.return_value.filter_by.assert_not_called()

    def test_service_role_sees_every_session(self):
        _, session_model = _list_sessions(service=True)
        session_model.query.filter_by.return_value.filter_by.assert_not_called()
