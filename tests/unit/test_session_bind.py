"""An end user's run binds to its session atomically, owner check included.

The run routes check that an end user may continue a ``session_id`` before
any work starts, but the session is bound later: in the SSE generator, after
the tools and sub-agents are built. Another user can create that session in
between. So the bind itself re-checks: ``get_or_create_session`` and
``get_or_create_orchestration_session`` take the caller as ``end_user_id``,
create the session with ``INSERT ... ON CONFLICT DO NOTHING`` (two creators of
one id cannot both win), and refuse any existing session that is not the
caller's own in this agent or orchestration.

DB-free: the data layer is stubbed. The database tier
(tests/test_session_bind_store.py) runs the same functions against Postgres.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from agentic_project_service.routes import agents as agents_route
from agentic_project_service.routes import orchestrations as orchestrations_route
from agentic_project_service.services import hook_loader
from agentic_project_service.services import orchestration as orchestration_service
from agentic_project_service.services.session import (
    SessionNotAccessible,
    get_or_create_session,
)

USER_ID = "11111111-1111-4111-8111-111111111111"
OTHER_USER_ID = "22222222-2222-4222-8222-222222222222"
AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"
OTHER_AGENT_ID = "8c2d6e7f-0a3b-4c66-8a8b-d27e1f9c6d25"
ORCH_ID = "4a8b2d3f-6c8e-4f22-8b4d-9e3a7c5f2b81"
OTHER_ORCH_ID = "6d0e4f5a-8b1c-4d44-8e6f-b05c9e7d4a03"
ROW_ID = "9d3e7f80-1b4c-4d77-9b9c-e38f2a0d7e36"
HEADERS = {"Authorization": "Bearer fake"}


class _FakeSessionTable:
    """A db session whose INSERT conflicts when ``existing`` is set.

    ``existing`` is the (id, owner-scope id, user_id) row a re-read returns.
    Every statement is recorded as (sql, params).
    """

    def __init__(self, existing=None):
        self.existing = existing
        self.statements: list[tuple[str, dict]] = []
        self.session = MagicMock()
        self.session.execute.side_effect = self._execute

    def _execute(self, statement, params=None):
        sql = " ".join(str(statement).split())
        self.statements.append((sql, params or {}))
        result = MagicMock()
        if sql.startswith("INSERT"):
            result.fetchone.return_value = None if self.existing else (ROW_ID,)
        else:
            result.fetchone.return_value = self.existing
        return result

    @property
    def inserts(self):
        return [(sql, params) for sql, params in self.statements if sql.startswith("INSERT")]


# ---------------------------------------------------------------------------
# services.session.get_or_create_session(..., end_user_id=...)
# ---------------------------------------------------------------------------


def _bind_agent_session(existing, *, session_id="sess_x", agent_id=AGENT_ID):
    table = _FakeSessionTable(existing)
    result = get_or_create_session(
        table.session,
        agent_id,
        session_id=session_id,
        user_id=USER_ID,
        end_user_id=USER_ID,
    )
    return result, table


class TestAgentSessionBind:
    def test_an_unknown_named_id_is_refused_and_nothing_is_created(self):
        """End users never name a new session: a backend's ids can be guessed."""
        table = _FakeSessionTable(None)
        with pytest.raises(SessionNotAccessible):
            get_or_create_session(table.session, AGENT_ID, session_id="sess_x", end_user_id=USER_ID)
        assert table.inserts == []

    def test_no_id_generates_one_for_the_caller_without_a_race(self):
        (row_id, session_id, is_new), table = _bind_agent_session(None, session_id=None)
        assert is_new is True and session_id.startswith("sess_")
        [(sql, params)] = table.inserts
        assert "ON CONFLICT (session_id) DO NOTHING" in sql
        assert "RETURNING id" in sql
        assert params["session_id"] == session_id
        assert params["user_id"] == USER_ID
        assert params["agent_id"] == AGENT_ID

    def test_own_session_of_this_agent_is_continued(self):
        (row_id, session_id, is_new), _ = _bind_agent_session((ROW_ID, AGENT_ID, USER_ID))
        assert (row_id, session_id, is_new) == (ROW_ID, "sess_x", False)

    def test_own_session_matches_an_upper_case_agent_id(self):
        """The path segment is compared as a uuid, not as a string."""
        result, _ = _bind_agent_session((ROW_ID, AGENT_ID, USER_ID), agent_id=AGENT_ID.upper())
        assert result[2] is False

    @pytest.mark.parametrize(
        "existing",
        [
            pytest.param((ROW_ID, AGENT_ID, OTHER_USER_ID), id="another-users-session"),
            pytest.param((ROW_ID, AGENT_ID, None), id="ownerless-session"),
            pytest.param((ROW_ID, OTHER_AGENT_ID, USER_ID), id="own-session-other-agent"),
            pytest.param((ROW_ID, None, USER_ID), id="own-session-deleted-agent"),
        ],
    )
    def test_any_other_existing_session_is_refused(self, existing):
        with pytest.raises(SessionNotAccessible):
            _bind_agent_session(existing)

    def test_without_end_user_id_the_service_path_is_unchanged(self):
        """The service role may continue any session; nothing is re-checked."""
        table = _FakeSessionTable((ROW_ID, "sess_x"))
        result = get_or_create_session(table.session, AGENT_ID, session_id="sess_x")
        assert result == (ROW_ID, "sess_x", False)
        assert table.inserts == []


# ---------------------------------------------------------------------------
# services.orchestration.get_or_create_orchestration_session(..., end_user_id=...)
# ---------------------------------------------------------------------------


def _bind_orchestration_session(existing, *, session_id="orch_sess_x", orch_id=ORCH_ID):
    table = _FakeSessionTable(existing)
    with patch.object(orchestration_service.db, "session", table.session):
        result = orchestration_service.get_or_create_orchestration_session(
            orchestration_id=orch_id,
            session_id=session_id,
            user_id=USER_ID,
            end_user_id=USER_ID,
        )
    return result, table


class TestOrchestrationSessionBind:
    def test_an_unknown_named_id_is_refused_and_nothing_is_created(self):
        """End users never name a new session: a backend's ids can be guessed."""
        with pytest.raises(SessionNotAccessible):
            _bind_orchestration_session(None)

    def test_no_id_generates_one_for_the_caller_without_a_race(self):
        (_, session_id, is_new), table = _bind_orchestration_session(None, session_id=None)
        assert is_new is True and session_id.startswith("orch_sess_")
        [(sql, params)] = table.inserts
        assert "ON CONFLICT (session_id) DO NOTHING" in sql
        assert "RETURNING id" in sql
        assert params["session_id"] == session_id
        assert params["user_id"] == USER_ID
        assert params["orchestration_id"] == ORCH_ID

    def test_own_session_of_this_orchestration_is_continued(self):
        (row_id, session_id, is_new), _ = _bind_orchestration_session((ROW_ID, ORCH_ID, USER_ID))
        assert (row_id, session_id, is_new) == (ROW_ID, "orch_sess_x", False)

    def test_own_session_matches_an_upper_case_orchestration_id(self):
        result, _ = _bind_orchestration_session((ROW_ID, ORCH_ID, USER_ID), orch_id=ORCH_ID.upper())
        assert result[2] is False

    @pytest.mark.parametrize(
        "existing",
        [
            pytest.param((ROW_ID, ORCH_ID, OTHER_USER_ID), id="another-users-session"),
            pytest.param((ROW_ID, ORCH_ID, None), id="ownerless-session"),
            pytest.param((ROW_ID, OTHER_ORCH_ID, USER_ID), id="own-session-other-orchestration"),
            pytest.param((ROW_ID, None, USER_ID), id="own-session-deleted-orchestration"),
        ],
    )
    def test_any_other_existing_session_is_refused(self, existing):
        with pytest.raises(SessionNotAccessible):
            _bind_orchestration_session(existing)


# ---------------------------------------------------------------------------
# The run routes bind as the caller, and a refused bind stops the run
# ---------------------------------------------------------------------------


def _as(service: bool):
    claims = (
        {"role": "service_role", "is_service_role": True}
        if service
        else {"sub": USER_ID, "role": "authenticated", "aud": "authenticated"}
    )
    return patch("agentic_project_service.auth.decode_jwt", return_value=claims)


def _sse_events(resp) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in resp.get_data(as_text=True).splitlines()
        if line.startswith("data: ")
    ]


def _agent_app():
    app = Flask(__name__)
    app.register_blueprint(agents_route.agents_bp)
    return app


def _post_agent_run(path, *, service=False, bind_error=None):
    """POST an agent run whose session bind is a mock; return (resp, mocks)."""
    session = MagicMock()
    session.execute.return_value.fetchone.return_value = (AGENT_ID, "A", "gpt-4o-mini", "", {})
    bind = MagicMock(side_effect=bind_error, return_value=(ROW_ID, "sess_x", False))
    with (
        _as(service),
        patch.object(agents_route.db, "session", session),
        patch.object(agents_route.billing, "check_balance"),
        patch.object(agents_route, "session_accessible_to", return_value=True),
        patch.object(agents_route, "check_model_available"),
        patch.object(agents_route, "resolve_api_key_or_raise_for_drop", return_value=None),
        patch.object(agents_route, "get_setting", return_value=4000),
        patch.object(agents_route, "get_or_create_session", bind),
        patch.object(agents_route, "load_session_history", side_effect=RuntimeError("stop")),
        patch.object(agents_route, "persist_agent_run") as persist,
        patch.object(agents_route, "update_agent_run") as update,
        _agent_app().test_client() as c,
    ):
        resp = c.post(
            path, headers=HEADERS, json={"message": "hi", "session_id": "sess_x"}, buffered=True
        )
    return resp, bind, persist, update


class TestAgentRunRoutesBindAsTheCaller:
    @pytest.mark.parametrize("stream", [False, True], ids=["run", "run-stream"])
    def test_end_user_bind_names_the_caller(self, stream):
        path = f"/api/agents/{AGENT_ID}/run" + ("/stream" if stream else "")
        _, bind, _, _ = _post_agent_run(path)
        first = bind.call_args_list[0]
        assert first.kwargs["end_user_id"] == USER_ID
        assert first.kwargs["session_id"] == "sess_x"

    @pytest.mark.parametrize("stream", [False, True], ids=["run", "run-stream"])
    def test_service_role_bind_names_no_end_user(self, stream):
        path = f"/api/agents/{AGENT_ID}/run" + ("/stream" if stream else "")
        _, bind, _, _ = _post_agent_run(path, service=True)
        assert bind.call_args_list[0].kwargs.get("end_user_id") is None

    def test_refused_bind_is_a_404_and_nothing_is_saved(self):
        resp, bind, persist, update = _post_agent_run(
            f"/api/agents/{AGENT_ID}/run", bind_error=SessionNotAccessible("sess_x")
        )
        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Session not found"}
        assert bind.call_count == 1  # no second bind to save a failed run
        persist.assert_not_called()
        update.assert_not_called()

    def test_refused_bind_ends_the_stream_with_an_error_and_nothing_is_saved(self):
        resp, bind, persist, update = _post_agent_run(
            f"/api/agents/{AGENT_ID}/run/stream", bind_error=SessionNotAccessible("sess_x")
        )
        assert resp.status_code == 200
        events = _sse_events(resp)
        assert events == [{"event": "error", "error": "Session not found"}]
        assert bind.call_count == 1
        persist.assert_not_called()
        update.assert_not_called()


def _post_orchestration_run(*, service=False, bind_error=None):
    app = Flask(__name__)
    app.register_blueprint(orchestrations_route.orchestrations_bp)
    session = MagicMock()
    session.get.return_value = None  # no orchestration row: model checks are skipped
    bind = MagicMock(side_effect=bind_error, return_value=(ROW_ID, "orch_sess_x", False))
    with (
        _as(service),
        patch.object(orchestrations_route.db, "session", session),
        patch.object(orchestrations_route.billing, "check_balance"),
        patch.object(orchestrations_route, "_end_user_may_run_in_session", return_value=True),
        patch.object(
            orchestration_service, "build_orchestration", return_value=(MagicMock(), MagicMock())
        ),
        patch.object(hook_loader, "load_hooks_for_orchestration", return_value=[]),
        patch.object(orchestration_service, "get_or_create_orchestration_session", bind),
        patch.object(
            orchestration_service, "create_orchestration_run", side_effect=RuntimeError("stop")
        ) as create_run,
        patch.object(orchestration_service, "update_orchestration_run") as update_run,
        app.test_client() as c,
    ):
        resp = c.post(
            f"/api/orchestrations/{ORCH_ID}/run/stream",
            headers=HEADERS,
            json={"message": "hi", "session_id": "orch_sess_x"},
            buffered=True,
        )
    return resp, bind, create_run, update_run


class TestOrchestrationRunBindsAsTheCaller:
    def test_end_user_bind_names_the_caller(self):
        _, bind, _, _ = _post_orchestration_run()
        assert bind.call_args.kwargs["end_user_id"] == USER_ID
        assert bind.call_args.kwargs["session_id"] == "orch_sess_x"

    def test_service_role_bind_names_no_end_user(self):
        _, bind, _, _ = _post_orchestration_run(service=True)
        assert bind.call_args.kwargs.get("end_user_id") is None

    def test_refused_bind_ends_the_stream_with_an_error_and_nothing_is_saved(self):
        resp, _, create_run, update_run = _post_orchestration_run(
            bind_error=SessionNotAccessible("orch_sess_x")
        )
        assert resp.status_code == 200
        assert _sse_events(resp) == [{"event": "error", "error": "Session not found"}]
        create_run.assert_not_called()
        update_run.assert_not_called()
