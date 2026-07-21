"""Tests for user-scoped access control on orchestration session endpoints.

Verifies that:
  - list_orchestration_sessions filters to the authenticated user's sessions
  - service-role callers see all sessions (bypass filter)
  - get_orchestration_session_messages returns 404 for another user's session
  - get_orchestration_session_messages returns 200 for the session owner

Mirrors the pattern from tests/test_agents.py:TestListSessionsScoping.
"""

import os
import uuid

import pytest
from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.services.orchestration import get_or_create_orchestration_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_orchestration(app):
    orch_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".orchestrations
                    (id, name, description, strategy, orchestrator_config, settings)
                VALUES (:id, 'access-test-orch', 'desc', 'supervisor', '{}', '{}')
                """
            ),
            {"id": orch_id},
        )
        db.session.commit()
    return orch_id


def _create_orchestration_session(app, orch_id, session_id, user_id):
    with app.app_context():
        get_or_create_orchestration_session(
            orchestration_id=orch_id,
            session_id=session_id,
            user_id=user_id,
        )
        db.session.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListOrchestrationSessionsScoping:
    """GET /<orch_id>/sessions respects user_id scoping."""

    def test_list_orchestration_sessions_excludes_other_users_sessions(self, client, app, mocker):
        """User A only sees their own sessions in the list; user B's are hidden."""
        user_a = str(uuid.uuid4())
        user_b = str(uuid.uuid4())
        orch_id = _create_orchestration(app)

        _create_orchestration_session(app, orch_id, "orch_sess_a1", user_a)
        _create_orchestration_session(app, orch_id, "orch_sess_b1", user_b)

        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": user_a, "role": "authenticated"},
        )

        resp = client.get(
            f"/api/orchestrations/{orch_id}/sessions",
            headers={"Authorization": "Bearer fake-token"},
        )
        assert resp.status_code == 200
        session_ids = {s["session_id"] for s in resp.get_json()["sessions"]}
        assert "orch_sess_a1" in session_ids
        assert "orch_sess_b1" not in session_ids

    def test_list_orchestration_sessions_service_role_sees_all(self, client, app, mocker):
        """A service-role caller sees all sessions regardless of user_id."""
        user_a = str(uuid.uuid4())
        user_b = str(uuid.uuid4())
        orch_id = _create_orchestration(app)

        _create_orchestration_session(app, orch_id, "orch_sess_sr_a", user_a)
        _create_orchestration_session(app, orch_id, "orch_sess_sr_b", user_b)

        mocker.patch.dict(os.environ, {"SERVICE_ROLE_KEY": "fake-service-role-key"})
        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={
                "sub": "service",
                "role": "service_role",
                "is_service_role": True,
            },
        )

        resp = client.get(
            f"/api/orchestrations/{orch_id}/sessions",
            headers={"Authorization": "Bearer fake-service-role-key"},
        )
        assert resp.status_code == 200
        session_ids = {s["session_id"] for s in resp.get_json()["sessions"]}
        assert {"orch_sess_sr_a", "orch_sess_sr_b"}.issubset(session_ids)


class TestGetOrchestrationSessionMessagesAccess:
    """GET /<orch_id>/sessions/<session_id>/messages enforces ownership."""

    def test_get_orchestration_session_messages_404_for_other_users_session(
        self, client, app, mocker
    ):
        """User A requesting user B's session_id receives 404 (not 403)."""
        user_a = str(uuid.uuid4())
        user_b = str(uuid.uuid4())
        orch_id = _create_orchestration(app)

        _create_orchestration_session(app, orch_id, "orch_sess_owned_by_b", user_b)

        # Authenticate as user A
        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": user_a, "role": "authenticated"},
        )

        resp = client.get(
            f"/api/orchestrations/{orch_id}/sessions/orch_sess_owned_by_b/messages",
            headers={"Authorization": "Bearer fake-token"},
        )
        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Session not found"}

    def test_get_orchestration_session_messages_owner_can_read(self, client, app, mocker):
        """Session owner gets 200 when requesting their own session's messages."""
        user_a = str(uuid.uuid4())
        orch_id = _create_orchestration(app)

        _create_orchestration_session(app, orch_id, "orch_sess_owned_by_a", user_a)

        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": user_a, "role": "authenticated"},
        )

        resp = client.get(
            f"/api/orchestrations/{orch_id}/sessions/orch_sess_owned_by_a/messages",
            headers={"Authorization": "Bearer fake-token"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["session_id"] == "orch_sess_owned_by_a"
        assert "messages" in data
