"""Tests for DELETE /api/agents/<agent_id>/sessions/<session_id>.

Mirrors the DB-free harness in test_agents_session_create.py: a minimal Flask
app registering only agents_bp, auth mocked via `auth.decode_jwt`, and the DB
layer stubbed rather than exercised for real (this repo's `tests/unit` tier
is deliberately Postgres-free — see .github/workflows/test.yml).

Ownership check mirrors `run_agent`'s (`get_session_owner`, service-role
bypass), except "session not found" and "owned by someone else" both 404
here (there is no request body that could be valid regardless of ownership,
unlike run_agent's optional session_id).
"""

from unittest.mock import MagicMock, patch

from agentic_project_service.routes import agents as agents_route

AGENT_ID = "agent-1"
SESSION_ID = "sess_abc123"
USER_ID = "user-1"


def _make_test_app():
    """Create a minimal Flask app with the agents blueprint for test_client use."""
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(agents_route.agents_bp)
    return app


def _auth_headers():
    return {"Authorization": "Bearer fake"}


def _authed_as(user_id=USER_ID):
    return patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": user_id, "role": "authenticated"},
    )


class TestDeleteSessionRoute:
    def test_delete_own_session_returns_204_and_deletes_runs_then_session(self):
        app = _make_test_app()
        fake_db_session = MagicMock()
        fake_db_session.execute.return_value.fetchone.return_value = ("db-uuid-1",)

        manager = MagicMock()
        with (
            patch.object(agents_route, "get_session_owner", return_value=USER_ID) as mock_owner,
            patch.object(agents_route.db, "session", fake_db_session),
        ):
            manager.attach_mock(fake_db_session.execute, "execute")
            manager.attach_mock(fake_db_session.commit, "commit")

            with _authed_as():
                with app.test_client() as client:
                    resp = client.delete(
                        f"/api/agents/{AGENT_ID}/sessions/{SESSION_ID}",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 204
        assert resp.get_data() == b""

        mock_owner.assert_called_once_with(fake_db_session, SESSION_ID)

        # SELECT the internal id, then DELETE agent_runs, then DELETE
        # agent_sessions, then commit — in that order.
        execute_calls = [c for c in manager.mock_calls if c[0] == "execute"]
        assert len(execute_calls) == 3

        select_sql = str(execute_calls[0].args[0])
        assert "SELECT" in select_sql and "agent_sessions" in select_sql

        runs_delete_sql = str(execute_calls[1].args[0])
        assert "DELETE" in runs_delete_sql and "agent_runs" in runs_delete_sql
        assert execute_calls[1].args[1] == {"id": "db-uuid-1"}

        session_delete_sql = str(execute_calls[2].args[0])
        assert "DELETE" in session_delete_sql and "agent_sessions" in session_delete_sql
        assert execute_calls[2].args[1] == {"id": "db-uuid-1"}

        # Top-level calls (excluding the chained `.fetchone()`), in order.
        top_level_calls = [c[0] for c in manager.mock_calls if "." not in c[0]]
        assert top_level_calls == ["execute", "execute", "execute", "commit"]

    def test_delete_nonexistent_session_returns_404(self):
        app = _make_test_app()
        fake_db_session = MagicMock()

        with (
            patch.object(agents_route, "get_session_owner", return_value=None),
            patch.object(agents_route.db, "session", fake_db_session),
        ):
            with _authed_as():
                with app.test_client() as client:
                    resp = client.delete(
                        f"/api/agents/{AGENT_ID}/sessions/does-not-exist",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Session not found"}
        fake_db_session.execute.assert_not_called()
        fake_db_session.commit.assert_not_called()

    def test_delete_other_users_session_returns_404(self):
        app = _make_test_app()
        fake_db_session = MagicMock()

        with (
            patch.object(agents_route, "get_session_owner", return_value="other-user"),
            patch.object(agents_route.db, "session", fake_db_session),
        ):
            with _authed_as(user_id=USER_ID):
                with app.test_client() as client:
                    resp = client.delete(
                        f"/api/agents/{AGENT_ID}/sessions/{SESSION_ID}",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Session not found"}
        fake_db_session.execute.assert_not_called()
        fake_db_session.commit.assert_not_called()

    def test_delete_session_requires_auth(self):
        app = _make_test_app()
        with app.test_client() as client:
            resp = client.delete(f"/api/agents/{AGENT_ID}/sessions/{SESSION_ID}")
        assert resp.status_code == 401
