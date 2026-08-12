"""Tests for DELETE /api/agents/<agent_id>/sessions/<session_id>.

Mirrors the DB-free harness in test_agents_session_create.py: a minimal Flask
app registering only agents_bp, auth mocked via `auth.decode_jwt`, and the DB
layer stubbed rather than exercised for real (this repo's `tests/unit` tier is
deliberately Postgres-free). The properties that need a real database — the
ON DELETE CASCADE this route relies on to remove the session's runs — are
covered in tests/test_session_seeding_roundtrip.py.

"Not found", "owned by someone else" and "belongs to a different agent" all
answer 404 with the same body, so the endpoint cannot be used to enumerate
sessions. Service role bypasses the ownership half of that; agent scoping
still applies to it.
"""

from unittest.mock import MagicMock, patch

from agentic_project_service.routes import agents as agents_route

AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"
OTHER_AGENT_ID = "9c1e7d40-2a55-4f8b-b0d3-6e5a1c9f8b22"
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


def _authed_as_service_role():
    """auth.decode_jwt marks the payload is_service_role=True when the token
    matches SERVICE_ROLE_KEY; patch that branch directly."""
    return patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "service", "role": "service_role", "is_service_role": True},
    )


def _db_session_returning(row):
    fake = MagicMock()
    fake.execute.return_value.fetchone.return_value = row
    return fake


class TestDeleteSessionRoute:
    def test_delete_own_session_returns_204(self):
        """One SELECT answers both checks — ownership and agent scoping — and
        one DELETE removes the session; its runs go with it via the
        `agent_runs.session_id` ON DELETE CASCADE."""
        app = _make_test_app()
        fake_db_session = _db_session_returning(("db-uuid-1", AGENT_ID, USER_ID))

        manager = MagicMock()
        with patch.object(agents_route.db, "session", fake_db_session):
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

        execute_calls = [c for c in manager.mock_calls if c[0] == "execute"]
        assert len(execute_calls) == 2

        select_sql = str(execute_calls[0].args[0])
        assert "SELECT" in select_sql and "agent_sessions" in select_sql
        assert execute_calls[0].args[1] == {"session_id": SESSION_ID}

        delete_sql = str(execute_calls[1].args[0])
        assert "DELETE" in delete_sql and "agent_sessions" in delete_sql
        assert execute_calls[1].args[1] == {"id": "db-uuid-1"}

        # Top-level calls (excluding the chained `.fetchone()`), in order.
        top_level_calls = [c[0] for c in manager.mock_calls if "." not in c[0]]
        assert top_level_calls == ["execute", "execute", "commit"]

    def test_delete_nonexistent_session_returns_404(self):
        app = _make_test_app()
        fake_db_session = _db_session_returning(None)

        with patch.object(agents_route.db, "session", fake_db_session):
            with _authed_as():
                with app.test_client() as client:
                    resp = client.delete(
                        f"/api/agents/{AGENT_ID}/sessions/does-not-exist",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Session not found"}
        fake_db_session.execute.assert_called_once()
        fake_db_session.commit.assert_not_called()

    def test_delete_other_users_session_returns_404(self):
        app = _make_test_app()
        fake_db_session = _db_session_returning(("db-uuid-1", AGENT_ID, "other-user"))

        with patch.object(agents_route.db, "session", fake_db_session):
            with _authed_as(user_id=USER_ID):
                with app.test_client() as client:
                    resp = client.delete(
                        f"/api/agents/{AGENT_ID}/sessions/{SESSION_ID}",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Session not found"}
        fake_db_session.execute.assert_called_once()
        fake_db_session.commit.assert_not_called()

    def test_delete_unowned_session_returns_404(self):
        """A session with a NULL user_id belongs to nobody, so no user-scoped
        caller may delete it."""
        app = _make_test_app()
        fake_db_session = _db_session_returning(("db-uuid-1", AGENT_ID, None))

        with patch.object(agents_route.db, "session", fake_db_session):
            with _authed_as():
                with app.test_client() as client:
                    resp = client.delete(
                        f"/api/agents/{AGENT_ID}/sessions/{SESSION_ID}",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Session not found"}
        fake_db_session.commit.assert_not_called()

    def test_agent_id_spelling_does_not_change_the_comparison(self):
        """The scoping check compares against `str(row.agent_id)`, which is
        always canonical lowercase-hyphenated. Uppercase, braced and `urn:uuid:`
        spellings all name the same agent and all cast in Postgres's uuid input
        (or, for `urn:uuid:`, would not cast at all), so the path segment is
        normalized before it is compared — otherwise a session created under
        one spelling could never be deleted under it."""
        app = _make_test_app()

        for spelling in (
            AGENT_ID.upper(),
            "{" + AGENT_ID + "}",
            f"urn:uuid:{AGENT_ID}",
            AGENT_ID.replace("-", ""),
        ):
            fake_db_session = _db_session_returning(("db-uuid-1", AGENT_ID, USER_ID))

            with patch.object(agents_route.db, "session", fake_db_session):
                with _authed_as():
                    with app.test_client() as client:
                        resp = client.delete(
                            f"/api/agents/{spelling}/sessions/{SESSION_ID}",
                            headers=_auth_headers(),
                        )

            assert resp.status_code == 204, spelling
            fake_db_session.commit.assert_called_once()

    def test_malformed_agent_id_returns_404_without_querying(self):
        """A non-uuid path segment cannot name any agent, so it 404s before the
        lookup rather than being compared as a string."""
        app = _make_test_app()
        fake_db_session = _db_session_returning(("db-uuid-1", AGENT_ID, USER_ID))

        with patch.object(agents_route.db, "session", fake_db_session):
            with _authed_as():
                with app.test_client() as client:
                    resp = client.delete(
                        f"/api/agents/does-not-exist/sessions/{SESSION_ID}",
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

    def test_session_of_different_agent_returns_404(self):
        """A session owned by the caller but belonging to a different agent
        than the one in the path 404s — mirrors remove_agent_tool's and
        delete_agent_hook's `str(row.agent_id) != agent_id` scoping for the
        same `/<agent_id>/<resource>/<resource_id>` DELETE shape."""
        app = _make_test_app()
        fake_db_session = _db_session_returning(("db-uuid-1", OTHER_AGENT_ID, USER_ID))

        with patch.object(agents_route.db, "session", fake_db_session):
            with _authed_as():
                with app.test_client() as client:
                    resp = client.delete(
                        f"/api/agents/{AGENT_ID}/sessions/{SESSION_ID}",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Session not found"}
        # Only the lookup SELECT ran — no DELETE, no commit.
        fake_db_session.execute.assert_called_once()
        fake_db_session.commit.assert_not_called()

    def test_orphaned_session_is_not_deletable_here(self):
        """`agent_sessions.agent_id` is ON DELETE SET NULL, so a session whose
        agent was deleted can never match the agent in the path. It stays
        deletable through DELETE /api/sessions/<session_id>."""
        app = _make_test_app()
        fake_db_session = _db_session_returning(("db-uuid-1", None, USER_ID))

        with patch.object(agents_route.db, "session", fake_db_session):
            with _authed_as():
                with app.test_client() as client:
                    resp = client.delete(
                        f"/api/agents/{AGENT_ID}/sessions/{SESSION_ID}",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 404
        fake_db_session.commit.assert_not_called()

    def test_service_role_deletes_a_session_it_does_not_own(self):
        """The bypass exists so a service caller can clean up any session;
        ownership is not consulted for it."""
        app = _make_test_app()
        fake_db_session = _db_session_returning(("db-uuid-1", AGENT_ID, "someone-else"))

        with patch.object(agents_route.db, "session", fake_db_session):
            with _authed_as_service_role():
                with app.test_client() as client:
                    resp = client.delete(
                        f"/api/agents/{AGENT_ID}/sessions/{SESSION_ID}",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 204
        fake_db_session.commit.assert_called_once()

    def test_service_role_still_gets_404_for_a_session_of_another_agent(self):
        """The bypass is about ownership only — agent scoping still applies."""
        app = _make_test_app()
        fake_db_session = _db_session_returning(("db-uuid-1", OTHER_AGENT_ID, "someone-else"))

        with patch.object(agents_route.db, "session", fake_db_session):
            with _authed_as_service_role():
                with app.test_client() as client:
                    resp = client.delete(
                        f"/api/agents/{AGENT_ID}/sessions/{SESSION_ID}",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Session not found"}
        fake_db_session.commit.assert_not_called()

    def test_service_role_gets_404_for_a_nonexistent_session(self):
        app = _make_test_app()
        fake_db_session = _db_session_returning(None)

        with patch.object(agents_route.db, "session", fake_db_session):
            with _authed_as_service_role():
                with app.test_client() as client:
                    resp = client.delete(
                        f"/api/agents/{AGENT_ID}/sessions/does-not-exist",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Session not found"}
        fake_db_session.commit.assert_not_called()
