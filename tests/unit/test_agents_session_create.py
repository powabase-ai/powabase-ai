"""Tests for POST /api/agents/<agent_id>/sessions — session creation with
optional conversation seeding.

Mirrors the DB-free harness in test_agents_route_billing.py: a minimal Flask
app registering only agents_bp, auth mocked via `auth.decode_jwt`, and the DB
layer stubbed rather than exercised for real (this repo's `tests/unit` tier
is deliberately Postgres-free — see .github/workflows/test.yml).

Two levels of coverage:
  * TestCreateSessionRoute — validation (alternation/shape) and wiring
    (get_or_create_session / seed_session_runs / commit called correctly),
    with those two service calls mocked out.
  * TestSeedSessionRuns — the seeding helper in isolation, mocking
    persist_agent_run to prove one synthetic COMPLETED run is written per
    user/assistant pair, in order — the property that keeps
    load_session_history's reconstruction from interleaving pairs.
"""

from unittest.mock import MagicMock, patch

from agentic_project_service.models.tenant import AgentRunStatus
from agentic_project_service.routes import agents as agents_route
from agentic_project_service.services.session import seed_session_runs

AGENT_ID = "agent-1"
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


# ---------------------------------------------------------------------------
# Route: validation + wiring
# ---------------------------------------------------------------------------


class TestCreateSessionRoute:
    def test_create_session_without_messages_returns_session_id(self):
        app = _make_test_app()
        with (
            patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
            patch.object(agents_route, "seed_session_runs") as mock_seed,
            patch.object(agents_route.db, "session", MagicMock()),
        ):
            mock_get_or_create.return_value = ("db-uuid-1", "sess_abc123", True)

            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 201
        assert resp.get_json() == {"session_id": "sess_abc123"}
        mock_seed.assert_not_called()

    def test_create_session_with_no_body_returns_session_id(self):
        """Omitted body (not even `{}`) is also a bare-session create."""
        app = _make_test_app()
        with (
            patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
            patch.object(agents_route, "seed_session_runs") as mock_seed,
            patch.object(agents_route.db, "session", MagicMock()),
        ):
            mock_get_or_create.return_value = ("db-uuid-1", "sess_abc123", True)

            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 201
        assert resp.get_json()["session_id"] == "sess_abc123"
        mock_seed.assert_not_called()

    def test_non_alternating_messages_rejected(self):
        """Starting with 'assistant' instead of 'user' is rejected."""
        app = _make_test_app()
        msgs = [{"role": "assistant", "content": "A1"}, {"role": "user", "content": "Q1"}]

        with patch.object(agents_route, "get_or_create_session") as mock_get_or_create:
            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={"initial_messages": msgs},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 400
        assert "error" in resp.get_json()
        mock_get_or_create.assert_not_called()

    def test_odd_count_rejected(self):
        app = _make_test_app()
        msgs = [{"role": "user", "content": "Q1"}]

        with patch.object(agents_route, "get_or_create_session") as mock_get_or_create:
            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={"initial_messages": msgs},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 400
        mock_get_or_create.assert_not_called()

    def test_role_mismatch_mid_sequence_rejected(self):
        """user, user, assistant, assistant breaks alternation at index 1."""
        app = _make_test_app()
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A1"},
            {"role": "assistant", "content": "A2"},
        ]

        with patch.object(agents_route, "get_or_create_session") as mock_get_or_create:
            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={"initial_messages": msgs},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 400
        mock_get_or_create.assert_not_called()

    def test_non_string_content_rejected(self):
        app = _make_test_app()
        msgs = [{"role": "user", "content": 123}, {"role": "assistant", "content": "A1"}]

        with patch.object(agents_route, "get_or_create_session") as mock_get_or_create:
            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={"initial_messages": msgs},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 400
        mock_get_or_create.assert_not_called()

    def test_non_list_initial_messages_rejected(self):
        app = _make_test_app()

        with patch.object(agents_route, "get_or_create_session") as mock_get_or_create:
            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={"initial_messages": "not-a-list"},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 400
        mock_get_or_create.assert_not_called()

    def test_non_dict_message_item_rejected(self):
        app = _make_test_app()

        with patch.object(agents_route, "get_or_create_session") as mock_get_or_create:
            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={"initial_messages": ["not-a-dict", "also-not"]},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 400
        mock_get_or_create.assert_not_called()

    def test_valid_messages_seeded_and_session_committed(self):
        """Valid alternating messages: get_or_create_session then
        seed_session_runs are called with the right arguments, in order,
        followed by a commit."""
        app = _make_test_app()
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        fake_db_session = MagicMock()

        manager = MagicMock()
        with (
            patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
            patch.object(agents_route, "seed_session_runs") as mock_seed,
            patch.object(agents_route.db, "session", fake_db_session),
        ):
            mock_get_or_create.return_value = ("db-uuid-1", "sess_xyz789", True)
            manager.attach_mock(mock_get_or_create, "get_or_create_session")
            manager.attach_mock(mock_seed, "seed_session_runs")
            manager.attach_mock(fake_db_session.commit, "commit")

            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={"initial_messages": msgs},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 201
        assert resp.get_json() == {"session_id": "sess_xyz789"}

        mock_get_or_create.assert_called_once_with(
            db_session=fake_db_session, agent_id=AGENT_ID, user_id=USER_ID
        )
        mock_seed.assert_called_once_with(fake_db_session, "db-uuid-1", msgs)

        # get_or_create_session, then seed_session_runs, then commit — in order.
        assert [c[0] for c in manager.mock_calls] == [
            "get_or_create_session",
            "seed_session_runs",
            "commit",
        ]

    def test_create_session_passes_caller_user_id(self):
        """The authenticated caller's user_id (JWT `sub`) is forwarded to
        get_or_create_session so the session records the right owner."""
        app = _make_test_app()
        other_user = "user-42"

        with (
            patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
            patch.object(agents_route.db, "session", MagicMock()),
        ):
            mock_get_or_create.return_value = ("db-uuid-2", "sess_def456", True)

            with _authed_as(user_id=other_user):
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 201
        assert mock_get_or_create.call_args.kwargs["user_id"] == other_user

    def test_create_session_requires_auth(self):
        app = _make_test_app()
        with app.test_client() as client:
            resp = client.post(f"/api/agents/{AGENT_ID}/sessions", json={})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# seed_session_runs — the seeding helper in isolation
# ---------------------------------------------------------------------------


class TestSeedSessionRuns:
    def test_creates_one_run_per_pair_in_order(self):
        """Two user/assistant pairs produce exactly two persist_agent_run
        calls, one per pair, in the given order — never packed into one run
        (which would reorder history as Q1, Q2, A1, A2 on reconstruction)."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        fake_db_session = MagicMock()

        with patch("agentic_project_service.services.session.persist_agent_run") as mock_persist:
            seed_session_runs(fake_db_session, "db-uuid-1", msgs)

        assert mock_persist.call_count == 2

        first_call, second_call = mock_persist.call_args_list
        assert first_call.kwargs["input_messages"] == [{"role": "user", "content": "Q1"}]
        assert first_call.kwargs["output_messages"] == [{"role": "assistant", "content": "A1"}]
        assert first_call.kwargs["content"] == "A1"

        assert second_call.kwargs["input_messages"] == [{"role": "user", "content": "Q2"}]
        assert second_call.kwargs["output_messages"] == [{"role": "assistant", "content": "A2"}]
        assert second_call.kwargs["content"] == "A2"

    def test_each_run_is_completed_status_for_the_given_session(self):
        msgs = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]
        fake_db_session = MagicMock()

        with patch("agentic_project_service.services.session.persist_agent_run") as mock_persist:
            seed_session_runs(fake_db_session, "db-uuid-1", msgs)

        call = mock_persist.call_args
        assert call.kwargs["status"] == AgentRunStatus.COMPLETED
        assert call.kwargs["db_session_uuid"] == "db-uuid-1"
        assert call.kwargs["db_session"] == fake_db_session
        assert call.kwargs["started_at"] is not None
        assert call.kwargs["completed_at"] is not None

    def test_run_id_is_seed_prefixed_and_unique_per_pair(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        fake_db_session = MagicMock()

        with patch("agentic_project_service.services.session.persist_agent_run") as mock_persist:
            seed_session_runs(fake_db_session, "db-uuid-1", msgs)

        run_ids = [c.kwargs["run_id"] for c in mock_persist.call_args_list]
        assert all(rid.startswith("seed_") for rid in run_ids)
        assert len(set(run_ids)) == len(run_ids)

    def test_empty_messages_is_a_noop(self):
        fake_db_session = MagicMock()
        with patch("agentic_project_service.services.session.persist_agent_run") as mock_persist:
            seed_session_runs(fake_db_session, "db-uuid-1", [])

        mock_persist.assert_not_called()
