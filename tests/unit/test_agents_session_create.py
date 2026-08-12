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

AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"
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

    def test_unknown_agent_id_returns_404(self):
        """A well-formed but nonexistent agent_id must be rejected with a
        clean 404 before any session-creating write is attempted — agent_id
        is a foreign key on the sessions table, so reaching
        get_or_create_session with a bad id would otherwise surface as an
        unhandled integrity error instead of a normal 404."""
        app = _make_test_app()
        fake_db_session = MagicMock()
        fake_db_session.execute.return_value.fetchone.return_value = None
        unknown_but_valid = "9c1e7d40-2a55-4f8b-b0d3-6e5a1c9f8b22"

        with (
            patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
            patch.object(agents_route, "seed_session_runs") as mock_seed,
            patch.object(agents_route.db, "session", fake_db_session),
        ):
            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{unknown_but_valid}/sessions",
                        json={},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Agent not found"}
        mock_get_or_create.assert_not_called()
        mock_seed.assert_not_called()
        fake_db_session.commit.assert_not_called()

    def test_malformed_agent_id_returns_404_without_querying(self):
        """`agents.id` is a uuid column, so a non-uuid path segment must be
        rejected before it reaches SQL — passing it through makes Postgres
        raise `invalid input syntax for type uuid` and poisons the
        transaction, turning the intended 404 into a 500."""
        app = _make_test_app()
        fake_db_session = MagicMock()

        with (
            patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
            patch.object(agents_route.db, "session", fake_db_session),
        ):
            mock_get_or_create.return_value = ("db-uuid-1", "sess_abc123", True)

            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        "/api/agents/does-not-exist/sessions",
                        json={},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Agent not found"}
        fake_db_session.execute.assert_not_called()
        mock_get_or_create.assert_not_called()

    def test_agent_deleted_between_check_and_commit_returns_404(self):
        """The existence pre-check is not a lock: an agent deleted between it
        and the commit surfaces as an IntegrityError on the session insert,
        which must roll back to a 404 rather than a 500."""
        from sqlalchemy.exc import IntegrityError

        app = _make_test_app()
        fake_db_session = MagicMock()
        fake_db_session.execute.return_value.fetchone.return_value = ("gpt-4o-mini",)
        fake_db_session.commit.side_effect = IntegrityError("stmt", {}, Exception("fk"))

        with (
            patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
            patch.object(agents_route.db, "session", fake_db_session),
        ):
            mock_get_or_create.return_value = ("db-uuid-1", "sess_abc123", True)

            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Agent not found"}
        fake_db_session.rollback.assert_called_once()

    def test_malformed_json_body_rejected(self):
        """A body that fails to parse must 400. Silently treating it as "no
        seed" would discard an imported conversation and report 201."""
        app = _make_test_app()

        with (
            patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
            patch.object(agents_route.db, "session", MagicMock()),
        ):
            mock_get_or_create.return_value = ("db-uuid-1", "sess_abc123", True)

            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        data='{"initial_messages": [{"role": "user", "content": "Q1"',
                        content_type="application/json",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 400
        assert resp.get_json() == {"error": "Request body must be valid JSON"}
        mock_get_or_create.assert_not_called()

    def test_body_without_json_content_type_rejected(self):
        """Valid JSON sent without `application/json` does not parse either —
        same silent-discard hazard, same 400."""
        app = _make_test_app()

        with (
            patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
            patch.object(agents_route.db, "session", MagicMock()),
        ):
            mock_get_or_create.return_value = ("db-uuid-1", "sess_abc123", True)

            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        data='{"initial_messages": []}',
                        content_type="text/plain",
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 400
        assert resp.get_json() == {"error": "Request body must be valid JSON"}
        mock_get_or_create.assert_not_called()

    def test_non_object_json_body_rejected(self):
        """A top-level array or string is valid JSON but has no `.get` —
        it must 400, not raise AttributeError into a 500."""
        app = _make_test_app()

        for body in ([1, 2], "x"):
            with (
                patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
                patch.object(agents_route.db, "session", MagicMock()),
            ):
                mock_get_or_create.return_value = ("db-uuid-1", "sess_abc123", True)

                with _authed_as():
                    with app.test_client() as client:
                        resp = client.post(
                            f"/api/agents/{AGENT_ID}/sessions",
                            json=body,
                            headers=_auth_headers(),
                        )

            assert resp.status_code == 400, body
            assert resp.get_json() == {"error": "Request body must be a JSON object"}
            mock_get_or_create.assert_not_called()

    def test_falsy_non_list_initial_messages_rejected(self):
        """`{}`, `""`, `0` and `false` are not lists either — they must be
        rejected like any other wrong type, not coerced to "no seed"."""
        app = _make_test_app()

        for value in ({}, "", 0, False):
            with (
                patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
                patch.object(agents_route.db, "session", MagicMock()),
            ):
                mock_get_or_create.return_value = ("db-uuid-1", "sess_abc123", True)

                with _authed_as():
                    with app.test_client() as client:
                        resp = client.post(
                            f"/api/agents/{AGENT_ID}/sessions",
                            json={"initial_messages": value},
                            headers=_auth_headers(),
                        )

            assert resp.status_code == 400, value
            assert resp.get_json() == {"error": "initial_messages must be a list"}
            mock_get_or_create.assert_not_called()

    def test_null_initial_messages_creates_bare_session(self):
        """Explicit `null` means "no seed" — only absence, not a wrong type."""
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
                        json={"initial_messages": None},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 201
        mock_seed.assert_not_called()


class TestCreateSessionOnBehalfOfAUser:
    """A service-role caller imports a conversation *for* an end user. Its JWT
    `sub` is not that user, so without an explicit owner the session is written
    with user_id NULL: invisible in list_sessions, and undeletable through the
    agent-scoped route."""

    def test_service_role_can_set_the_session_owner(self):
        app = _make_test_app()
        owner = "6b8f0c11-4d2a-4f5e-9b7c-1e3a5d7f9021"
        fake_db_session = MagicMock()
        fake_db_session.execute.return_value.fetchone.return_value = ("gpt-4o-mini",)

        with (
            patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
            patch.object(agents_route.db, "session", fake_db_session),
        ):
            mock_get_or_create.return_value = ("db-uuid-1", "sess_abc123", True)

            with _authed_as_service_role():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={"user_id": owner},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 201
        assert mock_get_or_create.call_args.kwargs["user_id"] == owner

    def test_user_scoped_caller_cannot_set_the_session_owner(self):
        app = _make_test_app()
        fake_db_session = MagicMock()
        fake_db_session.execute.return_value.fetchone.return_value = ("gpt-4o-mini",)

        with (
            patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
            patch.object(agents_route.db, "session", fake_db_session),
        ):
            mock_get_or_create.return_value = ("db-uuid-1", "sess_abc123", True)

            with _authed_as():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={"user_id": "6b8f0c11-4d2a-4f5e-9b7c-1e3a5d7f9021"},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 403
        assert resp.get_json() == {"error": "Service role required to set user_id"}
        mock_get_or_create.assert_not_called()

    def test_service_role_without_user_id_still_creates_a_session(self):
        """Omitting user_id keeps the existing behaviour — the JWT `sub`."""
        app = _make_test_app()
        fake_db_session = MagicMock()
        fake_db_session.execute.return_value.fetchone.return_value = ("gpt-4o-mini",)

        with (
            patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
            patch.object(agents_route.db, "session", fake_db_session),
        ):
            mock_get_or_create.return_value = ("db-uuid-1", "sess_abc123", True)

            with _authed_as_service_role():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 201
        assert mock_get_or_create.call_args.kwargs["user_id"] == "service"

    def test_malformed_user_id_rejected(self):
        """agent_sessions.user_id is a uuid column — a non-uuid owner would
        fail at the insert with 22P02 rather than a 400."""
        app = _make_test_app()
        fake_db_session = MagicMock()

        with (
            patch.object(agents_route, "get_or_create_session") as mock_get_or_create,
            patch.object(agents_route.db, "session", fake_db_session),
        ):
            mock_get_or_create.return_value = ("db-uuid-1", "sess_abc123", True)

            with _authed_as_service_role():
                with app.test_client() as client:
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/sessions",
                        json={"user_id": "not-a-uuid"},
                        headers=_auth_headers(),
                    )

        assert resp.status_code == 400
        assert resp.get_json() == {"error": "user_id must be a uuid"}
        mock_get_or_create.assert_not_called()


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

    def test_each_pair_gets_a_strictly_increasing_created_at(self):
        """load_session_history orders by created_at ASC with no tie-break, so
        the seeding loop must stamp each pair itself. Leaving created_at to a
        fresh datetime.now(UTC) inside persist_agent_run makes correct ordering
        a matter of how long the loop happens to take: pairs written in the
        same microsecond tie, and the sort may interleave them — exactly the
        corruption one-run-per-pair exists to prevent."""
        msgs = []
        for i in range(20):
            msgs.append({"role": "user", "content": f"Q{i}"})
            msgs.append({"role": "assistant", "content": f"A{i}"})
        fake_db_session = MagicMock()

        with patch("agentic_project_service.services.session.persist_agent_run") as mock_persist:
            seed_session_runs(fake_db_session, "db-uuid-1", msgs)

        stamps = [c.kwargs["created_at"] for c in mock_persist.call_args_list]
        assert len(stamps) == 20
        assert all(a < b for a, b in zip(stamps, stamps[1:], strict=False))

    def test_run_timestamps_match_the_pair_stamp(self):
        """started_at / completed_at track created_at so all three columns
        agree on the pair's position in the imported conversation."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        fake_db_session = MagicMock()

        with patch("agentic_project_service.services.session.persist_agent_run") as mock_persist:
            seed_session_runs(fake_db_session, "db-uuid-1", msgs)

        for call in mock_persist.call_args_list:
            assert call.kwargs["started_at"] == call.kwargs["created_at"]
            assert call.kwargs["completed_at"] == call.kwargs["created_at"]
