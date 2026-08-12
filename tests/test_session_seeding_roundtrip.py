"""Postgres-backed tests for POST /api/agents/<agent_id>/sessions.

The unit tier stubs `db.session` wholesale, so it cannot see anything that
only a real database decides: uuid casting, foreign keys, `ON DELETE CASCADE`,
or the `created_at ASC` ordering that `load_session_history` reconstructs
seeded conversations from. Those are the properties tested here, against the
`ai` schema the root-tier conftest builds.
"""

import uuid

from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.services.session import load_session_history


class TestAgentIdHandling:
    def test_malformed_agent_id_returns_404(self, client, mock_auth, auth_headers):
        """`agents.id` is a uuid column — a non-uuid path segment reaching SQL
        raises `invalid input syntax for type uuid` (22P02), which surfaces as
        a 500 with a poisoned transaction rather than the intended 404."""
        resp = client.post(
            "/api/agents/does-not-exist/sessions",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Agent not found"}

    def test_unknown_agent_id_returns_404(self, client, mock_auth, auth_headers):
        resp = client.post(
            f"/api/agents/{uuid.uuid4()}/sessions",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Agent not found"}

    def test_no_session_row_is_left_behind_by_a_rejected_create(
        self, client, mock_auth, auth_headers, app
    ):
        client.post("/api/agents/does-not-exist/sessions", json={}, headers=auth_headers)
        with app.app_context():
            count = db.session.execute(text('SELECT COUNT(*) FROM "ai".agent_sessions')).scalar()
        assert count == 0

    def test_a_session_is_deletable_under_the_spelling_it_was_created_with(
        self, client, mock_auth, auth_headers, test_agent
    ):
        """Create accepts any uuid spelling Python parses; delete compares
        against `str(row.agent_id)`, which is canonical. Unless both routes
        normalize, a session created under an uppercase or braced agent id can
        never be deleted under that same id. `urn:uuid:` is here too: Python
        parses it and Postgres's uuid cast does not, so passing it through raw
        is a 22P02 — a 500, not a 404."""
        canonical = test_agent["id"]
        for spelling in (
            canonical.upper(),
            "{" + canonical + "}",
            f"urn:uuid:{canonical}",
            canonical.replace("-", ""),
        ):
            created = client.post(
                f"/api/agents/{spelling}/sessions",
                json={},
                headers=auth_headers,
            )
            assert created.status_code == 201, (spelling, created.get_data(as_text=True))
            session_id = created.get_json()["session_id"]

            deleted = client.delete(
                f"/api/agents/{spelling}/sessions/{session_id}",
                headers=auth_headers,
            )
            assert deleted.status_code == 204, (spelling, deleted.get_data(as_text=True))


class TestSeedingOnBehalfOfAUser:
    def test_service_role_seeded_session_belongs_to_the_named_user(
        self, client, mock_auth, auth_headers, test_agent, app, mocker
    ):
        """The import use case: a service-role caller seeds a conversation for
        an end user, who must then see it. Naming the owner is the only way a
        service-role create can produce a session that shows up in
        list_sessions and is deletable through the agent-scoped route."""
        target_user = str(uuid.uuid4())
        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "service", "role": "service_role", "is_service_role": True},
        )
        resp = client.post(
            f"/api/agents/{test_agent['id']}/sessions",
            json={
                "user_id": target_user,
                "initial_messages": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                ],
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        session_id = resp.get_json()["session_id"]

        with app.app_context():
            owner = db.session.execute(
                text('SELECT user_id FROM "ai".agent_sessions WHERE session_id = :sid'),
                {"sid": session_id},
            ).scalar()
        assert str(owner) == target_user

        # The named user can now list it, and delete it through this blueprint.
        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": target_user, "role": "authenticated"},
        )
        listed = client.get(f"/api/agents/{test_agent['id']}/sessions", headers=auth_headers)
        assert listed.status_code == 200
        assert session_id in {s["session_id"] for s in listed.get_json()["sessions"]}

        deleted = client.delete(
            f"/api/agents/{test_agent['id']}/sessions/{session_id}", headers=auth_headers
        )
        assert deleted.status_code == 204

    def test_service_role_bare_session_is_written_with_a_null_owner(
        self, client, mock_auth, auth_headers, test_agent, app, mocker
    ):
        """A service token's `sub` is not a project user and is not a uuid, so
        it is never forwarded as the owner — the column really goes to NULL.
        Forwarding it would fail the INSERT with 22P02 rather than write this
        row at all."""
        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "service", "role": "service_role", "is_service_role": True},
        )
        resp = client.post(
            f"/api/agents/{test_agent['id']}/sessions",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 201, resp.get_data(as_text=True)
        session_id = resp.get_json()["session_id"]

        with app.app_context():
            owner = db.session.execute(
                text('SELECT user_id FROM "ai".agent_sessions WHERE session_id = :sid'),
                {"sid": session_id},
            ).scalar()
        assert owner is None

    def test_service_role_cannot_seed_a_session_it_names_no_owner_for(
        self, client, mock_auth, auth_headers, test_agent, app, mocker
    ):
        """A NULL-owner session passes the `owner is not None` run/stream
        ownership checks for every authenticated user, so an ownerless session
        carrying an imported conversation hands that conversation to anyone who
        learns its id. Rejected before anything is written."""
        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "service", "role": "service_role", "is_service_role": True},
        )
        resp = client.post(
            f"/api/agents/{test_agent['id']}/sessions",
            json={
                "initial_messages": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                ]
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert resp.get_json() == {
            "error": "user_id is required to seed initial_messages as service role"
        }

        with app.app_context():
            sessions = db.session.execute(text('SELECT COUNT(*) FROM "ai".agent_sessions')).scalar()
            runs = db.session.execute(text('SELECT COUNT(*) FROM "ai".agent_runs')).scalar()
        assert sessions == 0
        assert runs == 0


class TestSeedRoundTrip:
    def _seed(self, client, auth_headers, agent_id, messages):
        resp = client.post(
            f"/api/agents/{agent_id}/sessions",
            json={"initial_messages": messages},
            headers=auth_headers,
        )
        assert resp.status_code == 201, resp.get_data(as_text=True)
        return resp.get_json()["session_id"]

    def _db_uuid(self, app, session_id):
        with app.app_context():
            return str(
                db.session.execute(
                    text('SELECT id FROM "ai".agent_sessions WHERE session_id = :sid'),
                    {"sid": session_id},
                ).scalar()
            )

    def test_ten_pairs_round_trip_in_order(self, client, mock_auth, auth_headers, test_agent, app):
        """The endpoint's central claim: what goes in as `initial_messages`
        comes back out of `load_session_history` as the same conversation, in
        the same order. `load_session_history` orders by `created_at ASC`, so
        ten pairs written in one tight loop must not tie on that column."""
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Q{i}"})
            messages.append({"role": "assistant", "content": f"A{i}"})

        session_id = self._seed(client, auth_headers, test_agent["id"], messages)
        db_uuid = self._db_uuid(app, session_id)

        with app.app_context():
            history = load_session_history(db.session, db_uuid)
            stamps = [
                row[0]
                for row in db.session.execute(
                    text(
                        """
                        SELECT created_at FROM "ai".agent_runs
                        WHERE session_id = :sid ORDER BY created_at ASC
                        """
                    ),
                    {"sid": db_uuid},
                ).fetchall()
            ]

        assert [(m.role, m.content) for m in history] == [
            (m["role"], m["content"]) for m in messages
        ]
        # No two runs may share the sort key the reconstruction above depends on.
        assert len(set(stamps)) == 10

    def test_citation_markers_are_stripped_from_seeded_assistant_content(
        self, client, mock_auth, auth_headers, test_agent, app
    ):
        """`load_session_history` runs `strip_citation_markers` over assistant
        content. Seeded text that happens to contain `[n]` is rewritten by it
        on the way out — pinned here so the behaviour is a decision, not a
        surprise found in production."""
        messages = [
            {"role": "user", "content": "What does the report say [1]?"},
            {"role": "assistant", "content": "The report says X [1] and Y [2]."},
        ]
        session_id = self._seed(client, auth_headers, test_agent["id"], messages)

        with app.app_context():
            history = load_session_history(db.session, self._db_uuid(app, session_id))

        # User content passes through untouched; assistant content loses the markers.
        assert history[0].content == "What does the report say [1]?"
        assert history[1].content == "The report says X and Y."

    def test_seeded_runs_are_completed_and_scoped_to_the_session(
        self, client, mock_auth, auth_headers, test_agent, app
    ):
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        session_id = self._seed(client, auth_headers, test_agent["id"], messages)
        db_uuid = self._db_uuid(app, session_id)

        with app.app_context():
            rows = db.session.execute(
                text(
                    """
                    SELECT status, run_id, agent_id, created_at
                    FROM "ai".agent_runs
                    WHERE session_id = :sid
                    ORDER BY created_at ASC
                    """
                ),
                {"sid": db_uuid},
            ).fetchall()

        assert len(rows) == 2
        assert all(row[0] == "completed" for row in rows)
        assert all(row[1].startswith("seed_") for row in rows)
        assert all(str(row[2]) == test_agent["id"] for row in rows)
        # Distinct created_at is what keeps the ASC ordering deterministic.
        assert rows[0][3] < rows[1][3]

    def test_session_delete_cascades_to_seeded_runs(
        self, client, mock_auth, auth_headers, test_agent, app
    ):
        """`agent_runs.session_id` is `ON DELETE CASCADE`, so deleting the
        session is enough — the route does not delete runs itself.

        The cascade proved here is the ORM-declared one: the conftest bootstraps
        with `db.metadata.create_all`, so this says nothing about whether the
        migration-applied schema declares the same FK.
        """
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        session_id = self._seed(client, auth_headers, test_agent["id"], messages)
        db_uuid = self._db_uuid(app, session_id)

        resp = client.delete(
            f"/api/agents/{test_agent['id']}/sessions/{session_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 204

        with app.app_context():
            runs = db.session.execute(
                text('SELECT COUNT(*) FROM "ai".agent_runs WHERE session_id = :sid'),
                {"sid": db_uuid},
            ).scalar()
            sessions = db.session.execute(
                text('SELECT COUNT(*) FROM "ai".agent_sessions WHERE id = :sid'),
                {"sid": db_uuid},
            ).scalar()
        assert runs == 0
        assert sessions == 0
