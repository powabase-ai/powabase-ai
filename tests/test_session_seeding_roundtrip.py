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

        with app.app_context():
            history = load_session_history(db.session, self._db_uuid(app, session_id))

        assert [(m.role, m.content) for m in history] == [
            (m["role"], m["content"]) for m in messages
        ]

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
        session is enough — the route does not delete runs itself."""
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
