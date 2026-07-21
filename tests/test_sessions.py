"""Tests for session CRUD routes."""

import json
import uuid

from sqlalchemy import text


class TestGetSession:
    def test_get(self, client, mock_auth, auth_headers, test_agent, app):
        from agentic_project_service.db import db

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agent_sessions (id, session_id, agent_id, user_id)
                    VALUES (:id, :session_id, :agent_id, :user_id)
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "agent_id": test_agent["id"],
                    "user_id": mock_auth,
                },
            )
            db.session.commit()

        resp = client.get(
            f"/api/sessions/{session_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["session_id"] == session_id

    def test_get_not_found(self, client, mock_auth, auth_headers):
        resp = client.get(
            "/api/sessions/nonexistent",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestDeleteSession:
    def test_delete(self, client, mock_auth, auth_headers, test_agent, app):
        from agentic_project_service.db import db

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agent_sessions (id, session_id, agent_id, user_id)
                    VALUES (:id, :session_id, :agent_id, :user_id)
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "agent_id": test_agent["id"],
                    "user_id": mock_auth,
                },
            )
            db.session.commit()

        resp = client.delete(
            f"/api/sessions/{session_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_delete_not_found(self, client, mock_auth, auth_headers):
        resp = client.delete(
            "/api/sessions/nonexistent",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestGetMessages:
    def test_get_empty(self, client, mock_auth, auth_headers, test_agent, app):
        from agentic_project_service.db import db

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        db_session_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agent_sessions (id, session_id, agent_id, user_id)
                    VALUES (:id, :session_id, :agent_id, :user_id)
                    """
                ),
                {
                    "id": db_session_id,
                    "session_id": session_id,
                    "agent_id": test_agent["id"],
                    "user_id": mock_auth,
                },
            )
            db.session.commit()

        resp = client.get(
            f"/api/sessions/{session_id}/messages",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["session_id"] == session_id
        assert data["messages"] == []


class TestGetRuns:
    def test_get_empty(self, client, mock_auth, auth_headers, test_agent, app):
        from agentic_project_service.db import db

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agent_sessions (id, session_id, agent_id, user_id)
                    VALUES (:id, :session_id, :agent_id, :user_id)
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "agent_id": test_agent["id"],
                    "user_id": mock_auth,
                },
            )
            db.session.commit()

        resp = client.get(
            f"/api/sessions/{session_id}/runs",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["session_id"] == session_id
        assert data["runs"] == []

    def test_get_runs_omits_retrieved_context_and_keeps_citations(
        self, client, mock_auth, auth_headers, test_agent, app
    ):
        from agentic_project_service.db import db

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        db_session_id = str(uuid.uuid4())
        run_db_id = str(uuid.uuid4())
        run_id = f"run_{uuid.uuid4().hex[:12]}"

        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agent_sessions (id, session_id, agent_id, user_id)
                    VALUES (:id, :session_id, :agent_id, :user_id)
                    """
                ),
                {
                    "id": db_session_id,
                    "session_id": session_id,
                    "agent_id": test_agent["id"],
                    "user_id": mock_auth,
                },
            )
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agent_runs
                    (id, session_id, run_id, status, input_messages, output_messages, content, usage, retrieved_context, error, created_at)
                    VALUES
                    (:id, :session_id, :run_id, 'completed',
                     CAST(:input_messages AS jsonb), CAST(:output_messages AS jsonb),
                     :content, CAST(:usage AS jsonb), CAST(:retrieved_context AS jsonb), NULL, NOW())
                    """
                ),
                {
                    "id": run_db_id,
                    "session_id": db_session_id,
                    "run_id": run_id,
                    "input_messages": json.dumps([{"role": "user", "content": "hi"}]),
                    "output_messages": json.dumps([{"role": "assistant", "content": "hello"}]),
                    "content": "hello",
                    "usage": json.dumps({"prompt_tokens": 3, "completion_tokens": 4}),
                    "retrieved_context": json.dumps(
                        [{"_type": "text_embedding_chunk", "id": "chunk-1"}]
                    ),
                },
            )
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".message_citations
                    (id, run_id, citation_key, item_id, source_id, text_excerpt, meta)
                    VALUES
                    (:id1, :run_id, 2, :item2, NULL, 'excerpt 2', '{}'::jsonb),
                    (:id2, :run_id, 1, :item1, NULL, 'excerpt 1', '{}'::jsonb)
                    """
                ),
                {
                    "id1": str(uuid.uuid4()),
                    "id2": str(uuid.uuid4()),
                    "run_id": run_db_id,
                    "item1": str(uuid.uuid4()),
                    "item2": str(uuid.uuid4()),
                },
            )
            db.session.commit()

        resp = client.get(f"/api/sessions/{session_id}/runs", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["session_id"] == session_id
        assert len(data["runs"]) == 1
        run = data["runs"][0]
        assert run["run_id"] == run_id
        assert "retrieved_context" not in run
        assert [c["key"] for c in run["citations"]] == ["1", "2"]


class TestGetRunRetrievedContext:
    def test_get_retrieved_context_for_run(self, client, mock_auth, auth_headers, test_agent, app):
        from agentic_project_service.db import db

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        db_session_id = str(uuid.uuid4())
        run_id = f"run_{uuid.uuid4().hex[:12]}"

        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agent_sessions (id, session_id, agent_id, user_id)
                    VALUES (:id, :session_id, :agent_id, :user_id)
                    """
                ),
                {
                    "id": db_session_id,
                    "session_id": session_id,
                    "agent_id": test_agent["id"],
                    "user_id": mock_auth,
                },
            )
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agent_runs
                    (id, session_id, run_id, status, retrieved_context, created_at)
                    VALUES
                    (:id, :session_id, :run_id, 'completed',
                     CAST(:retrieved_context AS jsonb), NOW())
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "session_id": db_session_id,
                    "run_id": run_id,
                    "retrieved_context": json.dumps(
                        [{"_type": "text_embedding_chunk", "id": "chunk-ctx"}]
                    ),
                },
            )
            db.session.commit()

        resp = client.get(
            f"/api/sessions/{session_id}/runs/{run_id}/retrieved-context",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["session_id"] == session_id
        assert data["run_id"] == run_id
        assert data["retrieved_context"] == [{"_type": "text_embedding_chunk", "id": "chunk-ctx"}]

    def test_get_retrieved_context_not_found(
        self, client, mock_auth, auth_headers, test_agent, app
    ):
        from agentic_project_service.db import db

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agent_sessions (id, session_id, agent_id, user_id)
                    VALUES (:id, :session_id, :agent_id, :user_id)
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "agent_id": test_agent["id"],
                    "user_id": mock_auth,
                },
            )
            db.session.commit()

        resp = client.get(
            f"/api/sessions/{session_id}/runs/run_nonexistent/retrieved-context",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestSessionOwnershipScoping:
    """Ownership checks on /api/sessions routes — cross-user access denied."""

    def _create_session_for_user(self, app, agent_id, user_id, session_id):
        from agentic_project_service.db import db

        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agent_sessions (id, session_id, agent_id, user_id)
                    VALUES (:id, :session_id, :agent_id, :user_id)
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "user_id": user_id,
                },
            )
            db.session.commit()

    def test_get_session_denies_cross_user(self, client, app, mocker, test_agent):
        """User B gets 404 when fetching user A's session."""
        user_a = str(uuid.uuid4())
        user_b = str(uuid.uuid4())
        session_id = f"sess_xuser_{uuid.uuid4().hex[:8]}"

        self._create_session_for_user(app, test_agent["id"], user_a, session_id)

        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": user_b, "role": "authenticated"},
        )

        resp = client.get(
            f"/api/sessions/{session_id}",
            headers={"Authorization": "Bearer fake-token"},
        )
        assert resp.status_code == 404

    def test_get_session_allows_owner(self, client, app, mocker, test_agent):
        """The owning user can fetch their own session."""
        user_a = str(uuid.uuid4())
        session_id = f"sess_own_{uuid.uuid4().hex[:8]}"

        self._create_session_for_user(app, test_agent["id"], user_a, session_id)

        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": user_a, "role": "authenticated"},
        )

        resp = client.get(
            f"/api/sessions/{session_id}",
            headers={"Authorization": "Bearer fake-token"},
        )
        assert resp.status_code == 200

    def test_get_runs_denies_cross_user(self, client, app, mocker, test_agent):
        """User B gets 404 when listing runs for user A's session."""
        user_a = str(uuid.uuid4())
        user_b = str(uuid.uuid4())
        session_id = f"sess_runs_{uuid.uuid4().hex[:8]}"

        self._create_session_for_user(app, test_agent["id"], user_a, session_id)

        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": user_b, "role": "authenticated"},
        )

        resp = client.get(
            f"/api/sessions/{session_id}/runs",
            headers={"Authorization": "Bearer fake-token"},
        )
        assert resp.status_code == 404

    def test_get_messages_denies_cross_user(self, client, app, mocker, test_agent):
        """User B gets 404 when fetching messages for user A's session."""
        user_a = str(uuid.uuid4())
        user_b = str(uuid.uuid4())
        session_id = f"sess_msgs_{uuid.uuid4().hex[:8]}"

        self._create_session_for_user(app, test_agent["id"], user_a, session_id)

        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": user_b, "role": "authenticated"},
        )

        resp = client.get(
            f"/api/sessions/{session_id}/messages",
            headers={"Authorization": "Bearer fake-token"},
        )
        assert resp.status_code == 404

    def test_delete_session_denies_cross_user(self, client, app, mocker, test_agent):
        """User B cannot delete user A's session."""
        from agentic_project_service.db import db

        user_a = str(uuid.uuid4())
        user_b = str(uuid.uuid4())
        session_id = f"sess_del_{uuid.uuid4().hex[:8]}"

        self._create_session_for_user(app, test_agent["id"], user_a, session_id)

        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": user_b, "role": "authenticated"},
        )

        resp = client.delete(
            f"/api/sessions/{session_id}",
            headers={"Authorization": "Bearer fake-token"},
        )
        assert resp.status_code == 404

        # Confirm the session still exists
        with app.app_context():
            row = db.session.execute(
                text('SELECT 1 FROM "ai".agent_sessions WHERE session_id = :sid'),
                {"sid": session_id},
            ).fetchone()
            assert row is not None


class TestAuthRequired:
    def test_no_token(self, client):
        resp = client.get("/api/sessions/some-id")
        assert resp.status_code == 401
