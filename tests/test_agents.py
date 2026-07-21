"""Tests for agent CRUD routes."""

from sqlalchemy import text


class TestCreateAgent:
    def test_create(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/agents",
            json={"name": "My Agent", "model": "gpt-4o"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["name"] == "My Agent"
        assert data["model"] == "gpt-4o"
        assert "id" in data

    def test_create_defaults(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/agents",
            json={"name": "Default Agent"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["model"]  # should have a default model

    def test_create_with_system_prompt(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/agents",
            json={
                "name": "Helper",
                "system_prompt": "You are a helpful assistant.",
                "settings": {"temperature": 0.7},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["system_prompt"] == "You are a helpful assistant."
        assert data["settings"]["temperature"] == 0.7

    def test_create_missing_name(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/agents",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400


class TestListAgents:
    def test_list(self, client, mock_auth, auth_headers, test_agent):
        resp = client.get("/api/agents", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] >= 1
        assert any(a["id"] == test_agent["id"] for a in data["agents"])

    def test_list_empty(self, client, mock_auth, auth_headers):
        resp = client.get("/api/agents", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["total"] == 0

    def test_list_pagination(self, client, mock_auth, auth_headers):
        # Create 3 agents
        for i in range(3):
            client.post(
                "/api/agents",
                json={"name": f"Agent {i}"},
                headers=auth_headers,
            )
        resp = client.get("/api/agents?limit=2&offset=0", headers=auth_headers)
        data = resp.get_json()
        assert len(data["agents"]) == 2
        assert data["total"] == 3


class TestGetAgent:
    def test_get(self, client, mock_auth, auth_headers, test_agent):
        resp = client.get(
            f"/api/agents/{test_agent['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] == test_agent["id"]
        assert data["name"] == test_agent["name"]

    def test_get_not_found(self, client, mock_auth, auth_headers):
        import uuid

        resp = client.get(
            f"/api/agents/{uuid.uuid4()}",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestUpdateAgent:
    def test_update_name(self, client, mock_auth, auth_headers, test_agent):
        resp = client.patch(
            f"/api/agents/{test_agent['id']}",
            json={"name": "Renamed Agent"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "Renamed Agent"

    def test_update_model(self, client, mock_auth, auth_headers, test_agent):
        resp = client.patch(
            f"/api/agents/{test_agent['id']}",
            json={"model": "gpt-4o"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["model"] == "gpt-4o"

    def test_update_system_prompt(self, client, mock_auth, auth_headers, test_agent):
        resp = client.patch(
            f"/api/agents/{test_agent['id']}",
            json={"system_prompt": "New prompt"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["system_prompt"] == "New prompt"

    def test_update_no_data(self, client, mock_auth, auth_headers, test_agent):
        resp = client.patch(
            f"/api/agents/{test_agent['id']}",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400


class TestDeleteAgent:
    def test_delete(self, client, mock_auth, auth_headers, test_agent):
        resp = client.delete(
            f"/api/agents/{test_agent['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["message"] == "Agent deleted"

        # Verify it's gone
        resp = client.get(
            f"/api/agents/{test_agent['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestAuthRequired:
    def test_no_token(self, client):
        resp = client.get("/api/agents")
        assert resp.status_code == 401


class TestRunStreamSessionOwnership:
    """The /run/stream endpoint must reject session_ids owned by another user."""

    def test_denies_cross_user_session_id(self, client, app, mocker):
        """User B cannot post to /run/stream with user A's session_id."""
        import uuid

        from sqlalchemy import text

        from agentic_project_service.db import db
        from agentic_project_service.services.session import get_or_create_session

        user_a = str(uuid.uuid4())
        user_b = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())
        session_id = f"sess_stream_{uuid.uuid4().hex[:8]}"

        with app.app_context():
            db.session.execute(
                text(
                    "INSERT INTO ai.agents (id, name, model) "
                    "VALUES (:id, 'stream-scope-test', 'gpt-4o')"
                ),
                {"id": agent_id},
            )
            get_or_create_session(db.session, agent_id, session_id=session_id, user_id=user_a)
            db.session.commit()

        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": user_b, "role": "authenticated"},
        )

        resp = client.post(
            f"/api/agents/{agent_id}/run/stream",
            headers={
                "Authorization": "Bearer fake-token",
                "Content-Type": "application/json",
            },
            json={"message": "hi", "session_id": session_id},
        )
        assert resp.status_code == 404


class TestListSessionsScoping:
    """Tests that /api/agents/<agent_id>/sessions filters by authenticated user."""

    def test_list_sessions_filters_by_authenticated_user(self, client, app, mocker):
        """A user listing sessions for an agent only sees their own sessions."""
        import uuid

        from sqlalchemy import text

        from agentic_project_service.db import db
        from agentic_project_service.services.session import get_or_create_session

        user_a = str(uuid.uuid4())
        user_b = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())

        with app.app_context():
            db.session.execute(
                text(
                    "INSERT INTO ai.agents (id, name, model) VALUES (:id, 'scope-test-a', 'gpt-4o')"
                ),
                {"id": agent_id},
            )
            get_or_create_session(db.session, agent_id, session_id="sess_a1", user_id=user_a)
            get_or_create_session(db.session, agent_id, session_id="sess_a2", user_id=user_a)
            get_or_create_session(db.session, agent_id, session_id="sess_b1", user_id=user_b)
            db.session.commit()

        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": user_a, "role": "authenticated"},
        )

        response = client.get(
            f"/api/agents/{agent_id}/sessions",
            headers={"Authorization": "Bearer fake-token"},
        )
        assert response.status_code == 200
        session_ids = {s["session_id"] for s in response.get_json()["sessions"]}
        assert session_ids == {"sess_a1", "sess_a2"}
        assert "sess_b1" not in session_ids

    def test_list_sessions_service_role_sees_all(self, client, app, mocker):
        """A service-role caller sees all sessions regardless of user_id."""
        import os
        import uuid

        from sqlalchemy import text

        from agentic_project_service.db import db
        from agentic_project_service.services.session import get_or_create_session

        user_a = str(uuid.uuid4())
        user_b = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())

        with app.app_context():
            db.session.execute(
                text(
                    "INSERT INTO ai.agents (id, name, model) VALUES (:id, 'scope-test-b', 'gpt-4o')"
                ),
                {"id": agent_id},
            )
            get_or_create_session(db.session, agent_id, session_id="sess_sra", user_id=user_a)
            get_or_create_session(db.session, agent_id, session_id="sess_srb", user_id=user_b)
            db.session.commit()

        # Service-role JWT path: auth.decode_jwt marks the payload with
        # is_service_role=True when the token matches SERVICE_ROLE_KEY.
        # For this test we patch decode_jwt to simulate that branch directly.
        mocker.patch.dict(os.environ, {"SERVICE_ROLE_KEY": "fake-service-role-key"})
        mocker.patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={
                "sub": "service",
                "role": "service_role",
                "is_service_role": True,
            },
        )

        response = client.get(
            f"/api/agents/{agent_id}/sessions",
            headers={"Authorization": "Bearer fake-service-role-key"},
        )
        assert response.status_code == 200
        session_ids = {s["session_id"] for s in response.get_json()["sessions"]}
        assert {"sess_sra", "sess_srb"}.issubset(session_ids)


class TestListSessionsValidation:
    """Tests that /api/agents/<agent_id>/sessions surfaces invalid filter
    params with a 400 instead of silently dropping the filter."""

    def _seed_agent(self, app) -> str:
        import uuid

        from sqlalchemy import text

        from agentic_project_service.db import db

        agent_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text(
                    "INSERT INTO ai.agents (id, name, model) "
                    "VALUES (:id, 'validation-test', 'gpt-4o')"
                ),
                {"id": agent_id},
            )
            db.session.commit()
        return agent_id

    def test_min_runs_non_integer_returns_400(self, client, app, mock_auth, auth_headers):
        agent_id = self._seed_agent(app)
        resp = client.get(
            f"/api/agents/{agent_id}/sessions?min_runs=abc", headers=auth_headers
        )
        assert resp.status_code == 400
        assert "min_runs" in resp.get_json()["error"]
        # Regression: an earlier version silently dropped the filter and
        # returned 200 with all sessions. Pin that the bad input is rejected.

    def test_max_runs_non_integer_returns_400(self, client, app, mock_auth, auth_headers):
        agent_id = self._seed_agent(app)
        resp = client.get(
            f"/api/agents/{agent_id}/sessions?max_runs=foo", headers=auth_headers
        )
        assert resp.status_code == 400
        assert "max_runs" in resp.get_json()["error"]

    def test_min_runs_integer_succeeds(self, client, app, mock_auth, auth_headers):
        """Valid integer keeps working — guards against an over-broad fix."""
        agent_id = self._seed_agent(app)
        resp = client.get(
            f"/api/agents/{agent_id}/sessions?min_runs=2", headers=auth_headers
        )
        assert resp.status_code == 200


class TestListAgentsExtended:
    def _seed_agents(self, app, count: int, name_prefix: str = "Agent"):
        from agentic_project_service.db import db

        with app.app_context():
            for i in range(count):
                db.session.execute(
                    text(
                        "INSERT INTO ai.agents (id, name, model, system_prompt, settings) "
                        "VALUES (gen_random_uuid(), :name, 'gpt-4o', NULL, '{}'::jsonb)"
                    ),
                    {"name": f"{name_prefix} {i:03d}"},
                )
            db.session.commit()

    def test_envelope_unchanged(self, client, mock_auth, auth_headers, app):
        """Public API back-compat: { agents, total, limit, offset } envelope preserved."""
        self._seed_agents(app, 3)
        resp = client.get("/api/agents", headers=auth_headers)
        data = resp.get_json()
        assert set(data.keys()) >= {"agents", "total", "limit", "offset"}
        assert isinstance(data["agents"], list)
        assert isinstance(data["total"], int)

    def test_search_by_name(self, client, mock_auth, auth_headers, app):
        self._seed_agents(app, 5, name_prefix="Foo")
        self._seed_agents(app, 5, name_prefix="Bar")
        resp = client.get("/api/agents?q=Foo", headers=auth_headers)
        data = resp.get_json()
        assert data["total"] == 5
        assert all("Foo" in a["name"] for a in data["agents"])

    def test_sort_by_name(self, client, mock_auth, auth_headers, app):
        self._seed_agents(app, 5)
        resp = client.get("/api/agents?sort=name&order=asc", headers=auth_headers)
        data = resp.get_json()
        names = [a["name"] for a in data["agents"]]
        assert names == sorted(names)

    def test_sort_invalid_returns_400(self, client, mock_auth, auth_headers):
        resp = client.get("/api/agents?sort=bogus", headers=auth_headers)
        assert resp.status_code == 400

    def test_response_includes_session_count(self, client, mock_auth, auth_headers, app):
        from agentic_project_service.db import db

        with app.app_context():
            agent_id = db.session.execute(
                text(
                    "INSERT INTO ai.agents (id, name, model, system_prompt, settings) "
                    "VALUES (gen_random_uuid(), 'Test Agent', 'gpt-4o', NULL, '{}'::jsonb) "
                    "RETURNING id"
                )
            ).scalar()
            for i in range(3):
                db.session.execute(
                    text(
                        "INSERT INTO ai.agent_sessions (id, session_id, agent_id) "
                        "VALUES (gen_random_uuid(), :session_id, :agent_id)"
                    ),
                    {"session_id": f"sess_sc_{i}_{str(agent_id)[:8]}", "agent_id": agent_id},
                )
            db.session.commit()
        resp = client.get("/api/agents?q=Test+Agent", headers=auth_headers)
        a = resp.get_json()["agents"][0]
        assert a["session_count"] == 3

    def test_response_includes_total_runs_and_last_run_at(
        self, client, mock_auth, auth_headers, app
    ):
        from agentic_project_service.db import db

        with app.app_context():
            agent_id = db.session.execute(
                text(
                    "INSERT INTO ai.agents (id, name, model, system_prompt, settings) "
                    "VALUES (gen_random_uuid(), 'Runs Agent', 'gpt-4o', NULL, '{}'::jsonb) "
                    "RETURNING id"
                )
            ).scalar()
            session_row_id = db.session.execute(
                text(
                    "INSERT INTO ai.agent_sessions (id, session_id, agent_id) "
                    "VALUES (gen_random_uuid(), :session_id, :agent_id) RETURNING id"
                ),
                {"session_id": f"sess_runs_{str(agent_id)[:8]}", "agent_id": agent_id},
            ).scalar()
            for i in range(4):
                db.session.execute(
                    text(
                        "INSERT INTO ai.agent_runs (id, run_id, agent_id, session_id, status, content) "
                        "VALUES (gen_random_uuid(), :run_id, :agent_id, :session_id, 'completed', 'x')"
                    ),
                    {
                        "run_id": f"run_ra_{i}_{str(agent_id)[:8]}",
                        "agent_id": agent_id,
                        "session_id": session_row_id,
                    },
                )
            db.session.commit()
        resp = client.get("/api/agents?q=Runs+Agent", headers=auth_headers)
        a = resp.get_json()["agents"][0]
        assert a["total_runs"] == 4
        assert a["last_run_at"] is not None

    def test_last_run_at_null_for_unused_agent(self, client, mock_auth, auth_headers, app):
        self._seed_agents(app, 1, name_prefix="Unused")
        resp = client.get("/api/agents?q=Unused", headers=auth_headers)
        a = resp.get_json()["agents"][0]
        assert a["last_run_at"] is None
        assert a["session_count"] == 0
        assert a["total_runs"] == 0

    def test_sort_by_last_run_at_nulls_last(self, client, mock_auth, auth_headers, app):
        # Create 3 agents — one with a run, two without
        from agentic_project_service.db import db

        with app.app_context():
            ids = []
            for name in ("A_sort", "B_sort", "C_sort"):
                aid = db.session.execute(
                    text(
                        "INSERT INTO ai.agents (id, name, model, system_prompt, settings) "
                        "VALUES (gen_random_uuid(), :name, 'gpt-4o', NULL, '{}'::jsonb) "
                        "RETURNING id"
                    ),
                    {"name": name},
                ).scalar()
                ids.append(aid)
            # Give B_sort a run
            session_row_id = db.session.execute(
                text(
                    "INSERT INTO ai.agent_sessions (id, session_id, agent_id) "
                    "VALUES (gen_random_uuid(), :session_id, :aid) RETURNING id"
                ),
                {"session_id": f"sess_sort_{str(ids[1])[:8]}", "aid": ids[1]},
            ).scalar()
            db.session.execute(
                text(
                    "INSERT INTO ai.agent_runs (id, run_id, agent_id, session_id, status, content) "
                    "VALUES (gen_random_uuid(), :run_id, :aid, :sid, 'completed', 'x')"
                ),
                {"run_id": f"run_sort_{str(ids[1])[:8]}", "aid": ids[1], "sid": session_row_id},
            )
            db.session.commit()
        resp = client.get("/api/agents?sort=last_run_at&order=desc&q=_sort", headers=auth_headers)
        names = [a["name"] for a in resp.get_json()["agents"]]
        # B_sort (has run) sorts first; A_sort and C_sort (NULL) sort last
        assert names[0] == "B_sort"
        assert set(names[1:]) == {"A_sort", "C_sort"}
