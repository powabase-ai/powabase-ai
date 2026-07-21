"""Tests for list_orchestrations() pagination, search/sort, and per-row aggregates."""

from sqlalchemy import text


class TestListOrchestrationsExtended:
    def _seed_orchs(self, app, count: int, name_prefix: str = "Orch"):
        from agentic_project_service.db import db

        with app.app_context():
            for i in range(count):
                db.session.execute(
                    text(
                        "INSERT INTO ai.orchestrations (id, name, description, strategy, settings) "
                        "VALUES (gen_random_uuid(), :name, NULL, 'supervisor', '{}'::jsonb)"
                    ),
                    {"name": f"{name_prefix} {i:03d}"},
                )
            db.session.commit()

    def test_envelope_unchanged(self, client, mock_auth, auth_headers, app):
        """Public API back-compat: { orchestrations: [...] } key preserved."""
        self._seed_orchs(app, 3)
        resp = client.get("/api/orchestrations", headers=auth_headers)
        data = resp.get_json()
        assert "orchestrations" in data
        assert isinstance(data["orchestrations"], list)

    def test_unparameterized_returns_all(self, client, mock_auth, auth_headers, app):
        """Back-compat: no params -> return every orchestration, no pagination keys."""
        self._seed_orchs(app, 75)
        resp = client.get("/api/orchestrations", headers=auth_headers)
        data = resp.get_json()
        assert len(data["orchestrations"]) == 75
        # When unparameterized, no pagination metadata should be added
        assert "total" not in data
        assert "limit" not in data
        assert "offset" not in data

    def test_explicit_limit_opts_into_pagination(self, client, mock_auth, auth_headers, app):
        self._seed_orchs(app, 75)
        resp = client.get("/api/orchestrations?limit=10", headers=auth_headers)
        data = resp.get_json()
        assert len(data["orchestrations"]) == 10
        assert data["total"] == 75
        assert data["limit"] == 10
        assert data["offset"] == 0

    def test_search_by_name(self, client, mock_auth, auth_headers, app):
        self._seed_orchs(app, 5, name_prefix="Foo")
        self._seed_orchs(app, 5, name_prefix="Bar")
        resp = client.get("/api/orchestrations?q=Foo&limit=20", headers=auth_headers)
        data = resp.get_json()
        assert data["total"] == 5
        assert all("Foo" in o["name"] for o in data["orchestrations"])

    def test_sort_by_name(self, client, mock_auth, auth_headers, app):
        self._seed_orchs(app, 5)
        resp = client.get("/api/orchestrations?sort=name&order=asc&limit=20", headers=auth_headers)
        names = [o["name"] for o in resp.get_json()["orchestrations"]]
        assert names == sorted(names)

    def test_sort_invalid_returns_400(self, client, mock_auth, auth_headers):
        resp = client.get("/api/orchestrations?sort=bogus", headers=auth_headers)
        assert resp.status_code == 400

    def test_response_includes_entity_count(self, client, mock_auth, auth_headers, app):
        from agentic_project_service.db import db

        with app.app_context():
            orch_id = db.session.execute(
                text(
                    "INSERT INTO ai.orchestrations (id, name, description, strategy, settings) "
                    "VALUES (gen_random_uuid(), 'With Entities', NULL, 'supervisor', '{}'::jsonb) "
                    "RETURNING id"
                )
            ).scalar()
            for _ in range(3):
                db.session.execute(
                    text(
                        "INSERT INTO ai.orchestration_entities "
                        "(id, orchestration_id, entity_type, entity_ref_id) "
                        "VALUES (gen_random_uuid(), :orch_id, 'agent', gen_random_uuid())"
                    ),
                    {"orch_id": orch_id},
                )
            db.session.commit()
        resp = client.get("/api/orchestrations", headers=auth_headers)
        o = next(x for x in resp.get_json()["orchestrations"] if x["name"] == "With Entities")
        assert o["entity_count"] == 3

    def test_response_includes_session_count_and_last_run_at(
        self, client, mock_auth, auth_headers, app
    ):
        from agentic_project_service.db import db

        with app.app_context():
            orch_id = db.session.execute(
                text(
                    "INSERT INTO ai.orchestrations (id, name, description, strategy, settings) "
                    "VALUES (gen_random_uuid(), 'With Sessions', NULL, 'supervisor', '{}'::jsonb) "
                    "RETURNING id"
                )
            ).scalar()
            session_id = db.session.execute(
                text(
                    "INSERT INTO ai.orchestration_sessions (id, orchestration_id, session_id) "
                    "VALUES (gen_random_uuid(), :orch_id, :session_id) RETURNING id"
                ),
                {"orch_id": orch_id, "session_id": "test-session-with-sessions"},
            ).scalar()
            db.session.execute(
                text(
                    "INSERT INTO ai.orchestration_runs (id, session_id, run_id, status) "
                    "VALUES (gen_random_uuid(), :session_id, :run_id, 'completed')"
                ),
                {"session_id": session_id, "run_id": "test-run-with-sessions"},
            )
            db.session.commit()
        resp = client.get("/api/orchestrations", headers=auth_headers)
        o = next(x for x in resp.get_json()["orchestrations"] if x["name"] == "With Sessions")
        assert o["session_count"] == 1
        assert o["last_run_at"] is not None

    def test_unused_orchestration_has_zero_aggregates(self, client, mock_auth, auth_headers, app):
        self._seed_orchs(app, 1, name_prefix="Unused")
        resp = client.get("/api/orchestrations", headers=auth_headers)
        o = next(x for x in resp.get_json()["orchestrations"] if "Unused" in x["name"])
        assert o["entity_count"] == 0
        assert o["session_count"] == 0
        assert o["last_run_at"] is None


class TestLoadSubAgentModels:
    """Direct DB-fixture coverage for ``routes/orchestrations._load_sub_agent_models``.

    The fail-fast tests in unit/test_byok_fail_fast.py (line 323-326, 371-374)
    ALL patch this helper via ``patch.object(orchestrations_route,
    "_load_sub_agent_models", return_value=[...])`` — so a typo'd
    ``entity.entity_type != "agent"`` filter, or a wrong column on the
    AgentModel join, would never be caught by those tests. This class seeds
    the real ORM tables and exercises the helper directly.
    """

    def test_returns_only_agent_entity_models_not_tools(self, app):
        """Seed an orchestration with one ``agent`` entity + one ``tool`` entity.

        Both entities' ``entity_ref_id`` columns point at real ``ai.agents``
        rows with distinct models. The tool entity's ``entity_ref_id``
        deliberately resolves to an agent row that DOES exist — so if the
        ``entity_type != 'agent'`` filter were removed, the helper would
        return BOTH models (2 items), not just the agent's (1 item). This
        is the load-bearing piece of the counterfactual: a naive seed where
        the tool entity points at a non-existent agent ID would silently
        pass the buggy version of the helper because ``AgentModel.get``
        returns None.

        Counterfactual: comment out the ``if entity.entity_type != 'agent':
        continue`` block in the helper — the test fails because the
        tool-entity branch now resolves to ``anthropic/claude-sonnet-4-6``
        and the assertion of a single-item list breaks.
        """
        import uuid

        from agentic_project_service.db import db
        from agentic_project_service.routes.orchestrations import _load_sub_agent_models

        orch_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        # tool_ref_agent_id is what the tool ENTITY points at via
        # entity_ref_id. We put it in ai.agents so a broken filter would
        # successfully resolve it through AgentModel and return its model.
        tool_ref_agent_id = uuid.uuid4()
        tool_id = uuid.uuid4()

        with app.app_context():
            db.session.execute(
                text(
                    "INSERT INTO ai.orchestrations (id, name, strategy, settings) "
                    "VALUES (:id, 'TestOrch', 'supervisor', '{}'::jsonb)"
                ),
                {"id": str(orch_id)},
            )
            # Sub-agent with a distinctive model string.
            db.session.execute(
                text(
                    "INSERT INTO ai.agents (id, name, model, system_prompt, settings) "
                    "VALUES (:id, 'SubAgent', 'openai/gpt-4o', '', '{}'::jsonb)"
                ),
                {"id": str(agent_id)},
            )
            # Plant an agent row at the tool entity's entity_ref_id, with a
            # DIFFERENT model — so removing the entity_type filter produces
            # a visible failure (2 models returned instead of 1).
            db.session.execute(
                text(
                    "INSERT INTO ai.agents (id, name, model, system_prompt, settings) "
                    "VALUES (:id, 'TrapAgent', 'anthropic/claude-sonnet-4-6', '', '{}'::jsonb)"
                ),
                {"id": str(tool_ref_agent_id)},
            )
            # A real tool in ai.tools (presence is not load-bearing here,
            # but keeps the seed shape realistic).
            db.session.execute(
                text(
                    "INSERT INTO ai.tools (id, name, description, type, input_schema, config) "
                    "VALUES (:id, 'SomeTool', 'a tool', 'mcp', '{}'::jsonb, '{}'::jsonb)"
                ),
                {"id": str(tool_id)},
            )
            # entity_type='agent' → points at the real sub-agent.
            db.session.execute(
                text(
                    "INSERT INTO ai.orchestration_entities "
                    "(id, orchestration_id, entity_type, entity_ref_id, position) "
                    "VALUES (gen_random_uuid(), :oid, 'agent', :eid, 0)"
                ),
                {"oid": str(orch_id), "eid": str(agent_id)},
            )
            # entity_type='tool' → entity_ref_id points at the trap agent
            # row, so a buggy filter that doesn't gate on entity_type would
            # return its model too.
            db.session.execute(
                text(
                    "INSERT INTO ai.orchestration_entities "
                    "(id, orchestration_id, entity_type, entity_ref_id, position) "
                    "VALUES (gen_random_uuid(), :oid, 'tool', :eid, 1)"
                ),
                {"oid": str(orch_id), "eid": str(tool_ref_agent_id)},
            )
            db.session.commit()

            models = _load_sub_agent_models(str(orch_id))

        # ONLY the agent-entity's model — the tool-entity is filtered out
        # even though its entity_ref_id resolves to a valid agent row.
        assert models == ["openai/gpt-4o"]
        assert "anthropic/claude-sonnet-4-6" not in models

    def test_returns_empty_for_orchestration_with_only_tool_entities(self, app):
        """Orchestration with zero agent entities → helper returns ``[]``.

        Edge case: if the filter were inverted (``entity_type == 'tool'``)
        the helper would return one model here; the empty-list assertion
        catches that.
        """
        import uuid

        from agentic_project_service.db import db
        from agentic_project_service.routes.orchestrations import _load_sub_agent_models

        orch_id = uuid.uuid4()
        # Plant an agent row that the tool entity's entity_ref_id resolves
        # to — so a buggy inverted filter would return its model.
        tool_ref_agent_id = uuid.uuid4()

        with app.app_context():
            db.session.execute(
                text(
                    "INSERT INTO ai.orchestrations (id, name, strategy, settings) "
                    "VALUES (:id, 'ToolOnlyOrch', 'supervisor', '{}'::jsonb)"
                ),
                {"id": str(orch_id)},
            )
            db.session.execute(
                text(
                    "INSERT INTO ai.agents (id, name, model, system_prompt, settings) "
                    "VALUES (:id, 'TrapAgent2', 'openrouter/some-model', '', '{}'::jsonb)"
                ),
                {"id": str(tool_ref_agent_id)},
            )
            db.session.execute(
                text(
                    "INSERT INTO ai.orchestration_entities "
                    "(id, orchestration_id, entity_type, entity_ref_id, position) "
                    "VALUES (gen_random_uuid(), :oid, 'tool', :eid, 0)"
                ),
                {"oid": str(orch_id), "eid": str(tool_ref_agent_id)},
            )
            db.session.commit()

            models = _load_sub_agent_models(str(orch_id))

        assert models == []
