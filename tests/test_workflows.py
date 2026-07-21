"""Tests for workflow CRUD routes."""

import pytest
from sqlalchemy import text


@pytest.fixture(autouse=True)
def _truncate_workflows(app):
    """Clean ai.workflows between tests (conftest's autouse omits it)."""
    from agentic_project_service.db import db
    yield
    with app.app_context():
        # workflow_executions is in conftest's autouse cleanup; workflows isn't.
        # CASCADE handles workflow_blocks/workflow_edges/copilot_sessions.
        db.session.execute(text('TRUNCATE "ai".workflows CASCADE'))
        db.session.commit()


@pytest.fixture(scope="module", autouse=True)
def _ensure_workflows_schema(app):
    """Add columns + executions table the ORM stub omits.

    The Workflow / WorkflowExecution ORM classes in models/tenant.py are FK-resolution
    stubs only (id column). Routes use raw SQL against the full ai_schema.sql
    schema. Apply the columns + ai.workflow_executions table this module needs.
    """
    from agentic_project_service.db import db

    with app.app_context():
        db.session.execute(
            text(
                """
                ALTER TABLE ai.workflows ADD COLUMN IF NOT EXISTS name VARCHAR(255);
                ALTER TABLE ai.workflows ADD COLUMN IF NOT EXISTS description TEXT;
                ALTER TABLE ai.workflows ADD COLUMN IF NOT EXISTS variables JSONB DEFAULT '{}';
                ALTER TABLE ai.workflows ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1;
                ALTER TABLE ai.workflows ADD COLUMN IF NOT EXISTS color VARCHAR(50);
                ALTER TABLE ai.workflows ADD COLUMN IF NOT EXISTS state VARCHAR(20) DEFAULT 'internal';
                ALTER TABLE ai.workflows ADD COLUMN IF NOT EXISTS schedule_config JSONB DEFAULT NULL;
                ALTER TABLE ai.workflows ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
                ALTER TABLE ai.workflows ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
                """
            )
        )
        db.session.execute(
            text(
                """
                ALTER TABLE ai.workflow_executions ADD COLUMN IF NOT EXISTS workflow_id UUID;
                ALTER TABLE ai.workflow_executions ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'pending';
                ALTER TABLE ai.workflow_executions ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
                """
            )
        )
        db.session.commit()
    yield


class TestListWorkflowsExtended:
    def _seed_workflows(self, app, count: int, name_prefix: str = "WF"):
        from agentic_project_service.db import db
        with app.app_context():
            for i in range(count):
                db.session.execute(
                    text(
                        "INSERT INTO ai.workflows (id, name, description, variables, version, state) "
                        "VALUES (gen_random_uuid(), :name, NULL, '{}'::jsonb, 1, 'internal')"
                    ),
                    {"name": f"{name_prefix} {i:03d}"},
                )
            db.session.commit()

    def test_envelope_unchanged(self, client, mock_auth, auth_headers, app):
        """Public envelope { workflows, total, limit, offset } preserved."""
        self._seed_workflows(app, 3)
        resp = client.get("/api/workflows", headers=auth_headers)
        data = resp.get_json()
        assert set(data.keys()) >= {"workflows", "total", "limit", "offset"}
        assert isinstance(data["workflows"], list)

    def test_search_by_name(self, client, mock_auth, auth_headers, app):
        self._seed_workflows(app, 5, name_prefix="Foo")
        self._seed_workflows(app, 5, name_prefix="Bar")
        resp = client.get("/api/workflows?q=Foo", headers=auth_headers)
        data = resp.get_json()
        assert data["total"] == 5
        assert all("Foo" in w["name"] for w in data["workflows"])

    def test_sort_by_name(self, client, mock_auth, auth_headers, app):
        self._seed_workflows(app, 5)
        resp = client.get("/api/workflows?sort=name&order=asc", headers=auth_headers)
        names = [w["name"] for w in resp.get_json()["workflows"]]
        assert names == sorted(names)

    def test_sort_invalid_returns_400(self, client, mock_auth, auth_headers):
        resp = client.get("/api/workflows?sort=bogus", headers=auth_headers)
        assert resp.status_code == 400

    def test_response_includes_execution_count(self, client, mock_auth, auth_headers, app):
        from agentic_project_service.db import db
        with app.app_context():
            wf_id = db.session.execute(
                text(
                    "INSERT INTO ai.workflows (id, name, variables, version, state) "
                    "VALUES (gen_random_uuid(), 'WF with runs', '{}'::jsonb, 1, 'internal') "
                    "RETURNING id"
                )
            ).scalar()
            for _ in range(3):
                db.session.execute(
                    text(
                        "INSERT INTO ai.workflow_executions (id, workflow_id, status) "
                        "VALUES (gen_random_uuid(), :wf_id, 'completed')"
                    ),
                    {"wf_id": wf_id},
                )
            db.session.commit()
        resp = client.get("/api/workflows?q=WF with runs", headers=auth_headers)
        w = resp.get_json()["workflows"][0]
        assert w["execution_count"] == 3
        assert w["last_execution_at"] is not None

    def test_unused_workflow_has_zero_aggregates(self, client, mock_auth, auth_headers, app):
        self._seed_workflows(app, 1, name_prefix="Unused")
        resp = client.get("/api/workflows?q=Unused", headers=auth_headers)
        w = resp.get_json()["workflows"][0]
        assert w["execution_count"] == 0
        assert w["last_execution_at"] is None

    def test_sort_by_last_execution_at_nulls_last(self, client, mock_auth, auth_headers, app):
        from agentic_project_service.db import db
        with app.app_context():
            ids = []
            for name in ("A_wfsort", "B_wfsort", "C_wfsort"):
                wid = db.session.execute(
                    text(
                        "INSERT INTO ai.workflows (id, name, variables, version, state) "
                        "VALUES (gen_random_uuid(), :name, '{}'::jsonb, 1, 'internal') "
                        "RETURNING id"
                    ),
                    {"name": name},
                ).scalar()
                ids.append(wid)
            # Give B_wfsort an execution
            db.session.execute(
                text(
                    "INSERT INTO ai.workflow_executions (id, workflow_id, status) "
                    "VALUES (gen_random_uuid(), :wid, 'completed')"
                ),
                {"wid": ids[1]},
            )
            db.session.commit()
        resp = client.get("/api/workflows?sort=last_execution_at&order=desc&q=_wfsort", headers=auth_headers)
        names = [w["name"] for w in resp.get_json()["workflows"]]
        assert names[0] == "B_wfsort"
        assert set(names[1:]) == {"A_wfsort", "C_wfsort"}
