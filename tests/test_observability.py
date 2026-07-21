"""Tests for the observability dashboard routes (C2.1)."""

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import text


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _insert_workflow(db, workflow_id: str) -> None:
    """workflow_executions.workflow_id is NOT NULL + FK'd to ai.workflows."""
    db.session.execute(
        text("INSERT INTO ai.workflows (id, name) VALUES (:id, 'wf')"),
        {"id": workflow_id},
    )


class TestAgentRunsEndpoint:
    def test_since_required(self, client, mock_auth, auth_headers):
        resp = client.get("/api/observability/agent-runs", headers=auth_headers)
        assert resp.status_code == 400

    def test_returns_rows_in_window(self, client, mock_auth, auth_headers, test_agent, app):
        from agentic_project_service.db import db

        agent_id = test_agent["id"]
        now = datetime.now(timezone.utc)
        with app.app_context():
            # In-window run.
            db.session.execute(
                text(
                    'INSERT INTO "ai".agent_runs '
                    "(id, run_id, agent_id, status, model, created_at, prompt_tokens, total_tokens) "
                    "VALUES (gen_random_uuid(), :rid, :aid, 'completed', 'gpt-5-mini', :ts, 10, 15)"
                ),
                {"rid": f"run-{uuid.uuid4()}", "aid": agent_id, "ts": now},
            )
            # Out-of-window run (2 days ago) — must be excluded by `since`.
            db.session.execute(
                text(
                    'INSERT INTO "ai".agent_runs '
                    "(id, run_id, agent_id, status, model, created_at) "
                    "VALUES (gen_random_uuid(), :rid, :aid, 'completed', 'gpt-5-mini', :ts)"
                ),
                {"rid": f"run-{uuid.uuid4()}", "aid": agent_id, "ts": now - timedelta(days=2)},
            )
            db.session.commit()

        since = _iso(now - timedelta(hours=1))
        resp = client.get(
            "/api/observability/agent-runs",
            query_string={"since": since},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["runs"]) == 1
        assert data["runs"][0]["model"] == "gpt-5-mini"
        assert data["runs"][0]["prompt_tokens"] == 10
        assert data["runs"][0]["total_tokens"] == 15
        assert data["truncated"] is False

    def test_filters_by_model_and_agent_id(self, client, mock_auth, auth_headers, test_agent, app):
        from agentic_project_service.db import db

        agent_id = test_agent["id"]
        other_agent_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        with app.app_context():
            db.session.execute(
                text("INSERT INTO ai.agents (id, name, model) VALUES (:id, 'other', 'gpt-4')"),
                {"id": other_agent_id},
            )
            for aid, model in ((agent_id, "gpt-5-mini"), (other_agent_id, "gpt-4")):
                db.session.execute(
                    text(
                        'INSERT INTO "ai".agent_runs '
                        "(id, run_id, agent_id, status, model, created_at) "
                        "VALUES (gen_random_uuid(), :rid, :aid, 'completed', :model, :ts)"
                    ),
                    {"rid": f"run-{uuid.uuid4()}", "aid": aid, "model": model, "ts": now},
                )
            db.session.commit()

        since = _iso(now - timedelta(hours=1))
        resp = client.get(
            "/api/observability/agent-runs",
            query_string={"since": since, "models": "gpt-5-mini"},
            headers=auth_headers,
        )
        runs = resp.get_json()["runs"]
        assert len(runs) == 1
        assert runs[0]["model"] == "gpt-5-mini"

        resp2 = client.get(
            "/api/observability/agent-runs",
            query_string={"since": since, "agent_ids": agent_id},
            headers=auth_headers,
        )
        runs2 = resp2.get_json()["runs"]
        assert len(runs2) == 1
        assert runs2[0]["agent_id"] == agent_id

    def test_truncated_flag_when_limit_hit(self, client, mock_auth, auth_headers, test_agent, app):
        from agentic_project_service.db import db

        agent_id = test_agent["id"]
        now = datetime.now(timezone.utc)
        with app.app_context():
            for _ in range(3):
                db.session.execute(
                    text(
                        'INSERT INTO "ai".agent_runs '
                        "(id, run_id, agent_id, status, created_at) "
                        "VALUES (gen_random_uuid(), :rid, :aid, 'completed', :ts)"
                    ),
                    {"rid": f"run-{uuid.uuid4()}", "aid": agent_id, "ts": now},
                )
            db.session.commit()

        since = _iso(now - timedelta(hours=1))
        resp = client.get(
            "/api/observability/agent-runs",
            query_string={"since": since, "limit": "2"},
            headers=auth_headers,
        )
        data = resp.get_json()
        assert len(data["runs"]) == 2
        assert data["truncated"] is True


class TestOrchestrationRunsAndWorkflowBlockLogsEndpoints:
    def test_orchestration_runs_since_required(self, client, mock_auth, auth_headers):
        resp = client.get("/api/observability/orchestration-runs", headers=auth_headers)
        assert resp.status_code == 400

    def test_orchestration_runs_returns_typed_columns(self, client, mock_auth, auth_headers, app):
        from agentic_project_service.db import db

        now = datetime.now(timezone.utc)
        with app.app_context():
            db.session.execute(
                text(
                    'INSERT INTO "ai".orchestration_runs '
                    "(id, run_id, status, model, created_at, total_tokens) "
                    "VALUES (gen_random_uuid(), :rid, 'completed', 'claude-opus', :ts, 99)"
                ),
                {"rid": f"orun-{uuid.uuid4()}", "ts": now},
            )
            db.session.commit()

        since = _iso(now - timedelta(hours=1))
        resp = client.get(
            "/api/observability/orchestration-runs",
            query_string={"since": since},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        runs = resp.get_json()["runs"]
        assert len(runs) == 1
        assert runs[0]["model"] == "claude-opus"
        assert runs[0]["total_tokens"] == 99

    def test_workflow_block_logs_only_returns_agent_blocks(
        self, client, mock_auth, auth_headers, app
    ):
        from agentic_project_service.db import db

        now = datetime.now(timezone.utc)
        workflow_id = str(uuid.uuid4())
        exec_id = str(uuid.uuid4())
        with app.app_context():
            _insert_workflow(db, workflow_id)
            db.session.execute(
                text(
                    "INSERT INTO ai.workflow_executions (id, workflow_id, status) "
                    "VALUES (:id, :wid, 'completed')"
                ),
                {"id": exec_id, "wid": workflow_id},
            )
            for block_type, model in (("agent", "gpt-5-mini"), ("condition", None)):
                db.session.execute(
                    text(
                        'INSERT INTO "ai".workflow_block_logs '
                        "(id, execution_id, block_id, block_type, status, execution_order, "
                        "model, created_at) "
                        "VALUES (gen_random_uuid(), :eid, gen_random_uuid(), :bt, 'success', 0, "
                        ":model, :ts)"
                    ),
                    {"eid": exec_id, "bt": block_type, "model": model, "ts": now},
                )
            db.session.commit()

        since = _iso(now - timedelta(hours=1))
        resp = client.get(
            "/api/observability/workflow-block-logs",
            query_string={"since": since},
            headers=auth_headers,
        )
        logs = resp.get_json()["logs"]
        assert len(logs) == 1
        assert logs[0]["block_type"] == "agent"
        assert logs[0]["model"] == "gpt-5-mini"


class TestToolCallsEndpoint:
    def test_since_required(self, client, mock_auth, auth_headers):
        resp = client.get("/api/observability/tool-calls", headers=auth_headers)
        assert resp.status_code == 400

    def test_returns_and_filters_events(self, client, mock_auth, auth_headers, test_agent, app):
        from agentic_project_service.db import db

        agent_id = test_agent["id"]
        now = datetime.now(timezone.utc)
        with app.app_context():
            for tool, status in (("web_search", "success"), ("knowledge_search", "error")):
                db.session.execute(
                    text(
                        'INSERT INTO "ai".tool_call_events '
                        "(id, agent_id, model, tool_name, status, duration_ms, occurred_at) "
                        "VALUES (gen_random_uuid(), :aid, 'gpt-5-mini', :tool, :status, 120, :ts)"
                    ),
                    {"aid": agent_id, "tool": tool, "status": status, "ts": now},
                )
            db.session.commit()

        since = _iso(now - timedelta(hours=1))
        resp = client.get(
            "/api/observability/tool-calls",
            query_string={"since": since},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        events = resp.get_json()["events"]
        assert len(events) == 2
        assert {e["tool_name"] for e in events} == {"web_search", "knowledge_search"}

        resp2 = client.get(
            "/api/observability/tool-calls",
            query_string={"since": since, "agent_ids": agent_id},
            headers=auth_headers,
        )
        assert len(resp2.get_json()["events"]) == 2


class TestExtractionStatusEndpoint:
    def test_returns_status_arrays(self, client, mock_auth, auth_headers, test_indexed_source):
        resp = client.get("/api/observability/extraction-status", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert any(s["extraction_status"] == "extracted" for s in data["sources"])
        assert any(i["index_status"] == "completed" for i in data["indexed_sources"])


class TestFilterOptionsEndpoint:
    def test_distinct_models_and_agents(self, client, mock_auth, auth_headers, test_agent, app):
        from agentic_project_service.db import db

        agent_id = test_agent["id"]
        now = datetime.now(timezone.utc)
        with app.app_context():
            # Two rows with the SAME model — must dedupe to one entry.
            for _ in range(2):
                db.session.execute(
                    text(
                        'INSERT INTO "ai".agent_runs '
                        "(id, run_id, agent_id, status, model, created_at) "
                        "VALUES (gen_random_uuid(), :rid, :aid, 'completed', 'gpt-5-mini', :ts)"
                    ),
                    {"rid": f"run-{uuid.uuid4()}", "aid": agent_id, "ts": now},
                )
            db.session.commit()

        resp = client.get("/api/observability/filter-options", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["models"] == ["gpt-5-mini"]
        assert any(a["id"] == agent_id for a in data["agents"])


class TestAgentsLookupEndpoint:
    def test_empty_ids_returns_empty(self, client, mock_auth, auth_headers):
        resp = client.get("/api/observability/agents-lookup", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["agents"] == []

    def test_resolves_names_for_given_ids(self, client, mock_auth, auth_headers, test_agent, app):
        from agentic_project_service.db import db

        other_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text("INSERT INTO ai.agents (id, name, model) VALUES (:id, 'unwanted', 'gpt-4')"),
                {"id": other_id},
            )
            db.session.commit()

        resp = client.get(
            "/api/observability/agents-lookup",
            query_string={"ids": test_agent["id"]},
            headers=auth_headers,
        )
        agents = resp.get_json()["agents"]
        assert len(agents) == 1
        assert agents[0]["id"] == test_agent["id"]
        assert agents[0]["name"] == test_agent["name"]


class TestHealthEndpoint:
    def test_zero_state(self, client, mock_auth, auth_headers):
        resp = client.get("/api/observability/health", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {
            "activeRuns": 0,
            "failedRuns24h": 0,
            "stuckExtractions": 0,
            "failedIndexedSources": 0,
            "runningWorkflows": 0,
        }

    def test_counts_active_and_failed_runs(self, client, mock_auth, auth_headers, test_agent, app):
        from agentic_project_service.db import db

        agent_id = test_agent["id"]
        now = datetime.now(timezone.utc)
        with app.app_context():
            db.session.execute(
                text(
                    'INSERT INTO "ai".agent_runs (id, run_id, agent_id, status, created_at) '
                    "VALUES (gen_random_uuid(), :rid, :aid, 'running', :ts)"
                ),
                {"rid": f"run-{uuid.uuid4()}", "aid": agent_id, "ts": now},
            )
            db.session.execute(
                text(
                    'INSERT INTO "ai".agent_runs (id, run_id, agent_id, status, created_at) '
                    "VALUES (gen_random_uuid(), :rid, :aid, 'failed', :ts)"
                ),
                {"rid": f"run-{uuid.uuid4()}", "aid": agent_id, "ts": now},
            )
            # Failed run OUTSIDE the 24h window — must not count.
            db.session.execute(
                text(
                    'INSERT INTO "ai".agent_runs (id, run_id, agent_id, status, created_at) '
                    "VALUES (gen_random_uuid(), :rid, :aid, 'failed', :ts)"
                ),
                {
                    "rid": f"run-{uuid.uuid4()}",
                    "aid": agent_id,
                    "ts": now - timedelta(hours=25),
                },
            )
            db.session.commit()

        resp = client.get("/api/observability/health", headers=auth_headers)
        data = resp.get_json()
        assert data["activeRuns"] == 1
        assert data["failedRuns24h"] == 1

    def test_counts_stuck_extraction(self, client, mock_auth, auth_headers, app):
        from agentic_project_service.db import db

        with app.app_context():
            # Stuck: extracting, last updated 15 minutes ago (> 10min threshold).
            db.session.execute(
                text(
                    'INSERT INTO "ai".sources '
                    "(id, name, file_type, storage_path, extraction_status, updated_at) "
                    "VALUES (gen_random_uuid(), 'stuck.pdf', 'application/pdf', 'sources/x', "
                    "'extracting', NOW() - INTERVAL '15 minutes')"
                )
            )
            # Not stuck: extracting but recently touched.
            db.session.execute(
                text(
                    'INSERT INTO "ai".sources '
                    "(id, name, file_type, storage_path, extraction_status, updated_at) "
                    "VALUES (gen_random_uuid(), 'fresh.pdf', 'application/pdf', 'sources/y', "
                    "'extracting', NOW())"
                )
            )
            db.session.commit()

        resp = client.get("/api/observability/health", headers=auth_headers)
        assert resp.get_json()["stuckExtractions"] == 1

    def test_counts_failed_indexed_sources_and_running_workflows(
        self, client, mock_auth, auth_headers, test_indexed_source, app
    ):
        from agentic_project_service.db import db

        workflow_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text("UPDATE \"ai\".indexed_sources SET index_status = 'failed' WHERE id = :id"),
                {"id": test_indexed_source["id"]},
            )
            # Stuck workflow: running, started 10 minutes ago (> 5min threshold).
            _insert_workflow(db, workflow_id)
            db.session.execute(
                text(
                    "INSERT INTO ai.workflow_executions (id, workflow_id, status, started_at) "
                    "VALUES (gen_random_uuid(), :wid, 'running', NOW() - INTERVAL '10 minutes')"
                ),
                {"wid": workflow_id},
            )
            db.session.commit()

        resp = client.get("/api/observability/health", headers=auth_headers)
        data = resp.get_json()
        assert data["failedIndexedSources"] == 1
        assert data["runningWorkflows"] == 1


class TestAuthRequired:
    def test_no_token(self, client):
        resp = client.get("/api/observability/health")
        assert resp.status_code == 401
