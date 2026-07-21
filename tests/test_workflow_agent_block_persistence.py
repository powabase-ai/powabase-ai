"""Integration test: AgentBlock executions persist as agent_runs rows.

Task 5b — verifies that when a workflow with an AgentBlock runs, the
block's execution is persisted as an ai.agent_runs row with
parent_workflow_execution_id set, and the corresponding
workflow_block_logs row has agent_run_id pointing to that agent_runs row.
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text

from agentic_project_service.db import db


# ---------------------------------------------------------------------------
# DB setup helpers
# ---------------------------------------------------------------------------


def _insert_workflow(app, name="Test WF"):
    wf_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".workflows (id, name, description, variables)
                VALUES (:id, :name, '', '{}')
                """
            ),
            {"id": wf_id, "name": name},
        )
        db.session.commit()
    return wf_id


def _insert_blocks(app, wf_id, blocks):
    """Insert workflow_blocks rows. blocks is a list of dicts with id/type/name/config."""
    with app.app_context():
        for b in blocks:
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".workflow_blocks
                        (id, workflow_id, type, name, position_x, position_y, config, enabled)
                    VALUES
                        (:id, :wid, :type, :name, 0, 0, CAST(:config AS jsonb), true)
                    """
                ),
                {
                    "id": b["id"],
                    "wid": wf_id,
                    "type": b["type"],
                    "name": b.get("name", b["type"]),
                    "config": json.dumps(b.get("config", {})),
                },
            )
        db.session.commit()


def _insert_edge(app, wf_id, source_id, target_id):
    edge_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".workflow_edges
                    (id, workflow_id, source_block_id, target_block_id,
                     source_handle, target_handle)
                VALUES
                    (:id, :wid, :src, :tgt, 'output', 'input')
                """
            ),
            {"id": edge_id, "wid": wf_id, "src": source_id, "tgt": target_id},
        )
        db.session.commit()


def _make_fake_agent_output(content="agent said hi"):
    """Build a MagicMock that looks like AgentOutput."""
    from agentic.execution.status import ExecutionStatus

    fake = MagicMock()
    fake.content = content
    fake.status = ExecutionStatus.COMPLETED
    fake.error = None
    fake.usage = {"total_tokens": 7, "prompt_tokens": 5, "completion_tokens": 2}
    fake.steps = 1
    fake.events = []
    fake.tool_calls = []
    fake.reasoning_steps = []
    fake.messages = []
    fake.started_at = None
    fake.completed_at = None
    return fake


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_agent_block_persists_agent_run(client, app, mock_auth, auth_headers):
    """Executing a workflow with one AgentBlock creates:
    - an ai.agent_runs row with parent_workflow_execution_id = execution_id,
      session_id IS NULL, and content = 'agent said hi'.
    - a workflow_block_logs row (block_type = 'agent') with agent_run_id pointing
      to that agent_runs row.
    """
    wf_id = _insert_workflow(app)
    starter_id = str(uuid.uuid4())
    agent_block_id = str(uuid.uuid4())

    _insert_blocks(
        app,
        wf_id,
        [
            {"id": starter_id, "type": "starter", "name": "Start", "config": {}},
            {
                "id": agent_block_id,
                "type": "agent",
                "name": "My Agent",
                "config": {"model": "gpt-4o-mini", "prompt": "say hi"},
            },
        ],
    )
    _insert_edge(app, wf_id, starter_id, agent_block_id)

    fake_output = _make_fake_agent_output("agent said hi")

    with patch("agentic.agent.agent.Agent.arun", new_callable=AsyncMock, return_value=fake_output):
        resp = client.post(
            f"/api/workflows/{wf_id}/execute",
            json={"variables": {}},
            headers=auth_headers,
        )

    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    execution_id = data["execution_id"]

    with app.app_context():
        # 1. Exactly one agent_runs row with parent_workflow_execution_id
        agent_runs = db.session.execute(
            text(
                "SELECT id, session_id, content, parent_workflow_execution_id "
                'FROM "ai".agent_runs '
                "WHERE parent_workflow_execution_id = :exec_id"
            ),
            {"exec_id": execution_id},
        ).fetchall()

        assert len(agent_runs) == 1, (
            f"Expected 1 agent_runs row with parent_workflow_execution_id={execution_id}, "
            f"got {len(agent_runs)}"
        )
        ar = agent_runs[0]
        assert ar[1] is None, "session_id should be NULL for workflow AgentBlock runs"
        assert ar[2] == "agent said hi", f"Expected content='agent said hi', got '{ar[2]}'"
        assert str(ar[3]) == execution_id

        agent_run_uuid = str(ar[0])

        # 2. workflow_block_logs row for block_type='agent' has agent_run_id set
        log_row = db.session.execute(
            text(
                "SELECT block_type, agent_run_id "
                'FROM "ai".workflow_block_logs '
                "WHERE execution_id = :exec_id AND block_type = 'agent'"
            ),
            {"exec_id": execution_id},
        ).fetchone()

        assert log_row is not None, "Expected a workflow_block_logs row with block_type='agent'"
        assert str(log_row[1]) == agent_run_uuid, (
            f"workflow_block_logs.agent_run_id should be {agent_run_uuid}, got {log_row[1]}"
        )


@pytest.mark.integration
def test_execution_logs_endpoint_includes_agent_run_id(client, app, mock_auth, auth_headers):
    """GET /api/workflows/<wf_id>/executions/<exec_id>/logs returns agent_run_id on AgentBlock rows."""
    wf_id = _insert_workflow(app, "Test WF Logs")
    starter_id = str(uuid.uuid4())
    agent_block_id = str(uuid.uuid4())

    _insert_blocks(
        app,
        wf_id,
        [
            {"id": starter_id, "type": "starter", "name": "Start", "config": {}},
            {
                "id": agent_block_id,
                "type": "agent",
                "name": "Log Agent",
                "config": {"model": "gpt-4o-mini", "prompt": "say hello"},
            },
        ],
    )
    _insert_edge(app, wf_id, starter_id, agent_block_id)

    fake_output = _make_fake_agent_output("hello from agent")

    with patch("agentic.agent.agent.Agent.arun", new_callable=AsyncMock, return_value=fake_output):
        exec_resp = client.post(
            f"/api/workflows/{wf_id}/execute",
            json={"variables": {}},
            headers=auth_headers,
        )

    assert exec_resp.status_code == 200, exec_resp.get_data(as_text=True)
    execution_id = exec_resp.get_json()["execution_id"]

    logs_resp = client.get(
        f"/api/workflows/{wf_id}/executions/{execution_id}/logs",
        headers=auth_headers,
    )
    assert logs_resp.status_code == 200, logs_resp.get_data(as_text=True)
    logs_data = logs_resp.get_json()

    agent_logs = [b for b in logs_data["block_logs"] if b["block_type"] == "agent"]
    assert len(agent_logs) == 1, f"Expected 1 agent block log, got {len(agent_logs)}"

    agent_log = agent_logs[0]
    assert "agent_run_id" in agent_log, "agent_run_id key missing from block_logs entry"
    assert agent_log["agent_run_id"] is not None, "agent_run_id should not be None for AgentBlock"

    # Cross-check: agent_run_id in the log matches the actual agent_runs row
    with app.app_context():
        ar = db.session.execute(
            text('SELECT id FROM "ai".agent_runs WHERE parent_workflow_execution_id = :exec_id'),
            {"exec_id": execution_id},
        ).fetchone()
        assert ar is not None, "Expected agent_runs row"
        assert agent_log["agent_run_id"] == str(ar[0])


@pytest.mark.integration
def test_single_agent_run_endpoint_workflow(client, app, mock_auth, auth_headers):
    """GET /api/agents/runs/<run_id> returns the run with parent_workflow_execution_id set."""
    wf_id = _insert_workflow(app, "Test WF Single Run")
    starter_id = str(uuid.uuid4())
    agent_block_id = str(uuid.uuid4())

    _insert_blocks(
        app,
        wf_id,
        [
            {"id": starter_id, "type": "starter", "name": "Start", "config": {}},
            {
                "id": agent_block_id,
                "type": "agent",
                "name": "Fetch Agent",
                "config": {"model": "gpt-4o-mini", "prompt": "answer"},
            },
        ],
    )
    _insert_edge(app, wf_id, starter_id, agent_block_id)

    fake_output = _make_fake_agent_output("agent answer")

    with patch("agentic.agent.agent.Agent.arun", new_callable=AsyncMock, return_value=fake_output):
        exec_resp = client.post(
            f"/api/workflows/{wf_id}/execute",
            json={"variables": {}},
            headers=auth_headers,
        )

    assert exec_resp.status_code == 200
    execution_id = exec_resp.get_json()["execution_id"]

    # Get run_id from DB
    with app.app_context():
        ar = db.session.execute(
            text(
                'SELECT id, run_id FROM "ai".agent_runs '
                "WHERE parent_workflow_execution_id = :exec_id"
            ),
            {"exec_id": execution_id},
        ).fetchone()
        assert ar is not None, "Expected agent_runs row"
        run_id = ar[1]

    run_resp = client.get(f"/api/agents/runs/{run_id}", headers=auth_headers)
    assert run_resp.status_code == 200, run_resp.get_data(as_text=True)
    run_data = run_resp.get_json()

    assert run_data["run_id"] == run_id
    assert run_data["content"] == "agent answer"
    assert run_data["parent_workflow_execution_id"] == execution_id
    assert run_data["parent_orchestration_run_id"] is None
    assert run_data["session_id"] is None


@pytest.mark.integration
def test_single_agent_run_endpoint_not_found(client, app, mock_auth, auth_headers):
    """GET /api/agents/runs/<run_id> returns 404 for unknown run_id."""
    resp = client.get("/api/agents/runs/nonexistent-run-id-xyz", headers=auth_headers)
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "Run not found"


@pytest.mark.integration
def test_failed_agent_block_persists_with_failed_status_and_error(
    client, app, mock_auth, auth_headers
):
    """When Agent.arun returns AgentOutput(status=FAILED, error=<msg>), the
    resulting agent_runs row must be persisted with status='failed' and the
    error string — NOT silently labelled 'completed'.
    """
    from agentic.execution.status import ExecutionStatus

    wf_id = _insert_workflow(app, "Failed-Status WF")
    starter_id = str(uuid.uuid4())
    agent_block_id = str(uuid.uuid4())
    _insert_blocks(
        app,
        wf_id,
        [
            {"id": starter_id, "type": "starter", "name": "Start", "config": {}},
            {
                "id": agent_block_id,
                "type": "agent",
                "name": "Bad Agent",
                "config": {"model": "gpt-4o-mini", "prompt": "will fail"},
            },
        ],
    )
    _insert_edge(app, wf_id, starter_id, agent_block_id)

    failed_output = MagicMock()
    failed_output.content = None
    failed_output.status = ExecutionStatus.FAILED
    failed_output.error = "LLM unavailable"
    failed_output.usage = {}
    failed_output.steps = 0
    failed_output.events = []
    failed_output.tool_calls = []
    failed_output.reasoning_steps = []
    failed_output.messages = []
    failed_output.started_at = None
    failed_output.completed_at = None

    with patch(
        "agentic.agent.agent.Agent.arun", new_callable=AsyncMock, return_value=failed_output
    ):
        resp = client.post(
            f"/api/workflows/{wf_id}/execute",
            json={"variables": {}},
            headers=auth_headers,
        )

    assert resp.status_code == 200, resp.get_data(as_text=True)
    execution_id = resp.get_json()["execution_id"]

    with app.app_context():
        row = db.session.execute(
            text(
                'SELECT status, error, content FROM "ai".agent_runs '
                "WHERE parent_workflow_execution_id = :eid"
            ),
            {"eid": execution_id},
        ).fetchone()
        assert row is not None, "Expected an agent_runs row even on failure"
        assert row[0] == "failed", f"Expected status='failed', got '{row[0]}'"
        assert row[1] == "LLM unavailable"
        assert row[2] is None  # content was None in the failed output


@pytest.mark.integration
def test_single_agent_run_endpoint_denies_cross_user(client, app, mock_auth, auth_headers):
    """GET /api/agents/runs/<run_id> returns 404 when a run belongs to a session
    owned by a different user (404 not 403, to avoid leaking existence).

    mock_auth fixture authenticates the caller as some user U. We create a
    session owned by a DIFFERENT user, put a run in it, and expect 404.
    """
    other_user_id = str(uuid.uuid4())
    assert other_user_id != mock_auth  # sanity: fixture users differ

    agent_id = str(uuid.uuid4())
    session_db_id = str(uuid.uuid4())
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    with app.app_context():
        db.session.execute(
            text("INSERT INTO \"ai\".agents (id, name, model) VALUES (:id, 'test', 'gpt-4o-mini')"),
            {"id": agent_id},
        )
        db.session.execute(
            text(
                'INSERT INTO "ai".agent_sessions (id, session_id, agent_id, user_id) '
                "VALUES (:id, :sid, :aid, :uid)"
            ),
            {"id": session_db_id, "sid": session_id, "aid": agent_id, "uid": other_user_id},
        )
        db.session.execute(
            text(
                'INSERT INTO "ai".agent_runs (id, session_id, run_id, status) '
                "VALUES (:id, :sid, :rid, 'completed')"
            ),
            {"id": str(uuid.uuid4()), "sid": session_db_id, "rid": run_id},
        )
        db.session.commit()

    resp = client.get(f"/api/agents/runs/{run_id}", headers=auth_headers)
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "Run not found"


@pytest.mark.integration
def test_single_agent_run_endpoint_allows_session_owner(client, app, mock_auth, auth_headers):
    """Conversely, a session owned by the calling user IS accessible.

    Guards against over-scoping: the ownership check must not false-positive
    and deny legitimate same-user reads.
    """
    agent_id = str(uuid.uuid4())
    session_db_id = str(uuid.uuid4())
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    with app.app_context():
        db.session.execute(
            text("INSERT INTO \"ai\".agents (id, name, model) VALUES (:id, 'test', 'gpt-4o-mini')"),
            {"id": agent_id},
        )
        db.session.execute(
            text(
                'INSERT INTO "ai".agent_sessions (id, session_id, agent_id, user_id) '
                "VALUES (:id, :sid, :aid, :uid)"
            ),
            {"id": session_db_id, "sid": session_id, "aid": agent_id, "uid": mock_auth},
        )
        db.session.execute(
            text(
                'INSERT INTO "ai".agent_runs (id, session_id, run_id, status) '
                "VALUES (:id, :sid, :rid, 'completed')"
            ),
            {"id": str(uuid.uuid4()), "sid": session_db_id, "rid": run_id},
        )
        db.session.commit()

    resp = client.get(f"/api/agents/runs/{run_id}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.get_json()["run_id"] == run_id


@pytest.mark.integration
def test_multi_agent_block_workflow_persists_each_distinctly(client, app, mock_auth, auth_headers):
    """Workflow with 2 AgentBlocks must produce 2 distinct agent_runs rows,
    each linked to the correct workflow_block_logs row (no id swapping).

    Structure: starter → agent_block_a → agent_block_b
    """
    wf_id = _insert_workflow(app, "Multi-Agent WF")
    starter_id = str(uuid.uuid4())
    block_a_id = str(uuid.uuid4())
    block_b_id = str(uuid.uuid4())

    _insert_blocks(
        app,
        wf_id,
        [
            {"id": starter_id, "type": "starter", "name": "Start", "config": {}},
            {
                "id": block_a_id,
                "type": "agent",
                "name": "Agent A",
                "config": {"model": "gpt-4o-mini", "prompt": "say alpha", "block_id": block_a_id},
            },
            {
                "id": block_b_id,
                "type": "agent",
                "name": "Agent B",
                "config": {"model": "gpt-4o-mini", "prompt": "say beta", "block_id": block_b_id},
            },
        ],
    )
    _insert_edge(app, wf_id, starter_id, block_a_id)
    _insert_edge(app, wf_id, block_a_id, block_b_id)

    call_count = {"n": 0}

    async def fake_arun(self, prompt, **kwargs):
        call_count["n"] += 1
        return _make_fake_agent_output(f"output_{call_count['n']}")

    with patch("agentic.agent.agent.Agent.arun", new=fake_arun):
        resp = client.post(
            f"/api/workflows/{wf_id}/execute",
            json={"variables": {}},
            headers=auth_headers,
        )

    assert resp.status_code == 200, resp.get_data(as_text=True)
    execution_id = resp.get_json()["execution_id"]

    with app.app_context():
        # 1. Exactly 2 agent_runs rows
        agent_runs = db.session.execute(
            text(
                "SELECT id, run_id "
                'FROM "ai".agent_runs '
                "WHERE parent_workflow_execution_id = :exec_id "
                "ORDER BY created_at ASC"
            ),
            {"exec_id": execution_id},
        ).fetchall()

        assert len(agent_runs) == 2, f"Expected 2 agent_runs rows, got {len(agent_runs)}"

        run_uuid_a = str(agent_runs[0][0])
        run_uuid_b = str(agent_runs[1][0])
        assert run_uuid_a != run_uuid_b, "The two agent_runs must have distinct UUIDs"

        # 2. 2 workflow_block_logs rows of block_type='agent'
        block_logs = db.session.execute(
            text(
                "SELECT block_id, agent_run_id "
                'FROM "ai".workflow_block_logs '
                "WHERE execution_id = :exec_id AND block_type = 'agent' "
                "ORDER BY execution_order ASC"
            ),
            {"exec_id": execution_id},
        ).fetchall()

        assert len(block_logs) == 2, f"Expected 2 agent block_logs rows, got {len(block_logs)}"

        # 3. Each block_log.agent_run_id points to a DIFFERENT agent_runs row
        log_run_ids = {str(row[1]) for row in block_logs}
        assert len(log_run_ids) == 2, (
            f"Both block_logs rows must point to different agent_runs, got ids: {log_run_ids}"
        )

        # 4. The full set of agent_run_ids in logs matches the actual agent_runs rows
        assert log_run_ids == {run_uuid_a, run_uuid_b}, (
            f"block_logs agent_run_ids {log_run_ids} don't match "
            f"agent_runs ids {{{run_uuid_a}, {run_uuid_b}}}"
        )

        # 5. The block_id column maps each log to the correct block (no id swap)
        block_log_map = {str(row[0]): str(row[1]) for row in block_logs}
        if block_a_id in block_log_map and block_b_id in block_log_map:
            assert block_log_map[block_a_id] != block_log_map[block_b_id], (
                "block_a and block_b must link to different agent_runs (ids should not be swapped)"
            )
