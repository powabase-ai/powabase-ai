"""Integration tests: orchestration delegated agents persist as child agent_runs."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from agentic_project_service.db import db


def _create_agent(app, name="Specialist"):
    """Insert an ai.agents row and return its id."""
    agent_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".agents (id, name, model, system_prompt)
                VALUES (:id, :name, 'gpt-4o-mini', 'You are helpful.')
                """
            ),
            {"id": agent_id, "name": name},
        )
        db.session.commit()
    return agent_id


def _create_orchestration(app, strategy="supervisor", agent_ids=None):
    """Insert an orchestration + entity rows, return orchestration id."""
    orch_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".orchestrations
                    (id, name, description, strategy, orchestrator_config, settings)
                VALUES (:id, 'test-orch', 'desc', :strategy, '{}', '{}')
                """
            ),
            {"id": orch_id, "strategy": strategy},
        )
        for i, aid in enumerate(agent_ids or []):
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".orchestration_entities
                        (id, orchestration_id, entity_type, entity_ref_id,
                         role_description, position, config)
                    VALUES (:eid, :orch, 'agent', :aid, 'role', :pos, '{}')
                    """
                ),
                {
                    "eid": str(uuid.uuid4()),
                    "orch": orch_id,
                    "aid": aid,
                    "pos": i,
                },
            )
        db.session.commit()
    return orch_id


def _make_fake_agent_output(content="delegation result"):
    """Build a MagicMock that looks like an AgentOutput."""
    from agentic.execution.status import ExecutionStatus

    fake_output = MagicMock()
    fake_output.content = content
    fake_output.status = ExecutionStatus.COMPLETED
    fake_output.error = None
    fake_output.usage = {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}
    fake_output.steps = 1
    fake_output.events = []
    fake_output.tool_calls = []
    fake_output.reasoning_steps = []
    fake_output.messages = []
    fake_output.started_at = None
    fake_output.completed_at = None
    return fake_output


@pytest.mark.integration
def test_supervisor_delegates_persist_as_child_agent_runs(client, app, mock_auth, auth_headers):
    """After a supervisor orchestration runs, child agent_runs exist with parent_orchestration_run_id.

    The supervisor strategy delegates via DelegateTool, which fires on_run_complete inside
    Agent.run's ReAct loop.  To avoid a real LLM call we patch Orchestration.run to directly
    invoke the on_delegate_complete callback (simulating what DelegateTool would do) and return
    a valid OrchestrationOutput.  This tests that the route (a) passes the callback, and
    (b) the callback correctly persists an agent_runs row.
    """
    from agentic.execution.status import ExecutionStatus
    from agentic.orchestration.output import OrchestrationOutput

    agent_id = _create_agent(app, "Alpha")
    orch_id = _create_orchestration(app, "supervisor", agent_ids=[agent_id])

    child_execution_id = str(uuid.uuid4())

    def fake_orchestration_run(input, context=None, *, history=None, on_delegate_complete=None):
        """Simulate a supervisor run that delegates once to 'Alpha' agent."""
        # Fire the hook as DelegateTool would, using the orchestration_run_id from context
        if on_delegate_complete is not None:
            on_delegate_complete(
                {
                    "agent_name": "Alpha",
                    "child_execution_id": child_execution_id,
                    "orchestration_run_id": context.orchestration_run_id if context else None,
                    "task": input if isinstance(input, str) else "task",
                    "content": "delegation result",
                    "status": ExecutionStatus.COMPLETED,
                    "error": None,
                    "usage": {"total_tokens": 10},
                    "steps": 1,
                    "events": [],
                    "tool_calls": [],
                    "messages": [],
                    "started_at": None,
                    "completed_at": None,
                }
            )
        out = OrchestrationOutput(execution_id=context.execution_id if context else "test")
        out.status = ExecutionStatus.COMPLETED
        out.content = "done"
        out.steps = 1
        out.usage = {"total_tokens": 10}
        out.events = []
        return out

    with patch(
        "agentic.orchestration.orchestration.Orchestration.run",
        side_effect=fake_orchestration_run,
    ):
        # buffered=True forces the test client to fully consume the SSE generator
        # before returning, preventing GeneratorExit from aborting the stream early.
        resp = client.post(
            f"/api/orchestrations/{orch_id}/run/stream",
            json={"message": "do it"},
            headers=auth_headers,
            buffered=True,
        )
        assert resp.status_code == 200

    # Verify: an orchestration_runs row was created
    with app.app_context():
        orch_run = db.session.execute(
            text('SELECT id FROM "ai".orchestration_runs WHERE orchestration_id = :oid'),
            {"oid": orch_id},
        ).fetchone()
        assert orch_run is not None, "Expected an orchestration_runs row"

        # Verify: at least one agent_runs row with parent_orchestration_run_id set
        child_runs = db.session.execute(
            text(
                "SELECT parent_orchestration_run_id, session_id, content "
                'FROM "ai".agent_runs WHERE parent_orchestration_run_id = :orid'
            ),
            {"orid": str(orch_run[0])},
        ).fetchall()
        assert len(child_runs) >= 1, (
            f"Expected at least 1 child agent_run with parent_orchestration_run_id={orch_run[0]}"
        )
        for row in child_runs:
            assert str(row[0]) == str(orch_run[0])
            assert row[1] is None  # delegated runs have no session


@pytest.mark.integration
def test_sequential_delegates_persist_as_child_agent_runs(client, app, mock_auth, auth_headers):
    """SequentialEngine with 2 agents persists 2 child agent_runs via the real hook chain.

    Unlike the supervisor test above (which patches Orchestration.run and fires the callback
    directly), this test patches Agent.run at the leaf level so that the real SequentialEngine
    executes the entity loop, calls on_delegate_complete after each agent, and the callback
    persists rows through the live _persist_delegate_run closure.  This exercises the full
    route → SequentialEngine → on_delegate_complete → persist_agent_run wiring.
    """
    a = _create_agent(app, "Step1")
    b = _create_agent(app, "Step2")
    orch_id = _create_orchestration(app, "sequential", agent_ids=[a, b])

    fake_output = _make_fake_agent_output("step result")

    with patch("agentic.agent.agent.Agent.run", return_value=fake_output):
        resp = client.post(
            f"/api/orchestrations/{orch_id}/run/stream",
            json={"message": "run sequentially"},
            headers=auth_headers,
            buffered=True,
        )
        assert resp.status_code == 200

    with app.app_context():
        orch_run = db.session.execute(
            text('SELECT id FROM "ai".orchestration_runs WHERE orchestration_id = :oid'),
            {"oid": orch_id},
        ).fetchone()
        assert orch_run is not None, "Expected an orchestration_runs row"

        child_rows = db.session.execute(
            text(
                'SELECT parent_orchestration_run_id, session_id FROM "ai".agent_runs '
                "WHERE parent_orchestration_run_id = :orid"
            ),
            {"orid": str(orch_run[0])},
        ).fetchall()
        assert len(child_rows) == 2, f"Expected 2 child agent_runs, got {len(child_rows)}"
        for row in child_rows:
            assert str(row[0]) == str(orch_run[0])
            assert row[1] is None  # delegated runs have no session


@pytest.mark.integration
def test_parallel_strategy_persists_all_child_runs(client, app, mock_auth, auth_headers):
    """ParallelEngine with N agents produces N child agent_runs, all pointing at the same parent."""
    a = _create_agent(app, "A")
    b = _create_agent(app, "B")
    c = _create_agent(app, "C")
    orch_id = _create_orchestration(app, "parallel", agent_ids=[a, b, c])

    fake_output = _make_fake_agent_output("done")

    with patch("agentic.agent.agent.Agent.run", return_value=fake_output):
        # buffered=True forces full consumption of the SSE generator in the test client
        resp = client.post(
            f"/api/orchestrations/{orch_id}/run/stream",
            json={"message": "go"},
            headers=auth_headers,
            buffered=True,
        )
        assert resp.status_code == 200

    with app.app_context():
        orch_run = db.session.execute(
            text('SELECT id FROM "ai".orchestration_runs WHERE orchestration_id = :oid'),
            {"oid": orch_id},
        ).fetchone()
        assert orch_run is not None, "Expected an orchestration_runs row"

        child_rows = db.session.execute(
            text(
                'SELECT parent_orchestration_run_id, session_id FROM "ai".agent_runs '
                "WHERE parent_orchestration_run_id = :orid"
            ),
            {"orid": str(orch_run[0])},
        ).fetchall()
        assert len(child_rows) == 3, f"Expected 3 child agent_runs, got {len(child_rows)}"
        for row in child_rows:
            assert str(row[0]) == str(orch_run[0])
            assert row[1] is None  # delegated runs have no session


@pytest.mark.integration
def test_single_agent_run_endpoint_orchestration(client, app, mock_auth, auth_headers):
    """GET /api/agents/runs/<run_id> returns parent_orchestration_run_id for child runs."""
    from agentic.execution.status import ExecutionStatus
    from agentic.orchestration.output import OrchestrationOutput

    agent_id = _create_agent(app, "Orch-Fetch")
    orch_id = _create_orchestration(app, "supervisor", agent_ids=[agent_id])

    def fake_orchestration_run(input, context=None, *, history=None, on_delegate_complete=None):
        if on_delegate_complete is not None:
            on_delegate_complete(
                {
                    "agent_name": "Orch-Fetch",
                    "child_execution_id": str(uuid.uuid4()),
                    "orchestration_run_id": context.orchestration_run_id if context else None,
                    "task": input if isinstance(input, str) else "task",
                    "content": "orchestration result",
                    "status": ExecutionStatus.COMPLETED,
                    "error": None,
                    "usage": {"total_tokens": 5},
                    "steps": 1,
                    "events": [],
                    "tool_calls": [],
                    "messages": [],
                    "started_at": None,
                    "completed_at": None,
                }
            )
        out = OrchestrationOutput(execution_id=context.execution_id if context else "test")
        out.status = ExecutionStatus.COMPLETED
        out.content = "done"
        out.steps = 1
        out.usage = {"total_tokens": 5}
        out.events = []
        return out

    with patch(
        "agentic.orchestration.orchestration.Orchestration.run",
        side_effect=fake_orchestration_run,
    ):
        resp = client.post(
            f"/api/orchestrations/{orch_id}/run/stream",
            json={"message": "fetch it"},
            headers=auth_headers,
            buffered=True,
        )
        assert resp.status_code == 200

    with app.app_context():
        orch_run = db.session.execute(
            text('SELECT id FROM "ai".orchestration_runs WHERE orchestration_id = :oid'),
            {"oid": orch_id},
        ).fetchone()
        assert orch_run is not None

        child = db.session.execute(
            text(
                'SELECT run_id, parent_orchestration_run_id FROM "ai".agent_runs '
                "WHERE parent_orchestration_run_id = :orid LIMIT 1"
            ),
            {"orid": str(orch_run[0])},
        ).fetchone()
        assert child is not None, "Expected at least one child agent_run"
        run_id = child[0]
        expected_orch_run_id = str(child[1])

    run_resp = client.get(f"/api/agents/runs/{run_id}", headers=auth_headers)
    assert run_resp.status_code == 200, run_resp.get_data(as_text=True)
    run_data = run_resp.get_json()

    assert run_data["run_id"] == run_id
    assert run_data["content"] == "orchestration result"
    assert run_data["parent_orchestration_run_id"] == expected_orch_run_id
    assert run_data["parent_workflow_execution_id"] is None
    assert run_data["session_id"] is None


@pytest.mark.integration
def test_delegate_payload_missing_orch_run_id_falls_back(client, app, mock_auth, auth_headers):
    """If a strategy engine forgets to set payload['orchestration_run_id'], the
    route still stamps the child agent_run with the outer orchestration_run uuid.

    Guards against regressions where a new engine / DelegateTool code path drops
    context.orchestration_run_id — without this fallback the child row would be
    written with parent_orchestration_run_id=NULL and orphaned from its parent.
    """
    from agentic.execution.status import ExecutionStatus
    from agentic.orchestration.output import OrchestrationOutput

    agent_id = _create_agent(app, "Fallback")
    orch_id = _create_orchestration(app, "supervisor", agent_ids=[agent_id])
    child_execution_id = str(uuid.uuid4())

    def fake_orchestration_run(input, context=None, *, history=None, on_delegate_complete=None):
        if on_delegate_complete is not None:
            # Intentionally omit 'orchestration_run_id' from the payload to
            # simulate an engine that forgot to propagate context.
            on_delegate_complete(
                {
                    "agent_name": "Fallback",
                    "child_execution_id": child_execution_id,
                    "task": input if isinstance(input, str) else "task",
                    "content": "result",
                    "status": ExecutionStatus.COMPLETED,
                    "error": None,
                    "usage": {"total_tokens": 3},
                    "steps": 1,
                    "events": [],
                    "tool_calls": [],
                    "messages": [],
                    "started_at": None,
                    "completed_at": None,
                }
            )
        out = OrchestrationOutput(execution_id=context.execution_id if context else "test")
        out.status = ExecutionStatus.COMPLETED
        out.content = "done"
        out.steps = 1
        out.usage = {}
        out.events = []
        return out

    with patch(
        "agentic.orchestration.orchestration.Orchestration.run",
        side_effect=fake_orchestration_run,
    ):
        resp = client.post(
            f"/api/orchestrations/{orch_id}/run/stream",
            json={"message": "go"},
            headers=auth_headers,
            buffered=True,
        )
        assert resp.status_code == 200

    with app.app_context():
        orch_run = db.session.execute(
            text('SELECT id FROM "ai".orchestration_runs WHERE orchestration_id = :oid'),
            {"oid": orch_id},
        ).fetchone()
        assert orch_run is not None

        child = db.session.execute(
            text('SELECT parent_orchestration_run_id FROM "ai".agent_runs WHERE run_id = :rid'),
            {"rid": child_execution_id},
        ).fetchone()
        assert child is not None, "Expected child agent_run even when payload omits orch id"
        assert str(child[0]) == str(orch_run[0])
