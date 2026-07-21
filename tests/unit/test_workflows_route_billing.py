"""Tests for billing wiring in routes/workflows.py.

Covers the pre-op balance check on /execute and /execute/stream, routed
through the billing port (services/billing_port.py) via a
RecordingBillingAdapter (tests/support/billing.py). The workflow runner is
heavyweight (build_workflow_from_db, DB writes, asyncio.run), so tests patch
the helpers and verify the contract.
"""

from unittest.mock import MagicMock, patch

from agentic_project_service.routes import workflows as workflows_route
from agentic_project_service.services import billing_port
from tests.support.billing import RecordingBillingAdapter


def _make_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(workflows_route.workflows_bp)
    return app


def _auth_headers():
    return {"Authorization": "Bearer fake"}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_workflow_run_estimated_cost_constant():
    assert workflows_route._WORKFLOW_RUN_ESTIMATED_COST == 2_000


# ---------------------------------------------------------------------------
# execute_workflow — balance check fires up front
# ---------------------------------------------------------------------------


def test_execute_workflow_balance_check_fires_first():
    """402 from billing.check_balance propagates without touching the workflow.

    load_blocks is patched so _workflow_pre_check can reach billing.check_balance.
    Without the patch, load_blocks hits the DB (no app context) → fail-closed 503
    instead of the 402 from billing.check_balance that this test verifies.
    """
    app = _make_test_app()
    rec = RecordingBillingAdapter(raise_402=True)
    billing_port.set_billing_adapter(rec)

    with (
        # load_blocks is called by _workflow_pre_check before billing.check_balance.
        # Return empty list so _workflow_pre_check reaches the balance check.
        patch.object(workflows_route, "load_blocks", return_value=[]),
        # Patch build_workflow_from_db so if billing somehow doesn't fire
        # first, we'd at least notice (it'd return None and reach a 404).
        patch.object(workflows_route, "build_workflow_from_db") as mock_build,
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/workflows/wf-1/execute",
                json={"variables": {"foo": "bar"}},
                headers=_auth_headers(),
            )

    assert resp.status_code == 402
    assert rec.balance_checks == [2_000]
    # Workflow never ran → no charges, no build.
    assert rec.charges == []
    mock_build.assert_not_called()


def test_execute_workflow_propagates_503():
    """R3-F2: 503 from billing.check_balance propagates to the response.

    load_blocks must be patched (return []) so _workflow_pre_check reaches
    billing.check_balance. Without this patch, load_blocks hits the DB (no app
    context) → fail-closed 503 from the pre-check's own except handler →
    test would observe 503 from the WRONG source and pass for the wrong reason
    (billing.check_balance never gets called).

    rec.balance_checks pins that the 503 genuinely came from billing.check_balance
    (not from load_blocks' own fail-closed path or some other ambient 503 source).
    """
    app = _make_test_app()
    rec = RecordingBillingAdapter(raise_503=True)
    billing_port.set_billing_adapter(rec)

    with (
        patch.object(workflows_route, "load_blocks", return_value=[]),
        patch.object(workflows_route, "build_workflow_from_db"),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/workflows/wf-1/execute",
                json={"variables": {}},
                headers=_auth_headers(),
            )

    assert resp.status_code == 503
    assert rec.balance_checks == [2_000]


# ---------------------------------------------------------------------------
# execute_workflow_stream — balance check fires up front
# ---------------------------------------------------------------------------


def test_execute_workflow_stream_balance_check_fires_first():
    """402 from billing.check_balance propagates before SSE generator starts.

    load_blocks is patched so _workflow_pre_check can reach billing.check_balance.
    Without the patch, load_blocks hits the DB (no app context) → fail-closed 503
    instead of the 402 from billing.check_balance that this test verifies.
    """
    app = _make_test_app()
    rec = RecordingBillingAdapter(raise_402=True)
    billing_port.set_billing_adapter(rec)

    with (
        patch.object(workflows_route, "load_blocks", return_value=[]),
        patch.object(workflows_route, "build_workflow_from_db") as mock_build,
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/workflows/wf-1/execute/stream",
                json={"variables": {}},
                headers=_auth_headers(),
            )

    assert resp.status_code == 402
    assert rec.balance_checks == [2_000]
    assert rec.charges == []
    mock_build.assert_not_called()


# ---------------------------------------------------------------------------
# Success path: billing.charge fires after a successful workflow run
# ---------------------------------------------------------------------------


def _mock_async_arun_detailed(block_outputs=None):
    """Build a Workflow mock whose arun_detailed returns (output, events)."""
    block_outputs = block_outputs or {"b1": {"output": "result"}}

    async def _run(**_):
        return block_outputs, []

    wf = MagicMock()
    wf.arun_detailed = _run
    return wf


def _mock_db_for_workflow_execute():
    """db.session mock that absorbs INSERT/UPDATE calls."""
    mock_session = MagicMock()
    mock_session.execute.return_value = MagicMock()
    return mock_session


def test_execute_workflow_posts_charge_on_success(recording_billing):
    """A successful sync run posts the workflow_run dispatch fee + per-block."""
    app = _make_test_app()
    block_outputs = {"b1": {"output": "ok"}}
    blocks_data = [{"id": "b1", "type": "code"}]

    with (
        patch.object(workflows_route, "charge_workflow_blocks") as mock_block_charge,
        patch.object(
            workflows_route,
            "build_workflow_from_db",
            return_value=_mock_async_arun_detailed(block_outputs),
        ),
        patch.object(workflows_route, "load_blocks", return_value=blocks_data),
        patch.object(workflows_route, "load_edges", return_value=[]),
        patch.object(workflows_route, "make_agent_run_recorder", return_value=(lambda _: None, {})),
        patch.object(workflows_route, "make_services", return_value={}),
        patch.object(workflows_route, "build_block_logs", return_value=[]),
        patch.object(workflows_route, "persist_block_logs", return_value=True),
        patch(
            "agentic_project_service.routes.workflows.db.session", _mock_db_for_workflow_execute()
        ),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/workflows/wf-1/execute",
                json={"variables": {}},
                headers=_auth_headers(),
            )

    assert resp.status_code == 200
    # billing.charge for workflow_run was called exactly once
    assert len(recording_billing.charges) == 1
    charge = recording_billing.charges[0]
    assert charge["action"] == "workflow_run"
    assert charge["quantity"] == 1
    assert charge["ref_type"] == "workflow_run"
    # charge_workflow_blocks was also called with the right context.
    mock_block_charge.assert_called_once()
    block_kwargs = mock_block_charge.call_args.kwargs
    assert block_kwargs["block_outputs"] == block_outputs
    assert block_kwargs["blocks_data"] == blocks_data


def test_execute_workflow_skips_charge_on_failure(recording_billing):
    """A run that raises during arun_detailed does NOT post a workflow_run charge."""
    app = _make_test_app()

    async def _raise(**_):
        raise RuntimeError("boom")

    wf = MagicMock()
    wf.arun_detailed = _raise

    with (
        patch.object(workflows_route, "charge_workflow_blocks") as mock_block_charge,
        patch.object(workflows_route, "build_workflow_from_db", return_value=wf),
        patch.object(workflows_route, "load_blocks", return_value=[]),
        patch.object(workflows_route, "load_edges", return_value=[]),
        patch.object(workflows_route, "make_agent_run_recorder", return_value=(lambda _: None, {})),
        patch.object(workflows_route, "make_services", return_value={}),
        patch(
            "agentic_project_service.routes.workflows.db.session", _mock_db_for_workflow_execute()
        ),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/workflows/wf-1/execute",
                json={"variables": {}},
                headers=_auth_headers(),
            )

    # 500 because the run raised — billing never posted.
    assert resp.status_code == 500
    assert recording_billing.charges == []
    mock_block_charge.assert_not_called()


def test_execute_workflow_uses_run_id_in_idempotency_key(recording_billing):
    """ref_id is the execution id; idempotency_parts is deterministic from it."""
    app = _make_test_app()
    block_outputs = {"b1": {"output": "ok"}}
    blocks_data = [{"id": "b1", "type": "code"}]

    with (
        patch.object(workflows_route, "charge_workflow_blocks"),
        patch.object(
            workflows_route,
            "build_workflow_from_db",
            return_value=_mock_async_arun_detailed(block_outputs),
        ),
        patch.object(workflows_route, "load_blocks", return_value=blocks_data),
        patch.object(workflows_route, "load_edges", return_value=[]),
        patch.object(workflows_route, "make_agent_run_recorder", return_value=(lambda _: None, {})),
        patch.object(workflows_route, "make_services", return_value={}),
        patch.object(workflows_route, "build_block_logs", return_value=[]),
        patch.object(workflows_route, "persist_block_logs", return_value=True),
        patch(
            "agentic_project_service.routes.workflows.db.session", _mock_db_for_workflow_execute()
        ),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/workflows/wf-1/execute",
                json={"variables": {}},
                headers=_auth_headers(),
            )

    body = resp.get_json()
    charge = recording_billing.charges[0]
    assert charge["ref_id"] == body["execution_id"]
    assert charge["idempotency_parts"] == (body["execution_id"],)
