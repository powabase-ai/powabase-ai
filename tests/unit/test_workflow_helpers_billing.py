"""Tests for the workflow_helpers billing wiring.

Covers the block_type → action mapping and the charge_workflow_blocks
helper that posts one billing.charge (services/billing_port.py) per
successful block in a workflow execution, via a RecordingBillingAdapter
(tests/support/billing.py).
"""

from agentic_project_service.routes import _workflow_helpers
from agentic_project_service.services import billing_port
from tests.support.billing import RecordingBillingAdapter

# ---------------------------------------------------------------------------
# _resolve_block_billing_action
# ---------------------------------------------------------------------------


def test_resolve_block_billing_action_for_external_api():
    """general_api / api_call / platform_api → workflow_block_external_api."""
    resolve = _workflow_helpers._resolve_block_billing_action
    assert resolve("general_api") == "workflow_block_external_api"
    assert resolve("api_call") == "workflow_block_external_api"
    assert resolve("platform_api") == "workflow_block_external_api"


def test_resolve_block_billing_action_for_code():
    """code / function → workflow_block_code."""
    resolve = _workflow_helpers._resolve_block_billing_action
    assert resolve("code") == "workflow_block_code"
    assert resolve("function") == "workflow_block_code"


def test_resolve_block_billing_action_defaults_to_other():
    """Everything else → workflow_block_other (cost 0)."""
    resolve = _workflow_helpers._resolve_block_billing_action
    assert resolve("starter") == "workflow_block_other"
    assert resolve("condition") == "workflow_block_other"
    assert resolve("response") == "workflow_block_other"
    assert resolve("split") == "workflow_block_other"
    assert resolve("webhook") == "workflow_block_other"
    # Agent + orchestration blocks: charged via their inner runs, not here.
    assert resolve("agent") == "workflow_block_other"
    assert resolve("orchestration") == "workflow_block_other"
    # Unknown type also falls through to _other so a new block type can't
    # raise KeyError during billing.
    assert resolve("totally_new") == "workflow_block_other"


# ---------------------------------------------------------------------------
# charge_workflow_blocks
# ---------------------------------------------------------------------------


def test_charge_workflow_blocks_one_charge_per_block(recording_billing):
    """A workflow with N blocks → exactly N billing.charge calls."""
    blocks_data = [
        {"id": "b1", "type": "general_api"},
        {"id": "b2", "type": "code"},
        {"id": "b3", "type": "starter"},
    ]
    block_outputs = {"b1": {"output": "ok"}, "b2": {"output": 42}, "b3": {"output": "x"}}

    _workflow_helpers.charge_workflow_blocks(
        execution_id="exec-1",
        block_outputs=block_outputs,
        blocks_data=blocks_data,
    )

    assert len(recording_billing.charges) == 3


def test_charge_workflow_blocks_uses_correct_actions(recording_billing):
    """Each block bills its mapped action."""
    blocks_data = [
        {"id": "b1", "type": "general_api"},
        {"id": "b2", "type": "code"},
        {"id": "b3", "type": "starter"},
    ]
    block_outputs = {
        "b1": {"output": "ok"},
        "b2": {"output": 42},
        "b3": {"output": "x"},
    }

    _workflow_helpers.charge_workflow_blocks(
        execution_id="exec-1",
        block_outputs=block_outputs,
        blocks_data=blocks_data,
    )

    actions = sorted(c["action"] for c in recording_billing.charges)
    assert actions == [
        "workflow_block_code",
        "workflow_block_external_api",
        "workflow_block_other",
    ]


def test_charge_workflow_blocks_skips_failed_blocks(recording_billing):
    """A block whose output dict carries `error` is not billed."""
    blocks_data = [
        {"id": "b1", "type": "general_api"},
        {"id": "b2", "type": "code"},
    ]
    block_outputs = {
        "b1": {"output": "ok"},
        "b2": {"error": "syntax error"},
    }

    _workflow_helpers.charge_workflow_blocks(
        execution_id="exec-1",
        block_outputs=block_outputs,
        blocks_data=blocks_data,
    )

    assert len(recording_billing.charges) == 1
    assert recording_billing.charges[0]["ref_id"] == "b1"


def test_charge_workflow_blocks_idempotency_parts_is_execution_and_block_id(recording_billing):
    """idempotency_parts is exactly the (execution_id, block_id) tail.

    The cloud adapter prepends org_id + action to reconstruct
    ``sha256(org_id + action + execution_id + block_id)`` — the same key
    retry-dedup always relied on. Dropping block_id here would collapse
    every block in an execution onto the same key and double-charge on
    retry (the plan's only multi-part idempotency tail).
    """
    blocks_data = [{"id": "b1", "type": "general_api"}]
    block_outputs = {"b1": {"output": "ok"}}

    _workflow_helpers.charge_workflow_blocks(
        execution_id="exec-1",
        block_outputs=block_outputs,
        blocks_data=blocks_data,
    )

    assert recording_billing.charges[0]["idempotency_parts"] == ("exec-1", "b1")


def test_charge_workflow_blocks_idempotency_parts_unique_per_block(recording_billing):
    """Same execution, different blocks → different idempotency_parts tails."""
    blocks_data = [
        {"id": "b1", "type": "general_api"},
        {"id": "b2", "type": "general_api"},
    ]
    block_outputs = {"b1": {"output": "ok"}, "b2": {"output": "ok"}}

    _workflow_helpers.charge_workflow_blocks(
        execution_id="exec-1",
        block_outputs=block_outputs,
        blocks_data=blocks_data,
    )

    parts = [c["idempotency_parts"] for c in recording_billing.charges]
    assert len(parts) == 2
    assert parts[0] != parts[1]


def test_charge_workflow_blocks_ref_type_is_workflow_block(recording_billing):
    """ref_type identifies the row in billing's audit logs."""
    blocks_data = [{"id": "b1", "type": "code"}]
    block_outputs = {"b1": {"output": 42}}

    _workflow_helpers.charge_workflow_blocks(
        execution_id="exec-1",
        block_outputs=block_outputs,
        blocks_data=blocks_data,
    )

    charge = recording_billing.charges[0]
    assert charge["ref_type"] == "workflow_block"
    assert charge["ref_id"] == "b1"


def test_charge_workflow_blocks_handles_unknown_block_id(recording_billing):
    """A block_output for a block_id not in blocks_data → falls back to _other."""
    blocks_data = [{"id": "b1", "type": "code"}]
    block_outputs = {"orphan": {"output": "x"}}

    _workflow_helpers.charge_workflow_blocks(
        execution_id="exec-1",
        block_outputs=block_outputs,
        blocks_data=blocks_data,
    )

    # Block type lookup misses → action falls through to _other (free).
    assert len(recording_billing.charges) == 1
    assert recording_billing.charges[0]["action"] == "workflow_block_other"


def test_charge_workflow_blocks_continues_after_charge_reports_insufficient():
    """A charge that reports insufficient credits does not raise or halt the
    loop — charge_workflow_blocks doesn't inspect the ChargeOutcome per
    block; billing failures are bounded-loss per the credits_client v1
    design."""
    blocks_data = [
        {"id": "b1", "type": "general_api"},
        {"id": "b2", "type": "code"},
    ]
    block_outputs = {"b1": {"output": "ok"}, "b2": {"output": 42}}

    rec = RecordingBillingAdapter(insufficient=True)
    billing_port.set_billing_adapter(rec)

    # Should not raise.
    _workflow_helpers.charge_workflow_blocks(
        execution_id="exec-1",
        block_outputs=block_outputs,
        blocks_data=blocks_data,
    )

    # Both blocks were charged despite each outcome reporting insufficient
    # credits — the loop is fire-and-forget per block, not abort-on-402.
    assert len(rec.charges) == 2
