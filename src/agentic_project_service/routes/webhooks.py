"""Webhook trigger endpoint for workflows.

Unauthenticated endpoint — security is via webhook secret token
stored in the block config, validated via Authorization header or query param.
"""

import asyncio
import hmac
import json
import logging
import uuid
from datetime import UTC, datetime

from flask import Blueprint, jsonify, request
from sqlalchemy import text

from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
from ..services.llm_availability import check_model_available
from ..services.run_context import (
    reset_run_id,
    set_run_id,
)
from ._workflow_helpers import (
    ErrorCode,
    build_block_logs,
    build_workflow_from_db,
    charge_workflow_blocks,
    error_response,
    get_final_output,
    load_blocks,
    load_edges,
    make_agent_run_recorder,
    make_services,
    persist_block_logs,
    serialize_outputs,
)
from .workflows import _workflow_pre_check

logger = logging.getLogger(__name__)

EXECUTION_TIMEOUT = 300  # 5 min for webhook-triggered runs

webhooks_bp = Blueprint("webhooks", __name__, url_prefix="/api/webhooks")


@webhooks_bp.route("/<webhook_id>", methods=["POST"])
def trigger_webhook(webhook_id: str):
    """Trigger a workflow via webhook.

    Security: validates secret token from Authorization header or ?token= query param.
    """
    # 0. Validate webhook_id is a UUID
    try:
        uuid.UUID(webhook_id)
    except ValueError:
        return error_response("Invalid webhook ID", ErrorCode.VALIDATION_ERROR, 400)

    # 1. Look up the block with this webhook_id
    block_row = db.session.execute(
        text(f"""
            SELECT wb.id, wb.workflow_id, wb.config
            FROM "{AI_SCHEMA}".workflow_blocks wb
            WHERE wb.config->>'webhook_id' = :webhook_id
              AND wb.type = 'webhook'
            LIMIT 1
        """),
        {"webhook_id": webhook_id},
    ).fetchone()

    if not block_row:
        return error_response("Webhook not found", ErrorCode.WORKFLOW_NOT_FOUND, 404)

    config = block_row[2] or {}
    workflow_id = block_row[1]

    # 2. Validate secret token (timing-attack safe) — BEFORE state gate
    #    so an invalid secret cannot consume the single-use arm slot.
    expected_secret = config.get("webhook_secret", "")
    provided_token = _extract_token()

    if not expected_secret or not hmac.compare_digest(provided_token, expected_secret):
        return jsonify({"error": "Unauthorized"}), 401

    # 3. Check workflow deployment state
    wf_row = db.session.execute(
        text(f'SELECT state, webhook_armed_until FROM "{AI_SCHEMA}".workflows WHERE id = :id'),
        {"id": workflow_id},
    ).fetchone()

    if not wf_row:
        return error_response("Workflow not found", ErrorCode.WORKFLOW_NOT_FOUND, 404)

    wf_state = wf_row[0] or "internal"

    if wf_state != "deployed":
        armed_until = wf_row[1]
        now = datetime.now(UTC)
        logger.info(
            "Webhook gate: workflow=%s state=%s armed_until=%s now=%s",
            workflow_id,
            wf_state,
            armed_until,
            now,
        )
        if not armed_until or armed_until < now:
            return jsonify({"error": "Webhook is not active"}), 403
        # Atomic disarm — only first concurrent request succeeds
        disarmed = db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".workflows
                SET webhook_armed_until = NULL
                WHERE id = :id AND webhook_armed_until IS NOT NULL
                RETURNING id
            """),
            {"id": workflow_id},
        ).fetchone()
        db.session.commit()
        if not disarmed:
            return jsonify({"error": "Webhook is not active"}), 403

    # 4. Parse request body as webhook payload
    payload = request.get_json(silent=True) or {}

    # 4a. Pre-op balance check (free-tier hard cap) via the billing port.
    # Done after auth+state checks so an unauthorized or undeployed webhook
    # can't trigger billing traffic. The no-op adapter makes this inert in
    # OSS/unit-test/local-dev builds.
    _workflow_pre_check(workflow_id)

    # 5. Build and execute workflow
    wf = build_workflow_from_db(workflow_id)
    if wf is None:
        return error_response("Workflow not found", ErrorCode.WORKFLOW_NOT_FOUND, 404)

    blocks_data = load_blocks(workflow_id)
    edges_data = load_edges(workflow_id)

    # Fail-fast on any agent block whose resolved model has neither a project
    # BYOK key nor a platform env key. Mirrors routes/workflows.py:503-515:
    # without this, a webhook-triggered run for a BYOK-only block model
    # bypasses the gate and surfaces LiteLLM's generic "Missing API Key"
    # deep in the engine. Runs BEFORE the workflow_executions INSERT so a
    # 400 doesn't leave an orphan "running" execution behind.
    for block in blocks_data:
        if block.get("type") != "agent":
            continue
        cfg = block.get("config") or {}
        block_model = cfg.get("model")
        if not block_model and cfg.get("agent_id"):
            agent_row = db.session.execute(
                text(f'SELECT model FROM "{AI_SCHEMA}".agents WHERE id = :id'),
                {"id": cfg["agent_id"]},
            ).fetchone()
            block_model = agent_row[0] if agent_row else None
        if block_model:
            check_model_available(block_model)

    # Create execution record
    exec_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    db.session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".workflow_executions
                (id, workflow_id, status, input, started_at)
            VALUES (:id, :wid, 'running', CAST(:input AS jsonb), :started)
        """),
        {"id": exec_id, "wid": workflow_id, "input": json.dumps(payload), "started": now},
    )
    db.session.commit()

    # Bind webhook exec_id as the billing run_id for the entire async run.
    # asyncio.run() copies the current Context into the loop so coroutines
    # spawned by the workflow engine (and any inner agent runs that don't
    # re-bind) see this id when deriving idempotency keys. Without this,
    # webhook-triggered tool calls mint fresh uuid4 keys and double-charge
    # on retry (spec line 132).
    #
    # Retry-determinism caveat: webhook providers (Stripe, GitHub, etc.)
    # generally retry failed deliveries with the same body but our exec_id
    # is regenerated per POST. So provider-triggered HTTP retries produce
    # different exec_ids -> different keys -> will double-charge. The
    # in-loop idempotency this run_id buys protects only PS-side bounded
    # retries at the billing layer. Threading the provider's
    # delivery-id header (Stripe-Signature event id, X-GitHub-Delivery,
    # ...) into exec_id is the proper fix; deferred to a follow-up.
    _wh_run_id_token = set_run_id(exec_id)
    try:
        recorder, agent_run_ids_by_block = make_agent_run_recorder(exec_id, log_context="webhook")
        services = make_services(auth_token="", agent_run_recorder=recorder)

        async def _run_with_timeout():
            return await asyncio.wait_for(
                wf.arun_detailed(variables=payload, services=services),
                timeout=EXECUTION_TIMEOUT,
            )

        with billing.llm_call_scope():
            output, events = asyncio.run(_run_with_timeout())

        duration_map = {
            e.block_id: e.duration_ms for e in events if e.type in ("block_complete", "block_error")
        }

        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".workflow_executions
                SET status = 'completed',
                    output = CAST(:output AS jsonb),
                    block_outputs = CAST(:block_outputs AS jsonb),
                    completed_at = :completed
                WHERE id = :id
            """),
            {
                "id": exec_id,
                "output": json.dumps(get_final_output(output, blocks_data)),
                "block_outputs": json.dumps(serialize_outputs(output)),
                "completed": datetime.now(UTC),
            },
        )
        db.session.commit()

        block_logs = build_block_logs(
            output,
            blocks_data,
            edges_data,
            duration_map,
            agent_run_ids_by_block=agent_run_ids_by_block,
        )
        if not persist_block_logs(exec_id, block_logs):
            logger.warning("Block logs failed to persist for execution %s", exec_id)

        # Billing: post the dispatch fee for the workflow run + per-block
        # charges via the billing port. Webhook-triggered runs share the
        # same charge schedule as /execute (workflow_run + workflow_block_*)
        # because they exercise the same workflow.arun_detailed path.
        billing.charge(
            action="workflow_run",
            quantity=1,
            ref_type="workflow_run",
            ref_id=exec_id,
            idempotency_parts=(exec_id,),
            metadata={
                "workflow_id": str(workflow_id),
                "blocks": len(output),
                "trigger": "webhook",
            },
        )
        charge_workflow_blocks(
            execution_id=exec_id,
            block_outputs=output,
            blocks_data=blocks_data,
        )

        return jsonify(
            {
                "execution_id": exec_id,
                "status": "completed",
                "output": get_final_output(output, blocks_data),
            }
        )

    except asyncio.TimeoutError:
        error_msg = f"Execution timed out after {EXECUTION_TIMEOUT}s"
        logger.error("Webhook workflow %s timed out", workflow_id)
        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".workflow_executions
                SET status = 'failed', error = :error, completed_at = :completed
                WHERE id = :id
            """),
            {"id": exec_id, "error": error_msg, "completed": datetime.now(UTC)},
        )
        db.session.commit()
        return error_response(
            error_msg,
            ErrorCode.EXECUTION_TIMEOUT,
            504,
            execution_id=exec_id,
        )

    except Exception as e:
        logger.error("Webhook workflow execution failed: %s", e, exc_info=True)
        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".workflow_executions
                SET status = 'failed', error = :error, completed_at = :completed
                WHERE id = :id
            """),
            {"id": exec_id, "error": str(e), "completed": datetime.now(UTC)},
        )
        db.session.commit()
        return error_response(
            "Workflow execution failed",
            ErrorCode.EXECUTION_FAILED,
            500,
            execution_id=exec_id,
        )
    finally:
        reset_run_id(_wh_run_id_token)


# ---------------------------------------------------------------------------
# Helpers (webhook-specific)
# ---------------------------------------------------------------------------


def _extract_token() -> str:
    """Extract webhook secret from Authorization header or query param."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return request.args.get("token", "")
