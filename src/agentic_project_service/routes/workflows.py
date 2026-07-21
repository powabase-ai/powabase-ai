"""Workflow management and execution routes for the project service."""

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from flask import Blueprint, Response, jsonify, request, stream_with_context
from sqlalchemy import text

from ..auth import require_auth
from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
from ..services.llm_availability import check_model_available
from ..services.rate_limit import rate_limit_executions
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

# Pre-op estimate. v1.5 smoke-test bump from 51 mc — see
# orchestrations.py:_ORCHESTRATION_RUN_ESTIMATED_COST for full rationale.
# 2,000 mc ≈ $0.02 covers a workflow with ~2 average LLM-driven blocks
# (workflow_run dispatch fee + 1-2 agent blocks). Bigger workflows can
# still overrun mid-stream (post-charge architectural limitation).
_WORKFLOW_RUN_ESTIMATED_COST = 2_000

# Conservative per-block max cost for pre-check scaling. Block dispatch fees
# range from 1-5 credits in the catalog; LLM-completion blocks add llm_call
# recoup (~30-60 credits per call for typical models). 75 covers both.
_WORKFLOW_PER_BLOCK_MAX_CREDITS: int = 75

# Floor when block lookup fails or workflow has 0 blocks (rare).
_WORKFLOW_FALLBACK_ESTIMATE: int = _WORKFLOW_RUN_ESTIMATED_COST  # 2000

logger = logging.getLogger(__name__)

EXECUTION_TIMEOUT = 300  # 5 min for HTTP-triggered runs
STREAM_EXECUTION_TIMEOUT = 600  # 10 min for streaming runs

workflows_bp = Blueprint("workflows", __name__, url_prefix="/api/workflows")


# ---------------------------------------------------------------------------
# Pre-checks & Helpers
# ---------------------------------------------------------------------------


def _workflow_pre_check(workflow_id: str) -> None:
    """Pre-op balance check that scales with the workflow's actual block count.

    Replaces the fixed 2000-credit pre-check that allowed a 100-block workflow
    on a 100-credit customer to run to completion before billing caught up.
    Routed through the billing port — the no-op adapter makes this inert in
    OSS/unit-test/local-dev builds; the cloud adapter enforces the cap.
    """
    try:
        blocks = load_blocks(workflow_id)
    except Exception as exc:
        logger.error(
            "load_blocks failed for workflow=%s during pre-check; failing closed: %s",
            workflow_id,
            exc,
        )
        from werkzeug.exceptions import ServiceUnavailable

        raise ServiceUnavailable(
            "Balance check failed (workflow block lookup error). Retry shortly."
        )
    estimated_cost = max(
        _WORKFLOW_FALLBACK_ESTIMATE,
        len(blocks) * _WORKFLOW_PER_BLOCK_MAX_CREDITS,
    )
    billing.check_balance(estimated_cost=estimated_cost)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@workflows_bp.route("", methods=["GET"])
@require_auth
def list_workflows():
    """List all workflows, paginated, with execution aggregates."""
    from ..services.list_params import parse_list_params, escape_like, ListParamsError

    try:
        limit, offset, q, sort, order = parse_list_params(
            request,
            sort_allowed={"created_at", "name", "updated_at", "last_execution_at"},
        )
    except ListParamsError as e:
        return jsonify({"error": str(e)}), e.status

    where_clause = ""
    params: dict = {"limit": limit, "offset": offset}
    if q:
        where_clause = "WHERE w.name ILIKE :q_like"
        params["q_like"] = f"%{escape_like(q)}%"

    # `last_execution_at` is a computed column; reference it by SELECT alias.
    if sort == "last_execution_at":
        order_by = f"last_execution_at {order.upper()} NULLS LAST, w.id ASC"
    else:
        order_by = f"w.{sort} {order.upper()}, w.id ASC"

    count_sql = f'SELECT COUNT(*) FROM "{AI_SCHEMA}".workflows w {where_clause}'
    total = db.session.execute(text(count_sql), params).scalar()

    rows_sql = f"""
        SELECT
          w.id, w.name, w.description, w.variables, w.version, w.color,
          w.created_at, w.updated_at, w.state, w.schedule_config,
          (SELECT COUNT(*) FROM "{AI_SCHEMA}".workflow_executions WHERE workflow_id = w.id) AS execution_count,
          (SELECT MAX(created_at) FROM "{AI_SCHEMA}".workflow_executions WHERE workflow_id = w.id) AS last_execution_at
        FROM "{AI_SCHEMA}".workflows w
        {where_clause}
        ORDER BY {order_by}
        LIMIT :limit OFFSET :offset
    """
    rows = db.session.execute(text(rows_sql), params)

    workflows = []
    for row in rows:
        workflows.append(
            {
                "id": str(row.id),
                "name": row.name,
                "description": row.description,
                "variables": row.variables,
                "version": row.version,
                "color": row.color,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "state": row.state or "internal",
                "schedule_config": row.schedule_config,
                "execution_count": int(row.execution_count),
                "last_execution_at": row.last_execution_at.isoformat()
                if row.last_execution_at
                else None,
            }
        )

    return jsonify(
        {
            "workflows": workflows,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@workflows_bp.route("", methods=["POST"])
@require_auth
def create_workflow():
    """Create a new workflow."""
    data = request.get_json() or {}
    if not data.get("name"):
        return jsonify({"error": "Name is required"}), 400

    wf_id = str(uuid.uuid4())
    db.session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".workflows (id, name, description, variables, color)
            VALUES (:id, :name, :description, CAST(:variables AS jsonb), :color)
        """),
        {
            "id": wf_id,
            "name": data["name"],
            "description": data.get("description"),
            "variables": json.dumps(data.get("variables", {})),
            "color": data.get("color"),
        },
    )
    db.session.commit()
    return jsonify({"id": wf_id, "name": data["name"]}), 201


@workflows_bp.route("/<workflow_id>", methods=["GET"])
@require_auth
def get_workflow(workflow_id: str):
    """Get a workflow with its blocks and edges."""
    row = db.session.execute(
        text(f"""
            SELECT id, name, description, variables, version, color,
                   created_at, updated_at, state, schedule_config
            FROM "{AI_SCHEMA}".workflows WHERE id = :id
        """),
        {"id": workflow_id},
    ).fetchone()

    if not row:
        return error_response("Workflow not found", ErrorCode.WORKFLOW_NOT_FOUND, 404)

    blocks = load_blocks(workflow_id)
    edges = load_edges(workflow_id)

    return jsonify(
        {
            "id": str(row[0]),
            "name": row[1],
            "description": row[2],
            "variables": row[3],
            "version": row[4],
            "color": row[5],
            "created_at": row[6].isoformat() if row[6] else None,
            "updated_at": row[7].isoformat() if row[7] else None,
            "state": row[8] or "internal",
            "schedule_config": row[9],
            "blocks": blocks,
            "edges": edges,
        }
    )


@workflows_bp.route("/<workflow_id>", methods=["PATCH"])
@require_auth
def update_workflow(workflow_id: str):
    """Update workflow metadata."""
    data = request.get_json() or {}
    sets = []
    params: dict[str, Any] = {"id": workflow_id}

    for field in ("name", "description", "color"):
        if field in data:
            sets.append(f"{field} = :{field}")
            params[field] = data[field]
    if "variables" in data:
        sets.append("variables = CAST(:variables AS jsonb)")
        params["variables"] = json.dumps(data["variables"])

    if not sets:
        return jsonify({"error": "No fields to update"}), 400

    db.session.execute(
        text(f'UPDATE "{AI_SCHEMA}".workflows SET {", ".join(sets)} WHERE id = :id'),
        params,
    )
    db.session.commit()
    return jsonify({"ok": True})


@workflows_bp.route("/<workflow_id>", methods=["DELETE"])
@require_auth
def delete_workflow(workflow_id: str):
    """Delete a workflow and its blocks/edges (cascade)."""
    db.session.execute(
        text(f'DELETE FROM "{AI_SCHEMA}".workflows WHERE id = :id'),
        {"id": workflow_id},
    )
    db.session.commit()
    return "", 204


# ---------------------------------------------------------------------------
# Deploy / Undeploy / Arm
# ---------------------------------------------------------------------------


@workflows_bp.route("/<workflow_id>/deploy", methods=["POST"])
@require_auth
def deploy_workflow(workflow_id: str):
    """Set workflow state to deployed (webhooks always listening)."""
    # Extract schedule config from the starter block
    starter_row = db.session.execute(
        text(f"""
            SELECT config FROM "{AI_SCHEMA}".workflow_blocks
            WHERE workflow_id = :wid AND type = 'starter'
            LIMIT 1
        """),
        {"wid": workflow_id},
    ).fetchone()

    schedule_config = None
    if starter_row and starter_row[0]:
        cfg = starter_row[0]
        if cfg.get("schedule_enabled"):
            sched: dict[str, Any] = {
                "enabled": True,
                "type": cfg.get("schedule_type", "interval"),
                "timezone": cfg.get("schedule_timezone", "UTC"),
                "start_at": cfg.get("schedule_start_at") or None,
                "end_at": cfg.get("schedule_end_at") or None,
                "max_runs": None,
            }
            raw_max = cfg.get("schedule_max_runs")
            try:
                sched["max_runs"] = int(raw_max) if raw_max else None
            except (ValueError, TypeError):
                sched["max_runs"] = None
            if sched["max_runs"] is not None:
                sched["max_runs"] = max(1, sched["max_runs"])
            if sched["type"] == "interval":
                raw_val = cfg.get("schedule_interval_value")
                try:
                    val = int(raw_val) if raw_val else 5
                except (ValueError, TypeError):
                    val = 5
                val = max(1, val)
                unit = cfg.get("schedule_interval_unit", "minutes")
                multiplier = {"minutes": 60, "hours": 3600, "days": 86400}
                sched["interval_seconds"] = max(60, val * multiplier.get(unit, 60))
                sched["cron"] = None
            else:
                sched["interval_seconds"] = None
                sched["cron"] = cfg.get("schedule_cron", "0 * * * *")
            schedule_config = sched

    result = db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".workflows
            SET state = 'deployed', webhook_armed_until = NULL,
                schedule_config = CAST(:sched AS jsonb),
                schedule_run_count = 0
            WHERE id = :id
            RETURNING id
        """),
        {"id": workflow_id, "sched": json.dumps(schedule_config)},
    ).fetchone()
    if not result:
        return jsonify({"error": "Workflow not found"}), 404
    db.session.commit()
    return jsonify({"ok": True, "state": "deployed"})


@workflows_bp.route("/<workflow_id>/undeploy", methods=["POST"])
@require_auth
def undeploy_workflow(workflow_id: str):
    """Set workflow state back to internal (webhooks dormant)."""
    result = db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".workflows
            SET state = 'internal', webhook_armed_until = NULL,
                schedule_config = NULL, last_scheduled_at = NULL
            WHERE id = :id
            RETURNING id
        """),
        {"id": workflow_id},
    ).fetchone()
    if not result:
        return jsonify({"error": "Workflow not found"}), 404
    db.session.commit()
    return jsonify({"ok": True, "state": "internal"})


@workflows_bp.route("/<workflow_id>/arm", methods=["POST"])
@require_auth
def arm_webhook(workflow_id: str):
    """Arm the webhook for a single execution (10-minute TTL)."""
    result = db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".workflows
            SET webhook_armed_until = NOW() + interval '10 minutes'
            WHERE id = :id
            RETURNING webhook_armed_until
        """),
        {"id": workflow_id},
    ).fetchone()
    if not result:
        return jsonify({"error": "Workflow not found"}), 404
    db.session.commit()
    logger.info("Webhook armed: workflow=%s armed_until=%s", workflow_id, result[0])
    return jsonify({"ok": True, "armed_until": result[0].isoformat()})


# ---------------------------------------------------------------------------
# Graph save
# ---------------------------------------------------------------------------


@workflows_bp.route("/<workflow_id>/graph", methods=["PUT"])
@require_auth
def save_graph(workflow_id: str):
    """Save the full graph (blocks + edges) for a workflow."""
    data = request.get_json() or {}
    blocks = data.get("blocks", [])
    edges = data.get("edges", [])

    # Verify workflow exists
    exists = db.session.execute(
        text(f'SELECT 1 FROM "{AI_SCHEMA}".workflows WHERE id = :id'),
        {"id": workflow_id},
    ).fetchone()
    if not exists:
        return error_response("Workflow not found", ErrorCode.WORKFLOW_NOT_FOUND, 404)

    # Validate required fields
    for b in blocks:
        if "id" not in b or "type" not in b:
            return error_response(
                "Each block must have 'id' and 'type'",
                ErrorCode.VALIDATION_ERROR,
                400,
            )
    for e in edges:
        if "source" not in e or "target" not in e:
            return error_response(
                "Each edge must have 'source' and 'target'",
                ErrorCode.VALIDATION_ERROR,
                400,
            )

    # Validate block types against engine registry
    from agentic.workflow.block import BlockRegistry

    for b in blocks:
        if BlockRegistry.get(b["type"]) is None:
            return error_response(
                f"Unknown block type: {b['type']}",
                ErrorCode.VALIDATION_ERROR,
                400,
            )

    # Validate edge references point to blocks in this graph
    block_ids = {b["id"] for b in blocks}
    for e in edges:
        if e["source"] not in block_ids or e["target"] not in block_ids:
            return error_response(
                "Edge references a block not in this graph",
                ErrorCode.VALIDATION_ERROR,
                400,
            )

    # Clear existing blocks and edges
    db.session.execute(
        text(f'DELETE FROM "{AI_SCHEMA}".workflow_edges WHERE workflow_id = :wid'),
        {"wid": workflow_id},
    )
    db.session.execute(
        text(f'DELETE FROM "{AI_SCHEMA}".workflow_blocks WHERE workflow_id = :wid'),
        {"wid": workflow_id},
    )

    # Insert blocks
    for b in blocks:
        db.session.execute(
            text(f"""
                INSERT INTO "{AI_SCHEMA}".workflow_blocks
                    (id, workflow_id, type, name, position_x, position_y, config, enabled)
                VALUES
                    (:id, :wid, :type, :name, :px, :py, CAST(:config AS jsonb), :enabled)
            """),
            {
                "id": b["id"],
                "wid": workflow_id,
                "type": b["type"],
                "name": b.get("name", ""),
                "px": b.get("position", {}).get("x", 0),
                "py": b.get("position", {}).get("y", 0),
                "config": json.dumps(b.get("config", {})),
                "enabled": b.get("enabled", True),
            },
        )

    # Insert edges
    for e in edges:
        db.session.execute(
            text(f"""
                INSERT INTO "{AI_SCHEMA}".workflow_edges
                    (id, workflow_id, source_block_id, target_block_id,
                     source_handle, target_handle, condition)
                VALUES
                    (:id, :wid, :src, :tgt, :sh, :th, :cond)
            """),
            {
                "id": e.get("id", str(uuid.uuid4())),
                "wid": workflow_id,
                "src": e["source"],
                "tgt": e["target"],
                "sh": e.get("sourceHandle", "output"),
                "th": e.get("targetHandle", "input"),
                "cond": e.get("condition"),
            },
        )

    # Bump version
    db.session.execute(
        text(f'UPDATE "{AI_SCHEMA}".workflows SET version = version + 1 WHERE id = :id'),
        {"id": workflow_id},
    )
    db.session.commit()

    return jsonify({"ok": True, "blocks": len(blocks), "edges": len(edges)})


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@workflows_bp.route("/<workflow_id>/execute", methods=["POST"])
@require_auth
@rate_limit_executions
def execute_workflow(workflow_id: str):
    """Execute a workflow synchronously and return the result."""
    data = request.get_json() or {}
    input_variables = data.get("variables", data.get("input", {}))

    # Pre-op balance check (free-tier hard cap) via the billing port.
    # Propagates 402/503 to the caller before any workflow side-effects so
    # callers don't get a half-run. The no-op adapter makes this inert in
    # OSS/unit-test/local-dev builds.
    _workflow_pre_check(workflow_id)

    wf = build_workflow_from_db(workflow_id)
    if wf is None:
        return error_response("Workflow not found", ErrorCode.WORKFLOW_NOT_FOUND, 404)

    # Build block name/config maps for logs
    blocks_data = load_blocks(workflow_id)

    # Fail fast on any agent block whose resolved model has neither a project
    # BYOK key nor a platform env key. ``build_workflow_from_db`` already
    # resolves each agent block's model from its linked agent row; we mirror
    # that lookup here and check each one BEFORE creating the
    # workflow_executions row so a BYOK-only model doesn't leave an orphan
    # "running" execution behind on 400.
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

    # Only store input if a Starter block exists
    has_entry = any(b["type"] in ("starter", "webhook") for b in blocks_data)
    exec_input = input_variables if has_entry else None

    # Create execution record
    exec_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    db.session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".workflow_executions
                (id, workflow_id, status, input, started_at)
            VALUES (:id, :wid, 'running', CAST(:input AS jsonb), :started)
        """),
        {"id": exec_id, "wid": workflow_id, "input": json.dumps(exec_input), "started": now},
    )
    db.session.commit()
    edges_data = load_edges(workflow_id)

    # Bind workflow exec_id as the billing run_id for the entire async run.
    # asyncio.run() copies the current Context into the loop, so coroutines
    # spawned inside the workflow engine (and any inner agent runs that
    # don't re-bind) see this run_id when deriving idempotency keys.
    # Reset in finally so a worker pod reuse never inherits stale state.
    #
    # Retry-determinism caveat: each HTTP POST to this endpoint generates
    # a FRESH ``exec_id = str(uuid.uuid4())``. So a user-triggered HTTP
    # retry of the same logical request produces a different exec_id ->
    # different idempotency keys -> WILL double-charge. The in-loop
    # idempotency this run_id buys protects only against PS-side bounded
    # retries (HTTP timeout / 5xx → credits_client retries 3x with the
    # SAME idempotency_key). Threading a client-supplied Idempotency-Key
    # header through into ``exec_id`` would extend protection to HTTP
    # retries; deferred to a follow-up per spec line 139.
    _wf_run_id_token = set_run_id(exec_id)
    try:
        auth_token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()

        recorder, agent_run_ids_by_block = make_agent_run_recorder(exec_id, log_context="sync")
        services = make_services(auth_token=auth_token, agent_run_recorder=recorder)

        async def _run_with_timeout():
            with billing.llm_call_scope():
                return await asyncio.wait_for(
                    wf.arun_detailed(variables=input_variables, services=services),
                    timeout=EXECUTION_TIMEOUT,
                )

        output, events = asyncio.run(_run_with_timeout())

        # Build duration map from events
        duration_map = {
            e.block_id: e.duration_ms for e in events if e.type in ("block_complete", "block_error")
        }

        # Persist result
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

        # Billing: post the dispatch fee for the workflow run + one charge per
        # successful block via the billing port. Failures are bounded-loss;
        # never let a billing error fail the user-facing run.
        billing.charge(
            action="workflow_run",
            quantity=1,
            ref_type="workflow_run",
            ref_id=exec_id,
            idempotency_parts=(exec_id,),
            metadata={"workflow_id": workflow_id, "blocks": len(output)},
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
                "block_outputs": serialize_outputs(output),
                "block_logs": block_logs,
            }
        )

    except asyncio.TimeoutError:
        error_msg = f"Execution timed out after {EXECUTION_TIMEOUT}s"
        logger.error("Workflow %s timed out", workflow_id)
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
        logger.error("Workflow execution failed: %s", e, exc_info=True)
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
            str(e),
            ErrorCode.EXECUTION_FAILED,
            500,
            execution_id=exec_id,
        )
    finally:
        reset_run_id(_wf_run_id_token)


@workflows_bp.route("/<workflow_id>/execute/stream", methods=["POST"])
@require_auth
@rate_limit_executions
def execute_workflow_stream(workflow_id: str):
    """Execute a workflow with streaming (SSE)."""
    data = request.get_json() or {}
    input_variables = data.get("variables", data.get("input", {}))

    # Pre-op balance check (free-tier hard cap) via the billing port.
    # Propagates 402/503 to the caller before any workflow side-effects so
    # callers don't get a half-run. The no-op adapter makes this inert in
    # OSS/unit-test/local-dev builds.
    _workflow_pre_check(workflow_id)

    wf = build_workflow_from_db(workflow_id)
    if wf is None:
        return error_response("Workflow not found", ErrorCode.WORKFLOW_NOT_FOUND, 404)

    # Load block metadata for log enrichment
    blocks_data = load_blocks(workflow_id)

    # Fail fast on any agent block whose model has neither a project BYOK key
    # nor a platform env key. Mirrors execute_workflow's pre-check so the SSE
    # stream never opens with a doomed model. Aborts 400 BEFORE entering the
    # generator so the error propagates as a normal HTTP response.
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

    # Only store input if a Starter block exists
    has_entry = any(b["type"] in ("starter", "webhook") for b in blocks_data)
    exec_input = input_variables if has_entry else None

    exec_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    db.session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".workflow_executions
                (id, workflow_id, status, input, started_at)
            VALUES (:id, :wid, 'running', CAST(:input AS jsonb), :started)
        """),
        {"id": exec_id, "wid": workflow_id, "input": json.dumps(exec_input), "started": now},
    )
    db.session.commit()

    auth_token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    edges_data = load_edges(workflow_id)

    recorder, agent_run_ids_by_block = make_agent_run_recorder(exec_id, log_context="stream")
    services = make_services(auth_token=auth_token, agent_run_recorder=recorder)

    async def _stream_events():
        """Pure async generator — yields (sse_string, event_data) tuples, no DB calls."""
        try:
            async with asyncio.timeout(STREAM_EXECUTION_TIMEOUT):
                with billing.llm_call_scope():
                    async for event in wf.astream(variables=input_variables, services=services):
                        payload = {
                            "type": event.type,
                            "block_id": event.block_id,
                            "block_type": event.block_type,
                        }
                        if event.type == "block_chunk":
                            payload["chunk"] = event.data
                        elif event.type in ("block_complete", "block_error"):
                            payload["data"] = event.data
                            payload["duration_ms"] = event.duration_ms

                        yield f"data: {json.dumps(payload)}\n\n", payload

            # Final done event
            yield f"data: {json.dumps({'type': 'done', 'execution_id': exec_id})}\n\n", None

        except asyncio.TimeoutError:
            error_msg = f"Execution timed out after {STREAM_EXECUTION_TIMEOUT}s"
            logger.error("Workflow stream %s timed out", workflow_id)
            yield (
                f"data: {json.dumps({'type': 'error', 'error': error_msg, 'error_code': ErrorCode.EXECUTION_TIMEOUT})}\n\n",
                {"type": "error", "error": error_msg},
            )

        except Exception as e:
            logger.error("Workflow stream failed: %s", e, exc_info=True)
            yield (
                f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n",
                {"type": "error", "error": str(e)},
            )

    def _persist_execution_result(eid, block_outputs, error_msg, duration_map=None):
        """Persist execution result to DB (called on the Flask thread after streaming)."""
        try:
            if error_msg:
                db.session.execute(
                    text(f"""
                        UPDATE "{AI_SCHEMA}".workflow_executions
                        SET status = 'failed', error = :error, completed_at = :completed
                        WHERE id = :id
                    """),
                    {"id": eid, "error": error_msg, "completed": datetime.now(UTC)},
                )
            else:
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
                        "id": eid,
                        "output": json.dumps(get_final_output(block_outputs, blocks_data)),
                        "block_outputs": json.dumps(serialize_outputs(block_outputs)),
                        "completed": datetime.now(UTC),
                    },
                )
            db.session.commit()

            # Build and persist block logs (including partial results on error)
            if block_outputs:
                dur_map = duration_map or {}
                block_logs = build_block_logs(
                    block_outputs,
                    blocks_data,
                    edges_data,
                    dur_map,
                    agent_run_ids_by_block=agent_run_ids_by_block,
                )
                if not persist_block_logs(eid, block_logs):
                    logger.warning("Block logs failed to persist for execution %s", eid)

            # Billing: only charge on success — failed/timeout runs don't pay
            # the dispatch fee, via the billing port.
            if not error_msg:
                billing.charge(
                    action="workflow_run",
                    quantity=1,
                    ref_type="workflow_run",
                    ref_id=eid,
                    idempotency_parts=(eid,),
                    metadata={"workflow_id": workflow_id, "blocks": len(block_outputs)},
                )
                charge_workflow_blocks(
                    execution_id=eid,
                    block_outputs=block_outputs,
                    blocks_data=blocks_data,
                )
        except Exception as persist_err:
            logger.error("Failed to persist execution result: %s", persist_err, exc_info=True)

    def _sync_stream():
        # Bind workflow exec_id as the billing run_id for the streaming run.
        # _stream_events runs inside the loop spawned below; asyncio inherits
        # the current Context into coroutines, so this binding flows down
        # into every inner agent run that doesn't re-bind its own run_id.
        _wf_stream_token = set_run_id(exec_id)
        loop = asyncio.new_event_loop()
        agen = _stream_events()
        block_outputs = {}
        duration_map: dict[str, float | None] = {}
        error_msg = None
        try:
            while True:
                try:
                    event_str, event_data = loop.run_until_complete(agen.__anext__())
                    if event_data and event_data.get("type") == "block_complete":
                        block_outputs[event_data["block_id"]] = event_data.get("data")
                        duration_map[event_data["block_id"]] = event_data.get("duration_ms")
                    elif event_data and event_data.get("type") == "block_error":
                        block_outputs[event_data["block_id"]] = event_data.get("data")
                        duration_map[event_data["block_id"]] = event_data.get("duration_ms")
                    elif event_data and event_data.get("type") == "error":
                        error_msg = event_data.get("error")
                    yield event_str
                except StopAsyncIteration:
                    break
        except Exception as e:
            error_msg = str(e)
        finally:
            loop.run_until_complete(agen.aclose())
            loop.close()
            _persist_execution_result(exec_id, block_outputs, error_msg, duration_map)
            reset_run_id(_wf_stream_token)

    return Response(
        stream_with_context(_sync_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@workflows_bp.route("/<workflow_id>/executions", methods=["GET"])
@require_auth
def list_executions(workflow_id: str):
    """List execution history for a workflow (lightweight — no block_outputs)."""
    try:
        limit = max(1, min(int(request.args.get("limit", 20)), 100))
    except ValueError:
        limit = 20

    rows = db.session.execute(
        text(f"""
            SELECT id, status, input, output, error,
                   started_at, completed_at, created_at
            FROM "{AI_SCHEMA}".workflow_executions
            WHERE workflow_id = :wid
            ORDER BY created_at DESC
            LIMIT :limit
        """),
        {"wid": workflow_id, "limit": limit},
    ).fetchall()

    executions = [
        {
            "id": str(r[0]),
            "status": r[1],
            "input": r[2],
            "output": r[3],
            "error": r[4],
            "started_at": r[5].isoformat() if r[5] else None,
            "completed_at": r[6].isoformat() if r[6] else None,
            "created_at": r[7].isoformat() if r[7] else None,
        }
        for r in rows
    ]

    return jsonify({"executions": executions})


@workflows_bp.route("/<workflow_id>/executions/<execution_id>/logs", methods=["GET"])
@require_auth
def get_execution_logs(workflow_id: str, execution_id: str):
    """Get per-block execution logs for a specific workflow execution."""
    # Verify execution belongs to this workflow
    exists = db.session.execute(
        text(f"""
            SELECT 1 FROM "{AI_SCHEMA}".workflow_executions
            WHERE id = :eid AND workflow_id = :wid
        """),
        {"eid": execution_id, "wid": workflow_id},
    ).fetchone()
    if not exists:
        return jsonify({"error": "Execution not found"}), 404

    rows = db.session.execute(
        text(f"""
            SELECT block_id, block_type, block_name, status, duration_ms,
                   input, output, error, config_snapshot, execution_order,
                   agent_run_id
            FROM "{AI_SCHEMA}".workflow_block_logs
            WHERE execution_id = :eid
            ORDER BY execution_order
        """),
        {"eid": execution_id},
    ).fetchall()

    block_logs = [
        {
            "block_id": str(r[0]),
            "block_type": r[1],
            "block_name": r[2],
            "status": r[3],
            "duration_ms": r[4],
            "input": r[5],
            "output": r[6],
            "error": r[7],
            "config_snapshot": r[8],
            "execution_order": r[9],
            "agent_run_id": str(r[10]) if r[10] else None,
        }
        for r in rows
    ]

    return jsonify({"block_logs": block_logs})
