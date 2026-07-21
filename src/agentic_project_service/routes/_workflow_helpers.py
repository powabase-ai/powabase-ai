"""Shared helper functions for workflow and webhook routes."""

import json
import logging
import uuid as _uuid
from collections.abc import Callable
from typing import Any

from agentic.execution.status import ExecutionStatus
from agentic.workflow import Workflow
from sqlalchemy import text

from ..db import db, AI_SCHEMA
from ..models.tenant import AgentRunStatus
from ..services import billing_port as billing
from ..services.ai_provider_keys_resolver import resolve_api_key_or_raise_for_drop
from ..services.context_handler import execute_retrieval_async
from ..services.session import persist_agent_run
from ..services.settings_registry import get_setting

logger = logging.getLogger(__name__)

# Keys that must never appear in config_snapshot logs
_SECRET_CONFIG_KEYS = frozenset({"webhook_secret"})


# Block-type → billing action. Anything not in this map falls back to
# ``workflow_block_other`` (cost 0 per the pricing catalog).
#
# ``agent`` and ``orchestration`` blocks are deliberately mapped to ``_other``
# (the free bucket) — their actual cost is billed through the sub-runs they
# spawn (the agent's tool calls / retrievals via Task 15, and the
# orchestration's own ``orchestration_run`` charge via the orchestration
# route). Charging ``workflow_block_*`` at the block layer for agent and
# orchestration blocks would double-bill the dispatch fee.
_BLOCK_BILLING_ACTION: dict[str, str] = {
    "general_api": "workflow_block_external_api",
    "api_call": "workflow_block_external_api",
    "platform_api": "workflow_block_external_api",
    "code": "workflow_block_code",
    "function": "workflow_block_code",
}


def _resolve_block_billing_action(block_type: str) -> str:
    """Map a workflow block type to its billing action."""
    return _BLOCK_BILLING_ACTION.get(block_type, "workflow_block_other")


def charge_workflow_blocks(
    *,
    execution_id: str,
    block_outputs: dict[str, Any],
    blocks_data: list[dict],
) -> None:
    """Post per-block billing charges for a completed workflow execution.

    One ``billing.charge`` call per block_id present in ``block_outputs``,
    using the block_type → action mapping. Failed blocks are skipped (output
    carries an ``error`` key) — failed blocks are not billed for the
    dispatch fee.

    Idempotency: ``sha256(org_id + action + execution_id + block_id)`` —
    stable across retries of the same execution, distinct per block. Per
    spec line 131: workflow blocks key by run_id + step_index; we use
    block_id as the step identifier because the engine processes each
    block exactly once per execution. ``org_id`` and ``action`` are
    prepended by the billing adapter; this function passes only the
    ``(execution_id, block_id)`` tail via ``idempotency_parts``.
    """
    block_type_map = {b["id"]: b["type"] for b in blocks_data}

    for block_id, output in block_outputs.items():
        # Skip blocks that errored — only successful dispatches are billed.
        if isinstance(output, dict) and output.get("error"):
            continue

        block_type = block_type_map.get(block_id, "")
        action = _resolve_block_billing_action(block_type)

        billing.charge(
            action=action,
            quantity=1,
            ref_type="workflow_block",
            ref_id=block_id,
            idempotency_parts=(execution_id, block_id),
            metadata={"execution_id": execution_id, "block_type": block_type},
        )


def load_blocks(workflow_id: str) -> list[dict]:
    rows = db.session.execute(
        text(f"""
            SELECT id, type, name, position_x, position_y, config, enabled
            FROM "{AI_SCHEMA}".workflow_blocks
            WHERE workflow_id = :wid
        """),
        {"wid": workflow_id},
    ).fetchall()

    return [
        {
            "id": str(r[0]),
            "type": r[1],
            "name": r[2],
            "position": {"x": r[3], "y": r[4]},
            "config": r[5],
            "enabled": r[6],
        }
        for r in rows
    ]


def load_edges(workflow_id: str) -> list[dict]:
    rows = db.session.execute(
        text(f"""
            SELECT id, source_block_id, target_block_id,
                   source_handle, target_handle, condition
            FROM "{AI_SCHEMA}".workflow_edges
            WHERE workflow_id = :wid
        """),
        {"wid": workflow_id},
    ).fetchall()

    return [
        {
            "id": str(r[0]),
            "source": str(r[1]),
            "target": str(r[2]),
            "sourceHandle": r[3],
            "targetHandle": r[4],
            "condition": r[5],
        }
        for r in rows
    ]


def build_workflow_from_db(workflow_id: str) -> Workflow | None:
    """Load a Workflow object from the database."""
    row = db.session.execute(
        text(f'SELECT name, description, variables FROM "{AI_SCHEMA}".workflows WHERE id = :id'),
        {"id": workflow_id},
    ).fetchone()
    if not row:
        return None

    blocks = load_blocks(workflow_id)
    edges = load_edges(workflow_id)

    # Resolve agent_id for agent-type blocks
    for block in blocks:
        if block["type"] == "agent" and block["config"].get("agent_id"):
            agent_row = db.session.execute(
                text(f'SELECT model, system_prompt FROM "{AI_SCHEMA}".agents WHERE id = :id'),
                {"id": block["config"]["agent_id"]},
            ).fetchone()
            if agent_row:
                block["config"]["model"] = agent_row[0] or get_setting("AGENT_DEFAULT_MODEL")
                if agent_row[1]:
                    block["config"]["system_prompt"] = agent_row[1]

            # Fetch agent's pre-configured KBs and override block-level KBs
            kb_rows = db.session.execute(
                text(
                    f'SELECT knowledge_base_id FROM "{AI_SCHEMA}"'
                    f".agent_knowledge_bases WHERE agent_id = :id"
                ),
                {"id": block["config"]["agent_id"]},
            ).fetchall()
            if kb_rows:
                block["config"]["knowledge_bases"] = [{"id": str(r[0])} for r in kb_rows]
            else:
                block["config"]["knowledge_bases"] = []

    # Resolve orchestration_id for orchestration-type blocks
    for block in blocks:
        if block["type"] == "orchestration" and block["config"].get("orchestration_id"):
            orch_row = db.session.execute(
                text(f'SELECT id, name, strategy FROM "{AI_SCHEMA}".orchestrations WHERE id = :id'),
                {"id": block["config"]["orchestration_id"]},
            ).fetchone()
            if not orch_row:
                raise ValueError(f"Orchestration {block['config']['orchestration_id']} not found")

    # Option A: inject block_id into each block's config so AgentBlock.execute
    # can include it in the on_agent_run_complete payload.
    for block in blocks:
        config = block.get("config") or {}
        config["block_id"] = block["id"]
        block["config"] = config

    # Convert edges to engine format
    engine_edges = [
        {
            "source_block_id": e["source"],
            "target_block_id": e["target"],
            "source_handle": e.get("sourceHandle", "output"),
            "target_handle": e.get("targetHandle", "input"),
            "condition": e.get("condition"),
        }
        for e in edges
    ]

    return Workflow.from_graph(
        name=row[0] or "workflow",
        blocks=blocks,
        edges=engine_edges,
        variables=row[2] or {},
        description=row[1] or "",
    )


def make_services(
    auth_token: str = "",
    agent_run_recorder: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Create the services dict for workflow execution."""

    async def _retrieve_kb_context(query: str, knowledge_base_configs: list[dict]) -> str:
        result = await execute_retrieval_async(db.session, query, knowledge_base_configs)
        return result.get("formatted_context", "")

    async def _run_orchestration(orchestration_id: str, message: str) -> dict:
        """Run an orchestration and return the result (non-streaming).

        Note: orchestration.run() is synchronous and will block, but this is
        acceptable because the workflow engine runs inside asyncio.run() from
        a sync Flask route — the event loop has no other concurrent tasks.
        We cannot use asyncio.to_thread here because build_orchestration
        accesses Flask-SQLAlchemy's thread-local db.session.
        """
        from ..services.orchestration import build_orchestration

        _orch_row, orchestration = build_orchestration(orchestration_id)
        with billing.llm_call_scope():
            output = orchestration.run(input=message)
        return {
            "content": output.content or "",
            "status": "completed" if output.status.is_success() else "failed",
            "steps": output.steps,
            "usage": output.usage,
        }

    services: dict[str, Any] = {
        "retrieve_kb_context": _retrieve_kb_context,
        "run_orchestration": _run_orchestration,
        "platform_api_base_url": "http://localhost:5000",
        "platform_api_token": auth_token,
        # Injected so AgentBlock can pass the user's BYOK key to Agent, ensuring
        # the billing.llm_call_scope() wrap correctly signals "user paid" to
        # BillingLogger (CRIT-1 from PR #440 review).
        "resolve_agent_api_key": resolve_api_key_or_raise_for_drop,
    }
    if agent_run_recorder is not None:
        services["on_agent_run_complete"] = agent_run_recorder
    return services


def make_agent_run_recorder(
    exec_id: str,
    *,
    log_context: str = "workflow",
) -> tuple[Callable[[dict], None], dict[str, str]]:
    """Create an on_agent_run_complete callback that persists AgentBlock runs.

    Returns (recorder, agent_run_ids_by_block). The recorder calls
    persist_agent_run with parent_workflow_execution_id=exec_id, captures the
    returned UUID, and stores it in the map keyed by block_id. The map is
    later passed to build_block_logs so workflow_block_logs.agent_run_id can
    point at the run.

    Failures are logged and swallowed — persistence is observability, not
    correctness.
    """
    agent_run_ids_by_block: dict[str, str] = {}

    # Translate the core-lib ExecutionStatus enum into AgentRunStatus.
    # Parallels the orchestration recorder in routes/orchestrations.py.
    _status_map = {
        ExecutionStatus.COMPLETED: AgentRunStatus.COMPLETED,
        ExecutionStatus.FAILED: AgentRunStatus.FAILED,
        ExecutionStatus.CANCELLED: AgentRunStatus.FAILED,
    }

    def _recorder(payload: dict) -> None:
        try:
            block_id = payload.get("block_id")
            system_prompt = payload.get("system_prompt") or ""
            prompt = payload.get("prompt") or ""
            input_messages = [
                *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                {"role": "user", "content": prompt},
            ]
            run_id = f"wfblk_{_uuid.uuid4().hex[:12]}"
            # Default to FAILED when the payload carries no status — conservative
            # (a missing status is more likely a bug than a silent success).
            mapped_status = _status_map.get(payload.get("status"), AgentRunStatus.FAILED)
            ar_uuid = persist_agent_run(
                db_session=db.session,
                run_id=run_id,
                status=mapped_status,
                input_messages=input_messages,
                output_messages=[{"role": "assistant", "content": payload.get("content") or ""}],
                content=payload.get("content"),
                usage=payload.get("usage"),
                error=payload.get("error"),
                parent_workflow_execution_id=exec_id,
                model=payload.get("model"),
            )
            db.session.commit()
            if block_id:
                agent_run_ids_by_block[str(block_id)] = ar_uuid
        except Exception:
            db.session.rollback()
            logger.exception(
                "Failed to persist AgentBlock agent_run (%s exec_id=%s); continuing",
                log_context,
                exec_id,
            )

    return _recorder, agent_run_ids_by_block


def get_final_output(
    block_outputs: dict[str, Any],
    blocks_data: list[dict[str, Any]] | None = None,
) -> Any | None:
    """Extract the final output from the Response block, or None if absent."""
    if blocks_data:
        response_ids = {b["id"] for b in blocks_data if b["type"] == "response"}
        for block_id in response_ids:
            output = block_outputs.get(block_id)
            if isinstance(output, dict) and "output" in output:
                return output["output"]
        return None
    # Fallback when blocks_data not provided (legacy callers)
    for _block_id, output in reversed(list(block_outputs.items())):
        if isinstance(output, dict) and "output" in output:
            return output["output"]
    return None


def serialize_outputs(block_outputs: dict[str, Any]) -> dict[str, Any]:
    """Ensure all outputs are JSON-serializable."""
    result = {}
    for k, v in block_outputs.items():
        try:
            json.dumps(v)
            result[k] = v
        except (TypeError, ValueError):
            result[k] = str(v)
    return result


def safe_json(value: Any) -> Any:
    """Return value if JSON-serializable, otherwise convert to string."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _sanitize_config(config: dict) -> dict:
    """Strip secret keys from config before logging."""
    if not _SECRET_CONFIG_KEYS.intersection(config):
        return config
    return {k: v for k, v in config.items() if k not in _SECRET_CONFIG_KEYS}


def build_block_logs(
    output: dict[str, Any],
    blocks_data: list[dict],
    edges_data: list[dict],
    duration_map: dict[str, float | None],
    agent_run_ids_by_block: dict[str, str] | None = None,
) -> list[dict]:
    """Build block log entries from execution output.

    Extracts the duplicated block-log construction logic used by workflows,
    webhooks, and the scheduler into a single helper.
    """
    block_name_map = {b["id"]: b.get("name", b["id"]) for b in blocks_data}
    block_config_map = {b["id"]: b.get("config", {}) for b in blocks_data}
    block_type_map = {b["id"]: b["type"] for b in blocks_data}
    _agent_run_ids = agent_run_ids_by_block or {}

    upstream_map: dict[str, list[str]] = {}
    for e in edges_data:
        upstream_map.setdefault(e["target"], []).append(e["source"])

    block_logs = []
    for block_id, block_data in output.items():
        block_type = block_type_map.get(block_id, "")
        upstream_ids = upstream_map.get(block_id, [])
        log_input = (
            {src: output.get(src) for src in upstream_ids if src in output}
            if upstream_ids
            else None
        )
        block_logs.append(
            {
                "block_id": block_id,
                "block_type": block_type,
                "block_name": block_name_map.get(block_id, block_id),
                "status": "error"
                if isinstance(block_data, dict) and block_data.get("error")
                else "success",
                "duration_ms": duration_map.get(block_id),
                "config": block_config_map.get(block_id, {}),
                "output": block_data,
                "input": log_input,
                "agent_run_id": _agent_run_ids.get(block_id),
            }
        )
    return block_logs


def persist_block_logs(exec_id: str, block_logs: list[dict]) -> bool:
    """Persist per-block execution logs to the workflow_block_logs table.

    For agent blocks we also promote model + token counts from the block's
    `output` JSONB (shape: {content, model, usage: {...}}) into typed columns
    so the observability dashboard can aggregate without parsing JSONB.

    Returns True on success, False if the DB write failed.
    """
    try:
        for idx, log in enumerate(block_logs):
            config = log.get("config", {})
            config = _sanitize_config(config)

            # Extract typed token cols for agent blocks. Non-agent blocks get NULLs.
            model = None
            tokens: dict[str, int | None] = {
                "prompt_tokens": None,
                "completion_tokens": None,
                "reasoning_tokens": None,
                "total_tokens": None,
            }
            output_data = log.get("output")
            if log.get("block_type") == "agent" and isinstance(output_data, dict):
                model = output_data.get("model")
                usage = output_data.get("usage") or {}
                if isinstance(usage, dict):
                    try:
                        if usage.get("prompt_tokens") is not None:
                            tokens["prompt_tokens"] = int(usage["prompt_tokens"])
                        if usage.get("completion_tokens") is not None:
                            tokens["completion_tokens"] = int(usage["completion_tokens"])
                        if usage.get("reasoning_tokens") is not None:
                            tokens["reasoning_tokens"] = int(usage["reasoning_tokens"])
                        if usage.get("total_tokens") is not None:
                            tokens["total_tokens"] = int(usage["total_tokens"])
                    except (TypeError, ValueError):
                        # If any field isn't an int, leave the rest as captured.
                        pass

            db.session.execute(
                text(f"""
                    INSERT INTO "{AI_SCHEMA}".workflow_block_logs
                        (execution_id, block_id, block_type, block_name, status,
                         execution_order, duration_ms, input, output, error, config_snapshot,
                         agent_run_id, model,
                         prompt_tokens, completion_tokens, reasoning_tokens, total_tokens)
                    VALUES (:eid, :bid, :btype, :bname, :status,
                            :ord, :dur, CAST(:input AS jsonb), CAST(:output AS jsonb),
                            :error, CAST(:config AS jsonb),
                            :agent_run_id, :model,
                            :prompt_tokens, :completion_tokens, :reasoning_tokens, :total_tokens)
                """),
                {
                    "eid": exec_id,
                    "bid": log["block_id"],
                    "btype": log.get("block_type", ""),
                    "bname": log.get("block_name", ""),
                    "status": log.get("status", "success"),
                    "ord": idx,
                    "dur": log.get("duration_ms"),
                    "input": json.dumps(safe_json(log.get("input", {}))),
                    "output": json.dumps(safe_json(log.get("output"))),
                    "error": log.get("error"),
                    "config": json.dumps(config),
                    "agent_run_id": log.get("agent_run_id"),
                    "model": model,
                    "prompt_tokens": tokens["prompt_tokens"],
                    "completion_tokens": tokens["completion_tokens"],
                    "reasoning_tokens": tokens["reasoning_tokens"],
                    "total_tokens": tokens["total_tokens"],
                },
            )
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to persist block logs: %s", e, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Structured error responses
# ---------------------------------------------------------------------------


class ErrorCode:
    """Standardized error codes for workflow routes."""

    WORKFLOW_NOT_FOUND = "WORKFLOW_NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"


def error_response(error: str, error_code: str, status: int, **extra):
    """Standardized error response for workflow routes."""
    from flask import jsonify

    body = {"error": error, "error_code": error_code, **extra}
    return jsonify(body), status
