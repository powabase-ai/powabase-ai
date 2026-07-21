"""Orchestration CRUD, entity management, and execution endpoints."""

import contextvars
import json
import logging
import os
import queue
import threading
import time
import uuid as _uuid

import litellm
from flask import Blueprint, Response, current_app, g, jsonify, request, stream_with_context
from sqlalchemy import text

from agentic.agent.message import Message
from agentic.execution.status import ExecutionStatus

from ..auth import get_current_user_id, require_auth
from ..db import db, AI_SCHEMA
from ..models.tenant import (
    Agent as AgentModel,
    AgentRun,
    AgentRunStatus,
    Hook,
    OrchestrationEntityModel,
    OrchestrationModel,
    OrchestrationRunModel,
    OrchestrationSessionModel,
)
from ..services import billing_port as billing
from ..services.ai_provider_keys_resolver import ProviderKeyDecryptDropped
from ..services.context_handler import resolve_tool_call_image_refs
from ..services.llm_availability import check_model_available
from ..services.run_context import (
    reset_run_id,
    set_run_id,
)
from ..services.session import _load_tool_calls_for_runs, persist_agent_run

logger = logging.getLogger(__name__)

# Pre-op estimate: 1 orchestration_run base fee + headroom for ~50 internal
# ops (delegated agent tool calls, retrievals).
# Pre-op balance gate: minimum free-tier balance (in millicents) required
# to START an orchestration. Caught during v1.5 launch smoke test 2026-05-27
# (post-merge, in feat/v15-key-gate-ux): a 5-agent Opus-4-1 supervisor cost
# ~50,000 mc in practice but the original 51 mc gate let any non-empty
# org start, so balance ran out mid-stream and 6 llm_calls returned
# 402 from billing-service — the LLM call had already executed (post-
# charge metering) so the platform absorbed ~$0.20 of Anthropic spend per
# blocked run with NO ledger debit. This bump catches the obvious "user
# is broke" case at run start. It does NOT eliminate mid-stream
# overrun for larger runs — that requires pre-authorization / reservation
# (architectural, follow-up issue). 20,000 mc ≈ $0.20 covers ~10 average
# Opus-4-1 calls which matches typical multi-agent supervisor cost.
_ORCHESTRATION_RUN_ESTIMATED_COST = 20_000

orchestrations_bp = Blueprint("orchestrations", __name__, url_prefix="/api/orchestrations")


def _load_sub_agent_models(orch_id: str) -> list[str]:
    """Return the model strings for every sub-agent referenced by the orchestration.

    Sub-agents are invoked via INTERNAL Python code (DelegateTool /
    SequentialEngine / ParallelEngine), not via HTTP, so they bypass
    ``run_agent``'s own check_model_available gate. We enumerate them here so
    the orchestration run handler can fail fast at entry when any sub-agent's
    model has neither a project BYOK key nor a platform env key — instead of
    surfacing LiteLLM's generic "Missing API Key" deep in a sub-agent's
    ReAct loop. Mirrors the build_orchestration enumeration in
    services/orchestration.py (entity_type='agent' rows only).
    """
    entities = (
        OrchestrationEntityModel.query.filter_by(orchestration_id=orch_id)
        .order_by(OrchestrationEntityModel.position)
        .all()
    )
    models: list[str] = []
    for entity in entities:
        if entity.entity_type != "agent":
            continue
        agent_row = db.session.get(AgentModel, entity.entity_ref_id)
        if agent_row is None or not agent_row.model:
            continue
        models.append(agent_row.model)
    return models


def _verify_orchestration_session_access(session_id: str):
    """Return None if caller may access session, else a (response, 404) tuple.

    Service-role callers bypass the check. For user-scoped callers, return 404
    (not 403) on both "not found" and "owned by someone else" to avoid leaking
    session existence. Mirrors routes/sessions.py:_verify_session_access.
    """
    jwt_payload = getattr(g, "jwt_payload", None) or {}
    if jwt_payload.get("is_service_role", False):
        return None

    session = OrchestrationSessionModel.query.filter_by(session_id=session_id).first()
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    caller = get_current_user_id()
    if session.user_id is None or str(session.user_id) != caller:
        return jsonify({"error": "Session not found"}), 404
    return None


def _require_uuid(value: str, label: str = "id"):
    """Return a 404 response tuple if value is not a valid UUID, else None."""
    try:
        _uuid.UUID(value)
        return None
    except ValueError:
        return jsonify({"error": f"Invalid {label}"}), 404


# ---------------------------------------------------------------------------
# Orchestration CRUD
# ---------------------------------------------------------------------------


@orchestrations_bp.route("", methods=["POST"])
@require_auth
def create_orchestration():
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"error": "name is required"}), 400

    orch = OrchestrationModel(
        name=name,
        description=data.get("description", ""),
        strategy=data.get("strategy", "supervisor"),
        orchestrator_config=data.get("orchestrator_config", {}),
        settings=data.get("settings", {}),
    )
    db.session.add(orch)
    db.session.commit()

    return (
        jsonify(
            {
                "id": str(orch.id),
                "name": orch.name,
                "description": orch.description,
                "strategy": orch.strategy,
                "orchestrator_config": orch.orchestrator_config,
                "settings": orch.settings,
            }
        ),
        201,
    )


@orchestrations_bp.route("", methods=["GET"])
@require_auth
def list_orchestrations():
    """List orchestrations.

    Back-compat: calls with no ``limit`` or ``offset`` return every row in
    the existing ``{ orchestrations: [...] }`` envelope (no pagination
    fields) to preserve the public API contract documented in
    api/_data/guides/orchestration.ts. Calls that send either param opt
    into the paginated envelope with ``total/limit/offset``.

    All calls receive the new per-row aggregate fields ``entity_count``,
    ``session_count``, and ``last_run_at`` (additive — existing customer
    code ignoring them keeps working).
    """
    from ..services.list_params import parse_list_params, escape_like, ListParamsError

    try:
        limit, offset, q, sort, order = parse_list_params(
            request,
            sort_allowed={"created_at", "name", "updated_at", "last_run_at"},
            default_unpaginated=True,
        )
    except ListParamsError as e:
        return jsonify({"error": str(e)}), e.status

    where_clause = ""
    params: dict = {}
    if q:
        where_clause = "WHERE o.name ILIKE :q_like"
        params["q_like"] = f"%{escape_like(q)}%"

    if sort == "last_run_at":
        order_by = f"last_run_at {order.upper()} NULLS LAST, o.id ASC"
    else:
        order_by = f"o.{sort} {order.upper()}, o.id ASC"

    limit_offset_clause = ""
    # When default_unpaginated=True, parse_list_params returns both as None or
    # both as ints — checking either is sufficient. Consistent with the
    # pagination-metadata gate further down (`if limit is not None`).
    if limit is not None:
        limit_offset_clause = "LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset

    rows_sql = f"""
        SELECT
          o.id, o.name, o.description, o.strategy, o.settings,
          o.created_at, o.updated_at,
          (SELECT COUNT(*) FROM "{AI_SCHEMA}".orchestration_entities WHERE orchestration_id = o.id) AS entity_count,
          (SELECT COUNT(*) FROM "{AI_SCHEMA}".orchestration_sessions WHERE orchestration_id = o.id) AS session_count,
          (SELECT MAX(orun.created_at)
             FROM "{AI_SCHEMA}".orchestration_runs orun
             JOIN "{AI_SCHEMA}".orchestration_sessions os ON os.id = orun.session_id
             WHERE os.orchestration_id = o.id) AS last_run_at
        FROM "{AI_SCHEMA}".orchestrations o
        {where_clause}
        ORDER BY {order_by}
        {limit_offset_clause}
    """
    rows = db.session.execute(text(rows_sql), params)

    orchestrations = []
    for row in rows:
        orchestrations.append(
            {
                "id": str(row.id),
                "name": row.name,
                "description": row.description,
                "strategy": row.strategy,
                "settings": row.settings,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "entity_count": int(row.entity_count),
                "session_count": int(row.session_count),
                "last_run_at": row.last_run_at.isoformat() if row.last_run_at else None,
            }
        )

    response: dict = {"orchestrations": orchestrations}

    # Include pagination metadata only when the caller opted into pagination.
    if limit is not None:
        count_sql = f'SELECT COUNT(*) FROM "{AI_SCHEMA}".orchestrations o {where_clause}'
        count_params = {k: v for k, v in params.items() if k not in ("limit", "offset")}
        total = db.session.execute(text(count_sql), count_params).scalar()
        response["total"] = total
        response["limit"] = limit
        response["offset"] = offset

    return jsonify(response)


@orchestrations_bp.route("/<orch_id>", methods=["GET"])
@require_auth
def get_orchestration(orch_id):
    orch = db.session.get(OrchestrationModel, orch_id)
    if not orch:
        return jsonify({"error": "Orchestration not found"}), 404

    entities = (
        OrchestrationEntityModel.query.filter_by(orchestration_id=orch_id)
        .order_by(OrchestrationEntityModel.position)
        .all()
    )

    return jsonify(
        {
            "id": str(orch.id),
            "name": orch.name,
            "description": orch.description,
            "strategy": orch.strategy,
            "orchestrator_config": orch.orchestrator_config,
            "settings": orch.settings,
            "entities": [
                {
                    "id": str(e.id),
                    "entity_type": e.entity_type,
                    "entity_ref_id": str(e.entity_ref_id),
                    "role_description": e.role_description,
                    "config": e.config,
                    "position": e.position,
                }
                for e in entities
            ],
        }
    )


@orchestrations_bp.route("/<orch_id>", methods=["PUT"])
@require_auth
def update_orchestration(orch_id):
    orch = db.session.get(OrchestrationModel, orch_id)
    if not orch:
        return jsonify({"error": "Orchestration not found"}), 404

    data = request.get_json()
    for field in ["name", "description", "strategy", "orchestrator_config", "settings"]:
        if field in data:
            setattr(orch, field, data[field])
    db.session.commit()

    return jsonify(
        {
            "id": str(orch.id),
            "name": orch.name,
            "description": orch.description,
            "strategy": orch.strategy,
            "orchestrator_config": orch.orchestrator_config,
            "settings": orch.settings,
            "created_at": orch.created_at.isoformat() if orch.created_at else None,
            "updated_at": orch.updated_at.isoformat() if orch.updated_at else None,
        }
    )


@orchestrations_bp.route("/<orch_id>", methods=["DELETE"])
@require_auth
def delete_orchestration(orch_id):
    orch = db.session.get(OrchestrationModel, orch_id)
    if not orch:
        return jsonify({"error": "Orchestration not found"}), 404
    db.session.delete(orch)
    db.session.commit()
    return jsonify({"deleted": True})


# ---------------------------------------------------------------------------
# Entity management
# ---------------------------------------------------------------------------


@orchestrations_bp.route("/<orch_id>/entities", methods=["POST"])
@require_auth
def add_entity(orch_id):
    orch = db.session.get(OrchestrationModel, orch_id)
    if not orch:
        return jsonify({"error": "Orchestration not found"}), 404

    data = request.get_json()
    entity_type = data.get("entity_type")
    entity_ref_id = data.get("entity_ref_id")
    if not entity_type or not entity_ref_id:
        return jsonify({"error": "entity_type and entity_ref_id are required"}), 400

    entity = OrchestrationEntityModel(
        orchestration_id=orch_id,
        entity_type=entity_type,
        entity_ref_id=entity_ref_id,
        role_description=data.get("role_description"),
        config=data.get("config", {}),
        position=data.get("position", 0),
    )
    db.session.add(entity)
    db.session.commit()

    return (
        jsonify(
            {
                "id": str(entity.id),
                "entity_type": entity_type,
                "entity_ref_id": str(entity.entity_ref_id),
                "role_description": entity.role_description,
            }
        ),
        201,
    )


@orchestrations_bp.route("/<orch_id>/entities", methods=["GET"])
@require_auth
def list_entities(orch_id):
    orch = db.session.get(OrchestrationModel, orch_id)
    if not orch:
        return jsonify({"error": "Orchestration not found"}), 404

    entities = (
        OrchestrationEntityModel.query.filter_by(orchestration_id=orch_id)
        .order_by(OrchestrationEntityModel.position)
        .all()
    )

    return jsonify(
        {
            "entities": [
                {
                    "id": str(e.id),
                    "entity_type": e.entity_type,
                    "entity_ref_id": str(e.entity_ref_id),
                    "role_description": e.role_description,
                    "config": e.config,
                    "position": e.position,
                }
                for e in entities
            ]
        }
    )


@orchestrations_bp.route("/<orch_id>/entities/<entity_id>", methods=["PUT"])
@require_auth
def update_entity(orch_id, entity_id):
    entity = db.session.get(OrchestrationEntityModel, entity_id)
    if not entity or str(entity.orchestration_id) != orch_id:
        return jsonify({"error": "Entity not found"}), 404

    data = request.get_json()
    for field in ["role_description", "config", "position"]:
        if field in data:
            setattr(entity, field, data[field])
    db.session.commit()

    return jsonify({"id": str(entity.id), "entity_type": entity.entity_type})


@orchestrations_bp.route("/<orch_id>/entities/<entity_id>", methods=["DELETE"])
@require_auth
def remove_entity(orch_id, entity_id):
    entity = db.session.get(OrchestrationEntityModel, entity_id)
    if not entity or str(entity.orchestration_id) != orch_id:
        return jsonify({"error": "Entity not found"}), 404
    db.session.delete(entity)
    db.session.commit()
    return jsonify({"deleted": True})


# ---------------------------------------------------------------------------
# Hook management
# ---------------------------------------------------------------------------


@orchestrations_bp.route("/<orch_id>/hooks", methods=["POST"])
@require_auth
def add_orchestration_hook(orch_id):
    """Add a hook to an orchestration."""
    orch = db.session.get(OrchestrationModel, orch_id)
    if not orch:
        return jsonify({"error": "Orchestration not found"}), 404

    data = request.get_json()
    event = data.get("event")
    hook_type = data.get("type")
    config = data.get("config")
    if not event or not hook_type or config is None:
        return jsonify({"error": "event, type, and config are required"}), 400

    hook = Hook(
        orchestration_id=orch_id,
        event=event,
        type=hook_type,
        matcher=data.get("matcher"),
        config=config,
        enabled=data.get("enabled", True),
        position=data.get("position", 0),
    )
    db.session.add(hook)
    db.session.commit()

    return jsonify(
        {
            "id": str(hook.id),
            "orchestration_id": orch_id,
            "event": hook.event,
            "matcher": hook.matcher,
            "type": hook.type,
            "config": hook.config,
            "enabled": hook.enabled,
            "position": hook.position,
            "created_at": hook.created_at.isoformat() if hook.created_at else None,
        }
    ), 201


@orchestrations_bp.route("/<orch_id>/hooks", methods=["GET"])
@require_auth
def list_orchestration_hooks(orch_id):
    """List hooks configured for an orchestration."""
    orch = db.session.get(OrchestrationModel, orch_id)
    if not orch:
        return jsonify({"error": "Orchestration not found"}), 404

    hooks = Hook.query.filter_by(orchestration_id=orch_id).order_by(Hook.position).all()
    return jsonify(
        {
            "hooks": [
                {
                    "id": str(h.id),
                    "event": h.event,
                    "matcher": h.matcher,
                    "type": h.type,
                    "config": h.config,
                    "enabled": h.enabled,
                    "position": h.position,
                    "created_at": h.created_at.isoformat() if h.created_at else None,
                }
                for h in hooks
            ]
        }
    )


@orchestrations_bp.route("/<orch_id>/hooks/<hook_id>", methods=["DELETE"])
@require_auth
def delete_orchestration_hook(orch_id, hook_id):
    """Remove a hook from an orchestration."""
    err = _require_uuid(orch_id, "orchestration id") or _require_uuid(hook_id, "hook id")
    if err:
        return err
    hook = db.session.get(Hook, hook_id)
    if not hook or str(hook.orchestration_id) != orch_id:
        return jsonify({"error": "Hook not found"}), 404
    db.session.delete(hook)
    db.session.commit()
    return jsonify({"deleted": True})


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@orchestrations_bp.route("/<orch_id>/sessions", methods=["GET"])
@require_auth
def list_orchestration_sessions(orch_id: str):
    """List sessions for an orchestration with run counts.

    Scoped to the authenticated user unless the caller is service-role
    (which sees all sessions). Mirrors the agent path's pattern at
    routes/agents.py:list_sessions.
    """
    is_service_role = (getattr(g, "jwt_payload", None) or {}).get("is_service_role", False)
    scoped_user_id = None if is_service_role else get_current_user_id()

    query = OrchestrationSessionModel.query.filter_by(orchestration_id=orch_id)
    if scoped_user_id is not None:
        query = query.filter_by(user_id=scoped_user_id)
    sessions = query.order_by(OrchestrationSessionModel.created_at.desc()).limit(100).all()

    result = []
    for s in sessions:
        runs = OrchestrationRunModel.query.filter_by(session_id=s.id).all()
        last_run = (
            max(runs, key=lambda r: r.created_at or r.started_at, default=None) if runs else None
        )
        first_input = None
        for r in runs:
            if r.input_messages and isinstance(r.input_messages, list):
                for msg in r.input_messages:
                    if isinstance(msg, dict) and msg.get("content"):
                        first_input = msg["content"]
                        break
            if first_input:
                break

        result.append(
            {
                "session_id": s.session_id,
                "run_count": len(runs),
                "first_message": first_input,
                "last_activity_at": (
                    last_run.completed_at or last_run.started_at or last_run.created_at
                ).isoformat()
                if last_run
                else None,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
        )

    return jsonify({"sessions": result, "total": len(result)})


def _summary_from_events(events: list[dict]) -> str | None:
    """Reconstruct a reasoning summary from persisted terminal `reasoning`
    events (the route filters out reasoning_delta but keeps terminal
    `reasoning` per agent.py:577). Returns None when no reasoning text is
    present so the FE pill renders 'done-empty' rather than 'done-full' with
    blank content."""
    parts: list[str] = []
    for e in events or []:
        if not isinstance(e, dict):
            continue
        if e.get("type") == "reasoning":
            content = e.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
    if not parts:
        return None
    return "\n\n".join(parts)


@orchestrations_bp.route("/<orch_id>/sessions/<session_id>/messages", methods=["GET"])
@require_auth
def get_orchestration_session_messages(orch_id: str, session_id: str):
    """Get assembled messages for an orchestration session.

    Each assistant message carries per-run reasoning replay metadata so the
    FE ReasoningPill renders correctly after page refresh:
      - ``reasoning_requested`` (bool) — gates whether the pill renders
      - ``reasoning_duration_ms`` (int | None) — drives "Thought for X.Xs"
      - ``reasoning`` ({summary_text, thinking_blocks}) — drives done-empty
        vs done-full classification
      - ``events`` (list) — raw events for buildReasoningSteps replay
    The legacy top-level ``events`` field is preserved for back-compat with
    older FE builds; new FE code reads per-message events instead.
    """
    denial = _verify_orchestration_session_access(session_id)
    if denial is not None:
        return denial

    session = OrchestrationSessionModel.query.filter_by(session_id=session_id).first()
    if not session:
        return jsonify({"error": "Session not found"}), 404

    runs = (
        OrchestrationRunModel.query.filter_by(session_id=session.id)
        .order_by(OrchestrationRunModel.created_at.asc())
        .all()
    )

    # Hydrate tool_calls for delegated child agent_runs of this session's
    # orchestration runs. tool_call_events.agent_run_id references
    # ai.agent_runs, never ai.orchestration_runs — so we resolve the
    # parent_orchestration_run_id → agent_run_id mapping first, then load
    # tool calls keyed by agent_run UUID. We build this BEFORE the message
    # loop so each assistant_msg can carry its own run's tool_calls — the
    # FE's per-message buildTraceTree(typed, msg.tool_calls) needs them
    # attached on the message, not flattened at the top level.
    orch_run_uuids = [str(r.id) for r in runs]
    child_agent_runs = (
        AgentRun.query.filter(AgentRun.parent_orchestration_run_id.in_(orch_run_uuids))
        .order_by(AgentRun.parent_orchestration_run_id, AgentRun.created_at.asc())
        .all()
        if orch_run_uuids
        else []
    )
    agent_run_uuids = [str(ar.id) for ar in child_agent_runs]
    tool_calls_by_agent_run = _load_tool_calls_for_runs(db.session, agent_run_uuids)

    child_runs_by_orch: dict[str, list[str]] = {}
    for ar in child_agent_runs:
        child_runs_by_orch.setdefault(str(ar.parent_orchestration_run_id), []).append(str(ar.id))

    messages = []
    all_events = []
    all_tool_calls: list[dict] = []
    for run in runs:
        if run.input_messages:
            for msg in (
                run.input_messages if isinstance(run.input_messages, list) else [run.input_messages]
            ):
                if isinstance(msg, dict):
                    messages.append(
                        {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                    )

        run_events = run.events if isinstance(run.events, list) else []

        # Tool calls for this orchestration run, in (agent_run.created_at,
        # tool_call_events.step) order — matches event ordering inside
        # nested delegation envelopes so `buildTraceTree`'s tool_name +
        # order-of-appearance enrichment lines up.
        run_tool_calls: list[dict] = []
        for ar_uuid in child_runs_by_orch.get(str(run.id), []):
            run_tool_calls.extend(tool_calls_by_agent_run.get(ar_uuid, []))
        # Resolve image_ref blocks to inline base64 data URLs so the
        # browser can render them without a separate signed-URL fetch.
        run_tool_calls = resolve_tool_call_image_refs(run_tool_calls)

        assistant_msg: dict | None = None
        if run.content:
            assistant_msg = {
                "role": "assistant",
                "content": run.content,
                "run_id": run.run_id,
            }
        elif run.error:
            assistant_msg = {
                "role": "assistant",
                "content": f"Error: {run.error}",
                "run_id": run.run_id,
            }

        if assistant_msg is not None:
            if run.reasoning_requested:
                assistant_msg["reasoning_requested"] = True
                if run.started_at is not None and run.completed_at is not None:
                    assistant_msg["reasoning_duration_ms"] = int(
                        (run.completed_at - run.started_at).total_seconds() * 1000
                    )
                summary = _summary_from_events(run_events)
                # Keep thinking_blocks empty — we filter reasoning_delta from
                # persistence so we can't reconstruct structured blocks. The
                # FE only uses thinking_blocks to detect redacted_thinking
                # (Anthropic safety policy, rare). summary_text drives the
                # done-empty/done-full classification.
                assistant_msg["reasoning"] = {
                    "summary_text": summary,
                    "thinking_blocks": [],
                }
            if run_events:
                assistant_msg["events"] = run_events
            if run_tool_calls:
                assistant_msg["tool_calls"] = run_tool_calls
            messages.append(assistant_msg)

        if run_events:
            all_events.extend(run_events)
        if run_tool_calls:
            all_tool_calls.extend(run_tool_calls)

    return jsonify(
        {
            "session_id": session_id,
            "messages": messages,
            "events": all_events,
            "tool_calls": all_tool_calls,
        }
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@orchestrations_bp.route("/<orch_id>/run/stream", methods=["POST"])
@require_auth
def run_orchestration_stream(orch_id: str):
    """Run an orchestration with SSE streaming."""
    data = request.get_json()
    message = data.get("message")
    if not message:
        return jsonify({"error": "message is required"}), 400

    session_id = data.get("session_id")
    user_id = get_current_user_id()

    # Pre-op balance check (free-tier hard cap) via the billing port. Done
    # outside the generator so 402/503 propagate as a normal HTTP error
    # response BEFORE the SSE stream begins. The no-op adapter makes this
    # inert in OSS/unit-test/local-dev builds.
    billing.check_balance(estimated_cost=_ORCHESTRATION_RUN_ESTIMATED_COST)

    # Fail fast when the orchestrator model has neither a project BYOK key nor
    # a platform env key. Done outside the SSE generator so 400 propagates as
    # a normal HTTP error response (not as a stream event). Mirrors the
    # generator's own model lookup precedence (orchestrator_config.model >
    # settings.model > default).
    orch_row = db.session.get(OrchestrationModel, orch_id)
    if orch_row is not None:
        raw_settings = orch_row.settings or {}
        orch_config = raw_settings.get("orchestrator_config") or orch_row.orchestrator_config or {}
        orchestrator_model_to_check = (
            orch_config.get("model") or raw_settings.get("model") or "gpt-5.4"
        )
        check_model_available(orchestrator_model_to_check)

        # Also validate every sub-agent's model. Sub-agents are invoked
        # internally (not via HTTP), so they bypass run_agent's own gate —
        # without this loop, a BYOK-only sub-agent surfaces its missing key
        # mid-delegation rather than at run entry.
        for sub_agent_model in _load_sub_agent_models(orch_id):
            check_model_available(sub_agent_model)

    def generate_sse():
        run_id = None
        abort_event = threading.Event()
        event_queue = queue.Queue()

        # Streaming flag + content buffer + delta counter — initialized once at
        # request entry (issue #106 / Task 13). The buffer accumulates
        # ``content_delta`` payloads across all step iterations and feeds three
        # downstream uses: the terminal SSE ``chunk`` event, the ``complete``
        # event's ``content`` field, and the persisted ``run.content`` column.
        # ``events_for_db`` is declared here too so the outer failure handler
        # can pass it to ``update_orchestration_run`` (M2 v3).
        streaming_enabled = os.getenv("AGENT_LLM_STREAMING_ENABLED", "true").lower() == "true"
        content_buffer = ""
        delta_count = 0
        events_for_db: list[dict] = []
        started_at_monotonic = time.monotonic()

        try:
            from agentic.execution.context import ExecutionContext

            from ..services.hook_loader import load_hooks_for_orchestration
            from ..services.orchestration import (
                build_orchestration,
                create_orchestration_run,
                get_or_create_orchestration_session,
                update_orchestration_run,
            )

            orch_row, orchestration = build_orchestration(orch_id)

            hooks = load_hooks_for_orchestration(orch_id)  # noqa: F841 — used when orchestration engine supports hooks

            db_session_uuid, actual_session_id, is_new = get_or_create_orchestration_session(
                orchestration_id=orch_id,
                session_id=session_id,
                user_id=user_id,
            )

            # Load chat history from prior completed runs in this session
            history: list[dict] = []
            if not is_new:
                prior_runs = (
                    OrchestrationRunModel.query.filter_by(session_id=db_session_uuid)
                    .filter(OrchestrationRunModel.status == "completed")
                    .order_by(OrchestrationRunModel.created_at.asc())
                    .all()
                )
                for prior_run in prior_runs:
                    if prior_run.input_messages:
                        msgs = (
                            prior_run.input_messages
                            if isinstance(prior_run.input_messages, list)
                            else [prior_run.input_messages]
                        )
                        for msg in msgs:
                            if isinstance(msg, dict):
                                history.append(
                                    {
                                        "role": msg.get("role", "user"),
                                        "content": msg.get("content", ""),
                                    }
                                )
                    if prior_run.content:
                        history.append({"role": "assistant", "content": prior_run.content})

            # Compute reasoning_requested for the orchestrator agent so the FE
            # ReasoningPill renders. Mirrors the agents-route precheck pattern
            # (effort set + model supports reasoning); the agents route can
            # reuse the constructed Agent's _resolved_effort_for, but here the
            # orchestrator Agent isn't built until inside the worker thread, so
            # we run the same litellm.supports_reasoning probe inline. We
            # compute this BEFORE create_orchestration_run so the row is
            # persisted with the flag — the messages endpoint reads it back on
            # replay (post-refresh) to render the pill on historical runs.
            orchestrator_effort = (orchestration.orchestrator_config or {}).get("reasoning_effort")
            orchestrator_model = (
                (orchestration.orchestrator_config or {}).get("model")
                or (orchestration.settings or {}).get("model")
                or "gpt-5.4"
            )
            reasoning_requested_flag = False
            if orchestrator_effort:
                try:
                    reasoning_requested_flag = bool(
                        litellm.supports_reasoning(model=orchestrator_model)
                    )
                except Exception:
                    reasoning_requested_flag = False

            db_run_uuid, run_id = create_orchestration_run(
                session_uuid=db_session_uuid,
                orchestration_id=orch_id,
                message=message,
                reasoning_requested=reasoning_requested_flag,
            )
            db.session.commit()
            start_event = {
                "event": "start",
                "run_id": run_id,
                "session_id": actual_session_id,
                "reasoning_requested": reasoning_requested_flag,
            }
            yield f"data: {json.dumps(start_event)}\n\n"

            def on_event(e):
                event_queue.put(e)

            context = ExecutionContext(
                execution_id=run_id,
                session_id=actual_session_id,
                user_id=user_id,
                orchestration_run_id=db_run_uuid,
                on_event=on_event,
                abort_signal=abort_event,
            )

            # Capture app object before entering threads (current_app proxy is not safe across threads)
            app_for_threads = current_app._get_current_object()

            # Build persistence callback for delegated agent runs.
            # Called by DelegateTool / SequentialEngine / ParallelEngine after each sub-agent run.
            _status_map = {
                ExecutionStatus.COMPLETED: AgentRunStatus.COMPLETED,
                ExecutionStatus.FAILED: AgentRunStatus.FAILED,
                ExecutionStatus.CANCELLED: AgentRunStatus.FAILED,
            }

            def _persist_delegate_run(payload: dict) -> None:
                from flask import has_app_context

                from .agents import extract_reasoning_steps

                def _do_persist() -> None:
                    mapped_status = _status_map.get(payload.get("status"), AgentRunStatus.FAILED)
                    messages = payload.get("messages") or []
                    reasoning = extract_reasoning_steps(messages) if messages else None
                    # Fall back to the known orch run uuid if the engine/tool hook
                    # didn't propagate context.orchestration_run_id. Prevents child
                    # agent_runs from being orphaned (parent FK NULL) when a future
                    # code path forgets to thread the context through.
                    parent_orch_id = payload.get("orchestration_run_id") or db_run_uuid
                    try:
                        persist_agent_run(
                            db_session=db.session,
                            run_id=payload.get("child_execution_id")
                            or f"delegate_{_uuid.uuid4().hex[:12]}",
                            status=mapped_status,
                            input_messages=[{"role": "user", "content": payload.get("task") or ""}],
                            output_messages=[
                                Message(
                                    role="assistant",
                                    content=payload.get("content") or "",
                                ).model_dump(exclude_none=True)
                            ],
                            content=payload.get("content"),
                            usage=payload.get("usage"),
                            error=payload.get("error"),
                            started_at=payload.get("started_at"),
                            completed_at=payload.get("completed_at"),
                            steps=payload.get("steps"),
                            events=payload.get("events"),
                            tool_calls=payload.get("tool_calls"),
                            reasoning_steps=reasoning,
                            parent_orchestration_run_id=parent_orch_id,
                            # agent_id is threaded from services/orchestration.py
                            # → OrchestrationEntity → DelegateTool →
                            # on_run_complete payload. Without this, the
                            # delegate path passes db_session_uuid=None and
                            # _resolve_agent_identity early-returns, persisting
                            # agent_id NULL on every child agent_run.
                            agent_id=payload.get("agent_id"),
                            model=payload.get("model"),
                        )
                        db.session.commit()
                    except Exception:
                        db.session.rollback()
                        logger.exception(
                            "Failed to persist delegate agent_run for orch %s; continuing",
                            parent_orch_id,
                        )

                if has_app_context():
                    _do_persist()
                else:
                    with app_for_threads.app_context():
                        _do_persist()

            # Run orchestration in background thread so we can yield events in real-time
            result_holder: list = []
            error_holder: list = []

            def run_orchestration():
                # ContextVar values don't propagate across raw threading.Thread
                # boundaries. Bind orchestration run_id here so wrapped tool
                # handlers fired in sub-agent ReAct loops derive deterministic
                # idempotency keys instead of falling back to uuid4. Spec
                # line 132: retry must collide on UNIQUE(org_id, idem_key).
                _orch_run_id_token = set_run_id(run_id)
                try:
                    with billing.llm_call_scope():
                        output = orchestration.run(
                            input=message,
                            context=context,
                            history=history if history else None,
                            on_delegate_complete=_persist_delegate_run,
                        )
                    result_holder.append(output)
                except Exception as e:
                    error_holder.append(e)
                finally:
                    reset_run_id(_orch_run_id_token)
                    event_queue.put(None)  # sentinel to signal completion

            # Propagate Flask before_request contextvars (current_byok_providers,
            # byok_lookup_degraded, run_id_var, etc.) into the worker
            # thread. Raw threading.Thread does NOT inherit context — the
            # default frozenset() leaks through and BillingLogger's BYOK skip
            # at billing_litellm.py:137 reads as empty, charging every call
            # against AI-on-us even when the project has a valid BYOK key
            # registered. Caught during local v1.5 smoke test 2026-05-27:
            # 11 llm_call rows landed with non-zero unit_credits despite a
            # valid is_valid=true row in ai.ai_provider_keys for anthropic.
            # Same pattern as agentic.agent.agent + agentic.orchestration.
            # strategies which already use copy_context().run.
            _captured_ctx = contextvars.copy_context()
            worker = threading.Thread(
                target=lambda: _captured_ctx.run(run_orchestration), daemon=True
            )
            worker.start()

            # Yield events as they arrive (keeps the SSE stream alive).
            # Buffer-aware drain (β): content_delta accumulates into
            # content_buffer and is NOT persisted; reasoning_delta is forwarded
            # but NOT persisted; terminal events are persisted + forwarded.
            while True:
                try:
                    event = event_queue.get(timeout=30)
                except queue.Empty:
                    # Send SSE comment as keepalive to prevent proxy timeout
                    yield ": keepalive\n\n"
                    continue
                if event is None:  # sentinel — orchestration finished
                    break

                event_type = event.get("type")

                # β: capture content_delta into the buffer
                if event_type == "content_delta":
                    content_buffer += event.get("delta", "")
                    delta_count += 1
                    yield f"data: {json.dumps({'event': event_type, **event})}\n\n"
                    continue

                # reasoning_delta: forwarded only, NOT persisted
                if event_type == "reasoning_delta":
                    delta_count += 1
                    yield f"data: {json.dumps({'event': event_type, **event})}\n\n"
                    continue

                # Terminal events: persist + forward
                events_for_db.append(event)
                yield f"data: {json.dumps({'event': event.get('type', 'event'), **event})}\n\n"

            worker.join(timeout=5)

            # Handle errors from the worker thread
            if error_holder:
                raise error_holder[0]

            if not result_holder:
                raise RuntimeError("Orchestration thread finished without producing a result")

            output = result_holder[0]

            # β: terminal chunk + complete + persist all use the same
            # final_content to keep live SSE, post-complete refetch, and
            # reload-from-DB consistent (B1 fix).
            final_content = content_buffer if streaming_enabled else output.content

            # M7 observability log — one line per terminal-of-run
            duration_ms = int((time.monotonic() - started_at_monotonic) * 1000)
            logger.info(
                "agent_stream_run completed run=%s status=%s deltas=%d streamed_bytes=%d duration_ms=%d",
                run_id,
                output.status.name,
                delta_count,
                len(content_buffer),
                duration_ms,
            )

            # Terminal chunk
            if final_content:
                yield f"data: {json.dumps({'event': 'chunk', 'content': final_content})}\n\n"

            # Persist result (B1: content=final_content, M2 v3: events=events_for_db).
            # The supervisor's model lives on the OrchestrationOutput.coordination_metadata
            # dict (set by the SupervisorEngine); other strategies leave it empty, in
            # which case we fall back to the orchestration's configured model.
            run_status = "completed" if output.status.is_success() else "failed"
            supervisor_model = (
                (output.coordination_metadata or {}).get("model")
                or (orchestration.settings or {}).get("model")
                or (orch_row.settings or {}).get("model")
            )
            update_orchestration_run(
                run_id=run_id,
                status=run_status,
                content=final_content,
                events=events_for_db,
                usage=output.usage,
                error=output.error,
                model=supervisor_model,
            )
            db.session.commit()

            # Billing: post the orchestration_run dispatch fee only on
            # success — failed runs aren't charged. Delegated agent runs are
            # tracked via parent_orchestration_run_id but the dispatch fee for
            # those inner agents is intentionally NOT posted here. Their
            # internal atomic ops (tool calls, retrievals) are already billed
            # by Task 15 inside tool_registry / knowledge_search.
            if run_status == "completed":
                billing.charge(
                    action="orchestration_run",
                    quantity=1,
                    ref_type="orchestration_run",
                    ref_id=run_id,
                    idempotency_parts=(run_id,),
                    metadata={
                        "orchestration_id": str(orch_id),
                        "model": supervisor_model,
                    },
                )

            # Complete event — content from final_content (B1)
            complete_event = {
                "event": "complete",
                "run_id": run_id,
                "session_id": actual_session_id,
                "content": final_content,
                "status": run_status,
                "steps": output.steps,
                "usage": output.usage,
                # Surface the agent/orchestration's error message in the
                # complete envelope so clients that only read `complete`
                # still see what went wrong (the dedicated `event=error`
                # only fires on outer exceptions, not on handled-and-stored
                # failures from a downstream LLM rejection).
                "error": output.error if run_status == "failed" else None,
            }
            yield f"data: {json.dumps(complete_event)}\n\n"

        except GeneratorExit:
            abort_event.set()
            logger.info(
                "Client disconnected during orchestration run %s",
                run_id or "?",
            )
            return

        except ProviderKeyDecryptDropped as e:
            # Same actionable error contract as routes/agents.py 400 response
            # (issue #246). Run isn't persisted because the typed exception is
            # raised by build_orchestration BEFORE create_orchestration_run,
            # so run_id is still None.
            logger.warning("Orchestration aborted: provider key decrypt drop (%s)", e.provider)
            err_payload = {
                "event": "error",
                "error": str(e),
                "code": "provider_key_decrypt_failed",
                "provider": e.provider,
            }
            yield f"data: {json.dumps(err_payload)}\n\n"
            return

        except Exception as e:
            logger.exception("Orchestration run failed")
            error_msg = str(e)
            try:
                if run_id is not None:
                    from ..services.orchestration import update_orchestration_run

                    # M2 v3: persist events_for_db on failure too — synthetic
                    # terminal chunk/reasoning events emitted before the error
                    # must reach the events JSONB column.
                    update_orchestration_run(
                        run_id=run_id,
                        status="failed",
                        error=error_msg,
                        events=events_for_db,
                    )
                    db.session.commit()
            except Exception:
                logger.exception("Failed to persist orchestration error")
            yield f"data: {json.dumps({'event': 'error', 'error': error_msg})}\n\n"

    return Response(stream_with_context(generate_sse()), mimetype="text/event-stream")


@orchestrations_bp.route("/runs/<run_id>", methods=["GET"])
@require_auth
def get_orchestration_run(run_id: str):
    """Get orchestration run status, events, and child runs."""
    run = OrchestrationRunModel.query.filter_by(run_id=run_id).first()
    if not run:
        return jsonify({"error": "Run not found"}), 404

    # Get child agent runs
    child_runs = AgentRun.query.filter_by(parent_orchestration_run_id=str(run.id)).all()

    # Hydrate tool_calls (with full multimodal results) for each delegated
    # agent_run. tool_call_events.agent_run_id references ai.agent_runs only,
    # so we look up by the child runs' UUIDs. The orchestration_run row
    # itself never has rows in tool_call_events. Without this the UI falls
    # back to the truncated `result_preview` in the streamed events, which
    # renders multimodal payloads as the literal "[multimodal content]".
    child_run_uuids = [str(cr.id) for cr in child_runs]
    tool_calls_by_run = _load_tool_calls_for_runs(db.session, child_run_uuids)

    return jsonify(
        {
            "run_id": run.run_id,
            "orchestration_id": (str(run.orchestration_id) if run.orchestration_id else None),
            "status": run.status,
            "content": run.content,
            "events": run.events,
            "usage": run.usage,
            "model": run.model,
            "error": run.error,
            "started_at": (run.started_at.isoformat() if run.started_at else None),
            "completed_at": (run.completed_at.isoformat() if run.completed_at else None),
            "child_runs": [
                {
                    "run_id": cr.run_id,
                    "status": cr.status,
                    "steps": cr.steps,
                    "content": cr.content,
                    "events": cr.events,
                    # Resolve image_ref blocks to inline base64 data URLs so
                    # the browser can render them without a separate signed-URL
                    # fetch — same treatment as sessions.get_runs.
                    "tool_calls": resolve_tool_call_image_refs(
                        tool_calls_by_run.get(str(cr.id), [])
                    ),
                    "usage": cr.usage,
                    "model": cr.model,
                }
                for cr in child_runs
            ],
        }
    )
