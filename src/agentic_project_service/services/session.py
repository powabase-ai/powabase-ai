"""
Session Service for agent conversations.

Provides functions for managing agent sessions and run history.
Sessions are stored in ai.agent_sessions table.
Runs are stored in ai.agent_runs table.
"""

import json
import logging
import re
import uuid
from datetime import UTC, datetime
from typing import Any

import litellm
from agentic.agent.message import Message
from agentic.execution.context import ExecutionContext
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from ..db import AI_SCHEMA
from ..models.tenant import AgentRunStatus
from .context_handler import _resolve_image_refs

logger = logging.getLogger(__name__)


def _safe_json_dumps(obj: Any, fallback: str = "null") -> str:
    """Serialize to JSON, falling back gracefully on non-serializable data."""
    try:
        return json.dumps(obj)
    except (TypeError, ValueError) as e:
        logger.warning("JSON serialization failed, using fallback: %s", e)
        return fallback


_CITATION_MARKER_RE = re.compile(r"\s*\[\d+\]")


def strip_citation_markers(text: str) -> str:
    """Remove citation markers like [1], [2] from text."""
    return _CITATION_MARKER_RE.sub("", text).strip()


# =============================================================================
# Typed usage + tool-call helpers (replaces `usage` / `tool_calls` JSONB)
# =============================================================================


def _as_int(value: Any) -> int | None:
    """Best-effort int coerce for dicts coming from LLM usage payloads."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unpack_usage(usage: dict | None) -> dict[str, int | None]:
    """Flatten a litellm usage dict into the typed agent_runs columns.

    Accepts both flat keys (prompt_tokens, completion_tokens, total_tokens)
    and litellm's nested details dicts (prompt_tokens_details, completion_tokens_details)
    for reasoning_tokens / cached_tokens.
    """
    if not usage:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "cached_tokens": None,
            "total_tokens": None,
        }

    prompt = _as_int(usage.get("prompt_tokens"))
    completion = _as_int(usage.get("completion_tokens"))
    total = _as_int(usage.get("total_tokens"))
    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)

    reasoning = _as_int(usage.get("reasoning_tokens"))
    if reasoning is None:
        details = usage.get("completion_tokens_details") or {}
        if isinstance(details, dict):
            reasoning = _as_int(details.get("reasoning_tokens"))

    cached = _as_int(usage.get("cached_tokens"))
    if cached is None:
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            cached = _as_int(details.get("cached_tokens"))

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "cached_tokens": cached,
        "total_tokens": total,
    }


def _summarize_tool_calls(tool_calls: list[dict] | None) -> dict[str, int]:
    """Aggregate a list of ToolCallRecord dicts into the 3 run-level summary cols.

    An error is heuristically detected by an 'Error...' prefix in the result
    string (matching how agentic.agent.agent records failures).
    """
    if not tool_calls:
        return {"count": 0, "error_count": 0, "duration_ms_total": 0}

    count = len(tool_calls)
    error_count = 0
    duration_total = 0
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        result_str = call.get("result")
        if isinstance(result_str, str) and result_str.lower().startswith("error"):
            error_count += 1
        dur = _as_int(call.get("duration_ms"))
        if dur is not None:
            duration_total += dur
    return {
        "count": count,
        "error_count": error_count,
        "duration_ms_total": duration_total,
    }


# =============================================================================
# Session Management
# =============================================================================


def get_or_create_session(
    db_session: Session,
    agent_id: str,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, str, bool]:
    """
    Get an existing session or create a new one.

    Args:
        db_session: SQLAlchemy session
        agent_id: UUID of the agent
        session_id: Optional user-facing session ID to look up
        user_id: Optional user ID to associate with new session
        metadata: Optional metadata for new session

    Returns:
        Tuple of (db_session_uuid, session_id, is_new)
    """
    if session_id:
        # Try to find existing session
        result = db_session.execute(
            text(
                f"""
                SELECT id, session_id FROM "{AI_SCHEMA}".agent_sessions
                WHERE session_id = :session_id AND agent_id = :agent_id
            """
            ),
            {"session_id": session_id, "agent_id": agent_id},
        )
        row = result.fetchone()
        if row:
            return str(row[0]), row[1], False

    # Create new session
    db_session_uuid = str(uuid.uuid4())
    new_session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
    now = datetime.now(UTC)

    db_session.execute(
        text(
            f"""
            INSERT INTO "{AI_SCHEMA}".agent_sessions
            (id, session_id, agent_id, user_id, session_data, metadata, created_at, updated_at)
            VALUES (:id, :session_id, :agent_id, :user_id, :session_data, :metadata, :created_at, :updated_at)
        """
        ),
        {
            "id": db_session_uuid,
            "session_id": new_session_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "session_data": "{}",
            "metadata": json.dumps(metadata) if metadata else "{}",
            "created_at": now,
            "updated_at": now,
        },
    )

    return db_session_uuid, new_session_id, True


def get_session_by_id(
    db_session: Session,
    session_id: str,
) -> dict[str, Any] | None:
    """
    Get a session by its user-facing session_id.

    Returns:
        Session dict or None if not found
    """
    result = db_session.execute(
        text(
            f"""
            SELECT id, session_id, agent_id, user_id, session_data, metadata, created_at, updated_at
            FROM "{AI_SCHEMA}".agent_sessions
            WHERE session_id = :session_id
        """
        ),
        {"session_id": session_id},
    )
    row = result.fetchone()
    if not row:
        return None

    return {
        "id": str(row[0]),
        "session_id": row[1],
        "agent_id": str(row[2]) if row[2] else None,
        "user_id": str(row[3]) if row[3] else None,
        "session_data": row[4] or {},
        "metadata": row[5] or {},
        "created_at": row[6].isoformat() if row[6] else None,
        "updated_at": row[7].isoformat() if row[7] else None,
    }


def update_session_timestamp(
    db_session: Session,
    db_session_uuid: str,
) -> None:
    """
    Update the session's updated_at timestamp.

    Called after each run to keep session timestamps current.
    """
    db_session.execute(
        text(
            f"""
            UPDATE "{AI_SCHEMA}".agent_sessions
            SET updated_at = :updated_at
            WHERE id = :id
        """
        ),
        {"id": db_session_uuid, "updated_at": datetime.now(UTC)},
    )


def list_sessions_for_agent(
    db_session: Session,
    agent_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    List all sessions for an agent.

    Returns:
        List of session dicts with run counts
    """
    result = db_session.execute(
        text(
            f"""
            SELECT
                s.id, s.session_id, s.agent_id, s.user_id,
                s.metadata, s.created_at, s.updated_at,
                COUNT(r.id) as run_count,
                MAX(r.created_at) as last_run_at
            FROM "{AI_SCHEMA}".agent_sessions s
            LEFT JOIN "{AI_SCHEMA}".agent_runs r ON r.session_id = s.id
            WHERE s.agent_id = :agent_id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            LIMIT :limit OFFSET :offset
        """
        ),
        {"agent_id": agent_id, "limit": limit, "offset": offset},
    )

    sessions = []
    for row in result:
        sessions.append(
            {
                "id": str(row[0]),
                "session_id": row[1],
                "agent_id": str(row[2]) if row[2] else None,
                "user_id": str(row[3]) if row[3] else None,
                "metadata": row[4] or {},
                "created_at": row[5].isoformat() if row[5] else None,
                "updated_at": row[6].isoformat() if row[6] else None,
                "run_count": row[7],
                "last_run_at": row[8].isoformat() if row[8] else None,
            }
        )

    return sessions


def get_session_owner(db_session: Session, session_id: str) -> str | None:
    """
    Return the user_id that owns a session, or None if session not found
    or has NULL user_id.

    Used for ownership checks before returning session data to a caller.
    """
    result = db_session.execute(
        text(
            f"""
            SELECT user_id FROM "{AI_SCHEMA}".agent_sessions
            WHERE session_id = :session_id
            """
        ),
        {"session_id": session_id},
    )
    row = result.fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])


def delete_session(
    db_session: Session,
    session_id: str,
) -> bool:
    """
    Delete a session and all its runs (via CASCADE).

    Returns:
        True if session was found and deleted, False otherwise
    """
    # Get DB UUID from session_id
    result = db_session.execute(
        text(
            f"""
            SELECT id FROM "{AI_SCHEMA}".agent_sessions
            WHERE session_id = :session_id
        """
        ),
        {"session_id": session_id},
    )
    row = result.fetchone()
    if not row:
        return False

    # Delete (runs cascade due to FK)
    db_session.execute(
        text(
            f"""
            DELETE FROM "{AI_SCHEMA}".agent_sessions
            WHERE id = :id
        """
        ),
        {"id": str(row[0])},
    )

    return True


# =============================================================================
# Run History
# =============================================================================


def load_session_history(
    db_session: Session,
    db_session_uuid: str,
    limit: int | None = None,
) -> list[Message]:
    """Load conversation history from agent_runs for a session.

    Returns Message objects in chronological order. Pydantic validates each
    message and preserves reasoning artifacts, tool_calls, and tool_call_id —
    all fields the prior dict-flatten implementation silently dropped.

    Citation markers are stripped from string content; everything else passes
    through unmodified.
    """
    query = f"""
        SELECT input_messages, output_messages
        FROM "{AI_SCHEMA}".agent_runs
        WHERE session_id = :session_id AND status = 'completed'
        ORDER BY created_at ASC
    """
    params: dict[str, Any] = {"session_id": db_session_uuid}
    if limit is not None:
        query += " LIMIT :limit"
        params["limit"] = limit

    result = db_session.execute(text(query), params)
    history: list[Message] = []

    for row in result:
        for raw in row[0] or []:
            if not isinstance(raw, dict) or raw.get("role") != "user":
                continue
            history.append(Message.model_validate(raw))
        for raw in row[1] or []:
            if not isinstance(raw, dict) or raw.get("role") != "assistant":
                continue
            msg = Message.model_validate(raw)
            if isinstance(msg.content, str):
                msg.content = strip_citation_markers(msg.content)
            history.append(msg)

    return history


def _resolve_agent_identity(
    db_session: Session,
    *,
    db_session_uuid: str | None,
    agent_id: str | None,
    model: str | None,
) -> tuple[str | None, str | None]:
    """Resolve (agent_id, model) from the parent session when the caller did
    not pass them. Keeps writer call-sites terse — they already know the
    session; the session knows its agent.
    """
    if agent_id and model:
        return agent_id, model
    if not db_session_uuid:
        return agent_id, model
    row = db_session.execute(
        text(
            f"""
            SELECT s.agent_id, a.model
            FROM "{AI_SCHEMA}".agent_sessions s
            LEFT JOIN "{AI_SCHEMA}".agents a ON a.id = s.agent_id
            WHERE s.id = :sid
            """
        ),
        {"sid": db_session_uuid},
    ).fetchone()
    if row is None:
        return agent_id, model
    return agent_id or (str(row[0]) if row[0] else None), model or row[1]


def _resolve_provider(model: str) -> str | None:
    """Return canonical provider name, or None if undeterminable.

    None preserves artifacts (custom endpoints / proxy URLs may legitimately
    resolve to unexpected providers). False drops cause silent quality loss.
    """
    try:
        _, provider, _, _ = litellm.get_llm_provider(model)
        return provider
    except Exception:
        return None


def build_messages_for_llm(
    session_history: list[Message],
    target_model: str,
    context: ExecutionContext,
    user_input: str | list[dict],
) -> list[dict]:
    """Convert session history Messages to LiteLLM input dicts, dropping
    cross-provider reasoning artifacts and emitting drop events on the SSE
    stream via context.emit_event.

    Spec §5.5 — runs at the route layer where both Message objects (from
    load_session_history) and the SSE context are in scope.
    """
    target_provider = _resolve_provider(target_model)
    messages_for_llm: list[dict] = []
    for msg in session_history:
        msg_dict = msg.to_litellm_input()
        if (
            target_provider is not None
            and msg.role == "assistant"
            and msg.reasoning is not None
            and msg.reasoning.provider != target_provider
        ):
            msg_dict.pop("thinking_blocks", None)
            psf = msg_dict.get("provider_specific_fields", {})
            psf.pop("encrypted_content_items", None)
            psf.pop("thought_signatures", None)
            if not psf:
                msg_dict.pop("provider_specific_fields", None)
            else:
                msg_dict["provider_specific_fields"] = psf
            context.emit_event(
                {
                    "type": "reasoning_dropped_at_provider_switch",
                    "from_provider": msg.reasoning.provider,
                    "to_provider": target_provider,
                }
            )
        messages_for_llm.append(msg_dict)
    if isinstance(user_input, str):
        messages_for_llm.append({"role": "user", "content": user_input})
    elif user_input and all(isinstance(item, dict) and "role" in item for item in user_input):
        messages_for_llm.extend(user_input)
    else:
        # Multimodal content array (items have `type` but no `role`) — wrap as
        # the content of one user message. Spreading these as top-level
        # messages produces role-less dicts that break agentic's
        # normalize_messages and get rejected by OpenAI as
        # "content: expected a string, got null."
        messages_for_llm.append({"role": "user", "content": user_input})
    return messages_for_llm


def persist_agent_run(
    db_session: Session,
    run_id: str,
    status: AgentRunStatus,
    input_messages: list[dict],
    *,
    db_session_uuid: str | None = None,
    output_messages: list[dict] | None = None,
    content: str | None = None,
    usage: dict | None = None,
    retrieved_context: list[dict] | None = None,
    error: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    context_handler_id: str | None = None,
    steps: int | None = None,
    events: list[dict] | None = None,
    tool_calls: list[dict] | None = None,
    reasoning_steps: list[dict] | None = None,
    parent_orchestration_run_id: str | None = None,
    parent_workflow_execution_id: str | None = None,
    agent_id: str | None = None,
    model: str | None = None,
) -> str:
    """Persist an agent run to the database.

    `usage` and `tool_calls` kwargs remain for caller ergonomics — they are
    unpacked into typed columns by _unpack_usage / _summarize_tool_calls
    (the ai.agent_runs.usage / tool_calls JSONB columns were dropped in 0019).

    When db_session_uuid is None (delegated / block runs), the row is written with
    session_id NULL and no session timestamp update is performed.

    Returns the run's UUID.
    """
    run_uuid = str(uuid.uuid4())
    now = datetime.now(UTC)

    resolved_agent_id, resolved_model = _resolve_agent_identity(
        db_session,
        db_session_uuid=db_session_uuid,
        agent_id=agent_id,
        model=model,
    )
    tokens = _unpack_usage(usage)
    tc = _summarize_tool_calls(tool_calls)

    db_session.execute(
        text(
            f"""
            INSERT INTO "{AI_SCHEMA}".agent_runs
            (id, session_id, run_id, context_handler_id, status, input_messages,
             output_messages, content, retrieved_context, error,
             started_at, completed_at, created_at, steps, events,
             reasoning_steps, parent_orchestration_run_id, parent_workflow_execution_id,
             agent_id, model,
             prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens, total_tokens,
             tool_call_count, tool_call_error_count, tool_call_duration_ms_total)
            VALUES (:id, :session_id, :run_id, :context_handler_id, :status,
                    CAST(:input_messages AS jsonb), CAST(:output_messages AS jsonb),
                    :content, CAST(:retrieved_context AS jsonb),
                    :error, :started_at, :completed_at, :created_at,
                    :steps, CAST(:events AS jsonb),
                    CAST(:reasoning_steps AS jsonb),
                    :parent_orchestration_run_id, :parent_workflow_execution_id,
                    :agent_id, :model,
                    :prompt_tokens, :completion_tokens, :reasoning_tokens,
                    :cached_tokens, :total_tokens,
                    :tool_call_count, :tool_call_error_count, :tool_call_duration_ms_total)
            """
        ),
        {
            "id": run_uuid,
            "session_id": db_session_uuid,
            "run_id": run_id,
            "context_handler_id": context_handler_id,
            "status": status.value,
            "input_messages": json.dumps(input_messages),
            "output_messages": json.dumps(output_messages) if output_messages else None,
            "content": content,
            "retrieved_context": json.dumps(retrieved_context) if retrieved_context else None,
            "error": error,
            "started_at": started_at or now,
            "completed_at": completed_at,
            "created_at": now,
            "steps": steps,
            "events": _safe_json_dumps(events, "[]") if events else None,
            "reasoning_steps": _safe_json_dumps(reasoning_steps, "[]") if reasoning_steps else None,
            "parent_orchestration_run_id": parent_orchestration_run_id,
            "parent_workflow_execution_id": parent_workflow_execution_id,
            "agent_id": resolved_agent_id,
            "model": resolved_model,
            "prompt_tokens": tokens["prompt_tokens"],
            "completion_tokens": tokens["completion_tokens"],
            "reasoning_tokens": tokens["reasoning_tokens"],
            "cached_tokens": tokens["cached_tokens"],
            "total_tokens": tokens["total_tokens"],
            "tool_call_count": tc["count"],
            "tool_call_error_count": tc["error_count"],
            "tool_call_duration_ms_total": tc["duration_ms_total"],
        },
    )

    # Persist per-tool detail rows into ai.tool_call_events.
    if tool_calls:
        _insert_tool_call_events(
            db_session,
            agent_run_uuid=run_uuid,
            agent_id=resolved_agent_id,
            model=resolved_model,
            tool_calls=tool_calls,
        )

    if db_session_uuid is not None:
        update_session_timestamp(db_session, db_session_uuid)

    return run_uuid


def _insert_tool_call_events(
    db_session: Session,
    *,
    agent_run_uuid: str,
    agent_id: str | None,
    model: str | None,
    tool_calls: list[dict],
) -> None:
    """Insert one ai.tool_call_events row per ToolCallRecord dict.

    Errors are heuristically detected by an 'Error...' prefix in the result
    string (matching how the ReAct loop records failures today). Both the
    full `arguments` / `result` JSONB and short text previews are written —
    the JSONB columns let API readers round-trip multimodal tool results
    (image_ref blocks etc.) without truncation.
    """
    if not tool_calls:
        return
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        args = call.get("arguments")
        # Text previews — None inputs stay None (the preview cols are nullable);
        # we don't want a missing field to be stored as the literal "null".
        result_str = (
            result
            if isinstance(result, str)
            else (_safe_json_dumps(result, "") if result is not None else None)
        )
        args_str = (
            args
            if isinstance(args, str)
            else (_safe_json_dumps(args, "") if args is not None else None)
        )
        is_error = isinstance(result_str, str) and result_str.lower().startswith("error")
        # JSON-serialize arguments/result for the JSONB columns. _safe_json_dumps
        # falls back to "null" rather than dropping the row, so the ledger
        # always advances even if a tool returns weird objects.
        args_json = _safe_json_dumps(args, "null") if args is not None else None
        result_json = _safe_json_dumps(result, "null") if result is not None else None
        db_session.execute(
            text(
                f"""
                INSERT INTO "{AI_SCHEMA}".tool_call_events
                (agent_run_id, agent_id, model, tool_name, status, duration_ms,
                 arguments, result, arguments_preview, result_preview, error, step)
                VALUES (:run_id, :agent_id, :model, :tool_name, :status,
                        :duration_ms,
                        CAST(:arguments AS jsonb), CAST(:result AS jsonb),
                        :arguments_preview, :result_preview,
                        :error, :step)
                """
            ),
            {
                "run_id": agent_run_uuid,
                "agent_id": agent_id,
                "model": model,
                "tool_name": str(call.get("tool_name") or "unknown")[:255],
                "status": "error" if is_error else "success",
                "duration_ms": _as_int(call.get("duration_ms")),
                "arguments": args_json,
                "result": result_json,
                "arguments_preview": (args_str or "")[:500] or None,
                "result_preview": (result_str or "")[:500] or None,
                "error": (result_str or "")[:1000] if is_error else None,
                "step": _as_int(call.get("step")),
            },
        )


def update_agent_run(
    db_session: Session,
    run_id: str,
    status: "AgentRunStatus | None" = None,
    content: str | None = None,
    output_messages: list[dict] | None = None,
    usage: dict | None = None,
    retrieved_context: list[dict] | None = None,
    error: str | None = None,
    completed_at: datetime | None = None,
    context_handler_id: str | None = None,
    steps: int | None = None,
    events: list[dict] | None = None,
    tool_calls: list[dict] | None = None,
    reasoning_steps: list[dict] | None = None,
    model: str | None = None,
) -> None:
    """
    Update an existing agent run by its user-facing run_id.

    Only updates fields that are explicitly provided (not None). The `usage`
    and `tool_calls` kwargs are unpacked into typed columns (the JSONB
    columns were dropped in migration 0019). When `tool_calls` is provided,
    existing ai.tool_call_events rows for this run are replaced with the new
    set so a partial update stays consistent.
    """
    updates = []
    params: dict[str, Any] = {"run_id": run_id}

    if status is not None:
        updates.append("status = :status")
        params["status"] = status.value
    if content is not None:
        updates.append("content = :content")
        params["content"] = content
    if output_messages is not None:
        updates.append("output_messages = CAST(:output_messages AS jsonb)")
        params["output_messages"] = json.dumps(output_messages)
    if usage is not None:
        tokens = _unpack_usage(usage)
        updates.append("prompt_tokens = :prompt_tokens")
        updates.append("completion_tokens = :completion_tokens")
        updates.append("reasoning_tokens = :reasoning_tokens")
        updates.append("cached_tokens = :cached_tokens")
        updates.append("total_tokens = :total_tokens")
        params.update(tokens)
    if retrieved_context is not None:
        updates.append("retrieved_context = CAST(:retrieved_context AS jsonb)")
        params["retrieved_context"] = json.dumps(retrieved_context)
    if error is not None:
        updates.append("error = :error")
        params["error"] = error
    if completed_at is not None:
        updates.append("completed_at = :completed_at")
        params["completed_at"] = completed_at
    if context_handler_id is not None:
        updates.append("context_handler_id = :context_handler_id")
        params["context_handler_id"] = context_handler_id
    if steps is not None:
        updates.append("steps = :steps")
        params["steps"] = steps
    if events is not None:
        updates.append("events = CAST(:events AS jsonb)")
        params["events"] = _safe_json_dumps(events, "[]")
    if tool_calls is not None:
        tc = _summarize_tool_calls(tool_calls)
        updates.append("tool_call_count = :tool_call_count")
        updates.append("tool_call_error_count = :tool_call_error_count")
        updates.append("tool_call_duration_ms_total = :tool_call_duration_ms_total")
        params["tool_call_count"] = tc["count"]
        params["tool_call_error_count"] = tc["error_count"]
        params["tool_call_duration_ms_total"] = tc["duration_ms_total"]
    if reasoning_steps is not None:
        updates.append("reasoning_steps = CAST(:reasoning_steps AS jsonb)")
        params["reasoning_steps"] = _safe_json_dumps(reasoning_steps, "[]")
    if model is not None:
        updates.append("model = :model")
        params["model"] = model

    if not updates:
        return

    db_session.execute(
        text(
            f"""
            UPDATE "{AI_SCHEMA}".agent_runs
            SET {", ".join(updates)}
            WHERE run_id = :run_id
        """
        ),
        params,
    )

    # Re-materialize tool_call_events for this run (idempotent replace).
    if tool_calls is not None:
        run_row = db_session.execute(
            text(
                f"""
                SELECT id, agent_id, model FROM "{AI_SCHEMA}".agent_runs
                WHERE run_id = :run_id
                """
            ),
            {"run_id": run_id},
        ).fetchone()
        if run_row:
            run_uuid = str(run_row[0])
            db_session.execute(
                text(
                    f"""
                    DELETE FROM "{AI_SCHEMA}".tool_call_events
                    WHERE agent_run_id = :run_id
                    """
                ),
                {"run_id": run_uuid},
            )
            _insert_tool_call_events(
                db_session,
                agent_run_uuid=run_uuid,
                agent_id=str(run_row[1]) if run_row[1] else None,
                model=run_row[2],
                tool_calls=tool_calls,
            )

    # Also update the parent session timestamp
    session_result = db_session.execute(
        text(
            f"""
            SELECT session_id FROM "{AI_SCHEMA}".agent_runs
            WHERE run_id = :run_id
        """
        ),
        {"run_id": run_id},
    )
    row = session_result.fetchone()
    if row and row[0]:
        update_session_timestamp(db_session, str(row[0]))


def get_run_by_id(
    db_session: Session,
    run_id: str,
) -> dict[str, Any] | None:
    """
    Get a run by its user-facing run_id.

    Returns:
        Run dict or None if not found
    """
    result = db_session.execute(
        text(
            f"""
            SELECT id, session_id, run_id, context_handler_id, status,
                   input_messages, output_messages, content,
                   retrieved_context, error, started_at, completed_at, created_at,
                   steps, events, reasoning_steps,
                   agent_id, model,
                   prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens, total_tokens,
                   tool_call_count, tool_call_error_count, tool_call_duration_ms_total
            FROM "{AI_SCHEMA}".agent_runs
            WHERE run_id = :run_id
        """
        ),
        {"run_id": run_id},
    )
    row = result.fetchone()
    if not row:
        return None

    run_uuid = str(row[0])
    tool_calls = _load_tool_calls_for_run(db_session, run_uuid)

    return {
        "id": run_uuid,
        "session_id": str(row[1]) if row[1] else None,
        "run_id": row[2],
        "context_handler_id": str(row[3]) if row[3] else None,
        "status": row[4],
        "input_messages": row[5] or [],
        "output_messages": row[6] or [],
        "content": row[7],
        "usage": _pack_usage(row),
        "retrieved_context": _resolve_image_refs(row[8]) if row[8] else row[8],
        "error": row[9],
        "started_at": row[10].isoformat() if row[10] else None,
        "completed_at": row[11].isoformat() if row[11] else None,
        "created_at": row[12].isoformat() if row[12] else None,
        "steps": row[13],
        "events": row[14] or [],
        "tool_calls": tool_calls,
        "reasoning_steps": row[15] or [],
        "agent_id": str(row[16]) if row[16] else None,
        "model": row[17],
    }


def _pack_usage(row: Any) -> dict[str, int] | None:
    """Re-assemble a `usage` dict from the typed cols for API back-compat.

    Row layout (get_run_by_id): [..., prompt_tokens, completion_tokens,
    reasoning_tokens, cached_tokens, total_tokens, tool_call_count,
    tool_call_error_count, tool_call_duration_ms_total]. Returns None when
    no token info is present.
    """
    prompt, completion, reasoning, cached, total = row[18], row[19], row[20], row[21], row[22]
    if all(v is None for v in (prompt, completion, reasoning, cached, total)):
        return None
    usage: dict[str, int] = {}
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if completion is not None:
        usage["completion_tokens"] = completion
    if reasoning is not None:
        usage["reasoning_tokens"] = reasoning
    if cached is not None:
        usage["cached_tokens"] = cached
    if total is not None:
        usage["total_tokens"] = total
    return usage


def _load_tool_calls_for_run(db_session: Session, run_uuid: str) -> list[dict]:
    """Hydrate the legacy tool_calls list from ai.tool_call_events.

    Prefers the JSONB `arguments` / `result` columns so multimodal tool
    payloads round-trip; falls back to the text previews for events that
    pre-date the JSONB columns. API shape (`tool_calls`: list of
    ToolCallRecord-ish dicts) stays stable for existing UI code.
    """
    result = db_session.execute(
        text(
            f"""
            SELECT step, tool_name, arguments, result, arguments_preview,
                   result_preview, duration_ms
            FROM "{AI_SCHEMA}".tool_call_events
            WHERE agent_run_id = :run_id
            ORDER BY COALESCE(step, 0), occurred_at
            """
        ),
        {"run_id": run_uuid},
    )
    calls: list[dict] = []
    for r in result:
        # `arguments` / `result` come back as already-parsed Python values
        # (psycopg2 maps JSONB → dict/list/scalar). Fall back to the preview
        # text when the JSONB col is null (legacy backfilled rows or
        # unserializable inputs).
        args_full, result_full = r[2], r[3]
        args_preview, result_preview = r[4], r[5]
        calls.append(
            {
                "step": r[0],
                "tool_name": r[1],
                "arguments": args_full if args_full is not None else args_preview,
                "result": result_full if result_full is not None else result_preview,
                "duration_ms": r[6],
            }
        )
    return calls


def _get_db_session_uuid(db_session: Session, session_id: str) -> str | None:
    """Resolve user-facing session_id to internal UUID."""
    session_result = db_session.execute(
        text(
            f"""
            SELECT id FROM "{AI_SCHEMA}".agent_sessions
            WHERE session_id = :session_id
        """
        ),
        {"session_id": session_id},
    )
    session_row = session_result.fetchone()
    if not session_row:
        return None
    return str(session_row[0])


def get_run_retrieved_context(
    db_session: Session,
    session_id: str,
    run_id: str,
) -> list[dict] | None:
    """Get retrieved_context for a run scoped to a session."""
    db_session_uuid = _get_db_session_uuid(db_session, session_id)
    if not db_session_uuid:
        return None

    result = db_session.execute(
        text(
            f"""
            SELECT retrieved_context
            FROM "{AI_SCHEMA}".agent_runs
            WHERE session_id = :session_id AND run_id = :run_id
        """
        ),
        {"session_id": db_session_uuid, "run_id": run_id},
    )
    row = result.fetchone()
    if not row:
        return None
    raw = row[0]
    return _resolve_image_refs(raw) if raw else raw


def list_runs_for_session(
    db_session: Session,
    session_id: str,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    List all runs for a session.

    Args:
        session_id: The user-facing session_id

    Returns:
        List of run dicts ordered by created_at DESC
    """
    db_session_uuid = _get_db_session_uuid(db_session, session_id)
    if not db_session_uuid:
        return []

    result = db_session.execute(
        text(
            f"""
            SELECT id, run_id, context_handler_id, status, input_messages,
                   output_messages, content, error,
                   started_at, completed_at, created_at, steps, events,
                   reasoning_steps,
                   prompt_tokens, completion_tokens, reasoning_tokens,
                   cached_tokens, total_tokens,
                   agent_id, model
            FROM "{AI_SCHEMA}".agent_runs
            WHERE session_id = :session_id
            ORDER BY created_at ASC
            LIMIT :limit OFFSET :offset
        """
        ),
        {"session_id": db_session_uuid, "limit": limit, "offset": offset},
    )

    rows = list(result)
    run_ids = [str(row[0]) for row in rows]
    citations_by_run = fetch_citations_for_runs(db_session, run_ids)
    tool_calls_by_run = _load_tool_calls_for_runs(db_session, run_ids)

    runs = []
    for row in rows:
        run_uuid = str(row[0])
        citations = citations_by_run.get(run_uuid, [])

        prompt, completion, reasoning, cached, total = row[14], row[15], row[16], row[17], row[18]
        usage: dict[str, int] | None = None
        if any(v is not None for v in (prompt, completion, reasoning, cached, total)):
            usage = {}
            if prompt is not None:
                usage["prompt_tokens"] = prompt
            if completion is not None:
                usage["completion_tokens"] = completion
            if reasoning is not None:
                usage["reasoning_tokens"] = reasoning
            if cached is not None:
                usage["cached_tokens"] = cached
            if total is not None:
                usage["total_tokens"] = total

        run_dict: dict[str, Any] = {
            "id": run_uuid,
            "run_id": row[1],
            "context_handler_id": str(row[2]) if row[2] else None,
            "status": row[3],
            "input_messages": row[4] or [],
            "output_messages": row[5] or [],
            "content": row[6],
            "usage": usage,
            "error": row[7],
            "started_at": row[8].isoformat() if row[8] else None,
            "completed_at": row[9].isoformat() if row[9] else None,
            "created_at": row[10].isoformat() if row[10] else None,
            "steps": row[11],
            "events": row[12] or [],
            "tool_calls": tool_calls_by_run.get(run_uuid, []),
            "reasoning_steps": row[13] or [],
            "agent_id": str(row[19]) if row[19] else None,
            "model": row[20],
        }
        if citations:
            run_dict["citations"] = citations
        runs.append(run_dict)

    return runs


def _load_tool_calls_for_runs(db_session: Session, run_uuids: list[str]) -> dict[str, list[dict]]:
    """Batch-hydrate tool_calls lists for many runs. Avoids N+1 reads."""
    if not run_uuids:
        return {}
    stmt = text(
        f"""
        SELECT agent_run_id, step, tool_name, arguments, result,
               arguments_preview, result_preview, duration_ms
        FROM "{AI_SCHEMA}".tool_call_events
        WHERE agent_run_id IN :run_ids
        ORDER BY agent_run_id, COALESCE(step, 0), occurred_at
        """
    ).bindparams(bindparam("run_ids", expanding=True))
    result = db_session.execute(stmt, {"run_ids": run_uuids})
    out: dict[str, list[dict]] = {}
    for r in result:
        rid = str(r[0])
        args_full, result_full = r[3], r[4]
        args_preview, result_preview = r[5], r[6]
        out.setdefault(rid, []).append(
            {
                "step": r[1],
                "tool_name": r[2],
                "arguments": args_full if args_full is not None else args_preview,
                "result": result_full if result_full is not None else result_preview,
                "duration_ms": r[7],
            }
        )
    return out


def fetch_citations_for_runs(db_session: Session, run_uuids: list[str]) -> dict[str, list[dict]]:
    """Fetch citations for many runs in one query."""
    if not run_uuids:
        return {}
    stmt = text(
        f"""
        SELECT c.run_id, c.citation_key, c.item_id, c.source_id, c.text_excerpt,
               c.meta, s.name AS source_name
        FROM "{AI_SCHEMA}".message_citations c
        LEFT JOIN "{AI_SCHEMA}".sources s ON s.id = c.source_id
        WHERE c.run_id IN :run_ids
        ORDER BY c.run_id, c.citation_key
    """
    ).bindparams(bindparam("run_ids", expanding=True))
    result = db_session.execute(stmt, {"run_ids": run_uuids})
    citations_by_run: dict[str, list[dict]] = {}
    for row in result:
        run_id = str(row[0])
        payload = {
            "key": str(row[1]),
            "item_id": str(row[2]) if row[2] else None,
            "source_id": str(row[3]) if row[3] else None,
            "text_excerpt": row[4],
            "meta": row[5] or {},
            "source_name": row[6] or "",
        }
        citations_by_run.setdefault(run_id, []).append(payload)
    return citations_by_run


def fetch_citations_for_run(db_session: Session, run_uuid: str) -> list[dict]:
    """Fetch citations for a run by its internal UUID."""
    return fetch_citations_for_runs(db_session, [run_uuid]).get(run_uuid, [])


def get_chat_messages(
    db_session: Session,
    session_id: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Get chat messages for a session in chronological order.

    This reconstructs the conversation as a list of user/assistant messages,
    suitable for displaying in a chat UI.

    Returns:
        List of message dicts with role, content, and metadata
    """
    # First get DB UUID from session_id
    session_result = db_session.execute(
        text(
            f"""
            SELECT id FROM "{AI_SCHEMA}".agent_sessions
            WHERE session_id = :session_id
        """
        ),
        {"session_id": session_id},
    )
    session_row = session_result.fetchone()
    if not session_row:
        return []

    db_session_uuid = str(session_row[0])

    # Get all completed runs
    query = f"""
        SELECT id, run_id, input_messages, output_messages, content, created_at,
               started_at, completed_at
        FROM "{AI_SCHEMA}".agent_runs
        WHERE session_id = :session_id AND status = 'completed'
        ORDER BY created_at ASC
    """
    params: dict[str, Any] = {"session_id": db_session_uuid}
    if limit is not None:
        query += " LIMIT :limit"
        params["limit"] = limit

    result = db_session.execute(
        text(query),
        params,
    )

    messages = []
    for row in result:
        run_uuid = str(row[0])
        run_id = row[1]
        input_msgs = row[2] or []
        output_msgs = row[3] or []
        content = row[4]
        created_at = row[5]
        started_at = row[6]
        completed_at = row[7]
        reasoning_duration_ms: int | None = None
        if started_at is not None and completed_at is not None:
            reasoning_duration_ms = int((completed_at - started_at).total_seconds() * 1000)

        # Add user message
        for msg in input_msgs:
            if msg.get("role") == "user":
                messages.append(
                    {
                        "role": "user",
                        "content": msg.get("content", ""),
                        "run_id": run_id,
                        "timestamp": created_at.isoformat() if created_at else None,
                    }
                )

        # Fetch citations for this run
        citations = fetch_citations_for_run(db_session, run_uuid)

        # Find the assistant message in output_msgs (used for reasoning replay
        # fields whether or not we use it as the primary message body).
        assistant_output_msg: dict[str, Any] | None = next(
            (m for m in output_msgs if isinstance(m, dict) and m.get("role") == "assistant"),
            None,
        )

        # Add assistant message
        if content:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "run_id": run_id,
                "timestamp": created_at.isoformat() if created_at else None,
            }
            if citations:
                assistant_msg["citations"] = citations
            if assistant_output_msg is not None:
                if assistant_output_msg.get("reasoning_requested"):
                    assistant_msg["reasoning_requested"] = assistant_output_msg[
                        "reasoning_requested"
                    ]
                    if reasoning_duration_ms is not None:
                        assistant_msg["reasoning_duration_ms"] = reasoning_duration_ms
                if assistant_output_msg.get("reasoning"):
                    assistant_msg["reasoning"] = assistant_output_msg["reasoning"]
            messages.append(assistant_msg)
        else:
            for msg in output_msgs:
                if msg.get("role") == "assistant":
                    assistant_msg = {
                        "role": "assistant",
                        "content": msg.get("content", ""),
                        "run_id": run_id,
                        "timestamp": created_at.isoformat() if created_at else None,
                    }
                    if citations:
                        assistant_msg["citations"] = citations
                    if msg.get("reasoning_requested"):
                        assistant_msg["reasoning_requested"] = msg["reasoning_requested"]
                        if reasoning_duration_ms is not None:
                            assistant_msg["reasoning_duration_ms"] = reasoning_duration_ms
                    if msg.get("reasoning"):
                        assistant_msg["reasoning"] = msg["reasoning"]
                    messages.append(assistant_msg)

    return messages
