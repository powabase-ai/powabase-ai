"""Copilot routes — AI-powered workflow building assistant."""

import contextvars
import json
import logging
import queue
import threading
import uuid
from queue import Empty

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from sqlalchemy import text

from ..auth import require_auth
from ..db import db, AI_SCHEMA
from ..services.copilot import run_copilot_chat
from ..services.copilot_config import (
    COPILOT_MODEL_OPTIONS,
    STATUS_MESSAGES,
    TOOL_STATUS,
)
from ..services.run_context import run_scope
from ..services.settings_registry import get_setting, validate_setting, SETTINGS_REGISTRY

logger = logging.getLogger(__name__)

copilot_bp = Blueprint("copilot", __name__, url_prefix="/api/copilot")


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


@copilot_bp.route("/sessions", methods=["POST"])
@require_auth
def create_session():
    """Create a new copilot session for a workflow."""
    data = request.get_json() or {}
    workflow_id = data.get("workflow_id")
    if not workflow_id:
        return jsonify({"error": "workflow_id is required"}), 400

    # Verify workflow exists
    exists = db.session.execute(
        text(f'SELECT 1 FROM "{AI_SCHEMA}".workflows WHERE id = :id'),
        {"id": workflow_id},
    ).fetchone()
    if not exists:
        return jsonify({"error": "Workflow not found"}), 404

    session_id = str(uuid.uuid4())
    db.session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".copilot_sessions (id, workflow_id)
            VALUES (:id, :wid)
        """),
        {"id": session_id, "wid": workflow_id},
    )
    db.session.commit()

    return jsonify({"id": session_id, "workflow_id": workflow_id}), 201


@copilot_bp.route("/sessions", methods=["GET"])
@require_auth
def get_session():
    """Get existing copilot session for a workflow."""
    workflow_id = request.args.get("workflow_id")
    if not workflow_id:
        return jsonify({"error": "workflow_id query param required"}), 400

    row = db.session.execute(
        text(f"""
            SELECT id, workflow_id, created_at, updated_at
            FROM "{AI_SCHEMA}".copilot_sessions
            WHERE workflow_id = :wid
            ORDER BY created_at DESC
            LIMIT 1
        """),
        {"wid": workflow_id},
    ).fetchone()

    if not row:
        return jsonify({"session": None})

    return jsonify(
        {
            "session": {
                "id": str(row[0]),
                "workflow_id": str(row[1]),
                "created_at": row[2].isoformat() if row[2] else None,
                "updated_at": row[3].isoformat() if row[3] else None,
            }
        }
    )


@copilot_bp.route("/sessions/<session_id>", methods=["DELETE"])
@require_auth
def delete_session(session_id: str):
    """Delete a copilot session (cascade deletes messages)."""
    db.session.execute(
        text(f'DELETE FROM "{AI_SCHEMA}".copilot_sessions WHERE id = :id'),
        {"id": session_id},
    )
    db.session.commit()
    return "", 204


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@copilot_bp.route("/sessions/<session_id>/messages", methods=["GET"])
@require_auth
def get_messages(session_id: str):
    """Get conversation history for a session."""
    rows = db.session.execute(
        text(f"""
            SELECT id, session_id, role, content, workflow_diff, pre_snapshot, created_at
            FROM "{AI_SCHEMA}".copilot_messages
            WHERE session_id = :sid
            ORDER BY created_at ASC
        """),
        {"sid": session_id},
    ).fetchall()

    messages = [
        {
            "id": str(r[0]),
            "session_id": str(r[1]),
            "role": r[2],
            "content": r[3],
            "workflow_diff": r[4],
            "pre_snapshot": r[5],
            "created_at": r[6].isoformat() if r[6] else None,
        }
        for r in rows
    ]

    return jsonify({"messages": messages})


@copilot_bp.route("/sessions/<session_id>/messages/<message_id>/snapshot", methods=["POST"])
@require_auth
def save_snapshot(session_id: str, message_id: str):
    """Store the pre-application snapshot on an assistant message."""
    data = request.get_json() or {}
    pre_snapshot = data.get("pre_snapshot")
    if not pre_snapshot:
        return jsonify({"error": "pre_snapshot is required"}), 400

    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".copilot_messages
            SET pre_snapshot = CAST(:snap AS jsonb)
            WHERE id = :mid AND session_id = :sid
        """),
        {"snap": json.dumps(pre_snapshot), "mid": message_id, "sid": session_id},
    )
    db.session.commit()

    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Chat (streaming)
# ---------------------------------------------------------------------------


@copilot_bp.route("/sessions/<session_id>/chat", methods=["POST"])
@require_auth
def chat(session_id: str):
    """Send a user message and stream the assistant response (SSE)."""
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    workflow_state = data.get("workflow_state", {"nodes": [], "edges": []})

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    # Verify session exists and get workflow_id
    session_row = db.session.execute(
        text(f'SELECT workflow_id FROM "{AI_SCHEMA}".copilot_sessions WHERE id = :id'),
        {"id": session_id},
    ).fetchone()
    if not session_row:
        return jsonify({"error": "Session not found"}), 404

    # Inject workflow_id so the copilot can query execution logs
    workflow_state["workflow_id"] = str(session_row[0])

    # Persist user message
    user_msg_id = str(uuid.uuid4())
    db.session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".copilot_messages (id, session_id, role, content)
            VALUES (:id, :sid, 'user', :content)
        """),
        {"id": user_msg_id, "sid": session_id, "content": user_message},
    )
    db.session.commit()

    # Load conversation history
    history_rows = db.session.execute(
        text(f"""
            SELECT role, content FROM "{AI_SCHEMA}".copilot_messages
            WHERE session_id = :sid
            ORDER BY created_at ASC
        """),
        {"sid": session_id},
    ).fetchall()

    messages = [{"role": r[0], "content": r[1]} for r in history_rows]

    # Update session timestamp
    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".copilot_sessions
            SET updated_at = now()
            WHERE id = :id
        """),
        {"id": session_id},
    )
    db.session.commit()

    def generate():
        q: queue.Queue = queue.Queue()
        assistant_msg_id = str(uuid.uuid4())

        # Tag every llm_call charge for this turn with assistant_msg_id.
        # Wrapping the entire body (including copy_context() below) keeps
        # the contextvar set when the worker-thread snapshot is taken,
        # and guarantees reset on every exit path.
        with run_scope(assistant_msg_id):
            yield from _do_generate(q, assistant_msg_id)

    def _do_generate(q, assistant_msg_id):
        """Body of ``generate``, separated so the public closure can wrap
        the entire SSE stream in a ``run_scope`` context. Closes over
        ``messages``, ``workflow_state``, ``session_id`` from ``chat``."""

        def on_event(event):
            """Translate ReAct events to SSE events and push to queue."""
            event_type = event.get("type", "")

            # tool_call → emit both tool_call event (existing contract) AND status
            if event_type == "tool_call":
                tool_name = event.get("tool_name", "")
                q.put(
                    {
                        "event": "tool_call",
                        "tool_call": {
                            "name": tool_name,
                            "arguments": event.get("arguments", {}),
                        },
                    }
                )
                status = TOOL_STATUS.get(tool_name, f"Using {tool_name}...")
                q.put({"event": "status", "message": status})
                return

            # reasoning_delta → stream the chunk; tag step so the FE can
            # group deltas per ReAct step
            if event_type == "reasoning_delta":
                q.put(
                    {
                        "event": "reasoning_delta",
                        "step": event.get("step"),
                        "delta": event.get("delta", ""),
                    }
                )
                return

            # Other events → emit status message if we have one
            status = STATUS_MESSAGES.get(event_type)
            if status:
                q.put({"event": "status", "message": status})

        # Capture the Flask app for the background thread — tool handlers
        # access db.session which requires an active application context.
        app = current_app._get_current_object()

        def run_agent():
            """Run the copilot agent and persist the assistant message.

            Persistence happens here (in the background thread) rather than
            in the SSE generator so the message is saved even if the client
            disconnects before the agent finishes.
            """
            with app.app_context():
                try:
                    content, diff = run_copilot_chat(
                        messages,
                        workflow_state,
                        on_event=on_event,
                    )
                    # Persist assistant message in the background thread
                    try:
                        db.session.execute(
                            text(f"""
                                INSERT INTO "{AI_SCHEMA}".copilot_messages
                                    (id, session_id, role, content, workflow_diff)
                                VALUES (:id, :sid, 'assistant', :content, CAST(:diff AS jsonb))
                            """),
                            {
                                "id": assistant_msg_id,
                                "sid": session_id,
                                "content": content or "(empty)",
                                "diff": json.dumps(diff) if diff else None,
                            },
                        )
                        db.session.commit()
                    except Exception as persist_err:
                        logger.error("Failed to persist assistant message: %s", persist_err)
                    q.put(("DONE", content, diff, None))
                except Exception as e:
                    # Persist error message so the user sees it on return
                    error_content = f"Error: {e}"
                    try:
                        db.session.execute(
                            text(f"""
                                INSERT INTO "{AI_SCHEMA}".copilot_messages
                                    (id, session_id, role, content, workflow_diff)
                                VALUES (:id, :sid, 'assistant', :content, NULL)
                            """),
                            {
                                "id": assistant_msg_id,
                                "sid": session_id,
                                "content": error_content,
                            },
                        )
                        db.session.commit()
                    except Exception as persist_err:
                        logger.error("Failed to persist error assistant message: %s", persist_err)
                    q.put(("DONE", None, None, e))
                finally:
                    db.session.remove()

        # Propagate Flask before_request contextvars (current_byok_providers,
        # byok_lookup_degraded, run_id_var) into the worker thread.
        # Same fix as orchestrations.py + agents.py; raw threading.Thread
        # doesn't inherit context, so BillingLogger inside the copilot's
        # LLM calls would read an empty BYOK set and charge every call to
        # AI-on-us even with a valid BYOK key. v1.5 BYOK-bypass invariant.
        _captured_ctx = contextvars.copy_context()
        thread = threading.Thread(target=lambda: _captured_ctx.run(run_agent), daemon=True)
        thread.start()

        assistant_content = ""
        workflow_diff = None

        try:
            while True:
                try:
                    item = q.get(timeout=300)
                except Empty:
                    raise TimeoutError("Copilot agent timed out after 300 seconds")
                if isinstance(item, tuple) and item[0] == "DONE":
                    _, assistant_content, workflow_diff, error = item
                    if error:
                        raise error
                    break
                # Forward tool_call and status events as SSE
                yield f"data: {json.dumps(item)}\n\n"

            # Message already persisted by run_agent thread
            complete_event = {
                "event": "complete",
                "message_id": assistant_msg_id,
                "content": assistant_content,
                "workflow_diff": workflow_diff,
            }
            yield f"data: {json.dumps(complete_event)}\n\n"

        except Exception as e:
            logger.error("Copilot chat error: %s", e, exc_info=True)
            error_event = {"event": "error", "error": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Project settings — copilot model
# ---------------------------------------------------------------------------


@copilot_bp.route("/settings/model", methods=["GET"])
@require_auth
def get_model_setting():
    """Get the configured copilot model for this project."""
    return jsonify(
        {
            "model": get_setting("copilot_model"),
            "default": SETTINGS_REGISTRY["copilot_model"].default,
            "options": [{"label": label, "value": v} for label, v in COPILOT_MODEL_OPTIONS],
        }
    )


@copilot_bp.route("/settings/model", methods=["PUT"])
@require_auth
def set_copilot_model():
    """Set the copilot model for this project."""
    data = request.get_json() or {}
    model = data.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    ok, msg = validate_setting("copilot_model", model)
    if not ok:
        return jsonify({"error": msg}), 400

    db.session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".project_settings (key, value, updated_at)
            VALUES ('copilot_model', :model, now())
            ON CONFLICT (key) DO UPDATE SET value = :model, updated_at = now()
        """),
        {"model": model},
    )
    db.session.commit()

    from flask import g

    g._settings_cache = None

    return jsonify({"ok": True, "model": model})
