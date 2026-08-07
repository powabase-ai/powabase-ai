"""Copilot routes — AI-powered workflow building assistant."""

import contextvars
import json
import logging
import os
import queue
import threading
import uuid
from queue import Empty

import redis
from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from sqlalchemy import text

from ..auth import require_auth
from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
from ..services.ai_provider_keys_resolver import project_has_byok_for_model
from ..services.copilot import get_copilot_model, run_copilot_chat
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
# Credit gate (BYOK-aware)
# ---------------------------------------------------------------------------
# A copilot turn's only credit charge is the recoupable `llm_call` (no flat
# dispatch fee, unlike agent/workflow runs). We gate on "any positive balance"
# rather than a precise per-turn estimate: a rejected turn never runs, and a turn
# that runs charges its real cost recoupably — driving the balance negative and
# blocking the NEXT turn. That stops unlimited turns at zero balance without
# guessing a turn's cost (a too-high estimate would wrongly block small balances).
_COPILOT_TURN_ESTIMATED_CREDITS = 1


def _gate_copilot_turn(model: str) -> None:
    """Pre-turn credit gate. No-op under BYOK; otherwise the port decides.

    Raises whatever the installed billing adapter raises — 402 out of credits,
    503 billing unreachable (fail-closed). NoopBillingAdapter never raises, which
    is the OSS-edition contract.
    """
    if project_has_byok_for_model(model):
        return
    billing.check_balance(estimated_cost=_COPILOT_TURN_ESTIMATED_CREDITS)


# A workflow's copilot session is long-lived, so history grows without bound.
# Feed the agent only a trailing window of the most recent messages (mirrors
# routes/project_copilot.py) — otherwise a long-lived session eventually
# exceeds the model context window, and an orphaned trailing 'user' row (see
# the turn-lock block below) would wedge the session FOREVER instead of aging
# out of the window.
_MAX_HISTORY_MESSAGES = 40

# ---------------------------------------------------------------------------
# In-flight turn lock (concurrent-turn safety)
# ---------------------------------------------------------------------------
# Nothing else serializes concurrent POST .../chat calls against a session.
# Two overlapping turns would each insert a 'user' message before either
# assistant reply lands (the reply is persisted by the background worker
# seconds later), producing two consecutive 'user' rows — the LLM provider
# 400s on the *next* turn. Reject a second turn outright instead of letting
# that happen: a Redis SET-NX+EX marker keyed on the session id, released via
# compare-and-delete when the turn finishes (mirrors routes/project_copilot.py).
# The EX TTL is a backstop only — it guarantees the marker can never wedge a
# session forever if a worker crashes/is killed before its release runs.
_TURN_LOCK_TTL_SECONDS = 360  # exceeds the 300s in-flight q.get timeout + margin
_RELEASE_TURN_LOCK_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] "
    "then return redis.call('del', KEYS[1]) else return 0 end"
)

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"))
    return _redis_client


def _turn_lock_key(session_id: str) -> str:
    return f"copilot_turn_lock:{session_id}"


def _try_acquire_turn_lock(session_id: str) -> str | None:
    """Mark a turn in-flight for this session. Returns a release token on
    success, or None if another turn already holds the lock.

    Fails OPEN on a Redis error (mirrors ``routes/project_copilot.py``): this
    lock exists to avoid a role-alternation 400 from the LLM provider, not to
    protect a safety-critical resource, so a Redis outage should degrade to
    "no serialization" rather than take chat down entirely.
    """
    token = uuid.uuid4().hex
    try:
        r = _get_redis()
        acquired = r.set(_turn_lock_key(session_id), token, nx=True, ex=_TURN_LOCK_TTL_SECONDS)
    except Exception:
        logger.warning("Failed to acquire copilot turn lock; failing open", exc_info=True)
        return token
    return token if acquired else None


def _release_turn_lock(session_id: str, token: str) -> None:
    """Compare-and-delete: only release if we still hold it, so a stale token
    (e.g. after our own TTL already expired) can't release a newer turn's lock."""
    try:
        _get_redis().eval(_RELEASE_TURN_LOCK_LUA, 1, _turn_lock_key(session_id), token)
    except Exception:
        logger.warning("Failed to release copilot turn lock", exc_info=True)


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

    # Reject a second concurrent turn on this session outright (see the
    # "In-flight turn lock" block above) rather than letting it race the
    # in-flight one and corrupt the history's user/assistant alternation.
    turn_token = _try_acquire_turn_lock(session_id)
    if turn_token is None:
        return jsonify({"error": "a copilot turn is already in progress"}), 409

    try:
        # Credit gate (parity with the project copilot): an AI-on-us turn bills
        # credits, so refuse it (402) when out of credits — a broke AI-on-us project
        # must not run unlimited turns into a negative balance. BYOK / billing-off
        # no-op. Before any persist so a rejected turn leaves no orphan row.
        _gate_copilot_turn(get_copilot_model())

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

        # Trailing window: take the most recent N by created_at, then re-order ASC
        # so the agent sees them chronologically (the just-inserted user message
        # is the newest, so it is always included). Mirrors routes/project_copilot.py.
        history_rows = db.session.execute(
            text(f"""
                SELECT role, content FROM (
                    SELECT role, content, created_at
                    FROM "{AI_SCHEMA}".copilot_messages
                    WHERE session_id = :sid
                    ORDER BY created_at DESC
                    LIMIT :lim
                ) recent
                ORDER BY created_at ASC
            """),
            {"sid": session_id, "lim": _MAX_HISTORY_MESSAGES},
        ).fetchall()

        messages = [{"role": r[0], "content": r[1]} for r in history_rows]
        # The window can begin with an assistant message (history is user/assistant
        # alternating and the newest row — always included — is the just-inserted
        # user turn, so an even-length window starts assistant-first). Anthropic 400s
        # unless the first non-system message is 'user', which would make EVERY turn
        # fail once a session passes _MAX_HISTORY_MESSAGES. Trim to a user boundary.
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        # A turn that died between its user-row commit and its assistant insert
        # (pod eviction/OOM — even the placeholder persist in run_agent below
        # never ran) leaves an orphaned 'user' row, so the window contains a
        # user,user pair the LLM provider 400s on. Repair the alternation for
        # the model's eyes only — substitute a structural placeholder assistant
        # message between consecutive user rows instead of dropping a row
        # (dropping would lose the user's words; placeholder-not-drop mirrors
        # project_copilot's _build_input_messages). The DB keeps the orphan;
        # the trailing window ages it out.
        repaired: list[dict] = []
        for m in messages:
            if repaired and m["role"] == "user" and repaired[-1]["role"] == "user":
                repaired.append(
                    {
                        "role": "assistant",
                        "content": "(No reply was recorded for the previous message.)",
                    }
                )
            repaired.append(m)
        messages = repaired

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
    except Exception:
        # Nothing kicked off a background turn (the worker thread below owns the
        # release from this point on) — release the lock ourselves so a 402/DB
        # failure here doesn't wedge the session until the TTL backstop expires.
        _release_turn_lock(session_id, turn_token)
        raise

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
                        # The reply was generated (and billed) but not saved. Don't
                        # report DONE-success: the history reload won't contain this
                        # answer, so the UI would flash it and then lose it on the
                        # next mount. Surface it as an error turn instead (parity
                        # with routes/project_copilot.py).
                        logger.error(
                            "Failed to persist assistant message: %s", persist_err, exc_info=True
                        )
                        db.session.rollback()
                        # The user row for this turn is already committed (~204-211).
                        # Without also landing an assistant row here, the session is
                        # left with a mid-history user,user pair that this copilot has
                        # no trailing window to age out of — 400ing every subsequent
                        # turn until the session is deleted. Persist a placeholder
                        # assistant row (best-effort) to keep the alternation valid,
                        # mirroring routes/project_copilot.py's round-2 I1 fix.
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
                                    "content": "Sorry — the reply could not be saved. Please try again.",
                                },
                            )
                            db.session.commit()
                        except Exception as placeholder_err:
                            logger.error(
                                "Failed to persist placeholder assistant message: %s",
                                placeholder_err,
                            )
                            db.session.rollback()
                        q.put(("DONE", None, None, persist_err))
                        return
                    q.put(("DONE", content, diff, None))
                except Exception as e:
                    # Persist error message so the user sees it on return. Log the
                    # real exception server-side but never echo it to the client —
                    # str(e) can leak internal details (parity with
                    # routes/project_copilot.py's generic error handling).
                    logger.error("Copilot agent error: %s", e, exc_info=True)
                    error_content = (
                        "Sorry — something went wrong while answering. Please try again."
                    )
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
                    # The turn is over (success or failure) — free the session
                    # for the next turn. This is the ONLY release path once the
                    # worker thread has started (see the try/except in `chat`
                    # for the pre-thread failure paths).
                    _release_turn_lock(session_id, turn_token)

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
            # Log the real exception server-side but never echo it to the
            # client — str(e) can leak internal details (parity with
            # routes/project_copilot.py's generic error handling).
            logger.error("Copilot chat error: %s", e, exc_info=True)
            _generic = "Sorry — something went wrong while answering. Please try again."
            error_event = {"event": "error", "error": _generic}
            yield f"data: {json.dumps(error_event)}\n\n"

    response = Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    # Belt-and-braces release: if the SSE generator is never iterated (e.g. the
    # client disconnects before the WSGI server starts streaming), the worker
    # thread that owns the "normal" release path (the `finally` in `run_agent`
    # above) never even starts, leaking the lock for up to the TTL. Release on
    # response teardown too — the compare-and-delete in `_release_turn_lock`
    # makes a double release (this plus the worker's) harmless.
    response.call_on_close(lambda: _release_turn_lock(session_id, turn_token))
    return response


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
