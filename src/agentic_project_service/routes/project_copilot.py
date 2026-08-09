"""Project Copilot routes — project-scoped onboarding/guidance assistant.

Mirrors the workflow copilot (routes/copilot.py): session fetch/create plus an
SSE chat stream. Differences: the session is project-scoped (one resumable
session per project, no workflow_id), and the stream can emit a ``trigger_guide``
event telling the front-end to launch a guide-bubble walkthrough.

Billing parity with the workflow copilot: the whole stream is wrapped in
``run_scope(assistant_msg_id)`` so each LLM call inside the run is charged
(or BYOK-skipped) exactly like an agent/workflow LLM call.
"""

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
from sqlalchemy.exc import IntegrityError

from ..auth import require_auth
from ..db import db, AI_SCHEMA
from ..services.project_copilot import PROJECT_COPILOT_MODEL, run_project_copilot_chat
from ..services.run_context import run_scope
from .copilot import _gate_copilot_turn

logger = logging.getLogger(__name__)

project_copilot_bp = Blueprint("project_copilot", __name__, url_prefix="/api/project-copilot")

# There is exactly one non-expiring session per project, so history grows without
# bound. Feed the agent only a trailing window of the most recent messages —
# otherwise a long-lived project eventually exceeds the model context window and
# every turn fails until the session is reset.
_MAX_HISTORY_MESSAGES = 40

# ---------------------------------------------------------------------------
# In-flight turn lock (concurrent-turn safety)
# ---------------------------------------------------------------------------
# The session is a per-project singleton, but nothing else serializes concurrent
# POST .../chat calls against it. Two overlapping turns would each insert a
# 'user' message before either assistant reply lands (the reply is persisted by
# the background worker seconds later), producing two consecutive 'user' rows —
# Anthropic 400s on the *next* turn because the trailing-window trim only strips
# a leading assistant message, not a duplicated user pair. Reject a second turn
# outright instead of letting that happen: a Redis SET-NX+EX marker keyed on the
# session id, released via compare-and-delete when the turn finishes (mirrors
# the lock pattern in tasks/docs_refresh.py). The EX TTL is a backstop only —
# it guarantees the marker can never wedge a session forever if a worker
# crashes/is killed before its release runs.
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
    return f"project_copilot_turn_lock:{session_id}"


def _try_acquire_turn_lock(session_id: str) -> str | None:
    """Mark a turn in-flight for this session. Returns a release token on
    success, or None if another turn already holds the lock.

    Fails OPEN on a Redis error (mirrors ``routes/internal_docs.py``'s rate
    limiter): this lock exists to avoid a role-alternation 400 from the LLM
    provider, not to protect a safety-critical resource, so a Redis outage
    should degrade to "no serialization" rather than take chat down entirely.
    """
    token = uuid.uuid4().hex
    try:
        r = _get_redis()
        acquired = r.set(_turn_lock_key(session_id), token, nx=True, ex=_TURN_LOCK_TTL_SECONDS)
    except Exception:
        logger.warning("Failed to acquire project copilot turn lock; failing open", exc_info=True)
        return token
    return token if acquired else None


def _release_turn_lock(session_id: str, token: str) -> None:
    """Compare-and-delete: only release if we still hold it, so a stale token
    (e.g. after our own TTL already expired) can't release a newer turn's lock."""
    try:
        _get_redis().eval(_RELEASE_TURN_LOCK_LUA, 1, _turn_lock_key(session_id), token)
    except Exception:
        logger.warning("Failed to release project copilot turn lock", exc_info=True)


# ---------------------------------------------------------------------------
# Session (one resumable session per project)
# ---------------------------------------------------------------------------


@project_copilot_bp.route("/sessions", methods=["POST"])
@require_auth
def create_session():
    """Get-or-create the project's single copilot session.

    The table is a per-project singleton (enforced by a one-row unique index), so
    concurrent creates (two tabs, React strict-mode double-mount) converge on the
    same row instead of splitting the chat history across duplicates.
    """
    existing = _current_session_id()
    if existing:
        return jsonify({"id": existing}), 200

    session_id = str(uuid.uuid4())
    try:
        db.session.execute(
            text(f'INSERT INTO "{AI_SCHEMA}".project_copilot_sessions (id) VALUES (:id)'),
            {"id": session_id},
        )
        db.session.commit()
        return jsonify({"id": session_id}), 201
    except IntegrityError:
        # Lost the race to a concurrent create — return the winning row.
        db.session.rollback()
        existing = _current_session_id()
        return jsonify({"id": existing}), 200


def _current_session_id() -> str | None:
    row = db.session.execute(
        text(f"""
            SELECT id FROM "{AI_SCHEMA}".project_copilot_sessions
            ORDER BY created_at DESC LIMIT 1
        """)
    ).fetchone()
    return str(row[0]) if row else None


@project_copilot_bp.route("/sessions", methods=["GET"])
@require_auth
def get_session():
    """Get the project's most recent copilot session (or null)."""
    row = db.session.execute(
        text(f"""
            SELECT id, created_at, updated_at
            FROM "{AI_SCHEMA}".project_copilot_sessions
            ORDER BY created_at DESC LIMIT 1
        """)
    ).fetchone()
    if not row:
        return jsonify({"session": None})
    return jsonify(
        {
            "session": {
                "id": str(row[0]),
                "created_at": row[1].isoformat() if row[1] else None,
                "updated_at": row[2].isoformat() if row[2] else None,
            }
        }
    )


@project_copilot_bp.route("/sessions/<session_id>/messages", methods=["GET"])
@require_auth
def get_messages(session_id: str):
    """Conversation history for a session."""
    rows = db.session.execute(
        text(f"""
            SELECT id, session_id, role, content, guide_event, created_at
            FROM "{AI_SCHEMA}".project_copilot_messages
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
            "guide_event": r[4],
            "created_at": r[5].isoformat() if r[5] else None,
        }
        for r in rows
    ]
    return jsonify({"messages": messages})


@project_copilot_bp.route("/sessions/<session_id>", methods=["DELETE"])
@require_auth
def delete_session(session_id: str):
    """Delete a session (cascade deletes messages) — lets a user reset the chat."""
    db.session.execute(
        text(f'DELETE FROM "{AI_SCHEMA}".project_copilot_sessions WHERE id = :id'),
        {"id": session_id},
    )
    db.session.commit()
    return "", 204


# ---------------------------------------------------------------------------
# Chat (streaming)
# ---------------------------------------------------------------------------


@project_copilot_bp.route("/sessions/<session_id>/chat", methods=["POST"])
@require_auth
def chat(session_id: str):
    """Send a user message and stream the assistant response (SSE)."""
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    session_row = db.session.execute(
        text(f'SELECT 1 FROM "{AI_SCHEMA}".project_copilot_sessions WHERE id = :id'),
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
        # Credit gate: an AI-on-us turn (no BYOK key for the model) bills credits,
        # so refuse it (402) when the project is out of credits — otherwise a
        # broke AI-on-us project could run unlimited turns into a negative
        # balance. BYOK and billing-off both no-op. Raised BEFORE persisting the
        # user message so a rejected turn leaves no orphan row.
        _gate_copilot_turn(PROJECT_COPILOT_MODEL)

        # Persist user message
        db.session.execute(
            text(f"""
                INSERT INTO "{AI_SCHEMA}".project_copilot_messages (id, session_id, role, content)
                VALUES (:id, :sid, 'user', :content)
            """),
            {"id": str(uuid.uuid4()), "sid": session_id, "content": user_message},
        )
        db.session.commit()

        # Trailing window: take the most recent N by created_at, then re-order ASC
        # so the agent sees them chronologically (the just-inserted user message
        # is the newest, so it is always included).
        history_rows = db.session.execute(
            text(f"""
                SELECT role, content, guide_event FROM (
                    SELECT role, content, guide_event, created_at
                    FROM "{AI_SCHEMA}".project_copilot_messages
                    WHERE session_id = :sid
                    ORDER BY created_at DESC
                    LIMIT :lim
                ) recent
                ORDER BY created_at ASC
            """),
            {"sid": session_id, "lim": _MAX_HISTORY_MESSAGES},
        ).fetchall()
        messages = [{"role": r[0], "content": r[1], "guide_event": r[2]} for r in history_rows]
        # The window can begin with an assistant message (history is user/assistant
        # alternating and the newest row — always included — is the just-inserted
        # user turn, so an even-length window starts assistant-first). Anthropic 400s
        # unless the first non-system message is 'user', which would make EVERY turn
        # fail once a session passes _MAX_HISTORY_MESSAGES. Trim to a user boundary.
        while messages and messages[0]["role"] != "user":
            messages.pop(0)

        db.session.execute(
            text(
                f'UPDATE "{AI_SCHEMA}".project_copilot_sessions SET updated_at = now() WHERE id = :id'
            ),
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
        # Tag every llm_call charge for this turn (parity with workflow copilot).
        #
        # KNOWN LIMITATION (accepted, rare — not currently fixed): billing is
        # charged per LLM call INSIDE agent.run(), before we know whether the turn
        # will be reported to the user as failed. If the generation SUCCEEDS (and
        # is charged) but the turn is then surfaced as an error — either the
        # assistant-message INSERT below fails, or this generator's 300s q.get
        # timeout fires while the daemon worker is still running — the user sees an
        # error and retries, and the retry runs (and charges) a second time. So an
        # AI-on-us turn can be double-charged; BYOK just re-bills the user's own
        # provider. Ordinary generation failures (agent.run raising before any LLM
        # call) do NOT double-charge. A real fix needs turn-level idempotency, a
        # cancellation token for the abandoned worker, or ledger reconciliation —
        # all touch the shared billing/agent path; deferred as low-frequency.
        with run_scope(assistant_msg_id):
            yield from _do_generate(q, assistant_msg_id)

    def _do_generate(q, assistant_msg_id):
        def on_event(event):
            # Surface tool calls so the UI can show "Searching the docs…" etc.,
            # and stream reasoning deltas so the UI can show the model's live
            # thinking. (The copilot streams no assistant content tokens; the
            # final text arrives in the `complete` event and is reloaded from
            # the persisted history.)
            event_type = event.get("type", "")
            if event_type == "tool_call":
                q.put(
                    {
                        "event": "tool_call",
                        "tool_call": {
                            "name": event.get("tool_name", ""),
                            "arguments": event.get("arguments", {}),
                        },
                    }
                )
            elif event_type == "reasoning_delta":
                # Live extended-thinking chunk (Opus 4.8 @ medium effort). `step`
                # groups deltas per ReAct step (mirrors the workflow copilot).
                q.put(
                    {
                        "event": "reasoning_delta",
                        "step": event.get("step"),
                        "delta": event.get("delta", ""),
                    }
                )

        app = current_app._get_current_object()

        def run_agent():
            with app.app_context():
                try:
                    content, guide_sequence_id, docs_notice = run_project_copilot_chat(
                        messages, on_event=on_event
                    )
                    try:
                        db.session.execute(
                            text(f"""
                                INSERT INTO "{AI_SCHEMA}".project_copilot_messages
                                    (id, session_id, role, content, guide_event)
                                VALUES (:id, :sid, 'assistant', :content, CAST(:guide AS jsonb))
                            """),
                            {
                                "id": assistant_msg_id,
                                "sid": session_id,
                                # Persist exactly what was streamed in the `complete`
                                # event below (including "" for a guide-only turn with
                                # no prose) — the column is NOT NULL but allows "", so
                                # a reload renders identically to what streamed instead
                                # of showing a literal "(empty)" placeholder string.
                                "content": content or "",
                                "guide": json.dumps({"sequence_id": guide_sequence_id})
                                if guide_sequence_id
                                else None,
                            },
                        )
                        db.session.commit()
                    except Exception as persist_err:
                        # The reply was generated (and billed) but not saved. Don't
                        # report DONE-success: the history reload won't contain this
                        # answer, so the UI would flash it and then lose it on the
                        # next mount. Surface it as an error turn (consistent with a
                        # worker crash) so the user gets a clear failure + retry.
                        logger.error(
                            "Failed to persist assistant message: %s", persist_err, exc_info=True
                        )
                        db.session.rollback()
                        # The user row for this turn is already committed (~248-255).
                        # Without also landing an assistant row here, the session is
                        # left with a mid-window user,user pair that the leading-
                        # window trim can't remove — 400ing every turn until it ages
                        # out of the trailing window. Persist a placeholder assistant
                        # row (best-effort) to keep the alternation valid, mirroring
                        # the agent-error path below.
                        try:
                            db.session.execute(
                                text(f"""
                                    INSERT INTO "{AI_SCHEMA}".project_copilot_messages
                                        (id, session_id, role, content)
                                    VALUES (:id, :sid, 'assistant', :content)
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
                        q.put(("DONE", None, None, persist_err, None))
                        return
                    q.put(("DONE", content, guide_sequence_id, None, docs_notice))
                except Exception as e:
                    logger.error("Project copilot worker error: %s", e, exc_info=True)
                    error_content = "Sorry — something went wrong while answering. Please try again."
                    try:
                        db.session.execute(
                            text(f"""
                                INSERT INTO "{AI_SCHEMA}".project_copilot_messages
                                    (id, session_id, role, content)
                                VALUES (:id, :sid, 'assistant', :content)
                            """),
                            {"id": assistant_msg_id, "sid": session_id, "content": error_content},
                        )
                        db.session.commit()
                    except Exception as persist_err:
                        logger.error("Failed to persist error message: %s", persist_err)
                    q.put(("DONE", None, None, e, None))
                finally:
                    db.session.remove()
                    # The turn is over (success or failure) — free the session
                    # for the next turn. This is the ONLY release path once the
                    # worker thread has started (see the try/except above for
                    # the pre-thread failure paths).
                    _release_turn_lock(session_id, turn_token)

        # Propagate BYOK/billing contextvars into the worker thread (see
        # routes/copilot.py for the v1.5 BYOK-bypass rationale).
        _captured_ctx = contextvars.copy_context()
        thread = threading.Thread(target=lambda: _captured_ctx.run(run_agent), daemon=True)
        thread.start()

        assistant_content = ""
        guide_sequence_id = None
        docs_notice = None
        try:
            while True:
                try:
                    item = q.get(timeout=300)
                except Empty:
                    raise TimeoutError("Project copilot agent timed out after 300 seconds")
                if isinstance(item, tuple) and item[0] == "DONE":
                    _, assistant_content, guide_sequence_id, error, docs_notice = item
                    if error:
                        raise error
                    break
                yield f"data: {json.dumps(item)}\n\n"

            # Warn the UI if the answer wasn't grounded (docs endpoint unreachable/
            # unconfigured) so it can show a "answered without docs" notice rather
            # than silently presenting a confident, unsourced reply.
            if docs_notice:
                yield f"data: {json.dumps({'event': 'notice', 'kind': docs_notice})}\n\n"

            # Tell the UI to launch a guide-bubble walkthrough, if one was chosen.
            if guide_sequence_id:
                yield f"data: {json.dumps({'event': 'trigger_guide', 'sequence_id': guide_sequence_id})}\n\n"

            complete_event = {
                "event": "complete",
                "message_id": assistant_msg_id,
                "content": assistant_content,
            }
            yield f"data: {json.dumps(complete_event)}\n\n"
        except Exception as e:
            logger.error("Project copilot chat error: %s", e, exc_info=True)
            _generic = "Sorry — something went wrong while answering. Please try again."
            yield f"data: {json.dumps({'event': 'error', 'error': _generic})}\n\n"

    response = Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    # Belt-and-braces release: if the SSE generator is never iterated (e.g. the
    # client disconnects before the WSGI server starts streaming), the worker
    # thread that owns the "normal" release path (the `finally` in `run_agent`
    # above) never even starts, leaking the lock for up to the TTL. Release on
    # response teardown too — the compare-and-delete in `_release_turn_lock`
    # makes a double release (this plus the worker's) harmless.
    response.call_on_close(lambda: _release_turn_lock(session_id, turn_token))
    return response
