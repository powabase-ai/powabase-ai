"""Session management routes for the project service."""

import logging

from flask import Blueprint, g, jsonify, request

from ..auth import get_current_user_id, require_auth
from ..db import db
from ..services.context_handler import resolve_tool_call_image_refs
from ..services.session import (
    delete_session,
    get_chat_messages,
    get_run_retrieved_context,
    get_session_by_id,
    get_session_owner,
    list_runs_for_session,
)

logger = logging.getLogger(__name__)

sessions_bp = Blueprint("sessions", __name__, url_prefix="/api/sessions")


def _verify_session_access(session_id: str):
    """Return None if caller may access session, else a (response, 404) tuple.

    Service-role callers bypass the check. For user-scoped callers, return 404
    (not 403) on both "not found" and "owned by someone else" to avoid leaking
    session existence.
    """
    jwt_payload = getattr(g, "jwt_payload", None) or {}
    if jwt_payload.get("is_service_role", False):
        return None

    owner = get_session_owner(db.session, session_id)
    caller = get_current_user_id()
    if owner is None or owner != caller:
        return jsonify({"error": "Session not found"}), 404
    return None


@sessions_bp.route("/<session_id>", methods=["GET"])
@require_auth
def get_session(session_id: str):
    """Get a session by its session_id."""
    denial = _verify_session_access(session_id)
    if denial is not None:
        return denial

    session = get_session_by_id(db.session, session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    return jsonify(session)


@sessions_bp.route("/<session_id>/messages", methods=["GET"])
@require_auth
def get_messages(session_id: str):
    """
    Get chat messages for a session.

    Returns messages in chronological order, suitable for displaying in a chat UI.

    Query params:
        limit: Optional max number of messages to return

    Returns:
        {
            "session_id": "sess_xxx",
            "messages": [
                {"role": "user", "content": "...", "run_id": "...", "timestamp": "..."},
                {"role": "assistant", "content": "...", "run_id": "...", "timestamp": "..."},
                ...
            ]
        }
    """
    denial = _verify_session_access(session_id)
    if denial is not None:
        return denial

    raw_limit = request.args.get("limit", type=int)
    limit = max(1, min(raw_limit, 200)) if raw_limit is not None else None

    messages = get_chat_messages(db.session, session_id, limit=limit)

    return jsonify(
        {
            "session_id": session_id,
            "messages": messages,
        }
    )


@sessions_bp.route("/<session_id>/runs", methods=["GET"])
@require_auth
def get_runs(session_id: str):
    """
    Get all runs for a session.

    Query params:
        limit: Max number of runs (default 100)
        offset: Offset for pagination

    Returns:
        {
            "session_id": "sess_xxx",
            "runs": [...]
        }
    """
    denial = _verify_session_access(session_id)
    if denial is not None:
        return denial

    try:
        limit = max(1, min(int(request.args.get("limit", 100)), 100))
    except ValueError:
        limit = 100

    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        offset = 0

    runs = list_runs_for_session(db.session, session_id, limit=limit, offset=offset)

    for run in runs:
        if run.get("tool_calls"):
            run["tool_calls"] = resolve_tool_call_image_refs(run["tool_calls"])

    return jsonify(
        {
            "session_id": session_id,
            "runs": runs,
            "limit": limit,
            "offset": offset,
        }
    )


@sessions_bp.route("/<session_id>/runs/<run_id>/retrieved-context", methods=["GET"])
@require_auth
def get_run_retrieved_context_route(session_id: str, run_id: str):
    """Get retrieved context for a single run in a session."""
    denial = _verify_session_access(session_id)
    if denial is not None:
        return denial

    retrieved_context = get_run_retrieved_context(db.session, session_id, run_id)
    if retrieved_context is None:
        return jsonify({"error": "Run not found"}), 404
    return jsonify(
        {
            "session_id": session_id,
            "run_id": run_id,
            "retrieved_context": retrieved_context,
        }
    )


@sessions_bp.route("/<session_id>", methods=["DELETE"])
@require_auth
def delete_session_route(session_id: str):
    """
    Delete a session and all its runs.

    The runs are deleted via CASCADE in the database.
    """
    denial = _verify_session_access(session_id)
    if denial is not None:
        return denial

    deleted = delete_session(db.session, session_id)

    if not deleted:
        return jsonify({"error": "Session not found"}), 404

    db.session.commit()

    return jsonify({"success": True, "message": "Session deleted"})
