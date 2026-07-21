"""Standalone context handler routes.

Provides API endpoints for creating and retrieving context handlers
independently from agent runs.
"""

import logging

from flask import Blueprint, jsonify, request

from ..auth import require_auth
from ..db import db
from ..services.settings_registry import get_setting
from ..services.context_handler import (
    create_and_execute,
    get_context_handler,
    list_context_handlers,
)

logger = logging.getLogger(__name__)

context_handlers_bp = Blueprint("context_handlers", __name__, url_prefix="/api/context-handlers")


@context_handlers_bp.route("", methods=["GET"])
@require_auth
def list_handlers():
    """
    List context handlers with pagination.

    Query params:
        limit: int (default 50, max 100)
        offset: int (default 0)

    Returns:
        200 with paginated list of handlers (without retrieved_context/formatted_context)
    """
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 100))
    except ValueError:
        limit = 50

    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        offset = 0
    handlers, total = list_context_handlers(db.session, limit, offset)
    return jsonify(
        {
            "context_handlers": handlers,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@context_handlers_bp.route("", methods=["POST"])
@require_auth
def create_context_handler():
    """
    Create and execute a context handler.

    Request body:
        query: str (required) - The search query
        knowledge_bases: list[dict] (required) - KB configs with 'id' and optional params
        max_context_tokens: int (optional, default 32000) - Token limit

    Returns:
        201 with full handler object on success
    """
    data = request.get_json() or {}

    query = data.get("query")
    if not query:
        return jsonify({"error": "query is required"}), 400

    knowledge_bases = data.get("knowledge_bases", [])
    if not knowledge_bases:
        return jsonify({"error": "knowledge_bases is required and must not be empty"}), 400

    max_context_tokens = data.get("max_context_tokens", get_setting("DEFAULT_MAX_CONTEXT_TOKENS"))

    try:
        handler_id, result = create_and_execute(
            db_session=db.session,
            query=query,
            knowledge_base_configs=knowledge_bases,
            max_context_tokens=max_context_tokens,
        )
        db.session.commit()

        return jsonify(
            {
                "id": handler_id,
                "query": query,
                "status": result["status"],
                "knowledge_base_configs": knowledge_bases,
                "max_context_tokens": max_context_tokens,
                "formatted_context": result["formatted_context"],
                "retrieved_context": result["retrieved_context"],
                "metadata": result["metadata"],
                "errors": result["errors"],
            }
        ), 201

    except Exception as e:
        logger.exception("Context handler creation failed")
        return jsonify({"error": str(e)}), 500


@context_handlers_bp.route("/<handler_id>", methods=["GET"])
@require_auth
def get_handler(handler_id: str):
    """
    Fetch a context handler by ID.

    Returns:
        200 with full handler object, or 404 if not found
    """
    handler = get_context_handler(db.session, handler_id)
    if not handler:
        return jsonify({"error": "Context handler not found"}), 404

    return jsonify(handler)
