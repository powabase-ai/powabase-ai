"""Agent management routes for the project service.

This module uses the same approach as the backend for agent runs:
- Session history loading for multi-turn conversations
- Knowledge base search with token limiting
- agentic.Agent class for LLM streaming
"""

import collections
import contextvars
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from agentic import Agent
from agentic.agent.message import Message
from agentic.execution.context import ExecutionContext
from flask import Blueprint, Response, current_app, g, jsonify, request, stream_with_context
from sqlalchemy import text

from ..auth import get_current_user_id, require_auth
from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
from ..services.llm_availability import check_model_available
from ..services.run_context import (
    reset_run_id,
    set_run_id,
)
from ..services.settings_registry import get_setting
from sqlalchemy.exc import IntegrityError

from ..models.tenant import (
    AgentKnowledgeBase,
    AgentMcpServer,
    AgentRun,
    AgentRunStatus,
    AgentSession,
    AgentTool,
    Hook,
)
from ..services.context_handler import (
    create_and_execute,
    get_context_handler,
    make_lightweight_retrieved_context,
    strip_tool_call_images,
)
from ..services.citations import (
    build_citation_instruction,
    build_citation_map,
    parse_citations_from_response,
    persist_citations,
)
from ..services.session import (
    build_messages_for_llm,
    get_or_create_session,
    get_session_owner,
    load_session_history,
    persist_agent_run,
    update_agent_run,
)
from ..services.run_registry import get_active_run_context, register_run, unregister_run
from ..services.ai_provider_keys_resolver import (
    ProviderKeyDecryptDropped,
    resolve_api_key_or_raise_for_drop,
)

logger = logging.getLogger(__name__)

# Pre-op estimate: 1 agent_run base fee + headroom for ~50 internal ops
# (tool calls, retrievals). Picked to match the spec's "few tool calls"
# heuristic. Atomic ops inside the agent loop are independently billed
# in tool_registry / knowledge_search via Task 15 — this constant is the
# pre-flight check threshold only.
# v1.5 smoke-test bump from 51 mc — see
# orchestrations.py:_ORCHESTRATION_RUN_ESTIMATED_COST for full rationale.
# 1,000 mc ≈ $0.01 covers a typical mid-tier LLM call; bigger Opus runs
# can still overrun mid-stream (post-charge architectural limitation).
_AGENT_RUN_ESTIMATED_COST = 1_000

agents_bp = Blueprint("agents", __name__, url_prefix="/api/agents")


def extract_reasoning_steps(messages: list[dict]) -> list[dict]:
    """Extract per-step assistant reasoning from full message history."""
    steps = []
    step_num = 0
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content") or ""
            tool_call_names = [
                tc.get("function", {}).get("name", "") for tc in (msg.get("tool_calls") or [])
            ]
            if (isinstance(content, str) and content.strip()) or tool_call_names:
                step_num += 1
                entry: dict = {"step": step_num}
                if isinstance(content, str) and content.strip():
                    entry["content"] = content.strip()
                if tool_call_names:
                    entry["tool_calls"] = tool_call_names
                steps.append(entry)
    return steps


def _resolve_context_items(
    db_session, context_items: list[dict], ai_schema: str = "ai"
) -> list[dict]:
    """
    Resolve context_items: look up by-reference items from any content table,
    pass through by-value items as-is.

    Searches all 5 content tables: chunks, graph_index_nodes, page_index_nodes,
    full_documents, doc2json_documents.
    """
    by_ref_indices = []
    item_ids = []
    resolved = []

    for i, item in enumerate(context_items):
        ref_id = item.get("item_id")
        if ref_id:
            by_ref_indices.append(i)
            item_ids.append(ref_id)
            resolved.append(None)  # placeholder
        elif "text" in item:
            resolved.append(
                {
                    "text": item["text"],
                    "meta": item.get("meta", {}),
                }
            )
        else:
            logger.warning("context_item %d has neither item_id nor text, skipping", i)
            resolved.append(None)

    if item_ids:
        # Search content tables via UNION.  The text column differs by table
        # (mirrors BasePgVectorStore.TEXT_COL per store subclass):
        #   chunks / graph_index_nodes / page_index_nodes → "text"
        #   full_documents → "full_text_path" (resolved from Supabase storage below)
        #   doc2json_documents → "summary"
        content_tables = {
            "chunks": "text",
            "graph_index_nodes": "text",
            "page_index_nodes": "text",
            "full_documents": "full_text_path",
            "doc2json_documents": "summary",
        }
        union_parts = []
        for tbl, text_col in content_tables.items():
            union_parts.append(f"""
                SELECT t.id, t.{text_col} AS text, t.source_id, t.meta,
                       s.name AS source_name, '{tbl}' AS tbl
                FROM "{ai_schema}".{tbl} t
                LEFT JOIN "{ai_schema}".sources s ON s.id = t.source_id
                WHERE t.id = ANY(:ids)
            """)
        query = " UNION ALL ".join(union_parts)
        result = db_session.execute(text(query), {"ids": item_ids})

        item_lookup = {}
        full_doc_paths = {}  # id → full_text_path
        for row in result:
            row_id = str(row[0])
            item_lookup[row_id] = {
                "id": row_id,
                "text": row[1],
                "source_id": str(row[2]) if row[2] else None,
                "meta": row[3] or {},
                "source_name": row[4] or "",
            }
            if row[5] == "full_documents":
                full_doc_paths[row_id] = row[1]

        # Resolve full_documents: download full text from Supabase storage
        # (same as FullDocumentStore._resolve_text in the normal retrieval flow)
        if full_doc_paths:
            from ..services.storage import get_storage

            storage = get_storage()
            for row_id, path in full_doc_paths.items():
                try:
                    item_lookup[row_id]["text"] = storage.download_from_path(path).decode("utf-8")
                except Exception:
                    logger.warning("Failed to download full_text for item %s from %s", row_id, path)

        for idx, item_id in zip(by_ref_indices, item_ids):
            if item_id in item_lookup:
                resolved[idx] = item_lookup[item_id]
            else:
                logger.warning(
                    "context_item item_id %s not found in any content table, skipping", item_id
                )

    return [item for item in resolved if item is not None]


def _finish_run_in_background(
    flask_app,
    run_id: str,
    llm_gen,
    content_chunks: list[str],
    message: str,
    query_enrichment: dict | None,
    retrieved_context_for_db: list[dict] | None,
    context_handler_id: str | None,
    started_at: datetime,
    reasoning_requested: bool,
    citation_map: dict | None = None,
):
    """Consume remaining LLM chunks and persist the completed run in a background thread."""
    # Rebind run_id in the worker thread — ContextVar values do NOT propagate
    # across raw threading.Thread boundaries, so without this the background
    # charge call (and any tool/KB billing that fires from drained chunks)
    # would derive a uuid4 fallback, defeating spec line 132.
    _bg_token = set_run_id(run_id)
    try:
        with flask_app.app_context():
            try:
                llm_error = "Unknown error — no output from agent"

                # Drain remaining chunks from the LLM generator
                try:
                    output = None
                    while True:
                        chunk = next(llm_gen)
                        content_chunks.append(chunk)
                except StopIteration as e:
                    output = e.value
                except Exception as e:
                    logger.exception(
                        "Background thread: error consuming LLM chunks for run %s",
                        run_id,
                    )
                    output = None
                    llm_error = str(e)

                final_content = "".join(content_chunks)

                citations = []
                if citation_map:
                    final_content, citations = parse_citations_from_response(
                        final_content, citation_map
                    )

                run_status = (
                    AgentRunStatus.COMPLETED
                    if output and output.status.value == "completed"
                    else AgentRunStatus.FAILED
                )
                run_error = output.error if output else llm_error

                reasoning = (
                    extract_reasoning_steps(output.messages) if output and output.messages else None
                )

                # Mirror the foreground path's terminal `reasoning` synthesis
                # so disconnect-then-reload still populates the FE pill's
                # expanded panel. Without this, a client that closes the tab
                # mid-stream gets the same "Thought for Xs · 0 steps" bug the
                # foreground fix addresses.
                bg_events_for_db: list[dict] = []
                bg_artifact = output.reasoning_artifact if output else None
                bg_reasoning_text = (
                    getattr(bg_artifact, "summary_text", None) if bg_artifact else None
                )
                if bg_reasoning_text:
                    bg_events_for_db.append(
                        {
                            "type": "reasoning",
                            "step": 1,
                            "source": "thinking",
                            "content": bg_reasoning_text,
                        }
                    )

                update_agent_run(
                    db_session=db.session,
                    run_id=run_id,
                    status=run_status,
                    content=final_content,
                    output_messages=[
                        Message(
                            role="assistant",
                            content=final_content,
                            tool_calls=(
                                [tc.to_dict() for tc in output.tool_calls]
                                if output and output.tool_calls
                                else None
                            ),
                            reasoning=output.reasoning_artifact if output else None,
                            reasoning_requested=reasoning_requested,
                        ).model_dump(exclude_none=True)
                    ],
                    usage=output.usage if output else None,
                    error=run_error if run_status == AgentRunStatus.FAILED else None,
                    completed_at=datetime.now(UTC),
                    events=bg_events_for_db if bg_events_for_db else None,
                    reasoning_steps=reasoning,
                )
                db.session.commit()

                if citations:
                    persist_citations(db.session, run_id, citations)
                    db.session.commit()

                # Billing: post the agent_run dispatch fee on success. The
                # client-disconnected-mid-stream path still runs the LLM to
                # completion, so it owes the same charge as a fully-streamed
                # success. Idempotent via the run_id-derived key — if the
                # foreground generator already charged before disconnect, this
                # is a no-op on the billing side.
                if run_status == AgentRunStatus.COMPLETED:
                    billing.charge(
                        action="agent_run",
                        quantity=1,
                        ref_type="agent_run",
                        ref_id=run_id,
                        idempotency_parts=(run_id,),
                        metadata={
                            "streaming": True,
                            "react_loop": False,
                            "finished_in_background": True,
                        },
                    )
                logger.info(
                    "Background thread: finished run %s with status %s",
                    run_id,
                    run_status.value,
                )
            except Exception:
                logger.exception("Background thread: failed to finish run %s", run_id)
            finally:
                db.session.remove()
    except Exception:
        logger.exception(
            "Background thread: unexpected error for run %s (outside app context)", run_id
        )
    finally:
        reset_run_id(_bg_token)


@agents_bp.route("", methods=["GET"])
@require_auth
def list_agents():
    """List all agents, paginated, with usage aggregates."""
    from ..services.list_params import parse_list_params, escape_like, ListParamsError

    try:
        limit, offset, q, sort, order = parse_list_params(
            request,
            sort_allowed={"created_at", "name", "updated_at", "last_run_at"},
        )
    except ListParamsError as e:
        return jsonify({"error": str(e)}), e.status

    where_clause = ""
    params: dict = {"limit": limit, "offset": offset}
    if q:
        where_clause = "WHERE a.name ILIKE :q_like"
        params["q_like"] = f"%{escape_like(q)}%"

    # `last_run_at` is a computed column; reference it by SELECT alias.
    if sort == "last_run_at":
        order_by = f"last_run_at {order.upper()} NULLS LAST, a.id ASC"
    else:
        order_by = f"a.{sort} {order.upper()}, a.id ASC"

    count_sql = f'SELECT COUNT(*) FROM "{AI_SCHEMA}".agents a {where_clause}'
    total = db.session.execute(text(count_sql), params).scalar()

    rows_sql = f"""
        SELECT
          a.id, a.name, a.model, a.system_prompt, a.settings,
          a.created_at, a.updated_at,
          (SELECT COUNT(*) FROM "{AI_SCHEMA}".agent_sessions WHERE agent_id = a.id) AS session_count,
          (SELECT COUNT(*) FROM "{AI_SCHEMA}".agent_runs WHERE agent_id = a.id) AS total_runs,
          (SELECT MAX(created_at) FROM "{AI_SCHEMA}".agent_runs WHERE agent_id = a.id) AS last_run_at
        FROM "{AI_SCHEMA}".agents a
        {where_clause}
        ORDER BY {order_by}
        LIMIT :limit OFFSET :offset
    """
    rows = db.session.execute(text(rows_sql), params)

    agents = []
    for row in rows:
        agents.append(
            {
                "id": str(row.id),
                "name": row.name,
                "model": row.model,
                "system_prompt": row.system_prompt,
                "settings": row.settings,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "session_count": int(row.session_count),
                "total_runs": int(row.total_runs),
                "last_run_at": row.last_run_at.isoformat() if row.last_run_at else None,
            }
        )

    return jsonify(
        {
            "agents": agents,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@agents_bp.route("", methods=["POST"])
@require_auth
def create_agent():
    """Create a new agent."""
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Name is required"}), 400

    agent_id = str(uuid.uuid4())

    db.session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".agents (
                id, name, model, system_prompt, settings
            ) VALUES (
                :id, :name, :model, :system_prompt, CAST(:settings AS jsonb)
            )
        """),
        {
            "id": agent_id,
            "name": data["name"],
            "model": data.get("model", get_setting("AGENT_DEFAULT_MODEL")),
            "system_prompt": data.get("system_prompt"),
            "settings": json.dumps(data.get("settings", {})),
        },
    )
    db.session.commit()

    return jsonify(
        {
            "id": agent_id,
            "name": data["name"],
            "model": data.get("model", get_setting("AGENT_DEFAULT_MODEL")),
            "system_prompt": data.get("system_prompt"),
            "settings": data.get("settings", {}),
        }
    ), 201


@agents_bp.route("/<agent_id>", methods=["GET"])
@require_auth
def get_agent(agent_id: str):
    """Get a specific agent."""
    result = db.session.execute(
        text(f"""
            SELECT id, name, model, system_prompt, settings, created_at, updated_at
            FROM "{AI_SCHEMA}".agents
            WHERE id = :id
        """),
        {"id": agent_id},
    )

    row = result.fetchone()
    if not row:
        return jsonify({"error": "Agent not found"}), 404

    return jsonify(
        {
            "id": str(row[0]),
            "name": row[1],
            "model": row[2],
            "system_prompt": row[3],
            "settings": row[4],
            "created_at": row[5].isoformat() if row[5] else None,
            "updated_at": row[6].isoformat() if row[6] else None,
        }
    )


@agents_bp.route("/<agent_id>", methods=["PATCH"])
@require_auth
def update_agent(agent_id: str):
    """Update an agent."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    updates = []
    params = {"id": agent_id}

    if "name" in data:
        updates.append("name = :name")
        params["name"] = data["name"]
    if "model" in data:
        updates.append("model = :model")
        params["model"] = data["model"]
    if "system_prompt" in data:
        updates.append("system_prompt = :system_prompt")
        params["system_prompt"] = data["system_prompt"]
    if "settings" in data:
        updates.append("settings = CAST(:settings AS jsonb)")
        params["settings"] = json.dumps(data["settings"])

    if not updates:
        return jsonify({"error": "No valid fields to update"}), 400

    updates.append("updated_at = NOW()")

    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".agents
            SET {", ".join(updates)}
            WHERE id = :id
        """),
        params,
    )
    db.session.commit()

    return get_agent(agent_id)


@agents_bp.route("/<agent_id>", methods=["DELETE"])
@require_auth
def delete_agent(agent_id: str):
    """Delete an agent."""
    db.session.execute(
        text(f'DELETE FROM "{AI_SCHEMA}".agents WHERE id = :id'),
        {"id": agent_id},
    )
    db.session.commit()

    return jsonify({"message": "Agent deleted"})


# =============================================================================
# Agent Tool Assignment Endpoints
# =============================================================================


@agents_bp.route("/<agent_id>/tools", methods=["POST"])
@require_auth
def assign_tool(agent_id: str):
    data = request.get_json()
    tool_type = data.get("tool_type")
    tool_name = data.get("tool_name")
    if not tool_type or not tool_name:
        return jsonify({"error": "tool_type and tool_name are required"}), 400

    assignment = AgentTool(
        agent_id=agent_id,
        tool_id=data.get("tool_id"),
        tool_type=tool_type,
        tool_name=tool_name,
        config_override=data.get("config_override", {}),
    )
    db.session.add(assignment)
    db.session.commit()
    return jsonify({"id": str(assignment.id), "tool_name": tool_name}), 201


@agents_bp.route("/<agent_id>/tools", methods=["GET"])
@require_auth
def list_agent_tools(agent_id: str):
    assignments = AgentTool.query.filter_by(agent_id=agent_id).all()
    return jsonify(
        {
            "tools": [
                {
                    "id": str(a.id),
                    "tool_type": a.tool_type,
                    "tool_name": a.tool_name,
                    "tool_id": str(a.tool_id) if a.tool_id else None,
                    "config_override": a.config_override,
                }
                for a in assignments
            ]
        }
    )


@agents_bp.route("/<agent_id>/tools/<assignment_id>", methods=["DELETE"])
@require_auth
def remove_agent_tool(agent_id: str, assignment_id: str):
    assignment = db.session.get(AgentTool, assignment_id)
    if not assignment or str(assignment.agent_id) != agent_id:
        return jsonify({"error": "Assignment not found"}), 404
    db.session.delete(assignment)
    db.session.commit()
    return jsonify({"deleted": True})


@agents_bp.route("/<agent_id>/tools/<assignment_id>", methods=["PATCH"])
@require_auth
def update_agent_tool(agent_id: str, assignment_id: str):
    """Update a tool assignment's config_override."""
    assignment = AgentTool.query.filter_by(id=assignment_id, agent_id=agent_id).first()
    if not assignment:
        return jsonify({"error": "Tool assignment not found"}), 404

    data = request.get_json(silent=True) or {}
    if "config_override" in data:
        config = data["config_override"]
        # Validate schema config if present
        if isinstance(config, dict) and "schemas" in config:
            from ..routes.database import SYSTEM_SCHEMAS

            _ID_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")
            schemas = config["schemas"]
            if not isinstance(schemas, dict):
                return jsonify({"error": "schemas must be a dict"}), 400
            for schema_name, tables in schemas.items():
                if not _ID_RE.match(schema_name):
                    return jsonify({"error": f"Invalid schema name: {schema_name}"}), 400
                if schema_name in SYSTEM_SCHEMAS or schema_name.startswith("pg_"):
                    return jsonify({"error": f"System schema '{schema_name}' is not allowed"}), 400
                if not isinstance(tables, list) or not all(
                    isinstance(t, str) and _ID_RE.match(t) for t in tables
                ):
                    return jsonify({"error": f"Invalid table names in schema '{schema_name}'"}), 400
        assignment.config_override = config
    db.session.commit()

    return jsonify(
        {
            "id": str(assignment.id),
            "tool_type": assignment.tool_type,
            "tool_name": assignment.tool_name,
            "config_override": assignment.config_override,
        }
    )


@agents_bp.route("/<agent_id>/knowledge-bases", methods=["POST"])
@require_auth
def assign_knowledge_base(agent_id: str):
    data = request.get_json()
    kb_id = data.get("knowledge_base_id")
    if not kb_id:
        return jsonify({"error": "knowledge_base_id is required"}), 400

    assignment = AgentKnowledgeBase(
        agent_id=agent_id,
        knowledge_base_id=kb_id,
        config=data.get("config", {}),
    )
    try:
        db.session.add(assignment)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "KB already assigned to this agent"}), 409

    return jsonify(
        {
            "id": str(assignment.id),
            "agent_id": agent_id,
            "knowledge_base_id": kb_id,
            "config": assignment.config,
        }
    ), 201


@agents_bp.route("/<agent_id>/knowledge-bases", methods=["GET"])
@require_auth
def list_agent_knowledge_bases(agent_id: str):
    assignments = AgentKnowledgeBase.query.filter_by(agent_id=agent_id).all()
    return jsonify(
        {
            "knowledge_bases": [
                {
                    "id": str(a.id),
                    "knowledge_base_id": str(a.knowledge_base_id),
                    "config": a.config,
                }
                for a in assignments
            ]
        }
    )


@agents_bp.route("/<agent_id>/knowledge-bases/<assignment_id>", methods=["DELETE"])
@require_auth
def remove_agent_knowledge_base(agent_id: str, assignment_id: str):
    assignment = db.session.get(AgentKnowledgeBase, assignment_id)
    if not assignment or str(assignment.agent_id) != agent_id:
        return jsonify({"error": "Assignment not found"}), 404
    db.session.delete(assignment)
    db.session.commit()
    return jsonify({"deleted": True})


# =============================================================================
# Agent MCP Server Endpoints
# =============================================================================


@agents_bp.route("/<agent_id>/mcp-servers", methods=["POST"])
@require_auth
def add_mcp_server(agent_id: str):
    """Add an MCP server to an agent."""
    data = request.get_json()
    name = data.get("name")
    url = data.get("url")
    if not name or not url:
        return jsonify({"error": "name and url are required"}), 400

    server = AgentMcpServer(
        agent_id=agent_id,
        name=name,
        transport=data.get("transport", "http"),
        url=url,
        headers=data.get("headers", {}),
        config=data.get("config", {}),
        enabled=data.get("enabled", True),
    )
    try:
        db.session.add(server)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "MCP server with this name already exists for agent"}), 409

    return jsonify(
        {
            "id": str(server.id),
            "agent_id": agent_id,
            "name": server.name,
            "transport": server.transport,
            "url": server.url,
            "headers": server.headers,
            "config": server.config,
            "enabled": server.enabled,
            "created_at": server.created_at.isoformat() if server.created_at else None,
            "updated_at": server.updated_at.isoformat() if server.updated_at else None,
        }
    ), 201


@agents_bp.route("/<agent_id>/mcp-servers", methods=["GET"])
@require_auth
def list_mcp_servers(agent_id: str):
    """List MCP servers configured for an agent."""
    servers = AgentMcpServer.query.filter_by(agent_id=agent_id).all()
    return jsonify(
        {
            "mcp_servers": [
                {
                    "id": str(s.id),
                    "name": s.name,
                    "transport": s.transport,
                    "url": s.url,
                    "headers": s.headers,
                    "config": s.config,
                    "enabled": s.enabled,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                }
                for s in servers
            ]
        }
    )


@agents_bp.route("/<agent_id>/mcp-servers/<server_id>", methods=["PUT"])
@require_auth
def update_mcp_server(agent_id: str, server_id: str):
    """Update an MCP server configuration."""
    server = db.session.get(AgentMcpServer, server_id)
    if not server or str(server.agent_id) != agent_id:
        return jsonify({"error": "MCP server not found"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    if "name" in data:
        server.name = data["name"]
    if "transport" in data:
        server.transport = data["transport"]
    if "url" in data:
        server.url = data["url"]
    if "headers" in data:
        server.headers = data["headers"]
    if "config" in data:
        server.config = data["config"]
    if "enabled" in data:
        server.enabled = data["enabled"]

    server.updated_at = datetime.now(UTC)

    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "MCP server with this name already exists for agent"}), 409

    return jsonify(
        {
            "id": str(server.id),
            "agent_id": agent_id,
            "name": server.name,
            "transport": server.transport,
            "url": server.url,
            "headers": server.headers,
            "config": server.config,
            "enabled": server.enabled,
            "created_at": server.created_at.isoformat() if server.created_at else None,
            "updated_at": server.updated_at.isoformat() if server.updated_at else None,
        }
    )


@agents_bp.route("/<agent_id>/mcp-servers/<server_id>", methods=["DELETE"])
@require_auth
def delete_mcp_server(agent_id: str, server_id: str):
    """Remove an MCP server from an agent."""
    server = db.session.get(AgentMcpServer, server_id)
    if not server or str(server.agent_id) != agent_id:
        return jsonify({"error": "MCP server not found"}), 404
    db.session.delete(server)
    db.session.commit()
    return jsonify({"deleted": True})


@agents_bp.route("/<agent_id>/hooks", methods=["POST"])
@require_auth
def add_agent_hook(agent_id: str):
    """Add a hook to an agent."""
    data = request.get_json()
    event = data.get("event")
    hook_type = data.get("type")
    config = data.get("config")
    if not event or not hook_type or config is None:
        return jsonify({"error": "event, type, and config are required"}), 400

    hook = Hook(
        agent_id=agent_id,
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
            "agent_id": agent_id,
            "event": hook.event,
            "matcher": hook.matcher,
            "type": hook.type,
            "config": hook.config,
            "enabled": hook.enabled,
            "position": hook.position,
            "created_at": hook.created_at.isoformat() if hook.created_at else None,
        }
    ), 201


@agents_bp.route("/<agent_id>/hooks", methods=["GET"])
@require_auth
def list_agent_hooks(agent_id: str):
    """List hooks configured for an agent."""
    hooks = Hook.query.filter_by(agent_id=agent_id).order_by(Hook.position).all()
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


@agents_bp.route("/<agent_id>/hooks/<hook_id>", methods=["DELETE"])
@require_auth
def delete_agent_hook(agent_id: str, hook_id: str):
    """Remove a hook from an agent."""
    hook = db.session.get(Hook, hook_id)
    if not hook or str(hook.agent_id) != agent_id:
        return jsonify({"error": "Hook not found"}), 404
    db.session.delete(hook)
    db.session.commit()
    return jsonify({"deleted": True})


@agents_bp.route("/<agent_id>/sessions", methods=["GET"])
@require_auth
def list_sessions(agent_id: str):
    """List sessions for an agent with optional search and filters."""
    from ..services.list_params import escape_like

    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 200))
    except ValueError:
        limit = 50

    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        offset = 0

    # Optional filter params
    search = request.args.get("search", "").strip()
    created_after = request.args.get("created_after", "").strip()
    created_before = request.args.get("created_before", "").strip()
    min_runs = request.args.get("min_runs", "").strip()
    max_runs = request.args.get("max_runs", "").strip()

    # Scope to authenticated user unless caller is service-role
    is_service_role = (getattr(g, "jwt_payload", None) or {}).get("is_service_role", False)
    scoped_user_id = None if is_service_role else get_current_user_id()

    params: dict[str, object] = {"agent_id": agent_id, "limit": limit, "offset": offset}
    where_clauses: list[str] = []
    having_clauses: list[str] = []

    if scoped_user_id is not None:
        params["scoped_user_id"] = scoped_user_id
        where_clauses.append("s.user_id = :scoped_user_id")

    if search:
        params["search_pattern"] = f"%{escape_like(search)}%"
        where_clauses.append(f"""
            (
                EXISTS (
                    SELECT 1 FROM "{AI_SCHEMA}".agent_runs sr
                    WHERE sr.session_id = s.id
                    AND (
                        sr.content ILIKE :search_pattern
                        OR sr.input_messages::text ILIKE :search_pattern
                        OR sr.output_messages::text ILIKE :search_pattern
                    )
                )
                OR s.session_id ILIKE :search_pattern
            )
        """)

    if created_after:
        params["created_after"] = created_after
        where_clauses.append("s.created_at >= CAST(:created_after AS timestamptz)")

    if created_before:
        params["created_before"] = created_before
        where_clauses.append("s.created_at < CAST(:created_before AS date) + interval '1 day'")

    if min_runs:
        try:
            params["min_runs"] = int(min_runs)
        except ValueError:
            return (
                jsonify({"error": f"min_runs must be an integer, got {min_runs!r}"}),
                400,
            )
        having_clauses.append("COUNT(r.id) >= :min_runs")

    if max_runs:
        try:
            params["max_runs"] = int(max_runs)
        except ValueError:
            return (
                jsonify({"error": f"max_runs must be an integer, got {max_runs!r}"}),
                400,
            )
        having_clauses.append("COUNT(r.id) <= :max_runs")

    extra_where = (" AND " + " AND ".join(where_clauses)) if where_clauses else ""
    having_sql = ("HAVING " + " AND ".join(having_clauses)) if having_clauses else ""

    result = db.session.execute(
        text(f"""
            SELECT s.session_id,
                   COUNT(r.id) as run_count,
                   MAX(r.created_at) as last_activity_at,
                   s.created_at as created_at,
                   (
                     SELECT SUBSTRING(
                       (first_run.input_messages->0->>'content')
                       FROM 1 FOR 100
                     )
                     FROM "{AI_SCHEMA}".agent_runs first_run
                     WHERE first_run.session_id = s.id
                     ORDER BY first_run.created_at ASC
                     LIMIT 1
                   ) as first_message
            FROM "{AI_SCHEMA}".agent_sessions s
            LEFT JOIN "{AI_SCHEMA}".agent_runs r ON r.session_id = s.id
            WHERE s.agent_id = :agent_id
            {extra_where}
            GROUP BY s.id, s.session_id, s.created_at
            {having_sql}
            ORDER BY last_activity_at DESC NULLS LAST
            LIMIT :limit OFFSET :offset
        """),
        params,
    )

    sessions = []
    for row in result:
        sessions.append(
            {
                "session_id": row[0],
                "run_count": row[1],
                "last_activity_at": row[2].isoformat() if row[2] else None,
                "created_at": row[3].isoformat() if row[3] else None,
                "first_message": row[4],
            }
        )

    # Count query must apply the same filters
    count_result = db.session.execute(
        text(f"""
            SELECT COUNT(*) FROM (
                SELECT s.id
                FROM "{AI_SCHEMA}".agent_sessions s
                LEFT JOIN "{AI_SCHEMA}".agent_runs r ON r.session_id = s.id
                WHERE s.agent_id = :agent_id
                {extra_where}
                GROUP BY s.id
                {having_sql}
            ) filtered
        """),
        params,
    )
    total = count_result.scalar()

    return jsonify(
        {
            "sessions": sessions,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


# =============================================================================
# Agent Run Endpoint (Non-Streaming Mode)
# =============================================================================


@agents_bp.route("/<agent_id>/run", methods=["POST"])
@require_auth
def run_agent(agent_id: str):
    """
    Run an agent (non-streaming).

    Uses the same approach as the backend:
    - Session history for multi-turn conversations
    - Knowledge base search with token limiting
    - agentic.Agent class for LLM calls
    """
    data = request.get_json() or {}
    message = data.get("message")
    if not message:
        return jsonify({"error": "message is required"}), 400

    session_id = data.get("session_id")
    knowledge_bases = data.get("knowledge_bases", [])

    # Validate mutual exclusivity of context sources
    context_sources = sum(
        [
            bool(data.get("knowledge_bases")),
            bool(data.get("context_handler_id")),
            bool(data.get("context_override")),
            bool(data.get("context_items")),
        ]
    )
    if context_sources > 1:
        return jsonify(
            {
                "error": "Only one of 'knowledge_bases', 'context_handler_id', 'context_override', or 'context_items' may be provided"
            }
        ), 400

    # Pre-op balance check (free-tier hard cap) via the billing port — the
    # no-op adapter makes this inert in OSS/unit-test/local-dev builds; the
    # cloud adapter enforces the cap against BILLING_ORG_ID. Internal ops
    # inside the agent loop (tool calls, retrievals) charge independently via
    # tool_registry / knowledge_search — see Task 15.
    billing.check_balance(estimated_cost=_AGENT_RUN_ESTIMATED_COST)

    # Get user_id from auth context
    user_id = get_current_user_id()

    # Ownership check: if the client provided a session_id for an existing
    # session, it must belong to this user (service-role bypasses).
    is_service_role = (getattr(g, "jwt_payload", None) or {}).get("is_service_role", False)
    if session_id and not is_service_role:
        owner = get_session_owner(db.session, session_id)
        if owner is not None and owner != user_id:
            return jsonify({"error": "Session not found"}), 404

    # Fetch agent from DB
    result = db.session.execute(
        text(f"""
            SELECT id, name, model, system_prompt, settings
            FROM "{AI_SCHEMA}".agents
            WHERE id = :id
        """),
        {"id": agent_id},
    )
    agent_row = result.fetchone()
    if not agent_row:
        return jsonify({"error": "Agent not found"}), 404

    agent_name = agent_row[1]
    agent_model = agent_row[2] or get_setting("AGENT_DEFAULT_MODEL")
    agent_system_prompt = agent_row[3] or ""
    agent_settings = agent_row[4] or {}

    # Fail fast when the agent's model has neither a project BYOK key nor a
    # platform env key — avoids surfacing LiteLLM's generic "Missing API Key"
    # mid-loop. Aborts 400 with an actionable Settings → LLM Provider Keys
    # pointer when the model is BYOK-only and unconfigured.
    check_model_available(agent_model)

    # Resolve provider API key from DB. The helper self-heals undecryptable
    # rows (issue #246) and raises ProviderKeyDecryptDropped if the model's
    # provider was just dropped — surface that as an actionable error.
    try:
        provider_api_key = resolve_api_key_or_raise_for_drop(agent_model)
    except ProviderKeyDecryptDropped as exc:
        return jsonify(
            {"error": str(exc), "code": "provider_key_decrypt_failed", "provider": exc.provider}
        ), 400

    # Agent-level default, overridden by request-level param
    max_context_tokens = data.get(
        "max_context_tokens",
        agent_settings.get("max_context_tokens", get_setting("DEFAULT_MAX_CONTEXT_TOKENS")),
    )

    # Generate run_id
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    started_at = datetime.now(UTC)
    context_handler_id: str | None = None

    # Bind run_id into the billing contextvar so KB-search / tool-call /
    # query-enrichment idempotency keys are deterministic on agent_run
    # replay (spec line 132). Reset in finally so a downstream handler in
    # the same worker never sees a stale id.
    _run_id_token = set_run_id(run_id)
    try:
        # Get or create session
        db_session_uuid, actual_session_id, is_new_session = get_or_create_session(
            db_session=db.session,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
        )

        # Load session history for multi-turn conversations
        session_history = load_session_history(
            db_session=db.session,
            db_session_uuid=db_session_uuid,
        )

        # Resolve context via one of three modes
        rag_context = ""
        retrieved_context_for_db: list[dict] | None = None
        query_enrichment: dict[str, Any] | None = None
        rag_warning: str | None = None

        if data.get("context_handler_id"):
            # Mode 1: Reuse existing context handler
            # formatted_context doesn't depend on resolve_text, so a single
            # resolve_text=False call avoids unnecessary storage downloads.
            handler = get_context_handler(
                db.session, data["context_handler_id"], resolve_text=False
            )
            if not handler:
                return jsonify({"error": "Context handler not found"}), 404
            rag_context = handler.get("formatted_context")
            if rag_context is None:
                rag_context = ""
            retrieved_context_for_db = handler.get("retrieved_context")
            context_handler_id = handler["id"]

        elif data.get("context_override"):
            # Mode 2: Manual context injection
            rag_context = data["context_override"]
            retrieved_context_for_db = [
                {"_type": "retrieval_diagnostics", "source": "manual_override"}
            ]

        elif knowledge_bases:
            # Mode 3: Knowledge base retrieval (default, backward-compatible)
            try:
                handler_id, retrieval_result = create_and_execute(
                    db_session=db.session,
                    query=message,
                    knowledge_base_configs=knowledge_bases,
                    max_context_tokens=max_context_tokens,
                    session_history=[m.to_litellm_input() for m in session_history],
                )
                rag_context = retrieval_result["formatted_context"]
                retrieved_context_for_db = make_lightweight_retrieved_context(
                    retrieval_result["retrieved_context"]
                )
                context_handler_id = handler_id
                query_enrichment = retrieval_result.get("query_enrichment")
            except Exception as e:
                logger.warning("Failed to retrieve RAG context for agent %s: %s", agent_id, e)
                db.session.rollback()
                rag_warning = f"Failed to retrieve knowledge base context: {e}. The agent responded without document context."

        elif data.get("context_items"):
            resolved_items = _resolve_context_items(db.session, data["context_items"])
            if resolved_items:
                from agentic.knowledge.models import RetrievedItem
                from agentic_project_service.services.knowledge_search import (
                    format_items_as_context,
                )

                retrieved_items = [
                    RetrievedItem(
                        item_id=item.get("id", ""),
                        text=item["text"],
                        score=0.0,
                        source_id=item.get("source_id"),
                        knowledge_base_id=None,
                        meta=item.get("meta", {}),
                    )
                    for item in resolved_items
                ]
                citations_enabled_flag = data.get("citations_enabled", False)
                rag_context, _diag = format_items_as_context(
                    retrieved_items,
                    max_tokens=max_context_tokens,
                    citations_enabled=citations_enabled_flag,
                )
                retrieved_context_for_db = [
                    {"_type": "retrieval_diagnostics", "source": "context_items"},
                ] + resolved_items

        # Build system prompt and user message based on context type
        full_system_prompt = agent_system_prompt

        if isinstance(rag_context, list):
            # Multimodal context — put images in user message content array
            logger.info("Using multimodal context with %d content blocks", len(rag_context))
            user_content: str | list[dict] = [
                {"type": "text", "text": "Context from relevant documents:"},
                *rag_context,
                {"type": "text", "text": f"\n\n{message}"},
            ]
        elif rag_context:
            # Text-only context (current behavior)
            full_system_prompt = f"{agent_system_prompt}\n\nContext:\n{rag_context}"
            user_content = message
        else:
            user_content = message

        # Citation handling — build map and append instruction
        citations_enabled = data.get("citations_enabled", False)
        citation_map: dict = {}

        if citations_enabled and retrieved_context_for_db:
            citation_map = build_citation_map(retrieved_context_for_db)
            if citation_map:
                full_system_prompt = full_system_prompt + "\n\n" + build_citation_instruction()

        # Create the agentic Agent instance
        agent = Agent(
            model=agent_model,
            system_prompt=full_system_prompt,
            name=agent_name,
            api_key=provider_api_key,
            reasoning_effort=agent_settings.get("reasoning_effort"),
        )

        # Build messages with history (drops cross-provider reasoning artifacts)
        messages_for_llm = build_messages_for_llm(
            session_history=session_history,
            target_model=agent_model,
            context=ExecutionContext(),
            user_input=user_content,
        )

        # Run the agent (non-streaming)
        with billing.llm_call_scope():
            output = agent.run(messages_for_llm)
        final_content = output.content or ""

        # Parse and validate citations
        citations: list[dict] = []
        if citation_map:
            final_content, citations = parse_citations_from_response(final_content, citation_map)

        if output.status.value == "completed" and not output.content:
            logger.warning(
                "Agent returned completed status but empty content. "
                "This may indicate the model could not process multimodal input."
            )

        # Persist the run
        input_messages: list[dict[str, Any]] = [{"role": "user", "content": message}]
        if query_enrichment:
            input_messages.append({"_type": "query_enrichment", **query_enrichment})

        reasoning = extract_reasoning_steps(output.messages) if output.messages else None

        persist_agent_run(
            db_session=db.session,
            db_session_uuid=db_session_uuid,
            run_id=run_id,
            status=AgentRunStatus.COMPLETED
            if output.status.value == "completed"
            else AgentRunStatus.FAILED,
            input_messages=input_messages,
            output_messages=[
                Message(
                    role="assistant",
                    content=final_content,
                    tool_calls=(
                        [tc.to_dict() for tc in output.tool_calls] if output.tool_calls else None
                    ),
                    reasoning=output.reasoning_artifact,
                    reasoning_requested=output.reasoning_requested,
                ).model_dump(exclude_none=True)
            ],
            content=final_content,
            usage=output.usage,
            retrieved_context=retrieved_context_for_db,
            error=output.error,
            started_at=started_at,
            completed_at=output.completed_at,
            context_handler_id=context_handler_id,
            reasoning_steps=reasoning,
        )
        db.session.commit()

        if citations:
            persist_citations(db.session, run_id, citations)
            db.session.commit()

        # Billing: post the agent_run dispatch fee on completed runs only.
        # Internal atomic ops (tool calls, retrievals) charge themselves
        # inside the loop via Task 15 — no double-billing here.
        if output.status.value == "completed":
            billing.charge(
                action="agent_run",
                quantity=1,
                ref_type="agent_run",
                ref_id=run_id,
                idempotency_parts=(run_id,),
                metadata={
                    "agent_id": str(agent_id),
                    "model": agent_model,
                    "streaming": False,
                },
            )

        response = {
            "run_id": run_id,
            "session_id": actual_session_id,
            "context_handler_id": context_handler_id,
            "content": final_content,
            "error": output.error,
            "usage": output.usage,
            "retrieved_items": len(retrieved_context_for_db) - 1 if retrieved_context_for_db else 0,
            "status": "completed" if output.status.value == "completed" else "failed",
            "is_new_session": is_new_session,
        }
        if citations_enabled and citation_map:
            response["citation_candidates"] = list(citation_map.values())
            response["citations"] = citations
        if rag_warning:
            response["warning"] = rag_warning
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Agent run failed")
        db.session.rollback()

        # Try to persist failed run
        try:
            db_session_uuid, actual_session_id, _ = get_or_create_session(
                db_session=db.session,
                agent_id=agent_id,
                session_id=session_id,
                user_id=user_id,
            )
            persist_agent_run(
                db_session=db.session,
                db_session_uuid=db_session_uuid,
                run_id=run_id,
                status=AgentRunStatus.FAILED,
                input_messages=[{"role": "user", "content": message}],
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(UTC),
                context_handler_id=context_handler_id,
            )
            db.session.commit()
        except Exception:
            logger.exception("Failed to persist failed run")

        error_msg = f"Agent run failed: {e}. If this persists, verify that referenced knowledge bases and sources still exist in the ai schema."
        return jsonify({"error": error_msg, "run_id": run_id}), 500
    finally:
        reset_run_id(_run_id_token)


# =============================================================================
# Agent Run Endpoint (Streaming Mode via SSE)
# =============================================================================


@agents_bp.route("/<agent_id>/run/stream", methods=["POST"])
@require_auth
def run_agent_stream(agent_id: str):
    """
    Stream an agent's response using Server-Sent Events (SSE).

    Uses the same approach as the backend:
    - Session history for multi-turn conversations
    - Knowledge base search with token limiting
    - agentic.Agent.stream() for LLM streaming

    Events sent:
    - `start`: Initial metadata (run_id, session_id)
    - `chunk`: Content chunk as it streams
    - `complete`: Final response with metadata
    - `error`: Error information if something fails
    """
    # Parse request
    data = request.get_json() or {}
    message = data.get("message")
    if not message:
        return jsonify({"error": "message is required"}), 400

    session_id = data.get("session_id")
    knowledge_bases = data.get("knowledge_bases", [])

    # Validate mutual exclusivity of context sources
    context_sources = sum(
        [
            bool(data.get("knowledge_bases")),
            bool(data.get("context_handler_id")),
            bool(data.get("context_override")),
            bool(data.get("context_items")),
        ]
    )
    if context_sources > 1:
        return jsonify(
            {
                "error": "Only one of 'knowledge_bases', 'context_handler_id', 'context_override', or 'context_items' may be provided"
            }
        ), 400

    # Pre-op balance check (free-tier hard cap). Done BEFORE entering the
    # SSE generator so 402/503 propagate as a normal HTTP error to the
    # caller. Internal atomic ops are billed by Task 15 — this is just the
    # dispatch-fee pre-check.
    billing.check_balance(estimated_cost=_AGENT_RUN_ESTIMATED_COST)

    # Get user_id from auth context
    user_id = get_current_user_id()

    # Ownership check: if the client provided a session_id for an existing
    # session, it must belong to this user (service-role bypasses).
    is_service_role = (getattr(g, "jwt_payload", None) or {}).get("is_service_role", False)
    if session_id and not is_service_role:
        owner = get_session_owner(db.session, session_id)
        if owner is not None and owner != user_id:
            return jsonify({"error": "Session not found"}), 404

    # Fetch agent from DB
    result = db.session.execute(
        text(f"""
            SELECT id, name, model, system_prompt, settings
            FROM "{AI_SCHEMA}".agents
            WHERE id = :id
        """),
        {"id": agent_id},
    )
    agent_row = result.fetchone()
    if not agent_row:
        return jsonify({"error": "Agent not found"}), 404

    agent_name = agent_row[1]
    agent_model = agent_row[2] or get_setting("AGENT_DEFAULT_MODEL")
    agent_system_prompt = agent_row[3] or ""
    agent_settings = agent_row[4] or {}

    # Fail fast when the agent's model has neither a project BYOK key nor a
    # platform env key — avoids surfacing LiteLLM's generic "Missing API Key"
    # mid-stream. Aborts 400 BEFORE entering the SSE generator so the error
    # propagates as a normal HTTP response (not as a stream event).
    check_model_available(agent_model)

    # Resolve provider API key from DB. The helper self-heals undecryptable
    # rows (issue #246) and raises ProviderKeyDecryptDropped if the model's
    # provider was just dropped — surface that as an actionable error.
    try:
        provider_api_key = resolve_api_key_or_raise_for_drop(agent_model)
    except ProviderKeyDecryptDropped as exc:
        return jsonify(
            {"error": str(exc), "code": "provider_key_decrypt_failed", "provider": exc.provider}
        ), 400

    # Agent-level default, overridden by request-level param
    max_context_tokens = data.get(
        "max_context_tokens",
        agent_settings.get("max_context_tokens", get_setting("DEFAULT_MAX_CONTEXT_TOKENS")),
    )

    # Generate run_id
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    started_at = datetime.now(UTC)

    if data.get("context_handler_id"):
        _preflight_handler = get_context_handler(
            db.session, data["context_handler_id"], resolve_text=False
        )
        if not _preflight_handler:
            return jsonify({"error": "Context handler not found"}), 404

    # Capture app reference for background thread context
    app = current_app._get_current_object()

    def generate_sse():
        """Generator function for SSE stream."""
        nonlocal run_id, started_at
        # Bind run_id into the billing contextvar for the foreground stream.
        # Set inside the generator so the token's reset can run in a tail
        # finally regardless of how the stream exits (success, error, client
        # disconnect). The background continuation re-binds in its own
        # thread inside _finish_run_in_background.
        _stream_run_id_token = set_run_id(run_id)
        context_handler_id: str | None = None
        llm_gen = None
        run_persisted = False
        content_chunks: list[str] = []
        query_enrichment: dict[str, Any] | None = None
        retrieved_context_for_db: list[dict] | None = None

        # β streaming flag + content buffer + delta counter — initialized once
        # at request entry (issue #106 / Task 14, mirroring Task 13). The
        # buffer accumulates ``content_delta`` payloads across all ReAct step
        # iterations and feeds three downstream uses: the terminal SSE
        # ``chunk`` event, the ``complete`` event's ``content`` field, and the
        # persisted ``run.content`` column. ``events_for_db`` is declared at
        # function scope so the outer ``except Exception`` handler can pass it
        # to ``update_agent_run`` (M2 v3). These vars are unused by the
        # chat-style branch (no tools); they remain inert there.
        streaming_enabled = os.getenv("AGENT_LLM_STREAMING_ENABLED", "true").lower() == "true"
        content_buffer = ""
        delta_count = 0
        events_for_db: list[dict] = []
        started_at_monotonic = time.monotonic()

        try:
            # Get or create session
            db_session_uuid, actual_session_id, is_new_session = get_or_create_session(
                db_session=db.session,
                agent_id=agent_id,
                session_id=session_id,
                user_id=user_id,
            )
            # Commit session so it survives any RAG rollback
            db.session.commit()

            # Load session history for multi-turn conversations
            session_history = load_session_history(
                db_session=db.session,
                db_session_uuid=db_session_uuid,
            )

            # Resolve context via one of three modes
            rag_context = ""

            if data.get("context_handler_id"):
                handler = get_context_handler(
                    db.session, data["context_handler_id"], resolve_text=False
                )
                if handler:
                    rag_context = handler.get("formatted_context")
                    if rag_context is None:
                        rag_context = ""
                    context_handler_id = handler["id"]
                    retrieved_context_for_db = handler.get("retrieved_context")
                else:
                    logger.warning(f"Context handler not found: {data['context_handler_id']}")

            elif data.get("context_override"):
                rag_context = data["context_override"]
                retrieved_context_for_db = [
                    {"_type": "retrieval_diagnostics", "source": "manual_override"}
                ]

            elif knowledge_bases:
                try:
                    handler_id, retrieval_result = create_and_execute(
                        db_session=db.session,
                        query=message,
                        knowledge_base_configs=knowledge_bases,
                        max_context_tokens=max_context_tokens,
                        session_history=[m.to_litellm_input() for m in session_history],
                    )
                    rag_context = retrieval_result["formatted_context"]
                    retrieved_context_for_db = make_lightweight_retrieved_context(
                        retrieval_result["retrieved_context"]
                    )
                    context_handler_id = handler_id
                    query_enrichment = retrieval_result.get("query_enrichment")
                except Exception as e:
                    logger.warning(f"Failed to retrieve RAG context: {e}")
                    db.session.rollback()

            elif data.get("context_items"):
                resolved_items = _resolve_context_items(db.session, data["context_items"])
                if resolved_items:
                    from agentic.knowledge.models import RetrievedItem
                    from agentic_project_service.services.knowledge_search import (
                        format_items_as_context,
                    )

                    retrieved_items = [
                        RetrievedItem(
                            item_id=item.get("id", ""),
                            text=item["text"],
                            score=0.0,
                            source_id=item.get("source_id"),
                            knowledge_base_id=None,
                            meta=item.get("meta", {}),
                        )
                        for item in resolved_items
                    ]
                    citations_enabled_flag = data.get("citations_enabled", False)
                    rag_context, _diag = format_items_as_context(
                        retrieved_items,
                        max_tokens=max_context_tokens,
                        citations_enabled=citations_enabled_flag,
                    )
                    retrieved_context_for_db = [
                        {"_type": "retrieval_diagnostics", "source": "context_items"},
                    ] + resolved_items

            # Build system prompt and user message based on context type
            full_system_prompt = agent_system_prompt

            if isinstance(rag_context, list):
                logger.info(
                    "Using multimodal context with %d content blocks (streaming)", len(rag_context)
                )
                user_content: str | list[dict] = [
                    {"type": "text", "text": "Context from relevant documents:"},
                    *rag_context,
                    {"type": "text", "text": f"\n\n{message}"},
                ]
            elif rag_context:
                full_system_prompt = f"{agent_system_prompt}\n\nContext:\n{rag_context}"
                user_content = message
            else:
                user_content = message

            # ================================================================
            # ReAct loop path — when agent has tools assigned
            # ================================================================
            from ..services.tool_registry import load_all_tools_for_agent

            max_tool_output = get_setting("MAX_TOOL_OUTPUT_LENGTH")
            max_result_chars = get_setting("DEFAULT_MAX_RESULT_CHARS")
            tools = load_all_tools_for_agent(
                agent_id,
                db.session,
                max_tool_output_length=max_tool_output,
                default_max_result_chars=max_result_chars,
            )

            # Citation handling — gate on context being available either from
            # pre-fetched context_items or from tool calls (e.g. knowledge_search).
            citations_enabled = data.get("citations_enabled", False)
            citation_map: dict = {}
            if citations_enabled and (retrieved_context_for_db or tools):
                full_system_prompt = full_system_prompt + "\n\n" + build_citation_instruction()
                if retrieved_context_for_db:
                    citation_map = build_citation_map(retrieved_context_for_db)

            # abort_event must be defined before any try/except so the
            # GeneratorExit handler can reference it regardless of code path
            import threading as _threading

            abort_event = _threading.Event()

            if tools:
                from ..services.hook_loader import load_hooks_for_agent, load_tool_rules_for_agent

                hooks = load_hooks_for_agent(agent_id)
                tool_rules = load_tool_rules_for_agent(agent_id)
                fallback_model = agent_settings.get("fallback_model")

                # Agent execution settings (from agent config, with per-run overrides)
                temperature = data.get("temperature", agent_settings.get("temperature"))
                max_tokens_setting = agent_settings.get("max_tokens")
                timeout_seconds = data.get("timeout_seconds", agent_settings.get("timeout_seconds"))
                response_format = data.get("response_format")

                event_queue: queue.Queue = queue.Queue()

                context = ExecutionContext(
                    execution_id=run_id,
                    session_id=actual_session_id,
                    on_event=lambda e: event_queue.put(e),
                    session_history=[m.to_litellm_input() for m in session_history],
                    abort_signal=abort_event,
                )

                react_agent = Agent(
                    model=agent_model,
                    system_prompt=full_system_prompt,
                    name=agent_name,
                    temperature=temperature,
                    max_tokens=max_tokens_setting,
                    api_key=provider_api_key,
                    reasoning_effort=agent_settings.get("reasoning_effort"),
                )

                # Build messages with history (drops cross-provider reasoning artifacts;
                # drop events flow through context.on_event into the SSE event_queue)
                messages_for_llm = build_messages_for_llm(
                    session_history=session_history,
                    target_model=agent_model,
                    context=context,
                    user_input=user_content,
                )

                # Early persist: create the run as RUNNING before the ReAct loop
                input_messages: list[dict[str, Any]] = [{"role": "user", "content": message}]
                if query_enrichment:
                    input_messages.append({"_type": "query_enrichment", **query_enrichment})

                persist_agent_run(
                    db_session=db.session,
                    db_session_uuid=db_session_uuid,
                    run_id=run_id,
                    status=AgentRunStatus.RUNNING,
                    input_messages=input_messages,
                    content="",
                    retrieved_context=retrieved_context_for_db,
                    started_at=started_at,
                    context_handler_id=context_handler_id,
                )
                db.session.commit()
                run_persisted = True

                # Send start event
                start_event = {
                    "event": "start",
                    "run_id": run_id,
                    "session_id": actual_session_id,
                    "context_handler_id": context_handler_id,
                    "reasoning_requested": (
                        react_agent._resolved_effort_for(react_agent.model) is not None
                    ),
                }
                yield f"data: {json.dumps(start_event)}\n\n"

                # Register the context so the approval endpoint can resume it
                register_run(run_id, context)

                # Run the ReAct loop in a background thread so events
                # stream to the client as they are emitted (live SSE).
                result_holder: list = []
                error_holder: list = []

                def run_agent():
                    # ContextVar values do NOT propagate across raw
                    # threading.Thread boundaries. Only asyncio.Task
                    # copies the calling context automatically;
                    # ThreadPoolExecutor.submit and bare threading.Thread
                    # do NOT (see services/run_context.py for the
                    # full propagation-rules table). Re-bind run_id here
                    # so the wrapped tool handlers fired inside
                    # react_agent.run() derive deterministic idempotency
                    # keys instead of falling back to uuid4. Spec line
                    # 132: retry of an agent_run must collide on
                    # UNIQUE(org_id, idem_key).
                    _worker_run_id_token = set_run_id(run_id)
                    try:
                        with billing.llm_call_scope():
                            out = react_agent.run(
                                messages_for_llm,
                                context=context,
                                tools=tools,
                                max_steps=agent_settings.get("max_steps", 25),
                                fallback_model=fallback_model,
                                tool_rules=tool_rules if tool_rules else None,
                                hooks=hooks if hooks else None,
                                response_format=response_format,
                                timeout_seconds=timeout_seconds,
                            )
                        result_holder.append(out)
                    except Exception as e:
                        error_holder.append(e)
                    finally:
                        reset_run_id(_worker_run_id_token)
                        unregister_run(run_id)
                        event_queue.put(None)  # sentinel

                # Propagate Flask before_request contextvars into the worker
                # thread (BYOK provider set, byok_lookup_degraded, billing run
                # id). Raw threading.Thread doesn't inherit context, so without
                # this snapshot the BillingLogger inside the LLM call reads an
                # empty current_byok_providers and charges every call against
                # AI-on-us — even when the project has a valid BYOK key. Same
                # fix as orchestrations.py:972; mirrors copy_context().run
                # pattern used in agentic.agent.agent + agentic.orchestration.
                _captured_ctx = contextvars.copy_context()
                worker = threading.Thread(target=lambda: _captured_ctx.run(run_agent), daemon=True)
                worker.start()

                # Drain events from the queue and yield them live.
                # Buffer-aware drain (β): content_delta accumulates into
                # content_buffer and is NOT persisted; reasoning_delta is
                # forwarded but NOT persisted; terminal events are persisted
                # + forwarded. ``events_for_db`` is at function scope so the
                # outer except handler can persist it on failure.
                while True:
                    try:
                        event = event_queue.get(timeout=30)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    if event is None:
                        break

                    event_type = event.get("type")

                    # β: capture content_delta into the buffer
                    if event_type == "content_delta":
                        content_buffer += event.get("delta", "")
                        delta_count += 1
                        try:
                            payload = json.dumps({"event": event_type, **event})
                        except (TypeError, ValueError):
                            payload = json.dumps(
                                {"event": event_type, "error": "non-serializable event data"}
                            )
                        yield f"data: {payload}\n\n"
                        continue

                    # reasoning_delta: forwarded only, NOT persisted
                    if event_type == "reasoning_delta":
                        delta_count += 1
                        try:
                            payload = json.dumps({"event": event_type, **event})
                        except (TypeError, ValueError):
                            payload = json.dumps(
                                {"event": event_type, "error": "non-serializable event data"}
                            )
                        yield f"data: {payload}\n\n"
                        continue

                    # Terminal events: persist + forward
                    events_for_db.append(event)
                    try:
                        event_payload = json.dumps({"event": event.get("type", "step"), **event})
                    except (TypeError, ValueError):
                        event_payload = json.dumps(
                            {
                                "event": event.get("type", "step"),
                                "error": "non-serializable event data",
                            }
                        )
                    yield f"data: {event_payload}\n\n"

                worker.join(timeout=5)

                if error_holder:
                    raise error_holder[0]
                if not result_holder:
                    raise RuntimeError("Agent thread finished without producing a result")

                output = result_holder[0]

                # β: terminal chunk + complete + persist all use the same
                # final_content to keep live SSE, post-complete refetch, and
                # reload-from-DB consistent (B1 fix).
                final_content = content_buffer if streaming_enabled else (output.content or "")

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

                # Yield final content
                if final_content:
                    yield f"data: {json.dumps({'event': 'chunk', 'content': final_content})}\n\n"

                # Persist the run
                run_status = (
                    AgentRunStatus.COMPLETED
                    if output.status.is_success()
                    else AgentRunStatus.FAILED
                )

                # Collect context_handler_ids from tool-based retrievals
                tool_handler_ids = [
                    e["context_handler_id"]
                    for e in events_for_db
                    if e.get("type") == "context_handler_created"
                ]
                if tool_handler_ids and not context_handler_id:
                    context_handler_id = tool_handler_ids[0]

                # Merge retrieved_context from tool-based retrievals
                final_retrieved_context = retrieved_context_for_db
                if tool_handler_ids:
                    all_tool_items: list[dict] = []
                    for hid in tool_handler_ids:
                        handler = get_context_handler(db.session, hid, resolve_text=False)
                        if handler and handler.get("retrieved_context"):
                            for rc_item in handler["retrieved_context"]:
                                if rc_item.get("_type") != "retrieval_diagnostics":
                                    all_tool_items.append(rc_item)
                    # Dedup tool-retrieved chunks by UUID so multi-knowledge_search
                    # runs don't produce ambiguous [N] citation keys. Same chunk
                    # returned from two searches occupies one position in the merged
                    # list, not two. Order-preserving: first occurrence wins
                    # (highest-scored copy from the first search that returned it).
                    _seen_ids: set[str] = set()
                    _deduped: list[dict] = []
                    for _item in all_tool_items:
                        _iid = _item.get("id")
                        if _iid is None:
                            # Defensive: items without an id field are kept as-is and never
                            # deduplicated (shouldn't happen in practice, but if a future
                            # storage migration changes the chunk schema, we don't want to
                            # silently drop chunks).
                            _deduped.append(_item)
                            continue
                        if _iid in _seen_ids:
                            continue
                        _seen_ids.add(_iid)
                        _deduped.append(_item)
                    all_tool_items = _deduped

                    if all_tool_items:
                        tool_retrieved_context = [
                            {
                                "_type": "retrieval_diagnostics",
                                "source": "tool_calls",
                                "total_items": len(all_tool_items),
                            },
                        ] + all_tool_items
                        if final_retrieved_context:
                            existing_items = [
                                i
                                for i in final_retrieved_context
                                if i.get("_type") != "retrieval_diagnostics"
                            ]
                            new_items = [
                                i
                                for i in tool_retrieved_context
                                if i.get("_type") != "retrieval_diagnostics"
                            ]
                            final_retrieved_context = (
                                [
                                    {
                                        "_type": "retrieval_diagnostics",
                                        "total_items": len(existing_items) + len(new_items),
                                    },
                                ]
                                + existing_items
                                + new_items
                            )
                        else:
                            final_retrieved_context = tool_retrieved_context
                        final_retrieved_context = make_lightweight_retrieved_context(
                            final_retrieved_context
                        )

                # Phase 2 patch — mirror the pre-fetched-context citation parsing
                # so tool-based agentic runs also populate ai.message_citations and
                # ship structured citations on the SSE complete event. Without this,
                # [N] markers in the LLM's response decay into dead text.
                citations_for_run: list[dict] = []
                if citations_enabled and final_retrieved_context:
                    _citation_map = build_citation_map(final_retrieved_context)
                    if _citation_map:
                        final_content, citations_for_run = parse_citations_from_response(
                            final_content, _citation_map
                        )

                reasoning = extract_reasoning_steps(output.messages) if output.messages else None

                update_agent_run(
                    db_session=db.session,
                    run_id=run_id,
                    status=run_status,
                    content=final_content,
                    output_messages=[
                        Message(
                            role="assistant",
                            content=final_content,
                            tool_calls=(
                                [tc.to_dict() for tc in output.tool_calls]
                                if output.tool_calls
                                else None
                            ),
                            reasoning=output.reasoning_artifact,
                            reasoning_requested=output.reasoning_requested,
                        ).model_dump(exclude_none=True)
                    ],
                    usage=output.usage,
                    error=output.error if not output.status.is_success() else None,
                    completed_at=output.completed_at,
                    steps=output.steps,
                    events=events_for_db,
                    tool_calls=strip_tool_call_images(
                        [tc.to_dict() for tc in output.tool_calls],
                        events_for_db,
                        db.session,
                    ),
                    context_handler_id=context_handler_id,
                    reasoning_steps=reasoning,
                    retrieved_context=final_retrieved_context,
                )
                db.session.commit()

                if citations_for_run:
                    persist_citations(db.session, run_id, citations_for_run)
                    db.session.commit()

                # Billing: post the agent_run dispatch fee on success only.
                # Tool calls inside the ReAct loop are independently billed
                # in tool_registry via Task 15 — this is the dispatch fee
                # only. Failed runs are not charged.
                if run_status == AgentRunStatus.COMPLETED:
                    billing.charge(
                        action="agent_run",
                        quantity=1,
                        ref_type="agent_run",
                        ref_id=run_id,
                        idempotency_parts=(run_id,),
                        metadata={
                            "agent_id": str(agent_id),
                            "model": agent_model,
                            "streaming": True,
                            "react_loop": True,
                            "tool_calls": len(output.tool_calls) if output.tool_calls else 0,
                        },
                    )

                complete_event = {
                    "event": "complete",
                    "run_id": run_id,
                    "session_id": actual_session_id,
                    "content": final_content,
                    "status": run_status.value,
                    "steps": output.steps,
                    "tool_calls": [tc.to_dict() for tc in output.tool_calls],
                    "usage": output.usage,
                    # Include error in the complete envelope when the run
                    # finished in a failed state. Without this, FE clients
                    # that only listen for `event=error` (which fires only
                    # on outer exceptions, not handled-and-stored failures)
                    # see a "complete" with status=failed and no message —
                    # users perceive this as the platform suppressing the
                    # error.
                    "error": output.error if not output.status.is_success() else None,
                }
                if citations_for_run:
                    complete_event["citations"] = citations_for_run
                try:
                    complete_payload = json.dumps(complete_event)
                except (TypeError, ValueError) as e:
                    logger.warning("Failed to serialize complete event: %s", e)
                    complete_payload = json.dumps(
                        {
                            "event": "complete",
                            "run_id": run_id,
                            "session_id": actual_session_id,
                            "content": final_content,
                            "status": run_status.value,
                            "steps": output.steps,
                            "tool_calls": [],
                            "usage": output.usage,
                            "error": output.error if not output.status.is_success() else None,
                            "serialization_warning": str(e),
                        }
                    )
                yield f"data: {complete_payload}\n\n"
                return  # Don't fall through to the existing streaming path

            # ================================================================
            # Standard streaming path — no tools (existing behavior unchanged)
            # ================================================================

            # Create the agentic Agent instance
            agent = Agent(
                model=agent_model,
                system_prompt=full_system_prompt,
                name=agent_name,
                api_key=provider_api_key,
                reasoning_effort=agent_settings.get("reasoning_effort"),
            )

            # Build messages with history (drops cross-provider reasoning artifacts;
            # drop events are collected for SSE forwarding before streaming starts)
            stream_drop_events: list[dict] = []
            stream_drop_context = ExecutionContext(
                on_event=lambda e: stream_drop_events.append(e),
            )
            messages_for_llm = build_messages_for_llm(
                session_history=session_history,
                target_model=agent_model,
                context=stream_drop_context,
                user_input=user_content,
            )

            # --- Early persist: create the run as RUNNING before streaming ---
            input_messages: list[dict[str, Any]] = [{"role": "user", "content": message}]
            if query_enrichment:
                input_messages.append({"_type": "query_enrichment", **query_enrichment})

            persist_agent_run(
                db_session=db.session,
                db_session_uuid=db_session_uuid,
                run_id=run_id,
                status=AgentRunStatus.RUNNING,
                input_messages=input_messages,
                content="",
                retrieved_context=retrieved_context_for_db,
                started_at=started_at,
                context_handler_id=context_handler_id,
            )
            db.session.commit()
            run_persisted = True

            # Cache once — used by start event, persistence, and the background
            # finisher.
            reasoning_was_requested = agent._resolved_effort_for(agent.model) is not None

            # Send initial event — session and run are now committed to DB
            start_event = {
                "event": "start",
                "run_id": run_id,
                "session_id": actual_session_id,
                "context_handler_id": context_handler_id,
                "reasoning_requested": reasoning_was_requested,
            }
            if citation_map:
                start_event["citation_candidates"] = list(citation_map.values())
            yield f"data: {json.dumps(start_event)}\n\n"

            # Forward any cross-provider reasoning drop events collected during
            # build_messages_for_llm
            for drop_evt in stream_drop_events:
                yield f"data: {json.dumps({'event': drop_evt.get('type', 'step'), **drop_evt})}\n\n"

            # Per-token delta buffer; drained between generator yields. Empty
            # in kill-switch-off mode (callbacks are None).
            naive_delta_buf: collections.deque[dict] = collections.deque()
            # Flipped to False before handing off to _finish_run_in_background
            # so the background-drain doesn't keep appending into a buffer
            # nobody reads.
            callbacks_active = [True]

            def _on_content_delta(d: str) -> None:
                if not callbacks_active[0]:
                    return
                naive_delta_buf.append(
                    {"event": "content_delta", "type": "content_delta", "delta": d}
                )

            def _on_reasoning_delta(d: str) -> None:
                # step=1 + source="thinking" match the ReAct payload shape so
                # the FE's buildReasoningSteps (which gates accumulation on
                # ev.step) attributes these to a step and the pill leaves
                # `pre-stream`. The naive branch has no loop, so step is 1.
                if not callbacks_active[0]:
                    return
                naive_delta_buf.append(
                    {
                        "event": "reasoning_delta",
                        "type": "reasoning_delta",
                        "step": 1,
                        "source": "thinking",
                        "delta": d,
                    }
                )

            llm_gen = agent.stream(
                messages_for_llm,
                context=ExecutionContext(
                    session_id=actual_session_id,
                    abort_signal=abort_event,
                ),
                on_content_delta=_on_content_delta if streaming_enabled else None,
                on_reasoning_delta=_on_reasoning_delta if streaming_enabled else None,
            )
            output = None

            with billing.llm_call_scope():
                try:
                    while True:
                        chunk = next(llm_gen)
                        while naive_delta_buf:
                            evt = naive_delta_buf.popleft()
                            yield f"data: {json.dumps(evt)}\n\n"
                        content_chunks.append(chunk)
                        if not streaming_enabled:
                            yield f"data: {json.dumps({'event': 'chunk', 'content': chunk})}\n\n"
                except StopIteration as e:
                    output = e.value
                except GeneratorExit:
                    logger.info(
                        "Client disconnected during stream for run %s, finishing in background",
                        run_id,
                    )
                    callbacks_active[0] = False
                    # PR 421 R4 C9: snapshot caller contextvars
                    # (current_byok_providers / byok_lookup_degraded /
                    # run_id_var) so the background-finish thread sees
                    # them. Without this wrap, the disconnect path re-introduces
                    # the BYOK bypass regression — LLM completion continues in
                    # the worker but BillingLogger reads frozenset() and bills
                    # the call as AI-on-us.
                    _disconnect_ctx = contextvars.copy_context()
                    _finish_kwargs = {
                        "citation_map": citation_map if citation_map else None,
                    }
                    t = threading.Thread(
                        target=lambda: _disconnect_ctx.run(
                            _finish_run_in_background,
                            app,
                            run_id,
                            llm_gen,
                            content_chunks,
                            message,
                            query_enrichment,
                            retrieved_context_for_db,
                            context_handler_id,
                            started_at,
                            reasoning_was_requested,
                            **_finish_kwargs,
                        ),
                        daemon=True,
                    )
                    t.start()
                    return

            while naive_delta_buf:
                evt = naive_delta_buf.popleft()
                yield f"data: {json.dumps(evt)}\n\n"

            final_content = "".join(content_chunks)

            # Parse and validate citations
            citations: list[dict] = []
            if citation_map:
                final_content, citations = parse_citations_from_response(
                    final_content, citation_map
                )

            # Terminal `chunk` carries full content (matches ReAct branch).
            if streaming_enabled and final_content:
                yield f"data: {json.dumps({'event': 'chunk', 'content': final_content})}\n\n"

            # Synthesize a terminal `reasoning` event from the assembled
            # artifact's summary_text and persist it. ReAct emits one per
            # iteration via the `reasoning_text` branch in `Agent._react_loop`,
            # which then lands in agent_runs.events; the FE's
            # buildReasoningSteps replays from that column on reload to
            # populate the pill's expanded panel. Without this, naive runs
            # render "Thought for Xs · 0 steps" with an empty expanded panel
            # after refresh, even though the message's reasoning artifact
            # persists correctly.
            artifact = output.reasoning_artifact if output else None
            reasoning_text = getattr(artifact, "summary_text", None) if artifact else None
            if reasoning_text:
                reasoning_event = {
                    "type": "reasoning",
                    "step": 1,
                    "source": "thinking",
                    "content": reasoning_text,
                }
                events_for_db.append(reasoning_event)
                yield f"data: {json.dumps({'event': 'reasoning', **reasoning_event})}\n\n"

            # Determine actual status from agent output
            run_status = (
                AgentRunStatus.COMPLETED
                if output and output.status.value == "completed"
                else AgentRunStatus.FAILED
            )
            run_error = output.error if output else "Unknown error — no output from agent"

            if run_status == AgentRunStatus.FAILED:
                logger.warning("Agent streaming completed with failed status: %s", run_error)
                yield f"data: {json.dumps({'event': 'error', 'error': run_error, 'run_id': run_id})}\n\n"

            # Update the existing run to COMPLETED/FAILED
            reasoning = (
                extract_reasoning_steps(output.messages) if output and output.messages else None
            )

            update_agent_run(
                db_session=db.session,
                run_id=run_id,
                status=run_status,
                content=final_content,
                output_messages=[
                    Message(
                        role="assistant",
                        content=final_content,
                        tool_calls=(
                            [tc.to_dict() for tc in output.tool_calls]
                            if output and output.tool_calls
                            else None
                        ),
                        reasoning=output.reasoning_artifact if output else None,
                        reasoning_requested=(
                            output.reasoning_requested if output else reasoning_was_requested
                        ),
                    ).model_dump(exclude_none=True)
                ],
                usage=output.usage if output else None,
                error=run_error if run_status == AgentRunStatus.FAILED else None,
                completed_at=datetime.now(UTC),
                events=events_for_db if events_for_db else None,
                reasoning_steps=reasoning,
            )
            db.session.commit()

            if citations:
                persist_citations(db.session, run_id, citations)
                db.session.commit()

            # Billing: post the agent_run dispatch fee on success only. The
            # standard (no-tools) streaming path has no internal atomic ops
            # to bill — retrievals would already have been billed in the
            # context-prep step above via knowledge_search.
            if run_status == AgentRunStatus.COMPLETED:
                billing.charge(
                    action="agent_run",
                    quantity=1,
                    ref_type="agent_run",
                    ref_id=run_id,
                    idempotency_parts=(run_id,),
                    metadata={
                        "agent_id": str(agent_id),
                        "model": agent_model,
                        "streaming": True,
                        "react_loop": False,
                    },
                )

            # Send completion event
            item_count = len(retrieved_context_for_db) - 1 if retrieved_context_for_db else 0
            complete_event = {
                "event": "complete",
                "run_id": run_id,
                "session_id": actual_session_id,
                "context_handler_id": context_handler_id,
                "content": final_content,
                "retrieved_items": item_count,
                "is_new_session": is_new_session,
                "status": run_status.value,
                # `error` is also sent ahead of this complete envelope as
                # its own event (see the `event: error` yield above), but
                # we duplicate here so any client that only watches
                # `complete` can still surface the underlying message.
                "error": run_error if run_status == AgentRunStatus.FAILED else None,
            }
            if citations:
                complete_event["citations"] = citations
            try:
                complete_payload = json.dumps(complete_event)
            except (TypeError, ValueError) as e:
                logger.warning("Failed to serialize complete event (streaming path): %s", e)
                complete_event.pop("citations", None)
                complete_payload = json.dumps(complete_event)
            yield f"data: {complete_payload}\n\n"

        except GeneratorExit:
            # Client disconnected before streaming started.
            abort_event.set()
            logger.info(
                "Client disconnected (pre-stream) for run %s, run_persisted=%s",
                run_id,
                run_persisted,
            )
            if llm_gen is not None:
                # PR 421 R4 C9: snapshot caller contextvars for the
                # pre-stream disconnect path — same rationale as the
                # mid-stream disconnect ~150 lines above.
                _predisconnect_ctx = contextvars.copy_context()
                _predisconnect_kwargs = {
                    "citation_map": citation_map if citation_map else None,
                }
                t = threading.Thread(
                    target=lambda: _predisconnect_ctx.run(
                        _finish_run_in_background,
                        app,
                        run_id,
                        llm_gen,
                        content_chunks,
                        message,
                        query_enrichment,
                        retrieved_context_for_db,
                        context_handler_id,
                        started_at,
                        reasoning_was_requested,
                        **_predisconnect_kwargs,
                    ),
                    daemon=True,
                )
                t.start()
            elif run_persisted:
                # Run exists in DB — update it to FAILED
                try:
                    update_agent_run(
                        db_session=db.session,
                        run_id=run_id,
                        status=AgentRunStatus.FAILED,
                        error="Client disconnected before LLM streaming started",
                        completed_at=datetime.now(UTC),
                    )
                    db.session.commit()
                except Exception:
                    logger.exception(
                        "Failed to mark run %s as failed after early disconnect", run_id
                    )
            else:
                # Run was never persisted — create session + run as FAILED
                try:
                    db.session.rollback()
                    db_session_uuid, _, _ = get_or_create_session(
                        db_session=db.session,
                        agent_id=agent_id,
                        session_id=session_id,
                        user_id=user_id,
                    )
                    db.session.commit()
                    persist_agent_run(
                        db_session=db.session,
                        db_session_uuid=db_session_uuid,
                        run_id=run_id,
                        status=AgentRunStatus.FAILED,
                        input_messages=[{"role": "user", "content": message}],
                        error="Client disconnected before run was persisted",
                        started_at=started_at,
                        completed_at=datetime.now(UTC),
                        context_handler_id=context_handler_id,
                    )
                    db.session.commit()
                except Exception:
                    logger.exception(
                        "Failed to persist failed run %s after early disconnect", run_id
                    )
            return

        except Exception as e:
            logger.exception("Streaming agent run failed")
            db.session.rollback()

            # Try to update the existing run to FAILED (it was already persisted as RUNNING)
            # M2 v3: persist events_for_db on failure too — synthetic terminal
            # chunk/reasoning events emitted before the error must reach the
            # events JSONB column. (events_for_db is at function scope; for the
            # chat-style branch it remains [] which is harmless.)
            try:
                update_agent_run(
                    db_session=db.session,
                    run_id=run_id,
                    status=AgentRunStatus.FAILED,
                    error=str(e),
                    completed_at=datetime.now(UTC),
                    events=events_for_db,
                )
                db.session.commit()
            except Exception:
                logger.exception("Failed to update run %s to failed", run_id)
                # Last resort: try to create the run if early persist hadn't committed
                try:
                    db.session.rollback()
                    db_session_uuid, actual_session_id, _ = get_or_create_session(
                        db_session=db.session,
                        agent_id=agent_id,
                        session_id=session_id,
                        user_id=user_id,
                    )
                    persist_agent_run(
                        db_session=db.session,
                        db_session_uuid=db_session_uuid,
                        run_id=run_id,
                        status=AgentRunStatus.FAILED,
                        input_messages=[{"role": "user", "content": message}],
                        error=str(e),
                        started_at=started_at,
                        completed_at=datetime.now(UTC),
                        context_handler_id=context_handler_id,
                    )
                    db.session.commit()
                except Exception:
                    logger.exception("Failed to persist failed run %s", run_id)

            yield f"data: {json.dumps({'event': 'error', 'error': str(e), 'run_id': run_id, 'context_handler_id': context_handler_id})}\n\n"
        finally:
            # Unbind run_id from the streaming worker's contextvar so the
            # next unrelated request on this same worker thread doesn't
            # inherit a stale id. Runs on client disconnect as well.
            reset_run_id(_stream_run_id_token)

    return Response(
        stream_with_context(generate_sse()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@agents_bp.route("/runs/<run_id>/approve", methods=["POST"])
@require_auth
def approve_run(run_id: str):
    """Resume a paused ReAct run by supplying an approval decision.

    The run must be currently registered in the in-memory run registry,
    meaning it is actively waiting for a human-in-the-loop approval.
    """
    data = request.get_json(silent=True)
    if not data or "approved" not in data:
        return jsonify({"error": "Request body must include 'approved' (boolean)"}), 400
    if not isinstance(data["approved"], bool):
        return jsonify({"error": "'approved' must be a boolean"}), 400
    context = get_active_run_context(run_id)
    if not context:
        return jsonify({"error": "Run not found or not waiting for approval"}), 404
    context.set_approval_decision(data)
    return jsonify({"status": "resumed"})


@agents_bp.route("/runs/<run_id>", methods=["GET"])
@require_auth
def get_agent_run(run_id: str):
    """Fetch a single agent_run row by run_id, independent of session/parent context.

    Ownership scoping: when session_id IS NOT NULL, the run belongs to a session
    whose user_id must match the caller (service-role bypasses this check).
    For delegated/block runs (session_id IS NULL) there is no per-user ownership
    model yet, so require_auth is sufficient — tenant isolation is enforced at
    the project boundary (Kong + project-scoped auth).
    """
    row = AgentRun.query.filter_by(run_id=run_id).first()
    if not row:
        return jsonify({"error": "Run not found"}), 404

    if row.session_id:
        is_service_role = (getattr(g, "jwt_payload", None) or {}).get("is_service_role", False)
        if not is_service_role:
            # row.session_id is the UUID FK to ai.agent_sessions.id. The
            # get_session_owner helper keys by the user-facing sess_xxx string
            # (VARCHAR), so we query by UUID primary key here instead.
            session_row = AgentSession.query.filter_by(id=row.session_id).first()
            owner = str(session_row.user_id) if (session_row and session_row.user_id) else None
            if owner is not None and owner != get_current_user_id():
                # Return same shape as not-found to avoid leaking existence
                return jsonify({"error": "Run not found"}), 404

    return jsonify(
        {
            "id": str(row.id),
            "run_id": row.run_id,
            "session_id": str(row.session_id) if row.session_id else None,
            "parent_orchestration_run_id": (
                str(row.parent_orchestration_run_id) if row.parent_orchestration_run_id else None
            ),
            "parent_workflow_execution_id": (
                str(row.parent_workflow_execution_id) if row.parent_workflow_execution_id else None
            ),
            "status": row.status,
            "input_messages": row.input_messages,
            "output_messages": row.output_messages,
            "content": row.content,
            "usage": row.usage,
            "retrieved_context": row.retrieved_context,
            "error": row.error,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "steps": row.steps,
            "events": row.events,
            "tool_calls": row.tool_calls,
            "reasoning_steps": row.reasoning_steps,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
    )
