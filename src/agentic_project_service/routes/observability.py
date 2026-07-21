"""Observability dashboard routes for the project service (C2.1).

The Studio /observability page used to read ai.agent_runs / ai.sources /
ai.indexed_sources / ai.workflow_executions / ai.orchestration_runs /
ai.workflow_block_logs / ai.tool_call_events / ai.agents directly via
useProjectSupabaseClient().client — a PostgREST proxy onto the `ai` schema.
C2.2 removes `ai` from PGRST_DB_SCHEMAS, so every such read moves here first.

Every endpoint mirrors the exact bounded-row-fetch-then-aggregate-client-side
pattern the six data/observability/*.ts hooks already use: this file does
NOT reimplement their bucketing/percentile/tally aggregation logic — it just
proxies the same filtered SELECT the hooks used to run via PostgREST, so
that logic (already reviewed, already correct) stays untouched.
"""

from flask import Blueprint, jsonify, request
from sqlalchemy import text

from ..auth import require_auth
from ..db import db, AI_SCHEMA

observability_bp = Blueprint("observability", __name__, url_prefix="/api/observability")


def _parse_csv(param: str | None) -> list[str]:
    if not param:
        return []
    return [p.strip() for p in param.split(",") if p.strip()]


def _parse_limit(default: int, cap: int) -> int:
    try:
        return min(max(int(request.args.get("limit", default)), 1), cap)
    except ValueError:
        return default


@observability_bp.route("/agent-runs", methods=["GET"])
@require_auth
def list_observability_agent_runs():
    """Bounded ai.agent_runs rows for the runs-chart + tokens-by-agent-runs
    hooks. `since` is required (both callers always compute a window)."""
    since = request.args.get("since")
    if not since:
        return jsonify({"error": "since is required"}), 400
    until = request.args.get("until")
    models = _parse_csv(request.args.get("models"))
    agent_ids = _parse_csv(request.args.get("agent_ids"))
    limit = _parse_limit(default=5000, cap=50_000)

    where = ["created_at >= CAST(:since AS timestamptz)"]
    params: dict = {"since": since, "limit": limit}
    if until:
        where.append("created_at < CAST(:until AS timestamptz)")
        params["until"] = until
    if models:
        where.append("model = ANY(:models)")
        params["models"] = models
    if agent_ids:
        where.append("agent_id = ANY(CAST(:agent_ids AS uuid[]))")
        params["agent_ids"] = agent_ids
    where_sql = " AND ".join(where)

    rows = db.session.execute(
        text(f"""
            SELECT id, status, created_at, started_at, completed_at, error,
                   model, agent_id, prompt_tokens, completion_tokens,
                   reasoning_tokens, total_tokens
            FROM "{AI_SCHEMA}".agent_runs
            WHERE {where_sql}
            ORDER BY created_at DESC
            LIMIT :limit
        """),
        params,
    ).fetchall()

    runs = [
        {
            "id": str(r.id),
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            "error": r.error,
            "model": r.model,
            "agent_id": str(r.agent_id) if r.agent_id else None,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "reasoning_tokens": r.reasoning_tokens,
            "total_tokens": r.total_tokens,
        }
        for r in rows
    ]
    return jsonify({"runs": runs, "truncated": len(runs) == limit})


@observability_bp.route("/orchestration-runs", methods=["GET"])
@require_auth
def list_observability_orchestration_runs():
    """Bounded ai.orchestration_runs rows for the tokens dashboard."""
    since = request.args.get("since")
    if not since:
        return jsonify({"error": "since is required"}), 400
    until = request.args.get("until")
    models = _parse_csv(request.args.get("models"))
    limit = _parse_limit(default=50_000, cap=50_000)

    where = ["created_at >= CAST(:since AS timestamptz)"]
    params: dict = {"since": since, "limit": limit}
    if until:
        where.append("created_at < CAST(:until AS timestamptz)")
        params["until"] = until
    if models:
        where.append("model = ANY(:models)")
        params["models"] = models
    where_sql = " AND ".join(where)

    rows = db.session.execute(
        text(f"""
            SELECT id, created_at, status, model, prompt_tokens,
                   completion_tokens, reasoning_tokens, total_tokens
            FROM "{AI_SCHEMA}".orchestration_runs
            WHERE {where_sql}
            ORDER BY created_at DESC
            LIMIT :limit
        """),
        params,
    ).fetchall()

    runs = [
        {
            "id": str(r.id),
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "status": r.status,
            "model": r.model,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "reasoning_tokens": r.reasoning_tokens,
            "total_tokens": r.total_tokens,
        }
        for r in rows
    ]
    return jsonify({"runs": runs, "truncated": len(runs) == limit})


@observability_bp.route("/workflow-block-logs", methods=["GET"])
@require_auth
def list_observability_workflow_block_logs():
    """Bounded ai.workflow_block_logs rows (block_type='agent' only — the
    sole caller, the tokens dashboard, never wants other block types) for
    the tokens dashboard."""
    since = request.args.get("since")
    if not since:
        return jsonify({"error": "since is required"}), 400
    until = request.args.get("until")
    models = _parse_csv(request.args.get("models"))
    limit = _parse_limit(default=50_000, cap=50_000)

    where = ["block_type = 'agent'", "created_at >= CAST(:since AS timestamptz)"]
    params: dict = {"since": since, "limit": limit}
    if until:
        where.append("created_at < CAST(:until AS timestamptz)")
        params["until"] = until
    if models:
        where.append("model = ANY(:models)")
        params["models"] = models
    where_sql = " AND ".join(where)

    rows = db.session.execute(
        text(f"""
            SELECT id, created_at, status, block_type, model, prompt_tokens,
                   completion_tokens, reasoning_tokens, total_tokens
            FROM "{AI_SCHEMA}".workflow_block_logs
            WHERE {where_sql}
            ORDER BY created_at DESC
            LIMIT :limit
        """),
        params,
    ).fetchall()

    logs = [
        {
            "id": str(r.id),
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "status": r.status,
            "block_type": r.block_type,
            "model": r.model,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "reasoning_tokens": r.reasoning_tokens,
            "total_tokens": r.total_tokens,
        }
        for r in rows
    ]
    return jsonify({"logs": logs, "truncated": len(logs) == limit})


@observability_bp.route("/tool-calls", methods=["GET"])
@require_auth
def list_observability_tool_calls():
    """Bounded ai.tool_call_events rows for the tool-call panels."""
    since = request.args.get("since")
    if not since:
        return jsonify({"error": "since is required"}), 400
    until = request.args.get("until")
    models = _parse_csv(request.args.get("models"))
    agent_ids = _parse_csv(request.args.get("agent_ids"))
    limit = _parse_limit(default=50_000, cap=50_000)

    where = ["occurred_at >= CAST(:since AS timestamptz)"]
    params: dict = {"since": since, "limit": limit}
    if until:
        where.append("occurred_at < CAST(:until AS timestamptz)")
        params["until"] = until
    if models:
        where.append("model = ANY(:models)")
        params["models"] = models
    if agent_ids:
        where.append("agent_id = ANY(CAST(:agent_ids AS uuid[]))")
        params["agent_ids"] = agent_ids
    where_sql = " AND ".join(where)

    rows = db.session.execute(
        text(f"""
            SELECT tool_name, status, duration_ms, agent_id, model, occurred_at
            FROM "{AI_SCHEMA}".tool_call_events
            WHERE {where_sql}
            ORDER BY occurred_at DESC
            LIMIT :limit
        """),
        params,
    ).fetchall()

    events = [
        {
            "tool_name": r.tool_name,
            "status": r.status,
            "duration_ms": r.duration_ms,
            "agent_id": str(r.agent_id) if r.agent_id else None,
            "model": r.model,
            "occurred_at": r.occurred_at.isoformat() if r.occurred_at else None,
        }
        for r in rows
    ]
    return jsonify({"events": events, "truncated": len(events) == limit})


@observability_bp.route("/extraction-status", methods=["GET"])
@require_auth
def get_observability_extraction_status():
    """Raw (extraction_status | index_status) columns, bounded at 10k rows
    each — the extraction/indexing status donuts tally these client-side."""
    source_rows = db.session.execute(
        text(f"""
            SELECT extraction_status FROM "{AI_SCHEMA}".sources LIMIT 10000
        """)
    ).fetchall()
    indexed_rows = db.session.execute(
        text(f"""
            SELECT index_status FROM "{AI_SCHEMA}".indexed_sources LIMIT 10000
        """)
    ).fetchall()
    return jsonify(
        {
            "sources": [{"extraction_status": r.extraction_status} for r in source_rows],
            "indexed_sources": [{"index_status": r.index_status} for r in indexed_rows],
        }
    )


@observability_bp.route("/filter-options", methods=["GET"])
@require_auth
def get_observability_filter_options():
    """Distinct agent_runs.model values + all agents — populates the
    observability filter bar's model/agent dropdowns."""
    model_rows = db.session.execute(
        text(f"""
            SELECT DISTINCT model FROM "{AI_SCHEMA}".agent_runs
            WHERE model IS NOT NULL
            ORDER BY model
            LIMIT 10000
        """)
    ).fetchall()
    agent_rows = db.session.execute(
        text(f"""
            SELECT id, name FROM "{AI_SCHEMA}".agents
            ORDER BY name ASC
            LIMIT 500
        """)
    ).fetchall()
    return jsonify(
        {
            "models": [r.model for r in model_rows],
            "agents": [{"id": str(r.id), "name": r.name} for r in agent_rows],
        }
    )


@observability_bp.route("/agents-lookup", methods=["GET"])
@require_auth
def get_observability_agents_lookup():
    """Name resolution for a specific set of agent ids (tokens dashboard's
    "group by agent" dimension)."""
    ids = _parse_csv(request.args.get("ids"))
    if not ids:
        return jsonify({"agents": []})
    rows = db.session.execute(
        text(f"""
            SELECT id, name FROM "{AI_SCHEMA}".agents
            WHERE id = ANY(CAST(:ids AS uuid[]))
        """),
        {"ids": ids},
    ).fetchall()
    return jsonify({"agents": [{"id": str(r.id), "name": r.name} for r in rows]})


# Fixed thresholds match the FE's STUCK_EXTRACTION_THRESHOLD_MINUTES /
# STUCK_WORKFLOW_THRESHOLD_MINUTES constants (use-project-health-query.ts) —
# the hook accepts a `range` param but never actually uses it for these
# queries, so this endpoint takes no params either.
_STUCK_EXTRACTION_THRESHOLD_MINUTES = 10
_STUCK_WORKFLOW_THRESHOLD_MINUTES = 5


@observability_bp.route("/health", methods=["GET"])
@require_auth
def get_observability_health():
    """The 5 stat-card counts at the top of /observability."""
    row = db.session.execute(
        text(f"""
            SELECT
              (SELECT COUNT(*) FROM "{AI_SCHEMA}".agent_runs WHERE status = 'running') AS active_runs,
              (SELECT COUNT(*) FROM "{AI_SCHEMA}".agent_runs
                 WHERE status = 'failed' AND created_at >= NOW() - INTERVAL '24 hours') AS failed_runs_24h,
              (SELECT COUNT(*) FROM "{AI_SCHEMA}".sources
                 WHERE extraction_status = 'extracting'
                   AND updated_at < NOW() - (:extraction_minutes || ' minutes')::interval) AS stuck_extractions,
              (SELECT COUNT(*) FROM "{AI_SCHEMA}".indexed_sources
                 WHERE index_status = 'failed') AS failed_indexed_sources,
              (SELECT COUNT(*) FROM "{AI_SCHEMA}".workflow_executions
                 WHERE status = 'running'
                   AND started_at < NOW() - (:workflow_minutes || ' minutes')::interval) AS running_workflows
        """),
        {
            "extraction_minutes": _STUCK_EXTRACTION_THRESHOLD_MINUTES,
            "workflow_minutes": _STUCK_WORKFLOW_THRESHOLD_MINUTES,
        },
    ).fetchone()

    return jsonify(
        {
            "activeRuns": row.active_runs,
            "failedRuns24h": row.failed_runs_24h,
            "stuckExtractions": row.stuck_extractions,
            "failedIndexedSources": row.failed_indexed_sources,
            "runningWorkflows": row.running_workflows,
        }
    )
