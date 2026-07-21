"""Knowledge base management routes for the project service."""

import json
import logging
import uuid

from flask import Blueprint, jsonify, request
from sqlalchemy import text

from ..auth import require_auth
from ..celery import celery_app
from ..db import db, AI_SCHEMA
from ..services.ai_provider_keys_resolver import get_all_user_provider_keys
from ..services.settings_registry import get_setting
from ..services.sparse_retrieval import (
    SparseIndexStore,
    STRATEGY_TO_BM25_ITEM_TABLE as _STRATEGY_TO_ITEM_TABLE,
)
from ..services import billing_port as billing
from ..strategies import get_strategy
from ..tasks.indexing import (
    build_bm25_for_kb,
    index_source,
    reindex_knowledge_base,
    reenrich_graph_references,
)

logger = logging.getLogger(__name__)


# Conservative pre-charge estimate for an indexing op at queue time. The
# worker charges the actual 1k-token quantity once known. 200 units
# covers a typical mid-sized document (~200k tokens × catalog top price 5).
_INDEXING_ESTIMATED_QUANTITY: int = 200
# Per the catalog, the most expensive indexing action is indexing_graphindex
# at 5 credits/1k-tokens. Use it as the pre-charge upper bound so the check
# rejects the queue when the user's free-tier balance can't cover even the
# cheapest strategy.
_INDEXING_MAX_UNIT_CREDITS: int = 5


def _maybe_indexing_billing_kwargs(*, action: str, ref_id: str) -> dict[str, object]:
    """Build the billing key inputs threaded to an ``index_source`` task.

    Identity (org/project) and the idempotency key itself are no longer computed
    here — the billing port's adapter derives org from context and builds the
    key from ``(action, ref_id)``. The single-index route passes the resolved
    ``indexing_<strategy>`` as ``action`` (key action == billed action); the
    batch-reindex routes pass the literal ``"indexing"`` (key action != billed
    action — the split). Threaded unconditionally; the adapter no-ops the charge
    when billing is unconfigured."""
    return {
        "idempotency_action": action,
        "idempotency_parts": [ref_id],
    }


# Maps ai.enrichment_configs.status (DB CHECK: idle/enriching/completed/
# completed_with_errors/failed) to the API enrichment_status enum the
# frontend renders (none/enriching/enriched/failed).
#
# 'completed_with_errors' maps to 'enriched' because partial-success is
# still a completed enrichment; the UI surfaces the partial count via
# enrichment_progress.{enriched_count,total_count}.
_ENRICHMENT_STATUS_MAP = {
    None: "none",
    "idle": "none",
    "enriching": "enriching",
    "completed": "enriched",
    "completed_with_errors": "enriched",
    "failed": "failed",
}


def _count_graph_nodes_for_kb(kb_id: str, indexed_source_id: str | None) -> int:
    """Node count for a graph_index KB. Raises on lookup failure (fail-closed)."""
    from ..services.graph_index_store import GraphIndexStore

    store = GraphIndexStore(db_session=db.session, knowledge_base_id=kb_id)
    return store.count_nodes(indexed_source_id=indexed_source_id)


def _graph_check_balance(kb_id: str, indexed_source_id: str | None) -> None:
    """Size-aware pre-op balance check for graph enrichment/re-enrichment.

    Estimates node_count × _INDEXING_MAX_UNIT_CREDITS. Fails closed (503) on
    lookup error so a DB hiccup can't transiently restore the flat 1000 leak.
    Routed through the billing port — the no-op adapter makes this inert in
    OSS/unit-test/local-dev builds; the cloud adapter enforces the cap.
    """
    try:
        node_count = _count_graph_nodes_for_kb(kb_id, indexed_source_id)
    except Exception as exc:
        logger.error("graph node count lookup failed for kb=%s; failing closed: %s", kb_id, exc)
        from werkzeug.exceptions import ServiceUnavailable

        raise ServiceUnavailable(
            "Balance check failed (graph node count lookup error). Retry shortly."
        )
    estimated_cost = max(
        _INDEXING_ESTIMATED_QUANTITY * _INDEXING_MAX_UNIT_CREDITS,  # floor = old constant
        node_count * _INDEXING_MAX_UNIT_CREDITS,
    )
    billing.check_balance(estimated_cost=estimated_cost)


knowledge_bases_bp = Blueprint("knowledge_bases", __name__, url_prefix="/api/knowledge-bases")


def _require_uuid(value: str, label: str = "id", status_code: int = 404):
    """Return an error response tuple if `value` is not a valid UUID; else None."""
    try:
        uuid.UUID(value)
        return None
    except ValueError:
        return jsonify({"error": f"Invalid {label}"}), status_code


def _compute_drift(kb_id: str, current_config: dict) -> str:
    """Returns 'none' | 'enrichment_only' | 'full'.

    Compares each indexed_source's stored indexing_config_snapshot against
    the KB's current indexing_config. Excludes pending/indexing rows because
    their snapshot is set when they finish, not when they start.
    """
    current_json = json.dumps(current_config or {})

    has_drift = db.session.execute(
        text(f"""
            SELECT EXISTS (
                SELECT 1 FROM "{AI_SCHEMA}".indexed_sources
                WHERE knowledge_base_id = :kb_id
                  AND index_status NOT IN ('pending', 'indexing')
                  AND indexing_config_snapshot IS NOT NULL
                  AND indexing_config_snapshot != CAST(:current AS jsonb)
            )
        """),
        {"kb_id": kb_id, "current": current_json},
    ).scalar()
    if not has_drift:
        return "none"

    strategy = (current_config or {}).get("strategy", "chunk_embed")
    if strategy != "graph_index":
        return "full"

    # graph_index strategy: enrichment_model + embedding_model are reversible
    # via the lighter-weight reenrichment flow. If those are the *only* fields
    # that differ, the UI should offer reenrich instead of full reindex.
    rest = {
        k: v
        for k, v in (current_config or {}).items()
        if k not in ("enrichment_model", "embedding_model")
    }
    rest_json = json.dumps(rest)

    has_full_drift = db.session.execute(
        text(f"""
            SELECT EXISTS (
                SELECT 1 FROM "{AI_SCHEMA}".indexed_sources
                WHERE knowledge_base_id = :kb_id
                  AND index_status NOT IN ('pending', 'indexing')
                  AND indexing_config_snapshot IS NOT NULL
                  AND (indexing_config_snapshot - 'enrichment_model' - 'embedding_model')
                      != CAST(:rest AS jsonb)
            )
        """),
        {"kb_id": kb_id, "rest": rest_json},
    ).scalar()

    return "full" if has_full_drift else "enrichment_only"


def _fetch_kb_or_404(kb_id: str) -> dict | tuple:
    """Fetch a knowledge base row by id and return it as a dict.

    Returns a (response, 404) tuple if not found — callers should check
    ``isinstance(result, tuple)`` and return early.
    """
    row = db.session.execute(
        text(f"""
            SELECT id, name, description, indexing_config, retrieval_config,
                   created_at, updated_at
            FROM "{AI_SCHEMA}".knowledge_bases
            WHERE id = :id
        """),
        {"id": kb_id},
    ).fetchone()
    if not row:
        return jsonify({"error": "Knowledge base not found"}), 404
    return {
        "id": str(row[0]),
        "name": row[1],
        "description": row[2],
        "indexing_config": row[3],
        "retrieval_config": row[4],
        "created_at": row[5].isoformat() if row[5] else None,
        "updated_at": row[6].isoformat() if row[6] else None,
    }


def _read_existing_retrieval_config(kb_id: str) -> dict:
    """Read the current retrieval_config from the DB; returns {} if KB missing."""
    row = db.session.execute(
        text(f'SELECT retrieval_config FROM "{AI_SCHEMA}".knowledge_bases WHERE id = :id'),
        {"id": kb_id},
    ).fetchone()
    if row is None or row[0] is None:
        return {}
    return row[0]


def _count_items_for_kb_bm25(kb_id: str, item_table: str) -> int:
    """COUNT(*) of items in the right table for this KB, used for stale detection."""
    if item_table == "chunks":
        sql = (
            f'SELECT COUNT(*) FROM "{AI_SCHEMA}".chunks c '
            f'JOIN "{AI_SCHEMA}".indexed_sources i ON i.id = c.indexed_source_id '
            f"WHERE i.knowledge_base_id = :kb"
        )
    elif item_table == "full_documents":
        sql = (
            f'SELECT COUNT(*) FROM "{AI_SCHEMA}".full_documents d '
            f'JOIN "{AI_SCHEMA}".indexed_sources i ON i.id = d.indexed_source_id '
            f"WHERE i.knowledge_base_id = :kb"
        )
    elif item_table == "graph_index_nodes":
        sql = (
            f'SELECT COUNT(*) FROM "{AI_SCHEMA}".graph_index_nodes n '
            f'JOIN "{AI_SCHEMA}".indexed_sources i ON i.id = n.indexed_source_id '
            f"WHERE i.knowledge_base_id = :kb"
        )
    else:
        return 0
    row = db.session.execute(text(sql), {"kb": kb_id}).fetchone()
    return int(row[0]) if row else 0


def _compute_bm25_status(kb) -> str | None:
    """Returns 'absent' | 'stale' | 'ready', or None when not applicable.

    Returns None (caller should omit the field) when:
      - the KB's retrieval method does not use BM25, OR
      - BM25_AUTO_INDEXING is on (platform manages it; user has nothing to act on).
    """
    if isinstance(kb, dict):
        retrieval_config = kb.get("retrieval_config") or {}
        indexing_config = kb.get("indexing_config") or {}
        kb_id = kb["id"]
    else:
        retrieval_config = kb.retrieval_config or {}
        indexing_config = kb.indexing_config or {}
        kb_id = kb.id

    method = retrieval_config.get("method")
    if method not in ("hybrid", "full_text"):
        return None
    if get_setting("BM25_AUTO_INDEXING"):
        return None

    strategy = indexing_config.get("strategy")
    item_table = _STRATEGY_TO_ITEM_TABLE.get(strategy)
    if item_table is None:
        return None

    store = SparseIndexStore(knowledge_base_id=kb_id)
    if not store.index_exists(item_table):
        return "absent"
    metadata = store.read_metadata(item_table)
    if metadata is None:
        return "stale"
    current = _count_items_for_kb_bm25(kb_id, item_table)
    return "ready" if metadata.get("item_count") == current else "stale"


@knowledge_bases_bp.route("", methods=["GET"])
@require_auth
def list_knowledge_bases():
    """List knowledge bases with pagination, search, sort, and aggregates."""
    from ..services.list_params import parse_list_params, escape_like, ListParamsError

    try:
        limit, offset, q, sort, order = parse_list_params(
            request,
            sort_allowed={"created_at", "name", "updated_at"},
        )
    except ListParamsError as e:
        return jsonify({"error": str(e)}), e.status

    where_clause = ""
    params: dict = {"limit": limit, "offset": offset}
    if q:
        where_clause = "WHERE kb.name ILIKE :q_like"
        params["q_like"] = f"%{escape_like(q)}%"

    # Build ORDER BY safely from the validated `sort` (allow-listed above).
    order_by = f"kb.{sort} {order.upper()}, kb.id ASC"

    # Total count (filtered)
    count_sql = f"""
        SELECT COUNT(*) FROM "{AI_SCHEMA}".knowledge_bases kb
        {where_clause}
    """
    total = db.session.execute(text(count_sql), params).scalar()

    rows_sql = f"""
        SELECT
          kb.id, kb.name, kb.description,
          kb.indexing_config, kb.retrieval_config,
          kb.created_at, kb.updated_at,
          COALESCE(SUM(CASE WHEN ix.index_status = 'pending'   THEN 1 ELSE 0 END), 0) AS pending,
          COALESCE(SUM(CASE WHEN ix.index_status = 'indexing'  THEN 1 ELSE 0 END), 0) AS indexing,
          COALESCE(SUM(CASE WHEN ix.index_status = 'indexed'   THEN 1 ELSE 0 END), 0) AS indexed,
          COALESCE(SUM(CASE WHEN ix.index_status = 'failed'    THEN 1 ELSE 0 END), 0) AS failed,
          COALESCE(SUM(CASE WHEN ix.index_status = 'cancelled' THEN 1 ELSE 0 END), 0) AS cancelled,
          COUNT(ix.id) AS total_sources,
          (SELECT COUNT(*) FROM "{AI_SCHEMA}".chunks WHERE knowledge_base_id = kb.id) AS chunk_count,
          ec.status AS enrichment_status,
          ec.enriched_count, ec.total_count
        FROM "{AI_SCHEMA}".knowledge_bases kb
        LEFT JOIN "{AI_SCHEMA}".indexed_sources ix ON ix.knowledge_base_id = kb.id
        LEFT JOIN "{AI_SCHEMA}".enrichment_configs ec ON ec.knowledge_base_id = kb.id
        {where_clause}
        GROUP BY kb.id, ec.status, ec.enriched_count, ec.total_count
        ORDER BY {order_by}
        LIMIT :limit OFFSET :offset
    """
    rows = db.session.execute(text(rows_sql), params)

    knowledge_bases = []
    for row in rows:
        enrichment_status = _ENRICHMENT_STATUS_MAP.get(row.enrichment_status, "none")
        if enrichment_status == "none":
            enrichment_progress = None
        else:
            enrichment_progress = {
                "enriched_count": row.enriched_count or 0,
                "total_count": row.total_count or 0,
            }
        knowledge_bases.append(
            {
                "id": str(row.id),
                "name": row.name,
                "description": row.description,
                "indexing_config": row.indexing_config,
                "retrieval_config": row.retrieval_config,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "source_counts": {
                    "pending": int(row.pending),
                    "indexing": int(row.indexing),
                    "indexed": int(row.indexed),
                    "failed": int(row.failed),
                    "cancelled": int(row.cancelled),
                    "total": int(row.total_sources),
                },
                "chunk_count": int(row.chunk_count),
                "enrichment_status": enrichment_status,
                "enrichment_progress": enrichment_progress,
            }
        )

    return jsonify(
        {
            "knowledge_bases": knowledge_bases,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@knowledge_bases_bp.route("", methods=["POST"])
@require_auth
def create_knowledge_base():
    """Create a new knowledge base."""
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Name is required"}), 400

    kb_id = str(uuid.uuid4())

    # Determine strategy and use registry defaults
    user_indexing_config = data.get("indexing_config", {})
    strategy_name = user_indexing_config.get("strategy", "chunk_embed")

    try:
        strategy_def = get_strategy(strategy_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    indexing_config = {**strategy_def["default_indexing_config"], **user_indexing_config}
    retrieval_config = data.get("retrieval_config", strategy_def["default_retrieval_config"])

    db.session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".knowledge_bases (
                id, name, description, indexing_config, retrieval_config
            ) VALUES (
                :id, :name, :description, CAST(:indexing_config AS jsonb), CAST(:retrieval_config AS jsonb)
            )
        """),
        {
            "id": kb_id,
            "name": data["name"],
            "description": data.get("description"),
            "indexing_config": json.dumps(indexing_config),
            "retrieval_config": json.dumps(retrieval_config),
        },
    )
    db.session.commit()

    return jsonify(
        {
            "id": kb_id,
            "name": data["name"],
            "description": data.get("description"),
            "indexing_config": indexing_config,
            "retrieval_config": retrieval_config,
        }
    ), 201


@knowledge_bases_bp.route("/<kb_id>", methods=["GET"])
@require_auth
def get_knowledge_base(kb_id: str):
    """Get a specific knowledge base.

    Returns metadata plus aggregated `source_counts` and a `drift` indicator.
    Per-source rows are served by the paginated /<kb_id>/sources route.
    """
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err

    kb = _fetch_kb_or_404(kb_id)
    if isinstance(kb, tuple):
        return kb

    # Aggregate counts in a single SQL — much smaller than the old bulk join.
    counts_rows = db.session.execute(
        text(f"""
            SELECT index_status, COUNT(*)
            FROM "{AI_SCHEMA}".indexed_sources
            WHERE knowledge_base_id = :kb_id
            GROUP BY index_status
        """),
        {"kb_id": kb_id},
    )
    source_counts = {
        "indexed": 0,
        "failed": 0,
        "pending": 0,
        "indexing": 0,
        "cancelled": 0,
        "total": 0,
    }
    for status, cnt in counts_rows:
        if status in source_counts:
            source_counts[status] = cnt
        source_counts["total"] += cnt

    drift = _compute_drift(kb_id, kb["indexing_config"] or {})

    response_body = {
        "id": kb["id"],
        "name": kb["name"],
        "description": kb["description"],
        "indexing_config": kb["indexing_config"],
        "retrieval_config": kb["retrieval_config"],
        "created_at": kb["created_at"],
        "updated_at": kb["updated_at"],
        "source_counts": source_counts,
        "drift": drift,
    }

    bm25_status = _compute_bm25_status(kb)
    if bm25_status is not None:
        response_body["bm25_status"] = bm25_status

    return jsonify(response_body)


@knowledge_bases_bp.route("/<kb_id>", methods=["PATCH"])
@require_auth
def update_knowledge_base(kb_id: str):
    """Update a knowledge base."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Capture old method BEFORE the UPDATE so we can detect transitions.
    old_method = None
    if "retrieval_config" in data:
        old_method = _read_existing_retrieval_config(kb_id).get("method")

    updates = []
    params = {"id": kb_id}

    if "name" in data:
        updates.append("name = :name")
        params["name"] = data["name"]
    if "description" in data:
        updates.append("description = :description")
        params["description"] = data["description"]
    if "indexing_config" in data:
        updates.append("indexing_config = CAST(:indexing_config AS jsonb)")
        params["indexing_config"] = json.dumps(data["indexing_config"])
    if "retrieval_config" in data:
        updates.append("retrieval_config = CAST(:retrieval_config AS jsonb)")
        params["retrieval_config"] = json.dumps(data["retrieval_config"])

    if not updates:
        return jsonify({"error": "No valid fields to update"}), 400

    updates.append("updated_at = NOW()")

    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".knowledge_bases
            SET {", ".join(updates)}
            WHERE id = :id
        """),
        params,
    )
    db.session.commit()

    if "retrieval_config" in data:
        new_method = (data.get("retrieval_config") or {}).get("method")
        transitioned_to_bm25 = old_method not in ("hybrid", "full_text") and new_method in (
            "hybrid",
            "full_text",
        )
        if transitioned_to_bm25 and get_setting("BM25_AUTO_INDEXING"):
            try:
                build_bm25_for_kb.delay(kb_id)
            except Exception:
                logger.warning(
                    "Failed to auto-dispatch build_bm25 for KB %s; "
                    "bm25_status will remain absent until manually triggered",
                    kb_id,
                    exc_info=True,
                )

    return get_knowledge_base(kb_id)


@knowledge_bases_bp.route("/<kb_id>/sources", methods=["GET"])
@require_auth
def list_indexed_sources(kb_id: str):
    """Paginated, filterable, sortable list of indexed_sources for a KB.

    Query params:
      q          - case-insensitive substring match on source name (optional)
      status     - exact match on index_status (optional)
      sort       - 'name' | 'created_at' (optional). Omitted = failed-first then created_at desc.
      order      - 'asc' | 'desc' (default 'desc'). Ignored when sort is omitted.
      limit      - int, default 50, capped at 200
      offset     - int, default 0
    """
    from ..services.list_params import escape_like

    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err

    kb_exists = db.session.execute(
        text(f"""SELECT 1 FROM "{AI_SCHEMA}".knowledge_bases WHERE id = :id"""),
        {"id": kb_id},
    ).fetchone()
    if not kb_exists:
        return jsonify({"error": "Knowledge base not found"}), 404

    q = (request.args.get("q") or "").strip()
    status = request.args.get("status")
    sort = request.args.get("sort")
    order = (request.args.get("order") or "desc").lower()
    if order not in ("asc", "desc"):
        order = "desc"

    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 200)
        offset = max(int(request.args.get("offset", 0)), 0)
    except ValueError:
        return jsonify({"error": "limit/offset must be integers"}), 400

    where = ["idx.knowledge_base_id = :kb_id"]
    params: dict = {"kb_id": kb_id, "limit": limit, "offset": offset}

    if status:
        where.append("idx.index_status = :status")
        params["status"] = status
    if q:
        where.append("s.name ILIKE :q")
        params["q"] = f"%{escape_like(q)}%"

    where_sql = " AND ".join(where)

    if sort == "name":
        order_sql = f"s.name {order.upper()}, s.id ASC"
    elif sort == "created_at":
        order_sql = f"s.created_at {order.upper()}, s.id ASC"
    else:
        # Default: failed-first, newest first. Tie-break on s.id for stable pagination.
        order_sql = "(idx.index_status = 'failed') DESC, s.created_at DESC, s.id ASC"

    total = (
        db.session.execute(
            text(f"""
            SELECT COUNT(*)
            FROM "{AI_SCHEMA}".indexed_sources idx
            JOIN "{AI_SCHEMA}".sources s ON idx.source_id = s.id
            WHERE {where_sql}
        """),
            params,
        ).scalar()
        or 0
    )

    rows = db.session.execute(
        text(f"""
            SELECT idx.id, idx.source_id, idx.index_status, idx.indexed_at,
                   idx.stats, idx.error_message,
                   s.name as source_name, s.file_type, s.created_at as source_created_at
            FROM "{AI_SCHEMA}".indexed_sources idx
            JOIN "{AI_SCHEMA}".sources s ON idx.source_id = s.id
            WHERE {where_sql}
            ORDER BY {order_sql}
            LIMIT :limit OFFSET :offset
        """),
        params,
    )

    items = []
    for r in rows:
        items.append(
            {
                "id": str(r[0]),
                "source_id": str(r[1]),
                "index_status": r[2],
                "indexed_at": r[3].isoformat() if r[3] else None,
                "stats": r[4] or {},
                "error_message": r[5],
                "source_name": r[6],
                "file_type": r[7],
                "source_created_at": r[8].isoformat() if r[8] else None,
            }
        )

    return jsonify(
        {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@knowledge_bases_bp.route("/<kb_id>", methods=["DELETE"])
@require_auth
def delete_knowledge_base(kb_id: str):
    """Delete a knowledge base and all its chunks."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    # Drop dynamic metadata table if enrichment was configured
    from ..services.metadata_enricher import MetadataEnricher

    try:
        config_row = db.session.execute(
            text(
                f'SELECT metadata_table_name FROM "{AI_SCHEMA}".enrichment_configs '
                f"WHERE knowledge_base_id = :kb_id"
            ),
            {"kb_id": kb_id},
        ).fetchone()
        if config_row and config_row[0]:
            MetadataEnricher(db.session, kb_id).drop_metadata_table(config_row[0])
    except Exception:
        logger.warning(
            "Failed to clean up enrichment during KB deletion",
            exc_info=True,
        )

    # Check for agent dependencies before deleting
    agent_deps_result = db.session.execute(
        text(f"""
            SELECT name FROM "{AI_SCHEMA}".agents
            WHERE settings::text LIKE :kb_pattern
        """),
        {"kb_pattern": f"%{kb_id}%"},
    )
    agent_dep_names = [row[0] for row in agent_deps_result]

    db.session.execute(
        text(f'DELETE FROM "{AI_SCHEMA}".knowledge_bases WHERE id = :id'),
        {"id": kb_id},
    )
    db.session.commit()

    response = {"message": "Knowledge base deleted"}
    if agent_dep_names:
        response["warning"] = (
            f"This KB was referenced by {len(agent_dep_names)} agent(s): {', '.join(agent_dep_names)}. Those agents may fail to retrieve context."
        )
    return jsonify(response)


def index_source_into_kb(kb_id: str, source_id: str) -> dict:
    """Validate, insert, and dispatch indexing for a single source.

    Returns a dict with the result.  On validation errors the dict contains
    an ``error`` key.  Callers are responsible for translating to HTTP or
    tool-result format as appropriate.

    This is the **single code-path** for adding a source to a KB — used by
    both the REST route and the copilot tool handler.
    """
    # Check if source exists and is extracted
    src_row = db.session.execute(
        text(f"""
            SELECT id, name, extraction_status FROM "{AI_SCHEMA}".sources WHERE id = :id
        """),
        {"id": source_id},
    ).fetchone()
    if not src_row:
        return {"error": "Source not found", "status_code": 404}
    if src_row[2] != "extracted":
        return {
            "error": f"Source must be extracted first (status: {src_row[2]})",
            "status_code": 400,
        }

    # Get KB indexing config
    kb_row = db.session.execute(
        text(f"""
            SELECT indexing_config FROM "{AI_SCHEMA}".knowledge_bases WHERE id = :id
        """),
        {"id": kb_id},
    ).fetchone()
    if not kb_row:
        return {"error": "Knowledge base not found", "status_code": 404}

    indexing_config = kb_row[0] or {}

    # Create or update indexed_source record
    new_id = str(uuid.uuid4())
    result = db.session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".indexed_sources (
                id, knowledge_base_id, source_id, index_status, indexing_config_snapshot
            ) VALUES (
                :id, :kb_id, :source_id, 'pending', CAST(:config AS jsonb)
            )
            ON CONFLICT (knowledge_base_id, source_id)
            DO UPDATE SET
                index_status = 'pending',
                indexing_config_snapshot = CAST(:config AS jsonb),
                error_message = NULL,
                last_dispatched_at = NOW()
            RETURNING id
        """),
        {
            "id": new_id,
            "kb_id": kb_id,
            "source_id": source_id,
            "config": json.dumps(indexing_config),
        },
    )
    indexed_source_id = str(result.fetchone()[0])
    db.session.commit()

    # Pre-charge balance check. Maps strategy -> action just to derive the
    # right idempotency key — pre-charge uses the catalog max so the check
    # is strategy-agnostic and rejects on insufficient balance regardless
    # of which indexing path will run.
    strategy = (indexing_config or {}).get("strategy", "chunk_embed")
    from ..tasks.indexing import _resolve_indexing_action  # local: avoid circular at import

    indexing_action = _resolve_indexing_action(strategy)
    if strategy == "graph_index":
        _graph_check_balance(kb_id, indexed_source_id=indexed_source_id)
    else:
        billing.check_balance(
            estimated_cost=_INDEXING_ESTIMATED_QUANTITY * _INDEXING_MAX_UNIT_CREDITS
        )
    billing_kwargs = _maybe_indexing_billing_kwargs(
        action=indexing_action, ref_id=indexed_source_id
    )

    # Trigger indexing
    try:
        task = index_source.delay(
            kb_id,
            source_id,
            indexed_source_id,
            provider_keys=get_all_user_provider_keys(),
            **billing_kwargs,
        )
    except Exception:
        logger.error(
            "Failed to dispatch indexing task for source %s into KB %s",
            source_id,
            kb_id,
            exc_info=True,
        )
        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".indexed_sources
                SET index_status = 'failed',
                    error_message = 'Failed to dispatch indexing task (worker may be unavailable). Try reindexing.'
                WHERE id = :id
            """),
            {"id": indexed_source_id},
        )
        db.session.commit()
        return {"error": "Failed to dispatch indexing task", "status_code": 503}

    return {
        "id": indexed_source_id,
        "knowledge_base_id": kb_id,
        "source_id": source_id,
        "source_name": src_row[1],
        "index_status": "pending",
        "task_id": task.id,
    }


@knowledge_bases_bp.route("/<kb_id>/sources", methods=["POST"])
@require_auth
def add_source_to_kb(kb_id: str):
    """Add a source to a knowledge base for indexing."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    data = request.get_json()
    if not data or not data.get("source_id"):
        return jsonify({"error": "source_id is required"}), 400

    result = index_source_into_kb(kb_id, data["source_id"])

    if "error" in result:
        return jsonify({"error": result["error"]}), result.get("status_code", 400)

    return jsonify(result), 201


@knowledge_bases_bp.route("/<kb_id>/sources/<indexed_source_id>/cancel", methods=["POST"])
@require_auth
def cancel_indexing(kb_id: str, indexed_source_id: str):
    """Cancel an in-progress indexing task."""
    err = _require_uuid(kb_id, "knowledge base id") or _require_uuid(
        indexed_source_id, "indexed source id"
    )
    if err:
        return err
    result = db.session.execute(
        text(f"""
            SELECT index_status, celery_task_id
            FROM "{AI_SCHEMA}".indexed_sources
            WHERE id = :id AND knowledge_base_id = :kb_id
        """),
        {"id": indexed_source_id, "kb_id": kb_id},
    )
    row = result.fetchone()
    if not row:
        return jsonify({"error": "Indexed source not found"}), 404

    status, celery_task_id = row
    if status not in ("pending", "indexing"):
        return jsonify({"error": f"Cannot cancel indexing with status '{status}'"}), 409

    if celery_task_id:
        try:
            celery_app.control.revoke(celery_task_id, terminate=True)
        except Exception as exc:
            logger.warning("Failed to revoke Celery task %s: %s", celery_task_id, exc)

    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".indexed_sources
            SET index_status = 'cancelled',
                error_message = 'Cancelled by user'
            WHERE id = :id
        """),
        {"id": indexed_source_id},
    )
    db.session.commit()

    return jsonify({"message": "Indexing cancelled"})


@knowledge_bases_bp.route("/<kb_id>/sources/<indexed_source_id>", methods=["DELETE"])
@require_auth
def remove_source_from_kb(kb_id: str, indexed_source_id: str):
    """Remove a source from a knowledge base.

    Deletes the indexed_sources row, which cascades through the seven child
    tables defined in ai_schema.sql (chunks, page_index_nodes, full_documents,
    doc2json_documents, graph_index_nodes, embeddings, page_index_toc). The
    underlying ai.sources row is NOT touched — the source remains available
    to add to other KBs.

    If the indexed_source is mid-flight (status in {pending, indexing}) and
    has a celery_task_id, revokes the task before deleting. Failure to
    revoke logs a warning but does not block the delete (mirrors the
    cancel_indexing route's posture).
    """
    err = _require_uuid(kb_id, "knowledge base id", status_code=400) or _require_uuid(
        indexed_source_id, "indexed source id", status_code=400
    )
    if err:
        return err

    row = db.session.execute(
        text(f"""
            SELECT index_status, celery_task_id
            FROM "{AI_SCHEMA}".indexed_sources
            WHERE id = :id AND knowledge_base_id = :kb_id
        """),
        {"id": indexed_source_id, "kb_id": kb_id},
    ).fetchone()
    if not row:
        return jsonify({"error": "Indexed source not found"}), 404

    status, celery_task_id = row
    if status in ("pending", "indexing") and celery_task_id:
        try:
            celery_app.control.revoke(celery_task_id, terminate=True)
        except Exception as exc:
            logger.warning("Failed to revoke Celery task %s: %s", celery_task_id, exc)

    db.session.execute(
        text(f'DELETE FROM "{AI_SCHEMA}".indexed_sources WHERE id = :id'),
        {"id": indexed_source_id},
    )
    db.session.commit()

    return jsonify(
        {
            "message": "Source removed from knowledge base",
            "deleted_indexed_source_id": indexed_source_id,
            "kb_id": kb_id,
        }
    )


@knowledge_bases_bp.route("/<kb_id>/reindex", methods=["POST"])
@require_auth
def reindex_kb(kb_id: str):
    """Re-index sources in a knowledge base.

    Body (optional):
      - indexed_source_ids: list[str] — only reindex these rows
      - failed_only: bool — reindex every row currently in 'failed' status
    No body / empty body = reindex ALL rows in the KB (legacy behavior).
    """
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err

    data = request.get_json(silent=True) or {}
    indexed_source_ids = data.get("indexed_source_ids") or []
    failed_only = bool(data.get("failed_only"))

    # Validate any explicit IDs first so we don't kick off a partial job.
    for isid in indexed_source_ids:
        if isinstance(isid, str):
            uid_err = _require_uuid(isid, "indexed_source_id")
            if uid_err:
                return uid_err

    if indexed_source_ids:
        # Selective: reset only the requested rows.
        rows = db.session.execute(
            text(f"""
                SELECT id, source_id FROM "{AI_SCHEMA}".indexed_sources
                WHERE knowledge_base_id = :kb_id
                  AND id = ANY(:ids)
            """),
            {"kb_id": kb_id, "ids": indexed_source_ids},
        ).fetchall()
        target_ids = [str(r[0]) for r in rows]
        if not target_ids:
            return jsonify({"error": "No matching indexed_sources for the supplied ids"}), 404

        # Pre-charge balance check covering N sources before resetting state.
        billing.check_balance(
            estimated_cost=len(rows) * _INDEXING_ESTIMATED_QUANTITY * _INDEXING_MAX_UNIT_CREDITS
        )

        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".indexed_sources
                SET index_status = 'pending',
                    error_message = NULL,
                    last_dispatched_at = NOW()
                WHERE id = ANY(:ids)
            """),
            {"ids": target_ids},
        )
        db.session.commit()

        provider_keys = get_all_user_provider_keys()
        task_ids: list[str] = []
        for row in rows:
            t = index_source.delay(
                kb_id,
                str(row[1]),
                indexed_source_id=str(row[0]),
                provider_keys=provider_keys,
                **_maybe_indexing_billing_kwargs(action="indexing", ref_id=str(row[0])),
            )
            task_ids.append(t.id)
        return jsonify(
            {
                "status": "started",
                "task_ids": task_ids,
                "knowledge_base_id": kb_id,
                "count": len(task_ids),
                "scope": "selected",
            }
        )

    if failed_only:
        # Reset only the failed rows, then enqueue per-source.
        rows = db.session.execute(
            text(f"""
                SELECT id, source_id FROM "{AI_SCHEMA}".indexed_sources
                WHERE knowledge_base_id = :kb_id
                  AND index_status = 'failed'
            """),
            {"kb_id": kb_id},
        ).fetchall()
        target_ids = [str(r[0]) for r in rows]
        if not target_ids:
            return jsonify(
                {
                    "status": "noop",
                    "knowledge_base_id": kb_id,
                    "count": 0,
                    "scope": "failed_only",
                    "message": "No failed indexed_sources to retry.",
                }
            )

        # Pre-charge balance check covering N sources.
        billing.check_balance(
            estimated_cost=len(rows) * _INDEXING_ESTIMATED_QUANTITY * _INDEXING_MAX_UNIT_CREDITS
        )

        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".indexed_sources
                SET index_status = 'pending',
                    error_message = NULL,
                    last_dispatched_at = NOW()
                WHERE id = ANY(:ids)
            """),
            {"ids": target_ids},
        )
        db.session.commit()

        provider_keys = get_all_user_provider_keys()
        task_ids = []
        for row in rows:
            t = index_source.delay(
                kb_id,
                str(row[1]),
                indexed_source_id=str(row[0]),
                provider_keys=provider_keys,
                **_maybe_indexing_billing_kwargs(action="indexing", ref_id=str(row[0])),
            )
            task_ids.append(t.id)
        return jsonify(
            {
                "status": "started",
                "task_ids": task_ids,
                "knowledge_base_id": kb_id,
                "count": len(task_ids),
                "scope": "failed_only",
            }
        )

    # Default: reindex everything (legacy behavior).
    # Pre-charge balance check covers ALL sources in the KB; the per-source
    # the charge happens inside each index_source child task.
    source_count = (
        db.session.execute(
            text(
                f'SELECT COUNT(*) FROM "{AI_SCHEMA}".indexed_sources '
                f"WHERE knowledge_base_id = :kb_id"
            ),
            {"kb_id": kb_id},
        ).scalar()
        or 0
    )
    if source_count > 0:
        billing.check_balance(
            estimated_cost=source_count * _INDEXING_ESTIMATED_QUANTITY * _INDEXING_MAX_UNIT_CREDITS
        )

    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".indexed_sources
            SET index_status = 'pending',
                error_message = NULL,
                last_dispatched_at = NOW()
            WHERE knowledge_base_id = :kb_id
        """),
        {"kb_id": kb_id},
    )
    db.session.commit()

    # reindex_knowledge_base itself doesn't charge — it fans out to per-source
    # index_source tasks which each charge, threading their own key inputs
    # (literal "indexing" + idx_source_id/source_id). Identity comes from the
    # billing adapter, so no billing args are forwarded here.
    task = reindex_knowledge_base.delay(
        kb_id,
        provider_keys=get_all_user_provider_keys(),
    )

    return jsonify(
        {
            "status": "started",
            "task_id": task.id,
            "knowledge_base_id": kb_id,
            "scope": "all",
        }
    )


@knowledge_bases_bp.route("/<kb_id>/graph-enrichment/run", methods=["POST"])
@require_auth
def run_graph_reenrichment(kb_id: str):
    """Re-run graph reference enrichment (Stages 2+3) without full reindex."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    # Validate KB exists and uses graph_index strategy
    result = db.session.execute(
        text(f"""
            SELECT indexing_config FROM "{AI_SCHEMA}".knowledge_bases WHERE id = :id
        """),
        {"id": kb_id},
    )
    row = result.fetchone()
    if not row:
        return jsonify({"error": "Knowledge base not found"}), 404

    indexing_config = row[0] or {}
    strategy = indexing_config.get("strategy", "chunk_embed")
    if strategy != "graph_index":
        return jsonify(
            {
                "error": f"Graph re-enrichment is only supported for graph_index strategy, got '{strategy}'"
            }
        ), 400

    data = request.get_json(silent=True) or {}
    retry_failed = bool(data.get("retry_failed", False))
    indexed_source_id = data.get("indexed_source_id")

    # Pre-charge balance check. Scale with KB node count (fail-closed on lookup error).
    _graph_check_balance(kb_id, indexed_source_id=indexed_source_id)
    # reenrich has no main charge — its per-batch graph charging is enabled by
    # default in the billing port (ctx-gated by the adapter), so no billing key
    # inputs are threaded here.

    # Mark affected sources as "indexing" so frontend polling kicks in
    if indexed_source_id:
        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".indexed_sources
                SET index_status = 'indexing', error_message = NULL
                WHERE id = :id AND knowledge_base_id = :kb_id
            """),
            {"id": indexed_source_id, "kb_id": kb_id},
        )
    else:
        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".indexed_sources
                SET index_status = 'indexing', error_message = NULL
                WHERE knowledge_base_id = :kb_id
            """),
            {"kb_id": kb_id},
        )
    db.session.commit()

    try:
        task = reenrich_graph_references.delay(
            kb_id,
            retry_failed=retry_failed,
            indexed_source_id=indexed_source_id,
        )
    except Exception:
        # Restore status if task dispatch fails (e.g. Redis down)
        if indexed_source_id:
            db.session.execute(
                text(
                    f"UPDATE \"{AI_SCHEMA}\".indexed_sources SET index_status = 'indexed' WHERE id = :id"
                ),
                {"id": indexed_source_id},
            )
        else:
            db.session.execute(
                text(
                    f"UPDATE \"{AI_SCHEMA}\".indexed_sources SET index_status = 'indexed' WHERE knowledge_base_id = :kb_id"
                ),
                {"kb_id": kb_id},
            )
        db.session.commit()
        logger.exception("Failed to dispatch re-enrichment task for KB %s", kb_id)
        return jsonify({"error": "Failed to start re-enrichment task"}), 500

    return jsonify(
        {
            "status": "started",
            "task_id": task.id,
            "knowledge_base_id": kb_id,
            "retry_failed": retry_failed,
        }
    )


@knowledge_bases_bp.route("/<kb_id>/graph-enrichment/errors", methods=["GET"])
@require_auth
def get_graph_enrichment_errors(kb_id: str):
    """Get per-source enrichment error counts for a graph_index KB."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    result = db.session.execute(
        text(f"""
            SELECT indexing_config FROM "{AI_SCHEMA}".knowledge_bases WHERE id = :id
        """),
        {"id": kb_id},
    )
    row = result.fetchone()
    if not row:
        return jsonify({"error": "Knowledge base not found"}), 404

    indexing_config = row[0] or {}
    if indexing_config.get("strategy", "chunk_embed") != "graph_index":
        return jsonify({"error": "Enrichment errors only apply to graph_index strategy"}), 400

    from ..services.graph_index_store import GraphIndexStore

    store = GraphIndexStore(db_session=db.session, knowledge_base_id=kb_id)
    counts = store.get_enrichment_error_counts()
    return jsonify(counts)


@knowledge_bases_bp.route("/<kb_id>/items", methods=["POST"])
@require_auth
def get_items_by_sources(kb_id: str):
    """Fetch all indexed content items for specific source document(s)."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    data = request.get_json()
    if not data or not data.get("source_ids"):
        return jsonify({"error": "source_ids is required and must be a non-empty list"}), 400

    source_ids = data["source_ids"]
    if not isinstance(source_ids, list) or not all(isinstance(s, str) for s in source_ids):
        return jsonify({"error": "source_ids must be a list of UUID strings"}), 400

    try:
        limit = max(0, min(int(data.get("limit", 1000)), 10000))
        offset = max(0, int(data.get("offset", 0)))
    except (TypeError, ValueError):
        return jsonify({"error": "limit and offset must be integers"}), 400

    # Look up KB and determine strategy
    kb_row = db.session.execute(
        text(f'SELECT indexing_config FROM "{AI_SCHEMA}".knowledge_bases WHERE id = :id'),
        {"id": kb_id},
    ).fetchone()
    if not kb_row:
        return jsonify({"error": "Knowledge base not found"}), 404

    strategy = (kb_row[0] or {}).get("strategy", "chunk_embed")

    # Strategy-specific table, extra columns, and ordering
    strategy_configs = {
        "chunk_embed": {
            "table": "chunks",
            "text_col": "text",
            "extras": ["chunk_index", "start_char", "end_char", "tokens"],
            "order": "source_id, chunk_index",
        },
        "page_index": {
            "table": "page_index_nodes",
            "text_col": "text",
            "extras": ["node_id", "title", "depth", "parent_node_id"],
            "order": "source_id, depth, node_id",
        },
        "graph_index": {
            "table": "graph_index_nodes",
            "text_col": "text",
            "extras": ["node_id", "title", "depth", "parent_node_id"],
            "order": "source_id, depth, node_id",
        },
        "full_document": {
            "table": "full_documents",
            "text_col": "summary",
            "extras": ["full_text_path"],
            "order": "source_id, created_at",
        },
        "doc2json": {
            "table": "doc2json_documents",
            "text_col": "summary",
            "extras": ["extracted_json"],
            "order": "source_id, created_at",
        },
    }

    config = strategy_configs.get(strategy)
    if not config:
        return jsonify({"error": f"Unsupported strategy: {strategy}"}), 400

    table = config["table"]
    text_col = config["text_col"]
    extras = config["extras"]
    extra_cols = ", ".join(extras)

    source_ids_param = "{" + ",".join(source_ids) + "}"
    where = "knowledge_base_id = :kb_id AND source_id = ANY(CAST(:source_ids AS uuid[]))"
    params = {"kb_id": kb_id, "source_ids": source_ids_param, "limit": limit, "offset": offset}

    try:
        # Count query
        count_row = db.session.execute(
            text(f'SELECT COUNT(*) FROM "{AI_SCHEMA}".{table} WHERE {where}'),
            params,
        ).fetchone()
        total = count_row[0]

        # Main query
        rows = db.session.execute(
            text(
                f"SELECT id, source_id, {text_col}, meta, {extra_cols} "
                f'FROM "{AI_SCHEMA}".{table} '
                f"WHERE {where} "
                f"ORDER BY {config['order']} "
                f"LIMIT :limit OFFSET :offset"
            ),
            params,
        ).fetchall()
    except Exception as e:
        logger.error("Failed to fetch items for KB %s: %s", kb_id, e)
        return jsonify({"error": str(e)}), 500

    items = []
    for row in rows:
        item = {
            "id": str(row[0]),
            "source_id": str(row[1]),
            "text": row[2],
            "meta": row[3],
        }
        for i, col_name in enumerate(extras):
            val = row[4 + i]
            if col_name == "extracted_json":
                item[col_name] = val
            elif isinstance(val, uuid.UUID):
                item[col_name] = str(val)
            else:
                item[col_name] = val
        items.append(item)

    return jsonify(
        {
            "items": items,
            "total": total,
            "strategy": strategy,
            "source_ids": source_ids,
        }
    )


@knowledge_bases_bp.route("/<kb_id>/search", methods=["POST"])
@require_auth
def search_knowledge_base_route(kb_id: str):
    """Search the knowledge base."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    from ..services.knowledge_search import search_knowledge_base as do_search

    data = request.get_json()
    if not data or not data.get("query"):
        return jsonify({"error": "query is required"}), 400

    query = data["query"]
    top_k = data.get("top_k", 5)
    method = data.get("retrieval_method")
    similarity_threshold = data.get("similarity_threshold", 0.0)
    filter_metadata = data.get("filter_metadata")
    source_ids = data.get("source_ids")
    if source_ids is not None:
        if not isinstance(source_ids, list):
            return jsonify({"error": "source_ids must be a list of UUID strings"}), 400

    try:
        results = do_search(
            db_session=db.session,
            knowledge_base_id=kb_id,
            query=query,
            top_k=top_k,
            retrieval_method=method,
            similarity_threshold=similarity_threshold,
            filter_metadata=filter_metadata,
            source_ids=source_ids,
        )

        return jsonify(
            {
                "results": [
                    {
                        "chunk_id": r.item_id,
                        "text": r.text,
                        "score": r.score,
                        "source_id": r.source_id,
                        "meta": r.meta,
                    }
                    for r in results
                ],
                "query": query,
                "retrieval_method": method or "auto",
                "total_results": len(results),
            }
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({"error": str(e)}), 500


@knowledge_bases_bp.route("/<kb_id>/build-bm25", methods=["POST"])
@require_auth
def build_bm25_endpoint(kb_id: str):
    """Dispatch a one-shot BM25 rebuild for this KB.

    Manual operator path: re-tokenizes the entire item table for this
    KB's strategy (chunks/full_documents/graph_index_nodes) and writes
    a fresh BM25 index, replacing whatever was there.

    Returns 202 + the Celery task id. Caller can poll ``bm25_status`` on
    the KB to observe completion.
    """
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err

    kb = _fetch_kb_or_404(kb_id)
    if isinstance(kb, tuple):  # _fetch_kb_or_404 returns a response tuple on 404
        return kb

    method = (kb.get("retrieval_config") or {}).get("method")
    if method not in ("hybrid", "full_text"):
        return jsonify(
            {
                "error": (
                    f"KB retrieval method '{method}' does not use BM25; "
                    "this endpoint is only valid for hybrid or full_text KBs."
                )
            }
        ), 400

    try:
        t = build_bm25_for_kb.delay(kb_id)
    except Exception:
        logger.exception("Failed to dispatch build-bm25 task for KB %s", kb_id)
        return jsonify({"error": "Failed to start BM25 build task"}), 503
    return jsonify({"task_id": t.id, "knowledge_base_id": kb_id}), 202


# =============================================================================
# KB Inspector — content-viewer endpoints (C2.1)
#
# The Studio KB detail page's "inspect indexed source" modal used to read
# these tables directly via useProjectSupabaseClient().client (a PostgREST
# proxy onto the `ai` schema). C2.2 removes `ai` from PGRST_DB_SCHEMAS, so
# every such read moves here first. Response field names match the raw
# columns the old `.select('*')` / explicit-column calls actually returned
# (verified against what the page renders, not the possibly-stale
# ChunksListResponse/PageIndexNodeItem/etc. TS interfaces in ai-api.ts —
# some of those declare fields like `summary`/`content`/`level`/`parent_id`
# that no `.tsx` in the page actually reads).
# =============================================================================


def _require_indexed_source_in_kb(kb_id: str, indexed_source_id: str):
    """Error response tuple if `indexed_source_id` doesn't belong to `kb_id`; else None."""
    err = _require_uuid(indexed_source_id, "indexed source id")
    if err:
        return err
    row = db.session.execute(
        text(f"""
            SELECT 1 FROM "{AI_SCHEMA}".indexed_sources
            WHERE id = :id AND knowledge_base_id = :kb_id
        """),
        {"id": indexed_source_id, "kb_id": kb_id},
    ).fetchone()
    if not row:
        return jsonify({"error": "Indexed source not found in this knowledge base"}), 404
    return None


@knowledge_bases_bp.route("/<kb_id>/available-sources", methods=["GET"])
@require_auth
def list_available_sources(kb_id: str):
    """Extracted sources NOT yet indexed into this KB — the "Add source" modal.

    Wraps ai.list_sources_excluding_kb(...) (migration 0023), which pushes
    the NOT EXISTS dedup + search filter into Postgres instead of shipping
    the whole ai.sources table to the browser for client-side filtering.
    """
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err

    q = (request.args.get("q") or "").strip() or None
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 200)
        offset = max(int(request.args.get("offset", 0)), 0)
    except ValueError:
        return jsonify({"error": "limit/offset must be integers"}), 400

    rows = db.session.execute(
        text(f"""
            SELECT id, name, file_type, storage_path, extraction_status,
                   derivatives, metadata, total_count
            FROM "{AI_SCHEMA}".list_sources_excluding_kb(:kb_id, :q, :limit, :offset)
        """),
        {"kb_id": kb_id, "q": q, "limit": limit, "offset": offset},
    ).fetchall()

    sources = [
        {
            "id": str(r.id),
            "name": r.name,
            "file_type": r.file_type,
            "storage_path": r.storage_path,
            "extraction_status": r.extraction_status,
            "derivatives": r.derivatives,
            "metadata": r.metadata,
        }
        for r in rows
    ]
    # COUNT(*) OVER () only appears on rows that exist — zero matches means
    # zero rows back, not a single row with total_count=0. See migration 0023.
    total = rows[0].total_count if rows else 0
    return jsonify({"sources": sources, "total": total, "limit": limit, "offset": offset})


@knowledge_bases_bp.route("/<kb_id>/indexed-sources/<indexed_source_id>/chunks", methods=["GET"])
@require_auth
def list_chunks_for_indexed_source(kb_id: str, indexed_source_id: str):
    """Paginated chunk_embed chunks for one indexed source (inspector modal)."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    err = _require_indexed_source_in_kb(kb_id, indexed_source_id)
    if err:
        return err

    try:
        limit = min(max(int(request.args.get("limit", 10)), 1), 200)
        offset = max(int(request.args.get("offset", 0)), 0)
    except ValueError:
        return jsonify({"error": "limit/offset must be integers"}), 400

    params = {"isid": indexed_source_id, "kb_id": kb_id}
    total = (
        db.session.execute(
            text(f"""
                SELECT COUNT(*) FROM "{AI_SCHEMA}".chunks
                WHERE indexed_source_id = :isid AND knowledge_base_id = :kb_id
            """),
            params,
        ).scalar()
        or 0
    )

    rows = db.session.execute(
        text(f"""
            SELECT id, indexed_source_id, text, chunk_index, start_char, end_char,
                   tokens, meta
            FROM "{AI_SCHEMA}".chunks
            WHERE indexed_source_id = :isid AND knowledge_base_id = :kb_id
            ORDER BY chunk_index ASC NULLS LAST, id ASC
            LIMIT :limit OFFSET :offset
        """),
        {**params, "limit": limit, "offset": offset},
    ).fetchall()

    chunks = [
        {
            "id": str(r.id),
            "indexed_source_id": str(r.indexed_source_id) if r.indexed_source_id else None,
            "text": r.text,
            "chunk_index": r.chunk_index,
            "start_char": r.start_char,
            "end_char": r.end_char,
            "tokens": r.tokens,
            "meta": r.meta,
        }
        for r in rows
    ]
    return jsonify({"chunks": chunks, "total": total, "limit": limit, "offset": offset})


def _fetch_index_nodes(table: str, indexed_source_id: str) -> list[dict]:
    """Shared row-fetch for page_index_nodes / graph_index_nodes.

    `table` is always one of those two Python-literal constants passed by
    the route handlers below — never user input — so the f-string
    interpolation is not an injection surface.
    """
    rows = db.session.execute(
        text(f"""
            SELECT id, toc_id, indexed_source_id, knowledge_base_id, source_id,
                   node_id, title, depth, parent_node_id, text, line_num, meta,
                   created_at
            FROM "{AI_SCHEMA}".{table}
            WHERE indexed_source_id = :isid
            ORDER BY id ASC
        """),
        {"isid": indexed_source_id},
    ).fetchall()
    return [
        {
            "id": str(r.id),
            "toc_id": str(r.toc_id),
            "indexed_source_id": str(r.indexed_source_id) if r.indexed_source_id else None,
            "knowledge_base_id": str(r.knowledge_base_id),
            "source_id": str(r.source_id),
            "node_id": r.node_id,
            "title": r.title,
            "depth": r.depth,
            "parent_node_id": r.parent_node_id,
            "text": r.text,
            "line_num": r.line_num,
            "meta": r.meta,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


def _fetch_index_toc(table: str, indexed_source_id: str, kb_id: str) -> dict | None:
    """Shared row-fetch for page_index_toc / graph_index_toc. See _fetch_index_nodes
    docstring re: `table` not being an injection surface."""
    row = db.session.execute(
        text(f"""
            SELECT structure, doc_name, doc_description
            FROM "{AI_SCHEMA}".{table}
            WHERE indexed_source_id = :isid AND knowledge_base_id = :kb_id
        """),
        {"isid": indexed_source_id, "kb_id": kb_id},
    ).fetchone()
    if not row:
        return None
    return {
        "structure": row.structure,
        "doc_name": row.doc_name,
        "doc_description": row.doc_description,
    }


@knowledge_bases_bp.route(
    "/<kb_id>/indexed-sources/<indexed_source_id>/page-index-nodes", methods=["GET"]
)
@require_auth
def list_page_index_nodes(kb_id: str, indexed_source_id: str):
    """All page_index nodes for one indexed source (inspector modal)."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    err = _require_indexed_source_in_kb(kb_id, indexed_source_id)
    if err:
        return err
    return jsonify({"nodes": _fetch_index_nodes("page_index_nodes", indexed_source_id)})


@knowledge_bases_bp.route(
    "/<kb_id>/indexed-sources/<indexed_source_id>/page-index-toc", methods=["GET"]
)
@require_auth
def get_page_index_toc(kb_id: str, indexed_source_id: str):
    """The page_index table-of-contents row for one indexed source, or null."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    err = _require_indexed_source_in_kb(kb_id, indexed_source_id)
    if err:
        return err
    return jsonify({"toc": _fetch_index_toc("page_index_toc", indexed_source_id, kb_id)})


@knowledge_bases_bp.route(
    "/<kb_id>/indexed-sources/<indexed_source_id>/graph-index-nodes", methods=["GET"]
)
@require_auth
def list_graph_index_nodes(kb_id: str, indexed_source_id: str):
    """All graph_index nodes for one indexed source (inspector modal)."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    err = _require_indexed_source_in_kb(kb_id, indexed_source_id)
    if err:
        return err
    return jsonify({"nodes": _fetch_index_nodes("graph_index_nodes", indexed_source_id)})


@knowledge_bases_bp.route(
    "/<kb_id>/indexed-sources/<indexed_source_id>/graph-index-toc", methods=["GET"]
)
@require_auth
def get_graph_index_toc(kb_id: str, indexed_source_id: str):
    """The graph_index table-of-contents row for one indexed source, or null."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    err = _require_indexed_source_in_kb(kb_id, indexed_source_id)
    if err:
        return err
    return jsonify({"toc": _fetch_index_toc("graph_index_toc", indexed_source_id, kb_id)})


@knowledge_bases_bp.route(
    "/<kb_id>/indexed-sources/<indexed_source_id>/full-document", methods=["GET"]
)
@require_auth
def get_full_document(kb_id: str, indexed_source_id: str):
    """The full_document summary row for one indexed source, or null."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    err = _require_indexed_source_in_kb(kb_id, indexed_source_id)
    if err:
        return err

    row = db.session.execute(
        text(f"""
            SELECT id, source_id, summary, summary_model, summary_tokens,
                   full_text_tokens, meta
            FROM "{AI_SCHEMA}".full_documents
            WHERE indexed_source_id = :isid AND knowledge_base_id = :kb_id
        """),
        {"isid": indexed_source_id, "kb_id": kb_id},
    ).fetchone()
    if not row:
        return jsonify({"document": None})
    return jsonify(
        {
            "document": {
                "id": str(row.id),
                "source_id": str(row.source_id),
                "summary": row.summary,
                "summary_model": row.summary_model,
                "summary_tokens": row.summary_tokens,
                "full_text_tokens": row.full_text_tokens,
                "meta": row.meta,
            }
        }
    )


@knowledge_bases_bp.route(
    "/<kb_id>/indexed-sources/<indexed_source_id>/doc2json-document", methods=["GET"]
)
@require_auth
def get_doc2json_document(kb_id: str, indexed_source_id: str):
    """The doc2json extraction row for one indexed source (or null), plus the
    linked source's `derivatives` (the modal needs image/text derivative
    counts to decide image-mode vs text-mode rendering — previously a
    second PostgREST round trip from the browser)."""
    err = _require_uuid(kb_id, "knowledge base id")
    if err:
        return err
    err = _require_indexed_source_in_kb(kb_id, indexed_source_id)
    if err:
        return err

    row = db.session.execute(
        text(f"""
            SELECT id, source_id, summary, extracted_json, json_schema,
                   window_summaries, extraction_model, summary_tokens,
                   input_tokens, window_size, window_overlap, window_count, meta
            FROM "{AI_SCHEMA}".doc2json_documents
            WHERE indexed_source_id = :isid
        """),
        {"isid": indexed_source_id},
    ).fetchone()
    if not row:
        return jsonify({"document": None, "source_derivatives": None})

    source_row = db.session.execute(
        text(f"""SELECT derivatives FROM "{AI_SCHEMA}".sources WHERE id = :sid"""),
        {"sid": row.source_id},
    ).fetchone()

    return jsonify(
        {
            "document": {
                "id": str(row.id),
                "source_id": str(row.source_id),
                "summary": row.summary,
                "extracted_json": row.extracted_json,
                "json_schema": row.json_schema,
                "window_summaries": row.window_summaries,
                "extraction_model": row.extraction_model,
                "summary_tokens": row.summary_tokens,
                "input_tokens": row.input_tokens,
                "window_size": row.window_size,
                "window_overlap": row.window_overlap,
                "window_count": row.window_count,
                "meta": row.meta,
            },
            "source_derivatives": source_row.derivatives if source_row else None,
        }
    )
