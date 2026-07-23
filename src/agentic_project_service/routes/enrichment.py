"""Metadata enrichment CRUD routes for knowledge bases."""

import json
import logging
import uuid

from flask import Blueprint, jsonify, request
from sqlalchemy import text

from agentic.knowledge.model_config import METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS

from ..auth import require_auth
from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
from ..services.metadata_enricher import MetadataEnricher
from ..services.sql_utils import validate_sql_identifier
from ..tasks.enrichment import enrich_knowledge_base

logger = logging.getLogger(__name__)


# Conservative per-item cost estimate for the pre-check (flat metadata_enrichment
# fee + llm_call recoup at platform rates).
#
# Derivation:
#   - ~2500 tokens per item × 2 credits/1k = 5 credits flat fee
#   - + ~50 credits llm_call for gpt-5-mini-class models at $0.0005/call × the
#     configured markup × 100_000 (billing-adapter charge formula)
#   - Total ~55. Round up to 60 for batch overhead + variable model pricing.
#
# When bumping: re-derive the flat-fee and llm_call rates from the active
# billing adapter's pricing tables, don't guess.
_ENRICHMENT_PER_ITEM_MAX_CREDITS: int = 60

# Floor when the KB is empty (no items yet) or count lookup fails.
_ENRICHMENT_FALLBACK_ESTIMATE: int = 200


def _count_enrichable_items_for_kb(kb_id: str) -> int:
    """Total items the enrichment job would process for this KB.

    Mirrors MetadataEnricher.count_total_items but standalone so the pre-check
    can run before constructing an enricher. Looks up the KB's strategy from
    indexing_config.

    Raises ServiceUnavailable on any lookup failure (fail-closed) so a DB
    timeout does NOT silently fall back to the old 200-credit constant and
    restore the billing-leak behavior the PR fixed.
    """
    from ..services.metadata_enricher import MetadataEnricher
    from ..tasks.enrichment import _get_kb_strategy

    try:
        strategy = _get_kb_strategy(kb_id)
        enricher = MetadataEnricher(db.session, kb_id)
        return enricher.count_total_items(strategy)
    except Exception as exc:
        logger.error(
            "count_enrichable_items lookup failed for kb=%s; failing closed: %s",
            kb_id,
            exc,
        )
        from werkzeug.exceptions import ServiceUnavailable

        raise ServiceUnavailable(
            "Balance check failed (KB item count lookup error). Retry shortly."
        )


def _enrich_check_balance(kb_id: str) -> None:
    """Pre-op balance check that scales with actual KB size.

    Replaces the old 200-credit constant with `total_items ×
    _ENRICHMENT_PER_ITEM_MAX_CREDITS`. For empty KBs (count=0), falls back
    to _ENRICHMENT_FALLBACK_ESTIMATE so the pre-check is always non-zero.

    Raises ServiceUnavailable when the item count lookup fails — fail-closed
    so a DB timeout doesn't transiently restore the old leaky constant.

    See #437 root-cause discussion for why this matters: with the old
    constant, a 1725-item enrichment passed pre-check at 200 credits and
    incurred ~34,500 credits of platform-paid work that the customer never
    had to budget for.

    Routed through the billing port — the no-op adapter makes this inert in
    OSS/unit-test/local-dev builds; the cloud adapter enforces the cap.
    """
    total_items = _count_enrichable_items_for_kb(kb_id)
    estimated_cost = max(
        _ENRICHMENT_FALLBACK_ESTIMATE,
        total_items * _ENRICHMENT_PER_ITEM_MAX_CREDITS,
    )
    billing.check_balance(estimated_cost=estimated_cost)


enrichment_bp = Blueprint("enrichment", __name__, url_prefix="/api/knowledge-bases")

_VALID_FIELD_TYPES = {"text", "boolean", "number", "enum"}


_RESERVED_COLUMNS = {"id", "item_id", "item_type", "enriched_at", "_enrichment_error"}


def _validate_fields(fields: list) -> str | None:
    """Validate enrichment field definitions. Returns error message or None."""
    if not isinstance(fields, list) or not fields:
        return "fields must be a non-empty list"

    seen_names: set[str] = set()
    for i, f in enumerate(fields):
        if not isinstance(f, dict):
            return f"fields[{i}] must be an object"

        name = f.get("name")
        if not name or not isinstance(name, str):
            return f"fields[{i}].name is required"
        try:
            validate_sql_identifier(name)
        except ValueError:
            return (
                f"fields[{i}].name '{name}' is invalid. "
                "Must be alphanumeric + underscores, starting with a letter."
            )
        name_lower = name.lower()
        if name_lower in _RESERVED_COLUMNS:
            return f"fields[{i}].name '{name}' is reserved"
        if name_lower in seen_names:
            return f"Duplicate field name: '{name}'"
        seen_names.add(name_lower)

        desc = f.get("description")
        if not desc or not isinstance(desc, str):
            return f"fields[{i}].description is required"

        ftype = f.get("type")
        if ftype not in _VALID_FIELD_TYPES:
            return f"fields[{i}].type must be one of: {', '.join(sorted(_VALID_FIELD_TYPES))}"

        if ftype == "enum":
            enum_values = f.get("enum_values")
            if (
                not isinstance(enum_values, list)
                or len(enum_values) < 2
                or not all(isinstance(v, str) and v.strip() for v in enum_values)
            ):
                return (
                    f"fields[{i}].enum_values must be a list of at least 2 strings "
                    f"when type is 'enum'"
                )

    return None


def _get_existing_config(kb_id: str) -> dict | None:
    """Fetch existing enrichment config for a KB."""
    result = db.session.execute(
        text(
            f"SELECT id, fields, llm_model, max_tokens, use_multimodal, "
            f"metadata_table_name, status, "
            f"enriched_count, total_count, error_message, celery_task_id, "
            f"created_at, updated_at "
            f'FROM "{AI_SCHEMA}".enrichment_configs '
            f"WHERE knowledge_base_id = :kb_id"
        ),
        {"kb_id": kb_id},
    )
    row = result.fetchone()
    if not row:
        return None
    return {
        "id": str(row[0]),
        "fields": row[1] or [],
        "llm_model": row[2],
        "max_tokens": row[3] or METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS,
        "use_multimodal": bool(row[4]) if row[4] is not None else False,
        "metadata_table_name": row[5],
        "status": row[6],
        "enriched_count": row[7],
        "total_count": row[8],
        "error_message": row[9],
        "celery_task_id": row[10],
        "created_at": row[11].isoformat() if row[11] else None,
        "updated_at": row[12].isoformat() if row[12] else None,
    }


@enrichment_bp.route("/<kb_id>/enrichment", methods=["PUT"])
@require_auth
def put_enrichment_config(kb_id: str):
    """Create or replace enrichment config for a knowledge base."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    fields = data.get("fields")
    error = _validate_fields(fields)
    if error:
        return jsonify({"error": error}), 400

    llm_model = data.get("llm_model")
    max_tokens = data.get("max_tokens", METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS)
    if not isinstance(max_tokens, int) or max_tokens < 1:
        return jsonify({"error": "max_tokens must be a positive integer"}), 400
    use_multimodal = bool(data.get("use_multimodal", False))

    enricher = MetadataEnricher(db.session, kb_id)
    existing = _get_existing_config(kb_id)

    if existing:
        # Check if fields or model changed
        fields_changed = json.dumps(existing["fields"], sort_keys=True) != json.dumps(
            fields, sort_keys=True
        )
        model_changed = existing["llm_model"] != llm_model
        max_tokens_changed = (
            existing.get("max_tokens", METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS) != max_tokens
        )
        multimodal_changed = existing.get("use_multimodal", False) != use_multimodal

        if fields_changed or model_changed:
            if existing["status"] == "enriching":
                return jsonify(
                    {
                        "error": "Cannot update enrichment config while enrichment is in progress. "
                        "Please wait for the current run to complete.",
                    }
                ), 409

            # Drop old metadata table, create new one
            if existing["metadata_table_name"]:
                enricher.drop_metadata_table(existing["metadata_table_name"])

            table_name = enricher.create_metadata_table(fields)

            db.session.execute(
                text(
                    f'UPDATE "{AI_SCHEMA}".enrichment_configs '
                    f"SET fields = CAST(:fields AS jsonb), "
                    f"    llm_model = :llm_model, "
                    f"    max_tokens = :max_tokens, "
                    f"    use_multimodal = :use_multimodal, "
                    f"    metadata_table_name = :table_name, "
                    f"    status = 'idle', "
                    f"    enriched_count = 0, "
                    f"    total_count = 0, "
                    f"    error_message = NULL "
                    f"WHERE knowledge_base_id = :kb_id"
                ),
                {
                    "fields": json.dumps(fields),
                    "llm_model": llm_model,
                    "max_tokens": max_tokens,
                    "use_multimodal": use_multimodal,
                    "table_name": table_name,
                    "kb_id": kb_id,
                },
            )
            db.session.commit()

            # Trigger full enrichment
            _enrich_check_balance(kb_id)
            task = enrich_knowledge_base.delay(
                kb_id,
                incremental=False,
            )
            config = _get_existing_config(kb_id)
            config["task_id"] = task.id
            return jsonify({"config": config, "re_enrichment_triggered": True})
        elif multimodal_changed:
            if existing["status"] == "enriching":
                return jsonify(
                    {
                        "error": "Cannot update enrichment config while enrichment is in progress. "
                        "Please wait for the current run to complete.",
                    }
                ), 409

            # Multimodal toggle changed — update flag and trigger re-enrichment
            # (results will differ fundamentally with/without images)
            db.session.execute(
                text(
                    f'UPDATE "{AI_SCHEMA}".enrichment_configs '
                    f"SET use_multimodal = :use_multimodal, max_tokens = :max_tokens, "
                    f"error_message = NULL "
                    f"WHERE knowledge_base_id = :kb_id"
                ),
                {"use_multimodal": use_multimodal, "max_tokens": max_tokens, "kb_id": kb_id},
            )
            db.session.commit()
            _enrich_check_balance(kb_id)
            task = enrich_knowledge_base.delay(
                kb_id,
                incremental=False,
            )
            config = _get_existing_config(kb_id)
            config["task_id"] = task.id
            return jsonify({"config": config, "re_enrichment_triggered": True})
        elif max_tokens_changed:
            # Lightweight: just update max_tokens, no table changes needed
            db.session.execute(
                text(
                    f'UPDATE "{AI_SCHEMA}".enrichment_configs '
                    f"SET max_tokens = :max_tokens "
                    f"WHERE knowledge_base_id = :kb_id"
                ),
                {"max_tokens": max_tokens, "kb_id": kb_id},
            )
            db.session.commit()
            config = _get_existing_config(kb_id)
            return jsonify({"config": config, "re_enrichment_triggered": False})
        else:
            # No changes
            return jsonify({"config": existing, "re_enrichment_triggered": False})
    else:
        # Create new config
        config_id = str(uuid.uuid4())
        table_name = enricher.create_metadata_table(fields)

        db.session.execute(
            text(
                f'INSERT INTO "{AI_SCHEMA}".enrichment_configs '
                f"(id, knowledge_base_id, fields, llm_model, max_tokens, use_multimodal, metadata_table_name) "
                f"VALUES (:id, :kb_id, CAST(:fields AS jsonb), :llm_model, :max_tokens, :use_multimodal, :table_name)"
            ),
            {
                "id": config_id,
                "kb_id": kb_id,
                "fields": json.dumps(fields),
                "llm_model": llm_model,
                "max_tokens": max_tokens,
                "use_multimodal": use_multimodal,
                "table_name": table_name,
            },
        )
        db.session.commit()

        # Trigger enrichment
        _enrich_check_balance(kb_id)
        task = enrich_knowledge_base.delay(
            kb_id,
            incremental=False,
        )
        config = _get_existing_config(kb_id)
        config["task_id"] = task.id
        return jsonify({"config": config, "re_enrichment_triggered": True}), 201


@enrichment_bp.route("/<kb_id>/enrichment", methods=["GET"])
@require_auth
def get_enrichment_config(kb_id: str):
    """Get current enrichment config and status."""
    config = _get_existing_config(kb_id)
    if not config:
        return jsonify({"config": None})

    return jsonify({"config": config})


@enrichment_bp.route("/<kb_id>/enrichment", methods=["DELETE"])
@require_auth
def delete_enrichment_config(kb_id: str):
    """Remove enrichment config and all results."""
    existing = _get_existing_config(kb_id)
    if not existing:
        return jsonify({"message": "No enrichment config found"}), 404

    if existing["status"] == "enriching":
        return jsonify(
            {
                "error": "Cannot delete enrichment config while enrichment is in progress. "
                "Please wait for the current run to complete.",
            }
        ), 409

    # Drop the dynamic metadata table
    if existing["metadata_table_name"]:
        enricher = MetadataEnricher(db.session, kb_id)
        enricher.drop_metadata_table(existing["metadata_table_name"])

    # Delete the config row
    db.session.execute(
        text(f'DELETE FROM "{AI_SCHEMA}".enrichment_configs WHERE knowledge_base_id = :kb_id'),
        {"kb_id": kb_id},
    )
    db.session.commit()

    return jsonify({"message": "Enrichment config and results deleted"})


@enrichment_bp.route("/<kb_id>/enrichment/results", methods=["GET"])
@require_auth
def get_enrichment_results(kb_id: str):
    """Fetch enrichment metadata for specific items."""
    existing = _get_existing_config(kb_id)
    if not existing:
        return jsonify({"error": "No enrichment config found"}), 404
    if existing["status"] not in ("completed", "completed_with_errors"):
        return jsonify({"results": {}, "status": existing["status"]})

    raw_ids = request.args.get("item_ids", "")
    item_ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
    if not item_ids:
        return jsonify({"results": {}})

    table_name = existing["metadata_table_name"]
    fields = existing["fields"] or []
    field_names = [validate_sql_identifier(f["name"]) for f in fields]
    if not field_names or not table_name:
        return jsonify({"results": {}})

    safe_table = validate_sql_identifier(table_name)

    # Check if _enrichment_error column exists (for pre-existing tables)
    has_error_col = (
        db.session.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = :schema AND table_name = :table "
                "AND column_name = '_enrichment_error'"
            ),
            {"schema": AI_SCHEMA, "table": safe_table},
        ).fetchone()
        is not None
    )

    placeholders = ", ".join(f":id_{i}" for i in range(len(item_ids)))
    params = {f"id_{i}": iid for i, iid in enumerate(item_ids)}
    cols = ", ".join(f'"{fn}"' for fn in field_names)
    error_col = ', "_enrichment_error"' if has_error_col else ""
    rows = db.session.execute(
        text(
            f'SELECT item_id, {cols}{error_col} FROM "{AI_SCHEMA}"."{safe_table}" '
            f"WHERE item_id IN ({placeholders})"
        ),
        params,
    ).fetchall()

    results = {}
    item_errors = {}
    for row in rows:
        item_id_str = str(row[0])
        results[item_id_str] = {fn: row[i + 1] for i, fn in enumerate(field_names)}
        if has_error_col:
            error_val = row[len(field_names) + 1]
            if error_val:
                item_errors[item_id_str] = str(error_val)

    return jsonify({"results": results, "fields": fields, "item_errors": item_errors})


@enrichment_bp.route("/<kb_id>/enrichment/run", methods=["POST"])
@require_auth
def trigger_enrichment(kb_id: str):
    """Manually trigger (re-)enrichment."""
    existing = _get_existing_config(kb_id)
    if not existing:
        return jsonify({"error": "No enrichment config found"}), 404

    data = request.get_json() or {}
    incremental = data.get("incremental", False)
    retry_failed = data.get("retry_failed", False)

    # Pre-charge balance check before flipping status.
    _enrich_check_balance(kb_id)

    # Set status eagerly so API consumers see 'enriching' immediately
    db.session.execute(
        text(
            f'UPDATE "{AI_SCHEMA}".enrichment_configs '
            f"SET status = 'enriching', celery_task_id = NULL, error_message = NULL "
            f"WHERE knowledge_base_id = :kb_id"
        ),
        {"kb_id": kb_id},
    )
    db.session.commit()

    task = enrich_knowledge_base.delay(
        kb_id,
        incremental=incremental,
        retry_failed=retry_failed,
    )

    return jsonify(
        {
            "status": "started",
            "task_id": task.id,
            "incremental": incremental,
            "retry_failed": retry_failed,
        }
    )
