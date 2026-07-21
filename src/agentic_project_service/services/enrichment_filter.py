"""
Enrichment metadata helper.

Provides functions to fetch enrichment configs and retrieve per-item
metadata from the dynamic per-KB metadata tables.
"""

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import AI_SCHEMA
from .sql_utils import validate_sql_identifier

logger = logging.getLogger(__name__)


def get_enrichment_config(db_session: Session, knowledge_base_id: str) -> dict | None:
    """Fetch enrichment config for a KB. Returns None if not configured."""
    result = db_session.execute(
        text(
            f"SELECT id, fields, llm_model, metadata_table_name, status "
            f'FROM "{AI_SCHEMA}".enrichment_configs '
            f"WHERE knowledge_base_id = :kb_id"
        ),
        {"kb_id": knowledge_base_id},
    )
    row = result.fetchone()
    if not row:
        return None
    return {
        "id": str(row[0]),
        "fields": row[1] or [],
        "llm_model": row[2],
        "metadata_table_name": row[3],
        "status": row[4],
    }


def get_enrichment_metadata_for_items(
    db_session: Session,
    enrichment_config: dict,
    item_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch enrichment metadata for a set of item IDs.

    Returns {item_id: {field: value, ...}} with only non-null values.
    Returns empty dict on failure or if not configured.
    """
    if not item_ids:
        return {}

    table_name = enrichment_config.get("metadata_table_name")
    if not table_name:
        return {}

    fields = enrichment_config.get("fields", [])
    if not fields:
        return {}

    try:
        field_names = [validate_sql_identifier(f["name"]) for f in fields]
    except ValueError:
        logger.warning("Skipping enrichment metadata: invalid field name", exc_info=True)
        return {}

    qualified = f'"{AI_SCHEMA}"."{table_name}"'
    cols = ", ".join(f'"{fn}"' for fn in field_names)
    placeholders = ", ".join(f":id_{i}" for i in range(len(item_ids)))
    params = {f"id_{i}": iid for i, iid in enumerate(item_ids)}
    sql = f"SELECT item_id, {cols} FROM {qualified} WHERE item_id IN ({placeholders})"

    try:
        rows = db_session.execute(text(sql), params).fetchall()
    except Exception:
        logger.warning("Failed to fetch enrichment metadata from %s", table_name, exc_info=True)
        return {}

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        meta = {}
        for i, fn in enumerate(field_names):
            val = row[i + 1]
            if val is not None:
                meta[fn] = val
        if meta:
            result[str(row[0])] = meta
    return result
