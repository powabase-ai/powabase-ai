"""Validation for the per-request ``runtime_knowledge_bases`` field.

Deliberately strict (unlike the legacy preload ``knowledge_bases`` field,
which validates nothing): unknown ids, malformed entries, and oversized lists
all 400 before any stream opens.
"""

import uuid

from sqlalchemy import text

from ..db import AI_SCHEMA

RUNTIME_KB_MAX_ENTRIES = 10


def validate_runtime_knowledge_bases(data, db_session, ai_schema: str = AI_SCHEMA):
    raw = data.get("runtime_knowledge_bases")
    if raw is None:
        return [], None
    if not isinstance(raw, list) or not raw:
        return [], "'runtime_knowledge_bases' must be a non-empty list of objects"
    if len(raw) > RUNTIME_KB_MAX_ENTRIES:
        return [], f"'runtime_knowledge_bases' accepts at most {RUNTIME_KB_MAX_ENTRIES} entries"
    for entry in raw:
        if not isinstance(entry, dict) or not entry.get("id"):
            return [], "each 'runtime_knowledge_bases' entry must be an object with an 'id'"
        try:
            uuid.UUID(str(entry["id"]))
        except (ValueError, AttributeError, TypeError):
            return [], f"invalid knowledge base id: {entry.get('id')!r}"
    ids = [str(entry["id"]) for entry in raw]
    rows = db_session.execute(
        text(f'SELECT id FROM "{ai_schema}".knowledge_bases WHERE id = ANY(:ids)'),
        {"ids": ids},
    )
    found = {str(row[0]) for row in rows}
    missing = [i for i in ids if i not in found]
    if missing:
        return [], f"unknown knowledge base id(s): {', '.join(missing)}"
    return raw, None
