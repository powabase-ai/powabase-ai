"""Validation for the per-request ``runtime_knowledge_bases`` field.

Deliberately strict (unlike the legacy preload ``knowledge_bases`` field,
which validates nothing): unknown ids, malformed entries, out-of-range knobs,
and oversized lists all 400 before any stream opens.
"""

import uuid

from sqlalchemy import text

from ..db import AI_SCHEMA

RUNTIME_KB_MAX_ENTRIES = 10

_VALID_RETRIEVAL_METHODS = {"vector_search", "full_text", "hybrid", "tree_search"}


def _normalize_uuid(value):
    """Parse a UUID and return its canonical (lowercase, dashed) string form.

    Uppercase, braced, and undashed UUIDs all parse successfully but are not
    equal to psycopg's canonical form, which breaks a plain set-comparison
    against the DB and later misses the tool builder's kb_map lookup by
    string key — silently dropping the KB. Raises ValueError/TypeError on
    anything unparseable.
    """
    return str(uuid.UUID(str(value)))


def _parse_source_ids(value):
    """Validate+normalize a ``source_ids`` list. Returns (normalized, error)."""
    if not isinstance(value, list) or not value:
        return None, "'source_ids' must be a non-empty list of ids"
    normalized = []
    for raw_id in value:
        try:
            normalized.append(_normalize_uuid(raw_id))
        except (ValueError, AttributeError, TypeError):
            return None, f"invalid source id: {raw_id!r}"
    return normalized, None


def validate_runtime_knowledge_bases(data, db_session, ai_schema: str = AI_SCHEMA):
    raw = data.get("runtime_knowledge_bases")
    if raw is None:
        return [], None
    if not isinstance(raw, list) or not raw:
        return [], "'runtime_knowledge_bases' must be a non-empty list of objects"
    if len(raw) > RUNTIME_KB_MAX_ENTRIES:
        return [], f"'runtime_knowledge_bases' accepts at most {RUNTIME_KB_MAX_ENTRIES} entries"

    normalized_entries = []
    seen_ids: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict) or not entry.get("id"):
            return [], "each 'runtime_knowledge_bases' entry must be an object with an 'id'"
        try:
            norm_id = _normalize_uuid(entry["id"])
        except (ValueError, AttributeError, TypeError):
            return [], f"invalid knowledge base id: {entry.get('id')!r}"
        if norm_id in seen_ids:
            return [], f"duplicate knowledge base id: {norm_id}"
        seen_ids.add(norm_id)

        entry = dict(entry)  # build a normalized copy; never mutate the caller's object
        entry["id"] = norm_id

        top_k = entry.get("top_k")
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int) or not (1 <= top_k <= 100):
                return [], f"'top_k' must be an integer between 1 and 100: {top_k!r}"

        retrieval_method = entry.get("retrieval_method")
        if retrieval_method is not None and retrieval_method not in _VALID_RETRIEVAL_METHODS:
            return [], f"invalid retrieval_method: {retrieval_method!r}"

        similarity_threshold = entry.get("similarity_threshold")
        if similarity_threshold is not None:
            invalid_type = isinstance(similarity_threshold, bool) or not isinstance(
                similarity_threshold, (int, float)
            )
            if invalid_type or not (0 <= similarity_threshold <= 1):
                return [], (
                    f"'similarity_threshold' must be a number between 0 and 1: "
                    f"{similarity_threshold!r}"
                )

        filter_metadata = entry.get("filter_metadata")
        if filter_metadata is not None and not isinstance(filter_metadata, dict):
            return [], f"'filter_metadata' must be an object: {filter_metadata!r}"

        source_ids = entry.get("source_ids")
        if source_ids is not None:
            normalized_source_ids, err = _parse_source_ids(source_ids)
            if err:
                return [], err
            entry["source_ids"] = normalized_source_ids

        normalized_entries.append(entry)

    ids = [entry["id"] for entry in normalized_entries]
    rows = db_session.execute(
        text(f'SELECT id FROM "{ai_schema}".knowledge_bases WHERE id = ANY(:ids)'),
        {"ids": ids},
    )
    found = {str(row[0]) for row in rows}
    missing = [i for i in ids if i not in found]
    if missing:
        return [], f"unknown knowledge base id(s): {', '.join(missing)}"

    all_source_ids = sorted(
        {sid for entry in normalized_entries for sid in entry.get("source_ids") or []}
    )
    if all_source_ids:
        source_rows = db_session.execute(
            text(f'SELECT id FROM "{ai_schema}".sources WHERE id = ANY(:ids)'),
            {"ids": all_source_ids},
        )
        found_sources = {str(row[0]) for row in source_rows}
        missing_sources = [i for i in all_source_ids if i not in found_sources]
        if missing_sources:
            return [], f"unknown source id(s): {', '.join(missing_sources)}"

    return normalized_entries, None
