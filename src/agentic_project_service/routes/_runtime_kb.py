"""Validation for the per-request ``runtime_knowledge_bases`` field.

Deliberately strict (unlike the legacy preload ``knowledge_bases`` field,
which validates nothing): unknown ids, malformed entries, out-of-range knobs,
and oversized lists all 400 before any stream opens.
"""

import uuid

from sqlalchemy import text

from ..db import AI_SCHEMA
from ..services.settings_registry import SETTINGS_REGISTRY

RUNTIME_KB_MAX_ENTRIES = 10

_VALID_RETRIEVAL_METHODS = {"vector_search", "full_text", "hybrid", "tree_search"}

_ALLOWED_ENTRY_KEYS = {
    "id",
    "top_k",
    "retrieval_method",
    "similarity_threshold",
    "filter_metadata",
    "source_ids",
    "max_context_tokens",
}

# Bounds mirror the settings registry's KB_DEFAULT_MAX_CONTEXT_TOKENS
# definition — a per-entry override must stay inside the same range the
# project-wide default is constrained to.
_MAX_CONTEXT_TOKENS_DEF = SETTINGS_REGISTRY["KB_DEFAULT_MAX_CONTEXT_TOKENS"]
_MAX_CONTEXT_TOKENS_MIN = _MAX_CONTEXT_TOKENS_DEF.min
_MAX_CONTEXT_TOKENS_MAX = _MAX_CONTEXT_TOKENS_DEF.max


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
        unknown_keys = set(entry) - _ALLOWED_ENTRY_KEYS
        if unknown_keys:
            return [], f"unknown key(s) in runtime_knowledge_bases entry: {sorted(unknown_keys)}"
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

        # NOTE: this only validates the knob's range — it is silently ignored
        # at tool-build time whenever the run's knowledge_search tool covers
        # more than one KB (attached + runtime combined), which resets to
        # KB_DEFAULT_MAX_CONTEXT_TOKENS instead of picking a winner among
        # per-KB values. Honored only for single-KB runs.
        max_context_tokens = entry.get("max_context_tokens")
        if max_context_tokens is not None:
            invalid_type = isinstance(max_context_tokens, bool) or not isinstance(
                max_context_tokens, int
            )
            if invalid_type or not (
                _MAX_CONTEXT_TOKENS_MIN <= max_context_tokens <= _MAX_CONTEXT_TOKENS_MAX
            ):
                return [], (
                    f"'max_context_tokens' must be an integer between "
                    f"{_MAX_CONTEXT_TOKENS_MIN:.0f} and {_MAX_CONTEXT_TOKENS_MAX:.0f}: "
                    f"{max_context_tokens!r}"
                )

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

    # source_ids must be indexed into THAT entry's knowledge base, not just
    # present somewhere in the project — a source that exists but isn't
    # indexed into this KB silently filters retrieval to zero chunks
    # downstream, with no signal to the caller. One query per entry with
    # source_ids (the cap above bounds this to RUNTIME_KB_MAX_ENTRIES).
    for entry in normalized_entries:
        entry_source_ids = entry.get("source_ids")
        if not entry_source_ids:
            continue
        kb_id = entry["id"]
        indexed_rows = db_session.execute(
            text(
                f'SELECT source_id FROM "{ai_schema}".indexed_sources '
                "WHERE knowledge_base_id = :kb_id AND source_id = ANY(:ids)"
            ),
            {"kb_id": kb_id, "ids": entry_source_ids},
        )
        indexed = {str(row[0]) for row in indexed_rows}
        missing = [sid for sid in entry_source_ids if sid not in indexed]
        if missing:
            return [], f"source id(s) not in knowledge base {kb_id}: {', '.join(missing)}"

    return normalized_entries, None
