"""Citation labeling, parsing, and persistence for agent runs."""

import json
import logging
import re
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

TEXT_EXCERPT_MAX_LEN = 300
CITATION_PATTERN = re.compile(r"\[(\d+)\]")
AI_SCHEMA = "ai"


def build_citation_map(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Build a citation map from retrieved context items.

    Each item is assigned a sequential key ("1", "2", ...).
    Diagnostics items (with "_type": "retrieval_diagnostics") are skipped.

    Returns:
        Dict mapping citation key to metadata.
    """
    citation_map: dict[str, dict[str, Any]] = {}
    seq = 0
    for item in items:
        if item.get("_type") == "retrieval_diagnostics":
            continue
        seq += 1
        key = str(seq)
        text_val = item.get("text", "")
        citation_map[key] = {
            "key": key,
            "item_id": item.get("id"),
            "source_id": item.get("source_id"),
            "source_name": item.get("source_name", ""),
            "text_excerpt": text_val[:TEXT_EXCERPT_MAX_LEN] if text_val else "",
            "meta": item.get("meta", {}),
        }
    return citation_map


def build_citation_instruction() -> str:
    """Return the citation instruction to append to the system prompt."""
    return (
        "When referencing the provided context, include citations in brackets like [1], [2]. "
        "Each citation should be in its own brackets — use [1][2], not [1, 2]. "
        "If no specific context is referenced, do not include a citation. "
        "In tool results, each citable unit carries its key: a [n] label before the text, "
        'or a "cite": "[n]" field on a JSON item. Cite that key.'
    )


def parse_citations_from_response(
    content: str, citation_map: dict[str, dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """
    Parse citation markers from LLM response.

    Returns:
        Tuple of (cleaned_content, used_citations_list).
        Invalid/hallucinated markers are stripped from cleaned_content.
    """
    used_keys = set(CITATION_PATTERN.findall(content))
    valid_keys = used_keys & set(citation_map.keys())
    invalid_keys = used_keys - valid_keys

    cleaned = content
    for key in invalid_keys:
        cleaned = cleaned.replace(f"[{key}]", "")

    citations = [citation_map[k] for k in sorted(valid_keys, key=int)]
    return cleaned, citations


def _uuid_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        return None


SMALLINT_MAX = 32767


def _clean_text(value: Any) -> Any:
    """Strip NUL bytes, which Postgres text columns reject."""
    return value.replace("\x00", "") if isinstance(value, str) else value


def _meta_json(meta: Any) -> str:
    """Serialize ``meta`` for a jsonb column: no NaN/Infinity, no NUL characters."""
    try:
        encoded = json.dumps(meta or {}, default=str, allow_nan=False)
    except ValueError:
        logger.warning("Citation meta has non-finite numbers; storing it empty")
        return "{}"
    return encoded.replace("\\u0000", "")


def _key_in_range(cite: dict[str, Any]) -> bool:
    if int(cite["key"]) <= SMALLINT_MAX:
        return True
    logger.warning("Skipping citation %s: key exceeds the column range", cite["key"])
    return False


def persist_citations(
    db_session: Session,
    run_id: str,
    citations: list[dict[str, Any]],
) -> None:
    """
    Bulk-insert citation rows into ai.message_citations.

    Rows come from ``build_citation_map`` / ``parse_citations_from_response``,
    or from ``CitationRegistry.citation_map()`` with a ``cited`` flag added. A
    row without ``kind`` is a knowledge-base chunk, and a row without ``cited``
    is cited: that is the shape every caller wrote before run-wide keys.

    Every unit of a run is persisted, not only the cited ones, so one bad value
    must not lose them all. Ids that are not UUIDs are stored as NULL, and so is
    a ``source_id`` whose source no longer exists (it would violate the FK).

    Args:
        db_session: SQLAlchemy session
        run_id: The user-facing run_id string (e.g. "run_abc123")
        citations: Citation dicts keyed as above
    """
    if not citations:
        return

    # Look up the agent_runs.id from the user-facing run_id
    result = db_session.execute(
        text(f'SELECT id FROM "{AI_SCHEMA}".agent_runs WHERE run_id = :run_id'),
        {"run_id": run_id},
    )
    row = result.fetchone()
    if not row:
        logger.warning("Cannot persist citations: agent run %s not found", run_id)
        return
    run_uuid = str(row[0])

    citations = [c for c in citations if _key_in_range(c)]
    if not citations:
        return

    rows = [
        {
            "run_id": run_uuid,
            "citation_key": int(cite["key"]),
            "item_id": _uuid_or_none(cite.get("item_id")),
            "source_id": _uuid_or_none(cite.get("source_id")),
            "text_excerpt": _clean_text(cite.get("text_excerpt", "")),
            "meta": _meta_json(cite.get("meta")),
            "kind": cite.get("kind") or "kb_chunk",
            "tool_name": _clean_text(cite.get("tool_name")),
            "call_id": _clean_text(cite.get("call_id")),
            "title": _clean_text(cite.get("title")),
            "url": _clean_text(cite.get("url")),
            "knowledge_base_id": _uuid_or_none(cite.get("knowledge_base_id")),
            "cited": cite.get("cited") is not False,
        }
        for cite in citations
    ]
    db_session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".message_citations
            (run_id, citation_key, item_id, source_id, text_excerpt, meta,
             kind, tool_name, call_id, title, url, knowledge_base_id, cited)
            VALUES (
                :run_id, :citation_key, CAST(:item_id AS uuid),
                (SELECT id FROM "{AI_SCHEMA}".sources WHERE id = CAST(:source_id AS uuid)),
                :text_excerpt, CAST(:meta AS jsonb),
                :kind, :tool_name, :call_id, :title, :url,
                CAST(:knowledge_base_id AS uuid), :cited
            )
            ON CONFLICT (run_id, citation_key) DO NOTHING
        """),
        rows,
    )
