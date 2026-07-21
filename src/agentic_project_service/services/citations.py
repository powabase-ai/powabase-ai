"""Citation labeling, parsing, and persistence for agent runs."""

import json
import logging
import re
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
        "If no specific context is referenced, do not include a citation."
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


def persist_citations(
    db_session: Session,
    run_id: str,
    citations: list[dict[str, Any]],
) -> None:
    """
    Bulk-insert citations into ai.message_citations.

    Args:
        db_session: SQLAlchemy session
        run_id: The user-facing run_id string (e.g. "run_abc123")
        citations: List of citation dicts from parse_citations_from_response
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

    for cite in citations:
        db_session.execute(
            text(f"""
                INSERT INTO "{AI_SCHEMA}".message_citations
                (run_id, citation_key, item_id, source_id, text_excerpt, meta)
                VALUES (:run_id, :citation_key, :item_id, :source_id, :text_excerpt, CAST(:meta AS jsonb))
                ON CONFLICT (run_id, citation_key) DO NOTHING
            """),
            {
                "run_id": run_uuid,
                "citation_key": int(cite["key"]),
                "item_id": cite.get("item_id"),
                "source_id": cite.get("source_id"),
                "text_excerpt": cite.get("text_excerpt", ""),
                "meta": json.dumps(cite.get("meta", {})),
            },
        )
