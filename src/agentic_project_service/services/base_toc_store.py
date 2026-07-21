"""
Base ToC Store.

Shared base class for PageIndexStore and GraphIndexStore, providing
common CRUD operations for the two-table ToC + nodes schema pattern.
"""

import json
import logging
import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import AI_SCHEMA

logger = logging.getLogger(__name__)


class BaseTocStore:
    """Base class for ToC-based index stores.

    Subclasses must set TOC_TABLE and NODES_TABLE class attributes
    to the appropriate table names (without schema prefix).
    """

    TOC_TABLE: str  # e.g. "page_index_toc"
    NODES_TABLE: str  # e.g. "page_index_nodes"

    def __init__(self, db_session: Session, knowledge_base_id: str):
        self.session = db_session
        self.kb_id = knowledge_base_id

    def store_toc(
        self,
        indexed_source_id: str,
        source_id: str,
        structure: list,
        doc_name: str | None = None,
        doc_description: str | None = None,
    ) -> str:
        """Store a ToC structure in the database.

        Args:
            indexed_source_id: The indexed_source record ID
            source_id: The source record ID
            structure: The metadata-only tree hierarchy (no section text)
            doc_name: Optional document name
            doc_description: Optional document description

        Returns:
            UUID of the created toc record
        """
        toc_id = str(uuid.uuid4())

        self.session.execute(
            text(f"""
                INSERT INTO "{AI_SCHEMA}".{self.TOC_TABLE} (
                    id, indexed_source_id, knowledge_base_id, source_id,
                    doc_name, doc_description, structure
                ) VALUES (
                    :id, :indexed_source_id, :kb_id, :source_id,
                    :doc_name, :doc_description, CAST(:structure AS jsonb)
                )
            """),
            {
                "id": toc_id,
                "indexed_source_id": indexed_source_id,
                "kb_id": self.kb_id,
                "source_id": source_id,
                "doc_name": doc_name,
                "doc_description": doc_description,
                "structure": json.dumps(structure),
            },
        )

        logger.info(
            "Stored %s toc %s for source %s in KB %s",
            self.TOC_TABLE,
            toc_id,
            source_id,
            self.kb_id,
        )
        return toc_id

    def get_tocs(self) -> list[dict]:
        """Get all ToC records for this knowledge base (lightweight, no section text).

        Returns:
            List of toc record dicts with keys:
                id, indexed_source_id, source_id, doc_name,
                doc_description, structure, knowledge_base_id, created_at
        """
        result = self.session.execute(
            text(f"""
                SELECT id, indexed_source_id, knowledge_base_id, source_id,
                       doc_name, doc_description, structure, created_at
                FROM "{AI_SCHEMA}".{self.TOC_TABLE}
                WHERE knowledge_base_id = :kb_id
                ORDER BY created_at ASC
            """),
            {"kb_id": self.kb_id},
        )

        tocs = []
        for row in result:
            tocs.append(
                {
                    "id": str(row[0]),
                    "indexed_source_id": str(row[1]),
                    "knowledge_base_id": str(row[2]),
                    "source_id": str(row[3]),
                    "doc_name": row[4],
                    "doc_description": row[5],
                    "structure": row[6],
                    "created_at": row[7].isoformat() if row[7] else None,
                }
            )

        return tocs

    def get_nodes_by_ids(
        self,
        selections: list[tuple[str, str]],
    ) -> dict[tuple[str, str], dict]:
        """Fetch specific node rows by (toc_id, node_id) pairs.

        Args:
            selections: List of (toc_id, node_id) tuples to fetch

        Returns:
            Dict mapping (toc_id, node_id) -> node row dict
        """
        if not selections:
            return {}

        conditions = []
        params: dict = {}
        for i, (toc_id, node_id) in enumerate(selections):
            conditions.append(f"(toc_id = :toc_{i} AND node_id = :node_{i})")
            params[f"toc_{i}"] = toc_id
            params[f"node_{i}"] = node_id

        where_clause = " OR ".join(conditions)

        result = self.session.execute(
            text(f"""
                SELECT id, toc_id, node_id, title, text, depth,
                       parent_node_id, line_num, meta, source_id,
                       knowledge_base_id
                FROM "{AI_SCHEMA}".{self.NODES_TABLE}
                WHERE {where_clause}
            """),
            params,
        )

        nodes: dict[tuple[str, str], dict] = {}
        for row in result:
            key = (str(row[1]), row[2])
            nodes[key] = {
                "id": str(row[0]),
                "toc_id": str(row[1]),
                "node_id": row[2],
                "title": row[3],
                "text": row[4],
                "depth": row[5],
                "parent_node_id": row[6],
                "line_num": row[7],
                "meta": row[8],
                "source_id": str(row[9]),
                "knowledge_base_id": str(row[10]),
            }

        return nodes

    def get_children_by_parent_ids(
        self,
        parent_selections: list[tuple[str, str]],
    ) -> dict[tuple[str, str], list[dict]]:
        """Fetch direct child nodes for (toc_id, parent_node_id) pairs.

        Args:
            parent_selections: List of (toc_id, parent_node_id) tuples

        Returns:
            Dict mapping (toc_id, parent_node_id) -> [child row dicts]
        """
        if not parent_selections:
            return {}

        conditions = []
        params: dict = {}
        for i, (toc_id, parent_node_id) in enumerate(parent_selections):
            conditions.append(f"(toc_id = :toc_{i} AND parent_node_id = :parent_{i})")
            params[f"toc_{i}"] = toc_id
            params[f"parent_{i}"] = parent_node_id

        where_clause = " OR ".join(conditions)

        result = self.session.execute(
            text(f"""
                SELECT id, toc_id, node_id, title, text, depth,
                       parent_node_id, line_num, meta, source_id,
                       knowledge_base_id
                FROM "{AI_SCHEMA}".{self.NODES_TABLE}
                WHERE {where_clause}
            """),
            params,
        )

        children: dict[tuple[str, str], list[dict]] = {}
        for row in result:
            parent_key = (str(row[1]), row[6])  # (toc_id, parent_node_id)
            child = {
                "id": str(row[0]),
                "toc_id": str(row[1]),
                "node_id": row[2],
                "title": row[3],
                "text": row[4],
                "depth": row[5],
                "parent_node_id": row[6],
                "line_num": row[7],
                "meta": row[8],
                "source_id": str(row[9]),
                "knowledge_base_id": str(row[10]),
            }
            children.setdefault(parent_key, []).append(child)

        return children

    def delete_by_indexed_source(self, indexed_source_id: str) -> int:
        """Delete ToC records for a specific indexed source (sections cascade via FK).

        Args:
            indexed_source_id: The indexed_source record to delete data for

        Returns:
            Number of ToC records deleted
        """
        result = self.session.execute(
            text(f"""
                DELETE FROM "{AI_SCHEMA}".{self.TOC_TABLE}
                WHERE indexed_source_id = :indexed_source_id
                RETURNING id
            """),
            {"indexed_source_id": indexed_source_id},
        )
        deleted = result.rowcount
        self.session.commit()

        logger.info(
            "Deleted %d %s records (nodes cascaded) for indexed_source %s",
            deleted,
            self.TOC_TABLE,
            indexed_source_id,
        )
        return deleted

    def count_tocs(self) -> int:
        """Count ToC records in this knowledge base."""
        result = self.session.execute(
            text(f"""
                SELECT COUNT(*) FROM "{AI_SCHEMA}".{self.TOC_TABLE}
                WHERE knowledge_base_id = :kb_id
            """),
            {"kb_id": self.kb_id},
        )
        return result.scalar() or 0

    def count_nodes(self) -> int:
        """Count node rows in this knowledge base."""
        result = self.session.execute(
            text(f"""
                SELECT COUNT(*) FROM "{AI_SCHEMA}".{self.NODES_TABLE}
                WHERE knowledge_base_id = :kb_id
            """),
            {"kb_id": self.kb_id},
        )
        return result.scalar() or 0
