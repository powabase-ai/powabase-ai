"""
PageIndex Store.

Handles storage and retrieval of PageIndex ToC structures and node
rows across the ai.page_index_toc and ai.page_index_nodes tables.
"""

import json
import logging
import uuid

from sqlalchemy import text

from ..db import AI_SCHEMA
from .base_toc_store import BaseTocStore

logger = logging.getLogger(__name__)


class PageIndexStore(BaseTocStore):
    """Store for PageIndex ToC + node data.

    Provides CRUD operations for the two-table page_index schema:
    - page_index_toc: one lightweight row per document (structure only)
    - page_index_nodes: one row per section/node (with full text)
    """

    TOC_TABLE = "page_index_toc"
    NODES_TABLE = "page_index_nodes"

    def store_nodes(
        self,
        toc_id: str,
        indexed_source_id: str,
        source_id: str,
        nodes: list[dict],
    ) -> int:
        """Batch-insert node rows for a ToC record.

        Caller is responsible for committing the transaction.

        Args:
            toc_id: The parent toc record UUID
            indexed_source_id: The indexed_source record ID
            source_id: The source record ID
            nodes: List of node dicts, each with:
                node_id, title, text, depth, parent_node_id, line_num, meta

        Returns:
            Number of nodes inserted
        """
        if not nodes:
            return 0

        stmt = text(f"""
            INSERT INTO "{AI_SCHEMA}".page_index_nodes (
                id, toc_id, indexed_source_id, knowledge_base_id,
                source_id, node_id, title, depth, parent_node_id,
                text, line_num, meta
            ) VALUES (
                :id, :toc_id, :indexed_source_id, :kb_id,
                :source_id, :node_id, :title, :depth, :parent_node_id,
                :text, :line_num, CAST(:meta AS jsonb)
            )
        """)

        for node in nodes:
            node_row_id = str(uuid.uuid4())
            self.session.execute(
                stmt,
                {
                    "id": node_row_id,
                    "toc_id": toc_id,
                    "indexed_source_id": indexed_source_id,
                    "kb_id": self.kb_id,
                    "source_id": source_id,
                    "node_id": node["node_id"],
                    "title": node.get("title"),
                    "depth": node.get("depth", 0),
                    "parent_node_id": node.get("parent_node_id"),
                    "text": node["text"],
                    "line_num": node.get("line_num"),
                    "meta": json.dumps(node.get("meta", {})),
                },
            )

        logger.info(
            "Stored %d nodes for toc %s in KB %s",
            len(nodes),
            toc_id,
            self.kb_id,
        )
        return len(nodes)
