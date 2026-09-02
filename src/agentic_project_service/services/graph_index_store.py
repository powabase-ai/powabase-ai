"""
GraphIndex Store.

Handles storage and retrieval of GraphIndex ToC structures and node
rows across the ai.graph_index_toc and ai.graph_index_nodes tables.
"""

import json
import logging
import uuid

from sqlalchemy import text

from ..db import AI_SCHEMA
from .base_toc_store import BaseTocStore
from .base_vector_store import ensure_embedding_index

logger = logging.getLogger(__name__)


class GraphIndexStore(BaseTocStore):
    """Store for GraphIndex ToC + node data.

    Provides CRUD operations for the two-table graph_index schema:
    - graph_index_toc: one lightweight row per document (structure only)
    - graph_index_nodes: one row per section/node (with full text; embeddings in ai.embeddings)
    """

    TOC_TABLE = "graph_index_toc"
    NODES_TABLE = "graph_index_nodes"

    def store_nodes(
        self,
        toc_id: str,
        indexed_source_id: str,
        source_id: str,
        nodes: list[dict],
        embedding_model: str | None = None,
    ) -> tuple[int, list[str]]:
        """Batch-insert node rows for a ToC record.

        Each node dict has: node_id, title, text, depth,
        parent_node_id, line_num, meta, embedding (list[float] | None).

        Returns:
            Tuple of (number of nodes inserted, list of row IDs)
        """
        if not nodes:
            return 0, []

        stmt = text(f"""
            INSERT INTO "{AI_SCHEMA}".graph_index_nodes (
                id, toc_id, indexed_source_id, knowledge_base_id,
                source_id, node_id, title, depth, parent_node_id,
                text, line_num, meta
            ) VALUES (
                :id, :toc_id, :indexed_source_id, :kb_id,
                :source_id, :node_id, :title, :depth, :parent_node_id,
                :text, :line_num, CAST(:meta AS jsonb)
            )
        """)

        emb_stmt = text(f"""
            INSERT INTO "{AI_SCHEMA}".embeddings (
                item_id, item_table, indexed_source_id,
                knowledge_base_id, source_id,
                embedding_model, dims, embedding
            ) VALUES (
                :item_id, 'graph_index_nodes', :indexed_source_id,
                :kb_id, :source_id,
                :embedding_model, :dims, CAST(:embedding AS vector)
            )
        """)

        dims_seen: int | None = None
        node_row_ids: list[str] = []
        for node in nodes:
            node_row_id = str(uuid.uuid4())
            node_row_ids.append(node_row_id)
            embedding = node.get("embedding")

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

            if embedding and embedding_model:
                dims = len(embedding)
                dims_seen = dims
                embedding_str = f"[{','.join(str(x) for x in embedding)}]"
                self.session.execute(
                    emb_stmt,
                    {
                        "item_id": node_row_id,
                        "indexed_source_id": indexed_source_id,
                        "kb_id": self.kb_id,
                        "source_id": source_id,
                        "embedding_model": embedding_model,
                        "dims": dims,
                        "embedding": embedding_str,
                    },
                )

        if dims_seen is not None:
            ensure_embedding_index(self.session, AI_SCHEMA, dims_seen)

        logger.info(
            "Stored %d graph_index nodes for toc %s in KB %s",
            len(nodes),
            toc_id,
            self.kb_id,
        )
        return len(nodes), node_row_ids

    def update_node_meta(self, toc_id: str, node_id: str, meta: dict) -> None:
        """Update meta JSONB for a single node."""
        self.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".graph_index_nodes
                SET meta = CAST(:meta AS jsonb)
                WHERE toc_id = :toc_id AND node_id = :node_id
            """),
            {
                "toc_id": toc_id,
                "node_id": node_id,
                "meta": json.dumps(meta),
            },
        )

    def update_node_enrichment_error(self, toc_id: str, node_id: str, error: str | None) -> None:
        """Set or clear the enrichment_error for a single node."""
        self.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".graph_index_nodes
                SET enrichment_error = :error
                WHERE toc_id = :toc_id AND node_id = :node_id
            """),
            {"toc_id": toc_id, "node_id": node_id, "error": error},
        )

    def update_node_embedding(
        self,
        toc_id: str,
        node_id: str,
        embedding: list[float],
        embedding_model: str | None = None,
    ) -> None:
        """Upsert embedding for a single node into ai.embeddings."""
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        dims = len(embedding)

        # Look up the row id and FK columns for this node
        row = self.session.execute(
            text(f"""
                SELECT id, indexed_source_id, knowledge_base_id, source_id
                FROM "{AI_SCHEMA}".graph_index_nodes
                WHERE toc_id = :toc_id AND node_id = :node_id
            """),
            {"toc_id": toc_id, "node_id": node_id},
        ).fetchone()

        if not row:
            logger.warning(
                "update_node_embedding: node not found toc=%s node=%s",
                toc_id,
                node_id,
            )
            return

        item_id = str(row[0])
        model = embedding_model or "unknown"

        self.session.execute(
            text(f"""
                INSERT INTO "{AI_SCHEMA}".embeddings (
                    item_id, item_table, indexed_source_id,
                    knowledge_base_id, source_id,
                    embedding_model, dims, embedding
                ) VALUES (
                    :item_id, 'graph_index_nodes', :indexed_source_id,
                    :kb_id, :source_id,
                    :embedding_model, :dims, CAST(:embedding AS vector)
                )
                ON CONFLICT (item_id, embedding_model)
                DO UPDATE SET embedding = CAST(:embedding AS vector),
                             dims = :dims
            """),
            {
                "item_id": item_id,
                "indexed_source_id": str(row[1]) if row[1] else None,
                "kb_id": str(row[2]),
                "source_id": str(row[3]),
                "embedding_model": model,
                "dims": dims,
                "embedding": embedding_str,
            },
        )

    def get_toc_outline(self, toc_ids: list[str], limit: int) -> dict[str, dict]:
        """Fetch title-only outlines for whole documents.

        Deliberately not ``get_all_nodes_for_toc``, which selects every node's
        full text — retrieval wants the structure, not the document.

        Paged in SQL rather than in Python: this runs on the search path, and
        a 500-section document would otherwise ship 500 rows to render the
        first ``limit`` of them. ``total_nodes`` still reports the real count
        so the caller can say how many sections it left out.

        Returns:
            {toc_id: {"doc_name", "source_id", "total_nodes", "nodes":
            [{node_id, title, depth}]}}, nodes in document order.
        """
        if not toc_ids:
            return {}

        result = self.session.execute(
            text(f"""
                WITH ranked AS (
                    SELECT n.toc_id, n.node_id, n.title, n.depth,
                           row_number() OVER (
                               PARTITION BY n.toc_id ORDER BY n.node_id ASC
                           ) AS rn,
                           count(*) OVER (PARTITION BY n.toc_id) AS total_nodes
                    FROM "{AI_SCHEMA}".{self.NODES_TABLE} n
                    WHERE n.toc_id = ANY(:toc_ids)
                      AND n.knowledge_base_id = :kb_id
                )
                SELECT r.toc_id, r.node_id, r.title, r.depth, r.total_nodes,
                       t.doc_name, t.source_id
                FROM ranked r
                JOIN "{AI_SCHEMA}".{self.TOC_TABLE} t ON t.id = r.toc_id
                WHERE r.rn <= :limit
                ORDER BY r.toc_id, r.node_id ASC
            """),
            # The list is passed to the driver rather than hand-built into a
            # "{a,b}" literal: a toc_id containing a comma or brace would
            # otherwise produce a malformed array and a DataError.
            # knowledge_base_id is filtered here — not only at the caller —
            # because the outline item is stamped with the searching KB's id,
            # so a toc from another KB would be silently mislabelled.
            {
                "toc_ids": [uuid.UUID(tid) for tid in toc_ids],
                "kb_id": self.kb_id,
                "limit": limit,
            },
        )

        outlines: dict[str, dict] = {}
        for row in result:
            toc_id = str(row[0])
            outline = outlines.setdefault(
                toc_id,
                {
                    "doc_name": row[5] or "",
                    # str(None) would be the literal "None" and a phantom group.
                    "source_id": str(row[6]) if row[6] else None,
                    "total_nodes": int(row[4] or 0),
                    "nodes": [],
                },
            )
            outline["nodes"].append(
                {"node_id": row[1], "title": row[2] or "", "depth": row[3] or 0}
            )

        return outlines

    def count_nodes(self, indexed_source_id: str | None = None) -> int:
        """Total node count for this KB, optionally scoped to one source."""
        sql = f'SELECT COUNT(*) FROM "{AI_SCHEMA}".{self.NODES_TABLE} WHERE knowledge_base_id = :kb_id'
        params = {"kb_id": self.kb_id}
        if indexed_source_id:
            sql += " AND indexed_source_id = :sid"
            params["sid"] = indexed_source_id
        return int(self.session.execute(text(sql), params).scalar() or 0)

    def get_all_nodes_for_toc(self, toc_id: str) -> list[dict]:
        """Fetch all nodes for one ToC (used during enrichment)."""
        result = self.session.execute(
            text(f"""
                SELECT id, toc_id, node_id, title, text, depth,
                       parent_node_id, line_num, meta, source_id,
                       enrichment_error, indexed_source_id
                FROM "{AI_SCHEMA}".graph_index_nodes
                WHERE toc_id = :toc_id
                ORDER BY node_id ASC
            """),
            {"toc_id": toc_id},
        )

        nodes = []
        for row in result:
            nodes.append(
                {
                    "id": str(row[0]),
                    "toc_id": str(row[1]),
                    "node_id": row[2],
                    "title": row[3],
                    "text": row[4],
                    "depth": row[5],
                    "parent_node_id": row[6],
                    "line_num": row[7],
                    "meta": row[8] or {},
                    "source_id": str(row[9]),
                    "enrichment_error": row[10],
                    "indexed_source_id": str(row[11]) if row[11] else None,
                }
            )

        return nodes

    def get_enrichment_error_counts(self) -> dict[str, dict[str, int]]:
        """Return per-indexed-source error counts: {indexed_source_id: {total: N, failed: N}}."""
        result = self.session.execute(
            text(f"""
                SELECT indexed_source_id,
                       COUNT(*) AS total,
                       COUNT(enrichment_error) AS failed
                FROM "{AI_SCHEMA}".graph_index_nodes
                WHERE knowledge_base_id = :kb_id
                  AND indexed_source_id IS NOT NULL
                GROUP BY indexed_source_id
            """),
            {"kb_id": self.kb_id},
        )
        return {str(row[0]): {"total": row[1], "failed": row[2]} for row in result}
