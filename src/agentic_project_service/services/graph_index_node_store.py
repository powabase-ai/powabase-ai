"""
GraphIndex Node Store — Search via BasePgVectorStore.

Inherits vector_search(), full_text_search(), and hybrid_search()
from BasePgVectorStore.  Overrides _resolve_results to enrich search
results with title and document metadata from the DB (the base class
only selects the JSONB ``meta`` column, which doesn't include the
separate ``title`` column).
"""

from agentic.knowledge.models import RetrievedItem
from sqlalchemy import text

from .base_vector_store import BasePgVectorStore


class GraphIndexNodeStore(BasePgVectorStore):
    TABLE = "graph_index_nodes"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"

    def _resolve_results(self, items: list[RetrievedItem]) -> list[RetrievedItem]:
        items = super()._resolve_results(items)
        if not items:
            return items

        item_ids = [item.item_id for item in items]
        result = self.session.execute(
            text(f"""
                SELECT n.id, n.title, n.node_id, n.toc_id, n.depth,
                       t.doc_name, t.doc_description
                FROM "{self.schema}".{self.TABLE} n
                JOIN "{self.schema}".graph_index_toc t ON t.id = n.toc_id
                WHERE n.id = ANY(CAST(:ids AS uuid[]))
            """),
            {"ids": "{" + ",".join(item_ids) + "}"},
        )

        extra: dict[str, dict] = {}
        for row in result:
            extra[str(row[0])] = {
                "title": row[1] or "",
                "node_id": row[2],
                "toc_id": str(row[3]),
                "depth": row[4],
                "doc_name": row[5] or "",
                "doc_description": row[6] or "",
            }

        for item in items:
            if item.item_id in extra:
                if item.meta is None:
                    item.meta = {}
                for k, v in extra[item.item_id].items():
                    if k not in item.meta:  # don't overwrite existing
                        item.meta[k] = v

        return items
