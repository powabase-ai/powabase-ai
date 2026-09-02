"""Database-backed tests for GraphIndexStore.get_toc_outline.

The expansion unit tests fake the store, so nothing there exercises this
query — and it is the kind that unit fakes cannot vouch for: a ``uuid[]``
cast that rejects non-uuid text, a window function that pages per document,
and a count that has to survive that paging.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.services.graph_index_store import GraphIndexStore


@pytest.fixture
def toc_with_nodes(app, test_source, test_knowledge_base):
    """Insert one ToC and a handful of nodes, deliberately out of order.

    ``indexed_source_id`` is left NULL — it is nullable on both tables, and
    the outline query neither selects nor joins on it.
    """
    toc_id = str(uuid.uuid4())
    ids = {"kb_id": test_knowledge_base["id"], "source_id": test_source["id"]}
    with app.app_context():
        db.session.execute(
            text("""
                INSERT INTO "ai".graph_index_toc
                    (id, knowledge_base_id, source_id,
                     doc_name, doc_description, structure)
                VALUES (:id, :kb_id, :sid, :doc_name, '', CAST('[]' AS jsonb))
                """),
            {
                "id": toc_id,
                "kb_id": ids["kb_id"],
                "sid": ids["source_id"],
                "doc_name": "Master Services Agreement",
            },
        )
        # Inserted 0003, 0001, 0002 so row order differs from document order.
        for node_id, title, depth in [
            ("0003", "Scope of Indemnity", 1),
            ("0001", "Definitions", 0),
            ("0002", "Indemnification", 0),
        ]:
            db.session.execute(
                text("""
                    INSERT INTO "ai".graph_index_nodes
                        (id, toc_id, knowledge_base_id,
                         source_id, node_id, title, depth, text, meta)
                    VALUES (:id, :toc_id, :kb_id, :sid, :node_id,
                            :title, :depth, :text, CAST('{}' AS jsonb))
                    """),
                {
                    "id": str(uuid.uuid4()),
                    "toc_id": toc_id,
                    "kb_id": ids["kb_id"],
                    "sid": ids["source_id"],
                    "node_id": node_id,
                    "title": title,
                    "depth": depth,
                    "text": f"body of {node_id}",
                },
            )
        db.session.commit()
    return {"toc_id": toc_id, **ids}


@pytest.mark.integration
class TestGetTocOutline:
    def test_returns_nodes_in_document_order_with_document_metadata(self, app, toc_with_nodes):
        with app.app_context():
            store = GraphIndexStore(
                db_session=db.session, knowledge_base_id=toc_with_nodes["kb_id"]
            )
            outlines = store.get_toc_outline([toc_with_nodes["toc_id"]], 200)

        outline = outlines[toc_with_nodes["toc_id"]]
        assert [n["node_id"] for n in outline["nodes"]] == ["0001", "0002", "0003"]
        assert outline["nodes"][2] == {
            "node_id": "0003",
            "title": "Scope of Indemnity",
            "depth": 1,
        }
        assert outline["doc_name"] == "Master Services Agreement"
        assert outline["source_id"] == toc_with_nodes["source_id"]

    def test_limit_pages_the_query_but_total_counts_every_section(self, app, toc_with_nodes):
        """The renderer's "N more sections" marker is only honest if the count
        survives paging."""
        with app.app_context():
            store = GraphIndexStore(
                db_session=db.session, knowledge_base_id=toc_with_nodes["kb_id"]
            )
            outlines = store.get_toc_outline([toc_with_nodes["toc_id"]], 2)

        outline = outlines[toc_with_nodes["toc_id"]]
        assert [n["node_id"] for n in outline["nodes"]] == ["0001", "0002"]
        assert outline["total_nodes"] == 3

    def test_unknown_toc_id_returns_no_entry(self, app, toc_with_nodes):
        with app.app_context():
            store = GraphIndexStore(
                db_session=db.session, knowledge_base_id=toc_with_nodes["kb_id"]
            )
            outlines = store.get_toc_outline([str(uuid.uuid4())], 200)

        assert outlines == {}

    def test_empty_selection_does_not_query(self, app, toc_with_nodes):
        with app.app_context():
            store = GraphIndexStore(
                db_session=db.session, knowledge_base_id=toc_with_nodes["kb_id"]
            )
            assert store.get_toc_outline([], 200) == {}
