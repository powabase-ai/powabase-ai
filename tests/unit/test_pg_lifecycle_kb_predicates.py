"""I10: statements that find a knowledge base's rows by a narrower key also name the KB.

With the item tables partitioned by knowledge base, a statement that filters
only by ``indexed_source_id`` or ``toc_id`` cannot be pruned: it plans and
locks every partition (a generic plan locks them all), which costs planning
time and lock-table entries per partition. Adding ``knowledge_base_id`` prunes
it to the one partition that can hold the rows. The predicate never changes
the result: these ids only ever belong to the store's own knowledge base.
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import MagicMock

from agentic_project_service.services.base_vector_store import BasePgVectorStore
from agentic_project_service.services.full_document_store import FullDocumentStore
from agentic_project_service.services.graph_index_store import GraphIndexStore
from agentic_project_service.services.knowledge_store import PgVectorKnowledgeStore

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


def _spy():
    session = MagicMock()
    session.calls = []

    def execute(statement, params=None):
        session.calls.append((getattr(statement, "text", str(statement)), dict(params or {})))
        result = MagicMock()
        result.rowcount = 0
        result.fetchone.return_value = None
        result.__iter__.return_value = iter([])
        return result

    session.execute.side_effect = execute
    return session


def _assert_kb_scoped(session, table):
    statements = [(sql, params) for sql, params in session.calls if table in sql]
    assert statements, session.calls
    for sql, params in statements:
        assert re.search(r"knowledge_base_id\s*=\s*:kb_id", sql), sql
        assert params["kb_id"] == KB


def test_delete_chunks_is_scoped_to_the_kb():
    session = _spy()
    store = PgVectorKnowledgeStore(db_session=session, knowledge_base_id=KB)
    asyncio.run(store.delete_chunks("is-1"))
    _assert_kb_scoped(session, ".chunks")


def test_full_document_cleanup_is_scoped_to_the_kb():
    session = _spy()
    store = FullDocumentStore(db_session=session, knowledge_base_id=KB, storage=MagicMock())
    store.delete_by_indexed_source("is-1")
    _assert_kb_scoped(session, ".full_documents")


def test_graph_node_updates_and_reads_by_toc_are_scoped_to_the_kb():
    session = _spy()
    store = GraphIndexStore(db_session=session, knowledge_base_id=KB)
    store.update_node_meta("toc-1", "n1", {"a": 1})
    store.update_node_enrichment_error("toc-1", "n1", None)
    store.get_all_nodes_for_toc("toc-1")
    _assert_kb_scoped(session, "graph_index_nodes")


class _Chunks(BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


def test_fetch_items_by_ids_is_scoped_to_the_kb():
    session = _spy()
    store = _Chunks(db_session=session, knowledge_base_id=KB)
    asyncio.run(store._fetch_items_by_ids(["a", "b"]))
    _assert_kb_scoped(session, ".chunks")
