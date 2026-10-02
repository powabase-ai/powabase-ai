"""Statements that find a knowledge base's rows by a narrower key also name the KB.

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

from agentic.knowledge.models import RetrievedItem
from agentic_project_service.services.base_vector_store import BasePgVectorStore
from agentic_project_service.services.full_document_store import FullDocumentStore
from agentic_project_service.services.graph_index_node_store import GraphIndexNodeStore
from agentic_project_service.services.graph_index_store import GraphIndexStore
from agentic_project_service.services.knowledge_store import PgVectorKnowledgeStore
from agentic_project_service.services.page_index_store import PageIndexStore

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


# The retrieval-path lookups below run on every graph_index search, in the
# request's pooled connection. Unpruned, each one plans across every knowledge
# base's partition and opens all their indexes; the backend then keeps that
# relcache for the life of the connection. On a project with 512 graph_index
# KBs that was ~65 MB per connection, and a handful of them OOM-killed a 512 MiB
# Postgres in a loop.


def test_graph_expansion_node_lookups_are_scoped_to_the_kb():
    session = _spy()
    store = GraphIndexStore(db_session=session, knowledge_base_id=KB)
    store.get_nodes_by_ids([("toc-1", "0001"), ("toc-2", "0002")])
    store.get_children_by_parent_ids([("toc-1", "0001"), ("toc-2", "0002")])
    assert len(session.calls) == 2, session.calls
    _assert_kb_scoped(session, "graph_index_nodes")


def test_page_index_node_lookups_are_scoped_to_the_kb():
    session = _spy()
    store = PageIndexStore(db_session=session, knowledge_base_id=KB)
    store.get_nodes_by_ids([("toc-1", "0001")])
    store.get_children_by_parent_ids([("toc-1", "0001")])
    assert len(session.calls) == 2, session.calls
    _assert_kb_scoped(session, "page_index_nodes")


def test_or_of_node_pairs_cannot_escape_the_kb_predicate():
    # ``kb AND a OR b`` would scope only the first pair: the pairs must be
    # parenthesised as a group, or every pair after the first reads all partitions.
    session = _spy()
    store = GraphIndexStore(db_session=session, knowledge_base_id=KB)
    store.get_nodes_by_ids([("toc-1", "0001"), ("toc-2", "0002")])
    store.get_children_by_parent_ids([("toc-1", "0001"), ("toc-2", "0002")])
    for sql, _ in session.calls:
        flat = " ".join(sql.split())
        assert re.search(r"knowledge_base_id = :kb_id AND \(\(.*\) OR \(.*\)\)$", flat), flat


def test_graph_node_result_resolution_is_scoped_to_the_kb():
    session = _spy()
    store = GraphIndexNodeStore(db_session=session, knowledge_base_id=KB)
    item = RetrievedItem(
        item_id="8a1556b9-75cb-4cd3-83a7-5207e591cffb",
        text="t",
        score=1.0,
        source_id="s",
        meta={},
    )
    store._resolve_results([item])
    statements = [sql for sql, _ in session.calls if "graph_index_nodes" in sql]
    assert statements, session.calls
    for sql in statements:
        assert re.search(r"n\.knowledge_base_id\s*=\s*:kb_id", sql), sql
    assert all(params["kb_id"] == KB for sql, params in session.calls if "graph_index_nodes" in sql)
