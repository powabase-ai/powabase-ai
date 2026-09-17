"""Indexing takes the move gate before it touches an item table.

The gate is a per-item-table advisory lock: indexing takes it shared and
transaction-scoped, a partition move takes it exclusively. For it to keep a
re-index from holding DEFAULT into a move's lock tries, it has to come *first*
in every transaction indexing opens on the item tables -- before the read of
the ids it is about to delete, and before the rows a store writes.
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_project_service.services import pg_bm25_index as pgb
from agentic_project_service.services.full_document_store import FullDocumentStore
from agentic_project_service.services.graph_index_store import GraphIndexStore
from agentic_project_service.services.knowledge_store import PgVectorKnowledgeStore
from agentic_project_service.tasks import indexing

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
SRC = "11111111-1111-4111-8111-111111111111"
IS_ID = "22222222-2222-4222-8222-222222222222"

_ITEM_TABLE = re.compile(r"\.(chunks|full_documents|graph_index_nodes|graph_index_toc)\b")
_GATE = "pg_advisory_xact_lock_shared"
# An existence probe the cleanup makes, alone in its transaction, before it
# takes a table's gate: it holds nothing into a move, so it needs no gate.
_PROBE = "SELECT EXISTS"


def _recording_session(source_rows_exist=None):
    session = MagicMock()
    session.log = []

    def execute(statement, params=None):
        sql = getattr(statement, "text", str(statement))
        if _GATE in sql:
            sql = f"{sql} -- gates: {' '.join((params or {}).get('relations', []))}"
        session.log.append(sql)
        result = MagicMock()
        result.rowcount = 0
        result.fetchall.return_value = []
        result.fetchone.return_value = None
        result.scalar.return_value = source_rows_exist if sql.startswith(_PROBE) else None
        result.__iter__.return_value = iter([])
        return result

    session.execute.side_effect = execute
    session.commit.side_effect = lambda: session.log.append("COMMIT")
    session.rollback.side_effect = lambda: session.log.append("ROLLBACK")
    return session


# The gate that covers each table a statement can touch; the ToC's rows are
# written with their nodes, under the nodes' gate.
_GATE_OF = {
    "chunks": "chunks",
    "full_documents": "full_documents",
    "graph_index_nodes": "graph_index_nodes",
    "graph_index_toc": "graph_index_nodes",
}


def _assert_gate_opens_every_transaction_on_an_item_table(log) -> set[str]:
    """Before each statement on an item table, its transaction has taken that
    table's gate. Returns every gate taken."""
    transaction: list[str] = []
    checked = 0
    taken_anywhere: set[str] = set()
    entries = [*log, "COMMIT"]
    for position, entry in enumerate(entries):
        if entry in ("COMMIT", "ROLLBACK"):
            transaction = []
            continue
        transaction.append(entry)
        if _GATE in entry:
            taken_anywhere.update(entry.split("-- gates: ", 1)[1].split())
            continue
        if entry.startswith(_PROBE) and transaction == [entry]:
            assert entries[position + 1] in ("COMMIT", "ROLLBACK"), entries[position : position + 2]
            continue
        for table in _ITEM_TABLE.findall(entry):
            held = {
                relation
                for sql in transaction
                if _GATE in sql
                for relation in sql.split("-- gates: ", 1)[1].split()
            }
            assert pgb.move_gate_relation(_GATE_OF[table]) in held, transaction
            checked += 1
    assert checked, log
    return taken_anywhere


def test_the_gate_names_only_the_tables_asked_for_in_name_order():
    sql, params = pgb.move_gate_shared_sql("graph_index_nodes")
    assert _GATE in sql
    assert params["relations"] == [pgb.move_gate_relation("graph_index_nodes")]
    _, params = pgb.move_gate_shared_sql(pgb.PARTITIONED_ITEM_TABLES)
    assert params["relations"] == [
        pgb.move_gate_relation(t) for t in sorted(pgb.PARTITIONED_ITEM_TABLES)
    ]
    with pytest.raises(ValueError):
        pgb.move_gate_shared_sql([])
    with pytest.raises(ValueError):
        pgb.move_gate_shared_sql("doc2json_documents")
    # Its own key: never the build lock that serialises moves with each other.
    for table in pgb.PARTITIONED_ITEM_TABLES:
        assert pgb.move_gate_relation(table) != pgb.partition_build_lock_relation(table)


def test_chunk_writes_take_the_gate_first():
    session = _recording_session()
    store = PgVectorKnowledgeStore(db_session=session, knowledge_base_id=KB)
    asyncio.run(store.delete_chunks(IS_ID))
    session.log.append("COMMIT")
    asyncio.run(store.store_chunks(IS_ID, [{"text": "t", "source_id": SRC}]))
    gates = _assert_gate_opens_every_transaction_on_an_item_table(session.log)
    assert gates == {pgb.move_gate_relation("chunks")}


def test_full_document_writes_take_the_gate_first():
    session = _recording_session()
    store = FullDocumentStore(db_session=session, knowledge_base_id=KB, storage=MagicMock())
    store.delete_by_indexed_source(IS_ID)
    gates = _assert_gate_opens_every_transaction_on_an_item_table(session.log)
    assert gates == {pgb.move_gate_relation("full_documents")}


def test_storing_a_full_document_takes_the_gate_first():
    session = _recording_session()
    store = FullDocumentStore(db_session=session, knowledge_base_id=KB, storage=MagicMock())
    store.store_full_document(
        indexed_source_id=IS_ID,
        source_id=SRC,
        summary="Zusammenfassung",
        summary_embedding=[0.1, 0.2, 0.3],
        full_text="Volltext",
    )
    gates = _assert_gate_opens_every_transaction_on_an_item_table(session.log)
    assert gates == {pgb.move_gate_relation("full_documents")}


def test_graph_writes_take_the_gate_first():
    session = _recording_session()
    store = GraphIndexStore(db_session=session, knowledge_base_id=KB)
    store.delete_by_indexed_source(IS_ID)
    store.store_toc(indexed_source_id=IS_ID, source_id=SRC, structure=[], doc_name="d")
    store.store_nodes(
        "toc-1",
        IS_ID,
        SRC,
        [{"node_id": "n1", "title": "t", "text": "x", "depth": 1, "meta": {}}],
    )
    gates = _assert_gate_opens_every_transaction_on_an_item_table(session.log)
    # Only its own table's: a graph_index run's transaction stays open through
    # its LLM stages, and must not hold off moves on the other item tables.
    assert gates == {pgb.move_gate_relation("graph_index_nodes")}


@pytest.fixture
def run_body(monkeypatch, request):
    session = _recording_session(source_rows_exist=getattr(request, "param", True))
    monkeypatch.setattr(indexing, "db", MagicMock(session=session))
    monkeypatch.setattr(
        indexing, "get_knowledge_base", lambda _id: {"indexing_config": {"strategy": "chunk_embed"}}
    )
    monkeypatch.setattr(indexing, "get_source", lambda _id: {"name": "s", "auto_metadata": {}})
    monkeypatch.setattr(
        indexing, "_get_indexed_source_snapshot", lambda _id: {"strategy": "chunk_embed"}
    )
    monkeypatch.setattr(indexing, "update_indexed_source_config_snapshot", lambda *_a: None)
    monkeypatch.setattr(indexing, "get_storage", MagicMock())
    monkeypatch.setattr(indexing, "get_text_derivative_content", lambda *_a: None)
    monkeypatch.setattr(indexing, "_fenced_mark_failed", MagicMock())
    knowledge_store = MagicMock()
    knowledge_store.return_value.delete_chunks = AsyncMock(return_value=0)
    monkeypatch.setattr(indexing, "PgVectorKnowledgeStore", knowledge_store)
    for name in ("PageIndexStore", "FullDocumentStore", "GraphIndexStore", "Doc2JSONStore"):
        monkeypatch.setattr(indexing, name, MagicMock())
    return session


def test_the_reindex_cleanup_takes_the_gate_before_reading_the_ids_it_deletes(run_body):
    indexing._run_index_body(
        knowledge_base_id=KB,
        source_id=SRC,
        indexed_source_id=IS_ID,
        task_id="task-1",
        provider_keys=None,
    )
    _assert_gate_opens_every_transaction_on_an_item_table(run_body.log)


def _run_cleanup(session):
    indexing._run_index_body(
        knowledge_base_id=KB,
        source_id=SRC,
        indexed_source_id=IS_ID,
        task_id="task-1",
        provider_keys=None,
    )
    return session.log


def _gates_per_transaction(log) -> list[set[str]]:
    transactions: list[set[str]] = [set()]
    for entry in log:
        if entry in ("COMMIT", "ROLLBACK"):
            transactions.append(set())
        elif _GATE in entry:
            transactions[-1].update(entry.split("-- gates: ", 1)[1].split())
    return [gates for gates in transactions if gates]


def test_the_reindex_cleanup_takes_one_tables_gate_per_transaction(run_body):
    """A graph move waiting for its gate while a graph_index run holds it must
    not stall the cleanup of a chunk source: a transaction that held chunks'
    gate while it queued for graph_index_nodes' held up chunk moves and every
    cleanup behind it, for the whole 30 s wait."""
    log = _run_cleanup(run_body)
    _assert_gate_opens_every_transaction_on_an_item_table(log)
    gates = _gates_per_transaction(log)
    assert all(len(held) == 1 for held in gates), gates
    assert set().union(*gates) == {pgb.move_gate_relation(t) for t in pgb.PARTITIONED_ITEM_TABLES}


@pytest.mark.parametrize("run_body", [False], indirect=True)
def test_the_reindex_cleanup_takes_no_gate_for_a_table_holding_none_of_the_sources_rows(
    run_body,
):
    log = _run_cleanup(run_body)
    assert not [entry for entry in log if _GATE in entry], log
    probes = [entry for entry in log if entry.startswith(_PROBE)]
    # Each table is probed twice, each probe alone in its transaction: a probe
    # planned while a move of this KB held DEFAULT for its ATTACH reads the
    # table's old layout and answers "no rows" wrongly; the second is planned
    # after that move committed.
    assert len(probes) == 6, probes
    for table in ("chunks", "full_documents", "graph_index_nodes"):
        assert len([p for p in probes if f".{table} " in p]) == 2, (table, probes)
