"""A re-index's cleanup waits only for moves on the tables it deletes from.

``index_source`` clears a source's old rows from every item table before it
writes new ones. It used to take all three tables' move gates in one
statement: while a graph_index run held graph_index_nodes' gate through its
LLM stages and a graph move queued for it, every chunk re-index cleanup took
chunks' gate, then queued behind the waiting move -- for its whole 30 s wait,
holding up chunk moves meanwhile.

Same database and scratch schema as ``test_pg_bm25_live``.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from agentic_project_service.services import pg_bm25_index as pgb
from agentic_project_service.tasks import indexing
from tests.pg_search import test_pg_bm25_live as live
from tests.pg_search.test_pg_bm25_live import KB_A, KB_B, SCHEMA, SOURCE_1

migration = live.migration
engine = live.engine
scratch_schema = live.scratch_schema
session = live.session

INDEXED_SOURCE = str(uuid.uuid4())


@pytest.fixture(autouse=True)
def cleanup_schema(engine, scratch_schema, monkeypatch):
    """What the cleanup reads beyond the scratch tables: the source columns and the ToC."""
    monkeypatch.setattr(indexing, "AI_SCHEMA", SCHEMA)
    with engine.connect() as conn:
        conn.execute(text(f"ALTER TABLE {SCHEMA}.full_documents ADD COLUMN indexed_source_id uuid"))
        conn.execute(
            text(f"ALTER TABLE {SCHEMA}.graph_index_nodes ADD COLUMN indexed_source_id uuid")
        )
        conn.execute(
            text(
                f"CREATE TABLE {SCHEMA}.graph_index_toc "
                "(id uuid PRIMARY KEY DEFAULT gen_random_uuid(), indexed_source_id uuid)"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, indexed_source_id, source_id, text) "
                "SELECT CAST(:kb AS uuid), CAST(:is_id AS uuid), CAST(:src AS uuid), 'alt ' || g "
                "FROM generate_series(1, 5) g"
            ),
            {"kb": KB_A, "is_id": INDEXED_SOURCE, "src": SOURCE_1},
        )
        conn.commit()


def _delete_chunks(session):
    session.execute(
        text(
            f"DELETE FROM {SCHEMA}.chunks WHERE knowledge_base_id = CAST(:kb AS uuid) "
            "AND indexed_source_id = CAST(:is_id AS uuid)"
        ),
        {"kb": KB_A, "is_id": INDEXED_SOURCE},
    )


def _graph_run_holding_its_gate(engine):
    conn = engine.connect()
    pgb.hold_move_gate_shared(conn, "graph_index_nodes")
    conn.execute(
        text(
            f"INSERT INTO {SCHEMA}.graph_index_nodes (knowledge_base_id, source_id, title, text) "
            "VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), 'Titel', 'Knoten')"
        ),
        {"kb": KB_B, "src": SOURCE_1},
    )
    return conn


def _wait_for_a_queued_exclusive_gate(engine, seconds=10.0) -> bool:
    deadline = time.monotonic() + seconds
    with engine.connect() as probe:
        while time.monotonic() < deadline:
            waiting = probe.execute(
                text(
                    "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
                    "AND mode = 'ExclusiveLock' AND NOT granted"
                )
            ).scalar()
            probe.rollback()
            if waiting:
                return True
            time.sleep(0.05)
    return False


def test_a_waiting_graph_move_does_not_stall_a_chunk_sources_cleanup(engine, monkeypatch):
    monkeypatch.setattr(pgb, "MOVE_GATE_WAIT_SECONDS", 6.0)
    graph_run = _graph_run_holding_its_gate(engine)
    result: dict = {}

    def move_graph_nodes():
        try:
            result["outcome"] = pgb.create_partition(engine, KB_A, "graph_index_nodes")
        except Exception as exc:
            result["error"] = exc

    mover = threading.Thread(target=move_graph_nodes, daemon=True)
    deleted: list[str] = []
    try:
        mover.start()
        assert _wait_for_a_queued_exclusive_gate(engine), "the graph move never queued"

        with Session(engine) as cleanup:
            started = time.monotonic()
            removed = indexing._clear_source_item_rows(
                cleanup,
                KB_A,
                INDEXED_SOURCE,
                {
                    "chunks": lambda: _delete_chunks(cleanup),
                    "full_documents": lambda: deleted.append("full_documents"),
                    "graph_index_nodes": lambda: deleted.append("graph_index_nodes"),
                },
            )
            took = time.monotonic() - started
        assert mover.is_alive(), "the move stopped waiting before the cleanup was measured"
    finally:
        graph_run.rollback()
        graph_run.close()
        mover.join(timeout=30)

    assert took < 1.0, f"the cleanup waited {took:.2f} s behind the graph move"
    assert len(removed["chunks"]) == 5
    assert removed["full_documents"] == [] and removed["graph_index_nodes"] == []
    assert deleted == []
    with engine.connect() as conn:
        left = conn.execute(
            text(
                f"SELECT count(*) FROM {SCHEMA}.chunks "
                "WHERE indexed_source_id = CAST(:is_id AS uuid)"
            ),
            {"is_id": INDEXED_SOURCE},
        ).scalar()
    assert left == 0


def test_a_graph_sources_cleanup_still_waits_for_a_graph_move(engine, session):
    """Its rows are in the moving table, so it takes that table's gate."""
    with engine.connect() as conn:
        conn.execute(
            text(
                f"INSERT INTO {SCHEMA}.graph_index_nodes "
                "(knowledge_base_id, indexed_source_id, source_id, title, text) "
                "VALUES (CAST(:kb AS uuid), CAST(:is_id AS uuid), CAST(:src AS uuid), 't', 'x')"
            ),
            {"kb": KB_A, "is_id": INDEXED_SOURCE, "src": SOURCE_1},
        )
        conn.commit()
    gate_holder = engine.connect()
    gate_holder.execute(
        text("SELECT pg_advisory_lock(hashtextextended(:r, 0))"),
        {"r": pgb.move_gate_relation("graph_index_nodes")},
    )
    released_at: list[float] = []

    def release_later():
        time.sleep(1.0)
        released_at.append(time.monotonic())
        gate_holder.execute(
            text("SELECT pg_advisory_unlock(hashtextextended(:r, 0))"),
            {"r": pgb.move_gate_relation("graph_index_nodes")},
        )
        gate_holder.commit()

    releaser = threading.Thread(target=release_later, daemon=True)
    try:
        with Session(engine) as cleanup:
            releaser.start()
            removed = indexing._clear_source_item_rows(
                cleanup,
                KB_A,
                INDEXED_SOURCE,
                {
                    "chunks": lambda: _delete_chunks(cleanup),
                    "graph_index_nodes": lambda: cleanup.execute(
                        text(
                            f"DELETE FROM {SCHEMA}.graph_index_nodes "
                            "WHERE indexed_source_id = CAST(:is_id AS uuid)"
                        ),
                        {"is_id": INDEXED_SOURCE},
                    ),
                },
            )
            finished = time.monotonic()
    finally:
        releaser.join(timeout=10)
        gate_holder.close()

    assert len(removed["graph_index_nodes"]) == 1
    assert released_at and finished >= released_at[0]


def test_a_cleanup_that_races_the_kbs_attach_still_deletes_the_sources_rows(engine, monkeypatch):
    """A probe planned while the move holds DEFAULT for its ATTACH prunes to
    DEFAULT only, then runs after the commit and finds DEFAULT empty: it said
    "no rows", the delete was skipped and the re-index duplicated the rows."""
    real_lock = pgb._lock_default_exclusively
    calls: list[str] = []
    in_attach = threading.Event()

    def pause_inside_the_attach(conn, item_table, wait_seconds):
        real_lock(conn, item_table, wait_seconds)
        calls.append(item_table)
        if len(calls) == 2:  # the fence's lock first, then the ATTACH's
            in_attach.set()
            time.sleep(1.5)

    monkeypatch.setattr(pgb, "_lock_default_exclusively", pause_inside_the_attach)
    result: dict = {}

    def move():
        try:
            result["outcome"] = pgb.create_partition(engine, KB_A, "chunks", allow_row_move=True)
        except Exception as exc:
            result["error"] = exc

    mover = threading.Thread(target=move, daemon=True)
    mover.start()
    try:
        assert in_attach.wait(timeout=20), result
        time.sleep(0.1)
        with Session(engine) as cleanup:
            removed = indexing._clear_source_item_rows(
                cleanup,
                KB_A,
                INDEXED_SOURCE,
                {
                    "chunks": lambda: _delete_chunks(cleanup),
                    "full_documents": lambda: None,
                    "graph_index_nodes": lambda: None,
                },
            )
    finally:
        mover.join(timeout=30)

    assert "outcome" in result, result
    assert pgb.partition_exists(engine, KB_A, "chunks") is True
    with engine.connect() as conn:
        left = conn.execute(
            text(
                f"SELECT count(*) FROM {SCHEMA}.chunks "
                "WHERE indexed_source_id = CAST(:is_id AS uuid)"
            ),
            {"is_id": INDEXED_SOURCE},
        ).scalar()
    assert left == 0, f"the cleanup skipped the delete: {left} old rows of the source remain"
    assert len(removed["chunks"]) == 5
