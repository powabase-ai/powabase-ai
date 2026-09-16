"""Every lock a partition move waits for is bounded.

Same database and scratch schema as ``test_pg_bm25_live``.
"""

from __future__ import annotations

import threading
import time

from sqlalchemy import text

from agentic_project_service.services import pg_bm25_index as pgb
from tests.pg_search import test_pg_bm25_live as live
from tests.pg_search.test_pg_bm25_live import KB_A, KB_A_DOCS, KB_B, SCHEMA, _rows_in

migration = live.migration
engine = live.engine
scratch_schema = live.scratch_schema
session = live.session


def _run_in_thread(fn, timeout):
    outcome: dict = {}

    def target():
        started = time.monotonic()
        try:
            outcome["result"] = fn()
        except Exception as exc:
            outcome["error"] = exc
        outcome["seconds"] = time.monotonic() - started

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    return thread, outcome


def test_preparing_the_partition_gives_up_behind_a_writer_of_a_referenced_table(
    engine, session, monkeypatch
):
    """Adding the clone's foreign keys takes SHARE ROW EXCLUSIVE on the tables
    they reference, which waits for every open write there. Without a lock
    timeout the move sat behind one open transaction for as long as it stayed
    open, holding the item table's build lock all the while."""
    monkeypatch.setattr(pgb, "MOVE_LOCK_TIMEOUT_MS", 500)
    holder = engine.connect()
    holder.execute(
        text(
            f"UPDATE {SCHEMA}.knowledge_bases SET retrieval_config = retrieval_config "
            "WHERE id = CAST(:kb AS uuid)"
        ),
        {"kb": KB_B},
    )
    try:
        thread, outcome = _run_in_thread(lambda: pgb.create_partition(engine, KB_A, "chunks"), 20)
        assert not thread.is_alive(), "the move waited behind the writer without a bound"
    finally:
        holder.rollback()
        holder.close()
        thread.join(timeout=30)

    assert "error" in outcome, outcome
    assert pgb.is_lock_conflict(outcome["error"])
    assert outcome["seconds"] < 5
    # Nothing moved, and the next attempt goes through.
    assert _rows_in(session, "chunks_default", KB_A) == len(KB_A_DOCS)
    assert pgb.create_partition(engine, KB_A, "chunks")["rows_moved"] == len(KB_A_DOCS)
