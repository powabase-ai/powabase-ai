"""A bm25 index builds while the knowledge base's partition takes writes.

This is the one thing a stock pg_search 0.25.9 on Postgres 15 or 16 cannot do:
its ``CREATE INDEX CONCURRENTLY`` fails with XX000 "buffer ... is not owned by
resource owner" (the index left INVALID) or crashes the server, which
paradedb/paradedb#6211 fixes. Every other test here passes on the stock image,
so without this one CI could not tell the two apart.

The concurrent-build guard is overridden, so the build is attempted whatever
the server says about itself: on a stock image this test fails, on one with
the fix it passes.

Same database and scratch schema as ``test_pg_bm25_live``.
"""

from __future__ import annotations

import threading
import time

from sqlalchemy import text

from agentic_project_service.services import pg_bm25_index as pgb
from tests.pg_search import test_pg_bm25_live as live
from tests.pg_search.test_pg_bm25_live import KB_A, SCHEMA, _seed

migration = live.migration
engine = live.engine
scratch_schema = live.scratch_schema
session = live.session

ROWS = 200_000


def test_the_bm25_index_rebuilds_while_the_partition_takes_inserts(engine, session, monkeypatch):
    monkeypatch.setattr(pgb, "_concurrent_build_override", lambda: True)
    _seed(session, KB_A, ROWS)
    assert pgb.ensure_bm25_index(KB_A, engine=engine, allow_row_move=True)["status"] == "ready"
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(pgb.bm25_drop_ddl(KB_A, "chunks")))
    pgb.reset_pg_bm25_caches()

    stop = threading.Event()
    writes: dict = {"count": 0, "errors": []}

    def insert_through_the_parent():
        with engine.connect() as conn:
            while not stop.is_set():
                try:
                    conn.execute(
                        text(
                            f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, text) "
                            "SELECT CAST(:kb AS uuid), 'neue Wanderung ' || g "
                            "FROM generate_series(1, 20) g"
                        ),
                        {"kb": KB_A},
                    )
                    conn.commit()
                    writes["count"] += 1
                except Exception as exc:
                    conn.rollback()
                    writes["errors"].append(pgb.first_error_line(exc))
                    time.sleep(0.05)

    writer = threading.Thread(target=insert_through_the_parent, daemon=True)
    writer.start()
    try:
        time.sleep(0.2)
        outcome = pgb.ensure_bm25_index(KB_A, engine=engine)
    finally:
        stop.set()
        writer.join(timeout=30)

    assert outcome["status"] == "ready", outcome
    assert writes["count"] > 0
    assert writes["errors"] == []
    assert pgb.bm25_index_state(session, KB_A, "chunks") == "ready"
    session.rollback()
    with engine.connect() as conn:
        total = conn.execute(
            text(
                f"SELECT count(*) FROM {SCHEMA}.chunks WHERE knowledge_base_id = CAST(:kb AS uuid)"
            ),
            {"kb": KB_A},
        ).scalar()
    assert total == ROWS + 3 + 20 * writes["count"]
