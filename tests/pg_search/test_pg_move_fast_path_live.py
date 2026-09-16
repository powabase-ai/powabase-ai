"""A knowledge base with no rows in DEFAULT gets its partition without a move.

Every new knowledge base is one: KB create dispatches the index build before
anything is indexed. The full move would take SHARE on the parent -- holding
every writer of the item table -- for a VALIDATE scan of the whole DEFAULT
partition, which on an upgraded project never shrinks. With no rows to move
there is nothing that lock protects, so the partition is attached with only
brief ACCESS EXCLUSIVE tries on DEFAULT.

Same database and scratch schema as ``test_pg_bm25_live``.
"""

from __future__ import annotations

import json
import threading
import time

from sqlalchemy import text

from agentic_project_service.services import pg_bm25_index as pgb
from tests.pg_search import test_pg_bm25_live as live
from tests.pg_search.test_pg_bm25_live import KB_B, SCHEMA, SOURCE_1, _move_check_names, _rows_in

migration = live.migration
engine = live.engine
scratch_schema = live.scratch_schema
session = live.session

KB_NEW = "7d444840-9dc0-11d1-b245-5ffdce74fad2"


def _create_empty_kb(engine, kb_id=KB_NEW):
    with engine.connect() as conn:
        conn.execute(
            text(
                f"INSERT INTO {SCHEMA}.knowledge_bases (id, indexing_config, retrieval_config) "
                "VALUES (CAST(:id AS uuid), CAST(:ix AS jsonb), CAST(:rx AS jsonb))"
            ),
            {
                "id": kb_id,
                "ix": json.dumps({"strategy": "chunk_embed"}),
                "rx": json.dumps({"method": "hybrid", "ts_language": "english"}),
            },
        )
        conn.commit()


def _insert_chunk(conn, kb_id, body="neu"):
    conn.execute(
        text(
            f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text) "
            "VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), :body)"
        ),
        {"kb": kb_id, "src": SOURCE_1, "body": body},
    )


def test_an_empty_knowledge_base_is_attached_while_writers_keep_writing(
    engine, session, monkeypatch
):
    """The scan of DEFAULT is held open for 1.5 s; a write of another knowledge
    base made meanwhile must not wait for it."""
    _create_empty_kb(engine)
    parent_locks: list[str] = []
    real_parent_lock = pgb.partition_lock_parent_ddl
    real_validate = pgb.default_move_check_validate_ddl
    validating = threading.Event()

    def spy_parent_lock(*args, **kwargs):
        parent_locks.append("taken")
        return real_parent_lock(*args, **kwargs)

    def slow_validate(*args, **kwargs):
        validating.set()
        time.sleep(1.5)
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(pgb, "partition_lock_parent_ddl", spy_parent_lock)
    monkeypatch.setattr(pgb, "default_move_check_validate_ddl", slow_validate)
    writer: dict = {}

    def write_meanwhile():
        validating.wait(timeout=30)
        time.sleep(0.3)
        started = time.monotonic()
        with engine.connect() as conn:
            _insert_chunk(conn, KB_B, "geschrieben waehrend der Pruefung")
            conn.commit()
        writer["seconds"] = time.monotonic() - started

    thread = threading.Thread(target=write_meanwhile, daemon=True)
    thread.start()
    outcome = pgb.ensure_bm25_index(KB_NEW, engine=engine)
    thread.join(timeout=30)

    assert outcome["status"] == "ready"
    assert outcome["rows_moved"] == 0
    assert writer["seconds"] < 0.5, writer
    assert parent_locks == []
    assert pgb.partition_exists(session, KB_NEW, "chunks") is True
    assert _move_check_names(session) == []
    # The new partition takes the knowledge base's writes from now on.
    with engine.connect() as conn:
        _insert_chunk(conn, KB_NEW)
        conn.commit()
    assert _rows_in(session, pgb.partition_name(KB_NEW, "chunks")) == 1
    assert _rows_in(session, "chunks_default", KB_NEW) == 0


def test_a_row_that_arrives_before_the_check_goes_up_falls_back_to_the_full_move(
    engine, session, monkeypatch
):
    """The emptiness test and the check are not atomic. A row of the knowledge
    base landing in DEFAULT between them makes the check fail to validate; the
    call then moves the rows the ordinary way instead of failing."""
    _create_empty_kb(engine)
    real_lock = pgb._lock_default_exclusively
    arrived: list[bool] = []

    def lock_after_a_row_arrives(*args, **kwargs):
        if not arrived:
            arrived.append(True)

            def insert():
                with engine.connect() as conn:
                    _insert_chunk(conn, KB_NEW, "gerade noch rechtzeitig")
                    conn.commit()

            writer = threading.Thread(target=insert, daemon=True)
            writer.start()
            writer.join(timeout=3)
            assert not writer.is_alive(), "the row could not land before the check went up"
        return real_lock(*args, **kwargs)

    monkeypatch.setattr(pgb, "_lock_default_exclusively", lock_after_a_row_arrives)

    outcome = pgb.ensure_bm25_index(KB_NEW, engine=engine)

    assert outcome["status"] == "ready"
    assert outcome["rows_moved"] == 1
    assert _rows_in(session, pgb.partition_name(KB_NEW, "chunks")) == 1
    assert _rows_in(session, "chunks_default", KB_NEW) == 0
    assert _move_check_names(session) == []
