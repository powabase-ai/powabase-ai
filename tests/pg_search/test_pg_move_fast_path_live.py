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

import pytest
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


# ---------------------------------------------------------------------------
# An automatic ensure never moves rows
# ---------------------------------------------------------------------------


def _clone_keys_and_indexes(session, kb_id=KB_NEW) -> tuple[int, int]:
    relation = f"{SCHEMA}.{pgb.partition_name(kb_id, 'chunks')}"
    try:
        return tuple(
            session.execute(
                text(
                    "SELECT (SELECT count(*) FROM pg_constraint "
                    "        WHERE conrelid = to_regclass(:r) AND contype = 'f'), "
                    "       (SELECT count(*) FROM pg_index WHERE indrelid = to_regclass(:r))"
                ),
                {"r": relation},
            ).one()
        )
    finally:
        session.rollback()


def test_an_automatic_ensure_does_not_move_a_row_that_arrives_before_the_check(
    engine, session, monkeypatch
):
    """The same race as above, for an ensure dispatched at creation: the row is
    left where it is and the outcome says an operator has to move it."""
    _create_empty_kb(engine)
    real_lock = pgb._lock_default_exclusively
    arrived: list[bool] = []

    def lock_after_a_row_arrives(*args, **kwargs):
        if not arrived:
            arrived.append(True)
            with engine.connect() as conn:
                _insert_chunk(conn, KB_NEW, "gerade noch rechtzeitig")
                conn.commit()
        return real_lock(*args, **kwargs)

    monkeypatch.setattr(pgb, "_lock_default_exclusively", lock_after_a_row_arrives)

    outcome = pgb.ensure_bm25_index(KB_NEW, engine=engine, allow_row_move=False)

    assert outcome["status"] == "skipped"
    assert outcome["reason"] == "row_move_not_allowed"
    assert "rows_moved" not in outcome
    assert _rows_in(session, "chunks_default", KB_NEW) == 1
    assert pgb.partition_exists(session, KB_NEW, "chunks") is False
    assert _move_check_names(session) == []
    session.rollback()


def test_an_automatic_ensure_does_not_move_rows_that_arrive_while_it_waits_for_the_gate(
    engine, session, monkeypatch
):
    """Rows committed after the ensure found none, while it queued for the gate
    (its knowledge base's first sources being indexed, say), are seen under the
    gate and left alone."""
    _create_empty_kb(engine)
    real_acquire = pgb._acquire_move_gate

    def rows_arrive_first(conn, item_table):
        with engine.connect() as other:
            for n in range(50):
                _insert_chunk(other, KB_NEW, f"Quelle {n}")
            other.commit()
        return real_acquire(conn, item_table)

    monkeypatch.setattr(pgb, "_acquire_move_gate", rows_arrive_first)

    outcome = pgb.ensure_bm25_index(KB_NEW, engine=engine, allow_row_move=False)

    assert (outcome["status"], outcome["reason"]) == ("skipped", "row_move_not_allowed")
    assert _rows_in(session, "chunks_default", KB_NEW) == 50
    assert pgb.partition_exists(session, KB_NEW, "chunks") is False
    # The operator's build moves them.
    monkeypatch.setattr(pgb, "_acquire_move_gate", real_acquire)
    moved = pgb.ensure_bm25_index(KB_NEW, engine=engine)
    assert (moved["status"], moved["rows_moved"]) == ("ready", 50)


def test_a_failed_empty_attach_leaves_a_bare_clone_and_its_retry_never_holds_up_kb_readers(
    engine, session
):
    """Adding the clone's foreign keys takes SHARE ROW EXCLUSIVE on the tables
    they reference; dropping them takes ACCESS EXCLUSIVE there, which queues
    every reader of knowledge_bases behind it. So an attempt that gives up must
    leave no key behind, and its retry must not make a plain read of
    knowledge_bases wait."""
    _create_empty_kb(engine)

    # 1. A reader holding DEFAULT refuses the check's lock: the attempt gives up.
    default_reader = engine.connect()
    default_reader.execute(text(f"SELECT count(*) FROM {SCHEMA}.chunks_default"))
    try:
        with pytest.raises(Exception) as caught:
            pgb.ensure_bm25_index(KB_NEW, engine=engine, allow_row_move=False)
    finally:
        default_reader.rollback()
        default_reader.close()
    assert pgb.is_lock_conflict(caught.value), caught.value
    assert caught.value.bm25_move_step == "fence"
    assert _clone_keys_and_indexes(session) == (0, 0)

    # 2. A reader of knowledge_bases stays open throughout the retry, and plain
    # reads of knowledge_bases are timed while it runs.
    kb_reader = engine.connect()
    kb_reader.execute(text(f"SELECT count(*) FROM {SCHEMA}.knowledge_bases"))
    result: dict = {}

    def retry():
        try:
            result["outcome"] = pgb.ensure_bm25_index(KB_NEW, engine=engine, allow_row_move=False)
        except Exception as exc:
            result["error"] = exc

    waits: list[float] = []
    worker = threading.Thread(target=retry, daemon=True)
    try:
        worker.start()
        with engine.connect() as reader:
            while worker.is_alive():
                reader.execute(text("SET LOCAL lock_timeout = '10s'"))
                started = time.monotonic()
                reader.execute(text(f"SELECT count(*) FROM {SCHEMA}.knowledge_bases"))
                waits.append(time.monotonic() - started)
                reader.rollback()
                time.sleep(0.01)
        worker.join(timeout=60)
    finally:
        kb_reader.rollback()
        kb_reader.close()

    assert "error" not in result, result
    assert result["outcome"]["status"] == "ready"
    assert result["outcome"]["rows_moved"] == 0
    assert waits and max(waits) < 0.1, max(waits)


def test_a_stale_clone_with_foreign_keys_is_dropped_without_queueing_readers_of_kbs(
    engine, session
):
    """A clone an earlier layout left with its foreign keys: dropping it has to
    take ACCESS EXCLUSIVE on knowledge_bases. Behind a long transaction on
    knowledge_bases the prepare step gives up (retryable, named) instead of
    queueing every reader of the table for its lock timeout."""
    _create_empty_kb(engine)
    partition = pgb.partition_name(KB_NEW, "chunks")
    with engine.connect() as conn:
        conn.execute(
            text(
                f"CREATE TABLE {SCHEMA}.{partition} (LIKE {SCHEMA}.chunks_default "
                "INCLUDING DEFAULTS INCLUDING CONSTRAINTS)"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {SCHEMA}.{partition} ADD FOREIGN KEY (knowledge_base_id) "
                f"REFERENCES {SCHEMA}.knowledge_bases(id)"
            )
        )
        conn.commit()

    kb_reader = engine.connect()
    kb_reader.execute(text(f"SELECT count(*) FROM {SCHEMA}.knowledge_bases"))
    result: dict = {}

    def attempt():
        try:
            result["outcome"] = pgb.ensure_bm25_index(KB_NEW, engine=engine, allow_row_move=False)
        except Exception as exc:
            result["error"] = exc

    waits: list[float] = []
    worker = threading.Thread(target=attempt, daemon=True)
    try:
        worker.start()
        with engine.connect() as reader:
            while worker.is_alive():
                reader.execute(text("SET LOCAL lock_timeout = '10s'"))
                started = time.monotonic()
                reader.execute(text(f"SELECT count(*) FROM {SCHEMA}.knowledge_bases"))
                waits.append(time.monotonic() - started)
                reader.rollback()
                time.sleep(0.01)
        worker.join(timeout=60)
    finally:
        kb_reader.rollback()
        kb_reader.close()

    error = result.get("error")
    assert error is not None and pgb.is_lock_conflict(error), result
    assert error.bm25_move_step == "prepare"
    assert waits and max(waits) < 0.1, max(waits)

    # With the long transaction gone, the next attempt drops it and attaches.
    outcome = pgb.ensure_bm25_index(KB_NEW, engine=engine, allow_row_move=False)
    assert (outcome["status"], outcome["rows_moved"]) == ("ready", 0)


def test_a_role_statement_timeout_does_not_cancel_the_empty_attachs_scan_of_default(
    engine, session, monkeypatch
):
    """The fence's VALIDATE scans all of DEFAULT, which grows with every other
    knowledge base; a role's or database's statement_timeout must not cancel it."""
    from sqlalchemy import create_engine

    _create_empty_kb(engine)
    real_validate = pgb.default_move_check_validate_ddl
    monkeypatch.setattr(
        pgb,
        "default_move_check_validate_ddl",
        lambda *a, **k: f"{real_validate(*a, **k)}; SELECT pg_sleep(0.5)",
    )
    slow = create_engine(engine.url, connect_args={"options": "-c statement_timeout=200"})
    try:
        attached = pgb.create_partition(slow, KB_NEW, "chunks")
    finally:
        slow.dispose()

    assert attached["rows_moved"] == 0
    assert pgb.partition_exists(session, KB_NEW, "chunks") is True
