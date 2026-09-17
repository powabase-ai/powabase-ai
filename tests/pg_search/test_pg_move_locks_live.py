"""Every lock a partition move waits for is bounded.

Same database and scratch schema as ``test_pg_bm25_live``.
"""

from __future__ import annotations

import threading
import time
import uuid

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


# ---------------------------------------------------------------------------
# A clone left behind by a failed move
# ---------------------------------------------------------------------------


def test_a_stale_clone_missing_a_column_added_since_is_recreated(engine, session):
    """An unattached clone is not a partition, so a column added to the parent
    later never reaches it. Reusing it made every later move fail for good
    (``INSERT has more expressions than target columns``)."""
    with engine.connect() as conn:
        conn.execute(text(pgb.partition_create_ddl(KB_A, "chunks")))
        conn.commit()
        conn.execute(text(f"ALTER TABLE {SCHEMA}.chunks ADD COLUMN rank integer DEFAULT 7"))
        conn.commit()

    moved = pgb.create_partition(engine, KB_A, "chunks")

    assert moved["rows_moved"] == len(KB_A_DOCS)
    partition = pgb.partition_name(KB_A, "chunks")
    ranks = session.execute(text(f"SELECT DISTINCT rank FROM {SCHEMA}.{partition}")).scalars()
    assert list(ranks) == [7]
    session.rollback()


def test_a_stale_clone_missing_a_check_added_since_is_recreated(engine, session):
    """A CHECK added to the parent later reaches DEFAULT but not the unattached
    clone, and ATTACH then refuses the clone for good ("missing constraint")."""
    with engine.connect() as conn:
        conn.execute(text(pgb.partition_create_ddl(KB_A, "chunks")))
        conn.execute(text(pgb.partition_check_ddl(KB_A, "chunks")))
        conn.commit()
        conn.execute(
            text(
                f"ALTER TABLE {SCHEMA}.chunks ADD CONSTRAINT text_not_blank "
                "CHECK (text IS NULL OR length(text) > 0)"
            )
        )
        conn.commit()

    moved = pgb.create_partition(engine, KB_A, "chunks")

    assert moved["rows_moved"] == len(KB_A_DOCS)
    partition = pgb.partition_name(KB_A, "chunks")
    assert pgb.partition_exists(session, KB_A, "chunks") is True
    checks = session.execute(
        text(
            "SELECT conname FROM pg_constraint WHERE conrelid = to_regclass(:r) AND contype = 'c'"
        ),
        {"r": f"{SCHEMA}.{partition}"},
    ).scalars()
    assert "text_not_blank" in list(checks)
    session.rollback()


def test_an_up_to_date_clone_is_kept(engine, session, monkeypatch):
    """Comparing CHECKs must not make every clone look stale: its own
    partition-bound check, and a copied check that is NOT VALID on DEFAULT, match."""
    with engine.connect() as conn:
        conn.execute(
            text(
                f"ALTER TABLE {SCHEMA}.chunks ADD CONSTRAINT text_not_blank "
                "CHECK (text IS NULL OR length(text) > 0) NOT VALID"
            )
        )
        conn.execute(text(pgb.partition_create_ddl(KB_A, "chunks")))
        conn.execute(text(pgb.partition_check_ddl(KB_A, "chunks")))
        conn.commit()
    dropped: list = []
    real_drop = pgb.partition_drop_ddl
    monkeypatch.setattr(
        pgb, "partition_drop_ddl", lambda *a, **k: dropped.append(a) or real_drop(*a, **k)
    )

    assert pgb.create_partition(engine, KB_A, "chunks")["rows_moved"] == len(KB_A_DOCS)
    assert dropped == []


def test_a_stale_clone_that_somehow_holds_rows_is_not_dropped(engine, session):
    """A clone is empty whenever no move is in flight (the move is one
    transaction). One that is not is left for a person to look at, never
    dropped with the rows in it."""
    partition = pgb.partition_name(KB_A, "chunks")
    with engine.connect() as conn:
        conn.execute(text(pgb.partition_create_ddl(KB_A, "chunks")))
        conn.execute(
            text(
                f"INSERT INTO {SCHEMA}.{partition} (knowledge_base_id, text) "
                "VALUES (CAST(:kb AS uuid), 'verloren')"
            ),
            {"kb": KB_A},
        )
        conn.commit()
        conn.execute(text(f"ALTER TABLE {SCHEMA}.chunks ADD COLUMN rank integer"))
        conn.commit()

    try:
        pgb.create_partition(engine, KB_A, "chunks")
    except Exception as exc:
        assert "rows" in str(exc)
    else:
        raise AssertionError("a non-empty stale clone was dropped or reused")
    assert _rows_in(session, partition) == 1


# ---------------------------------------------------------------------------
# Who counts as a long holder of DEFAULT
# ---------------------------------------------------------------------------


def _hold_default_in_a_transaction(engine, seconds, started):
    def run():
        with engine.connect() as conn:
            conn.execute(
                text(
                    f"SELECT count(*) FROM {SCHEMA}.chunks "
                    "WHERE knowledge_base_id = CAST(:kb AS uuid)"
                ),
                {"kb": KB_B},
            ).scalar()
            started.set()
            time.sleep(seconds)
            conn.rollback()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _spy_parent_lock(monkeypatch) -> list:
    taken: list = []
    real = pgb.partition_lock_parent_ddl
    monkeypatch.setattr(
        pgb, "partition_lock_parent_ddl", lambda *a, **k: taken.append(1) or real(*a, **k)
    )
    return taken


def test_an_ordinary_short_read_of_default_does_not_make_the_move_give_up(
    engine, session, monkeypatch
):
    """A request reading the item table for well under a second is not a long
    transaction. The pre-flight used to call anything older than its own
    ~0.16 s probe window long, so such a read refused the move outright."""
    monkeypatch.setattr(pgb, "DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS", 3.0)
    monkeypatch.setattr(pgb, "_long_holder_seconds", lambda: 5)
    started = threading.Event()
    reader = _hold_default_in_a_transaction(engine, 0.8, started)
    assert started.wait(timeout=10)

    moved = pgb.create_partition(engine, KB_A, "chunks")
    reader.join(timeout=10)

    assert moved["rows_moved"] == len(KB_A_DOCS)


def test_a_transaction_older_than_the_setting_makes_the_move_give_up_before_locking(
    engine, session, monkeypatch
):
    monkeypatch.setattr(pgb, "_long_holder_seconds", lambda: 1)
    taken = _spy_parent_lock(monkeypatch)
    started = threading.Event()
    reader = _hold_default_in_a_transaction(engine, 3.0, started)
    assert started.wait(timeout=10)
    time.sleep(1.2)
    try:
        pgb.create_partition(engine, KB_A, "chunks")
    except Exception as exc:
        error = exc
    else:
        error = None
    reader.join(timeout=10)

    assert error is not None and pgb.is_lock_conflict(error)
    assert taken == []
    assert _rows_in(session, "chunks_default", KB_A) == len(KB_A_DOCS)


def test_the_long_holder_threshold_is_the_registry_setting(monkeypatch):
    """Outside an application context there is no settings table to read, so
    the registry default applies; inside one, the stored value."""
    from agentic_project_service.services import settings_registry

    assert (
        pgb._long_holder_seconds()
        == settings_registry.SETTINGS_REGISTRY["BM25_MOVE_LONG_HOLDER_SECONDS"].default
    )
    monkeypatch.setattr(pgb, "_has_app_context", lambda: True)
    monkeypatch.setattr(settings_registry, "get_setting", lambda key: 42)
    assert pgb._long_holder_seconds() == 42


# ---------------------------------------------------------------------------
# A statement_timeout set on the role or database
# ---------------------------------------------------------------------------


def _engine_with_statement_timeout(engine, ms):
    from sqlalchemy import create_engine

    return create_engine(engine.url, connect_args={"options": f"-c statement_timeout={int(ms)}"})


def test_a_role_statement_timeout_does_not_cancel_the_move(engine, session, monkeypatch):
    """The service role carries no statement_timeout today, but anon and
    authenticated do on the production image, and one set later would kill a
    large move half-way with nothing to retry it."""
    slow = _engine_with_statement_timeout(engine, 200)
    real = pgb.partition_lock_default_ddl
    monkeypatch.setattr(
        pgb, "partition_lock_default_ddl", lambda t: f"{real(t)}; SELECT pg_sleep(0.5)"
    )
    try:
        moved = pgb.create_partition(slow, KB_A, "chunks")
    finally:
        slow.dispose()
    assert moved["rows_moved"] == len(KB_A_DOCS)


def test_the_concurrent_index_builds_run_without_a_statement_timeout(engine, session):
    slow = _engine_with_statement_timeout(engine, 200)
    seen: list = []

    def before_execute(conn, cursor, statement, parameters, context, executemany):
        if "INDEX CONCURRENTLY" in statement and statement.startswith("CREATE"):
            seen.append(cursor.connection.execute("SHOW statement_timeout").fetchone()[0])

    from sqlalchemy import event

    event.listen(slow, "before_cursor_execute", before_execute)
    try:
        with slow.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(
                text(f"CREATE INDEX chunks_src_idx ON {SCHEMA}.chunks_default (source_id)")
            )
        outcome = pgb.ensure_bm25_index(KB_A, engine=slow)
        with slow.connect() as conn:
            after = conn.execute(text("SHOW statement_timeout")).scalar()
    finally:
        event.remove(slow, "before_cursor_execute", before_execute)
        slow.dispose()

    assert outcome["status"] == "ready"
    assert len(seen) >= 2 and set(seen) == {"0"}, seen
    assert after == "200ms"


# ---------------------------------------------------------------------------
# What a failed move says about who was in the way
# ---------------------------------------------------------------------------


def test_a_move_that_gives_up_names_the_reader_that_refused_its_lock(
    engine, session, monkeypatch, caplog
):
    monkeypatch.setattr(pgb, "MOVE_CHECK_CLEANUP_WAIT_SECONDS", 0.0)
    real = pgb.partition_lock_default_ddl
    reader: dict = {}

    def arrive_after_the_parent_lock(item_table):
        conn = engine.connect()
        reader["pid"] = conn.execute(text("SELECT pg_backend_pid()")).scalar()
        conn.execute(text(f"SELECT count(*) FROM {SCHEMA}.chunks_default")).scalar()
        reader["conn"] = conn
        return real(item_table)

    monkeypatch.setattr(pgb, "partition_lock_default_ddl", arrive_after_the_parent_lock)
    try:
        with caplog.at_level("WARNING"):
            try:
                pgb.create_partition(engine, KB_A, "chunks")
            except Exception as exc:
                error = exc
            else:
                error = None
    finally:
        if "conn" in reader:
            reader["conn"].rollback()
            reader["conn"].close()

    assert error is not None and pgb.is_lock_conflict(error)
    assert error.bm25_move_step == "attach"
    refusing = [h for h in error.bm25_lock_holders if h["pid"] == reader["pid"] and h["granted"]]
    assert refusing and refusing[0]["mode"] == "AccessShareLock", error.bm25_lock_holders
    message = next(r.getMessage() for r in caplog.records if "gave up at step" in r.getMessage())
    assert f"'pid': {reader['pid']}" in message


# ---------------------------------------------------------------------------
# The tables the partition's foreign keys reference
# ---------------------------------------------------------------------------


def _reference_indexed_sources(engine) -> list[str]:
    """chunks -> indexed_sources, as on the real schema; two unrelated sources."""
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    with engine.connect() as conn:
        conn.execute(text(f"CREATE TABLE {SCHEMA}.indexed_sources (id uuid PRIMARY KEY, n int)"))
        for source_id in ids:
            conn.execute(
                text(f"INSERT INTO {SCHEMA}.indexed_sources VALUES (CAST(:id AS uuid), 0)"),
                {"id": source_id},
            )
        conn.execute(
            text(
                f"ALTER TABLE {SCHEMA}.chunks ADD CONSTRAINT chunks_indexed_source_fk "
                f"FOREIGN KEY (indexed_source_id) REFERENCES {SCHEMA}.indexed_sources(id) "
                "ON DELETE CASCADE"
            )
        )
        conn.commit()
    return ids


def _open_update_of(engine, source_id):
    conn = engine.connect()
    conn.execute(
        text(f"UPDATE {SCHEMA}.indexed_sources SET n = n + 1 WHERE id = CAST(:id AS uuid)"),
        {"id": source_id},
    )
    pid = conn.execute(text("SELECT pg_backend_pid()")).scalar()
    return conn, pid


def test_a_writer_of_a_referenced_table_makes_the_move_give_up_fast_and_is_named(
    engine, session, monkeypatch
):
    """An open UPDATE of an unrelated indexed_sources row refused the foreign
    keys' SHARE ROW EXCLUSIVE, and the move waited its 5 s lock_timeout for it
    while holding every writer of the item table off -- then named nobody."""
    source_ids = _reference_indexed_sources(engine)
    monkeypatch.setattr(pgb, "MOVE_CHECK_CLEANUP_WAIT_SECONDS", 1.0)
    holder, holder_pid = _open_update_of(engine, source_ids[1])
    real = pgb.partition_lock_default_ddl
    writer: dict = {}

    def insert_for_another_kb():
        started = time.monotonic()
        with engine.connect() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, text) "
                    "VALUES (CAST(:kb AS uuid), 'anderswo')"
                ),
                {"kb": KB_B},
            )
            conn.commit()
        writer["waited"] = time.monotonic() - started

    def start_a_writer_once_writes_are_held(item_table):
        if "thread" not in writer:
            writer["thread"] = threading.Thread(target=insert_for_another_kb, daemon=True)
            writer["thread"].start()
        return real(item_table)

    monkeypatch.setattr(pgb, "partition_lock_default_ddl", start_a_writer_once_writes_are_held)
    started = time.monotonic()
    try:
        try:
            pgb.create_partition(engine, KB_A, "chunks")
        except Exception as exc:
            error = exc
        else:
            error = None
        took = time.monotonic() - started
        writer["thread"].join(timeout=30)
    finally:
        holder.rollback()
        holder.close()

    assert error is not None and pgb.is_lock_conflict(error), error
    assert error.bm25_move_step == "referenced tables"
    assert took < 3.0, took
    assert writer["waited"] < 2.0, writer
    named = [
        h
        for h in error.bm25_lock_holders
        if h["pid"] == holder_pid and h["granted"] and h["lock_on"].endswith("indexed_sources")
    ]
    assert named, error.bm25_lock_holders
    assert _rows_in(session, "chunks_default", KB_A) == len(KB_A_DOCS)
    # Nothing is left behind, and the next attempt goes through.
    assert pgb.create_partition(engine, KB_A, "chunks")["rows_moved"] == len(KB_A_DOCS)


def test_a_long_writer_of_a_referenced_table_stops_the_move_before_it_holds_writers(
    engine, session, monkeypatch
):
    source_ids = _reference_indexed_sources(engine)
    monkeypatch.setattr(pgb, "_long_holder_seconds", lambda: 1)
    taken = _spy_parent_lock(monkeypatch)
    holder, _pid = _open_update_of(engine, source_ids[0])
    try:
        time.sleep(1.2)
        try:
            pgb.create_partition(engine, KB_A, "chunks")
        except Exception as exc:
            error = exc
        else:
            error = None
    finally:
        holder.rollback()
        holder.close()

    assert error is not None and pgb.is_lock_conflict(error), error
    assert taken == []


def test_the_fast_path_gives_up_fast_on_a_writer_of_a_referenced_table(
    engine, session, monkeypatch
):
    source_ids = _reference_indexed_sources(engine)
    with engine.connect() as conn:
        conn.execute(
            text(f"DELETE FROM {SCHEMA}.chunks WHERE knowledge_base_id = CAST(:kb AS uuid)"),
            {"kb": KB_A},
        )
        conn.commit()
    holder, holder_pid = _open_update_of(engine, source_ids[1])
    started = time.monotonic()
    try:
        try:
            pgb.create_partition(engine, KB_A, "chunks")
        except Exception as exc:
            error = exc
        else:
            error = None
        took = time.monotonic() - started
    finally:
        holder.rollback()
        holder.close()

    assert error is not None and pgb.is_lock_conflict(error), error
    assert error.bm25_move_step == "referenced tables"
    assert took < 3.0, took
    assert any(h["pid"] == holder_pid for h in error.bm25_lock_holders), error.bm25_lock_holders


def test_a_long_writer_through_the_parent_stops_the_move_before_it_holds_writers(
    engine, session, monkeypatch
):
    """A long ungated transaction that wrote another, already partitioned,
    knowledge base's rows holds the parent, not DEFAULT: the move's SHARE lock
    on the parent queued behind it for 5 s, with every writer behind the move."""
    pgb.create_partition(engine, KB_B, "chunks")
    monkeypatch.setattr(pgb, "_long_holder_seconds", lambda: 1)
    taken = _spy_parent_lock(monkeypatch)
    holder = engine.connect()
    holder.execute(
        text(f"UPDATE {SCHEMA}.chunks SET text = text WHERE knowledge_base_id = CAST(:kb AS uuid)"),
        {"kb": KB_B},
    )
    holder_pid = holder.execute(text("SELECT pg_backend_pid()")).scalar()
    started = time.monotonic()
    try:
        time.sleep(1.2)
        started = time.monotonic()
        try:
            pgb.create_partition(engine, KB_A, "chunks")
        except Exception as exc:
            error = exc
        else:
            error = None
        took = time.monotonic() - started
    finally:
        holder.rollback()
        holder.close()

    assert error is not None and pgb.is_lock_conflict(error), error
    assert taken == []
    assert took < 2.0, took
    assert any(h["pid"] == holder_pid for h in error.bm25_lock_holders), error.bm25_lock_holders


def test_a_cascade_delete_during_the_move_wins_and_leaves_nothing_orphaned(
    engine, session, monkeypatch
):
    """A source deleted mid-move holds indexed_sources while its cascade waits
    for DEFAULT: the move gives up on the referenced table's lock instead of
    waiting for it, and the cascade then deletes the rows where they are."""
    source_ids = _reference_indexed_sources(engine)
    with engine.connect() as conn:
        conn.execute(
            text(
                f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, indexed_source_id, text) "
                "SELECT CAST(:kb AS uuid), CAST(:src AS uuid), 'weg ' || g "
                "FROM generate_series(1, 50) g"
            ),
            {"kb": KB_A, "src": source_ids[0]},
        )
        conn.commit()
    monkeypatch.setattr(pgb, "MOVE_CHECK_CLEANUP_WAIT_SECONDS", 1.0)
    real_validate = pgb.default_move_check_validate_ddl
    deleter: dict = {}

    def delete_the_source():
        started = time.monotonic()
        with engine.connect() as conn:
            conn.execute(
                text(f"DELETE FROM {SCHEMA}.indexed_sources WHERE id = CAST(:id AS uuid)"),
                {"id": source_ids[0]},
            )
            conn.commit()
        deleter["waited"] = time.monotonic() - started

    def a_source_is_deleted_mid_move(*args, **kwargs):
        if "thread" not in deleter:
            deleter["thread"] = threading.Thread(target=delete_the_source, daemon=True)
            deleter["thread"].start()
            time.sleep(0.3)
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(pgb, "default_move_check_validate_ddl", a_source_is_deleted_mid_move)
    try:
        pgb.create_partition(engine, KB_A, "chunks")
    except Exception as exc:
        error = exc
    else:
        error = None
    deleter["thread"].join(timeout=30)

    assert error is not None and pgb.is_transient_db_error(error), error
    assert deleter["waited"] < 3.0, deleter
    with engine.connect() as conn:
        orphans = conn.execute(
            text(
                f"SELECT count(*) FROM {SCHEMA}.chunks WHERE indexed_source_id = CAST(:id AS uuid)"
            ),
            {"id": source_ids[0]},
        ).scalar()
    assert orphans == 0
    assert _rows_in(session, "chunks", KB_A) == len(KB_A_DOCS)
