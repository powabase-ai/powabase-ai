"""A knowledge base's partition move against the transactions around it.

The move holds SHARE on the partitioned parent for its whole transaction, and
needs ACCESS EXCLUSIVE on the DEFAULT partition twice: once on a second
connection to put up its temporary check, and once for the ATTACH. These tests
drive real transactions against both of those steps -- a transaction that read
DEFAULT and then writes through the parent, a reader left idle in its
transaction, a long read that starts part-way through the move -- and pin who
gives way, what is left behind, and how long everyone else waited.

Same database and scratch schema as ``test_pg_bm25_live``.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.pool import NullPool

from agentic_project_service.services import pg_bm25_index as pgb
from tests.pg_search import test_pg_bm25_live as live
from tests.pg_search.test_pg_bm25_live import (
    KB_A,
    KB_A_DOCS,
    KB_B,
    KB_B_DOCS,
    KB_C,
    SCHEMA,
    SOURCE_1,
    _hold_the_move_open,
    _move_check_names,
    _rows_in,
    _seed,
)

# The live module's fixtures: its database, and the scratch schema every test
# here starts from (autouse there, so re-bound here to stay autouse).
migration = live.migration
engine = live.engine
scratch_schema = live.scratch_schema
session = live.session


def _kb_rows(session, kb_id) -> int:
    return _rows_in(session, "chunks", kb_id)


def _insert(conn, kb_id, body):
    conn.execute(
        text(
            f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text) "
            "VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), :body)"
        ),
        {"kb": kb_id, "src": SOURCE_1, "body": body},
    )


def _first_line(exc: BaseException) -> str:
    return str(getattr(exc, "orig", exc)).splitlines()[0]


# ---------------------------------------------------------------------------
# A transaction that reads DEFAULT and then writes through the parent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stall_seconds", [0.3, 1.5])
def test_a_transaction_that_read_default_then_writes_is_never_the_deadlock_victim(
    engine, session, monkeypatch, stall_seconds
):
    """The app transaction commits; the move gives way and succeeds on retry.

    What the re-index cleanup does: SELECT the chunk ids (ACCESS SHARE on
    DEFAULT), then write through the parent, which waits on the move's SHARE
    lock. If the move then *waited* for ACCESS EXCLUSIVE on DEFAULT, that is a
    lock cycle, and Postgres aborts whichever side runs its deadlock check
    first -- the app whenever it began waiting less than ``deadlock_timeout``
    before the ATTACH (the 0.3 s case), the move otherwise (1.5 s). The move
    must never wait in that queue, so it is the move that gives up, both times.
    """
    monkeypatch.setattr(pgb, "DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS", 0.5, raising=False)
    _seed(session, KB_A, 2_000)
    moving = threading.Event()
    _hold_the_move_open(monkeypatch, moving, stall_seconds)
    app: dict = {}

    def app_transaction():
        moving.wait(timeout=30)
        with engine.connect() as conn:
            try:
                conn.execute(
                    text(
                        f"SELECT count(*) FROM {SCHEMA}.chunks "
                        "WHERE knowledge_base_id = CAST(:kb AS uuid)"
                    ),
                    {"kb": KB_B},
                ).scalar()
                _insert(conn, KB_B, "rando pendant le deplacement")
                conn.execute(
                    text(
                        f"DELETE FROM {SCHEMA}.chunks WHERE knowledge_base_id = CAST(:kb AS uuid)"
                    ),
                    {"kb": KB_C},
                )
                conn.commit()
                app["committed"] = True
            except Exception as exc:
                conn.rollback()
                app["error"] = _first_line(exc)

    thread = threading.Thread(target=app_transaction, daemon=True)
    thread.start()
    first_attempt: BaseException | None = None
    try:
        pgb.create_partition(engine, KB_A, "chunks")
    except Exception as exc:
        first_attempt = exc
    thread.join(timeout=60)

    assert app == {"committed": True}
    if first_attempt is not None:
        assert pgb.is_transient_db_error(first_attempt), _first_line(first_attempt)
        assert _move_check_names(session) == []
        assert pgb.create_partition(engine, KB_A, "chunks")["rows_moved"] == 2_000 + len(KB_A_DOCS)
    partition = pgb.partition_name(KB_A, "chunks")
    assert _rows_in(session, partition) == 2_000 + len(KB_A_DOCS)
    assert _rows_in(session, "chunks_default", KB_A) == 0
    assert _kb_rows(session, KB_B) == len(KB_B_DOCS) + 1
    assert _kb_rows(session, KB_C) == 0


# ---------------------------------------------------------------------------
# The temporary check on DEFAULT, when a reader is in the way
# ---------------------------------------------------------------------------


def _read_default_and_hold(engine, kb_id=KB_B):
    """A connection whose open transaction holds ACCESS SHARE on DEFAULT."""
    conn = engine.connect()
    conn.execute(
        text(
            f"SELECT count(*) FROM {SCHEMA}.chunks_default "
            "WHERE knowledge_base_id = CAST(:kb AS uuid)"
        ),
        {"kb": kb_id},
    ).scalar()
    return conn


class _ReaderLoop:
    """Short reads of DEFAULT in a loop, recording how long each one took."""

    def __init__(self, engine, kb_id=KB_C):
        self.engine = engine
        self.kb_id = kb_id
        self.latencies: list[float] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            while not self._stop.is_set():
                issued = time.monotonic()
                try:
                    conn.execute(
                        text(
                            f"SELECT count(*) FROM {SCHEMA}.chunks "
                            "WHERE knowledge_base_id = CAST(:kb AS uuid)"
                        ),
                        {"kb": self.kb_id},
                    ).scalar()
                except Exception as exc:
                    self.errors.append(_first_line(exc))
                self.latencies.append(time.monotonic() - issued)
                time.sleep(0.02)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=30)
        return False


def _add_move_check(engine, kb_id):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(pgb.default_move_check_add_ddl(kb_id, "chunks")))


def test_a_move_whose_check_never_went_up_does_not_touch_default_again(
    engine, session, monkeypatch
):
    """A reader idle in its transaction stops the check going up at all.

    The move gives up and rolls back. Nothing was committed on DEFAULT, so
    there is nothing to clean up: no second try for its lock, the knowledge
    base's writes are accepted straight away (the reader is still open), and
    readers of DEFAULT never queued behind the move.
    """
    monkeypatch.setattr(pgb, "DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS", 0.5)
    monkeypatch.setattr(pgb, "MOVE_CHECK_CLEANUP_WAIT_SECONDS", 3.0, raising=False)
    _seed(session, KB_A, 2_000)
    holder = _read_default_and_hold(engine)
    try:
        with _ReaderLoop(engine) as readers:
            time.sleep(0.1)
            started = time.monotonic()
            with pytest.raises(Exception) as caught:
                pgb.create_partition(engine, KB_A, "chunks")
            elapsed = time.monotonic() - started
            time.sleep(0.1)
        assert pgb.is_transient_db_error(caught.value), _first_line(caught.value)
        assert elapsed < 2.0, elapsed
        assert _move_check_names(session) == []
        with engine.begin() as conn:
            _insert(conn, KB_A, "Wanderung waehrend der Leser wartet")
        assert readers.errors == []
        assert max(readers.latencies) < 0.3, max(readers.latencies)
    finally:
        holder.rollback()
        holder.close()

    assert pgb.create_partition(engine, KB_A, "chunks")["rows_moved"] == 2_000 + len(KB_A_DOCS) + 1


def test_a_failed_move_retries_dropping_its_check_until_the_reader_is_gone(
    engine, session, monkeypatch
):
    """A long read that starts after the check went up breaks the ATTACH.

    That reader is still there when the move rolls back, so one try at dropping
    the check fails too. The move keeps trying for a bounded time, and the
    check is gone once the reader finishes -- without readers ever queueing.
    """
    monkeypatch.setattr(pgb, "DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS", 0.5)
    monkeypatch.setattr(pgb, "MOVE_CHECK_CLEANUP_WAIT_SECONDS", 4.0, raising=False)
    _seed(session, KB_A, 2_000)
    moving = threading.Event()
    _hold_the_move_open(monkeypatch, moving, 0.3)

    def long_read():
        moving.wait(timeout=30)
        conn = _read_default_and_hold(engine)
        time.sleep(2.0)
        conn.rollback()
        conn.close()

    reader = threading.Thread(target=long_read, daemon=True)
    with _ReaderLoop(engine) as readers:
        reader.start()
        with pytest.raises(Exception) as caught:
            pgb.create_partition(engine, KB_A, "chunks")
        reader.join(timeout=30)

    assert pgb.is_transient_db_error(caught.value), _first_line(caught.value)
    assert _move_check_names(session) == []
    with engine.begin() as conn:
        _insert(conn, KB_A, "Wanderung nach dem Abbruch")
    assert _rows_in(session, "chunks_default", KB_A) == 2_000 + len(KB_A_DOCS) + 1
    assert readers.errors == []
    assert max(readers.latencies) < 0.3, max(readers.latencies)


def test_a_check_left_by_a_failed_move_is_cleared_by_the_next_ensure_first(
    engine, session, monkeypatch
):
    """A reader that outlives the cleanup leaves the check behind; the next
    ensure on that item table -- even for a knowledge base that already has
    its partition, so no move runs -- clears it before anything else."""
    monkeypatch.setattr(pgb, "DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS", 0.3)
    monkeypatch.setattr(pgb, "MOVE_CHECK_CLEANUP_WAIT_SECONDS", 0.5, raising=False)
    assert pgb.ensure_bm25_index(KB_B, engine=engine)["status"] == "ready"
    moving = threading.Event()
    _hold_the_move_open(monkeypatch, moving, 0.2)
    reader_done = threading.Event()

    def long_read():
        moving.wait(timeout=30)
        conn = _read_default_and_hold(engine, KB_C)
        reader_done.wait(timeout=30)
        conn.rollback()
        conn.close()

    reader = threading.Thread(target=long_read, daemon=True)
    reader.start()
    try:
        with pytest.raises(Exception) as caught:
            pgb.create_partition(engine, KB_A, "chunks")
        assert pgb.is_transient_db_error(caught.value), _first_line(caught.value)
        assert _move_check_names(session) == [pgb.default_move_check_name(KB_A)]
    finally:
        reader_done.set()
        reader.join(timeout=30)

    assert pgb.ensure_bm25_index(KB_B, engine=engine)["status"] == "ready"

    assert _move_check_names(session) == []
    with engine.begin() as conn:
        _insert(conn, KB_A, "Wanderung nach dem Aufraeumen")
    assert _rows_in(session, "chunks_default", KB_A) == len(KB_A_DOCS) + 1


def test_an_indexing_write_refused_by_a_leftover_check_clears_it_and_lands_on_retry(
    engine, session, monkeypatch
):
    """A long read that starts mid-move outlives the move's cleanup and leaves
    its check on DEFAULT. Indexing of that knowledge base does not wait for the
    next index build to clear it: the refused write clears it (one try, which
    gives up at once while the reader is still there) and the requeued attempt
    lands -- well inside the attempts bound."""
    from unittest.mock import MagicMock

    from agentic_project_service.tasks import indexing

    monkeypatch.setattr(pgb, "DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS", 0.3)
    monkeypatch.setattr(pgb, "MOVE_CHECK_CLEANUP_WAIT_SECONDS", 0.5, raising=False)
    moving = threading.Event()
    _hold_the_move_open(monkeypatch, moving, 0.2)
    reader_done = threading.Event()

    def long_read():
        moving.wait(timeout=30)
        conn = _read_default_and_hold(engine, KB_C)
        reader_done.wait(timeout=30)
        conn.rollback()
        conn.close()

    reader = threading.Thread(target=long_read, daemon=True)
    reader.start()
    try:
        with pytest.raises(Exception) as caught:
            pgb.create_partition(engine, KB_A, "chunks")
        assert pgb.is_transient_db_error(caught.value), _first_line(caught.value)
        assert _move_check_names(session) == [pgb.default_move_check_name(KB_A)]

        with pytest.raises(Exception) as refused:
            with engine.begin() as conn:
                _insert(conn, KB_A, "Wanderung gegen den Zaun")
        started = time.monotonic()
        assert pgb.clear_move_check_after_refusal(engine, refused.value) == []
        assert time.monotonic() - started < 0.5
        assert _move_check_names(session) == [pgb.default_move_check_name(KB_A)]
    finally:
        reader_done.set()
        reader.join(timeout=30)

    # index_source with its bookkeeping faked and its write real.
    fake_db = MagicMock()
    fake_db.engine = engine
    monkeypatch.setattr(indexing, "db", fake_db)
    monkeypatch.setattr(indexing, "get_knowledge_base", lambda _id: {"indexing_config": {}})
    monkeypatch.setattr(indexing, "get_source", lambda _id: {"extraction_status": "extracted"})
    attempts = {"n": 0}
    monkeypatch.setattr(indexing, "_claim_indexed_source", lambda *_a: attempts["n"])
    failed = MagicMock()
    monkeypatch.setattr(indexing, "_fenced_mark_failed", failed)
    outcomes: list = []

    def write(**_kwargs):
        with engine.begin() as conn:
            _insert(conn, KB_A, "Wanderung nach dem Zaun")
        return {"status": "success"}

    def attempt():
        attempts["n"] += 1
        return indexing.index_source.run(KB_A, SOURCE_1, indexed_source_id=str(uuid.uuid4()))

    def requeue(**kwargs):
        if attempts["n"] >= indexing.MAX_ATTEMPTS:
            outcomes.append("attempts exhausted")
        else:
            outcomes.append(attempt())

    monkeypatch.setattr(indexing, "_run_index_body", write)
    monkeypatch.setattr(indexing, "_handle_storage_error", requeue)

    attempt()

    assert outcomes == [{"status": "success"}]
    assert attempts["n"] == 2
    failed.assert_not_called()
    assert _move_check_names(session) == []
    assert _rows_in(session, "chunks_default", KB_A) == len(KB_A_DOCS) + 1


def test_ensure_leaves_a_check_alone_while_a_move_holds_the_table(engine, session):
    """The per-table build lock says a move is running, and its check is live."""
    assert pgb.ensure_bm25_index(KB_B, engine=engine)["status"] == "ready"
    _add_move_check(engine, KB_C)
    relation = pgb.partition_build_lock_relation("chunks")
    with engine.connect() as mover:
        mover.execute(text(pgb.partition_build_lock_sql()), {"relation": relation})
        mover.commit()
        try:
            assert pgb.ensure_bm25_index(KB_B, engine=engine)["status"] == "ready"
            assert _move_check_names(session) == [pgb.default_move_check_name(KB_C)]
        finally:
            mover.execute(text(pgb.partition_build_unlock_sql()), {"relation": relation})
            mover.commit()

    pgb.ensure_bm25_index(KB_B, engine=engine)
    assert _move_check_names(session) == []


def test_a_successful_move_drops_its_check_inside_its_own_transaction(engine, session, monkeypatch):
    """The move already holds ACCESS EXCLUSIVE on DEFAULT for the ATTACH, so
    the check goes in the same transaction: a reader that arrives while the move
    commits cannot leave it behind, and the move does not wait for that reader.
    """
    monkeypatch.setattr(pgb, "MOVE_CHECK_CLEANUP_WAIT_SECONDS", 3.0, raising=False)
    _seed(session, KB_A, 2_000)
    real_mirror = pgb.mirror_relation_settings_sql
    release = threading.Event()
    queued = threading.Event()

    def long_read():
        conn = _read_default_and_hold(engine, KB_C)
        release.wait(timeout=30)
        conn.rollback()
        conn.close()

    reader = threading.Thread(target=long_read, daemon=True)

    def _mirror_with_a_reader_queued(*args, **kwargs):
        # Start a reader that queues behind the ATTACH's lock and is granted it
        # the moment the move commits; wait until Postgres shows it waiting.
        reader.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not queued.is_set():
            with engine.connect() as probe:
                waiting = probe.execute(
                    text(
                        "SELECT count(*) FROM pg_locks WHERE NOT granted "
                        "AND relation = to_regclass(:rel)"
                    ),
                    {"rel": f"{SCHEMA}.chunks_default"},
                ).scalar()
                probe.rollback()
            if waiting:
                queued.set()
            else:
                time.sleep(0.01)
        return real_mirror(*args, **kwargs)

    monkeypatch.setattr(pgb, "mirror_relation_settings_sql", _mirror_with_a_reader_queued)
    try:
        started = time.monotonic()
        move = pgb.create_partition(engine, KB_A, "chunks")
        elapsed = time.monotonic() - started
        checks = _move_check_names(session)
    finally:
        release.set()
        reader.join(timeout=30)

    assert queued.is_set()
    assert move["rows_moved"] == 2_000 + len(KB_A_DOCS)
    assert checks == []
    assert elapsed < 2.0, elapsed


def test_start_up_clears_a_leftover_check_without_ever_blocking(engine, session):
    """A worker killed mid-move leaves its check behind. Start-up clears it when
    DEFAULT is free, and when it is not, returns at once and never raises."""
    _add_move_check(engine, KB_A)
    holder = _read_default_and_hold(engine)
    try:
        started = time.monotonic()
        outcome = pgb.clear_leftover_move_checks_at_start(engine)
        assert time.monotonic() - started < 1.0
        assert _move_check_names(session) == [pgb.default_move_check_name(KB_A)]
        assert outcome["chunks"] == "busy"
    finally:
        holder.rollback()
        holder.close()

    outcome = pgb.clear_leftover_move_checks_at_start(engine)

    assert outcome["chunks"] == [pgb.default_move_check_name(KB_A)]
    assert _move_check_names(session) == []
    with engine.begin() as conn:
        _insert(conn, KB_A, "Wanderung nach dem Neustart")


def test_start_up_check_sweep_never_raises_without_the_tables(engine, monkeypatch):
    monkeypatch.setattr(pgb, "AI_SCHEMA", f"absent_{uuid.uuid4().hex[:8]}")
    assert set(pgb.clear_leftover_move_checks_at_start(engine).values()) == {"not_partitioned"}


def test_the_move_is_not_starved_by_a_steady_stream_of_short_reads(engine, session):
    """Overlapping short reads leave DEFAULT without a single free moment, so
    ``NOWAIT`` tries alone never succeed. One short queued try lets the reads in
    flight drain while holding new ones back only briefly."""
    _seed(session, KB_A, 2_000)
    stop = threading.Event()
    latencies: list[float] = []

    def busy_reader():
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            while not stop.is_set():
                issued = time.monotonic()
                conn.execute(
                    text(
                        f"SELECT count(*), pg_sleep(0.03) FROM {SCHEMA}.chunks_default "
                        "WHERE knowledge_base_id = CAST(:kb AS uuid)"
                    ),
                    {"kb": KB_C},
                ).all()
                latencies.append(time.monotonic() - issued)

    readers = [threading.Thread(target=busy_reader, daemon=True) for _ in range(8)]
    for thread in readers:
        thread.start()
    try:
        time.sleep(0.3)
        move = pgb.create_partition(engine, KB_A, "chunks")
    finally:
        stop.set()
        for thread in readers:
            thread.join(timeout=30)

    assert move["rows_moved"] == 2_000 + len(KB_A_DOCS)
    # A read waits at most for one queued try, plus its own 30 ms.
    assert max(latencies) < pgb.DEFAULT_EXCLUSIVE_QUEUED_TRY_MS / 1000 + 0.3, max(latencies)


def test_a_queued_try_is_never_made_while_a_holder_of_default_waits_for_a_lock(engine, session):
    """The one case a queued request could close a lock cycle in: a transaction
    that holds DEFAULT and is itself waiting -- for the move's parent lock, or
    for anything that might be waiting on the move."""
    with engine.connect() as holder, engine.connect() as blocker, engine.connect() as probe:
        blocker.execute(text(f"LOCK TABLE {SCHEMA}.knowledge_bases IN ACCESS EXCLUSIVE MODE"))
        holder.execute(text(f"SELECT count(*) FROM {SCHEMA}.chunks_default")).scalar()
        waiter = threading.Thread(
            target=lambda: holder.execute(
                text(f"SELECT count(*) FROM {SCHEMA}.knowledge_bases")
            ).scalar(),
            daemon=True,
        )
        assert pgb._default_holder_is_waiting(probe, "chunks") is False
        probe.rollback()
        waiter.start()
        deadline = time.monotonic() + 10
        waiting = False
        while time.monotonic() < deadline and not waiting:
            waiting = pgb._default_holder_is_waiting(probe, "chunks")
            probe.rollback()
            time.sleep(0.02)
        blocker.rollback()
        waiter.join(timeout=10)
        holder.rollback()
    assert waiting is True


def test_a_holder_of_default_waiting_on_a_row_lock_counts_as_waiting(engine, session):
    """A row lock is waited for as a transaction id, a lock with no database:
    the database filter must not hide it."""
    with engine.connect() as holder, engine.connect() as blocker, engine.connect() as probe:
        blocker.execute(
            text(f"UPDATE {SCHEMA}.chunks_default SET text = text WHERE knowledge_base_id = :kb"),
            {"kb": KB_C},
        )
        holder.execute(text(f"SELECT count(*) FROM {SCHEMA}.chunks_default")).scalar()
        waiter = threading.Thread(
            target=lambda: holder.execute(
                text(
                    f"UPDATE {SCHEMA}.chunks_default SET text = text WHERE knowledge_base_id = :kb"
                ),
                {"kb": KB_C},
            ),
            daemon=True,
        )
        waiter.start()
        deadline = time.monotonic() + 10
        waiting = False
        while time.monotonic() < deadline and not waiting:
            waiting = pgb._default_holder_is_waiting(probe, "chunks")
            probe.rollback()
            time.sleep(0.02)
        blocker.rollback()
        waiter.join(timeout=10)
        holder.rollback()
    assert waiting is True


def test_a_waiting_holder_of_a_same_oid_table_in_another_database_does_not_count(engine):
    """``pg_locks`` spans the cluster, and a database copied from a template has
    the template's relation OIDs. A session in the copy that holds its own
    ``chunks_default`` and waits for a lock is no reason to skip the queued try
    in the original: nothing there can be waiting on this session."""
    suffix = uuid.uuid4().hex[:8]
    original, twin = f"bm25_oid_{suffix}", f"bm25_oid_twin_{suffix}"
    admin = engine.execution_options(isolation_level="AUTOCOMMIT")

    def _engine_for(name):
        from sqlalchemy import create_engine

        return create_engine(engine.url.set(database=name), poolclass=NullPool)

    with admin.connect() as conn:
        conn.execute(text(f"CREATE DATABASE {original} TEMPLATE template0"))
    try:
        seeded = _engine_for(original)
        with seeded.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))
            conn.execute(text(f"CREATE TABLE {SCHEMA}.chunks_default (id int)"))
            conn.execute(text(f"CREATE TABLE {SCHEMA}.knowledge_bases (id int)"))
            conn.execute(text(f"INSERT INTO {SCHEMA}.knowledge_bases VALUES (1)"))
        seeded.dispose()
        with admin.connect() as conn:
            conn.execute(text(f"CREATE DATABASE {twin} TEMPLATE {original}"))
        copied = _engine_for(twin)
        seeded = _engine_for(original)
        oid_sql = text(f"SELECT to_regclass('{SCHEMA}.chunks_default')::oid")
        with seeded.connect() as a, copied.connect() as b:
            assert a.execute(oid_sql).scalar() == b.execute(oid_sql).scalar()
        with (
            copied.connect() as holder,
            copied.connect() as blocker,
            copied.connect() as twin_probe,
            seeded.connect() as probe,
        ):
            # Waiting on a row lock -- a transaction id, which has no database --
            # so only the DEFAULT lock's own database tells the two apart.
            update = text(f"UPDATE {SCHEMA}.knowledge_bases SET id = id")
            blocker.execute(update)
            holder.execute(text(f"SELECT count(*) FROM {SCHEMA}.chunks_default")).scalar()
            waiter = threading.Thread(target=lambda: holder.execute(update), daemon=True)
            waiter.start()
            deadline = time.monotonic() + 10
            twin_waiting = False
            while time.monotonic() < deadline and not twin_waiting:
                twin_waiting = pgb._default_holder_is_waiting(twin_probe, "chunks")
                twin_probe.rollback()
                time.sleep(0.02)
            seen_from_original = pgb._default_holder_is_waiting(probe, "chunks")
            probe.rollback()
            blocker.rollback()
            waiter.join(timeout=10)
            holder.rollback()
        copied.dispose()
        seeded.dispose()
    finally:
        with admin.connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {twin} WITH (FORCE)"))
            conn.execute(text(f"DROP DATABASE IF EXISTS {original} WITH (FORCE)"))

    assert twin_waiting is True
    assert seen_from_original is False


def test_the_queued_try_is_skipped_when_it_would_land_on_a_waiting_transactions_deadlock_check(
    engine, session
):
    """Timed so that, were the queued try made, it would still be waiting when
    the application transaction's deadlock check runs -- which would find the
    cycle and abort the application. The try is skipped, the lock attempt gives
    up, and the application transaction commits once the mover rolls back."""
    with engine.connect() as probe:
        deadlock_seconds = (
            probe.execute(
                text("SELECT setting::int FROM pg_settings WHERE name = 'deadlock_timeout'")
            ).scalar()
            / 1000
        )
    app: dict = {}
    app_waiting = threading.Event()

    def app_transaction():
        with engine.connect() as conn:
            try:
                conn.execute(text(f"SELECT count(*) FROM {SCHEMA}.chunks_default")).scalar()
                app_waiting.set()
                _insert(conn, KB_B, "rando qui attend")
                conn.commit()
                app["committed"] = True
            except Exception as exc:
                conn.rollback()
                app["error"] = _first_line(exc)

    with engine.connect() as mover:
        mover.execute(text(pgb.partition_lock_parent_ddl("chunks")))
        thread = threading.Thread(target=app_transaction, daemon=True)
        thread.start()
        app_waiting.wait(timeout=10)
        # The queued try would start ~0.1 s into the call and wait up to 0.2 s,
        # straddling the moment the application's deadlock check runs.
        time.sleep(max(deadlock_seconds - 0.2, 0.0))
        with pytest.raises(Exception) as caught:
            pgb._lock_default_exclusively(mover, "chunks", 1.0)
        mover.rollback()
    thread.join(timeout=30)

    assert pgb.is_lock_conflict(caught.value), _first_line(caught.value)
    assert app == {"committed": True}
