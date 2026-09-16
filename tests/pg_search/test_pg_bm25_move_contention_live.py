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
