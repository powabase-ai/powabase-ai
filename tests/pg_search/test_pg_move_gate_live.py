"""Indexing transactions and the move take turns instead of racing.

A re-index reads a knowledge base's chunk ids through the parent (ACCESS SHARE
on DEFAULT) and then deletes and inserts through it. Against a move that holds
SHARE on the parent and needs ACCESS EXCLUSIVE on DEFAULT, such a transaction
kept DEFAULT held into the move's lock tries, and the move gave up every
time. So indexing takes a per-item-table advisory lock *shared* at the start
of each such transaction, and the move takes it *exclusively* before it
checks DEFAULT or takes any table lock: every indexing transaction runs
entirely before the move or entirely after it.

Same database and scratch schema as ``test_pg_bm25_live``.
"""

from __future__ import annotations

import threading
import time
import uuid

from sqlalchemy import text

from agentic_project_service.services import pg_bm25_index as pgb
from tests.pg_search import test_pg_bm25_live as live
from tests.pg_search.test_pg_bm25_live import (
    KB_A,
    KB_A_DOCS,
    KB_B,
    KB_B_DOCS,
    SCHEMA,
    SOURCE_1,
    _rows_in,
    _seed,
)

migration = live.migration
engine = live.engine
scratch_schema = live.scratch_schema
session = live.session


def _reindex_shaped_transaction(
    engine, kb_id, hold_seconds, errors, body="neu indexiert", source=SOURCE_1
):
    """What index_source's cleanup and write do, in one transaction."""
    with engine.connect() as conn:
        try:
            pgb.hold_move_gate_shared(conn, "chunks")
            ids = conn.execute(
                text(
                    f"SELECT id FROM {SCHEMA}.chunks WHERE knowledge_base_id = CAST(:kb AS uuid) "
                    "AND source_id = CAST(:src AS uuid)"
                ),
                {"kb": kb_id, "src": source},
            ).all()
            time.sleep(hold_seconds)
            conn.execute(
                text(
                    f"DELETE FROM {SCHEMA}.chunks WHERE knowledge_base_id = CAST(:kb AS uuid) "
                    "AND id = ANY(:ids)"
                ),
                {"kb": kb_id, "ids": [row[0] for row in ids]},
            )
            conn.execute(
                text(
                    f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text) "
                    "SELECT CAST(:kb AS uuid), CAST(:src AS uuid), :body FROM generate_series(1, :n)"
                ),
                {"kb": kb_id, "src": source, "body": body, "n": len(ids)},
            )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            errors.append(exc)


def test_a_reindex_transaction_that_holds_default_does_not_make_the_move_give_up(
    engine, session, monkeypatch
):
    """Without the gate every move failed this way: the transaction holds
    DEFAULT while its DELETE waits on the move's parent lock, and every lock try
    the move makes on DEFAULT is refused until it gives up."""
    monkeypatch.setattr(pgb, "DEFAULT_PREFLIGHT_WAIT_SECONDS", 0.0)
    _seed(session, KB_A, 2_000)
    errors: list = []
    started = threading.Event()
    real_hold = pgb.hold_move_gate_shared

    def hold_and_signal(conn, item_tables):
        real_hold(conn, item_tables)
        started.set()

    monkeypatch.setattr(pgb, "hold_move_gate_shared", hold_and_signal)
    worker = threading.Thread(
        target=_reindex_shaped_transaction, args=(engine, KB_B, 1.0, errors), daemon=True
    )
    worker.start()
    assert started.wait(timeout=10)

    moved = pgb.create_partition(engine, KB_A, "chunks")
    worker.join(timeout=30)

    assert errors == []
    assert moved["rows_moved"] == 2_000 + len(KB_A_DOCS)
    assert _rows_in(session, "chunks", KB_B) == len(KB_B_DOCS)


def test_a_waiting_move_is_not_starved_by_overlapping_indexing_transactions(
    engine, session, monkeypatch
):
    """Shared holders overlap without a gap, so a move that only *tried* for the
    lock would never get it. Queued, it is served once the holders ahead of it
    finish, and later indexing transactions queue behind it."""
    _seed(session, KB_A, 2_000)
    stop = threading.Event()
    errors: list = []
    sources = [str(uuid.uuid4()) for _ in range(3)]
    for source in sources:
        session.execute(
            text(
                f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text) "
                "SELECT CAST(:kb AS uuid), CAST(:src AS uuid), 'eigene Quelle' "
                "FROM generate_series(1, 2)"
            ),
            {"kb": KB_B, "src": source},
        )
    session.commit()

    def stream(source):
        while not stop.is_set():
            _reindex_shaped_transaction(engine, KB_B, 0.3, errors, source=source)

    workers = [threading.Thread(target=stream, args=(source,), daemon=True) for source in sources]
    for index, worker in enumerate(workers):
        worker.start()
        time.sleep(0.1 * (index + 1))
    try:
        started = time.monotonic()
        moved = pgb.create_partition(engine, KB_A, "chunks")
        took = time.monotonic() - started
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=30)

    assert errors == []
    assert moved["rows_moved"] == 2_000 + len(KB_A_DOCS)
    assert took < 5.0
    assert _rows_in(session, "chunks", KB_B) == len(KB_B_DOCS) + 2 * len(sources)


def test_the_move_gives_up_on_the_gate_after_its_bound_without_locking_the_table(
    engine, session, monkeypatch
):
    monkeypatch.setattr(pgb, "MOVE_GATE_WAIT_SECONDS", 0.5)
    parent_locks: list = []
    real_parent_lock = pgb.partition_lock_parent_ddl
    monkeypatch.setattr(
        pgb,
        "partition_lock_parent_ddl",
        lambda *a, **k: parent_locks.append(1) or real_parent_lock(*a, **k),
    )
    holder = engine.connect()
    pgb.hold_move_gate_shared(holder, "chunks")
    holder_pid = holder.execute(text("SELECT pg_backend_pid()")).scalar()
    try:
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

    assert error is not None and pgb.is_transient_db_error(error)
    assert error.bm25_move_step == "move gate"
    # The give-up names who held the gate, not only who held a table.
    assert any(
        h["pid"] == holder_pid and h["lock_on"] == "move gate" and h["granted"]
        for h in error.bm25_lock_holders
    ), error.bm25_lock_holders
    assert took < 3.0
    assert parent_locks == []
    assert _rows_in(session, "chunks_default", KB_A) == len(KB_A_DOCS)
    # The gate is released: the next attempt goes through.
    assert pgb.create_partition(engine, KB_A, "chunks")["rows_moved"] == len(KB_A_DOCS)


def test_indexing_waits_for_the_move_instead_of_failing(engine, session, monkeypatch):
    """An indexing transaction that starts during a move waits at the gate,
    holding nothing, and then writes into the new partition."""
    _seed(session, KB_A, 2_000)
    moving = threading.Event()
    live._hold_the_move_open(monkeypatch, moving, 1.0)
    errors: list = []

    def index_during_the_move():
        moving.wait(timeout=30)
        _reindex_shaped_transaction(engine, KB_A, 0.0, errors, body="nach dem Umzug")

    worker = threading.Thread(target=index_during_the_move, daemon=True)
    worker.start()
    pgb.create_partition(engine, KB_A, "chunks")
    worker.join(timeout=30)

    assert errors == []
    partition = pgb.partition_name(KB_A, "chunks")
    assert _rows_in(session, "chunks_default", KB_A) == 0
    assert _rows_in(session, partition) == 2_000 + len(KB_A_DOCS)


# ---------------------------------------------------------------------------
# One gate per item table
# ---------------------------------------------------------------------------


def _open_graph_index_run(engine, kb_id):
    """A graph_index run's transaction, as it stands during its LLM stages.

    It took graph_index_nodes' gate and wrote its nodes, and stays open (for
    minutes, in a real run) while the nodes are enriched and embedded.
    """
    conn = engine.connect()
    pgb.hold_move_gate_shared(conn, "graph_index_nodes")
    conn.execute(
        text(
            f"INSERT INTO {SCHEMA}.graph_index_nodes (knowledge_base_id, source_id, title, text) "
            "VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), 'Titel', 'Knoten')"
        ),
        {"kb": kb_id, "src": SOURCE_1},
    )
    return conn


def test_a_long_graph_index_run_does_not_hold_off_a_chunks_move(engine, session):
    _seed(session, KB_A, 2_000)
    graph_run = _open_graph_index_run(engine, KB_B)
    try:
        started = time.monotonic()
        moved = pgb.create_partition(engine, KB_A, "chunks")
        took = time.monotonic() - started
    finally:
        graph_run.rollback()
        graph_run.close()

    assert moved["rows_moved"] == 2_000 + len(KB_A_DOCS)
    assert took < 5.0


def test_chunk_indexing_does_not_queue_behind_a_graph_move_waiting_for_its_gate(
    engine, session, monkeypatch
):
    """A move on graph_index_nodes waits for a graph_index run and gives up;
    chunk indexing meanwhile goes straight through."""
    from agentic_project_service.tasks.indexing import _bm25_failure_reason

    monkeypatch.setattr(pgb, "MOVE_GATE_WAIT_SECONDS", 4.0)
    graph_run = _open_graph_index_run(engine, KB_B)
    result: dict = {}

    def move_graph_nodes():
        try:
            result["outcome"] = pgb.create_partition(engine, KB_A, "graph_index_nodes")
        except Exception as exc:
            result["error"] = exc

    mover = threading.Thread(target=move_graph_nodes, daemon=True)
    try:
        mover.start()
        deadline = time.monotonic() + 10
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
                    break
                time.sleep(0.05)
        assert waiting, "the graph move never queued for its gate"

        errors: list = []
        started = time.monotonic()
        _reindex_shaped_transaction(engine, KB_A, 0.0, errors, body="während des Wartens")
        chunk_indexing_took = time.monotonic() - started
        mover.join(timeout=30)
    finally:
        graph_run.rollback()
        graph_run.close()
        mover.join(timeout=30)

    assert errors == []
    assert chunk_indexing_took < 1.0
    error = result.get("error")
    assert error is not None and pgb.is_lock_conflict(error), result
    assert error.bm25_move_step == "move gate"
    reason = _bm25_failure_reason(error)
    assert "graph_index source is indexing" in reason, reason


def test_a_failed_move_lets_indexing_back_in_before_it_finishes_cleaning_up(
    engine, session, monkeypatch
):
    """A reader that arrives mid-move makes the ATTACH give up and keeps DEFAULT
    through the first tries to drop the move's check. The gate is released
    then, not after the whole cleanup wait, so indexing on the table is not
    held off while the rest of the cleanup waits for the reader."""
    _seed(session, KB_A, 2_000)
    monkeypatch.setattr(pgb, "MOVE_CHECK_CLEANUP_WAIT_SECONDS", 4.0)
    reader_holds_seconds = 3.0
    real_keys = pgb._build_move_indexes_and_keys
    reader_started = threading.Event()

    def a_reader_arrives(conn, kb_id, item_table, **kwargs):
        def read():
            with engine.connect() as reader:
                reader.execute(text(f"SELECT count(*) FROM {SCHEMA}.chunks_default"))
                reader_started.set()
                time.sleep(reader_holds_seconds)
                reader.rollback()

        threading.Thread(target=read, daemon=True).start()
        reader_started.wait(timeout=10)
        return real_keys(conn, kb_id, item_table, **kwargs)

    monkeypatch.setattr(pgb, "_build_move_indexes_and_keys", a_reader_arrives)
    released: list[float] = []
    real_release = pgb._MoveGate.release

    def spy_release(self):
        if self.held:
            released.append(time.monotonic())
        real_release(self)

    monkeypatch.setattr(pgb._MoveGate, "release", spy_release)

    started = time.monotonic()
    try:
        pgb.create_partition(engine, KB_A, "chunks")
    except Exception as exc:
        error = exc
    else:
        error = None
    failed_at = time.monotonic()

    assert error is not None and pgb.is_lock_conflict(error), error
    assert error.bm25_move_step == "attach"
    assert len(released) == 1
    # Released well before the cleanup ended: the cleanup ran on for at least
    # a second after it, until the reader let DEFAULT go.
    assert failed_at - released[0] > 1.0, (released[0] - started, failed_at - started)
    assert live._move_check_names(session) == []
    session.rollback()
    assert _rows_in(session, "chunks_default", KB_A) == 2_000 + len(KB_A_DOCS)


# ---------------------------------------------------------------------------
# The gate wait is reported on its own
# ---------------------------------------------------------------------------


def _hold_the_gate_shared_for(engine, item_table, seconds, started):
    def run():
        with engine.connect() as conn:
            pgb.hold_move_gate_shared(conn, item_table)
            started.set()
            time.sleep(seconds)
            conn.rollback()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _a_later_indexing_transaction(engine, item_table, waits):
    def run():
        with engine.connect() as conn:
            began = time.monotonic()
            pgb.hold_move_gate_shared(conn, item_table)
            waits.append(time.monotonic() - began)
            conn.rollback()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_the_move_reports_its_wait_for_the_gate_apart_from_the_write_block(engine, session):
    """While the move queues for the gate, indexing transactions that start on
    the table queue behind it: that stall is not in ``writes_blocked_seconds``."""
    _seed(session, KB_A, 2_000)
    started = threading.Event()
    holder = _hold_the_gate_shared_for(engine, "chunks", 1.5, started)
    assert started.wait(timeout=10)
    later_waits: list[float] = []
    later = None
    mover_result: dict = {}

    def move():
        mover_result["move"] = pgb.create_partition(engine, KB_A, "chunks")

    mover = threading.Thread(target=move, daemon=True)
    mover.start()
    time.sleep(0.3)
    later = _a_later_indexing_transaction(engine, "chunks", later_waits)
    mover.join(timeout=30)
    later.join(timeout=30)
    holder.join(timeout=30)

    result = mover_result["move"]
    assert result["gate_wait_seconds"] >= 1.0, result
    assert result["writes_blocked_seconds"] < result["gate_wait_seconds"], result
    # The indexing transaction that arrived meanwhile waited for the gate as well.
    assert later_waits and later_waits[0] >= 0.8, later_waits


def test_an_empty_knowledge_base_reports_its_gate_wait_too(engine, session):
    with engine.connect() as conn:
        conn.execute(
            text(f"DELETE FROM {SCHEMA}.chunks WHERE knowledge_base_id = CAST(:kb AS uuid)"),
            {"kb": KB_A},
        )
        conn.commit()
    started = threading.Event()
    holder = _hold_the_gate_shared_for(engine, "chunks", 1.0, started)
    assert started.wait(timeout=10)
    result = pgb.create_partition(engine, KB_A, "chunks")
    holder.join(timeout=30)

    assert result["rows_moved"] == 0
    assert result["gate_wait_seconds"] >= 0.7, result
    assert result["writes_blocked_seconds"] < 0.5, result


def test_ensure_passes_the_gate_wait_on(engine, session):
    outcome = pgb.ensure_bm25_index(KB_A, engine=engine, allow_row_move=True)
    assert outcome["status"] == "ready"
    assert outcome["gate_wait_seconds"] >= 0
    assert "writes_blocked_seconds" in outcome


def test_a_gate_give_up_logs_one_warning_that_says_what_held_it(
    engine, session, monkeypatch, caplog
):
    monkeypatch.setattr(pgb, "MOVE_GATE_WAIT_SECONDS", 0.5)
    holder = engine.connect()
    pgb.hold_move_gate_shared(holder, "chunks")
    try:
        with caplog.at_level("INFO", logger=pgb.logger.name):
            try:
                pgb.create_partition(engine, KB_A, "chunks")
            except Exception as exc:
                error = exc
            else:
                error = None
    finally:
        holder.rollback()
        holder.close()

    assert error is not None and pgb.is_lock_conflict(error)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, warnings
    assert "move gate" in warnings[0] and "indexing" in warnings[0]
    # Not yet known to be a move of rows: it may have been an empty attach.
    assert "Moving the rows" not in warnings[0]
