"""A move completes under ordinary indexing traffic, and nothing is lost.

The traffic that made every move give up: re-index transactions of 70-120 ms
running back to back on two workers, other writers that know nothing of the
move, and a stream of long re-index transactions (2-10 s, one starting every
second) that read the item table and hold it while they work. The move has to
finish within its task's retry budget, and every write that committed has to
be there afterwards, exactly once.

Same database and scratch schema as ``test_pg_bm25_live``.

The traffic runs on an engine of its own. The long re-indexers alone hold up to
ten connections at a time, and on the engine under test they exhausted its pool
(5 + 10 overflow), so the move failed for want of a connection -- a starvation
the test caused, not one the move has in a worker, whose pool is its own.
"""

from __future__ import annotations

import random
import threading
import time
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from agentic_project_service.services import pg_bm25_index as pgb
from tests.pg_search import test_pg_bm25_live as live
from tests.pg_search.test_pg_bm25_live import KB_A, KB_A_DOCS, KB_B, KB_C, SCHEMA, _rows_in, _seed

migration = live.migration
engine = live.engine
scratch_schema = live.scratch_schema
session = live.session

RETRY_BUDGET = 6

# Tags the traffic's backends, so teardown can find any that outlive their thread.
TRAFFIC_APPLICATION_NAME = "bm25_acceptance_traffic"
# A long re-index holds its connection for up to 10 s, so this covers the
# slowest one still running when the traffic is told to stop.
TRAFFIC_STOP_SECONDS = 30


class _Traffic:
    def __init__(self, engine):
        self.engine = engine
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.errors: list[str] = []
        self.writer_latencies: list[tuple[float, float]] = []
        self.expected_sources: dict[str, tuple[str, int]] = {}
        self.inserted_ids: list[str] = []
        self.threads: list[threading.Thread] = []

    def _source(self, kb_id, rows):
        source = str(uuid.uuid4())
        with self.engine.connect() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text) "
                    "SELECT CAST(:kb AS uuid), CAST(:src AS uuid), 'Quelle ' || g "
                    "FROM generate_series(1, :n) g"
                ),
                {"kb": kb_id, "src": source, "n": rows},
            )
            conn.commit()
        with self.lock:
            self.expected_sources[source] = (kb_id, rows)
        return source

    def _reindex(self, kb_id, source, hold_seconds):
        """index_source's cleanup and write: gate, read ids, (work), delete, insert."""
        with self.engine.connect() as conn:
            try:
                pgb.hold_move_gate_shared(conn)
                ids = conn.execute(
                    text(
                        f"SELECT id FROM {SCHEMA}.chunks "
                        "WHERE knowledge_base_id = CAST(:kb AS uuid) AND source_id = CAST(:src AS uuid)"
                    ),
                    {"kb": kb_id, "src": source},
                ).all()
                time.sleep(hold_seconds)
                conn.execute(
                    text(
                        f"DELETE FROM {SCHEMA}.chunks WHERE knowledge_base_id = CAST(:kb AS uuid) "
                        "AND source_id = CAST(:src AS uuid)"
                    ),
                    {"kb": kb_id, "src": source},
                )
                conn.execute(
                    text(
                        f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text) "
                        "SELECT CAST(:kb AS uuid), CAST(:src AS uuid), 'neu ' || g "
                        "FROM generate_series(1, :n) g"
                    ),
                    {"kb": kb_id, "src": source, "n": len(ids)},
                )
                conn.commit()
            except Exception as exc:
                conn.rollback()
                with self.lock:
                    self.errors.append(f"reindex: {str(getattr(exc, 'orig', exc)).splitlines()[0]}")

    def short_reindexer(self, kb_id):
        source = self._source(kb_id, 20)

        def run():
            while not self.stop.is_set():
                self._reindex(kb_id, source, random.uniform(0.07, 0.12))

        self._start(run)

    def long_reindexers(self):
        """One long re-index starting every second, each on a source of its own."""

        def run():
            while not self.stop.is_set():
                kb_id = random.choice([KB_A, KB_B])
                source = self._source(kb_id, 10)
                self._start(lambda k=kb_id, s=source: self._reindex(k, s, random.uniform(2, 10)))
                self.stop.wait(1.0)

        self._start(run)

    def other_writer(self, kb_id):
        """A writer that knows nothing of the move: short inserts through the parent."""

        def run():
            with self.engine.connect() as conn:
                while not self.stop.is_set():
                    started = time.monotonic()
                    new_id = str(uuid.uuid4())
                    try:
                        conn.execute(
                            text(
                                f"INSERT INTO {SCHEMA}.chunks (id, knowledge_base_id, text) "
                                "VALUES (CAST(:id AS uuid), CAST(:kb AS uuid), 'anderer Schreiber')"
                            ),
                            {"id": new_id, "kb": kb_id},
                        )
                        conn.commit()
                        with self.lock:
                            self.inserted_ids.append(new_id)
                            self.writer_latencies.append((started, time.monotonic() - started))
                    except Exception as exc:
                        conn.rollback()
                        with self.lock:
                            self.errors.append(
                                f"writer: {str(getattr(exc, 'orig', exc)).splitlines()[0]}"
                            )
                    self.stop.wait(0.02)

        self._start(run)

    def _start(self, target):
        thread = threading.Thread(target=target, daemon=True)
        with self.lock:
            self.threads.append(thread)
        thread.start()

    def finish(self):
        """Stop the traffic and wait for every thread; return any still alive."""
        self.stop.set()
        deadline = time.monotonic() + TRAFFIC_STOP_SECONDS
        while True:
            with self.lock:
                alive = [t for t in self.threads if t.is_alive()]
            if not alive or time.monotonic() > deadline:
                return alive
            alive[0].join(timeout=1)


def _terminate_traffic_backends(engine) -> None:
    """End any traffic session still connected, so the schema drop cannot meet it."""
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(
            text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE application_name = :name AND datname = current_database() "
                "AND pid <> pg_backend_pid()"
            ),
            {"name": TRAFFIC_APPLICATION_NAME},
        )


@pytest.fixture
def traffic(engine, scratch_schema):
    """The traffic mix, stopped and gone before the scratch schema is dropped.

    Depends on ``scratch_schema`` so it is torn down first. A test that fails
    mid-move would otherwise leave threads writing while that teardown runs
    ``DROP SCHEMA ... CASCADE``, which then deadlocks with them.
    """
    traffic_engine = create_engine(
        engine.url,
        poolclass=NullPool,
        connect_args={"application_name": TRAFFIC_APPLICATION_NAME},
    )
    traffic = _Traffic(traffic_engine)
    try:
        yield traffic
    finally:
        traffic.finish()
        _terminate_traffic_backends(engine)
        traffic_engine.dispose()


def test_the_move_completes_under_the_reindex_mix_and_loses_nothing(engine, session, traffic):
    _seed(session, KB_A, 20_000)
    traffic.short_reindexer(KB_A)
    traffic.short_reindexer(KB_B)
    traffic.other_writer(KB_B)
    traffic.other_writer(KB_C)
    traffic.long_reindexers()
    time.sleep(3.0)

    attempts: list[str] = []
    outcome = None
    move_started = time.monotonic()
    for _ in range(RETRY_BUDGET + 1):
        try:
            outcome = pgb.ensure_bm25_index(KB_A, engine=engine)
            attempts.append("ok")
            break
        except Exception as exc:
            assert pgb.is_transient_db_error(exc), str(exc)
            attempts.append(str(getattr(exc, "orig", exc)).splitlines()[0])
            time.sleep(1.0)
    move_ended = time.monotonic()
    time.sleep(2.0)
    still_running = traffic.finish()

    assert still_running == []
    assert outcome is not None and outcome["status"] == "ready", attempts
    assert traffic.errors == []
    during = [
        latency
        for started, latency in traffic.writer_latencies
        if started < move_ended and started + latency > move_started
    ]
    blocked = max(during) if during else 0.0
    print(
        f"acceptance: attempts={attempts} rows_moved={outcome.get('rows_moved')} "
        f"reported_block={outcome.get('writes_blocked_seconds')} "
        f"max_other_writer_wait={blocked:.3f}s"
    )

    # Nothing lost, nothing twice.
    partition = pgb.partition_name(KB_A, "chunks")
    assert _rows_in(session, "chunks_default", KB_A) == 0
    counts = dict(
        session.execute(
            text(f"SELECT source_id::text, count(*) FROM {SCHEMA}.chunks GROUP BY source_id")
        ).all()
    )
    duplicates = session.execute(
        text(f"SELECT count(*) - count(DISTINCT id) FROM {SCHEMA}.chunks")
    ).scalar()
    present = session.execute(
        text(f"SELECT count(*) FROM {SCHEMA}.chunks WHERE id = ANY(CAST(:ids AS uuid[]))"),
        {"ids": traffic.inserted_ids},
    ).scalar()
    session.rollback()
    assert duplicates == 0
    assert present == len(traffic.inserted_ids)
    for source, (kb_id, rows) in traffic.expected_sources.items():
        assert counts.get(source) == rows, (source, kb_id, counts.get(source), rows)
    assert _rows_in(session, partition) >= 20_000 + len(KB_A_DOCS)
