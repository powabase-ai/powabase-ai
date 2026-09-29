"""Two knowledge bases' index DDL on one table: the deadlock, and the gate that prevents it.

Issue #95, against a real server. The deadlock is reproduced here, not described:
every ``CREATE``/``DROP INDEX CONCURRENTLY`` on the shared ``embeddings`` table takes
its self-conflicting ``ShareUpdateExclusiveLock``, so a second knowledge base's DDL
queues inside Postgres behind a running build -- with its snapshot already taken --
and the build's last phase, ``WaitForOlderSnapshots``, waits on that snapshot. A
cycle, and the build is the victim, after all of its work.

What makes it deterministic rather than a race against a fast build is one open
write transaction. ``CREATE INDEX CONCURRENTLY``'s first phase waits for every
transaction holding a lock that conflicts with ``ShareLock`` -- an uncommitted
``INSERT`` does -- so build A is parked in that phase for as long as the test likes.
DDL B is issued while it is parked, queues on the table lock, and takes its snapshot
while the write transaction's xid is still running, so its ``xmin`` is at or below
that xid. The write transaction then ends and a few xids are consumed, so the
reference snapshot A takes for its validation phase has an ``xmin`` strictly above
B's. From there Postgres does the rest: A builds, validates, waits for B's older
snapshot, and B is waiting for A's table lock. The deadlock detector fires in A,
which began waiting last, one ``deadlock_timeout`` later.

The same choreography through the service is the proof of the fix: B's reconcile
meets the gate, returns ``table_busy`` at once having issued nothing, and leaves no
backend queued on the table -- so A finishes, and B builds afterwards.

Scratch schema of its own and about 9,000 rows at 384 dimensions, so the module
costs a few seconds; nothing here needs the planner-shaped fixture of
``test_per_kb_vector_index_live``.
"""

from __future__ import annotations

import io
import os
import threading
import time
import uuid

import numpy as np
import psycopg
import pytest
from sqlalchemy import create_engine, text

from agentic_project_service.services import pg_vector_index as pvi

SCHEMA = "vector_perkb_gate_live_test"
DIMS = 384

KB_A = "7c1e0a4d-0000-4000-8000-000000000001"
KB_B = "7c1e0a4d-0000-4000-8000-000000000002"
KB_FILLER = "7c1e0a4d-0000-4000-8000-000000000003"
SOURCE = "7c1e0a4d-0000-4000-8000-0000000000aa"

ROWS_A = 4_000
ROWS_B = 4_000
ROWS_FILLER = 1_000

# Thresholds far below both knowledge bases, so each of them is one reconcile away
# from a build; the numbers themselves are not under test.
_SETTINGS = {
    "VECTOR_PER_KB_INDEX_MIN_ROWS": 1_000,
    "VECTOR_PER_KB_INDEX_DROP_ROWS": 500,
    "VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB": 64,
}

# Every wait in this module polls for a state the choreography guarantees; a
# deadline is only there so a regression fails instead of hanging.
_DEADLINE_S = 30.0


def _dsn() -> str:
    dsn = os.environ.get("PG_SEARCH_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        if os.environ.get("PG_SEARCH_REQUIRED") == "1":
            pytest.fail("PG_SEARCH_REQUIRED=1 but no PG_SEARCH_TEST_DATABASE_URL or DATABASE_URL")
        pytest.skip("no PG_SEARCH_TEST_DATABASE_URL or DATABASE_URL to test pgvector against")
    return dsn


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(_dsn())
    where = eng.url.render_as_string(hide_password=True)
    try:
        with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as exc:
        eng.dispose()
        reason = f"pgvector is not available on {where}: {str(exc).splitlines()[0]}"
        if os.environ.get("PG_SEARCH_REQUIRED") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def raw_dsn(engine) -> str:
    return engine.url.set(drivername="postgresql").render_as_string(hide_password=False)


@pytest.fixture(scope="module")
def fixture_schema(raw_dsn):
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        conn.execute(f"CREATE SCHEMA {SCHEMA}")
        conn.execute(f"""
            CREATE TABLE {SCHEMA}.embeddings (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                item_id uuid NOT NULL,
                item_table varchar(50) NOT NULL,
                knowledge_base_id uuid NOT NULL,
                source_id uuid NOT NULL,
                embedding_model varchar(255) NOT NULL,
                dims smallint NOT NULL,
                embedding vector NOT NULL
            )
        """)
        # Serial builds: a parallel HNSW build asks for a shared memory segment
        # the size of maintenance_work_mem, more than a stock container's /dev/shm.
        conn.execute(f"ALTER TABLE {SCHEMA}.embeddings SET (parallel_workers = 0)")
        rng = np.random.default_rng(95)
        for kb_id, rows in ((KB_A, ROWS_A), (KB_B, ROWS_B), (KB_FILLER, ROWS_FILLER)):
            vectors = rng.standard_normal((rows, DIMS))
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            buf = io.StringIO()
            for vector in vectors:
                literal = "[" + ",".join(f"{x:.5f}" for x in vector) + "]"
                buf.write(
                    f"{uuid.uuid4()}\tchunks\t{kb_id}\t{SOURCE}\ttest-embed\t{DIMS}\t{literal}\n"
                )
            buf.seek(0)
            with conn.cursor() as cur:
                with cur.copy(
                    f"COPY {SCHEMA}.embeddings (item_id, item_table, knowledge_base_id, "
                    "source_id, embedding_model, dims, embedding) FROM STDIN"
                ) as copy:
                    copy.write(buf.read())
        conn.execute(f"VACUUM ANALYZE {SCHEMA}.embeddings")
    yield
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")


def _drop_all_partial_indexes(raw_dsn: str) -> None:
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = %s AND c.relkind = 'i' AND c.relname LIKE 'hnsw_kb_%%'",
                (SCHEMA,),
            ).fetchall()
        ]
        for name in names:
            conn.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{SCHEMA}".{name}')


@pytest.fixture
def schema(fixture_schema, raw_dsn, monkeypatch):
    monkeypatch.setattr(pvi, "AI_SCHEMA", SCHEMA)
    monkeypatch.setattr(pvi, "get_setting", lambda key: _SETTINGS[key])
    monkeypatch.setattr(pvi, "read_overrides", lambda conn, *keys: dict(_SETTINGS))
    _drop_all_partial_indexes(raw_dsn)
    yield SCHEMA
    _drop_all_partial_indexes(raw_dsn)


# ---------------------------------------------------------------------------
# Choreography
# ---------------------------------------------------------------------------


def _open_write_transaction(raw_dsn: str) -> psycopg.Connection:
    """An uncommitted INSERT into the table: a CIC's first phase waits for it."""
    conn = psycopg.connect(raw_dsn)
    conn.execute(
        f"INSERT INTO {SCHEMA}.embeddings (item_id, item_table, knowledge_base_id, source_id, "
        f"embedding_model, dims, embedding) VALUES (%s, 'chunks', %s, %s, 'test-embed', "
        f"{DIMS}, %s)",
        (str(uuid.uuid4()), KB_FILLER, SOURCE, "[" + ",".join(["0.01"] * DIMS) + "]"),
    )
    return conn


def _advance_the_xid_horizon(raw_dsn: str, n: int = 5) -> None:
    """Consume a few xids, so a snapshot taken from now on has an xmin above today's."""
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        for _ in range(n):
            conn.execute("SELECT txid_current()")


def _poll(raw_dsn: str, sql: str, params=()) -> list:
    """Run ``sql`` until it returns a row, or fail at the deadline."""
    deadline = time.monotonic() + _DEADLINE_S
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        while time.monotonic() < deadline:
            rows = conn.execute(sql, params).fetchall()
            if rows:
                return rows
            time.sleep(0.05)
    pytest.fail(f"timed out waiting for: {sql} {params}")


def _pid_building(raw_dsn: str, kb_id: str) -> int:
    """The backend building this knowledge base's index, once it *holds* the table.

    Not merely once its statement is visible: until it holds its
    ``ShareUpdateExclusiveLock`` a second DDL could still get there first and swap
    the two roles -- which a first version of this module did, and then asserted
    the deadlock on the wrong backend.
    """
    return _poll(
        raw_dsn,
        "SELECT a.pid FROM pg_stat_activity a JOIN pg_locks l ON l.pid = a.pid "
        "WHERE a.query ILIKE %s AND a.pid <> pg_backend_pid() AND l.locktype = 'relation' "
        "AND l.relation = to_regclass(%s) AND l.mode = 'ShareUpdateExclusiveLock' "
        "AND l.granted",
        (
            f"%CREATE INDEX CONCURRENTLY%{pvi.per_kb_index_name(kb_id, DIMS)}%",
            f"{SCHEMA}.embeddings",
        ),
    )[0][0]


def _deadlock_timeout_s(raw_dsn: str) -> float:
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        ms = conn.execute(
            "SELECT setting::int FROM pg_settings WHERE name = 'deadlock_timeout'"
        ).fetchone()[0]
    return ms / 1000.0


def _queued_on_the_table(raw_dsn: str) -> list:
    """Backends waiting, inside Postgres, for a lock on the embeddings table itself."""
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        return conn.execute(
            "SELECT l.pid, l.mode FROM pg_locks l WHERE l.locktype = 'relation' "
            "AND l.relation = to_regclass(%s) AND NOT l.granted",
            (f"{SCHEMA}.embeddings",),
        ).fetchall()


def _index_state(raw_dsn: str, kb_id: str):
    """``indisvalid`` of this knowledge base's index, or None when there is none."""
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        row = conn.execute(
            "SELECT indisvalid FROM pg_index WHERE indexrelid = to_regclass(%s)",
            (f'"{SCHEMA}".{pvi.per_kb_index_name(kb_id, DIMS)}',),
        ).fetchone()
    return None if row is None else row[0]


class _InThread:
    """Run ``fn`` in a daemon thread and keep its result or its exception."""

    def __init__(self, fn):
        self.result = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self._thread.start()

    def _run(self, fn):
        try:
            self.result = fn()
        except BaseException as exc:  # surfaced by the assertions
            self.error = exc

    def join(self) -> None:
        self._thread.join(timeout=_DEADLINE_S * 2)
        assert not self._thread.is_alive(), "the build never finished"


def _raw_ddl(raw_dsn: str, ddl: str):
    def run():
        with psycopg.connect(raw_dsn, autocommit=True) as conn:
            conn.execute(ddl)

    return run


# ---------------------------------------------------------------------------
# The deadlock itself, without the gate
# ---------------------------------------------------------------------------


@pytest.mark.timeout(90)
def test_ddl_queued_on_the_table_kills_a_running_build_at_its_end(schema, raw_dsn):
    """The mechanism of #95, reproduced with two plain statements and no service code.

    This is why the gate exists, and why it has to be in the application: the
    queued statement does nothing wrong, and neither does the build. The build is
    killed only after its catalog entry, its whole graph and its validation scan.
    """
    writer = _open_write_transaction(raw_dsn)
    try:
        build_a = _InThread(_raw_ddl(raw_dsn, pvi.per_kb_index_ddl(KB_A, DIMS)))
        _pid_building(raw_dsn, KB_A)
        # Parked in its first phase, waiting for the writer.
        ddl_b = _InThread(_raw_ddl(raw_dsn, pvi.per_kb_index_ddl(KB_B, DIMS)))
        # B queued inside Postgres on the table lock, with its snapshot -- taken
        # while the writer's xid was running -- already held.
        _poll(
            raw_dsn,
            "SELECT l.pid FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid "
            "WHERE l.locktype = 'relation' AND l.relation = to_regclass(%s) "
            "AND NOT l.granted AND a.backend_xmin IS NOT NULL",
            (f"{SCHEMA}.embeddings",),
        )
        # Let B's own deadlock check run first, and find nothing: it fires once,
        # ``deadlock_timeout`` after B starts waiting, and at that moment A is still
        # waiting on the writer. The cycle then closes only when A starts waiting on
        # B, so A's check is the one that finds it -- the order production had, where
        # B had been queued for hours. Without this pause a fast build can reach its
        # last phase inside B's first second, and B is the victim instead.
        time.sleep(1.5 * _deadlock_timeout_s(raw_dsn))
        _advance_the_xid_horizon(raw_dsn)
    finally:
        writer.rollback()
        writer.close()

    build_a.join()
    ddl_b.join()
    assert isinstance(build_a.error, psycopg.errors.DeadlockDetected), (
        f"expected the running build to be the deadlock victim, got {build_a.error!r}"
    )
    assert ddl_b.error is None, ddl_b.error
    # Killed after the work: the catalog entry is left behind INVALID, which is the
    # state every retry of it then has to repair -- with a drop that queues in turn.
    assert _index_state(raw_dsn, KB_A) is False
    assert _index_state(raw_dsn, KB_B) is True


# ---------------------------------------------------------------------------
# The same choreography through the service: the gate
# ---------------------------------------------------------------------------


@pytest.mark.timeout(90)
def test_a_second_reconcile_meets_the_gate_and_the_running_build_finishes(schema, engine, raw_dsn):
    writer = _open_write_transaction(raw_dsn)
    try:
        build_a = _InThread(lambda: pvi.ensure_per_kb_vector_index(KB_A, engine=engine))
        pid_a = _pid_building(raw_dsn, KB_A)

        started = time.monotonic()
        outcome_b = pvi.ensure_per_kb_vector_index(KB_B, engine=engine)
        elapsed = time.monotonic() - started

        assert outcome_b["status"] == "table_busy", outcome_b
        assert outcome_b["build_alive"] is True, outcome_b
        assert pid_a in [h["pid"] for h in outcome_b["table_holders"]], outcome_b
        assert outcome_b["built"] == [], outcome_b
        assert elapsed < 5, f"the refusal must not wait for the build ({elapsed:.1f} s)"
        # The whole point: nothing is left queued inside Postgres for A to wait on.
        assert _queued_on_the_table(raw_dsn) == []
        assert _index_state(raw_dsn, KB_B) is None, "and B issued nothing at all"
        _advance_the_xid_horizon(raw_dsn)
    finally:
        writer.rollback()
        writer.close()

    build_a.join()
    assert build_a.error is None, build_a.error
    assert build_a.result["built"] == [pvi.per_kb_index_name(KB_A, DIMS)], build_a.result
    assert _index_state(raw_dsn, KB_A) is True

    # B's task comes back once the table is free, and builds.
    outcome_b = pvi.ensure_per_kb_vector_index(KB_B, engine=engine)
    assert outcome_b["status"] == "ready", outcome_b
    assert outcome_b["built"] == [pvi.per_kb_index_name(KB_B, DIMS)], outcome_b
    assert _index_state(raw_dsn, KB_B) is True


@pytest.mark.timeout(90)
def test_an_ungated_build_on_the_table_is_waited_for_too(schema, engine, raw_dsn):
    """An operator's hand-run CIC, or a worker from before the gate: no advisory lock.

    The gate's own lock is free, so only the ``pg_locks`` evidence can refuse it --
    and refusing is what keeps the service's DDL out of the queue behind it.
    """
    writer = _open_write_transaction(raw_dsn)
    try:
        manual = _InThread(_raw_ddl(raw_dsn, pvi.per_kb_index_ddl(KB_A, DIMS)))
        pid_a = _pid_building(raw_dsn, KB_A)
        outcome_b = pvi.ensure_per_kb_vector_index(KB_B, engine=engine)
        assert outcome_b["status"] == "table_busy", outcome_b
        assert outcome_b["build_alive"] is True, outcome_b
        assert pid_a in [h["pid"] for h in outcome_b["table_holders"]], outcome_b
        assert _queued_on_the_table(raw_dsn) == []
        _advance_the_xid_horizon(raw_dsn)
    finally:
        writer.rollback()
        writer.close()
    manual.join()
    assert manual.error is None, manual.error
    assert _index_state(raw_dsn, KB_A) is True


# ---------------------------------------------------------------------------
# A held gate with nothing running
# ---------------------------------------------------------------------------


def _hold_the_gate(raw_dsn: str) -> psycopg.Connection:
    conn = psycopg.connect(raw_dsn, autocommit=True)
    got = conn.execute(
        "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (pvi.table_lock_relation(),)
    ).fetchone()[0]
    assert got is True
    return conn


def test_a_held_gate_with_no_build_running_is_reported_as_a_dead_holder(schema, engine, raw_dsn):
    """What the brief calls "held but nothing running": the counted, backed-off case."""
    holder = _hold_the_gate(raw_dsn)
    try:
        outcome = pvi.ensure_per_kb_vector_index(KB_B, engine=engine)
        assert outcome["status"] == "table_busy", outcome
        assert outcome["reason"] == "table_lock_held", outcome
        assert outcome["build_alive"] is False, outcome
        assert _queued_on_the_table(raw_dsn) == []
        assert _index_state(raw_dsn, KB_B) is None
        with engine.connect() as conn:
            waiting = conn.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
                    "AND datname = current_database()"
                )
            ).scalar()
            conn.rollback()
        assert waiting == 0, "no backend of the refused reconcile may be waiting on a lock"
    finally:
        holder.close()

    # Released: the build proceeds and goes valid.
    outcome = pvi.ensure_per_kb_vector_index(KB_B, engine=engine)
    assert outcome["status"] == "ready", outcome
    assert _index_state(raw_dsn, KB_B) is True


def test_the_deleted_kbs_drop_is_refused_by_a_held_gate_and_leaves_the_index(
    schema, engine, raw_dsn
):
    assert pvi.ensure_per_kb_vector_index(KB_B, engine=engine)["status"] == "ready"
    holder = _hold_the_gate(raw_dsn)
    try:
        with pytest.raises(pvi.PerKbVectorIndexTableBusy):
            pvi.drop_per_kb_vector_indexes(KB_B, engine=engine)
        assert _index_state(raw_dsn, KB_B) is True, "nothing was issued"
    finally:
        holder.close()
    assert pvi.drop_per_kb_vector_indexes(KB_B, engine=engine)["indexes"] == [
        pvi.per_kb_index_name(KB_B, DIMS)
    ]
    assert _index_state(raw_dsn, KB_B) is None


def test_the_gate_is_given_back_after_a_build(schema, engine, raw_dsn):
    """Session-scoped, so a pooled connection that kept it would gate the table for ever."""
    assert pvi.ensure_per_kb_vector_index(KB_A, engine=engine)["status"] == "ready"
    holder = _hold_the_gate(raw_dsn)
    holder.close()
