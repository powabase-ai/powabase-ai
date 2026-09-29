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


# How long the idle-holder specs sleep before looking. The ages are whole seconds
# computed by the server, and on a virtualised clock a 1.2 s client-side sleep has
# measured as 0 s there (2 failures in about 40 runs on Postgres 15), which reads the
# idle holder as one that has just finished a statement.
_IDLE_LONG_ENOUGH_S = 2.5


def _nothing_waits_on_a_lock(engine) -> None:
    with engine.connect() as conn:
        waiting = conn.execute(
            text(
                "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
                "AND datname = current_database()"
            )
        ).scalar()
        conn.rollback()
    assert waiting == 0, "no backend of the refused reconcile may be waiting on a lock"


def test_the_gate_holder_between_two_statements_is_found_and_counts_as_running(
    schema, engine, raw_dsn
):
    """The advisory holder is looked up by its key, which only a real server can check.

    ``pg_try_advisory_lock(bigint)`` files the key as ``classid``/``objid``; the lookup
    puts it back together. A holder that took the gate a moment ago and has shown no
    table lock yet is a caller mid-reconcile, and is waited for like a build.
    """
    holder = _hold_the_gate(raw_dsn)
    try:
        holder_pid = holder.info.backend_pid
        outcome = pvi.ensure_per_kb_vector_index(KB_B, engine=engine)
        assert outcome["status"] == "table_busy", outcome
        assert outcome["build_alive"] is True, outcome
        found = outcome["table_holders"]
        assert [(h["pid"], h["lock"], h["kind"]) for h in found] == [
            (holder_pid, "advisory", "running")
        ], found
        assert _queued_on_the_table(raw_dsn) == []
        assert _index_state(raw_dsn, KB_B) is None
        _nothing_waits_on_a_lock(engine)
    finally:
        holder.close()


def test_a_gate_held_by_an_idle_session_is_not_a_build(schema, engine, raw_dsn, monkeypatch):
    """What the brief calls "held but nothing running": the counted, backed-off case."""
    monkeypatch.setattr(pvi, "_GATE_HOLDER_IDLE_GRACE_S", 0)
    holder = _hold_the_gate(raw_dsn)
    try:
        time.sleep(_IDLE_LONG_ENOUGH_S)
        outcome = pvi.ensure_per_kb_vector_index(KB_B, engine=engine)
        assert outcome["status"] == "table_busy", outcome
        assert outcome["reason"] == "table_held_without_a_build", pvi.describe_table_holders(
            outcome["table_holders"]
        )
        assert outcome["build_alive"] is False, pvi.describe_table_holders(outcome["table_holders"])
        assert outcome["table_holders"][0]["kind"] == "stalled", pvi.describe_table_holders(
            outcome["table_holders"]
        )
        assert outcome["table_holders"][0]["idle_s"] >= 1, pvi.describe_table_holders(
            outcome["table_holders"]
        )
        assert _queued_on_the_table(raw_dsn) == []
        assert _index_state(raw_dsn, KB_B) is None
        _nothing_waits_on_a_lock(engine)
    finally:
        holder.close()

    # Released: the build proceeds and goes valid.
    outcome = pvi.ensure_per_kb_vector_index(KB_B, engine=engine)
    assert outcome["status"] == "ready", outcome
    assert _index_state(raw_dsn, KB_B) is True


def test_a_table_held_idle_in_transaction_refuses_the_gate_but_is_not_a_build(
    schema, engine, raw_dsn
):
    """A forgotten ``LOCK TABLE`` is not something that finishes; it must not buy 48 h of waits."""
    idle = psycopg.connect(raw_dsn)
    idle.execute(f"LOCK TABLE {SCHEMA}.embeddings IN SHARE MODE")
    try:
        time.sleep(_IDLE_LONG_ENOUGH_S)
        outcome = pvi.ensure_per_kb_vector_index(KB_B, engine=engine)
        assert outcome["status"] == "table_busy", outcome
        assert outcome["build_alive"] is False, outcome
        assert outcome["reason"] == "table_held_without_a_build", outcome
        (holder,) = outcome["table_holders"]
        assert holder["pid"] == idle.info.backend_pid, holder
        assert (holder["state"], holder["kind"], holder["mode"]) == (
            "idle in transaction",
            "stalled",
            "ShareLock",
        ), holder
        # The ages that tell an operator how long it has sat there -- the statement's
        # own age says almost nothing about that.
        assert holder["xact_s"] >= 1 and holder["idle_s"] >= 1, holder
        assert _queued_on_the_table(raw_dsn) == []
    finally:
        idle.rollback()
        idle.close()


# ---------------------------------------------------------------------------
# A role that cannot see other roles' backends
# ---------------------------------------------------------------------------

_READER_ROLE = "per_kb_gate_live_reader"


@pytest.fixture
def unprivileged_engine(raw_dsn, engine, schema):
    """The gate's own reads, as a role without pg_read_all_stats."""
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        conn.execute(f"DROP ROLE IF EXISTS {_READER_ROLE}")
        conn.execute(f"CREATE ROLE {_READER_ROLE} LOGIN PASSWORD 'reader'")
        conn.execute(f"GRANT USAGE ON SCHEMA {SCHEMA} TO {_READER_ROLE}")
    url = engine.url.set(username=_READER_ROLE, password="reader")
    eng = create_engine(url)
    yield eng
    eng.dispose()
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        conn.execute(f"DROP OWNED BY {_READER_ROLE}")
        conn.execute(f"DROP ROLE IF EXISTS {_READER_ROLE}")


@pytest.mark.timeout(90)
def test_a_holder_another_role_runs_is_unknown_without_pg_read_all_stats(
    schema, engine, raw_dsn, unprivileged_engine, monkeypatch
):
    """Without the privilege, a build and autovacuum look the same: all NULLs.

    So neither may buy the uncounted wait -- and the one warning says which grant
    makes the gate exact, which the second half of this spec then proves.
    """
    monkeypatch.setattr(pvi, "_warned_invisible_holders", False)
    writer = _open_write_transaction(raw_dsn)
    try:
        build = _InThread(_raw_ddl(raw_dsn, pvi.per_kb_index_ddl(KB_A, DIMS)))
        pid_a = _pid_building(raw_dsn, KB_A)

        with unprivileged_engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            holders = pvi.table_ddl_holders(conn)
        (holder,) = [h for h in holders if h["pid"] == pid_a]
        assert holder["kind"] == "unknown", holder
        assert holder["state"] is None and holder["backend_type"] is None, holder
        assert holder["query"] is None, "'<insufficient privilege>' is not a query"
        assert "query not visible" in pvi.describe_table_holders([holder])

        with psycopg.connect(raw_dsn, autocommit=True) as admin:
            admin.execute(f"GRANT pg_read_all_stats TO {_READER_ROLE}")
        with unprivileged_engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            holders = pvi.table_ddl_holders(conn)
        (holder,) = [h for h in holders if h["pid"] == pid_a]
        assert holder["kind"] == "running", holder
        _advance_the_xid_horizon(raw_dsn)
    finally:
        writer.rollback()
        writer.close()
    build.join()
    assert build.error is None, build.error


# ---------------------------------------------------------------------------
# A gate whose connection was replaced mid-dimension
# ---------------------------------------------------------------------------


def _invalidate(raw_dsn: str, kb_id: str) -> None:
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        conn.execute(
            "UPDATE pg_index SET indisvalid = false WHERE indexrelid = to_regclass(%s)::oid",
            (f'"{SCHEMA}".{pvi.per_kb_index_name(kb_id, DIMS)}',),
        )


def _advisory_holder_pids(raw_dsn: str, subject: str) -> list[int]:
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        return [
            row[0]
            for row in conn.execute(
                "SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND granted "
                "AND objsubid = 1 "
                "AND ((classid::bigint << 32) | objid::bigint) = hashtextextended(%s, 0)",
                (subject,),
            ).fetchall()
        ]


def _gate_holder_pids(raw_dsn: str) -> list[int]:
    return _advisory_holder_pids(raw_dsn, pvi.table_lock_relation())


@pytest.mark.timeout(90)
def test_a_build_after_a_lost_connection_runs_under_a_gate_taken_again(
    schema, engine, raw_dsn, monkeypatch
):
    """The repair DROP succeeds, its RESET finds the backend gone, and the build follows.

    Before the fix the gate's ``held`` flag outlived the backend its lock died with,
    so the build that followed -- hours of it, in production -- ran with no gate at
    all. Here the backend is terminated between the DROP and its RESET, and while
    the build is parked the gate must be held by the backend that is building.
    """
    assert pvi.ensure_per_kb_vector_index(KB_A, engine=engine)["status"] == "ready"
    _invalidate(raw_dsn, KB_A)

    real_drop = pvi._drop_index
    real_reset = pvi._reset_session_setting
    state: dict = {"in_drop": False, "terminated": None, "writer": None}

    def drop(conn, kb_id, dims):
        state["in_drop"] = True
        try:
            return real_drop(conn, kb_id, dims)
        finally:
            state["in_drop"] = False

    def reset(conn, name):
        if state["in_drop"] and state["terminated"] is None:
            # The DROP has finished. Lose the backend, and only now open the write
            # transaction that parks the build which follows -- opened earlier it
            # would have parked the DROP instead.
            pid = conn.execute(text("SELECT pg_backend_pid()")).scalar()
            with psycopg.connect(raw_dsn, autocommit=True) as admin:
                admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
            state["terminated"] = pid
            state["writer"] = _open_write_transaction(raw_dsn)
        return real_reset(conn, name)

    monkeypatch.setattr(pvi, "_drop_index", drop)
    monkeypatch.setattr(pvi, "_reset_session_setting", reset)

    repair = _InThread(lambda: pvi.ensure_per_kb_vector_index(KB_A, engine=engine))
    try:
        building = _pid_building(raw_dsn, KB_A)
        assert state["terminated"] is not None and building != state["terminated"]
        assert _gate_holder_pids(raw_dsn) == [building], (
            "the build must run under the gate, taken again on the new backend"
        )
        # And under this index's own lock, or a second task for the same knowledge
        # base would find it free and take the running build for an orphan.
        assert _advisory_holder_pids(raw_dsn, pvi.index_lock_relation(KB_A, DIMS)) == [building]
        _advance_the_xid_horizon(raw_dsn)
    finally:
        if state["writer"] is not None:
            state["writer"].rollback()
            state["writer"].close()
    repair.join()
    assert repair.error is None, repair.error
    assert repair.result["repaired_invalid_indexes"] == [pvi.per_kb_index_name(KB_A, DIMS)]
    assert _index_state(raw_dsn, KB_A) is True
    assert _gate_holder_pids(raw_dsn) == [], "and given back afterwards"


# ---------------------------------------------------------------------------
# The repair drop, choreographed against a running build
# ---------------------------------------------------------------------------


@pytest.mark.timeout(90)
def test_a_repair_drop_waits_for_another_kbs_build_instead_of_queueing(schema, engine, raw_dsn):
    """The statement that turned one killed build into a livelock in #95."""
    assert pvi.ensure_per_kb_vector_index(KB_A, engine=engine)["status"] == "ready"
    _invalidate(raw_dsn, KB_A)

    writer = _open_write_transaction(raw_dsn)
    try:
        build_b = _InThread(lambda: pvi.ensure_per_kb_vector_index(KB_B, engine=engine))
        pid_b = _pid_building(raw_dsn, KB_B)
        outcome_a = pvi.ensure_per_kb_vector_index(KB_A, engine=engine)
        assert outcome_a["status"] == "table_busy", outcome_a
        assert outcome_a["build_alive"] is True, outcome_a
        assert pid_b in [h["pid"] for h in outcome_a["table_holders"]], outcome_a
        assert "repaired_invalid_indexes" not in outcome_a, outcome_a
        assert _index_state(raw_dsn, KB_A) is False, "the INVALID index is left for later"
        assert _queued_on_the_table(raw_dsn) == []
        # Give a queued DROP -- if there were one -- time to be detected as the cycle.
        time.sleep(1.5 * _deadlock_timeout_s(raw_dsn))
        _advance_the_xid_horizon(raw_dsn)
    finally:
        writer.rollback()
        writer.close()
    build_b.join()
    assert build_b.error is None, build_b.error
    assert _index_state(raw_dsn, KB_B) is True

    outcome_a = pvi.ensure_per_kb_vector_index(KB_A, engine=engine)
    assert outcome_a["repaired_invalid_indexes"] == [pvi.per_kb_index_name(KB_A, DIMS)]
    assert _index_state(raw_dsn, KB_A) is True


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
