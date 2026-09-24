"""Per-knowledge-base partial HNSW index: names, DDL, thresholds and query shape.

These are the specs a live database cannot pin cheaply: that the emitted DDL
matches the expression and operator class the query orders by, that the
knowledge base id reaches SQL as a literal (so a prepared statement's generic
plan can still match the index predicate), and that the two thresholds keep
their hysteresis whatever is stored for them.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from unittest.mock import MagicMock

import pytest

from agentic_project_service.services import pg_vector_index as pvi
from agentic_project_service.services.base_vector_store import (
    BasePgVectorStore,
    kb_sql_literal,
    validated_top_k,
)
from agentic_project_service.services.settings_registry import (
    SETTINGS_REGISTRY,
    validate_setting,
)

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
KB_HEX = uuid.UUID(KB).hex


class _ChunkStore(BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


def _statements(coro_factory):
    """Run a store coroutine against a spy session and return the SQL it issued."""
    session = MagicMock()
    captured: list[str] = []

    def spy_execute(text_obj, params=None):
        captured.append(text_obj.text if hasattr(text_obj, "text") else str(text_obj))
        return iter([])

    session.execute = spy_execute
    store = _ChunkStore(db_session=session, knowledge_base_id=KB)
    asyncio.run(coro_factory(store))
    return captured


def _search_sql(statements):
    hits = [s for s in statements if "ORDER BY" in s]
    assert hits, f"no search query issued; statements: {statements}"
    return hits[0]


# ---------------------------------------------------------------------------
# Naming and DDL
# ---------------------------------------------------------------------------


def test_index_name_is_hex_and_dimension_and_fits_the_identifier_limit():
    name = pvi.per_kb_index_name(KB, 1536)
    assert name == f"hnsw_kb_{KB_HEX}_1536"
    assert "-" not in name, "a dashed name would need quoting everywhere it appears"
    assert len(name.encode()) <= 63, "Postgres truncates identifiers past 63 bytes"


def test_index_name_carries_the_dimension_so_two_models_do_not_collide():
    assert pvi.per_kb_index_name(KB, 768) != pvi.per_kb_index_name(KB, 1536)


@pytest.mark.parametrize("bad", ["kb-1", "", None, "3f2504e0-4f89-11d3-9a0c", 7])
def test_index_name_refuses_anything_that_is_not_a_uuid(bad):
    with pytest.raises(ValueError, match="knowledge_base_id"):
        pvi.per_kb_index_name(bad, 1536)


@pytest.mark.parametrize("bad", [0, -1, 8193, "many", None])
def test_index_name_refuses_a_dimension_outside_pgvectors_range(bad):
    with pytest.raises(ValueError, match="dims"):
        pvi.per_kb_index_name(KB, bad)


def test_create_ddl_matches_the_expression_and_operator_class_the_query_orders_by():
    ddl = pvi.per_kb_index_ddl(KB, 1536)
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" in ddl
    assert f"hnsw_kb_{KB_HEX}_1536" in ddl
    assert '"ai".embeddings' in ddl
    # The index is on an expression; a query whose ORDER BY lacks the identical
    # cast matches no HNSW index at all.
    assert "USING hnsw ((embedding::vector(1536)) vector_cosine_ops)" in ddl


def test_create_ddl_predicate_names_the_kb_and_the_dimension():
    ddl = pvi.per_kb_index_ddl(KB, 1536)
    assert f"WHERE knowledge_base_id = '{KB}' AND dims = 1536" in ddl


def test_create_ddl_is_concurrent_because_a_build_must_not_block_writes():
    assert "CONCURRENTLY" in pvi.per_kb_index_ddl(KB, 1536)
    assert "CONCURRENTLY" in pvi.per_kb_index_drop_ddl(KB, 1536)


def test_drop_ddl_is_idempotent_and_schema_qualified():
    drop = pvi.per_kb_index_drop_ddl(KB, 1536)
    assert drop == f'DROP INDEX CONCURRENTLY IF EXISTS "ai".hnsw_kb_{KB_HEX}_1536'


def test_lock_relation_is_per_index_not_per_table():
    assert pvi.index_lock_relation(KB, 1536) != pvi.index_lock_relation(KB, 768)
    assert pvi.index_lock_relation(KB, 1536).endswith(f"hnsw_kb_{KB_HEX}_1536")


# ---------------------------------------------------------------------------
# Settings and thresholds
# ---------------------------------------------------------------------------


def test_the_three_settings_are_registered_with_the_hysteresis_and_a_memory_bound():
    """The relations, not the numbers.

    The two row thresholds are tuned against measurements that move (the
    regression the embeddings-side predicate costs was re-measured at 80 ms at
    21% selectivity and 129 ms at 30%, both in knowledge bases of 12.6k-18k
    rows), so pinning their exact values here would only duplicate the registry.
    What must hold whatever they are is the hysteresis and a bounded build
    memory.
    """
    build = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"]
    drop = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_DROP_ROWS"]
    mem = SETTINGS_REGISTRY["VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB"]
    assert (build.type, drop.type, mem.type) == ("int", "int", "int")
    assert drop.default < build.default, "the defaults must carry the hysteresis"
    assert build.default >= (build.min or 0)
    assert (mem.default, mem.max) == (128, 4096), (
        "unbounded build memory would OOM the smallest project databases"
    )
    for d in (build, drop, mem):
        assert d.category == "knowledge-retrieval"
        assert d.advanced is True
        assert d.description


_ROW_SETTINGS = ("VECTOR_PER_KB_INDEX_MIN_ROWS", "VECTOR_PER_KB_INDEX_DROP_ROWS")
_ALL_SETTINGS = _ROW_SETTINGS + ("VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB",)


@pytest.mark.parametrize("key", _ALL_SETTINGS)
def test_validate_setting_enforces_each_settings_own_range(key):
    """Read off the registry, so tightening a bound cannot leave this test behind."""
    defn = SETTINGS_REGISTRY[key]
    for value, ok in (
        (defn.min, True),
        (defn.min - 1, False),
        (defn.max, True),
        (defn.max + 1, False),
        ("many", False),
    ):
        accepted, message = validate_setting(key, value)
        assert accepted is ok, f"{key}={value!r}: {message}"


def _stub_settings(monkeypatch, values: dict[str, int]):
    """Stub the settings read, filling in the build memory a caller did not name."""
    filled = {"VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB": 128, **values}
    monkeypatch.setattr(pvi, "get_setting", lambda key: filled[key])


def test_thresholds_default_to_the_registry_values(monkeypatch):
    build = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"].default
    drop = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_DROP_ROWS"].default
    _stub_settings(
        monkeypatch,
        {"VECTOR_PER_KB_INDEX_MIN_ROWS": build, "VECTOR_PER_KB_INDEX_DROP_ROWS": drop},
    )
    assert pvi.thresholds() == (build, drop)


def test_thresholds_clamp_a_stored_value_outside_the_registrys_bounds(monkeypatch):
    # get_setting coerces but does not range-check, so a row written before a
    # bound was tightened would otherwise be used as-is.
    build_min = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"].min
    drop_min = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_DROP_ROWS"].min
    _stub_settings(
        monkeypatch,
        {
            "VECTOR_PER_KB_INDEX_MIN_ROWS": build_min - 5,
            "VECTOR_PER_KB_INDEX_DROP_ROWS": max(0, drop_min - 5),
        },
    )
    build_at, drop_below = pvi.thresholds()
    assert build_at == build_min, "below the registry minimum must be pulled up to it"
    assert drop_below == drop_min


@pytest.mark.parametrize("stored_drop", [50000, 60000])
def test_thresholds_restore_the_hysteresis_when_drop_is_not_below_build(monkeypatch, stored_drop):
    # Equal or inverted thresholds would build an index and drop it again on
    # every dispatch, each build being a full index build.
    _stub_settings(
        monkeypatch,
        {
            "VECTOR_PER_KB_INDEX_MIN_ROWS": 50000,
            "VECTOR_PER_KB_INDEX_DROP_ROWS": stored_drop,
        },
    )
    build_at, drop_below = pvi.thresholds()
    assert (build_at, drop_below) == (50000, 25000)
    assert drop_below < build_at


def test_the_substituted_drop_threshold_is_never_zero(monkeypatch):
    """The correction is the one line that could re-admit the zero the registry forbids.

    ``VECTOR_PER_KB_INDEX_DROP_ROWS`` has a minimum of 1 because at 0 only a
    knowledge base with no embeddings at all can satisfy the drop test, so an
    index is held open for a handful of rows and paid for on every write. A
    *substituted* value has to obey the same floor. Out of reach through the
    build setting's current minimum of 1,000, so the bound is relaxed here: the
    guard has to hold for whatever the registry allows next, which is the only
    way this line is ever reached.
    """
    build = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"]
    monkeypatch.setattr(build, "min", 1)
    _stub_settings(
        monkeypatch,
        {"VECTOR_PER_KB_INDEX_MIN_ROWS": 1, "VECTOR_PER_KB_INDEX_DROP_ROWS": 1},
    )
    _, drop_below = pvi.thresholds()
    assert drop_below == 1, f"the hysteresis correction produced {drop_below}"
    assert drop_below >= SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_DROP_ROWS"].min


def test_build_memory_is_clamped_to_the_registry_bound(monkeypatch):
    monkeypatch.setattr(pvi, "get_setting", lambda key: 99999)
    assert pvi.maintenance_work_mem_mb() == 4096


# ---------------------------------------------------------------------------
# The query shape the index depends on
# ---------------------------------------------------------------------------


def test_kb_sql_literal_quotes_a_canonical_uuid():
    assert kb_sql_literal(KB) == f"'{KB}'"
    assert kb_sql_literal(uuid.UUID(KB)) == f"'{KB}'"
    # The canonical form, so two spellings of one id produce one statement text.
    assert kb_sql_literal(KB.upper()) == f"'{KB}'"


@pytest.mark.parametrize(
    "bad",
    ["kb-test", "", None, "'; DROP TABLE ai.embeddings; --", "3f2504e0-4f89-11d3"],
)
def test_kb_sql_literal_refuses_anything_that_is_not_a_uuid(bad):
    with pytest.raises(ValueError, match="knowledge_base_id"):
        kb_sql_literal(bad)


def test_vector_search_filters_the_embeddings_side_too():
    """Without this predicate no per-KB partial index can ever be used.

    PostgreSQL matches a partial index only from a restriction clause on the
    relation the index is on, and it does not reason through `e.item_id = c.id`
    to reach `c.knowledge_base_id`.
    """
    sql = _search_sql(_statements(lambda s: s.vector_search(embedding=[0.0] * 1536, top_k=10)))
    normalized = " ".join(sql.split())
    assert f"e.knowledge_base_id = '{KB}'" in normalized, sql


def test_vector_search_keeps_the_item_table_filter_as_well_as_the_embeddings_one():
    """The embeddings-side predicate is added, not substituted -- both must be there.

    The item-table filter is what keeps an embedding whose item has been
    deleted, or belongs to another knowledge base, out of the answer; the
    embeddings-side one is what a partial index can be matched from. Dropping
    either would pass a test that only looked for the other.

    Both are SQL literals of the same canonical id: a bound one on *either* side
    loses the partial index once PostgreSQL adopts the statement's generic plan,
    measured at ~135 ms against 0.98 ms for the life of that connection.
    """
    sql = _search_sql(_statements(lambda s: s.vector_search(embedding=[0.0] * 1536, top_k=10)))
    normalized = " ".join(sql.split())
    assert f"c.knowledge_base_id = '{KB}'" in normalized, sql
    assert f"e.knowledge_base_id = '{KB}'" in normalized, sql


def test_vector_search_passes_the_kb_id_as_a_literal_not_a_bind_parameter():
    """A bound id cannot prove the index predicate in a generic plan.

    psycopg prepares a repeated statement, and from the sixth execution of the
    prepared statement PostgreSQL may switch to its generic plan. Measured with
    the id bound: 1.1 ms on the partial index for ten executions, then 54-72 ms
    on a bitmap scan plus an exact sort for every execution after.
    """
    sql = _search_sql(_statements(lambda s: s.vector_search(embedding=[0.0] * 1536, top_k=10)))
    embeddings_side = [line for line in sql.splitlines() if "e.knowledge_base_id" in line]
    assert embeddings_side, sql
    for line in embeddings_side:
        assert ":kb_id" not in line, f"the embeddings-side predicate must be a literal: {line}"


def test_vector_search_per_source_uses_the_same_literal():
    sql = _search_sql(
        _statements(
            lambda s: s.vector_search_per_source(
                embedding=[0.0] * 1536, per_source_k=2, source_cap=3
            )
        )
    )
    normalized = " ".join(sql.split())
    assert f"e.knowledge_base_id = '{KB}'" in normalized, sql
    assert "e.knowledge_base_id = :kb_id" not in normalized


def test_hybrid_searchs_vector_leg_carries_the_predicate():
    """hybrid_search delegates to vector_search, so one change covers both."""
    statements = _statements(
        lambda s: s.vector_search(embedding=[0.0] * 1536, top_k=10, _resolve=False)
    )
    assert f"e.knowledge_base_id = '{KB}'" in " ".join(_search_sql(statements).split())


def test_vector_search_interpolates_dims_and_the_limit_too():
    """All three values the index predicate and the LIMIT need must be literals.

    Measured under ``plan_cache_mode = force_generic_plan``: with any one of the
    knowledge base id, ``dims`` or the ``LIMIT`` left as a bind parameter, the
    generic plan reaches no HNSW index at all and falls back to an exact sort.
    The KB id and ``dims`` are both named in the partial index's predicate, and
    an unknown ``LIMIT`` makes the planner assume a large fraction of the rows.
    """
    sql = _search_sql(_statements(lambda s: s.vector_search(embedding=[0.0] * 1536, top_k=7)))
    normalized = " ".join(sql.split())
    assert "e.dims = 1536" in normalized, sql
    assert "LIMIT 7" in normalized, sql
    assert ":dims" not in sql, f"dims must not be bound:\n{sql}"
    assert ":top_k" not in sql, f"the limit must not be bound:\n{sql}"


def test_vector_search_per_source_interpolates_dims_too():
    sql = _search_sql(
        _statements(
            lambda s: s.vector_search_per_source(
                embedding=[0.0] * 1536, per_source_k=2, source_cap=3
            )
        )
    )
    assert "e.dims = 1536" in " ".join(sql.split()), sql
    assert ":dims" not in sql, sql


@pytest.mark.parametrize("value,expected", [(0, 0), (1, 1), ("20", 20), (10_000, 10_000)])
def test_validated_top_k_accepts_what_a_caller_can_legitimately_ask_for(value, expected):
    # Zero is allowed because a bound ``LIMIT 0`` returned an empty answer
    # rather than erroring, and that is not a behaviour worth changing here.
    assert validated_top_k(value) == expected


@pytest.mark.parametrize("bad", [-1, 10_001, "many", None, 1.5e9])
def test_validated_top_k_refuses_anything_unsafe_to_interpolate(bad):
    with pytest.raises(ValueError, match="top_k"):
        validated_top_k(bad)


def test_an_out_of_range_top_k_never_reaches_sql():
    session = MagicMock()
    session.execute = lambda *a, **k: pytest.fail("no SQL may be built for a bad top_k")
    store = _ChunkStore(db_session=session, knowledge_base_id=KB)
    with pytest.raises(ValueError, match="top_k"):
        asyncio.run(store.vector_search(embedding=[0.0] * 1536, top_k=-5))


def test_order_by_still_carries_the_cast_the_index_expression_needs():
    sql = _search_sql(_statements(lambda s: s.vector_search(embedding=[0.0] * 768, top_k=10)))
    assert "ORDERBY(e.embedding::vector(768))" in "".join(sql.split())


# ---------------------------------------------------------------------------
# The cap on how many of these one project may hold
# ---------------------------------------------------------------------------


def test_the_index_cap_is_far_below_where_planning_and_locks_degrade():
    # Measured: 500 partial indexes on one table cost 1.9 ms of planning and
    # 513 locks per backend; 5,000 cost 25 ms and made the seventh concurrent
    # search fail with "out of shared memory".
    assert 0 < pvi.MAX_PER_KB_INDEXES <= 500


# ---------------------------------------------------------------------------
# A connection and an engine to run the service's own code against
# ---------------------------------------------------------------------------


class _Result:
    """Just enough of a SQLAlchemy result for the three shapes this module reads."""

    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._rows[0][0] if self._rows else None


class _FakeConn:
    """Records every statement, answers the ones a test names, fails one on request.

    ``answers`` are ``(sql fragment, rows)`` pairs matched by substring, first
    match winning, so a test names only the queries it cares about and
    everything else comes back empty. ``fail_on`` is the fragment whose
    statement raises, which is how the lifecycle specs put a failure exactly
    where it hurts.
    """

    def __init__(self, answers=(), fail_on=None, exc=None):
        self.answers = list(answers)
        self.statements: list[str] = []
        self.params: list[dict | None] = []
        self.invalidated = False
        self.rollbacks = 0
        self._fail_on = fail_on
        self._exc = exc if exc is not None else RuntimeError("statement failed")

    def execute(self, clause, params=None):
        sql = clause.text if hasattr(clause, "text") else str(clause)
        self.statements.append(" ".join(sql.split()))
        self.params.append(params)
        if self._fail_on is not None and self._fail_on in sql:
            raise self._exc
        for fragment, rows in self.answers:
            if fragment in sql:
                return _Result(rows(params) if callable(rows) else rows)
        return _Result([])

    def execution_options(self, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def rollback(self):
        self.rollbacks += 1

    def invalidate(self):
        self.invalidated = True

    def issued(self, fragment: str) -> list[str]:
        return [s for s in self.statements if fragment in s]


class _FakeEngine:
    def __init__(self, conn):
        self.conn = conn

    def connect(self):
        return self.conn


def _raiser(exc):
    def boom(*_args, **_kwargs):
        raise exc

    return boom


def _failure_comment(failures: int) -> str:
    """The comment the service writes, built by the service's own template.

    So a test says "two failures are already on record" rather than restating the
    wording, and a reworded comment the parser still reads keeps these passing.
    """
    return pvi._BUILD_FAILURES_COMMENT.format(n=failures)


# The catalog row shape both the ensure survey and the boot sweep read.
def _index_row(kb_id: str, dims: int, valid: bool = True):
    return (pvi.per_kb_index_name(kb_id, dims), valid)


def _ensure(monkeypatch, conn, kb_id=KB, build_at=10_000, drop_below=5_000, **kwargs):
    """Run the real ensure against a fake connection, with the thresholds fixed."""
    _stub_settings(
        monkeypatch,
        {
            "VECTOR_PER_KB_INDEX_MIN_ROWS": build_at,
            "VECTOR_PER_KB_INDEX_DROP_ROWS": drop_below,
        },
    )
    return pvi.ensure_per_kb_vector_index(kb_id, engine=_FakeEngine(conn), **kwargs)


# ---------------------------------------------------------------------------
# Lifecycle: a build's session settings never ride a pooled connection out
# ---------------------------------------------------------------------------


def test_a_build_sets_its_session_settings_before_the_ddl_and_resets_every_one_after(monkeypatch):
    monkeypatch.setattr(pvi, "maintenance_work_mem_mb", lambda: 256)
    conn = _FakeConn()
    pvi._create_index(conn, KB, 1536)
    assert conn.statements == [
        "SET statement_timeout = 0",
        "SET lock_timeout = 0",
        "SET maintenance_work_mem = '256MB'",
        " ".join(pvi.per_kb_index_ddl(KB, 1536).split()),
        "RESET maintenance_work_mem",
        "RESET lock_timeout",
        "RESET statement_timeout",
    ]


def test_a_drop_lifts_both_timeouts_and_puts_both_back(monkeypatch):
    """The drop waits in the same two ways the build does, so it lifts the same two."""
    conn = _FakeConn()
    pvi._drop_index(conn, KB, 1536)
    assert conn.statements == [
        "SET statement_timeout = 0",
        "SET lock_timeout = 0",
        " ".join(pvi.per_kb_index_drop_ddl(KB, 1536).split()),
        "RESET lock_timeout",
        "RESET statement_timeout",
    ]


def test_a_failed_second_setting_never_leaves_the_connection_without_a_timeout(monkeypatch):
    """``statement_timeout = 0`` must not outlive the build that needed it.

    A session-level ``SET`` survives the pool's rollback-on-return, so a
    connection handed back with no statement timeout carries that into
    unrelated work for the rest of its life. Both ``SET``s therefore sit inside
    the ``try`` whose ``finally`` puts them back -- if the second one raises,
    the first is still undone.
    """
    monkeypatch.setattr(pvi, "maintenance_work_mem_mb", lambda: 128)
    conn = _FakeConn(fail_on="SET maintenance_work_mem")
    with pytest.raises(RuntimeError):
        pvi._create_index(conn, KB, 1536)
    for setting in ("statement_timeout", "lock_timeout"):
        assert conn.issued(f"SET {setting} = 0"), conn.statements
        assert conn.issued(f"RESET {setting}") or conn.invalidated, conn.statements


def test_the_build_memory_is_read_before_any_session_setting_is_raised(monkeypatch):
    """The settings read is the other way into that leak, so it happens first."""
    monkeypatch.setattr(
        pvi, "maintenance_work_mem_mb", _raiser(RuntimeError("settings unreadable"))
    )
    conn = _FakeConn()
    with pytest.raises(RuntimeError, match="settings unreadable"):
        pvi._create_index(conn, KB, 1536)
    assert conn.statements == [], "nothing may be set on a session we then abandon"


def test_a_lock_release_that_fails_discards_the_connection():
    """The lock is session-scoped, so a connection that keeps it must not be pooled."""
    conn = _FakeConn(fail_on="pg_advisory_unlock")
    pvi._release_lock(conn, pvi.index_lock_relation(KB, 1536))
    assert conn.invalidated, "a pooled connection still holding the lock skips every later build"


def test_a_reset_that_fails_discards_the_connection():
    conn = _FakeConn(fail_on="RESET statement_timeout")
    pvi._reset_session_setting(conn, "statement_timeout")
    assert conn.invalidated


def test_a_build_does_not_hold_the_settings_session_idle_in_a_transaction(monkeypatch):
    """Two reads go through ``db.session``; neither ends the transaction they open.

    A build would then occupy two connections, the second idle in a transaction
    for the whole build -- and ``CREATE INDEX CONCURRENTLY`` waits for exactly
    such a transaction, so the build would be waiting on its own task.
    """
    from agentic_project_service import db as db_module

    session = MagicMock()
    monkeypatch.setattr(db_module.db, "session", session)
    _ensure(monkeypatch, _FakeConn())
    session.rollback.assert_called()


# ---------------------------------------------------------------------------
# The start-up sweep
# ---------------------------------------------------------------------------

# Distinct knowledge base ids, for the fan-out specs.
_KBS = [str(uuid.UUID(int=n)) for n in range(1, 40)]

_COUNT_QUERY = "GROUP BY 1, 2"
_CATALOG_QUERY = "i.indisvalid"


def _sweep(conn):
    return pvi.kbs_needing_a_per_kb_index(engine=_FakeEngine(conn))


_SETTINGS_QUERY = "project_settings"
_LOCK_TIMEOUT_SETTING = "set_config('lock_timeout'"


class _LockQueuedConn(_FakeConn):
    """A connection whose read of ``ai.project_settings`` is stuck behind a lock.

    Models the queue the boot sweep really meets: a long-lived snapshot holds a
    share lock on that table for hours, a single ``ALTER`` queues behind it, and
    every later reader queues behind the ``ALTER``. A reader with no
    ``lock_timeout`` gets no answer at all -- which at start-up is a process that
    never finishes booting -- so here it raises ``_NeverReturned`` rather than
    returning rows. A reader that bounded its wait first gets an answer: either
    the rows, once the queue clears, or the cancellation this fake's
    ``cancelling`` variant raises.

    The bound has to have been set **on this connection and before the read**,
    which is the whole of I1: a ``lock_timeout`` on another connection, or set
    after the statement it is meant to bound, changes nothing.
    """

    class _NeverReturned(Exception):
        """The read this fake was asked to make would still be waiting."""

    def __init__(self, rows, cancelling=False):
        super().__init__(answers=[(_SETTINGS_QUERY, rows)])
        self.bounded_at: int | None = None
        self.bound_before_the_read = False
        self._cancelling = cancelling

    def execute(self, clause, params=None):
        sql = clause.text if hasattr(clause, "text") else str(clause)
        if _LOCK_TIMEOUT_SETTING in sql:
            self.bounded_at = len(self.statements)
        elif _SETTINGS_QUERY in sql:
            if self.bounded_at is None:
                raise self._NeverReturned(
                    "the settings read is queued behind a DDL request and nothing bounds it"
                )
            self.bound_before_the_read = True
            if self._cancelling:
                self.statements.append(" ".join(sql.split()))
                self.params.append(params)
                raise RuntimeError("canceling statement due to lock timeout")
        return super().execute(clause, params)


def test_the_boot_settings_read_bounds_its_lock_wait_before_it_makes_it():
    """Without the bound this read is a start-up that never finishes.

    The bounded count beside it protects the expensive statement; this is the
    cheap one, and the only one touching a table the rest of the system takes
    DDL locks on. Proved by a connection that refuses to answer an unbounded
    read at all, so the overrides only come back if the bound was applied first
    -- not merely that the string appears somewhere.
    """
    conn = _LockQueuedConn([("VECTOR_PER_KB_INDEX_MIN_ROWS", "12000")])
    assert pvi.read_overrides(conn, *pvi._THRESHOLD_KEYS) == {
        "VECTOR_PER_KB_INDEX_MIN_ROWS": 12_000
    }
    assert conn.bound_before_the_read
    bound = conn.issued(_LOCK_TIMEOUT_SETTING)
    assert len(bound) == 1, f"one bound, on this connection: {conn.statements}"
    assert ", true)" in bound[0], (
        "transaction-local, so the bound cannot ride a pooled connection into "
        "unrelated work the way a session-level SET would"
    )
    assert conn.params[conn.bounded_at] == {"ms": str(pvi.SETTINGS_READ_LOCK_TIMEOUT_MS)}


def test_a_settings_read_the_bound_cancels_becomes_the_registry_defaults():
    """The bound turns a hang into an error, and the error into the defaults.

    Which is the whole reason a bound is safe here: a start-up that is a little
    wrong about a threshold reconciles the difference on the next indexed source
    or the next start-up, and a start-up that does not happen does not.
    """
    conn = _LockQueuedConn([("VECTOR_PER_KB_INDEX_MIN_ROWS", "12000")], cancelling=True)
    overrides = pvi.read_overrides(conn, *pvi._THRESHOLD_KEYS)
    assert overrides == {}
    assert pvi.thresholds(overrides) == (
        SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"].default,
        SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_DROP_ROWS"].default,
    )
    assert conn.rollbacks >= 1, "an aborted read must not leave the transaction open"


class _BlockedSettingsSweepConn(_FakeConn):
    """A sweep connection whose settings read is cancelled by its own bound."""

    def execute(self, clause, params=None):
        sql = clause.text if hasattr(clause, "text") else str(clause)
        if _SETTINGS_QUERY in sql:
            raise RuntimeError("canceling statement due to lock timeout")
        return super().execute(clause, params)


def test_a_settings_read_that_cannot_be_made_does_not_stop_the_boot_sweep():
    """End to end: the sweep still clears the INVALID indexes it exists for."""
    conn = _BlockedSettingsSweepConn(
        answers=[
            (_CATALOG_QUERY, [_index_row(_KBS[0], 1536, False)]),
            (_COUNT_QUERY, [(_KBS[0], 1536, 3)]),
        ]
    )
    assert _sweep(conn) == [_KBS[0]]


def test_an_abandoned_boot_count_does_not_dispatch_every_indexed_knowledge_base(caplog):
    """A count that times out says nothing about any knowledge base.

    On a database too slow to finish one ``GROUP BY`` in ``SWEEP_TIMEOUT_MS``,
    treating "no rows counted" as "no rows exist" dispatched a reconcile for
    every indexed knowledge base -- up to ``MAX_PER_KB_INDEXES`` unbounded
    builds against the database that was already struggling.
    """
    healthy = [_index_row(kb, 1536) for kb in _KBS[:5]]
    conn = _FakeConn(answers=[(_CATALOG_QUERY, healthy)], fail_on=_COUNT_QUERY)
    with caplog.at_level(logging.WARNING):
        assert _sweep(conn) == []
    assert "Could not count embeddings" in caplog.text


def test_an_invalid_index_is_still_dispatched_when_the_boot_count_is_abandoned():
    """The catalog cases need no count, which is what the count's warning promises."""
    rows = [_index_row(_KBS[0], 1536, False), _index_row(_KBS[1], 1536, True)]
    conn = _FakeConn(answers=[(_CATALOG_QUERY, rows)], fail_on=_COUNT_QUERY)
    assert _sweep(conn) == [_KBS[0]]


def test_a_knowledge_base_whose_rows_are_all_gone_is_dispatched_when_the_count_ran():
    """The other side of the guard: an emptied knowledge base has no group at all."""
    conn = _FakeConn(
        answers=[
            (_CATALOG_QUERY, [_index_row(_KBS[0], 1536)]),
            (_COUNT_QUERY, [(_KBS[1], 1536, 100)]),
        ]
    )
    assert _sweep(conn) == [_KBS[0]]


def test_the_boot_sweep_caps_how_many_reconciles_one_start_up_sets_off(caplog):
    """Every dispatch can start an unbounded build; the index cap does not bound those."""
    rows = [_index_row(kb, 1536, False) for kb in _KBS[: pvi.MAX_SWEEP_DISPATCH + 4]]
    conn = _FakeConn(answers=[(_CATALOG_QUERY, rows)])
    with caplog.at_level(logging.WARNING):
        pending = _sweep(conn)
    assert len(pending) == pvi.MAX_SWEEP_DISPATCH
    assert pending == _KBS[: pvi.MAX_SWEEP_DISPATCH]
    assert "4" in caplog.text, "the knowledge bases left for later must be counted in the log"


def test_the_capped_dispatch_keeps_the_invalid_indexes_ahead_of_the_rest():
    """An INVALID index answers no query and is maintained on every write."""
    over_threshold = _KBS[1 : pvi.MAX_SWEEP_DISPATCH + 5]
    conn = _FakeConn(
        answers=[
            (_CATALOG_QUERY, [_index_row(_KBS[0], 1536, False)]),
            (_COUNT_QUERY, [(kb, 1536, 60_000) for kb in over_threshold]),
        ]
    )
    pending = _sweep(conn)
    assert len(pending) == pvi.MAX_SWEEP_DISPATCH
    assert pending[0] == _KBS[0], "the INVALID index must not be the one left behind"


# ---------------------------------------------------------------------------
# Reconciling one knowledge base
# ---------------------------------------------------------------------------

_LOCK_QUERY = "pg_try_advisory_lock"
_FAILURE_RECORD_QUERY = "obj_description"
_FAILURE_COMMENT_DDL = "COMMENT ON INDEX"
_ROW_COUNT_QUERY = "LIMIT :cap) s"
_INDEX_COUNT_QUERY = "count(*) FROM pg_class"
_BUILD_RUNNING_QUERY = "pg_stat_progress_create_index"


def _ensure_conn(
    existing=(),
    dims_present=(1536,),
    rows_by_dims=None,
    index_count=1,
    cls=None,
    failures=0,
    **kwargs,
):
    """A connection that answers every read ``ensure_per_kb_vector_index`` makes.

    ``failures`` is what the index's own catalog comment already records, which is
    where a doomed build's history lives.
    """
    rows = dict(rows_by_dims or {})
    return (cls or _FakeConn)(
        answers=[
            (_FAILURE_RECORD_QUERY, [(_failure_comment(failures),)] if failures else []),
            (_CATALOG_QUERY, list(existing)),
            ("GROUP BY dims", [(d,) for d in dims_present]),
            (_INDEX_COUNT_QUERY, [(index_count,)]),
            (_ROW_COUNT_QUERY, lambda params: [(rows.get(int(params["dims"]), 0),)]),
            (_LOCK_QUERY, [(True,)]),
        ],
        **kwargs,
    )


# -- the disk-size log, which is the only free-space signal there can be ------


def test_the_build_log_gives_the_row_count_and_the_index_size_as_floors(monkeypatch, caplog):
    """The count stops just past the threshold, and the size is derived from it.

    There can be no free-space precheck, so this line is the operator's disk
    signal -- and for a knowledge base far over the threshold, which is what
    this feature is for, an unqualified figure understates the index by up to
    20x.
    """
    conn = _ensure_conn(rows_by_dims={1536: 10_001})
    with caplog.at_level(logging.INFO):
        outcome = _ensure(monkeypatch, conn, build_at=10_000)
    assert outcome["built"] == [pvi.per_kb_index_name(KB, 1536)]
    assert "at least 10001 rows" in caplog.text, caplog.text
    assert f"at least {pvi.estimated_index_mb(10_001, 1536)} MB" in caplog.text, caplog.text
    assert "floor" in caplog.text, "the qualifier has to be unmistakable, not implied"


def test_the_build_log_does_not_hedge_a_count_that_is_exact(monkeypatch, caplog):
    """The negative control: an unbounded count must be reported as the number it is."""
    conn = _ensure_conn(rows_by_dims={1536: 10_000})
    with caplog.at_level(logging.INFO):
        _ensure(monkeypatch, conn, build_at=10_000)
    assert "10000 rows" in caplog.text
    assert "at least" not in caplog.text, caplog.text


# -- pgvector's HNSW dimension limit -----------------------------------------


def test_a_dimension_above_pgvectors_hnsw_limit_is_never_built(monkeypatch, caplog):
    """A structurally doomed build must not be attempted, let alone repeatedly.

    pgvector refuses an HNSW index above 2,000 dimensions while ``MAX_DIMS``
    lets a 3,072-dimension model through, and dispatch runs after every source
    that finishes indexing -- so the build looped forever, each attempt holding
    a worker slot with ``statement_timeout = 0``.
    """
    conn = _ensure_conn(dims_present=(3072,), rows_by_dims={3072: 20_000})
    with caplog.at_level(logging.WARNING):
        outcome = _ensure(monkeypatch, conn)
    assert outcome["built"] == []
    assert outcome["status"] == "skipped", outcome
    assert outcome["reason"] == "dims_above_hnsw_limit", outcome
    assert conn.issued("CREATE INDEX") == [], conn.statements
    assert "3072" in caplog.text and str(pvi.MAX_HNSW_DIMS) in caplog.text, caplog.text


def test_a_dimension_above_the_hnsw_limit_is_not_dispatched_again(monkeypatch):
    """The other half of "do not retry": nothing may ask for that build."""
    _stub_settings(
        monkeypatch,
        {"VECTOR_PER_KB_INDEX_MIN_ROWS": 10_000, "VECTOR_PER_KB_INDEX_DROP_ROWS": 5_000},
    )
    conn = _ensure_conn(dims_present=(3072,), rows_by_dims={3072: 20_000})
    assert pvi.index_action(conn, KB) is None


def test_the_dimensions_below_the_limit_still_get_their_index(monkeypatch):
    conn = _ensure_conn(dims_present=(1536, 3072), rows_by_dims={1536: 20_000, 3072: 20_000})
    outcome = _ensure(monkeypatch, conn)
    assert outcome["built"] == [pvi.per_kb_index_name(KB, 1536)], outcome


def test_an_invalid_index_above_the_hnsw_limit_is_dropped_and_not_rebuilt(monkeypatch):
    """A failed build leaves an INVALID index; it costs every write and can never work."""
    name = pvi.per_kb_index_name(KB, 3072)
    conn = _ensure_conn(
        existing=[_index_row(KB, 3072, False)],
        dims_present=(3072,),
        rows_by_dims={3072: 20_000},
    )
    outcome = _ensure(monkeypatch, conn)
    assert outcome["repaired_invalid_indexes"] == [name], outcome
    assert outcome["built"] == [], outcome


# -- a build that can never succeed is given up on, durably ------------------


def test_a_failed_build_counts_itself_on_the_index_it_leaves_behind(monkeypatch):
    """The only durable record there is, and the only one that needs no migration.

    A failed ``CREATE INDEX CONCURRENTLY`` leaves the index in the catalog, so
    the attempt is recorded on the index itself. An in-process counter would
    forget on every worker restart and know nothing of the other workers, which
    is the population this reconcile runs across.
    """
    conn = _ensure_conn(rows_by_dims={1536: 20_000}, fail_on="CREATE INDEX")
    with pytest.raises(RuntimeError):
        _ensure(monkeypatch, conn)
    written = conn.issued(_FAILURE_COMMENT_DDL)
    assert len(written) == 1, conn.statements
    assert pvi.per_kb_index_name(KB, 1536) in written[0]
    assert _failure_comment(1) in written[0]


def test_each_failure_counts_on_from_the_last_one(monkeypatch):
    """The repair drop takes the record with it, so the count is carried over.

    Read before the drop and written back after the next failure -- otherwise
    every attempt records "1" and the bound is never reached.
    """
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536, False)],
        rows_by_dims={1536: 20_000},
        failures=1,
        fail_on="CREATE INDEX",
    )
    with pytest.raises(RuntimeError):
        _ensure(monkeypatch, conn)
    assert _failure_comment(2) in conn.issued(_FAILURE_COMMENT_DDL)[0], conn.statements


def test_a_build_that_succeeds_forgets_the_failures_before_it(monkeypatch):
    """Consecutive failures are what says a build is doomed, not lifetime ones.

    A build lost to a server restart or a killed worker is a failure a retry
    really does get past.
    """
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536, False)], rows_by_dims={1536: 20_000}, failures=2
    )
    outcome = _ensure(monkeypatch, conn)
    assert outcome["built"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    cleared = conn.issued(_FAILURE_COMMENT_DDL)
    assert len(cleared) == 1 and cleared[0].endswith("IS NULL"), conn.statements


def test_a_comment_this_module_did_not_write_is_not_a_build_history(monkeypatch):
    """The comment is a place anyone may write, and giving up needs evidence.

    An index someone has annotated by hand must not be read as doomed, and must
    not be read as having failed some number of times either.
    """
    conn = _ensure_conn(existing=[_index_row(KB, 1536, False)], rows_by_dims={1536: 20_000})
    conn.answers.insert(0, (_FAILURE_RECORD_QUERY, [("dropping this on Monday, see ticket 41",)]))
    assert pvi.recorded_build_failures(conn, KB, 1536) == 0
    outcome = _ensure(monkeypatch, conn)
    assert outcome["built"] == [pvi.per_kb_index_name(KB, 1536)], outcome


def _transient_exc(sqlstate: str = "55P03", statement: str = "CREATE INDEX"):
    """A database failure the pg_search path already classifies as transient.

    55P03 is ``lock_not_available``: a genuine lock conflict, which is what a
    concurrent build meets when another caller is holding the table -- and the
    Celery task retries it six times.
    """
    from sqlalchemy.exc import OperationalError

    class _Orig(Exception):
        pass

    orig = _Orig("canceling statement due to lock timeout")
    orig.sqlstate = sqlstate
    return OperationalError(statement, {}, orig)


class _CatalogStateConn(_FakeConn):
    """A connection whose index comment survives the reconcile that wrote it.

    The failure bound lives in the index's own ``pg_class`` comment, so a spec
    about the bound being *reached* has to read back what the previous attempt
    wrote. This keeps that one piece of catalog state, which is what lets a
    sequence of reconciles be driven the way the ones against a real server were.
    """

    def __init__(self, *args, comment=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.comment = comment

    def execute(self, clause, params=None):
        sql = clause.text if hasattr(clause, "text") else str(clause)
        if _FAILURE_COMMENT_DDL in sql:
            self.statements.append(" ".join(sql.split()))
            self.params.append(params)
            self.comment = None if sql.rstrip().endswith("IS NULL") else sql.split("'")[-2]
            return _Result([])
        if _FAILURE_RECORD_QUERY in sql:
            self.statements.append(" ".join(sql.split()))
            self.params.append(params)
            return _Result([(self.comment,)] if self.comment else [])
        return super().execute(clause, params)


def test_a_repair_drop_that_fails_counts_the_attempt_too(monkeypatch):
    """The other way a reconcile of an INVALID index ends without building.

    The attempt is counted in the build, so a repair drop that raises used to
    return before anything counted it: five consecutive reconciles against a real
    server each recorded ``failures=1`` and each asked for a build again, which is
    the unbounded drop-rebuild-fail loop with the bound stepped around.
    """
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536, False)],
        rows_by_dims={1536: 20_000},
        failures=1,
        fail_on="DROP INDEX",
    )
    with pytest.raises(RuntimeError):
        _ensure(monkeypatch, conn)
    assert conn.issued("CREATE INDEX") == [], "the drop failed; nothing was rebuilt"
    written = conn.issued(_FAILURE_COMMENT_DDL)
    assert written, "an uncounted attempt is an unbounded loop"
    assert _failure_comment(2) in written[0], written


def test_repair_drops_that_keep_failing_reach_the_give_up_bound(monkeypatch, caplog):
    """The loop itself, driven the way it was driven against a real server.

    Reconcile after reconcile, with the index's comment carried between them the
    way the catalog carries it. Without the attempt being counted this never
    terminates: every run reports ``failures=1`` and asks for a build again.
    """
    outcomes = []
    conn = _CatalogStateConn(
        answers=[
            (_CATALOG_QUERY, [_index_row(KB, 1536, False)]),
            ("GROUP BY dims", [(1536,)]),
            (_INDEX_COUNT_QUERY, [(1,)]),
            (_ROW_COUNT_QUERY, [(20_000,)]),
            (_LOCK_QUERY, [(True,)]),
        ],
        fail_on="DROP INDEX",
    )
    for _ in range(pvi.MAX_CONSECUTIVE_BUILD_FAILURES):
        with pytest.raises(RuntimeError):
            _ensure(monkeypatch, conn)
    with caplog.at_level(logging.ERROR):
        outcomes.append(_ensure(monkeypatch, conn))
    assert outcomes[-1]["reason"] == "build_repeatedly_failed", outcomes
    assert pvi.index_action(conn, KB) is None, "and nothing dispatches it again"
    assert "by hand" in caplog.text


def test_a_transient_build_failure_does_not_burn_an_attempt(monkeypatch):
    """Three failures disable the index until an operator drops it by hand.

    So the bound has to count only the failures a retry cannot get past. The task
    that runs this classifies a lock conflict as transient and retries it six
    times; one contention episode outlasting three of those retries would
    otherwise write "3 consecutive failed attempts" and turn the index off for
    good.
    """
    conn = _ensure_conn(
        rows_by_dims={1536: 20_000}, fail_on="CREATE INDEX", exc=_transient_exc()
    )
    with pytest.raises(Exception, match="lock timeout"):
        _ensure(monkeypatch, conn)
    assert conn.issued(_FAILURE_COMMENT_DDL) == [], conn.statements


def test_a_transient_build_failure_keeps_the_attempts_already_on_record(monkeypatch):
    """Not counting it must not un-count the ones before it either.

    The repair drop takes the record away with the index it is written on, so a
    transient failure after a repair would otherwise reset the count to zero and
    hand back the whole budget on every contention episode.
    """
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536, False)],
        rows_by_dims={1536: 20_000},
        failures=2,
        fail_on="CREATE INDEX",
        exc=_transient_exc(),
    )
    with pytest.raises(Exception, match="lock timeout"):
        _ensure(monkeypatch, conn)
    written = conn.issued(_FAILURE_COMMENT_DDL)
    assert written and _failure_comment(2) in written[0], written


def test_a_transient_repair_drop_failure_does_not_burn_an_attempt(monkeypatch):
    """The repair drop counts under the same rule, so it does not count this one.

    The count on record may be rewritten -- it is the same number -- but it may
    not advance: a lock conflict on the repair drop is what the task's own six
    retries are for.
    """
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536, False)],
        rows_by_dims={1536: 20_000},
        failures=1,
        fail_on="DROP INDEX",
        exc=_transient_exc(statement="DROP INDEX"),
    )
    with pytest.raises(Exception, match="lock timeout"):
        _ensure(monkeypatch, conn)
    for written in conn.issued(_FAILURE_COMMENT_DDL):
        assert _failure_comment(1) in written, written
        assert _failure_comment(2) not in written, written


def test_a_build_that_has_failed_the_limit_is_not_attempted_again(monkeypatch, caplog):
    """Where the drop-rebuild-fail loop stops.

    Measured against a real server on a build that could never succeed: three
    reconciles each dropped the INVALID index, rebuilt it and failed, and the
    fourth did it again, once per source that finished indexing. Now the third
    failure is the last, and the INVALID index stays as the record of it -- an
    operator dropping it by hand is what lets a later reconcile try again.
    """
    name = pvi.per_kb_index_name(KB, 1536)
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536, False)],
        rows_by_dims={1536: 20_000},
        failures=pvi.MAX_CONSECUTIVE_BUILD_FAILURES,
    )
    with caplog.at_level(logging.ERROR):
        outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "skipped" and outcome["reason"] == "build_repeatedly_failed"
    assert outcome["build_repeatedly_failed"] == [1536], outcome
    assert outcome["built"] == [] and outcome["dropped"] == [], outcome
    assert conn.issued("DROP INDEX") == [], "nothing to gain by dropping and rebuilding again"
    assert conn.issued("CREATE INDEX") == []
    assert pvi.outcome_needs_another_attempt(outcome) is False, "a retry cannot get past this"
    assert name in caplog.text and "by hand" in caplog.text
    assert [r for r in caplog.records if r.levelno >= logging.ERROR], caplog.text


def test_a_build_that_has_failed_the_limit_is_not_dispatched_either(monkeypatch):
    """Every source that finishes indexing would otherwise dispatch it again."""
    _stub_settings(
        monkeypatch,
        {"VECTOR_PER_KB_INDEX_MIN_ROWS": 10_000, "VECTOR_PER_KB_INDEX_DROP_ROWS": 5_000},
    )
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536, False)],
        rows_by_dims={1536: 20_000},
        failures=pvi.MAX_CONSECUTIVE_BUILD_FAILURES,
    )
    assert pvi.index_action(conn, KB) is None
    conn_below_limit = _ensure_conn(
        existing=[_index_row(KB, 1536, False)],
        rows_by_dims={1536: 20_000},
        failures=pvi.MAX_CONSECUTIVE_BUILD_FAILURES - 1,
    )
    assert pvi.index_action(conn_below_limit, KB) == "build", "one attempt still to go"


def test_an_invalid_index_on_a_shrunken_knowledge_base_asks_for_a_drop(monkeypatch):
    """An INVALID index used to mean "build" whatever the row count said.

    A knowledge base that fell below the drop threshold while its build was
    failing does not want that index rebuilt at a size it no longer is. The
    reconcile drops it either way, so asking for a build was asking for something
    it would decline.
    """
    _stub_settings(
        monkeypatch,
        {"VECTOR_PER_KB_INDEX_MIN_ROWS": 10_000, "VECTOR_PER_KB_INDEX_DROP_ROWS": 5_000},
    )
    conn = _ensure_conn(existing=[_index_row(KB, 1536, False)], rows_by_dims={1536: 100})
    assert pvi.index_action(conn, KB) == "drop"


def test_an_invalid_index_above_the_hnsw_limit_asks_for_a_drop_not_a_build(monkeypatch):
    """pgvector cannot build this one at any row count; the reconcile only drops it."""
    _stub_settings(
        monkeypatch,
        {"VECTOR_PER_KB_INDEX_MIN_ROWS": 10_000, "VECTOR_PER_KB_INDEX_DROP_ROWS": 5_000},
    )
    conn = _ensure_conn(
        existing=[_index_row(KB, 3072, False)], dims_present=(3072,), rows_by_dims={3072: 20_000}
    )
    assert pvi.index_action(conn, KB) == "drop"


def test_a_project_at_the_index_cap_dispatches_no_further_builds(monkeypatch):
    """The task at the cap can only report ``skipped``, once per indexed source."""
    _stub_settings(
        monkeypatch,
        {"VECTOR_PER_KB_INDEX_MIN_ROWS": 10_000, "VECTOR_PER_KB_INDEX_DROP_ROWS": 5_000},
    )
    at_cap = _ensure_conn(rows_by_dims={1536: 20_000}, index_count=pvi.MAX_PER_KB_INDEXES)
    assert pvi.index_action(at_cap, KB) is None
    below_cap = _ensure_conn(rows_by_dims={1536: 20_000}, index_count=pvi.MAX_PER_KB_INDEXES - 1)
    assert pvi.index_action(below_cap, KB) == "build"


# -- the drop threshold is reachable -----------------------------------------


def test_a_knowledge_base_at_exactly_the_drop_threshold_loses_its_index(monkeypatch):
    """``rows < drop_below`` made a drop threshold of 0 unreachable forever.

    The sweep then re-dispatched that knowledge base on every boot, nothing
    logged why, and the index stayed.
    """
    name = pvi.per_kb_index_name(KB, 1536)
    conn = _ensure_conn(existing=[_index_row(KB, 1536)], rows_by_dims={1536: 5_000})
    outcome = _ensure(monkeypatch, conn, drop_below=5_000)
    assert outcome["dropped"] == [name], outcome


def test_the_reconcile_check_agrees_with_the_drop_at_the_threshold(monkeypatch):
    _stub_settings(
        monkeypatch,
        {"VECTOR_PER_KB_INDEX_MIN_ROWS": 10_000, "VECTOR_PER_KB_INDEX_DROP_ROWS": 5_000},
    )
    conn = _ensure_conn(existing=[_index_row(KB, 1536)], rows_by_dims={1536: 5_000})
    assert pvi.index_action(conn, KB) == "drop"


# -- a declined INVALID repair is work left over, not a success --------------


def test_a_declined_invalid_repair_asks_to_be_run_again(monkeypatch, caplog):
    """Holding the lock while a build runs means an orphan backend from a dead worker.

    The index stays INVALID -- answering no query, maintained on every insert --
    until something comes back to it, and until now nothing did.
    """
    name = pvi.per_kb_index_name(KB, 1536)
    conn = _ensure_conn(existing=[_index_row(KB, 1536, False)], rows_by_dims={1536: 20_000})
    conn.answers.insert(0, (_BUILD_RUNNING_QUERY, [(1,)]))
    with caplog.at_level(logging.WARNING):
        outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "building", outcome
    assert outcome["reschedule"] is True, outcome
    assert pvi.outcome_needs_another_attempt(outcome) is True
    assert outcome["index"] == name
    assert outcome["built"] == [] and outcome["dropped"] == [], outcome
    assert "repaired_invalid_indexes" not in outcome, outcome
    assert conn.issued("DROP INDEX") == [], "the running build's index must stay"
    assert [r for r in caplog.records if r.levelno >= logging.WARNING], caplog.text


def test_ordinary_lock_contention_is_not_work_left_over(monkeypatch):
    """Whoever holds the lock is doing this index's work and will finish it."""
    conn = _ensure_conn(rows_by_dims={1536: 20_000})
    conn.answers.insert(0, (_LOCK_QUERY, [(False,)]))
    outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "building", outcome
    assert pvi.outcome_needs_another_attempt(outcome) is False, outcome
    assert outcome["built"] == [], outcome


# -- one dimension's outcome does not decide the others ----------------------


def test_a_dimension_whose_lock_is_held_does_not_abandon_the_others(monkeypatch):
    conn = _ensure_conn(dims_present=(768, 1536), rows_by_dims={768: 20_000, 1536: 20_000})
    conn.answers.insert(0, (_LOCK_QUERY, lambda p: [("_1536" not in p["relation"],)]))
    outcome = _ensure(monkeypatch, conn)
    assert outcome["built"] == [pvi.per_kb_index_name(KB, 768)], outcome
    assert outcome["status"] == "building", outcome


def test_the_index_cap_does_not_abandon_another_dimensions_drop(monkeypatch):
    """The cap is reached at 768; 1536's index still has to go."""
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536)],
        dims_present=(768, 1536),
        rows_by_dims={768: 20_000, 1536: 0},
        index_count=pvi.MAX_PER_KB_INDEXES,
    )
    outcome = _ensure(monkeypatch, conn)
    assert outcome["dropped"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    assert outcome["status"] == "skipped" and outcome["reason"] == "index_cap_reached", outcome


# ---------------------------------------------------------------------------
# Dropping a deleted knowledge base's indexes
# ---------------------------------------------------------------------------


class _LostConnection(Exception):
    """What SQLAlchemy raises for a connection that went away, as this module reads it."""

    connection_invalidated = True


def _drop_conn(dims=(1536,), cls=None, **kwargs):
    return (cls or _FakeConn)(
        answers=[
            (_CATALOG_QUERY, [_index_row(KB, d) for d in dims]),
            (_LOCK_QUERY, [(True,)]),
        ],
        **kwargs,
    )


def test_a_drop_reports_every_index_it_dropped():
    conn = _drop_conn(dims=(768, 1536))
    outcome = pvi.drop_per_kb_vector_indexes(KB, engine=_FakeEngine(conn))
    assert outcome == {
        "status": "dropped",
        "indexes": [pvi.per_kb_index_name(KB, 768), pvi.per_kb_index_name(KB, 1536)],
    }


def test_a_permanent_drop_failure_is_not_reported_as_a_success(caplog):
    """The knowledge base row is already gone, so nothing ever comes back to this.

    A non-transient failure used to return ``status: "partial"`` -- the task
    marked SUCCESS, one WARNING, and an index left behind that is named after a
    knowledge base that no longer exists and is maintained on every write. The
    give-up log the drop task advertises fires on retry exhaustion only, which a
    permanent failure never reaches.
    """
    name = pvi.per_kb_index_name(KB, 1536)
    conn = _drop_conn(fail_on="DROP INDEX")
    with caplog.at_level(logging.ERROR):
        with pytest.raises(pvi.PerKbVectorIndexDropFailed) as raised:
            pvi.drop_per_kb_vector_indexes(KB, engine=_FakeEngine(conn))
    assert raised.value.failed_indexes == [name]
    assert name in str(raised.value)
    assert [r for r in caplog.records if r.levelno >= logging.ERROR], caplog.text
    assert name in caplog.text


def test_a_drop_that_fails_at_one_dimension_still_drops_the_others():
    failing = pvi.per_kb_index_name(KB, 768)
    surviving = pvi.per_kb_index_name(KB, 1536)
    conn = _drop_conn(dims=(768, 1536), fail_on=failing)
    with pytest.raises(pvi.PerKbVectorIndexDropFailed) as raised:
        pvi.drop_per_kb_vector_indexes(KB, engine=_FakeEngine(conn))
    assert raised.value.failed_indexes == [failing]
    assert raised.value.dropped_indexes == [surviving]


def test_a_transient_drop_failure_is_re_raised_untouched_for_the_retry(caplog):
    """A retry can get past this one, so it must not be turned into a permanent failure."""
    conn = _drop_conn(fail_on="DROP INDEX", exc=_LostConnection("server closed the connection"))
    with caplog.at_level(logging.ERROR):
        with pytest.raises(_LostConnection):
            pvi.drop_per_kb_vector_indexes(KB, engine=_FakeEngine(conn))
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], caplog.text


# ---------------------------------------------------------------------------
# A drop waits as long as a build does
# ---------------------------------------------------------------------------


class _TimeoutCancellingConn(_FakeConn):
    """A server with a timeout of its own, as a role or database has.

    ``DROP INDEX CONCURRENTLY`` waits for every transaction whose snapshot could
    still be using the index, and ``CREATE INDEX CONCURRENTLY`` waits for every
    transaction that could still write a row it has not seen, so a timeout the
    DDL did not ask for cancels it mid-wait. Here that is modelled where it
    happens: the concurrent DDL raises unless ``SETTING`` was lifted on this
    connection first, and the ``RESET`` puts it back, so a second one is
    unprotected again if the lifting is not per-statement.

    Against a real server the cancelled drop leaves the index ``indisvalid =
    false`` -- maintained on every insert, answering no query -- which the
    deleted knowledge base's path can never come back to.

    ``SETTING`` is the name of the timeout, because there are two and they cover
    different waits: ``statement_timeout`` does not cover a lock wait at all, and
    both of these waits are lock waits (on a virtual transaction id). Subclassed
    rather than parametrized inside ``execute`` so each subclass is a server with
    exactly one of them set, which is how the bound is proved to be lifted for
    its own reason and not by the other one's ``SET``.
    """

    SETTING = "statement_timeout"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout_lifted = False
        self.cancelled: list[str] = []

    def execute(self, clause, params=None):
        sql = clause.text if hasattr(clause, "text") else str(clause)
        if f"SET {self.SETTING} = 0" in sql:
            self.timeout_lifted = True
        elif f"RESET {self.SETTING}" in sql:
            self.timeout_lifted = False
        elif "INDEX CONCURRENTLY" in sql and not self.timeout_lifted:
            self.statements.append(" ".join(sql.split()))
            self.cancelled.append(sql)
            raise RuntimeError(f"canceling statement due to {self.SETTING}")
        return super().execute(clause, params)


class _LockTimeoutCancellingConn(_TimeoutCancellingConn):
    """The same server with a role-level ``lock_timeout`` instead.

    ``statement_timeout`` does not bound a lock wait, and both phases these two
    statements block in are lock waits: ``CREATE INDEX CONCURRENTLY``'s
    ``WaitForLockers`` and ``DROP INDEX CONCURRENTLY``'s wait for conflicting
    snapshots both wait on a ``virtualxid`` lock. Measured against a real server
    with a role-level ``lock_timeout`` of 2 s and one open write transaction:
    the build failed in 2.02 s, five times out of five, and the drop in 2.01 s,
    leaving the index ``indisvalid = false, indisready = true``.
    """

    SETTING = "lock_timeout"


_TIMEOUT_SERVERS = [_TimeoutCancellingConn, _LockTimeoutCancellingConn]


@pytest.mark.parametrize("cls", _TIMEOUT_SERVERS)
def test_a_deleted_knowledge_bases_drops_are_not_cancelled_by_a_timeout(cls):
    """This is the path a cancelled drop strands for good.

    Both dimensions, because the timeout is lifted per drop: the ``RESET`` after
    the first one leaves the second exposed unless it lifts it again.
    """
    conn = _drop_conn(dims=(768, 1536), cls=cls)
    outcome = pvi.drop_per_kb_vector_indexes(KB, engine=_FakeEngine(conn))
    assert conn.cancelled == [], conn.statements
    assert outcome == {
        "status": "dropped",
        "indexes": [pvi.per_kb_index_name(KB, 768), pvi.per_kb_index_name(KB, 1536)],
    }
    assert not conn.timeout_lifted, (
        "the timeout has to be back before this connection can return to the pool"
    )


@pytest.mark.parametrize("cls", _TIMEOUT_SERVERS)
def test_a_drop_below_the_threshold_is_not_cancelled_by_a_timeout(monkeypatch, cls):
    conn = _ensure_conn(existing=[_index_row(KB, 1536)], rows_by_dims={1536: 0}, cls=cls)
    outcome = _ensure(monkeypatch, conn)
    assert conn.cancelled == [], conn.statements
    assert outcome["dropped"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    assert not conn.timeout_lifted


@pytest.mark.parametrize("cls", _TIMEOUT_SERVERS)
def test_the_repair_of_an_invalid_index_is_not_cancelled_by_a_timeout(monkeypatch, cls):
    """A cancelled repair drop is the one that loops: it leaves what it came to clear.

    The build that follows it is covered by the same run: it waits in
    ``WaitForLockers`` for the same reason and the fake cancels either statement.
    """
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536, False)],
        rows_by_dims={1536: 20_000},
        cls=cls,
    )
    outcome = _ensure(monkeypatch, conn)
    assert conn.cancelled == [], conn.statements
    assert outcome["repaired_invalid_indexes"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    assert outcome["built"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    assert not conn.timeout_lifted


def test_the_catalog_lookups_match_the_index_name_and_not_a_like_wildcard():
    """``_`` is a LIKE wildcard and every one of these names carries two of them.

    Without ``ESCAPE`` the patterns match names these functions' docstrings
    promise they cannot -- another prefix with any character where an underscore
    belongs. Asserted on the emitted pattern and clause rather than on rows,
    because the filtering happens in the server; the escaping itself is verified
    against a real PostgreSQL catalog in the pg_search tier.
    """
    conn = _FakeConn()
    pvi.existing_per_kb_indexes(conn, KB)
    pvi.per_kb_index_count(conn)
    _sweep(_FakeConn())
    assert conn.params[0]["prefix"] == f"hnsw\\_kb\\_{KB_HEX}\\_%"
    assert conn.params[1]["prefix"] == "hnsw\\_kb\\_%"
    for sql in conn.statements:
        assert "LIKE :prefix ESCAPE '\\'" in sql, sql


# ---------------------------------------------------------------------------
# What the progress hook reports
# ---------------------------------------------------------------------------


def _events(monkeypatch, conn, **kwargs):
    """Every progress callback the ensure makes, as ``(status, fields)``."""
    seen: list[tuple[str, dict]] = []
    _ensure(
        monkeypatch,
        conn,
        on_progress=lambda status, **fields: seen.append((status, fields)),
        **kwargs,
    )
    return seen


def test_the_progress_hook_carries_the_dimension_and_the_row_count_of_a_build(monkeypatch):
    """The row count is the field the caller's log cannot source any other way.

    A status on its own says a build happened; what an operator asks next is how
    big the knowledge base was when it crossed the threshold. The dimension is
    recoverable from the index name, the count is not.
    """
    conn = _ensure_conn(rows_by_dims={1536: 10_000})
    assert _events(monkeypatch, conn, build_at=10_000) == [
        ("building", {"dims": 1536, "rows": 10_000, "rows_are_a_floor": False})
    ]


def test_the_progress_hook_carries_them_for_a_drop_too(monkeypatch):
    conn = _ensure_conn(existing=[_index_row(KB, 1536)], rows_by_dims={1536: 5_000})
    assert _events(monkeypatch, conn, drop_below=5_000) == [
        ("dropping", {"dims": 1536, "rows": 5_000, "rows_are_a_floor": False})
    ]


def test_the_progress_hook_says_when_the_row_count_is_only_a_floor(monkeypatch):
    """The same bounded count that made the disk log understate by 20x.

    Reported as a number with no qualifier it would put that understatement
    straight back, in a structured field this time.
    """
    conn = _ensure_conn(rows_by_dims={1536: 10_001})
    assert _events(monkeypatch, conn, build_at=10_000) == [
        ("building", {"dims": 1536, "rows": 10_001, "rows_are_a_floor": True})
    ]


def test_a_failing_progress_hook_cannot_fail_the_build(monkeypatch, caplog):
    """It is a logging hook. Losing an event is acceptable; losing the index is not."""
    conn = _ensure_conn(rows_by_dims={1536: 20_000})
    with caplog.at_level(logging.WARNING):
        outcome = _ensure(monkeypatch, conn, on_progress=_raiser(RuntimeError("log sink down")))
    assert outcome["built"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    assert "log sink down" in caplog.text
    # Nothing else reports this failure, so the traceback is the only way to find
    # out where in the recorder it happened -- as the lock release beside it does.
    failed = [r for r in caplog.records if "progress hook failed" in r.getMessage()]
    assert failed and all(r.exc_info for r in failed), caplog.text
