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


def test_create_ddl_predicate_names_the_kb_the_dimension_and_the_one_population():
    """``ai.embeddings`` is polymorphic, so an index over all of it is not the index.

    Without ``item_table`` the index spans four item tables and the chunk search
    steered onto it walks entries that cannot join -- measured recall 0.858 against
    0.383 for the same 1,000 chunk rows once 9,000 document rows shared the index.
    """
    ddl = pvi.per_kb_index_ddl(KB, 1536)
    assert f"WHERE knowledge_base_id = '{KB}' AND dims = 1536" in ddl
    assert f"AND item_table = '{pvi.PER_KB_INDEX_ITEM_TABLE}'" in ddl


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


# "whatever definition the service emits today", so a helper's default is an index
# this version built rather than a drifted one.
_CURRENT = object()



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


def _recorded(comment: str | None) -> tuple[int, str | None]:
    """``(permanent failures, definition fingerprint)`` a written comment records.

    Read back through the service's own parsers rather than by matching the
    wording, so a reworded comment the parsers still read keeps these passing --
    and so a spec says "two failures are on record" about what the module can read
    back, not about a substring.
    """
    return pvi.build_failures_in(comment), pvi.definition_fingerprint_in(comment)


def _current_fingerprint(kb_id: str, dims: int) -> str:
    return pvi.per_kb_index_fingerprint(kb_id, dims)


# The one catalog row shape: name, validity, and the comment carrying both build
# counts and the definition the index was built from.
def _index_row(
    kb_id: str,
    dims: int,
    valid: bool = True,
    failures: int = 0,
    interrupted: int = 0,
    fingerprint=_CURRENT,
):
    if fingerprint is _CURRENT:
        fingerprint = _current_fingerprint(kb_id, dims)
    return (
        pvi.per_kb_index_name(kb_id, dims),
        valid,
        pvi.per_kb_index_comment(failures, interrupted, fingerprint),
    )


# The sweep reads the same row; the name is kept because the sweep specs are about
# what it concludes from the build history, which is the third column.
_sweep_index_row = _index_row


def _with_history(row, failures: int = 0, interrupted: int = 0):
    """The same catalog row, with a build history written into its comment.

    Keeps whatever definition fingerprint the row already carries, so saying "two
    failures are on record" does not accidentally also say "built from a definition
    this version no longer emits".
    """
    if not (failures or interrupted):
        return row
    relname, valid, comment = row
    return (
        relname,
        valid,
        pvi.per_kb_index_comment(failures, interrupted, pvi.definition_fingerprint_in(comment)),
    )


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
    assert [st for st in conn.statements if not st.startswith(_FAILURE_COMMENT_DDL)] == [
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
    assert conn.rollbacks >= 1, "and the loop it is shared with has to be able to go on"


def test_a_reset_that_fails_discards_the_connection():
    conn = _FakeConn(fail_on="RESET statement_timeout")
    pvi._reset_session_setting(conn, "statement_timeout")
    assert conn.invalidated
    assert conn.rollbacks >= 1


class _DiscardedConnectionConn(_FakeConn):
    """A connection that behaves as a real one does after ``invalidate()``.

    Two server rules, both measured against a real server on the AUTOCOMMIT
    connection this loop runs on, both modelled here because a fake that only
    records the calls cannot tell a fixed discard from a broken one.

    **The rollback is what lets the handle reconnect.** After ``invalidate()``
    the next statement raises ``PendingRollbackError`` ("Can't reconnect until
    invalid transaction is rolled back"), and it goes on raising until the
    connection is rolled back. ``PendingRollbackError`` is not classified as a
    transient database error either, so a connection lost mid-loop would fail the
    run without the retry that exists for exactly that.

    **Reconnecting loses the AUTOCOMMIT execution option.** It is held against
    the DBAPI connection the handle had, and SQLAlchemy does not carry it onto
    the new one, so a reconnected handle is inside an implicit transaction until
    the option is applied again -- and ``CREATE INDEX CONCURRENTLY`` and
    ``DROP INDEX CONCURRENTLY`` both refuse a transaction block, with
    ``25001``. Measured live on the scenario ``_discard_connection``'s own
    docstring gives: drop at 4 dimensions, connection lost at
    ``pg_advisory_unlock``, build at 8 -- the drop succeeded, the loop reached the
    next dimension, and the build raised
    ``ActiveSqlTransaction: CREATE INDEX CONCURRENTLY cannot run inside a
    transaction block``, which ``is_transient_db_error`` does not recognise.

    So the rule is read off the *statement* -- CONCURRENTLY or not -- against the
    connection's own transaction state, the way the server reads it, rather than
    from whether some method was called. That is what makes a missing
    re-application of the option fail a spec here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discarded = False
        # An opened connection is put into AUTOCOMMIT by ``_autocommit_connection``;
        # these fakes are handed straight to the code under test, so they start
        # there too.
        self.autocommit = True

    def invalidate(self):
        super().invalidate()
        self.discarded = True
        # A new backend, and the execution option did not come with it.
        self.autocommit = False

    def rollback(self):
        super().rollback()
        self.discarded = False

    def execution_options(self, **kwargs):
        if kwargs.get("isolation_level") == "AUTOCOMMIT":
            self.autocommit = True
        return self

    def execute(self, clause, params=None):
        if self.discarded:
            raise RuntimeError("Can't reconnect until invalid transaction is rolled back")
        sql = clause.text if hasattr(clause, "text") else str(clause)
        if "CONCURRENTLY" in sql and not self.autocommit:
            raise RuntimeError(
                "ActiveSqlTransaction: CREATE INDEX CONCURRENTLY cannot run inside a "
                "transaction block"
            )
        return super().execute(clause, params)


def test_a_discarded_loop_connection_is_given_back_so_the_rest_of_the_loop_runs(monkeypatch):
    """The reconcile shares one connection across every dimension in play.

    So a ``RESET`` that fails at the first dimension must not take the second with
    it: a knowledge base that has just changed embedding model has an index to drop
    at the old dimension and one to build at the new one.
    """
    conn = _ensure_conn(
        existing=[_index_row(KB, 768), _index_row(KB, 1536)],
        dims_present=(768, 1536),
        rows_by_dims={768: 0, 1536: 0},
        cls=_DiscardedConnectionConn,
        fail_on="RESET lock_timeout",
    )
    outcome = _ensure(monkeypatch, conn)
    assert conn.invalidated, "the connection that could not be reset must not be pooled"
    assert outcome["dropped"] == [
        pvi.per_kb_index_name(KB, 768),
        pvi.per_kb_index_name(KB, 1536),
    ], outcome


def test_a_discarded_connection_can_still_run_concurrently_ddl(monkeypatch):
    """The scenario ``_discard_connection`` exists for, end to end.

    A knowledge base that has changed embedding model: an index to drop at the old
    dimension, one to build at the new one, and the connection lost at the
    ``pg_advisory_unlock`` in between. Reconnecting gets a new backend without the
    AUTOCOMMIT execution option, and both ``DROP INDEX CONCURRENTLY`` and
    ``CREATE INDEX CONCURRENTLY`` refuse a transaction block -- so the dimension
    after the discard has to find the handle back in AUTOCOMMIT, not merely usable.

    Measured live before the option was re-applied: the 768 drop succeeded, the
    loop reached 1536, and the build raised ``25001``, which
    ``is_transient_db_error`` does not recognise -- so the one retry this whole
    function exists to preserve was lost.
    """
    conn = _ensure_conn(
        existing=[_index_row(KB, 768)],
        dims_present=(768, 1536),
        rows_by_dims={768: 0, 1536: 10_001},
        cls=_DiscardedConnectionConn,
        fail_on="pg_advisory_unlock",
    )
    outcome = _ensure(monkeypatch, conn)
    assert conn.invalidated, "the harness has to actually lose the backend"
    assert outcome["dropped"] == [pvi.per_kb_index_name(KB, 768)], outcome
    assert outcome["built"] == [pvi.per_kb_index_name(KB, 1536)], outcome


def test_the_discard_puts_the_handle_back_into_autocommit_itself(monkeypatch):
    """Not only via a caller that happens to re-open the connection.

    ``_discard_connection`` is reached from three places -- a lock release that
    failed, a session ``RESET`` that failed, and either of those inside a build's
    ``finally`` -- and in all three the caller goes on using the same handle. So
    the option is applied here rather than being left to whoever notices.
    """
    conn = _DiscardedConnectionConn()
    pvi._discard_connection(conn)
    assert conn.autocommit, "CREATE INDEX CONCURRENTLY cannot run on it otherwise"
    assert conn.rollbacks >= 1, "and the rollback is what lets it reconnect at all"


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
# One statement answers every question this module asks of ``pg_class`` about its
# own indexes -- a knowledge base's, the whole project's, the boot sweep's -- and
# the only thing that tells them apart is the LIKE prefix, which is what tells
# them apart on the server too. So the fakes answer it from the ``prefix`` bind
# rather than from three fragments of SQL.
_CATALOG_QUERY = "i.indisvalid"
_SWEEP_CATALOG_QUERY = _CATALOG_QUERY
_WHOLE_PROJECT_PREFIX = pvi._like_prefix(pvi.INDEX_NAME_PREFIX)


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
            (_SWEEP_CATALOG_QUERY, [_sweep_index_row(_KBS[0], 1536, False)]),
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
    healthy = [_sweep_index_row(kb, 1536) for kb in _KBS[:5]]
    conn = _FakeConn(answers=[(_SWEEP_CATALOG_QUERY, healthy)], fail_on=_COUNT_QUERY)
    with caplog.at_level(logging.WARNING):
        assert _sweep(conn) == []
    assert "Could not count embeddings" in caplog.text


def test_an_invalid_index_is_still_dispatched_when_the_boot_count_is_abandoned():
    """The catalog cases need no count, which is what the count's warning promises."""
    rows = [_sweep_index_row(_KBS[0], 1536, False), _sweep_index_row(_KBS[1], 1536, True)]
    conn = _FakeConn(answers=[(_SWEEP_CATALOG_QUERY, rows)], fail_on=_COUNT_QUERY)
    assert _sweep(conn) == [_KBS[0]]


def test_a_knowledge_base_whose_rows_are_all_gone_is_dispatched_when_the_count_ran():
    """The other side of the guard: an emptied knowledge base has no group at all."""
    conn = _FakeConn(
        answers=[
            (_SWEEP_CATALOG_QUERY, [_sweep_index_row(_KBS[0], 1536)]),
            (_COUNT_QUERY, [(_KBS[1], 1536, 100)]),
        ]
    )
    assert _sweep(conn) == [_KBS[0]]


def test_the_boot_sweep_caps_how_many_reconciles_one_start_up_sets_off(caplog):
    """Every dispatch can start an unbounded build; the index cap does not bound those."""
    rows = [_sweep_index_row(kb, 1536, False) for kb in _KBS[: pvi.MAX_SWEEP_DISPATCH + 4]]
    conn = _FakeConn(answers=[(_SWEEP_CATALOG_QUERY, rows)])
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
            (_SWEEP_CATALOG_QUERY, [_sweep_index_row(_KBS[0], 1536, False)]),
            (_COUNT_QUERY, [(kb, 1536, 60_000) for kb in over_threshold]),
        ]
    )
    pending = _sweep(conn)
    assert len(pending) == pvi.MAX_SWEEP_DISPATCH
    assert pending[0] == _KBS[0], "the INVALID index must not be the one left behind"


def test_the_boot_sweep_leaves_out_an_index_at_the_failure_bound(caplog):
    """A start-up budget spent on builds nothing will attempt.

    Measured against a real server: ``failures=3``, ``index_action`` already
    ``None``, and the sweep returned the knowledge base anyway. The list is
    INVALID-first and truncated to ``MAX_SWEEP_DISPATCH``, so that many given-up
    indexes consume the whole budget on every boot while a repairable one is never
    reached.
    """
    doomed = [
        _sweep_index_row(kb, 1536, False, failures=pvi.MAX_CONSECUTIVE_BUILD_FAILURES)
        for kb in _KBS[: pvi.MAX_SWEEP_DISPATCH]
    ]
    repairable = _sweep_index_row(_KBS[pvi.MAX_SWEEP_DISPATCH], 1536, False)
    conn = _FakeConn(answers=[(_SWEEP_CATALOG_QUERY, doomed + [repairable])], fail_on=_COUNT_QUERY)
    with caplog.at_level(logging.WARNING):
        assert _sweep(conn) == [_KBS[pvi.MAX_SWEEP_DISPATCH]]
    assert str(pvi.MAX_SWEEP_DISPATCH) in caplog.text, (
        "an index no reconcile will attempt again needs an operator, so the boot says so"
    )


def test_an_index_one_attempt_short_of_the_bound_is_still_dispatched():
    """The positive control: the bound is the bound, not a fear of failure."""
    rows = [_sweep_index_row(_KBS[0], 1536, False, failures=pvi.MAX_CONSECUTIVE_BUILD_FAILURES - 1)]
    conn = _FakeConn(answers=[(_SWEEP_CATALOG_QUERY, rows)], fail_on=_COUNT_QUERY)
    assert _sweep(conn) == [_KBS[0]]


def test_a_given_up_index_whose_knowledge_base_shrank_is_still_dispatched_to_be_dropped():
    """Only the *repair* is given up on; the index still costs every write.

    A knowledge base at or below the drop threshold wants that index gone, and
    dropping it is also what re-arms the build, so this is the one dispatch a
    given-up index must still get.
    """
    rows = [_sweep_index_row(_KBS[0], 1536, False, failures=pvi.MAX_CONSECUTIVE_BUILD_FAILURES)]
    conn = _FakeConn(
        answers=[
            (_SWEEP_CATALOG_QUERY, rows),
            (_COUNT_QUERY, [(_KBS[0], 1536, 1)]),
        ]
    )
    assert _sweep(conn) == [_KBS[0]]


def test_the_reconcile_a_given_up_index_is_dispatched_for_actually_drops_it(monkeypatch):
    """Which the spec above does not say, and for three rounds nothing did.

    The sweep returning the knowledge base is only half of it. The reconcile used to
    read the failure bound *before* the row count -- the opposite order to
    ``index_action`` -- so the dispatch asked for a drop, the reconcile reported
    ``build_repeatedly_failed``, and the index stayed ``indisvalid=false,
    indisready=true`` for ever: charged to every insert, answering no query, and
    reachable only by a manual ``DROP INDEX``. Measured exactly that, against a
    real server, on a knowledge base shrunk to three chunk rows.
    """
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536, False)],
        rows_by_dims={1536: 3},
        failures=pvi.MAX_CONSECUTIVE_BUILD_FAILURES,
    )
    assert pvi.index_action(conn, KB) == "drop", "the dispatch asks for the drop"
    outcome = _ensure(monkeypatch, conn)
    assert conn.issued("DROP INDEX CONCURRENTLY"), conn.statements
    assert outcome["repaired_invalid_indexes"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    assert outcome.get("build_repeatedly_failed") is None, (
        "giving up on the repair is not giving up on the drop"
    )
    assert conn.issued("CREATE INDEX") == [], "and it is not rebuilt at a size it no longer is"


def test_a_given_up_index_above_the_hnsw_limit_is_dropped_too(monkeypatch):
    """The other half of the same order: pgvector cannot build it at any row count.

    So the failure bound has nothing left to protect, and the index is pure cost.
    """
    conn = _ensure_conn(
        existing=[_index_row(KB, pvi.MAX_HNSW_DIMS + 1, False)],
        dims_present=(pvi.MAX_HNSW_DIMS + 1,),
        rows_by_dims={pvi.MAX_HNSW_DIMS + 1: 20_000},
        failures=pvi.MAX_CONSECUTIVE_BUILD_FAILURES,
    )
    assert pvi.index_action(conn, KB) == "drop"
    _ensure(monkeypatch, conn)
    assert conn.issued("DROP INDEX CONCURRENTLY"), conn.statements


def test_a_given_up_index_with_a_live_build_asks_to_be_run_again(monkeypatch):
    """``invalid_index_build_in_progress`` is the one reason that reschedules.

    A doomed index with a build still on it used to report
    ``build_repeatedly_failed`` instead, because the bound was read before
    ``_build_in_progress`` -- suppressing the reschedule for the single case the
    reschedule exists for. The build is an orphan backend; nothing else comes back
    to this knowledge base.
    """
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536, False)],
        rows_by_dims={1536: 20_000},
        failures=pvi.MAX_CONSECUTIVE_BUILD_FAILURES,
    )
    conn.answers.insert(0, (_BUILD_RUNNING_QUERY, [(1,)]))
    outcome = _ensure(monkeypatch, conn)
    assert outcome["status"] == "building", outcome
    assert outcome["reason"] == "invalid_index_build_in_progress", outcome
    assert pvi.outcome_needs_another_attempt(outcome) is True, outcome
    assert conn.issued("DROP INDEX") == [], "dropping it would pull the ground out from under it"


def test_the_sweeps_failure_history_comes_from_the_query_it_already_runs():
    """No second round trip per index: one column on the catalog SELECT."""
    conn = _FakeConn()
    _sweep(conn)
    catalog = conn.issued(_SWEEP_CATALOG_QUERY)
    assert len(catalog) == 1, conn.statements
    assert conn.issued("obj_description(to_regclass") == [], (
        "a per-index read would be one round trip per index on the boot path"
    )


# ---------------------------------------------------------------------------
# Reconciling one knowledge base
# ---------------------------------------------------------------------------

_LOCK_QUERY = "pg_try_advisory_lock"
# The one-index comment read, not the catalog SELECT's ``obj_description`` column:
# both name that function, and only this one names the index by ``to_regclass``.
_FAILURE_RECORD_QUERY = "obj_description(to_regclass(:index)"
_FAILURE_COMMENT_DDL = "COMMENT ON INDEX"
_ROW_COUNT_QUERY = "LIMIT :cap) s"
_BUILD_RUNNING_QUERY = "pg_stat_progress_create_index"


def _catalog_answer(own_rows, index_count=None):
    """Answer the one catalog SELECT the way ``pg_class`` would, from the prefix.

    A per-knowledge-base read gets that knowledge base's rows, and the project-wide
    read gets the whole project's -- which is those same rows unless a spec is about
    ``MAX_PER_KB_INDEXES`` and says the project holds ``index_count`` of them, in
    which case other knowledge bases' indexes make up the difference.

    Read from the bind, unlike ``_PopulationConn``, because here the prefix *is* the
    bind: one statement asks both questions and the bind is the only thing the
    server tells them apart by. A reader that stopped restricting itself to one
    knowledge base is answered with the whole project, exactly as Postgres would
    answer it.
    """
    own = list(own_rows)

    def answer(params):
        if index_count is None or (params or {}).get("prefix") != _WHOLE_PROJECT_PREFIX:
            return own
        filler = [
            _index_row(str(uuid.UUID(int=1_000 + n)), 1536)
            for n in range(max(0, int(index_count) - len(own)))
        ]
        return own + filler

    return answer


def _ensure_conn(
    existing=(),
    dims_present=(1536,),
    rows_by_dims=None,
    index_count=None,
    cls=None,
    failures=0,
    interrupted=0,
    **kwargs,
):
    """A connection that answers every read ``ensure_per_kb_vector_index`` makes.

    ``failures`` and ``interrupted`` are what this knowledge base's index already
    records in its own ``pg_class`` comment, which is where a doomed build's history
    lives -- and the reconcile reads that comment from the catalog SELECT, in the
    same pass as ``indisvalid``, so they are written onto the rows rather than
    answered separately.
    """
    rows = dict(rows_by_dims or {})
    own = [_with_history(row, failures, interrupted) for row in existing]
    return (cls or _FakeConn)(
        answers=[
            (_FAILURE_RECORD_QUERY, [(own[0][2],)] if own and own[0][2] else []),
            (_CATALOG_QUERY, _catalog_answer(own, index_count)),
            ("GROUP BY dims", [(d,) for d in dims_present]),
            (_ROW_COUNT_QUERY, lambda params: [(rows.get(int(params["dims"]), 0),)]),
            (_LOCK_QUERY, [(True,)]),
        ],
        **kwargs,
    )


# -- eligibility is decided on the population the index covers ---------------


class _PopulationConn(_FakeConn):
    """A connection backed by row counts per ``(dims, item_table)``, like the table.

    Both queries that decide eligibility are answered from the same rows, each
    honouring its own ``item_table`` bind when it has one and summing every
    population when it does not. So these specs are about what the decision comes
    out as for a given table, not about the text of a query -- a count that stops
    restricting itself answers with the sum and the decision moves.
    """

    def __init__(self, population, *args, kb_id=KB, **kwargs):
        super().__init__(*args, **kwargs)
        self.population = dict(population)
        self.kb_id = kb_id

    def _record(self, sql, params):
        self.statements.append(" ".join(sql.split()))
        self.params.append(params)

    @staticmethod
    def _restricted_to(sql, params):
        """The population this statement actually restricts itself to, or None.

        Read from the *statement*, not from the bound parameters, because that is
        what the server reads. A query that stops naming ``item_table`` while still
        binding a value for it is answered from every population here, exactly as
        Postgres would answer it -- which is the false green a bound-parameter
        assertion gives instead.
        """
        return (params or {}).get("item_table") if "item_table = :item_table" in sql else None

    def execute(self, clause, params=None):
        sql = clause.text if hasattr(clause, "text") else str(clause)
        if "GROUP BY dims" in sql:
            # The dimension survey, answered from the same rows and under the same
            # rule as the two counts: a survey that stops naming ``item_table`` is
            # answered from every population, which is how a dimension only another
            # population has reached the loop and crowded out one over the threshold.
            self._record(sql, params)
            wanted = self._restricted_to(sql, params)
            return _Result(
                [
                    (d,)
                    for d in sorted(
                        {d for d, table in self.population if wanted is None or table == wanted}
                    )
                ]
            )
        if _ROW_COUNT_QUERY in sql:
            self._record(sql, params)
            wanted = self._restricted_to(sql, params)
            rows = sum(
                n
                for (d, table), n in self.population.items()
                if d == int(params["dims"]) and (wanted is None or table == wanted)
            )
            return _Result([(min(rows, int(params["cap"])),)])
        if _COUNT_QUERY in sql:
            self._record(sql, params)
            wanted = self._restricted_to(sql, params)
            grouped: dict[int, int] = {}
            for (d, table), n in self.population.items():
                if wanted is None or table == wanted:
                    grouped[d] = grouped.get(d, 0) + n
            return _Result([(self.kb_id, d, n) for d, n in sorted(grouped.items())])
        return super().execute(clause, params)


def _population_conn(population, existing=(), index_count=None, **kwargs):
    return _PopulationConn(
        population,
        answers=[
            (_FAILURE_RECORD_QUERY, []),
            (_CATALOG_QUERY, _catalog_answer([_index_row(KB, d) for d in existing], index_count)),
            (_LOCK_QUERY, [(True,)]),
            (
                _SETTINGS_QUERY,
                [
                    ("VECTOR_PER_KB_INDEX_MIN_ROWS", "10000"),
                    ("VECTOR_PER_KB_INDEX_DROP_ROWS", "5000"),
                ],
            ),
        ],
        **kwargs,
    )


_MIXED = {(1536, "chunks"): 3_000, (1536, "full_documents"): 7_000}
_CHUNKS_ONLY = {(1536, "chunks"): 12_000}


def _at_ten_thousand(monkeypatch):
    """The thresholds ``_ensure`` fixes, for the dispatch check that reads them too."""
    _stub_settings(
        monkeypatch,
        {"VECTOR_PER_KB_INDEX_MIN_ROWS": 10_000, "VECTOR_PER_KB_INDEX_DROP_ROWS": 5_000},
    )


def test_a_knowledge_base_over_the_threshold_only_on_the_sum_gets_no_index(monkeypatch):
    """10,000 rows across four populations is not 10,000 rows the index can serve.

    The index covers one item table, so a knowledge base whose chunk population is
    3,000 would get an index that mostly indexes rows its chunk searches cannot
    join -- which is the recall loss measured at 0.858 -> 0.383.
    """
    _at_ten_thousand(monkeypatch)
    conn = _population_conn(_MIXED)
    assert pvi.index_action(conn, KB) is None
    outcome = _ensure(monkeypatch, conn, build_at=10_000)
    assert outcome["built"] == [], outcome
    assert conn.issued("CREATE INDEX") == [], conn.statements


def test_a_knowledge_base_over_the_threshold_on_its_own_population_still_gets_one(monkeypatch):
    """The positive control, on the same fake table with the other populations gone."""
    _at_ten_thousand(monkeypatch)
    conn = _population_conn(_CHUNKS_ONLY)
    assert pvi.index_action(conn, KB) == "build"
    outcome = _ensure(monkeypatch, conn, build_at=10_000)
    assert outcome["built"] == [pvi.per_kb_index_name(KB, 1536)], outcome


def test_an_index_whose_own_population_has_drained_is_dropped(monkeypatch):
    """The drop threshold cuts the same way, because the index covered only chunks."""
    _at_ten_thousand(monkeypatch)
    conn = _population_conn(
        {(1536, "chunks"): 10, (1536, "full_documents"): 40_000}, existing=(1536,)
    )
    assert pvi.index_action(conn, KB) == "drop"
    outcome = _ensure(monkeypatch, conn, build_at=10_000, drop_below=5_000)
    assert outcome["dropped"] == [pvi.per_kb_index_name(KB, 1536)], outcome


def test_another_population_cannot_crowd_the_chunk_dimension_out_of_the_survey(monkeypatch):
    """The survey gates the count, so an unrestricted survey is a decision too.

    Measured: 30,000 ``full_documents`` at 64 dimensions written first, then 30,000
    chunks at 128, threshold 1,000 -- the unrestricted survey returned ``[64]``, the
    count at 64 returned 0, ``index_action`` returned None, and 30,000 chunk rows
    thirty times over the threshold were never dispatched. Only the next boot
    recovered it, and a restart is not a recovery for a running project.
    """
    two_populations = {(64, "full_documents"): 30_000, (128, "chunks"): 30_000}
    conn = _population_conn(two_populations)
    _at_ten_thousand(monkeypatch)
    assert pvi.candidate_dims(conn, KB, 10_001) == [128], (
        "the survey has to look at the population the index would cover"
    )
    assert pvi.index_action(conn, KB) == "build"
    outcome = _ensure(monkeypatch, _population_conn(two_populations))
    assert outcome["built"] == [pvi.per_kb_index_name(KB, 128)], outcome


def test_the_survey_and_the_count_agree_about_which_rows_they_are_about(monkeypatch):
    """All four places that decide eligibility name the one constant.

    The predicate, the reconcile's count, the boot sweep's count and now the
    dimension survey. Read off the statements, because a query that still binds a
    value for ``item_table`` while no longer naming it in its SQL is answered by
    Postgres from every population.
    """
    conn = _population_conn({(1536, "chunks"): 12_000, (1536, "graph_index_nodes"): 40_000})
    _ensure(monkeypatch, conn)
    deciding = [
        st
        for st in conn.statements
        if "GROUP BY dims" in st or _ROW_COUNT_QUERY in st or _COUNT_QUERY in st
    ]
    assert deciding, conn.statements
    for statement in deciding:
        assert "item_table = :item_table" in statement, statement


def test_the_boot_sweep_counts_only_the_population_the_index_covers():
    """The sweep's grouped count is a second query and had the same defect.

    A knowledge base whose four populations sum past the build threshold but whose
    chunk population does not must not have a build dispatched at start-up: the
    reconcile counts the chunks and declines, so the dispatch is a build slot and a
    boot spent on nothing.
    """
    assert _sweep(_population_conn(_MIXED)) == []
    assert _sweep(_population_conn(_CHUNKS_ONLY)) == [KB]


# -- an index whose definition the module no longer emits --------------------


def _stale_row(kb_id=KB, dims=1536, **kwargs):
    """A valid index built from a definition this version no longer emits.

    Which is exactly what an index built by an earlier build of this branch is: the
    predicate gained ``item_table`` and the name did not change, so the catalog
    carries the old two-clause predicate under today's name. Measured live: the
    planner still matches it, because today's query *implies* the old predicate, so
    every chunk search on that knowledge base ran at recall 0.562 where 0.989 was
    available -- for ever, with no log line.
    """
    return _index_row(kb_id, dims, True, fingerprint=None, **kwargs)


def test_an_index_with_no_recorded_definition_has_drifted():
    """The index this exists to find carries no fingerprint at all.

    It was built by code that did not write one, so there is nothing to compare and
    "unknown" is not "current". Getting this backwards would leave every existing
    index unreconciled, which is the state the review found.
    """
    assert pvi.definition_has_drifted(KB, 1536, None) is True
    assert pvi.definition_has_drifted(KB, 1536, "dropping this on Monday, see ticket 41") is True


def test_an_index_built_from_todays_definition_has_not_drifted():
    current = pvi.per_kb_index_comment(0, 0, pvi.per_kb_index_fingerprint(KB, 1536))
    assert pvi.definition_has_drifted(KB, 1536, current) is False


def test_the_fingerprint_changes_with_the_definition_and_not_with_the_name():
    """Every part of the DDL the name does not carry has to move it.

    The predicate is the one that moved in this PR; the operator class, the cast and
    any storage parameter are the ones that can move next, and the same hole would
    swallow them.
    """
    kb_a, kb_b = KB, str(uuid.UUID(int=77))
    assert pvi.per_kb_index_fingerprint(kb_a, 1536) != pvi.per_kb_index_fingerprint(kb_b, 1536)
    assert pvi.per_kb_index_fingerprint(kb_a, 1536) != pvi.per_kb_index_fingerprint(kb_a, 768)
    before = pvi.per_kb_index_fingerprint(kb_a, 1536)
    real_ddl = pvi.per_kb_index_ddl
    try:
        pvi.per_kb_index_ddl = lambda kb, d: real_ddl(kb, d).replace(
            "vector_cosine_ops", "vector_l2_ops"
        )
        assert pvi.per_kb_index_fingerprint(kb_a, 1536) != before
    finally:
        pvi.per_kb_index_ddl = real_ddl


def test_drift_is_decided_without_reading_the_definition_back_from_postgres(monkeypatch):
    """The dangerous fix, kept out by what the reconcile actually asks the server.

    ``pg_get_indexdef`` normalises what this module emits in at least six ways --
    ``((embedding)::vector(N))``, ``'...'::uuid``, ``(item_table)::text``, the
    predicate's parentheses -- so the two strings are never equal. An equality test
    against it would mark *every* index stale, and because the failure record dies
    with the index a drift drop takes away, nothing would bound the resulting
    drop-and-rebuild: every reconcile of every knowledge base, for ever.

    Asserted on the statements rather than on the module's text, so the paragraph
    above may keep naming the functions it warns about.
    """
    conn = _ensure_conn(existing=[_stale_row()], rows_by_dims={1536: 20_000})
    _ensure(monkeypatch, conn)
    assert conn.issued("DROP INDEX CONCURRENTLY"), "the drift was found"
    for reader in ("pg_get_indexdef", "pg_get_expr", "indpred"):
        assert conn.issued(reader) == [], conn.statements


def test_a_stale_index_is_dispatched_dropped_and_rebuilt(monkeypatch):
    """The whole fix, in the order it happens."""
    conn = _ensure_conn(existing=[_stale_row()], rows_by_dims={1536: 20_000})
    _at_ten_thousand(monkeypatch)
    assert pvi.index_action(conn, KB) == "build", "nothing reconciled it before"
    outcome = _ensure(monkeypatch, conn)
    assert conn.issued("DROP INDEX CONCURRENTLY"), conn.statements
    assert outcome["rebuilt_stale_definitions"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    assert outcome["built"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    assert outcome["dropped"] == [], "a definition drop is not a threshold drop"


def test_the_rebuild_records_the_definition_so_it_happens_once(monkeypatch):
    """Otherwise this is the unbounded drop-and-rebuild loop the review warned about.

    The success path's comment is what makes the *next* reconcile a no-op. Measured
    live: after the rebuild, ``index_action`` is None and the boot sweep returns
    nothing.
    """
    conn = _CatalogStateConn(
        answers=[
            ("GROUP BY dims", [(1536,)]),
            (_ROW_COUNT_QUERY, [(20_000,)]),
            (_LOCK_QUERY, [(True,)]),
        ],
        comment=None,
        valid=True,
    )
    first = _ensure(monkeypatch, conn)
    assert first["rebuilt_stale_definitions"], first
    assert pvi.definition_has_drifted(KB, 1536, conn.comment) is False, conn.comment
    conn.statements.clear()
    second = _ensure(monkeypatch, conn)
    assert second.get("rebuilt_stale_definitions") is None, second
    assert conn.issued("DROP INDEX") == [], "a second reconcile has nothing to do"
    assert conn.issued("CREATE INDEX") == []


def test_a_definition_drop_is_not_counted_against_the_failure_budget(monkeypatch):
    """Three predicate changes would otherwise turn the feature off.

    Nothing failed: the index was working and this module's own DDL moved. The
    repair drop of an INVALID index counts against the bound, and a drift drop must
    not be confused with it.
    """
    conn = _ensure_conn(
        existing=[_stale_row()], rows_by_dims={1536: 20_000}, fail_on="DROP INDEX"
    )
    with pytest.raises(RuntimeError):
        _ensure(monkeypatch, conn)
    assert conn.issued(_FAILURE_COMMENT_DDL) == [], conn.statements


def test_a_stale_index_is_kept_at_the_index_cap(monkeypatch, caplog):
    """The drop frees a place another knowledge base's reconcile can take.

    This one would then be left with no index where it had a stale-but-usable one,
    and could not get one back until an operator made room. A stale index answers
    its searches; no index does not.
    """
    conn = _ensure_conn(
        existing=[_stale_row()],
        rows_by_dims={1536: 20_000},
        index_count=pvi.MAX_PER_KB_INDEXES,
    )
    with caplog.at_level(logging.WARNING):
        outcome = _ensure(monkeypatch, conn)
    assert conn.issued("DROP INDEX") == [], conn.statements
    assert conn.issued("CREATE INDEX") == []
    assert outcome["stale_definitions_kept"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    assert outcome["reason"] == "index_cap_reached", outcome
    assert "no index at all" in caplog.text
    assert pvi.index_action(conn, KB) is None, "and it is not dispatched again either"


def test_a_stale_index_inside_the_hysteresis_band_is_kept(monkeypatch, caplog):
    """A rebuild would be declined below the build threshold, so the drop is a loss.

    The hysteresis exists so an index at this size is kept; dropping it for a
    definition change and then declining the rebuild would delete it instead.
    """
    conn = _ensure_conn(existing=[_stale_row()], rows_by_dims={1536: 7_000})
    with caplog.at_level(logging.WARNING):
        outcome = _ensure(monkeypatch, conn)
    assert conn.issued("DROP INDEX") == [], conn.statements
    assert outcome["stale_definitions_kept"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    assert "below the build threshold" in caplog.text
    assert pvi.index_action(conn, KB) is None


def test_a_stale_index_below_the_drop_threshold_is_just_dropped(monkeypatch):
    """The threshold wins: there is no point rebuilding what is about to go."""
    conn = _ensure_conn(existing=[_stale_row()], rows_by_dims={1536: 10})
    _at_ten_thousand(monkeypatch)
    assert pvi.index_action(conn, KB) == "drop"
    outcome = _ensure(monkeypatch, conn)
    assert outcome["dropped"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    assert outcome.get("rebuilt_stale_definitions") is None, outcome
    assert conn.issued("CREATE INDEX") == []


def test_the_boot_sweep_dispatches_a_stale_index_after_the_urgent_ones():
    """A catalog fact, like INVALID -- but the least urgent of the four cases.

    An INVALID index answers nothing at all, so it must not lose a place in
    ``MAX_SWEEP_DISPATCH`` to one that answers at reduced recall.
    """
    invalid = [_sweep_index_row(kb, 1536, False) for kb in _KBS[:3]]
    stale = [_stale_row(kb) for kb in _KBS[3:6]]
    conn = _FakeConn(answers=[(_SWEEP_CATALOG_QUERY, stale + invalid)], fail_on=_COUNT_QUERY)
    assert _sweep(conn) == _KBS[:3] + _KBS[3:6]


def test_the_boot_sweep_leaves_a_current_index_alone():
    """The half that would make this a boot-time drop-and-rebuild loop."""
    build_at, drop_below = pvi.thresholds()
    settled = (build_at + drop_below) // 2
    assert drop_below < settled < build_at, "a knowledge base the sweep has no reason to touch"
    rows = [_index_row(_KBS[0], 1536)]
    conn = _FakeConn(
        answers=[(_SWEEP_CATALOG_QUERY, rows), (_COUNT_QUERY, [(_KBS[0], 1536, settled)])]
    )
    assert _sweep(conn) == []


# -- the comment carries three facts and the parse has to be strict ----------


@pytest.mark.parametrize(
    "failures, interrupted, fingerprint",
    [
        (0, 0, None),
        (2, 0, None),
        (0, 5, None),
        (0, 0, "abcdef012345"),
        (2, 0, "abcdef012345"),
        (0, 5, "abcdef012345"),
        (2, 5, "abcdef012345"),
    ],
)
def test_every_combination_of_the_three_facts_survives_a_round_trip(
    failures, interrupted, fingerprint
):
    """Both directions, because the counts and the fingerprint share one comment.

    A parse that reads one of them out of the other's sentence, or that stops
    finding a count once a fingerprint is beside it, is how a bound stops being
    reachable or an index stops looking stale.
    """
    comment = pvi.per_kb_index_comment(failures, interrupted, fingerprint)
    assert pvi.build_failures_in(comment) == failures
    assert pvi.interrupted_builds_in(comment) == interrupted
    assert pvi.definition_fingerprint_in(comment) == fingerprint


def test_nothing_to_record_is_no_comment_at_all():
    """Which is what ``COMMENT ON ... IS NULL`` writes, and what a clean index has."""
    assert pvi.per_kb_index_comment(0, 0, None) is None


@pytest.mark.parametrize(
    "garbage",
    [
        "dropping this on Monday, see ticket 41",
        "3 consecutive failures",
        "Built from definition NOTHEX12345.",
        "Built from definition abcdef0123456789.",
        "consecutive failed attempts to build this partial HNSW index.",
        "",
    ],
)
def test_a_comment_this_module_did_not_write_records_nothing(garbage):
    """A comment is a place anyone may write, and every reader has to say so.

    Reading a count out of one would give up on a healthy index; reading a
    fingerprint out of one would leave a stale index in place for ever.
    """
    assert pvi.build_failures_in(garbage) == 0
    assert pvi.interrupted_builds_in(garbage) == 0
    assert pvi.definition_fingerprint_in(garbage) is None
    assert pvi.build_is_given_up(garbage) is False


def test_a_comment_carrying_only_a_fingerprint_is_not_given_up_on():
    """Which is every index this version builds successfully."""
    comment = pvi.per_kb_index_comment(0, 0, pvi.per_kb_index_fingerprint(KB, 1536))
    assert pvi.build_is_given_up(comment) is False


def test_either_bound_on_its_own_is_enough_to_give_up():
    assert pvi.build_is_given_up(
        pvi.per_kb_index_comment(pvi.MAX_CONSECUTIVE_BUILD_FAILURES, 0, None)
    )
    assert pvi.build_is_given_up(
        pvi.per_kb_index_comment(0, pvi.MAX_CONSECUTIVE_INTERRUPTED_BUILDS, None)
    )
    assert not pvi.build_is_given_up(
        pvi.per_kb_index_comment(
            pvi.MAX_CONSECUTIVE_BUILD_FAILURES - 1,
            pvi.MAX_CONSECUTIVE_INTERRUPTED_BUILDS - 1,
            None,
        )
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
    assert pvi.build_failures_in(written[0]) == 1, written


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
    written = conn.issued(_FAILURE_COMMENT_DDL)
    assert pvi.build_failures_in(written[0]) == 2, written


def test_a_build_that_succeeds_forgets_the_failures_before_it(monkeypatch):
    """Consecutive failures are what says a build is doomed, not lifetime ones.

    A build lost to a server restart or a killed worker is a failure a retry
    really does get past. The one comment the success writes carries neither
    count, and carries the definition it was built from instead.
    """
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536, False)], rows_by_dims={1536: 20_000}, failures=2
    )
    outcome = _ensure(monkeypatch, conn)
    assert outcome["built"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    cleared = conn.issued(_FAILURE_COMMENT_DDL)
    assert len(cleared) == 1, conn.statements
    assert "consecutive" not in cleared[0], cleared
    assert pvi.per_kb_index_fingerprint(KB, 1536) in cleared[0], cleared


def test_a_build_that_succeeds_records_the_definition_it_built_from(monkeypatch):
    """On every success, not only the ones that follow a failure.

    The fingerprint is the only record that this index matches the definition the
    module now emits; without it on a first, clean build, the very next reconcile
    would read the index as drifted and drop and rebuild it -- on every reconcile,
    for ever.
    """
    conn = _ensure_conn(rows_by_dims={1536: 20_000})
    outcome = _ensure(monkeypatch, conn)
    assert outcome["built"] == [pvi.per_kb_index_name(KB, 1536)], outcome
    written = conn.issued(_FAILURE_COMMENT_DDL)
    assert len(written) == 1, conn.statements
    assert pvi.per_kb_index_fingerprint(KB, 1536) in written[0], written


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

    Both build bounds live in the index's own ``pg_class`` comment, so a spec about
    a bound being *reached* has to read back what the previous attempt wrote. This
    keeps that one piece of catalog state, which is what lets a sequence of
    reconciles be driven the way the ones against a real server were -- and it
    serves the comment through the *catalog row*, in the column the reconcile
    really reads it from, not only through the single-index read.

    The index is INVALID throughout, which is the state a failed
    ``CREATE INDEX CONCURRENTLY`` leaves and the state each of these reconciles
    finds.
    """

    def __init__(self, *args, comment=None, kb_id=KB, dims=1536, valid=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.comment = comment
        self.kb_id = kb_id
        self.dims = dims
        self.valid = valid

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
        if _CATALOG_QUERY in sql:
            self.statements.append(" ".join(sql.split()))
            self.params.append(params)
            return _Result(
                [(pvi.per_kb_index_name(self.kb_id, self.dims), self.valid, self.comment)]
            )
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
    assert pvi.build_failures_in(written[0]) == 2, written


def test_repair_drops_that_keep_failing_reach_the_give_up_bound(monkeypatch, caplog):
    """The loop itself, driven the way it was driven against a real server.

    Reconcile after reconcile, with the index's comment carried between them the
    way the catalog carries it. Without the attempt being counted this never
    terminates: every run reports ``failures=1`` and asks for a build again.
    """
    outcomes = []
    conn = _CatalogStateConn(
        answers=[
            ("GROUP BY dims", [(1536,)]),
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


def test_a_transient_build_failure_spends_the_other_bound_not_this_one(monkeypatch):
    """Three permanent failures disable the index until an operator drops it by hand.

    So a lock conflict may not spend one of them: the task that runs this
    classifies it as transient and retries it several times, and one contention
    episode outlasting three of those retries would otherwise write "3 consecutive
    failed attempts" and turn the index off for good.

    It still has to be counted against *something*, which is what this used to get
    wrong: writing nothing made the whole bound unreachable, because the repair
    drop takes the previous record away first.
    """
    conn = _ensure_conn(rows_by_dims={1536: 20_000}, fail_on="CREATE INDEX", exc=_transient_exc())
    with pytest.raises(Exception, match="lock timeout"):
        _ensure(monkeypatch, conn)
    written = conn.issued(_FAILURE_COMMENT_DDL)
    assert len(written) == 1, conn.statements
    assert pvi.build_failures_in(written[0]) == 0, written
    assert pvi.interrupted_builds_in(written[0]) == 1, written


def test_a_transient_build_failure_keeps_the_attempts_already_on_record(monkeypatch):
    """And does not un-count the permanent ones before it.

    The repair drop takes the record away with the index it is written on, so both
    counts have to be carried across it in memory and written back together.
    """
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536, False)],
        rows_by_dims={1536: 20_000},
        failures=2,
        interrupted=4,
        fail_on="CREATE INDEX",
        exc=_transient_exc(),
    )
    with pytest.raises(Exception, match="lock timeout"):
        _ensure(monkeypatch, conn)
    written = conn.issued(_FAILURE_COMMENT_DDL)
    assert written, "an uncounted attempt is an unbounded loop"
    assert pvi.build_failures_in(written[0]) == 2, written
    assert pvi.interrupted_builds_in(written[0]) == 5, written


def test_a_transient_repair_drop_failure_spends_the_larger_bound_too(monkeypatch):
    """The repair drop is classified under the same rule as the build it precedes."""
    conn = _ensure_conn(
        existing=[_index_row(KB, 1536, False)],
        rows_by_dims={1536: 20_000},
        failures=1,
        fail_on="DROP INDEX",
        exc=_transient_exc(statement="DROP INDEX"),
    )
    with pytest.raises(Exception, match="lock timeout"):
        _ensure(monkeypatch, conn)
    written = conn.issued(_FAILURE_COMMENT_DDL)
    assert written, conn.statements
    for one in written:
        assert pvi.build_failures_in(one) == 1, one
        assert pvi.interrupted_builds_in(one) == 1, one


def test_a_knowledge_base_whose_builds_only_ever_get_interrupted_is_given_up_on(monkeypatch):
    """The loop the larger bound exists to stop, driven the way it runs.

    Measured against a real server before there was a bound for these: seven
    consecutive reconciles of an INVALID index whose build fails ``55P03`` every
    time wrote no comment at all on any of the seven, ended with 0 recorded
    failures against a bound of 3, and asked for a build again every time. That is
    a drop-rebuild-fail loop with nothing bounding it, on every source that
    finishes indexing and every boot.
    """
    conn = _CatalogStateConn(
        answers=[
            ("GROUP BY dims", [(1536,)]),
            (_ROW_COUNT_QUERY, [(20_000,)]),
            (_LOCK_QUERY, [(True,)]),
        ],
        fail_on="CREATE INDEX",
        exc=_transient_exc(),
    )
    for _ in range(pvi.MAX_CONSECUTIVE_INTERRUPTED_BUILDS):
        with pytest.raises(Exception, match="lock timeout"):
            _ensure(monkeypatch, conn)
    assert pvi.interrupted_builds_in(conn.comment) == pvi.MAX_CONSECUTIVE_INTERRUPTED_BUILDS
    outcome = _ensure(monkeypatch, conn)
    assert outcome["reason"] == "build_repeatedly_failed", outcome
    assert pvi.index_action(conn, KB) is None, "and nothing dispatches it again"


def test_one_contention_episode_cannot_disable_an_index(monkeypatch):
    """Which is the whole reason the interrupted bound is a separate, larger one.

    The task retries a transient failure ``PG_BM25_TASK_MAX_RETRIES`` times, so one
    episode is that many attempts plus the first. The bound has to be comfortably
    above it, or a single burst of contention turns the index off until an operator
    drops it by hand.
    """
    from agentic_project_service.tasks.indexing import PG_BM25_TASK_MAX_RETRIES

    one_episode = PG_BM25_TASK_MAX_RETRIES + 1
    assert pvi.MAX_CONSECUTIVE_INTERRUPTED_BUILDS > 2 * one_episode, (
        "one contention episode must not spend the whole budget"
    )
    assert pvi.MAX_CONSECUTIVE_INTERRUPTED_BUILDS > pvi.MAX_CONSECUTIVE_BUILD_FAILURES


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
