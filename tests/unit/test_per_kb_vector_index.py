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


def test_the_three_settings_are_registered_with_conservative_defaults():
    build = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"]
    drop = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_DROP_ROWS"]
    mem = SETTINGS_REGISTRY["VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB"]
    assert (build.type, build.default) == ("int", 50000)
    assert (drop.type, drop.default) == ("int", 25000)
    assert (mem.type, mem.default) == ("int", 128)
    assert drop.default < build.default, "the defaults must carry the hysteresis"
    assert mem.max == 4096, "unbounded build memory would OOM the smallest project pods"
    for d in (build, drop, mem):
        assert d.category == "knowledge-retrieval"
        assert d.advanced is True
        assert d.description


@pytest.mark.parametrize(
    "key,value,ok",
    [
        ("VECTOR_PER_KB_INDEX_MIN_ROWS", 999, False),
        ("VECTOR_PER_KB_INDEX_MIN_ROWS", 1000, True),
        ("VECTOR_PER_KB_INDEX_MIN_ROWS", 10_000_001, False),
        ("VECTOR_PER_KB_INDEX_DROP_ROWS", 0, True),
        ("VECTOR_PER_KB_INDEX_DROP_ROWS", -1, False),
        ("VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB", 63, False),
        ("VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB", 64, True),
        ("VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB", 4097, False),
    ],
)
def test_validate_setting_enforces_the_ranges(key, value, ok):
    accepted, message = validate_setting(key, value)
    assert accepted is ok, message


def _stub_settings(monkeypatch, values: dict[str, int]):
    """Stub the settings read, filling in the build memory a caller did not name."""
    filled = {"VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB": 128, **values}
    monkeypatch.setattr(pvi, "get_setting", lambda key: filled[key])


def test_thresholds_default_to_the_registry_values(monkeypatch):
    _stub_settings(
        monkeypatch,
        {
            "VECTOR_PER_KB_INDEX_MIN_ROWS": 50000,
            "VECTOR_PER_KB_INDEX_DROP_ROWS": 25000,
        },
    )
    assert pvi.thresholds() == (50000, 25000)


def test_thresholds_clamp_a_stored_value_outside_the_registrys_bounds(monkeypatch):
    # get_setting coerces but does not range-check, so a row written before a
    # bound was tightened would otherwise be used as-is.
    _stub_settings(
        monkeypatch,
        {
            "VECTOR_PER_KB_INDEX_MIN_ROWS": 5,
            "VECTOR_PER_KB_INDEX_DROP_ROWS": 0,
        },
    )
    build_at, drop_below = pvi.thresholds()
    assert build_at == 1000, "below the registry minimum must be pulled up to it"
    assert drop_below == 0


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


def test_vector_search_keeps_the_item_table_filter():
    """The embeddings-side predicate is added, not substituted.

    The item-table filter is what keeps an embedding whose item has been
    deleted, or belongs to another knowledge base, out of the answer.
    """
    sql = _search_sql(_statements(lambda s: s.vector_search(embedding=[0.0] * 1536, top_k=10)))
    assert "c.knowledge_base_id = :kb_id" in " ".join(sql.split())


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
                return _Result(rows)
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


def test_a_build_sets_both_session_settings_before_the_ddl_and_resets_both_after(monkeypatch):
    monkeypatch.setattr(pvi, "maintenance_work_mem_mb", lambda: 256)
    conn = _FakeConn()
    pvi._create_index(conn, KB, 1536)
    assert conn.statements == [
        "SET statement_timeout = 0",
        "SET maintenance_work_mem = '256MB'",
        " ".join(pvi.per_kb_index_ddl(KB, 1536).split()),
        "RESET maintenance_work_mem",
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
    assert conn.issued("SET statement_timeout = 0"), conn.statements
    assert conn.issued("RESET statement_timeout") or conn.invalidated, conn.statements


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
