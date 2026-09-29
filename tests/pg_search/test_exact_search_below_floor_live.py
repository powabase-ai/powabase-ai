"""Database-backed checks for the exact search below the per-KB index floor.

``tests/unit/test_exact_search_below_per_kb_floor.py`` pins the statements the
store emits. Only a real planner shows what they are *for*: that a small
knowledge base with no index of its own is answered from the knowledge-base btree
and a sort -- never from an HNSW index, never by reading the whole
``embeddings`` table -- that the answer is exactly the true top-k, and that the
knowledge bases this path is not for keep the plan they had.

The regime the fixture represents
--------------------------------
One ``embeddings`` table of ~55,000 rows at 384 dimensions in which three filler
knowledge bases hold 36,000 and the ones under test are small shares of it, with
every knowledge base's rows **interleaved** through the heap the way rows arrive in
a shared table over time. Both properties are load-bearing, and both were found
by the fixture not reproducing the defect without them:

- with each knowledge base's rows contiguous (one ``COPY`` per knowledge base),
  a btree lookup reads a handful of adjacent pages and the planner chooses it
  unaided at every size tried, so there is nothing for the fix to change;
- interleaved, the same lookup touches a page per row, and the planner sends a
  5,000-row knowledge base at 9 % of the table to the *shared* HNSW index -- the
  plan this change exists to replace (``KB_EDGE``, and the control spec that
  asserts the fixture still reproduces it). The crossover sits near 6-7 %: 4,000
  rows went to the index at 7 % of a 55,000-row table and stayed exact at 6 % of
  a 66,000-row one, which is why the knowledge base sits at 9 % and at the cap
  itself, where the cap's inclusive edge is exercised too.

384 dimensions keeps the build to seconds. At 1536 a vector is stored out of line
and the plans' costs move; the cost race is a property of the table's shape, not
of the width (see ``BasePgVectorStore._preferring_this_kbs_partial_index``), and
what these specs pin -- the fenced statement cannot reach an HNSW index, and the
scan penalty keeps it off the heap -- holds at any width because it is structural.

A second, small schema (``SHARE_SCHEMA``) holds a knowledge base that is a third
of its table: the shape where the planner prefers to read the whole table, fence
or not, and where the ``enable_seqscan`` penalty is the only thing standing
between an exact search and a scan of everyone's rows.
"""

from __future__ import annotations

import asyncio
import io
import os
import time
import uuid

import numpy as np
import psycopg
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services import pg_vector_index as pvi

SCHEMA = "exact_below_floor_live_test"
SHARE_SCHEMA = "exact_below_floor_share_live_test"
DIMS = 384
TOP_K = 20
HNSW_SHARED = f"idx_ai_embeddings_hnsw_{DIMS}"
KB_BTREE = "embeddings_kb_btree"

# The knowledge bases under test, and the filler that makes them small shares.
KB_SMALL = "5b0e7a1c-0000-4000-8000-000000000001"  # 300 rows: small by any measure
KB_EDGE = "5b0e7a1c-0000-4000-8000-000000000002"  # 5,000, the cap: unaided, the planner picks HNSW
KB_GAP = "5b0e7a1c-0000-4000-8000-000000000003"  # 6,000: over the cap, no index
KB_INDEXED = "5b0e7a1c-0000-4000-8000-000000000004"  # 8,000: over the cap, its own index
FILLER_KBS = [f"5b0e7a1c-0000-4000-8000-0000000001{i:02d}" for i in range(3)]
SOURCE = "5b0e7a1c-0000-4000-8000-0000000000aa"
ROW_COUNTS = [
    (KB_SMALL, 300),
    (KB_EDGE, 5_000),
    (KB_GAP, 6_000),
    (KB_INDEXED, 8_000),
    *((kb, 12_000) for kb in FILLER_KBS),
]
# Document-level rows for the small knowledge base, so a store other than chunks
# can be searched: the per-KB index never covers these.
SMALL_DOCUMENTS = 60

KB_SHARE = "5b0e7a1c-0000-4000-8000-000000000201"
SHARE_ROW_COUNTS = [
    (KB_SHARE, 3_000),
    *((f"5b0e7a1c-0000-4000-8000-00000000021{i}", 3_000) for i in range(2)),
]

# The shipped default, deliberately: the specs are about which side of it each
# knowledge base falls on, and 5,000 separates KB_EDGE from KB_GAP.
CAP = 5_000


class _ChunkStore(bvs.BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


class _DocumentStore(bvs.BasePgVectorStore):
    """The real document store's table and columns, without its storage-backed text."""

    TABLE = "full_documents"
    TEXT_COL = "full_text_path"
    SEARCH_TEXT_COL = "summary"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _dsn() -> str:
    dsn = os.environ.get("PG_SEARCH_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        if os.environ.get("PG_SEARCH_REQUIRED") == "1":
            pytest.fail("PG_SEARCH_REQUIRED=1 but no PG_SEARCH_TEST_DATABASE_URL or DATABASE_URL")
        pytest.skip("no PG_SEARCH_TEST_DATABASE_URL or DATABASE_URL to test pgvector against")
    return dsn


def _vectors(rng, n: int) -> np.ndarray:
    """Clustered unit vectors, so a top-k is not a coin toss between near-ties."""
    centres = rng.standard_normal((40, DIMS))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    v = centres[rng.integers(0, 40, n)] + 0.05 * rng.standard_normal((n, DIMS))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _literal(vector) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vector) + "]"


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


def _create_tables(conn, schema: str) -> None:
    conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"""
        CREATE TABLE {schema}.chunks (
            id uuid PRIMARY KEY,
            knowledge_base_id uuid NOT NULL,
            source_id uuid NOT NULL,
            text text NOT NULL,
            meta jsonb DEFAULT '{{}}'::jsonb
        )
    """)
    conn.execute(f"""
        CREATE TABLE {schema}.full_documents (
            id uuid PRIMARY KEY,
            knowledge_base_id uuid NOT NULL,
            source_id uuid NOT NULL,
            full_text_path text NOT NULL,
            summary text,
            meta jsonb DEFAULT '{{}}'::jsonb
        )
    """)
    conn.execute(f"""
        CREATE TABLE {schema}.embeddings (
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
    conn.execute(f"CREATE INDEX ON {schema}.chunks (knowledge_base_id)")
    conn.execute(f"CREATE INDEX ON {schema}.full_documents (knowledge_base_id)")
    conn.execute(f"CREATE INDEX ON {schema}.embeddings (item_id)")
    conn.execute(f"CREATE INDEX {KB_BTREE} ON {schema}.embeddings (knowledge_base_id)")


def _load(conn, schema: str, row_counts, rng, documents: dict[str, int] | None = None) -> None:
    """Every knowledge base's rows, shuffled together so none of them is contiguous."""
    rows = []
    for kb_id, n in row_counts:
        rows += [("chunks", kb_id, v) for v in _vectors(rng, n)]
    for kb_id, n in (documents or {}).items():
        rows += [("full_documents", kb_id, v) for v in _vectors(rng, n)]
    items = {"chunks": io.StringIO(), "full_documents": io.StringIO()}
    embeddings = io.StringIO()
    for i in rng.permutation(len(rows)):
        table, kb_id, vector = rows[i]
        item_id = str(uuid.uuid4())
        items[table].write(f"{item_id}\t{kb_id}\t{SOURCE}\titem {i}\t{{}}\n")
        embeddings.write(
            f"{item_id}\t{table}\t{kb_id}\t{SOURCE}\ttest-embed\t{DIMS}\t{_literal(vector)}\n"
        )
    columns = {
        "chunks": "id, knowledge_base_id, source_id, text, meta",
        "full_documents": "id, knowledge_base_id, source_id, full_text_path, meta",
    }
    with conn.cursor() as cur:
        for table, buffer in items.items():
            buffer.seek(0)
            with cur.copy(f"COPY {schema}.{table} ({columns[table]}) FROM STDIN") as copy:
                copy.write(buffer.read())
        embeddings.seek(0)
        with cur.copy(
            f"COPY {schema}.embeddings (item_id, item_table, knowledge_base_id, source_id, "
            "embedding_model, dims, embedding) FROM STDIN"
        ) as copy:
            copy.write(embeddings.read())


def _build_shared_index(conn, schema: str) -> None:
    # Serial, but set on the session rather than as a table reloption: a parallel
    # HNSW build asks for a shared memory segment larger than a stock container's
    # /dev/shm, while the *query* side must keep its parallel plans available --
    # a parallel sequential scan is one of the plans the negative specs look for.
    conn.execute("SET maintenance_work_mem = '512MB'")
    conn.execute("SET max_parallel_maintenance_workers = 0")
    conn.execute(
        f"CREATE INDEX {HNSW_SHARED} ON {schema}.embeddings "
        f"USING hnsw ((embedding::vector({DIMS})) vector_cosine_ops) WHERE dims = {DIMS}"
    )


@pytest.fixture(scope="module")
def fixture_schema(engine):
    raw_dsn = engine.url.set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        _create_tables(conn, SCHEMA)
        _load(conn, SCHEMA, ROW_COUNTS, np.random.default_rng(2026), {KB_SMALL: SMALL_DOCUMENTS})
        _build_shared_index(conn, SCHEMA)
        # KB_INDEXED's own index, through the service's own DDL so its name and
        # predicate are exactly what the search's catalog probe looks for.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(pvi, "AI_SCHEMA", SCHEMA)
            conn.execute(pvi.per_kb_index_ddl(KB_INDEXED, DIMS))
        for table in ("embeddings", "chunks", "full_documents"):
            conn.execute(f"VACUUM ANALYZE {SCHEMA}.{table}")

        _create_tables(conn, SHARE_SCHEMA)
        _load(conn, SHARE_SCHEMA, SHARE_ROW_COUNTS, np.random.default_rng(7))
        _build_shared_index(conn, SHARE_SCHEMA)
        for table in ("embeddings", "chunks"):
            conn.execute(f"VACUUM ANALYZE {SHARE_SCHEMA}.{table}")
    yield
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        conn.execute(f"DROP SCHEMA IF EXISTS {SHARE_SCHEMA} CASCADE")


@pytest.fixture
def cap(monkeypatch):
    """``VECTOR_EXACT_SEARCH_MAX_ROWS``, injected; the real clamping still runs."""
    values = {bvs.EXACT_SEARCH_MAX_ROWS_SETTING: CAP}
    monkeypatch.setattr(bvs, "get_setting", lambda key: values[key])
    return values


@pytest.fixture
def schema(fixture_schema, monkeypatch, cap):
    monkeypatch.setattr(pvi, "AI_SCHEMA", SCHEMA)
    return SCHEMA


@pytest.fixture(scope="module")
def query_vectors():
    return _vectors(np.random.default_rng(99), 6)


# ---------------------------------------------------------------------------
# Running a search through the real store, and reading its plan
# ---------------------------------------------------------------------------


class _PlanRecordingSession:
    """A real session that EXPLAINs the search in place, then runs it.

    The plan is taken inside the store's own transaction, immediately before the
    real statement, so it is planned under exactly the settings the store has in
    force at that moment -- including the ones it puts back afterwards, which a
    plan taken later could not see.
    """

    def __init__(self, session):
        self._session = session
        self.statements: list[str] = []
        self.plans: list[str] = []
        self.searches: list[str] = []
        self.seqscan_at_the_search: list[str] = []

    def execute(self, clause, params=None):
        sql = clause.text
        self.statements.append(sql)
        if "ORDER BY" in sql:
            self.searches.append(sql)
            rows = self._session.execute(
                text("EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) " + sql), params
            ).all()
            self.plans.append("\n".join(str(r[0]) for r in rows))
            self.seqscan_at_the_search.append(
                self._session.execute(text("SELECT current_setting('enable_seqscan')")).scalar()
            )
        return self._session.execute(clause, params)

    def __getattr__(self, name):
        return getattr(self._session, name)


def _search(engine, kb_id, vector, *, store=_ChunkStore, schema_name=SCHEMA):
    """``(item ids in rank order, the recorder)`` for one real ``vector_search``."""
    with Session(engine) as session:
        recorder = _PlanRecordingSession(session)
        items = asyncio.run(
            store(db_session=recorder, knowledge_base_id=kb_id, schema=schema_name).vector_search(
                embedding=list(vector), top_k=TOP_K, _resolve=False
            )
        )
        session.rollback()
    assert len(recorder.plans) == 1, recorder.statements
    return [item.item_id for item in items], recorder


def _truth(engine, kb_id, vector, *, item_table="chunks", schema_name=SCHEMA) -> list[str]:
    """The exact top-k: every row of the knowledge base, ranked, no index scan possible."""
    with engine.connect() as conn:
        conn.execute(text("SET LOCAL enable_indexscan = off"))
        conn.execute(text("SET LOCAL enable_bitmapscan = on"))
        rows = conn.execute(
            text(
                f"SELECT item_id FROM {schema_name}.embeddings "
                f"WHERE knowledge_base_id = '{kb_id}' AND item_table = '{item_table}' "
                f"AND dims = {DIMS} "
                f"ORDER BY (embedding::vector({DIMS})) <=> CAST(:q AS vector({DIMS})) "
                f"LIMIT {TOP_K}"
            ),
            {"q": _literal(vector)},
        ).all()
        conn.rollback()
    return [str(r[0]) for r in rows]


def _embeddings_access(plan: str) -> list[str]:
    """The plan lines that read the ``embeddings`` relation or one of its indexes."""
    return [
        line.strip()
        for line in plan.splitlines()
        if "on embeddings" in line or HNSW_SHARED in line or "hnsw_kb_" in line or KB_BTREE in line
    ]


def _assert_exact_kb_btree_plan(plan: str) -> None:
    access = _embeddings_access(plan)
    assert any(
        f"Index Scan on {KB_BTREE}" in line or f"using {KB_BTREE}" in line for line in access
    ), f"the knowledge base's rows must come from its btree:\n{plan}"
    assert "hnsw" not in plan.lower(), f"no HNSW index may serve an exact search:\n{plan}"
    assert "Seq Scan on embeddings" not in plan, f"an exact search read the whole table:\n{plan}"
    assert "Sort" in plan, f"the rows are ranked by a sort:\n{plan}"


# ---------------------------------------------------------------------------
# 1. The small knowledge bases take the exact path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kb_id", [KB_SMALL, KB_EDGE], ids=["300-rows", "5000-rows"])
def test_a_small_unindexed_knowledge_base_is_a_btree_lookup_and_a_sort(
    engine, schema, query_vectors, kb_id
):
    _, recorder = _search(engine, kb_id, query_vectors[0])
    (plan,) = recorder.plans
    assert "OFFSET 0" in " ".join(recorder.searches[0].split())
    _assert_exact_kb_btree_plan(plan)
    assert recorder.seqscan_at_the_search == ["off"]


@pytest.mark.parametrize("kb_id", [KB_SMALL, KB_EDGE], ids=["300-rows", "5000-rows"])
def test_the_exact_path_returns_exactly_the_true_top_k(engine, schema, query_vectors, kb_id):
    for vector in query_vectors:
        found, _ = _search(engine, kb_id, vector)
        assert found == _truth(engine, kb_id, vector)


def test_the_fixture_reproduces_the_shared_index_plan_this_replaces(
    engine, schema, cap, query_vectors
):
    """The control: with the feature off, the planner sends KB_EDGE to the shared
    HNSW index. Without this, every spec above could pass on a fixture where the
    planner was already exact, and prove nothing."""
    cap[bvs.EXACT_SEARCH_MAX_ROWS_SETTING] = 0
    _, recorder = _search(engine, KB_EDGE, query_vectors[0])
    (plan,) = recorder.plans
    assert HNSW_SHARED in plan, f"the fixture no longer reproduces the defect:\n{plan}"
    assert "OFFSET 0" not in " ".join(recorder.searches[0].split())


def test_a_document_level_store_is_exact_below_the_cap(engine, schema, query_vectors):
    """The per-KB index covers chunks only, so this store never has one; the
    knowledge base's 60 document rows are all it may read."""
    for vector in query_vectors[:3]:
        found, recorder = _search(engine, KB_SMALL, vector, store=_DocumentStore)
        _assert_exact_kb_btree_plan(recorder.plans[0])
        assert found == _truth(engine, KB_SMALL, vector, item_table="full_documents")
    assert not [s for s in recorder.statements if "pg_class" in s or "to_regclass" in s], (
        "a store the per-KB index never covers must not ask the catalog"
    )


# ---------------------------------------------------------------------------
# 2. Everything else keeps its plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap_rows", [CAP, 10_000], ids=["over-the-cap", "under-the-cap"])
def test_a_knowledge_base_with_its_own_index_keeps_its_hnsw_path(
    engine, schema, cap, query_vectors, cap_rows
):
    """Over the cap, and under it too: an index of its own always wins."""
    cap[bvs.EXACT_SEARCH_MAX_ROWS_SETTING] = cap_rows
    _, recorder = _search(engine, KB_INDEXED, query_vectors[0])
    (plan,) = recorder.plans
    own = pvi.per_kb_index_name(KB_INDEXED, DIMS)
    assert own in plan, f"expected the knowledge base's own HNSW index:\n{plan}"
    assert "OFFSET 0" not in " ".join(recorder.searches[0].split())
    assert not [s for s in recorder.statements if "count(*)" in s], (
        "an indexed knowledge base must not pay for the count"
    )


def test_an_unindexed_knowledge_base_over_the_cap_keeps_todays_statement(
    engine, schema, query_vectors
):
    _, recorder = _search(engine, KB_GAP, query_vectors[0])
    assert "OFFSET 0" not in " ".join(recorder.searches[0].split())
    assert recorder.seqscan_at_the_search == ["on"], (
        "the penalty the decision ran under must be gone before today's plan is made"
    )


# ---------------------------------------------------------------------------
# 3. Never the whole table
# ---------------------------------------------------------------------------


def test_a_knowledge_base_that_is_a_third_of_its_table_is_still_a_lookup(
    engine, fixture_schema, monkeypatch, cap, query_vectors
):
    """Where the penalty is load-bearing. The control half shows the planner reading
    the whole table for the same fenced statement without it."""
    monkeypatch.setattr(pvi, "AI_SCHEMA", SHARE_SCHEMA)
    found, recorder = _search(engine, KB_SHARE, query_vectors[0], schema_name=SHARE_SCHEMA)
    (plan,) = recorder.plans
    _assert_exact_kb_btree_plan(plan)
    assert found == _truth(engine, KB_SHARE, query_vectors[0], schema_name=SHARE_SCHEMA)

    with engine.connect() as conn:
        rows = conn.execute(
            text("EXPLAIN (COSTS OFF) " + recorder.searches[0]),
            {"embedding": _literal(query_vectors[0])},
        ).all()
        conn.rollback()
    unaided = "\n".join(str(r[0]) for r in rows)
    assert "Seq Scan on embeddings" in unaided, (
        f"the control no longer shows the whole-table read the penalty prevents:\n{unaided}"
    )


def test_the_count_is_a_lookup_too(engine, fixture_schema, monkeypatch, cap):
    """The count runs under the same penalty and stops one row past the cap."""
    monkeypatch.setattr(pvi, "AI_SCHEMA", SHARE_SCHEMA)
    with Session(engine) as session:
        store = _ChunkStore(db_session=session, knowledge_base_id=KB_SHARE, schema=SHARE_SCHEMA)
        with store._scan_method_priced_out("enable_seqscan", not_applied="%s %s"):
            sql = (
                "EXPLAIN (ANALYZE, COSTS OFF) SELECT count(*) FROM (SELECT 1 FROM "
                f"\"{SHARE_SCHEMA}\".embeddings WHERE knowledge_base_id = '{KB_SHARE}' "
                f"AND item_table = 'chunks' AND dims = {DIMS} LIMIT {CAP + 1}) s"
            )
            plan = "\n".join(str(r[0]) for r in session.execute(text(sql)).all())
        assert store._capped_row_count(DIMS, CAP) == 3_000
        session.rollback()
    assert "Seq Scan on embeddings" not in plan, plan
    assert KB_BTREE in plan, plan


# ---------------------------------------------------------------------------
# 4. The plan cache cannot bring the HNSW index back
# ---------------------------------------------------------------------------


def _index_scans(engine, schema_name: str) -> dict[str, int]:
    with engine.connect() as conn:
        conn.execute(text("SELECT pg_stat_clear_snapshot()"))
        rows = conn.execute(
            text(
                "SELECT indexrelname, coalesce(idx_scan, 0) FROM pg_stat_all_indexes "
                "WHERE schemaname = :s"
            ),
            {"s": schema_name},
        ).all()
        conn.rollback()
    return {name: int(n) for name, n in rows}


def _wait_for_the_counters(engine, schema_name: str, before: dict[str, int], at_least: int):
    """A backend flushes its counters when it exits, and the exit is asynchronous."""
    deadline = time.monotonic() + 10
    while True:
        after = _index_scans(engine, schema_name)
        if (
            after.get(KB_BTREE, 0) - before.get(KB_BTREE, 0) >= at_least
            or time.monotonic() > deadline
        ):
            return after
        time.sleep(0.2)


def _settled_index_scans(engine, schema_name: str) -> dict[str, int]:
    """Counters with every earlier test's backend gone and its counts flushed.

    An idle pooled backend holds counts nobody can see yet, and adds them the next
    time it runs anything -- here, in the middle of the window being measured. So
    the pool is emptied and the counters read until two readings agree.
    """
    engine.dispose()
    deadline = time.monotonic() + 10
    previous = _index_scans(engine, schema_name)
    while True:
        time.sleep(0.3)
        current = _index_scans(engine, schema_name)
        if current == previous or time.monotonic() > deadline:
            engine.dispose()
            return current
        previous = current


@pytest.mark.parametrize("mode", ["auto", "force_generic_plan"])
def test_repeated_searches_on_one_connection_never_reach_an_hnsw_index(
    engine, schema, query_vectors, mode
):
    """Past psycopg's prepare threshold, so the later executions run a prepared
    statement -- under ``force_generic_plan``, a generic plan from the first."""
    runs = 12
    dedicated = create_engine(engine.url, poolclass=NullPool)
    truths = [_truth(engine, KB_EDGE, v) for v in query_vectors]
    before = _settled_index_scans(engine, SCHEMA)
    try:
        with Session(dedicated) as session:
            session.execute(text(f"SET plan_cache_mode = {mode}"))
            store = _ChunkStore(db_session=session, knowledge_base_id=KB_EDGE, schema=SCHEMA)
            for i in range(runs):
                vector = query_vectors[i % len(query_vectors)]
                items = asyncio.run(
                    store.vector_search(embedding=list(vector), top_k=TOP_K, _resolve=False)
                )
                assert [it.item_id for it in items] == truths[i % len(query_vectors)]
                session.commit()
    finally:
        dedicated.dispose()
    after = _wait_for_the_counters(engine, SCHEMA, before, at_least=runs)
    delta = {name: after.get(name, 0) - before.get(name, 0) for name in after}
    assert delta.get(HNSW_SHARED, 0) == 0, delta
    assert not {n: d for n, d in delta.items() if n.startswith("hnsw_kb_") and d}, delta
    # Each search is a count and a search on the btree, so at least one per run.
    assert delta.get(KB_BTREE, 0) >= runs, delta
