"""Database-backed checks for the exact search below the per-KB index floor.

``tests/unit/test_exact_search_below_per_kb_floor.py`` pins the statements the
store emits. Only a real planner shows what they are *for*: that a small
knowledge base with no index of its own is answered from the knowledge-base btree
and a sort -- never from an HNSW index, never by reading the whole
``embeddings`` table -- that the answer is exactly the true top-k, with the same
scores and columns the unfenced statement returns, and that the knowledge bases
this path is not for keep the plan they had.

The regime the main fixture represents
-------------------------------------
One ``embeddings`` table of ~55,000 rows at 384 dimensions in which three filler
knowledge bases hold 36,000 and the ones under test are small shares of it, with
every knowledge base's rows **interleaved** through the heap the way rows arrive in
a shared table over time. Both properties are load-bearing, and both were found
by the fixture not reproducing the defect without them:

- with each knowledge base's rows contiguous (one ``COPY`` per knowledge base),
  a btree lookup reads a handful of adjacent pages and the planner chooses it
  unaided at every size tried, so there is nothing for the fix to change;
- interleaved, the same lookup touches a page per row, and the planner sends
  ``KB_EDGE`` -- 5,000 rows, 9 % of this table -- to the *shared* HNSW index,
  the plan this change exists to replace; the control spec asserts the fixture
  still does, and that the answer it gets there is mostly wrong. In exploratory
  runs the crossover sat near 6-7 % (4,000 rows went to the index at 7 % of a
  55,000-row table and stayed exact at 6 % of a 66,000-row one), which is why
  ``KB_EDGE`` sits at 9 % -- and at the ceiling itself, where the ceiling's
  inclusive edge is exercised too.

384 dimensions keeps the build to seconds. A small 1536-dimension schema
(``WIDE_SCHEMA``) covers the width most embedding models produce, where a vector
is stored out of line and the costs move: plan shape and exactness only.

Two more small schemas cover the two bounds on what an exact search reads:

- ``SHARE_SCHEMA`` holds a knowledge base that is a third of its table: the
  shape where the planner prefers to read the whole table, fence or not, and
  where pricing sequential scans out is the only thing standing between an exact
  search and a scan of everyone's rows;
- ``MIXED_SCHEMA`` holds a knowledge base with 200 document rows beside 100,000
  chunk rows at another dimension. The only btree is on ``knowledge_base_id``, so
  an exact document search would read all 100,200 of them on every search; the
  decision's read budget is what keeps it on today's plan, and its count reads a
  bounded number of buffers doing so.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import time
import uuid

import numpy as np
import psycopg
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services import pg_vector_index as pvi

SCHEMA = "exact_below_floor_live_test"
SHARE_SCHEMA = "exact_below_floor_share_live_test"
MIXED_SCHEMA = "exact_below_floor_mixed_live_test"
WIDE_SCHEMA = "exact_below_floor_wide_live_test"
DIMS = 384
WIDE_DIMS = 1536
# The other population in MIXED_SCHEMA: a dimension no search here uses, so the
# vectors are cheap to load and nothing but the read budget separates the two.
OTHER_DIMS = 16
TOP_K = 20
KB_BTREE = "embeddings_kb_btree"


def _hnsw_shared(dims: int = DIMS) -> str:
    return f"idx_ai_embeddings_hnsw_{dims}"


HNSW_SHARED = _hnsw_shared()

# The knowledge bases under test, and the filler that makes them small shares.
KB_SMALL = "5b0e7a1c-0000-4000-8000-000000000001"  # 300 rows: small by any measure
KB_EDGE = "5b0e7a1c-0000-4000-8000-000000000002"  # 5,000, the cap: unaided, the planner picks HNSW
KB_GAP = "5b0e7a1c-0000-4000-8000-000000000003"  # 6,000: over the cap, no index
KB_INDEXED = "5b0e7a1c-0000-4000-8000-000000000004"  # 8,000: over the cap, its own index
FILLER_KBS = [f"5b0e7a1c-0000-4000-8000-0000000001{i:02d}" for i in range(3)]
SOURCE = "5b0e7a1c-0000-4000-8000-0000000000aa"
# A second source on every tenth item, so a ``source_ids`` restriction is a real
# restriction -- and so a search that mixed up ``source_id`` and ``meta`` could not
# return the right source by accident.
SOURCE_B = "5b0e7a1c-0000-4000-8000-0000000000ab"
SOURCE_B_EVERY = 10
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

KB_MIXED = "5b0e7a1c-0000-4000-8000-000000000301"  # 200 documents + 100,000 other rows
KB_MIXED_SMALL = "5b0e7a1c-0000-4000-8000-000000000302"  # 300 documents + 3,000 other rows
MIXED_DOCUMENTS = {KB_MIXED: 200, KB_MIXED_SMALL: 300}
MIXED_OTHER_ROWS = [(KB_MIXED, 100_000), (KB_MIXED_SMALL, 3_000)]

KB_WIDE = "5b0e7a1c-0000-4000-8000-000000000401"
WIDE_ROW_COUNTS = [
    (KB_WIDE, 600),
    *((f"5b0e7a1c-0000-4000-8000-00000000041{i}", 1_000) for i in range(3)),
]

# The shipped default, deliberately: the specs are about which side of it each
# knowledge base falls on, and 5,000 separates KB_EDGE from KB_GAP.
CAP = 5_000
BUDGET = bvs.exact_search_read_budget(CAP)


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


def _vectors(rng, n: int, dims: int = DIMS) -> np.ndarray:
    """Clustered unit vectors, so a top-k is not a coin toss between near-ties."""
    centres = rng.standard_normal((40, dims))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    v = centres[rng.integers(0, 40, n)] + 0.05 * rng.standard_normal((n, dims))
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


def _load(conn, schema: str, rows, rng) -> None:
    """``(item table, knowledge base, vector)`` rows, shuffled so none is contiguous.

    Every item carries its own ``meta`` and every tenth one the second source, so
    a search that returned the wrong column in either place is caught.
    """
    items = {"chunks": io.StringIO(), "full_documents": io.StringIO()}
    embeddings = io.StringIO()
    for i in rng.permutation(len(rows)):
        table, kb_id, vector = rows[i]
        item_id = str(uuid.uuid4())
        source = SOURCE_B if i % SOURCE_B_EVERY == 0 else SOURCE
        meta = json.dumps({"row": int(i)})
        items[table].write(f"{item_id}\t{kb_id}\t{source}\titem {i}\t{meta}\n")
        embeddings.write(
            f"{item_id}\t{table}\t{kb_id}\t{source}\ttest-embed\t{len(vector)}\t"
            f"{_literal(vector)}\n"
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
    for table in ("embeddings", "chunks", "full_documents"):
        conn.execute(f"VACUUM ANALYZE {schema}.{table}")


def _population(rng, table: str, counts, dims: int = DIMS) -> list:
    return [(table, kb_id, v) for kb_id, n in counts for v in _vectors(rng, n, dims)]


def _build_shared_index(conn, schema: str, dims: int = DIMS) -> None:
    # Serial, but set on the session rather than as a table reloption: a parallel
    # HNSW build asks for a shared memory segment larger than a stock container's
    # /dev/shm, while the *query* side must keep its parallel plans available --
    # a parallel sequential scan is one of the plans the negative specs look for.
    conn.execute("SET maintenance_work_mem = '512MB'")
    conn.execute("SET max_parallel_maintenance_workers = 0")
    conn.execute(
        f"CREATE INDEX {_hnsw_shared(dims)} ON {schema}.embeddings "
        f"USING hnsw ((embedding::vector({dims})) vector_cosine_ops) WHERE dims = {dims}"
    )
    conn.execute(f"ANALYZE {schema}.embeddings")


@pytest.fixture(scope="module")
def fixture_schema(engine):
    raw_dsn = engine.url.set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        rng = np.random.default_rng(2026)
        _create_tables(conn, SCHEMA)
        _load(
            conn,
            SCHEMA,
            _population(rng, "chunks", ROW_COUNTS)
            + _population(rng, "full_documents", [(KB_SMALL, SMALL_DOCUMENTS)]),
            rng,
        )
        _build_shared_index(conn, SCHEMA)
        # KB_INDEXED's own index, through the service's own DDL so its name and
        # predicate are exactly what the search's catalog probe looks for.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(pvi, "AI_SCHEMA", SCHEMA)
            conn.execute(pvi.per_kb_index_ddl(KB_INDEXED, DIMS))
        conn.execute(f"ANALYZE {SCHEMA}.embeddings")

        rng = np.random.default_rng(7)
        _create_tables(conn, SHARE_SCHEMA)
        _load(conn, SHARE_SCHEMA, _population(rng, "chunks", SHARE_ROW_COUNTS), rng)
        _build_shared_index(conn, SHARE_SCHEMA)

        rng = np.random.default_rng(11)
        _create_tables(conn, MIXED_SCHEMA)
        _load(
            conn,
            MIXED_SCHEMA,
            _population(rng, "full_documents", MIXED_DOCUMENTS.items())
            + _population(rng, "chunks", MIXED_OTHER_ROWS, OTHER_DIMS),
            rng,
        )

        rng = np.random.default_rng(13)
        _create_tables(conn, WIDE_SCHEMA)
        _load(conn, WIDE_SCHEMA, _population(rng, "chunks", WIDE_ROW_COUNTS, WIDE_DIMS), rng)
        _build_shared_index(conn, WIDE_SCHEMA, WIDE_DIMS)
    yield
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        for schema in (SCHEMA, SHARE_SCHEMA, MIXED_SCHEMA, WIDE_SCHEMA):
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


@pytest.fixture
def cap(monkeypatch):
    """``VECTOR_EXACT_SEARCH_MAX_ROWS``, injected; the real clamping still runs.

    Injected through ``get_setting_strict``, the reader the store uses: these
    specs run outside a Flask app context, where ``ai.project_settings`` cannot be
    read and an unreadable ceiling turns the exact path off.
    """
    values = {bvs.EXACT_SEARCH_MAX_ROWS_SETTING: CAP}
    monkeypatch.setattr(bvs, "get_setting_strict", lambda key: values[key])
    return values


@pytest.fixture
def schema(fixture_schema, monkeypatch, cap):
    monkeypatch.setattr(pvi, "AI_SCHEMA", SCHEMA)
    return SCHEMA


@pytest.fixture
def in_schema(fixture_schema, monkeypatch, cap):
    """Point the catalog reader at one of the small schemas."""

    def point(name: str) -> str:
        monkeypatch.setattr(pvi, "AI_SCHEMA", name)
        return name

    return point


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
        self.settings_at_the_search: list[dict[str, str]] = []

    def execute(self, clause, params=None):
        sql = clause.text
        self.statements.append(sql)
        if "ORDER BY" in sql:
            self.searches.append(sql)
            rows = self._session.execute(
                text("EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) " + sql), params
            ).all()
            self.plans.append("\n".join(str(r[0]) for r in rows))
            row = self._session.execute(
                text(
                    "SELECT current_setting('enable_seqscan'), "
                    "current_setting('enable_bitmapscan'), current_setting('enable_indexscan')"
                )
            ).one()
            self.settings_at_the_search.append(
                {"enable_seqscan": row[0], "enable_bitmapscan": row[1], "enable_indexscan": row[2]}
            )
        return self._session.execute(clause, params)

    def __getattr__(self, name):
        return getattr(self._session, name)


def _search(engine, kb_id, vector, *, store=_ChunkStore, schema_name=SCHEMA, **kwargs):
    """``(items in rank order, the recorder)`` for one real ``vector_search``."""
    with Session(engine) as session:
        recorder = _PlanRecordingSession(session)
        items = asyncio.run(
            store(db_session=recorder, knowledge_base_id=kb_id, schema=schema_name).vector_search(
                embedding=list(vector), top_k=TOP_K, _resolve=False, **kwargs
            )
        )
        session.rollback()
    assert len(recorder.plans) == 1, recorder.statements
    return items, recorder


def _ids(items) -> list[str]:
    return [item.item_id for item in items]


def _is_fenced(recorder) -> bool:
    return "OFFSET 0" in " ".join(recorder.searches[0].split())


def _truth(
    engine,
    kb_id,
    vector,
    *,
    item_table="chunks",
    schema_name=SCHEMA,
    source_id: str | None = None,
) -> list[str]:
    """The exact top-k: every matching row of the knowledge base, ranked, no index scan possible."""
    dims = len(vector)
    source = f"AND e.source_id = '{source_id}' " if source_id else ""
    with engine.connect() as conn:
        conn.execute(text("SET LOCAL enable_indexscan = off"))
        conn.execute(text("SET LOCAL enable_bitmapscan = on"))
        rows = conn.execute(
            text(
                f"SELECT e.item_id FROM {schema_name}.embeddings e "
                f"WHERE e.knowledge_base_id = '{kb_id}' AND e.item_table = '{item_table}' "
                f"AND e.dims = {dims} {source}"
                f"ORDER BY (e.embedding::vector({dims})) <=> CAST(:q AS vector({dims})) "
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
        if "on embeddings" in line or "hnsw" in line or KB_BTREE in line
    ]


def _assert_exact_kb_btree_plan(plan: str) -> None:
    access = _embeddings_access(plan)
    assert any(f"using {KB_BTREE}" in line or f"on {KB_BTREE}" in line for line in access), (
        f"the knowledge base's rows must come from its btree:\n{plan}"
    )
    assert "hnsw" not in plan.lower(), f"no HNSW index may serve an exact search:\n{plan}"
    assert "Seq Scan on embeddings" not in plan, f"an exact search read the whole table:\n{plan}"
    assert "Sort" in plan, f"the rows are ranked by a sort:\n{plan}"


def _buffers(plan: str) -> int:
    """Shared buffers the whole statement touched: the top node's hit + read."""
    line = next(line for line in plan.splitlines() if "Buffers: shared" in line)
    return sum(int(n) for n in re.findall(r"(?:hit|read)=(\d+)", line))


# ---------------------------------------------------------------------------
# 1. The small knowledge bases take the exact path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kb_id", [KB_SMALL, KB_EDGE], ids=["300-rows", "5000-rows"])
def test_a_small_unindexed_knowledge_base_is_a_btree_lookup_and_a_sort(
    engine, schema, query_vectors, kb_id
):
    _, recorder = _search(engine, kb_id, query_vectors[0])
    (plan,) = recorder.plans
    assert _is_fenced(recorder)
    _assert_exact_kb_btree_plan(plan)
    assert recorder.settings_at_the_search == [
        {"enable_seqscan": "off", "enable_bitmapscan": "off", "enable_indexscan": "on"}
    ]


@pytest.mark.parametrize("kb_id", [KB_SMALL, KB_EDGE], ids=["300-rows", "5000-rows"])
def test_the_exact_path_returns_exactly_the_true_top_k(engine, schema, query_vectors, kb_id):
    for vector in query_vectors:
        items, _ = _search(engine, kb_id, vector)
        assert _ids(items) == _truth(engine, kb_id, vector)


@pytest.mark.parametrize("kb_id", [KB_SMALL, KB_EDGE], ids=["300-rows", "5000-rows"])
def test_the_fenced_statement_returns_the_same_rows_scores_and_columns_as_the_unfenced_one(
    engine, schema, cap, query_vectors, kb_id
):
    """Not only the same ids: the same similarity, source and metadata per row.

    The unfenced statement is the one ``vector_search`` runs with the feature off,
    made exact here by pricing index scans out on its transaction. A fenced
    statement that computed its score with the wrong operator, or returned
    ``meta`` where ``source_id`` belongs, would still rank the right ids.
    """

    def rows(items):
        return [(i.item_id, round(i.score, 9), i.source_id, i.meta) for i in items]

    for vector in query_vectors[:3]:
        fenced_items, recorder = _search(engine, kb_id, vector)
        assert _is_fenced(recorder)
        cap[bvs.EXACT_SEARCH_MAX_ROWS_SETTING] = 0
        with Session(engine) as session:
            session.execute(text("SET LOCAL enable_indexscan = off"))
            unfenced_items = asyncio.run(
                _ChunkStore(
                    db_session=session, knowledge_base_id=kb_id, schema=SCHEMA
                ).vector_search(embedding=list(vector), top_k=TOP_K, _resolve=False)
            )
            session.rollback()
        cap[bvs.EXACT_SEARCH_MAX_ROWS_SETTING] = CAP
        assert rows(fenced_items) == rows(unfenced_items)
        assert {i.source_id for i in fenced_items} <= {SOURCE, SOURCE_B}
        assert all(isinstance(i.meta, dict) and "row" in i.meta for i in fenced_items)


def test_the_fixture_reproduces_the_shared_index_plan_this_replaces(
    engine, schema, cap, query_vectors
):
    """The control: with the feature off, the planner sends KB_EDGE to the shared
    HNSW index, and the answer it gets there is mostly not the true top-k. Without
    this, every spec above could pass on a fixture where the planner was already
    exact, and prove nothing."""
    cap[bvs.EXACT_SEARCH_MAX_ROWS_SETTING] = 0
    items, recorder = _search(engine, KB_EDGE, query_vectors[0])
    (plan,) = recorder.plans
    assert HNSW_SHARED in plan, f"the fixture no longer reproduces the defect:\n{plan}"
    assert not _is_fenced(recorder)
    recall = []
    for vector in query_vectors:
        items, _ = _search(engine, KB_EDGE, vector)
        truth = _truth(engine, KB_EDGE, vector)
        recall.append(len(set(_ids(items)) & set(truth)) / len(truth))
    # Mean recall under 0.5: measured 0.03-0.28 across HNSW builds (a single
    # query anything from 0.00 to 0.80), so half is a bound with headroom that
    # still says "mostly wrong".
    assert sum(recall) / len(recall) < 0.5, recall


def test_a_document_level_store_is_exact_below_the_cap(engine, schema, query_vectors):
    """The per-KB index covers chunks only, so this store never has one; the
    knowledge base's 60 document rows are all it ranks."""
    for vector in query_vectors[:3]:
        items, recorder = _search(engine, KB_SMALL, vector, store=_DocumentStore)
        _assert_exact_kb_btree_plan(recorder.plans[0])
        assert _ids(items) == _truth(engine, KB_SMALL, vector, item_table="full_documents")
    assert not [s for s in recorder.statements if "pg_class" in s or "to_regclass" in s], (
        "a store the per-KB index never covers must not ask the catalog"
    )


def test_at_1536_dimensions_the_exact_path_is_the_same_lookup_and_exact(engine, in_schema):
    """Where a vector is stored out of line -- one TOAST value in four chunks --
    and the planner's costs move. Plan shape and exactness only."""
    wide = in_schema(WIDE_SCHEMA)
    for vector in _vectors(np.random.default_rng(5), 3, WIDE_DIMS):
        items, recorder = _search(engine, KB_WIDE, vector, schema_name=wide)
        assert _is_fenced(recorder)
        _assert_exact_kb_btree_plan(recorder.plans[0])
        assert _ids(items) == _truth(engine, KB_WIDE, vector, schema_name=wide)


# ---------------------------------------------------------------------------
# 2. Everything else keeps its plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap_rows", [CAP, 10_000], ids=["over-the-cap", "under-the-cap"])
def test_a_knowledge_base_with_its_own_index_keeps_its_hnsw_path(
    engine, schema, cap, query_vectors, cap_rows
):
    """Over the cap, and at or below it too: an index of its own keeps the index path."""
    cap[bvs.EXACT_SEARCH_MAX_ROWS_SETTING] = cap_rows
    _, recorder = _search(engine, KB_INDEXED, query_vectors[0])
    (plan,) = recorder.plans
    own = pvi.per_kb_index_name(KB_INDEXED, DIMS)
    assert own in plan, f"expected the knowledge base's own HNSW index:\n{plan}"
    assert not _is_fenced(recorder)
    assert not [s for s in recorder.statements if "count(*)" in s], (
        "an indexed knowledge base must not pay for the count"
    )


def test_an_unindexed_knowledge_base_over_the_cap_keeps_todays_statement(
    engine, schema, query_vectors
):
    _, recorder = _search(engine, KB_GAP, query_vectors[0])
    assert not _is_fenced(recorder)
    assert recorder.settings_at_the_search == [
        {"enable_seqscan": "on", "enable_bitmapscan": "on", "enable_indexscan": "on"}
    ], "the penalty the decision ran under must be gone before today's plan is made"


def test_a_restricted_search_on_a_small_knowledge_base_stays_restricted_and_exact(
    engine, schema, query_vectors
):
    """A ``source_ids`` search keeps its own exact path: it never takes the fenced
    statement, which carries no restriction, and it never pays for the count."""
    for vector in query_vectors[:3]:
        items, recorder = _search(engine, KB_SMALL, vector, source_ids=[SOURCE_B])
        assert not _is_fenced(recorder)
        assert not [s for s in recorder.statements if "count(*)" in s], recorder.statements
        assert items, "the restriction matches rows; an empty answer proves nothing"
        assert {i.source_id for i in items} == {SOURCE_B}, [i.source_id for i in items]
        assert _ids(items) == _truth(engine, KB_SMALL, vector, source_id=SOURCE_B)


# ---------------------------------------------------------------------------
# 3. Never the whole table, and never the whole knowledge base past the budget
# ---------------------------------------------------------------------------


def test_a_knowledge_base_that_is_a_third_of_its_table_is_still_a_lookup(
    engine, in_schema, query_vectors
):
    """Where the penalty is load-bearing. The control half shows the planner reading
    the whole table for the same fenced statement without it."""
    share = in_schema(SHARE_SCHEMA)
    items, recorder = _search(engine, KB_SHARE, query_vectors[0], schema_name=share)
    (plan,) = recorder.plans
    _assert_exact_kb_btree_plan(plan)
    assert _ids(items) == _truth(engine, KB_SHARE, query_vectors[0], schema_name=share)

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


def _decision_count(recorder) -> str:
    (count,) = [s for s in recorder.statements if "count(*)" in s]
    return count


def _explain_under_the_penalty(engine, store_cls, kb_id, schema_name, sql) -> str:
    """EXPLAIN (ANALYZE, BUFFERS) of ``sql`` under the store's own scan penalty."""
    with Session(engine) as session:
        store = store_cls(db_session=session, knowledge_base_id=kb_id, schema=schema_name)
        with store._scan_methods_priced_out(
            "enable_seqscan", "enable_bitmapscan", not_applied="%s %s"
        ):
            plan = "\n".join(
                str(r[0])
                for r in session.execute(text("EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) " + sql))
            )
        session.rollback()
    return plan


def test_the_count_is_a_lookup_too(engine, in_schema, query_vectors):
    """On the large-share table the count, like the search, is a btree lookup."""
    share = in_schema(SHARE_SCHEMA)
    _, recorder = _search(engine, KB_SHARE, query_vectors[0], schema_name=share)
    plan = _explain_under_the_penalty(
        engine, _ChunkStore, KB_SHARE, share, _decision_count(recorder)
    )
    assert "Seq Scan on embeddings" not in plan, plan
    assert f"Index Scan using {KB_BTREE}" in plan, plan


def test_a_small_population_beside_a_large_one_keeps_todays_plan(engine, in_schema, query_vectors):
    """200 document rows beside 100,000 other rows of the same knowledge base.

    Before the read budget, the document search took the exact path -- 200 is
    far under the ceiling -- and its count and search each read every one of the
    knowledge base's 100,200 rows through the only btree it has. Now the count
    stops at the budget, sees there is more, and the search keeps today's plan.
    """
    mixed = in_schema(MIXED_SCHEMA)
    _, recorder = _search(
        engine, KB_MIXED, query_vectors[0], store=_DocumentStore, schema_name=mixed
    )
    assert not _is_fenced(recorder), "a knowledge base past the read budget is not read in full"


def test_the_decision_on_a_large_knowledge_base_reads_a_bounded_number_of_buffers(
    engine, in_schema, query_vectors
):
    """The count is a plain index scan that stops one row past the budget: its heap
    walk ends at ``BUDGET + 1`` rows and its btree walk with it, where a bitmap scan
    would first collect every one of the knowledge base's 100,200 entries."""
    mixed = in_schema(MIXED_SCHEMA)
    _, recorder = _search(
        engine, KB_MIXED, query_vectors[0], store=_DocumentStore, schema_name=mixed
    )
    count = _decision_count(recorder)
    assert f"LIMIT {BUDGET + 1}" in " ".join(count.split()), count
    plan = _explain_under_the_penalty(engine, _DocumentStore, KB_MIXED, mixed, count)
    assert f"Index Scan using {KB_BTREE}" in plan, plan
    assert "Bitmap" not in plan and "Seq Scan" not in plan, plan
    scanned = re.search(rf"Index Scan using {KB_BTREE} .*?rows=(\d+)", plan)
    assert scanned and int(scanned.group(1)) == BUDGET + 1, plan

    # The control is the count this replaced: the population filters inside the
    # limited scan, so the LIMIT bounded the rows it returned and not the rows it
    # read -- every row of the knowledge base, on the heap, to find 200.
    previous = _explain_under_the_penalty(
        engine,
        _DocumentStore,
        KB_MIXED,
        mixed,
        f'SELECT count(*) FROM (SELECT 1 FROM "{mixed}".embeddings '
        f"WHERE knowledge_base_id = '{KB_MIXED}' AND item_table = 'full_documents' "
        f"AND dims = {DIMS} LIMIT {CAP + 1}) s",
    )
    assert "Rows Removed by Filter: 100000" in previous, previous
    bounded, unbounded = _buffers(plan), _buffers(previous)
    # The budget is a fifth of this knowledge base, and the read shrinks with it.
    assert bounded * 3 < unbounded, (bounded, unbounded, plan, previous)


def test_other_populations_inside_the_budget_still_search_exactly(engine, in_schema, query_vectors):
    """The positive control: 300 documents beside 3,000 other rows is inside the
    budget, so the document search is exact -- and exactly right."""
    mixed = in_schema(MIXED_SCHEMA)
    for vector in query_vectors[:3]:
        items, recorder = _search(
            engine, KB_MIXED_SMALL, vector, store=_DocumentStore, schema_name=mixed
        )
        assert _is_fenced(recorder)
        _assert_exact_kb_btree_plan(recorder.plans[0])
        assert _ids(items) == _truth(
            engine, KB_MIXED_SMALL, vector, item_table="full_documents", schema_name=mixed
        )


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


def _settled_index_scans(engine, schema_name: str) -> dict[str, int]:
    """Counters with every earlier backend gone and its counts flushed.

    An idle pooled backend holds counts nobody can see yet, and adds them the next
    time it runs anything -- in the middle of a window being measured. So the pool
    is emptied and the counters read until two readings agree.
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


def _searches_on_one_connection(engine, kb_id, vectors, *, mode: str):
    """``vector_search`` once per vector, all on ONE connection in one transaction.

    One connection is the point: psycopg prepares a statement after five
    executions on the same connection, and PostgreSQL's plan cache lives in that
    backend -- so a spec about cached plans needs every execution on the same
    backend, under the ``plan_cache_mode`` it set there. Returns the answers and
    ``pg_prepared_statements``' ``(statement, generic_plans, custom_plans)`` for
    the fenced search, read on that connection before it closes.
    """
    probe = create_engine(_dsn())
    connection = probe.connect()
    answers = []
    try:
        with Session(bind=connection) as session:
            session.execute(text(f"SET plan_cache_mode = '{mode}'"))
            store = _ChunkStore(db_session=session, knowledge_base_id=kb_id, schema=SCHEMA)
            for vector in vectors:
                items = asyncio.run(
                    store.vector_search(embedding=list(vector), top_k=TOP_K, _resolve=False)
                )
                answers.append(_ids(items))
            prepared = session.execute(
                text(
                    "SELECT statement, generic_plans, custom_plans FROM pg_prepared_statements "
                    "WHERE statement LIKE '%OFFSET 0%'"
                )
            ).all()
            session.rollback()
    finally:
        connection.close()
        probe.dispose()
    return answers, prepared


@pytest.mark.parametrize("mode", ["auto", "force_generic_plan"])
def test_repeated_searches_on_one_connection_never_reach_an_hnsw_index(
    engine, schema, query_vectors, mode
):
    """Twelve searches on one backend, past psycopg's prepare threshold, so the
    later executions run a prepared statement -- and, as the control below
    requires, at least one of them on PostgreSQL's *generic* plan."""
    runs = 12
    vectors = [query_vectors[i % len(query_vectors)] for i in range(runs)]
    truths = [_truth(engine, KB_EDGE, v) for v in vectors]
    before = _settled_index_scans(engine, SCHEMA)
    answers, prepared = _searches_on_one_connection(engine, KB_EDGE, vectors, mode=mode)
    after = _settled_index_scans(engine, SCHEMA)
    assert answers == truths

    # The negative control: without a prepared statement that went generic, the
    # counters below would say nothing about the plan cache.
    assert prepared, "psycopg never prepared the fenced search on this connection"
    assert sum(row[1] for row in prepared) > 0, prepared

    delta = {name: after.get(name, 0) - before.get(name, 0) for name in after}
    assert delta.get(HNSW_SHARED, 0) == 0, delta
    assert not {n: d for n, d in delta.items() if n.startswith("hnsw_kb_") and d}, delta
    # Each search is the decision's count and the fenced search, both on the
    # knowledge-base btree: at least two scans a run.
    assert delta.get(KB_BTREE, 0) >= 2 * runs, delta
