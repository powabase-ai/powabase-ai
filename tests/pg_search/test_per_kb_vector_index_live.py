"""Database-backed checks for the per-knowledge-base partial HNSW index.

Unit specs show the DDL and the query text are shaped right. Only a real
Postgres with pgvector shows the things this feature actually rests on: that the
planner *chooses* the partial index, that it cannot choose it without the
embeddings-side predicate, that the index is complete (an exhaustive search
through it returns exactly an exact scan's top-k), and -- the one that nearly
sank the design -- that the same query issued through the real service code path
keeps the index after psycopg has prepared the statement.

It lives beside the pg_search suite because that suite's image is the one with
both extensions and the conftest that scopes timeouts and leaves the ``ai``
schema alone; nothing here needs ``pg_search`` itself. Every test works in a
scratch schema of its own, shaped like ``ai.chunks`` and ``ai.embeddings``.

The fixture is 12,000 embeddings of 1536 dimensions, 9,000 of them in one
knowledge base. That size is not arbitrary: below roughly a few thousand rows
per knowledge base the planner correctly prefers an exact bitmap scan and sort
over any HNSW index, so a smaller fixture could not show a plan choice at all.
It costs about 15 s to build, once per module.
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

from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services import pg_vector_index as pvi

SCHEMA = "vector_perkb_live_test"
DIMS = 1536

KB_BIG = "9f8b1c2e-0000-4000-8000-000000000001"
KB_SMALL = "9f8b1c2e-0000-4000-8000-000000000002"
SOURCE = "9f8b1c2e-0000-4000-8000-0000000000aa"

BIG_ROWS = 9_000
SMALL_ROWS = 3_000

# Thresholds each test wants, injected through the settings reader so the real
# clamping and hysteresis logic runs rather than being bypassed.
_MEM_MB = 256


class _ChunkStore(bvs.BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


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
    """Clustered unit vectors, so neighbourhoods are meaningful rather than uniform.

    Uniformly random high-dimensional vectors are nearly equidistant, so a
    top-k comparison against an exact scan would be a coin toss between tied
    candidates and would say nothing about the index.
    """
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


@pytest.fixture(scope="module")
def query_vectors(engine):
    return _vectors(np.random.default_rng(99), 6)


@pytest.fixture(scope="module")
def fixture_schema(engine):
    """The two tables the vector path joins, with a knowledge base worth indexing."""
    raw_dsn = engine.url.set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        conn.execute(f"CREATE SCHEMA {SCHEMA}")
        conn.execute(f"""
            CREATE TABLE {SCHEMA}.chunks (
                id uuid PRIMARY KEY,
                knowledge_base_id uuid NOT NULL,
                source_id uuid NOT NULL,
                text text NOT NULL,
                meta jsonb DEFAULT '{{}}'::jsonb
            )
        """)
        conn.execute(f"""
            CREATE TABLE {SCHEMA}.embeddings (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                item_id uuid NOT NULL,
                item_table varchar(50) NOT NULL,
                knowledge_base_id uuid NOT NULL,
                source_id uuid NOT NULL,
                embedding_model varchar(255) NOT NULL,
                dims smallint NOT NULL,
                embedding vector NOT NULL,
                CONSTRAINT embeddings_item_id_embedding_model_key UNIQUE (item_id, embedding_model)
            )
        """)
        conn.execute(f"CREATE INDEX ON {SCHEMA}.chunks (knowledge_base_id)")
        conn.execute(f"CREATE INDEX ON {SCHEMA}.embeddings (item_id)")
        conn.execute(f"CREATE INDEX ON {SCHEMA}.embeddings (knowledge_base_id)")

        rng = np.random.default_rng(4242)
        for kb_id, rows in ((KB_BIG, BIG_ROWS), (KB_SMALL, SMALL_ROWS)):
            vectors = _vectors(rng, rows)
            chunks = io.StringIO()
            embeddings = io.StringIO()
            for i in range(rows):
                item_id = str(uuid.uuid4())
                chunks.write(f"{item_id}\t{kb_id}\t{SOURCE}\tpassage {i}\t{{}}\n")
                embeddings.write(
                    f"{item_id}\tchunks\t{kb_id}\t{SOURCE}\ttest-embed\t{DIMS}\t"
                    f"{_literal(vectors[i])}\n"
                )
            chunks.seek(0)
            embeddings.seek(0)
            with conn.cursor() as cur:
                with cur.copy(
                    f"COPY {SCHEMA}.chunks (id, knowledge_base_id, source_id, text, meta) "
                    "FROM STDIN"
                ) as copy:
                    copy.write(chunks.read())
                with cur.copy(
                    f"COPY {SCHEMA}.embeddings (item_id, item_table, knowledge_base_id, "
                    "source_id, embedding_model, dims, embedding) FROM STDIN"
                ) as copy:
                    copy.write(embeddings.read())

        # The shared per-dimension index every project has today. It is what the
        # query falls back to, and what the old query shape picks even when a
        # partial index is available.
        conn.execute(f"SET maintenance_work_mem = '{_MEM_MB}MB'")
        conn.execute(
            f"CREATE INDEX idx_ai_embeddings_hnsw_{DIMS} ON {SCHEMA}.embeddings "
            f"USING hnsw ((embedding::vector({DIMS})) vector_cosine_ops) WHERE dims = {DIMS}"
        )
        conn.execute(f"VACUUM ANALYZE {SCHEMA}.embeddings")
        conn.execute(f"VACUUM ANALYZE {SCHEMA}.chunks")
    yield
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")


@pytest.fixture
def schema(engine, fixture_schema, monkeypatch):
    """Point the service's DDL at the scratch schema; leave no partial index behind."""
    monkeypatch.setattr(pvi, "AI_SCHEMA", SCHEMA)
    _drop_all_partial_indexes(engine)
    yield SCHEMA
    _drop_all_partial_indexes(engine)


@pytest.fixture
def settings(monkeypatch):
    """Injectable thresholds, read through the real clamping and hysteresis code."""
    values = {
        "VECTOR_PER_KB_INDEX_MIN_ROWS": 5_000,
        "VECTOR_PER_KB_INDEX_DROP_ROWS": 2_500,
        "VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB": _MEM_MB,
    }
    monkeypatch.setattr(pvi, "get_setting", lambda key: values[key])
    # The start-up sweep deliberately does not read through db.session, so it
    # has its own settings source; the tests drive both from here.
    monkeypatch.setattr(pvi, "read_overrides", lambda conn, *keys: dict(values))
    return values


def _drop_all_partial_indexes(engine) -> None:
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        names = [
            row[0]
            for row in conn.execute(
                text(
                    "SELECT c.relname FROM pg_class c JOIN pg_namespace n "
                    "ON n.oid = c.relnamespace WHERE n.nspname = :s AND c.relkind = 'i' "
                    "AND c.relname LIKE 'hnsw_kb_%'"
                ),
                {"s": SCHEMA},
            ).all()
        ]
        for name in names:
            conn.execute(text(f'DROP INDEX CONCURRENTLY IF EXISTS "{SCHEMA}".{name}'))


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


class _RecordingSession:
    """A real session that also keeps the statements the store hands it.

    The statements are taken before SQLAlchemy rewrites ``:name`` into the
    driver's own placeholders, so what the plan tests below examine is the
    service's own SQL, executable again as-is.
    """

    def __init__(self, session):
        self._session = session
        self.statements: list[tuple[str, dict]] = []

    def execute(self, clause, params=None):
        self.statements.append((clause.text, dict(params or {})))
        return self._session.execute(clause, params)

    def __getattr__(self, name):
        return getattr(self._session, name)


def _capture_search_sql(engine, kb_id, embedding):
    """The SQL the real store issues, and the parameters it binds."""
    with Session(engine) as session:
        recorder = _RecordingSession(session)
        store = _ChunkStore(db_session=recorder, knowledge_base_id=kb_id, schema=SCHEMA)
        asyncio.run(store.vector_search(embedding=list(embedding), top_k=20))
        session.rollback()
    searches = [pair for pair in recorder.statements if "ORDER BY" in pair[0]]
    assert searches, f"the store issued no search query: {recorder.statements}"
    return searches[0]


def _explain(session, sql: str, params: dict) -> str:
    _arm(session)
    rows = session.execute(text("EXPLAIN " + sql), params).all()
    plan = "\n".join(str(r[0]) for r in rows)
    session.rollback()
    return plan


def _arm(session) -> None:
    """Set the GUC the store sets, in the transaction the query will run in.

    ``SET LOCAL`` dies with its transaction, so a query run after a
    ``rollback()`` would run without it -- and without iterative scanning the
    shared index under-returns on a KB-filtered search, which is the whole
    reason the store sets it.
    """
    session.execute(text(f"SET LOCAL hnsw.iterative_scan = '{bvs.HNSW_ITERATIVE_SCAN_MODE}'"))


def _search(session, sql: str, params: dict) -> list:
    """Run one search the way the store runs it, then release the transaction."""
    _arm(session)
    try:
        return session.execute(text(sql), params).all()
    finally:
        session.rollback()


def _without_the_embeddings_predicate(sql: str) -> str:
    """Today's query shape: the same statement with the new line taken back out."""
    kept = [line for line in sql.splitlines() if "e.knowledge_base_id" not in line]
    assert len(kept) == len(sql.splitlines()) - 1, "expected exactly one such line"
    return "\n".join(kept)


def _idx_scans(engine, *names: str) -> dict[str, int]:
    """How many index scans each index has served, on a connection of its own.

    Two things about Postgres' statistics make this fiddly, and both bit:

    - a backend freezes its statistics view for the length of a transaction
      (``stats_fetch_consistency = cache``), so reading twice in one transaction
      reports the same numbers, hence ``pg_stat_clear_snapshot``;
    - a backend flushes its own counters no more often than once a second, so a
      connection that has run the work and gone idle in the pool can be sitting
      on counts nobody can see yet. The caller ends that backend before reading.
    """
    with engine.connect() as conn:
        conn.execute(text("SELECT pg_stat_clear_snapshot()"))
        rows = conn.execute(
            text(
                "SELECT indexrelname, coalesce(idx_scan, 0) FROM pg_stat_all_indexes "
                "WHERE schemaname = :s AND indexrelname = ANY(:n)"
            ),
            {"s": SCHEMA, "n": list(names)},
        ).all()
        conn.rollback()
    found = {name: int(count) for name, count in rows}
    missing = [name for name in names if name not in found]
    assert not missing, f"no such index in {SCHEMA}: {missing}"
    return {name: found[name] for name in names}


def _build_big_index(engine, settings) -> str:
    outcome = pvi.ensure_per_kb_vector_index(KB_BIG, engine=engine)
    name = pvi.per_kb_index_name(KB_BIG, DIMS)
    assert outcome["built"] == [name], outcome
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"ANALYZE {SCHEMA}.embeddings"))
    return name


# ---------------------------------------------------------------------------
# 1. The planner's choice
# ---------------------------------------------------------------------------


def test_the_planner_chooses_the_partial_index_when_it_exists(
    engine, schema, settings, query_vectors
):
    name = _build_big_index(engine, settings)
    sql, params = _capture_search_sql(engine, KB_BIG, query_vectors[0])
    with Session(engine) as session:
        plan = _explain(session, sql, params)
    assert f"Index Scan using {name}" in plan, plan
    assert f"idx_ai_embeddings_hnsw_{DIMS}" not in plan, plan


def test_todays_query_shape_cannot_use_the_partial_index(engine, schema, settings, query_vectors):
    """The load-bearing negative result: the index alone buys nothing.

    PostgreSQL matches a partial index only from a restriction clause on the
    relation the index is on, and there is no equivalence class linking
    ``c.knowledge_base_id`` to ``e.knowledge_base_id`` through the join.
    """
    name = _build_big_index(engine, settings)
    sql, params = _capture_search_sql(engine, KB_BIG, query_vectors[0])
    old_sql = _without_the_embeddings_predicate(sql)
    with Session(engine) as session:
        plan = _explain(session, old_sql, params)
    assert name not in plan, f"the old shape must not reach the partial index:\n{plan}"
    assert f"idx_ai_embeddings_hnsw_{DIMS}" in plan, plan


def test_the_new_predicate_alone_is_no_worse_than_today(engine, schema, settings, query_vectors):
    """With no partial index anywhere, the predicate must not cost anything.

    It can change the plan -- the planner gains a way to restrict
    ``ai.embeddings`` by its own knowledge_base_id index -- so what is pinned is
    the outcome rather than the shape: a full answer, and never fewer rows than
    today's query returns on the same state.
    """
    sql, params = _capture_search_sql(engine, KB_BIG, query_vectors[0])
    old_sql = _without_the_embeddings_predicate(sql)
    with Session(engine) as session:
        for vector in query_vectors:
            call = {**params, "embedding": _literal(vector)}
            today = _search(session, old_sql, call)
            now = _search(session, sql, call)
            assert len(now) >= len(today), (
                f"{len(now)} rows against today's {len(today)}:\n{_explain(session, sql, call)}"
            )
            assert len(now) == 20, _explain(session, sql, call)


# ---------------------------------------------------------------------------
# 2. Result quality
# ---------------------------------------------------------------------------


def test_the_partial_index_answers_only_from_its_own_knowledge_base_in_order(
    engine, schema, settings, query_vectors
):
    """Every row comes from this knowledge base, ranked by similarity.

    What the fixture cannot show is the benchmark's recall result. At this scale
    the indexed knowledge base is 75% of the table, so the shared index's
    post-filter throws almost nothing away and both indexes are equally
    approximate at ``hnsw.ef_search = 40`` (measured: 0.28 against 0.29
    recall@20, i.e. a wash). The quality gap the benchmark found belongs to the
    regime where one knowledge base is a small fraction of a large table, which
    would take a fixture two orders of magnitude bigger to reproduce. So what is
    pinned here is what does hold at every scale, and the completeness of the
    index itself is pinned by the next test.
    """
    name = _build_big_index(engine, settings)
    sql, params = _capture_search_sql(engine, KB_BIG, query_vectors[0])
    with Session(engine) as session:
        for vector in query_vectors:
            call = {**params, "embedding": _literal(vector)}
            plan = _explain(session, sql, call)
            assert f"Index Scan using {name}" in plan, plan
            rows = _search(session, sql, call)
            assert len(rows) == 20
            similarities = [float(r[2]) for r in rows]
            assert similarities == sorted(similarities, reverse=True), similarities

        ids = {str(r[0]) for r in rows}
        foreign = session.execute(
            text(
                f"SELECT count(*) FROM {SCHEMA}.chunks WHERE id = ANY(CAST(:ids AS uuid[])) "
                "AND knowledge_base_id <> CAST(:kb AS uuid)"
            ),
            {"ids": "{" + ",".join(ids) + "}", "kb": KB_BIG},
        ).scalar()
        session.rollback()
    assert foreign == 0


def test_the_partial_index_is_complete_and_only_the_search_is_approximate(
    engine, schema, settings, query_vectors
):
    """At a high ``ef_search`` the partial index is exactly an exact scan.

    Distinguishes an approximate *search* from a partial index that is missing
    rows: the latter could not be made exact by searching harder.
    """
    _build_big_index(engine, settings)
    sql, params = _capture_search_sql(engine, KB_BIG, query_vectors[0])
    with Session(engine) as session:
        for vector in query_vectors:
            call = {**params, "embedding": _literal(vector)}
            session.execute(text("SET LOCAL enable_indexscan = off"))
            exact = [str(r[0]) for r in session.execute(text(sql), call).all()]
            session.rollback()
            session.execute(text("SET LOCAL hnsw.ef_search = 1000"))
            got = [str(r[0]) for r in session.execute(text(sql), call).all()]
            session.rollback()
            assert got == exact


# ---------------------------------------------------------------------------
# 3. The prepared-statement trap
# ---------------------------------------------------------------------------


def test_the_real_code_path_keeps_the_index_past_the_prepare_threshold(
    engine, schema, settings, query_vectors
):
    """The finding that decides whether this ships.

    psycopg prepares a statement once it has seen it ``prepare_threshold`` (5)
    times on a connection, and from the sixth execution of the *prepared*
    statement PostgreSQL starts weighing its generic plan against the custom
    ones. With the knowledge base id bound as a parameter the generic plan
    cannot prove the index predicate, and the measured result was executions
    1-10 on the partial index at 1.1-1.8 ms and every execution from the 11th
    on a bitmap scan plus an exact sort at 54-72 ms.

    So this drives ``vector_search`` itself -- the real method, through
    SQLAlchemy and psycopg, on one connection -- 14 times, past both thresholds,
    and then checks three things: that the driver really did prepare the
    statement (otherwise the test proves nothing), that the plan the plan cache
    now holds for it still names the partial index, and that all 14 executions
    scanned the partial index and none the shared one.

    The searches deliberately share one transaction. psycopg discards its whole
    prepared-statement state when it sees a ROLLBACK, so a rollback between
    searches would reset the count before it ever reached the threshold -- which
    is also why this trap does not fire on every code path in the service, and
    no reason to leave it in place on the ones where it does.
    """
    name = _build_big_index(engine, settings)
    shared = f"idx_ai_embeddings_hnsw_{DIMS}"
    executions = 14

    before = _idx_scans(engine, name, shared)
    # An engine of its own, disposed before the counts are read: a backend
    # flushes its statistics at most once a second, or unconditionally when it
    # exits, so the searches must not be left in an idle pooled connection.
    probe = create_engine(_dsn())
    connection = probe.connect()
    try:
        with Session(bind=connection) as session:
            store = _ChunkStore(db_session=session, knowledge_base_id=KB_BIG, schema=SCHEMA)
            for i in range(executions):
                items = asyncio.run(
                    store.vector_search(
                        embedding=list(query_vectors[i % len(query_vectors)]), top_k=20
                    )
                )
                assert len(items) == 20

            prepared = session.execute(
                text("SELECT name, statement FROM pg_prepared_statements")
            ).all()
            searches = [row for row in prepared if "e.knowledge_base_id" in row[1]]
            assert searches, (
                "the driver never prepared the search statement, so this test could not "
                f"have seen a generic plan at all; prepared: {[r[0] for r in prepared]}"
            )
            # What the plan cache would use for the next execution -- generic or
            # custom, whichever it settled on over the 14.
            cached_plan = "\n".join(
                str(r[0])
                for r in session.execute(
                    text(
                        f"EXPLAIN EXECUTE {searches[0][0]} "
                        f"('{_literal(query_vectors[0])}', '{KB_BIG}'::uuid, {DIMS}, 20)"
                    )
                ).all()
            )
            # Statistics reach shared memory at transaction end, and EXPLAIN
            # plans nothing it does not execute, so the count below is the 14.
            session.commit()
    finally:
        connection.close()
        probe.dispose()

    after = _idx_scans(engine, name, shared)
    assert f"Index Scan using {name}" in cached_plan, (
        f"the cached plan no longer uses the partial index:\n{cached_plan}"
    )
    assert after[name] - before[name] == executions, (
        f"only {after[name] - before[name]} of {executions} executions used the partial "
        f"index; another plan took the rest (before {before}, after {after})"
    )
    assert after[shared] == before[shared], (
        f"no execution may fall back to the shared index (before {before}, after {after})"
    )


def _probe_sql(sql: str, embedding, *, bind_kb: bool) -> str:
    """The store's query with only the knowledge base id left as a parameter.

    Everything else becomes a literal, so ``PREPARE`` needs one parameter type
    and ``EXECUTE`` needs no protocol-level parameters at all. That isolates the
    one difference under test: whether the embeddings-side predicate names the
    knowledge base or a placeholder.
    """
    probe = (
        sql.replace(":embedding", f"'{_literal(embedding)}'")
        .replace(":dims", str(DIMS))
        .replace(":top_k", "20")
        .replace(":kb_id", "$1")
    )
    if bind_kb:
        probe = probe.replace(f"e.knowledge_base_id = '{KB_BIG}'", "e.knowledge_base_id = $1")
        assert "e.knowledge_base_id = $1" in probe
    return probe


def _generic_plan(session, statement: str, label: str) -> str:
    """The plan PostgreSQL builds for ``statement`` knowing none of its parameters.

    ``force_generic_plan`` is the same decision the planner makes on its own once
    a statement has been prepared and its generic plan costs no more than the
    custom ones. Forcing it removes the dependence on that cost comparison,
    which is fixture-specific, and leaves the structural question: can a plan
    built without the parameter's value prove the index predicate?
    """
    session.execute(text("SET LOCAL plan_cache_mode = 'force_generic_plan'"))
    session.execute(text(f"PREPARE {label} (uuid) AS {statement}"))
    plan = "\n".join(
        str(r[0])
        for r in session.execute(text(f"EXPLAIN EXECUTE {label} ('{KB_BIG}'::uuid)")).all()
    )
    session.rollback()
    return plan


def test_a_bound_kb_id_would_lose_the_index_in_a_generic_plan(
    engine, schema, settings, query_vectors
):
    """Why the id is a literal: a generic plan cannot prove the index predicate."""
    name = _build_big_index(engine, settings)
    sql, _ = _capture_search_sql(engine, KB_BIG, query_vectors[0])
    with Session(engine) as session:
        plan = _generic_plan(
            session, _probe_sql(sql, query_vectors[0], bind_kb=True), "bound_probe"
        )
    assert name not in plan, f"a generic plan must not be able to match the predicate:\n{plan}"


def test_the_literal_kb_id_keeps_the_index_in_a_generic_plan(
    engine, schema, settings, query_vectors
):
    """The other half: with the id as a literal the generic plan matches it too."""
    name = _build_big_index(engine, settings)
    sql, _ = _capture_search_sql(engine, KB_BIG, query_vectors[0])
    with Session(engine) as session:
        plan = _generic_plan(
            session, _probe_sql(sql, query_vectors[0], bind_kb=False), "literal_probe"
        )
    assert f"Index Scan using {name}" in plan, plan


# ---------------------------------------------------------------------------
# 4. Thresholds, hysteresis and dispatch
# ---------------------------------------------------------------------------


def test_nothing_is_built_for_a_knowledge_base_below_the_threshold(engine, schema, settings):
    outcome = pvi.ensure_per_kb_vector_index(KB_SMALL, engine=engine)
    assert outcome["built"] == [], outcome
    with engine.connect() as conn:
        assert pvi.existing_per_kb_indexes(conn, KB_SMALL) == {}


def test_an_index_is_built_at_or_above_the_threshold(engine, schema, settings):
    name = _build_big_index(engine, settings)
    with engine.connect() as conn:
        assert pvi.existing_per_kb_indexes(conn, KB_BIG) == {DIMS: True}
        indexdef = conn.execute(
            text("SELECT indexdef FROM pg_indexes WHERE schemaname = :s AND indexname = :n"),
            {"s": SCHEMA, "n": name},
        ).scalar()
    assert "USING hnsw" in indexdef
    assert f"(embedding)::vector({DIMS})" in indexdef
    assert "vector_cosine_ops" in indexdef
    assert KB_BIG in indexdef and f"dims = {DIMS}" in indexdef


def test_a_second_ensure_is_a_no_op(engine, schema, settings):
    _build_big_index(engine, settings)
    again = pvi.ensure_per_kb_vector_index(KB_BIG, engine=engine)
    assert again["built"] == [] and again["dropped"] == [], again


def test_the_index_is_kept_between_the_two_thresholds(engine, schema, settings, monkeypatch):
    """Hysteresis: a knowledge base that falls below the build threshold keeps its index."""
    _build_big_index(engine, settings)
    monkeypatch.setattr(
        pvi,
        "get_setting",
        lambda key: {
            "VECTOR_PER_KB_INDEX_MIN_ROWS": 20_000,
            "VECTOR_PER_KB_INDEX_DROP_ROWS": 5_000,
            "VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB": _MEM_MB,
        }[key],
    )
    outcome = pvi.ensure_per_kb_vector_index(KB_BIG, engine=engine)
    assert outcome["dropped"] == [], outcome
    with engine.connect() as conn:
        assert pvi.existing_per_kb_indexes(conn, KB_BIG) == {DIMS: True}
        assert pvi.index_action(conn, KB_BIG) is None


def test_the_index_is_dropped_below_the_drop_threshold(engine, schema, settings, monkeypatch):
    name = _build_big_index(engine, settings)
    monkeypatch.setattr(
        pvi,
        "get_setting",
        lambda key: {
            "VECTOR_PER_KB_INDEX_MIN_ROWS": 20_000,
            "VECTOR_PER_KB_INDEX_DROP_ROWS": 15_000,
            "VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB": _MEM_MB,
        }[key],
    )
    with engine.connect() as conn:
        assert pvi.index_action(conn, KB_BIG) == "drop"
    outcome = pvi.ensure_per_kb_vector_index(KB_BIG, engine=engine)
    assert outcome["dropped"] == [name], outcome
    with engine.connect() as conn:
        assert pvi.existing_per_kb_indexes(conn, KB_BIG) == {}


def test_index_action_is_the_gate_the_indexing_path_uses(engine, schema, settings):
    """A bounded row count, the one check that runs per source that finishes indexing."""
    with engine.connect() as conn:
        assert pvi.index_action(conn, KB_BIG) == "build"
        assert pvi.index_action(conn, KB_SMALL) is None
        # Bounded: it reads no further than the threshold, whatever the KB holds.
        cap = settings["VECTOR_PER_KB_INDEX_MIN_ROWS"] + 1
        assert pvi.bounded_row_count(conn, KB_BIG, DIMS, cap) == cap
        assert pvi.bounded_row_count(conn, KB_SMALL, DIMS, cap) == SMALL_ROWS
        assert pvi.candidate_dims(conn, KB_BIG, cap) == [DIMS]


def test_index_action_of_a_knowledge_base_with_no_embeddings_is_nothing(engine, schema, settings):
    with engine.connect() as conn:
        assert pvi.index_action(conn, str(uuid.uuid4())) is None


# ---------------------------------------------------------------------------
# 5. Repair, drop and the start-up sweep
# ---------------------------------------------------------------------------


def _invalidate(engine, name: str) -> None:
    """Make an index look like a failed CREATE INDEX CONCURRENTLY left it."""
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(
            text(
                "UPDATE pg_index SET indisvalid = false WHERE indexrelid = "
                "to_regclass(:relation)::oid"
            ),
            {"relation": f'"{SCHEMA}".{name}'},
        )


def test_an_invalid_index_is_dropped_and_rebuilt(engine, schema, settings):
    name = _build_big_index(engine, settings)
    _invalidate(engine, name)
    with engine.connect() as conn:
        assert pvi.existing_per_kb_indexes(conn, KB_BIG) == {DIMS: False}
        assert pvi.index_action(conn, KB_BIG) == "build"

    outcome = pvi.ensure_per_kb_vector_index(KB_BIG, engine=engine)
    assert outcome.get("repaired_invalid_indexes") == [name], outcome
    assert outcome["built"] == [name], outcome
    with engine.connect() as conn:
        assert pvi.existing_per_kb_indexes(conn, KB_BIG) == {DIMS: True}


def test_an_invalid_index_is_repaired_even_below_the_build_threshold(
    engine, schema, settings, monkeypatch
):
    """An index nothing can use is still maintained on every write; it must go."""
    name = _build_big_index(engine, settings)
    _invalidate(engine, name)
    monkeypatch.setattr(
        pvi,
        "get_setting",
        lambda key: {
            "VECTOR_PER_KB_INDEX_MIN_ROWS": 20_000,
            "VECTOR_PER_KB_INDEX_DROP_ROWS": 15_000,
            "VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB": _MEM_MB,
        }[key],
    )
    outcome = pvi.ensure_per_kb_vector_index(KB_BIG, engine=engine)
    assert outcome.get("repaired_invalid_indexes") == [name], outcome
    assert outcome["built"] == [], "below the threshold it must not be rebuilt"
    with engine.connect() as conn:
        assert pvi.existing_per_kb_indexes(conn, KB_BIG) == {}


def test_deleting_a_knowledge_base_drops_its_index(engine, schema, settings):
    name = _build_big_index(engine, settings)
    outcome = pvi.drop_per_kb_vector_indexes(KB_BIG, engine=engine)
    assert outcome == {"status": "dropped", "indexes": [name]}, outcome
    with engine.connect() as conn:
        assert pvi.existing_per_kb_indexes(conn, KB_BIG) == {}


def test_dropping_a_knowledge_base_that_never_had_an_index_is_a_no_op(engine, schema, settings):
    outcome = pvi.drop_per_kb_vector_indexes(KB_SMALL, engine=engine)
    assert outcome == {"status": "dropped", "indexes": []}, outcome


def test_the_start_up_sweep_finds_a_knowledge_base_that_needs_one(engine, schema, settings):
    found = pvi.kbs_needing_a_per_kb_index(engine)
    assert KB_BIG in found
    assert KB_SMALL not in found


def test_the_start_up_sweep_finds_an_invalid_index(engine, schema, settings):
    name = _build_big_index(engine, settings)
    _invalidate(engine, name)
    assert KB_BIG in pvi.kbs_needing_a_per_kb_index(engine)


def test_the_start_up_sweep_is_quiet_once_everything_is_reconciled(engine, schema, settings):
    _build_big_index(engine, settings)
    assert pvi.kbs_needing_a_per_kb_index(engine) == []


def test_the_start_up_sweep_survives_a_database_with_no_settings_table(engine, schema):
    """It runs inside the boot's migration transaction, possibly before that table exists.

    Deliberately without the ``settings`` fixture, so the sweep's own settings
    read runs for real against a schema that has no ``project_settings``. It
    must fall back to the registry defaults, leave its connection usable, and
    not raise -- a settings read through ``db.session`` would instead leave the
    boot's transaction aborted and fail the migrations that follow it.
    """
    with engine.connect() as conn:
        assert pvi.read_overrides(conn, *pvi._THRESHOLD_KEYS) == {}
        assert conn.execute(text("SELECT 1")).scalar() == 1
        conn.rollback()
    # 9,000 rows, and the registry default build threshold is 50,000.
    assert pvi.kbs_needing_a_per_kb_index(engine) == []


def test_a_build_held_by_another_caller_is_reported_not_duplicated(engine, schema, settings):
    """Two ensures for one index must not drop each other's in-flight build."""
    lock = pvi.index_lock_relation(KB_BIG, DIMS)
    holder = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        assert (
            holder.execute(text(pvi.partition_build_lock_sql()), {"relation": lock}).scalar()
            is True
        )
        outcome = pvi.ensure_per_kb_vector_index(KB_BIG, engine=engine)
        assert outcome["status"] == "building", outcome
        assert outcome["built"] == []
        with engine.connect() as conn:
            assert pvi.existing_per_kb_indexes(conn, KB_BIG) == {}
    finally:
        holder.execute(text(pvi.partition_build_unlock_sql()), {"relation": lock})
        holder.close()


def test_the_build_leaves_no_session_settings_behind(engine, schema, settings):
    """maintenance_work_mem and statement_timeout must not ride a pooled connection out.

    The build raises both on its own session, because CREATE INDEX CONCURRENTLY
    cannot run in a transaction and so SET LOCAL would do nothing.
    """
    _build_big_index(engine, settings)
    # The build's own connection is back in the pool by now, so this may well be
    # it. reset_val is what the session would hold with nothing set on it.
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT name, setting, reset_val FROM pg_settings "
                "WHERE name IN ('maintenance_work_mem', 'statement_timeout')"
            )
        ).all()
    assert len(rows) == 2, rows
    for name, setting, reset_val in rows:
        assert setting == reset_val, (name, setting, reset_val)


def test_a_concurrent_build_does_not_block_writes(engine, schema, settings):
    """CREATE INDEX CONCURRENTLY must not stall writes to ai.embeddings.

    Measured rather than asserted from the manual: single-row inserts into the
    same knowledge base run throughout the build, and none of them may wait for
    anything like the build's duration.
    """
    import threading

    stop = threading.Event()
    latencies: list[float] = []
    failures: list[BaseException] = []

    def writer():
        try:
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                while not stop.is_set():
                    started = time.monotonic()
                    conn.execute(
                        text(
                            f"INSERT INTO {SCHEMA}.embeddings (item_id, item_table, "
                            "knowledge_base_id, source_id, embedding_model, dims, embedding) "
                            "VALUES (gen_random_uuid(), 'chunks', CAST(:kb AS uuid), "
                            "CAST(:src AS uuid), :model, :dims, CAST(:v AS vector))"
                        ),
                        {
                            "kb": KB_BIG,
                            "src": SOURCE,
                            "model": f"probe-{uuid.uuid4()}",
                            "dims": DIMS,
                            "v": _literal(np.zeros(DIMS)),
                        },
                    )
                    latencies.append(time.monotonic() - started)
                    time.sleep(0.05)
        except BaseException as exc:  # pragma: no cover - reported by the assertions
            failures.append(exc)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        pvi.ensure_per_kb_vector_index(KB_BIG, engine=engine)
        build_seconds = time.monotonic() - started
    finally:
        stop.set()
        thread.join(timeout=30)

    assert not failures, failures
    assert latencies, "the writer never got to run"
    assert max(latencies) < max(1.0, build_seconds / 2), (
        f"a write waited {max(latencies):.2f} s during a {build_seconds:.2f} s build"
    )
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(
            text(
                f"DELETE FROM {SCHEMA}.embeddings WHERE embedding_model LIKE 'probe-%' "
                "AND knowledge_base_id = CAST(:kb AS uuid)"
            ),
            {"kb": KB_BIG},
        )
