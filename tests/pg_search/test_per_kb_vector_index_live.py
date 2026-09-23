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
schema alone; nothing here needs ``pg_search`` itself.

The regime the fixture represents
--------------------------------
60 knowledge bases in one ``embeddings`` table of ~40,000 rows, with the
indexed one at 30% of it and the others from 21% down to under 1%. That is the
shape this feature exists for -- one knowledge base is a *fraction* of a shared
table, and a much larger fraction than the average one -- and it is deliberately
not the shape an earlier version of this module used (2 knowledge bases, the
target at 75%). The difference is not cosmetic. Three things are true here and
false at 75%:

- the planner's estimate for a *bound* ``knowledge_base_id`` is 1/n_distinct,
  so 1.7% here against 50% there -- the regime a prepared statement's generic
  plan actually meets in production, and far enough below the indexed knowledge
  base's real 30% to price its index out (see ``FILLER_KBS``);
- a knowledge base below the build threshold is small in absolute terms as well
  as relative, which is the population the threshold decides for;
- the shared per-dimension index post-filters away 70-99% of what it returns,
  so recall through it is genuinely poor rather than a wash.

Sizes are chosen against the planner, not for roundness: the partial index has
to be the cheapest plan for the indexed knowledge base, or the plan tests would
be asserting a preference the planner does not have. Measured on this fixture,
the partial index is chosen at 30% and 21% selectivity and declined at 5% and
1% (where an exact scan really is cheaper, and exact).

Why 384 dimensions, and what that leaves unpinned
-------------------------------------------------
``dims`` is 384 -- a real embedding width, and the one that keeps the fixture at
~15 s. It is *not* the width most embedding models here produce, and the
difference matters more than a fixture's usually does.

At 384 the vector is stored in line, the exact scan and the ordered index scan
cost about the same, and the planner takes the index on its own. That is what
makes this width the right one for most of the module: the specs below about
*matchability* -- that the embeddings-side predicate is what lets the index be
used at all, that a prepared statement's generic plan can prove the index's
predicate only when all of the KB id, ``dims`` and the LIMIT are literals -- can
only be asserted where the planner has a preference to express.

At 1536 and up the vector is stored out of line, which leaves a small heap and a
one-tuple-per-page HNSW index, and PostgreSQL prices detoasting at nothing. The
two plans then cost within a small factor of each other, and which one wins is
settled by a join row estimate that is wrong by orders of magnitude -- so it
flips with the size of the table rather than tracking anything real. Measured
across all-1536 fixtures of one shape: the index was declined at 6,000 and
12,000 rows, chosen at 20,000, and declined again at 40,000. On the 40,000-row
one it stayed declined as the knowledge base's share was raised from 21% to 70%.

So at production widths the feature cannot be left to the planner, and
``BasePgVectorStore._preferring_this_kbs_partial_index`` prices the exact sort
out of the search when the knowledge base has a valid partial index to fall on.
Its docstring carries the 1536 measurements. **This module cannot reproduce them
cheaply.** A 1536-dimension knowledge base added to this fixture does not
reproduce them at all, and that is worth knowing: the 384 rows inflate the
embeddings heap, which is one of the two things that make the exact scan look
cheap, so the planner chooses the index there and the spec would assert a
preference the defect does not have. A faithful reproduction needs an
``embeddings`` table that is *mostly* 1536, and pinning it needs a second
module-scoped fixture and a row count on the lucky side of a coin flip. What is
pinned instead, deterministically, is that the store asks for the setting:
``tests/unit/test_vector_search_plan_shape.py``.

What this module does show about the fix is the part that is stable at any
width: that the setting is asked for only when there is an index to use, that
the answer for a knowledge base without one does not change, and that a filtered
search reaches the index -- which at 384 it does not do without the fix, because
a jsonb estimate is wrong the same way at every dimension.

Every test works in a scratch schema of its own, shaped like ``ai.chunks`` and
``ai.embeddings``, and the module fixture costs about 15 s, once.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import threading
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
DIMS = 384

# Five knowledge bases the tests name, and fifty-five more that exist only to
# put n_distinct(knowledge_base_id) at 60. The row counts are the selectivities
# the feature has to work at, as a share of the whole table (see the module
# docstring): 30, 21, 5, 2 and 1 per cent.
#
# Sixty rather than twenty, and it is load-bearing rather than tidy. A generic
# plan prices a *bound* knowledge base id at 1/n_distinct of the table, so how
# far that estimate is from the indexed knowledge base's real share is set by
# n_distinct and by that share together -- measured, the ordered index scan
# survives a bound item-side id up to about share x n_distinct = 12 and is
# priced out from about 18. At 20 knowledge bases and 30% this fixture sat at 6,
# on the safe side of that edge, so the shape the fix is *for* -- one large
# knowledge base among many average ones -- was the one shape the fixture did
# not represent, and a spec for it would have asserted a regression the fixture
# could not produce. At 60 it sits at 18. The filler rows shrink to keep the
# table at ~40,000 and every named share unchanged, so nothing else in the
# module moves.
KB_BIG = "9f8b1c2e-0000-4000-8000-000000000001"
KB_SMALL = "9f8b1c2e-0000-4000-8000-000000000002"
KB_MED = "9f8b1c2e-0000-4000-8000-000000000003"
KB_MID = "9f8b1c2e-0000-4000-8000-000000000004"
KB_THIN = "9f8b1c2e-0000-4000-8000-000000000005"
FILLER_KBS = [f"9f8b1c2e-0000-4000-8000-0000000000{10 + i:02d}" for i in range(55)]
SOURCE = "9f8b1c2e-0000-4000-8000-0000000000aa"

BIG_ROWS = 12_000
SMALL_ROWS = 800
MED_ROWS = 8_400
MID_ROWS = 2_000
THIN_ROWS = 400
FILLER_ROWS = 298

ROW_COUNTS: list[tuple[str, int]] = [
    (KB_BIG, BIG_ROWS),
    (KB_MED, MED_ROWS),
    (KB_MID, MID_ROWS),
    (KB_SMALL, SMALL_ROWS),
    (KB_THIN, THIN_ROWS),
    *((kb, FILLER_ROWS) for kb in FILLER_KBS),
]
TOTAL_ROWS = sum(rows for _, rows in ROW_COUNTS)

# Metadata the filter specs filter on. Every fifth chunk is "gold" (a 20%-
# selective filter inside a knowledge base) and every chunk carries its own
# knowledge base's tag (a 100%-selective one), so a filtered search can be
# measured without the filter's selectivity being the variable under test.
GOLD_EVERY = 5

# Any valid vector will do for a plan probe -- EXPLAIN prices the scan, it does
# not care where in the space the query sits.
_ZERO_VECTOR = "[" + ",".join(["0"] * DIMS) + "]"

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
    """The two tables the vector path joins, in the regime the docstring describes."""
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
        # Every index build in this module is serial. A 384-dimension vector is
        # stored in line, so the heap is big enough for PostgreSQL to want
        # parallel workers for an index build, and a parallel HNSW build asks
        # for a shared memory segment the size of maintenance_work_mem -- more
        # than the 64 MiB of /dev/shm a stock container has. The reloption is
        # the narrowest place to say "not on this table".
        conn.execute(f"ALTER TABLE {SCHEMA}.embeddings SET (parallel_workers = 0)")

        rng = np.random.default_rng(4242)
        for kb_id, rows in ROW_COUNTS:
            vectors = _vectors(rng, rows)
            chunks = io.StringIO()
            embeddings = io.StringIO()
            for i in range(rows):
                item_id = str(uuid.uuid4())
                meta = json.dumps(
                    {"tier": "gold" if i % GOLD_EVERY == 0 else "silver", "kb": kb_id}
                )
                chunks.write(f"{item_id}\t{kb_id}\t{SOURCE}\tpassage {i}\t{meta}\n")
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
        # The one build in the module that is worth giving more memory than the
        # service's own setting: it covers every row in the table.
        conn.execute("SET maintenance_work_mem = '512MB'")
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
    # 10,000 / 5,000 are the registry's intended defaults, so these tests run
    # the thresholds production runs. On this fixture only KB_BIG (12,000 rows)
    # is at or above the build threshold; KB_MED, at 8,400, is the knowledge
    # base those defaults leave without an index, which is what the regression
    # specs measure.
    values = {
        "VECTOR_PER_KB_INDEX_MIN_ROWS": 10_000,
        "VECTOR_PER_KB_INDEX_DROP_ROWS": 5_000,
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


def _capture_search_sql(engine, kb_id, embedding, **kwargs):
    """The SQL the real store issues, and the parameters it binds."""
    with Session(engine) as session:
        recorder = _RecordingSession(session)
        store = _ChunkStore(db_session=recorder, knowledge_base_id=kb_id, schema=SCHEMA)
        asyncio.run(store.vector_search(embedding=list(embedding), top_k=20, **kwargs))
        session.rollback()
    searches = [pair for pair in recorder.statements if "ORDER BY" in pair[0]]
    assert searches, f"the store issued no search query: {recorder.statements}"
    sql, params = searches[0]
    # Asserted here, where the statement is taken, rather than in each spec that
    # uses it: the knowledge base id reaches PostgreSQL as a literal on *both*
    # sides of the join and nothing binds it. This is the property every plan
    # spec below rests on, and it has to be checked positively -- an earlier
    # version of ``_probe_sql`` rewrote ``:kb_id`` into ``$1`` unconditionally,
    # so putting either id back on a parameter left the probes rewriting the
    # regression into the shape under test and the whole module green.
    assert ":kb_id" not in sql, (
        "the store bound the knowledge base id; a generic plan cannot prove the "
        f"partial index's predicate from a parameter:\n{sql}"
    )
    assert f"c.knowledge_base_id = '{kb_id}'" in sql, (
        f"the item-side knowledge base id must be a literal:\n{sql}"
    )
    assert f"e.knowledge_base_id = '{kb_id}'" in sql, (
        f"the embeddings-side knowledge base id must be a literal:\n{sql}"
    )
    return sql, params


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


def _execute_args(session, statement_name: str, vector) -> str:
    """Literal ``EXECUTE`` arguments matching a prepared statement's parameters.

    ``vector_search`` interpolates some values and binds others, and which is
    which is the subject of this whole module -- so the argument list is built
    from ``pg_prepared_statements.parameter_types`` rather than written out.
    Every parameter the statement has is either the embedding (text, cast in
    the statement itself) or the knowledge base id, and both are known here.

    The arguments are literals with casts, not bound parameters: an ``EXECUTE``
    whose own arguments arrive through the extended protocol cannot have their
    types inferred (``could not determine data type of parameter $1``).
    """
    # No rollback here, deliberately: psycopg throws away its whole
    # prepared-statement state when it sees one, which would take the statement
    # this is building arguments for with it.
    types = session.execute(
        text("SELECT parameter_types::text[] FROM pg_prepared_statements WHERE name = :n"),
        {"n": statement_name},
    ).scalar()
    args = []
    for pg_type in types or []:
        if pg_type == "uuid":
            args.append(f"'{KB_BIG}'::uuid")
        else:
            args.append(f"'{_literal(vector)}'::{pg_type}")
    return ", ".join(args)


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

    An earlier version of this docstring explained that the fixture could not
    show the benchmark's recall gap because the indexed knowledge base was 75%
    of the table, so the shared index's post-filter threw almost nothing away.
    The fixture is now in the regime where that post-filter does discard most of
    what it returns, and recall was measured rather than reasoned about: at 30%
    of the table the two indexes come out the same (0.41 recall at 20 either
    way), and at 21% the partial index is better (0.39 against 0.18). Neither
    number is asserted anywhere, on purpose -- absolute recall on a synthetic
    fixture is an artifact of how the vectors were generated, and a spec built
    on one would be pinning the generator. What is asserted is the part that
    does not depend on it: exactness, in
    ``test_the_new_shape_answers_exactly_where_the_old_one_was_approximate``.

    What stays here is the invariant that holds at every scale -- own knowledge
    base only, in similarity order -- with the index's completeness pinned by
    the next test.
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
            # The arguments are read off the prepared statement rather than
            # written out here: how many parameters production's statement
            # still has is exactly the thing this feature keeps changing, and a
            # hard-coded list turns that into an "Expected N parameters" error
            # instead of a result.
            cached_plan = "\n".join(
                str(r[0])
                for r in session.execute(
                    text(
                        f"EXPLAIN EXECUTE {searches[0][0]} "
                        f"({_execute_args(session, searches[0][0], query_vectors[0])})"
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


def _probe_sql(sql: str, embedding, *, bind: str) -> str:
    """The store's real query, prepared, with one value moved back to a parameter.

    Nothing is rewritten except the one thing under test. ``vector_search``
    already emits ``dims`` and the ``LIMIT`` as literals and the knowledge base
    id and embedding as parameters, so the "none" case below *is* production's
    statement — which the earlier version of this helper was not: it inlined
    ``:dims`` and ``:top_k`` itself and so pinned a shape the service never
    sends.

    ``bind`` names what to take back out of the SQL and hand to the planner as
    an unknown: ``"kb"`` (the embeddings side), ``"kb_chunks"`` (the item side),
    ``"dims"``, ``"limit"``, or ``"none"``.

    Only ``:embedding`` is rewritten unconditionally, because ``PREPARE`` speaks
    ``$n`` and that value is a parameter in production too. Nothing else is:
    this helper used to append ``.replace(":kb_id", "$1")``, which was a no-op
    against the fixed statement and an escape hatch against a broken one -- a
    store that bound the knowledge base id again would have had the bind
    rewritten back into a literal here, and every spec in the module would have
    kept passing. ``_capture_search_sql`` now asserts the literal instead, and
    the bound shapes are cases in the matrix below rather than accidents.
    """
    assert ":kb_id" not in sql, f"nothing in this helper may rewrite a bind away:\n{sql}"
    probe = sql.replace(":embedding", "$2")
    if bind == "kb":
        probe = probe.replace(f"e.knowledge_base_id = '{KB_BIG}'", "e.knowledge_base_id = $1")
        assert "e.knowledge_base_id = $1" in probe
    elif bind == "kb_chunks":
        probe = probe.replace(f"c.knowledge_base_id = '{KB_BIG}'", "c.knowledge_base_id = $1")
        assert "c.knowledge_base_id = $1" in probe
    elif bind == "dims":
        probe = probe.replace(f"e.dims = {DIMS}", "e.dims = $3::int")
        assert "e.dims = $3::int" in probe
    elif bind == "limit":
        probe = probe.replace("LIMIT 20", "LIMIT $3::int")
        assert "LIMIT $3::int" in probe
    else:
        assert bind == "none", bind
    return probe


def _generic_plan(session, statement: str, label: str, *, extra_arg: str | None = None) -> str:
    """The plan PostgreSQL builds for ``statement`` knowing none of its parameters.

    ``force_generic_plan`` is the decision the planner makes on its own once a
    statement has been prepared and its generic plan costs no more than the
    custom ones. Forcing it removes the dependence on that cost comparison,
    which is fixture-specific, and leaves the structural question: can a plan
    built without the parameters' values prove the index predicate?

    The ``EXECUTE`` arguments are literals with casts rather than bound
    parameters: an ``EXECUTE`` whose own arguments arrive through the extended
    protocol cannot have their types inferred (``could not determine data type
    of parameter $1``).
    """
    types = "uuid, text" + (", int" if extra_arg else "")
    args = f"'{KB_BIG}'::uuid, '{_ZERO_VECTOR}'::text" + (f", {extra_arg}" if extra_arg else "")
    session.execute(text("SET LOCAL plan_cache_mode = 'force_generic_plan'"))
    session.execute(text(f"PREPARE {label} ({types}) AS {statement}"))
    plan = "\n".join(
        str(r[0]) for r in session.execute(text(f"EXPLAIN EXECUTE {label} ({args})")).all()
    )
    session.rollback()
    return plan


def test_the_query_the_service_emits_keeps_the_index_in_a_generic_plan(
    engine, schema, settings, query_vectors
):
    """The central claim, on production's own statement.

    A prepared statement whose generic plan is chosen must still reach the
    partial index. This is the case the previous version of this test got wrong:
    it inlined ``dims`` and the ``LIMIT`` in the probe, so it passed on a query
    the service did not send while the real one fell back to an exact sort.
    """
    name = _build_big_index(engine, settings)
    sql, _ = _capture_search_sql(engine, KB_BIG, query_vectors[0])
    assert ":dims" not in sql and ":top_k" not in sql, (
        f"the service must emit dims and the limit as literals:\n{sql}"
    )
    with Session(engine) as session:
        plan = _generic_plan(session, _probe_sql(sql, query_vectors[0], bind="none"), "real_probe")
    assert f"Index Scan using {name}" in plan, plan


@pytest.mark.parametrize(
    "bind,extra_arg",
    [("kb", None), ("kb_chunks", None), ("dims", str(DIMS)), ("limit", "20")],
)
def test_binding_any_one_of_the_four_loses_the_index_in_a_generic_plan(
    engine, schema, settings, query_vectors, bind, extra_arg
):
    """Why all four values are literals, measured one at a time.

    The embeddings-side knowledge base id and ``dims`` are both in the index
    predicate, so a plan that cannot prove either cannot use the index; an
    unknown ``LIMIT`` makes the planner assume it will be asked for a large
    fraction of the rows, which prices the ordered index scan out. Any one of
    them left bound is enough to lose it -- which is what makes this a four-way
    requirement rather than the one-way one the first version of this PR
    claimed.

    ``kb_chunks`` -- the *item*-side id, which is in no index predicate at all --
    is the fourth, and it is here as a case because it used to be rewritten away:
    it is not matchability that a bound item-side id costs but the row estimate
    behind the cost comparison. A generic plan prices ``c.knowledge_base_id =
    $1`` at ``1/n_distinct``, so on a table of many knowledge bases it expects a
    fraction of the rows this one really has, which inflates the ordered index
    scan and deflates the sort. Both push the same way and the index goes. It is
    conditional on the fixture in a way none of the other three are: the effect
    needs the indexed knowledge base to be several times the average one --
    roughly ``share x n_distinct`` above 12-18 -- which is the population a
    per-knowledge-base index exists for and the regime this module's fixture is
    now built in (30% of 60 knowledge bases). At ``n_distinct`` 20 the same bind
    keeps the index, which is why the fixture's filler count is load-bearing and
    says so.
    """
    name = _build_big_index(engine, settings)
    sql, _ = _capture_search_sql(engine, KB_BIG, query_vectors[0])
    with Session(engine) as session:
        plan = _generic_plan(
            session,
            _probe_sql(sql, query_vectors[0], bind=bind),
            f"bound_{bind}_probe",
            extra_arg=extra_arg,
        )
    assert name not in plan, f"binding {bind} must lose the partial index:\n{plan}"


def test_the_custom_plan_reaches_the_index_whatever_is_bound(
    engine, schema, settings, query_vectors
):
    """The fallback is correct, not wrong: a custom plan always finds the index.

    So the failure mode the three literals avoid is latency, never a wrong
    answer -- the generic plan's alternative is an exact sort.
    """
    name = _build_big_index(engine, settings)
    sql, _ = _capture_search_sql(engine, KB_BIG, query_vectors[0])
    with Session(engine) as session:
        session.execute(text("SET LOCAL plan_cache_mode = 'force_custom_plan'"))
        session.execute(
            text(f"PREPARE custom_probe (uuid, text) AS {_probe_sql(sql, None, bind='kb')}")
        )
        plan = "\n".join(
            str(r[0])
            for r in session.execute(
                text(f"EXPLAIN EXECUTE custom_probe ('{KB_BIG}'::uuid, '{_ZERO_VECTOR}'::text)")
            ).all()
        )
        session.rollback()
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


def test_a_declined_repair_reports_building_and_never_claims_it_built(
    engine, schema, settings, monkeypatch
):
    """The one case that must not report success.

    When a build of this index is already running, the INVALID entry has to stay:
    dropping it would pull the ground out from under that build. But the caller
    must then stop, because ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` no-ops
    against the name the invalid index still holds -- so falling through would
    leave an index that answers no query, is maintained on every insert, and was
    reported as built, with nothing coming back to it until the next source
    finishes indexing or the pod restarts.

    The on-disk post-condition is what this asserts: the index is still INVALID
    afterwards, and its name appears in neither ``built`` nor
    ``repaired_invalid_indexes``.
    """
    name = _build_big_index(engine, settings)
    _invalidate(engine, name)
    monkeypatch.setattr(pvi, "_build_in_progress", lambda conn, kb_id, dims: True)

    outcome = pvi.ensure_per_kb_vector_index(KB_BIG, engine=engine)

    assert outcome["status"] == "building", outcome
    assert outcome["built"] == [], outcome
    assert outcome["dropped"] == [], outcome
    assert "repaired_invalid_indexes" not in outcome, outcome
    with engine.connect() as conn:
        assert pvi.existing_per_kb_indexes(conn, KB_BIG) == {DIMS: False}, (
            "the invalid index must be left in place for the running build"
        )


def test_a_build_of_another_knowledge_bases_index_does_not_block_this_repair(
    engine, schema, settings
):
    """``_build_in_progress`` is scoped to this index, not to ai.embeddings.

    Every one of these indexes lives on the one shared table, so a check scoped
    to the relation would report *any* concurrent build on it -- and the start-up
    sweep dispatches every out-of-step knowledge base at once, so with two large
    ones the builds overlap by construction, at exactly the boot meant to clear
    an INVALID index.
    """
    _build_big_index(engine, settings)
    ready = threading.Event()
    done = threading.Event()
    failures: list[BaseException] = []

    # A reader with an open transaction: CREATE INDEX CONCURRENTLY waits for it
    # before it starts building, and its progress row -- with index_relid already
    # set -- is visible for the whole wait. That is what makes this deterministic
    # rather than a race against a build that might finish first.
    holder = engine.connect()
    holder.execute(text(f"SELECT count(*) FROM {SCHEMA}.embeddings"))

    def build_the_other_kbs_index():
        try:
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text(pvi.per_kb_index_ddl(KB_SMALL, DIMS)))
        except BaseException as exc:  # pragma: no cover - surfaced by the assertions
            failures.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=build_the_other_kbs_index, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        with engine.connect() as conn:
            while time.monotonic() < deadline:
                conn.execute(text("SELECT pg_stat_clear_snapshot()"))
                if pvi._build_in_progress(conn, KB_SMALL, DIMS):
                    ready.set()
                    # The whole point: the same probe, for the knowledge base
                    # whose index is NOT being built, must be False -- even
                    # though the build is on the same relation.
                    assert pvi._build_in_progress(conn, KB_BIG, DIMS) is False, (
                        "another knowledge base's build must not look like this one's"
                    )
                    on_relation = conn.execute(
                        text(
                            "SELECT count(*) FROM pg_stat_progress_create_index "
                            "WHERE relid = to_regclass(:rel)"
                        ),
                        {"rel": f"{SCHEMA}.embeddings"},
                    ).scalar()
                    conn.rollback()
                    assert on_relation >= 1, (
                        "a relation-scoped check would have seen this build and blocked "
                        "the other knowledge base's repair"
                    )
                    break
                conn.rollback()
                time.sleep(0.02)
    finally:
        holder.rollback()
        holder.close()
        done.wait(timeout=60)
        thread.join(timeout=60)
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(pvi.per_kb_index_drop_ddl(KB_SMALL, DIMS)))

    assert not failures, failures
    assert ready.is_set(), "never observed the other knowledge base's build in progress"


def test_the_index_cap_stops_a_build_and_says_so(engine, schema, settings, monkeypatch):
    """The cap is a stated safety property, so it needs a test that fails without it.

    The planner opens and locks every index of a relation while planning any
    query on it, so unbounded per-knowledge-base indexes on ``ai.embeddings``
    degrade every query on the table and eventually fail them outright with
    ``out of shared memory``. Asserted on disk: no index exists afterwards.
    """
    monkeypatch.setattr(pvi, "per_kb_index_count", lambda conn: pvi.MAX_PER_KB_INDEXES)

    outcome = pvi.ensure_per_kb_vector_index(KB_BIG, engine=engine)

    assert outcome["status"] == "skipped", outcome
    assert outcome["reason"] == "index_cap_reached", outcome
    assert outcome["index_count"] == pvi.MAX_PER_KB_INDEXES, outcome
    assert outcome["built"] == [], outcome
    with engine.connect() as conn:
        assert pvi.existing_per_kb_indexes(conn, KB_BIG) == {}, (
            "the cap must be enforced before the index is created, not after"
        )


def test_the_cap_does_not_stop_a_build_one_below_it(engine, schema, settings, monkeypatch):
    """The other side, so the test above cannot pass by never building at all."""
    monkeypatch.setattr(pvi, "per_kb_index_count", lambda conn: pvi.MAX_PER_KB_INDEXES - 1)
    outcome = pvi.ensure_per_kb_vector_index(KB_BIG, engine=engine)
    assert outcome["built"] == [pvi.per_kb_index_name(KB_BIG, DIMS)], outcome
    with engine.connect() as conn:
        assert pvi.existing_per_kb_indexes(conn, KB_BIG) == {DIMS: True}


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
    # What the sweep then finds follows from the registry's own default, which
    # this PR's successors are expected to move: the assertion is derived from
    # it rather than from a number copied out of the registry, so lowering the
    # default changes this test's expectation instead of breaking it.
    build_at, _ = pvi.thresholds()
    expected = [KB_BIG] if BIG_ROWS >= build_at else []
    assert pvi.kbs_needing_a_per_kb_index(engine) == expected, (
        f"build threshold {build_at}, KB_BIG has {BIG_ROWS} rows"
    )


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


# ---------------------------------------------------------------------------
# 7. The generic plan, through the real driver
#
# Everything in section 3 above reaches the generic plan through a hand-written
# ``PREPARE``. That is the right tool for asking which of three values the
# planner needs, because it can take them out one at a time -- but it is not
# production's path, and there is one thing it cannot see: ``EXPLAIN`` on an
# un-prepared statement still has the parameter values in hand, so it reports
# the plan the planner *would* build knowing them, even under
# ``force_generic_plan``. Measured on this fixture, a filtered search EXPLAINs
# as using the partial index under ``force_generic_plan`` and then does not use
# it when psycopg actually prepares it. So these specs drive ``vector_search``
# itself, on one pooled connection, past psycopg's prepare threshold, and read
# the answer out of the index's own scan counters.
# ---------------------------------------------------------------------------

# Above psycopg's ``prepare_threshold`` (5) with room to spare, so the driver
# has prepared the statement and PostgreSQL has had several executions to settle
# on a plan for it.
_DRIVEN_EXECUTIONS = 12


def _drive_searches(engine, kb_id, vectors, *, plan_cache_mode, index_name, **kwargs):
    """Run ``vector_search`` for real, N times, on one connection.

    Returns ``(scans, prepared)``: how many scans each index served over the
    run, and the names of the prepared statements the driver ended up with.

    An engine of its own, disposed before the counters are read: a backend
    flushes its statistics at most once a second, or unconditionally when it
    exits, so the searches must not be left sitting in an idle pooled
    connection. The searches share one transaction because psycopg discards its
    prepared-statement state when it sees a ROLLBACK.
    """
    shared = f"idx_ai_embeddings_hnsw_{DIMS}"
    before = _idx_scans(engine, index_name, shared)
    probe = create_engine(_dsn())
    connection = probe.connect()
    prepared: list[str] = []
    try:
        with Session(bind=connection) as session:
            session.execute(text(f"SET plan_cache_mode = '{plan_cache_mode}'"))
            store = _ChunkStore(db_session=session, knowledge_base_id=kb_id, schema=SCHEMA)
            for i in range(_DRIVEN_EXECUTIONS):
                items = asyncio.run(
                    store.vector_search(
                        embedding=list(vectors[i % len(vectors)]), top_k=20, **kwargs
                    )
                )
                assert items, f"execution {i} returned nothing"
            prepared = [
                row[0]
                for row in session.execute(
                    text("SELECT name, statement FROM pg_prepared_statements")
                ).all()
                if "e.knowledge_base_id" in row[1]
            ]
            session.commit()
    finally:
        connection.close()
        probe.dispose()
    after = _idx_scans(engine, index_name, shared)
    return {name: after[name] - before[name] for name in after}, prepared


def test_the_real_code_path_keeps_the_index_under_a_forced_generic_plan(
    engine, schema, settings, query_vectors
):
    """Production's exact statement, prepared, planned without its parameters.

    ``force_generic_plan`` is the decision PostgreSQL makes on its own once a
    statement has been prepared and its generic plan costs no more than the
    custom ones; forcing it removes the dependence on that cost comparison,
    which is fixture-specific, and leaves the structural question. The negative
    control is the ``prepared`` assertion: without it a run in which psycopg
    never prepared anything would pass while proving nothing.

    Measured on this fixture, unfiltered: 12 of 12 executions on the partial
    index at 1.4-1.6 ms, generic and custom alike.
    """
    name = _build_big_index(engine, settings)
    scans, prepared = _drive_searches(
        engine, KB_BIG, query_vectors, plan_cache_mode="force_generic_plan", index_name=name
    )
    assert prepared, (
        "the driver never prepared the search statement, so this test could not "
        "have seen a generic plan at all"
    )
    assert scans[name] == _DRIVEN_EXECUTIONS, (
        f"only {scans[name]} of {_DRIVEN_EXECUTIONS} executions used the partial index "
        f"under a generic plan; another plan took the rest ({scans})"
    )
    assert scans[f"idx_ai_embeddings_hnsw_{DIMS}"] == 0, (
        f"no execution may fall back to the shared index ({scans})"
    )


# ---------------------------------------------------------------------------
# 8. A filtered search
#
# ``filter_metadata`` is a documented, shipped parameter on the knowledge-base
# search route, passed through unchanged by the search service and by the
# runtime and attached-knowledge-base configuration paths. It adds a fourth
# bound value to the statement -- ``c.meta @> $n`` -- and it is the one value the
# planner cannot be given as a literal, because it is caller data.
#
# Two different estimates go wrong, so two settings are needed. A *generic* plan
# has no value to estimate at all and falls back to a fixed guess, which is what
# ``_force_custom_plan`` is for. A *custom* plan does better, but not well: it
# estimates ``meta @> const`` by testing the constant against the column's MCV
# list, so the more keys the filter carries the fewer MCV entries it matches, and
# it then multiplies that by ``c.knowledge_base_id = ...`` as if the two were
# independent -- which they are not, when one of the filter's own keys is the
# knowledge base. Measured on this fixture, ``top_k`` 20:
#
#   filter                          estimate   rows that really match
#   {"tier": "gold"}                     722                    2,400
#   {"kb": KB_BIG}                     1,076                   12,000
#   {"tier": "gold", "kb": KB_BIG}       217                    2,400
#
# The first two keep the ordered index scan; the third, 11 times low, makes a
# sort of 217 rows cost 3,837 against the index scan's 5,060 and loses it. Which
# is why the other setting, ``_preferring_this_kbs_partial_index``, is about the
# sort rather than about the filter. Nothing else in this suite passes a filter.
# ---------------------------------------------------------------------------

# 20% of the rows in any knowledge base, and 100% of them: the same shape with
# the filter's own selectivity moved from one end to the other, so a difference
# between the two cannot be read as "the filter was too selective".
FILTER_ONE_IN_FIVE = {"tier": "gold"}
FILTER_EVERYTHING = {"kb": KB_BIG}
# Two keys, in one bound `@>` over the whole object since the metadata-filter
# hardening landed. Collapsing two operators into one did not bring the index
# back, and that is the point: the estimate is not made per operator, it is made
# by matching the whole constant against `meta`'s MCV list, so one `@>` over two
# keys estimates as low as two `@>` did.
FILTER_TWO_KEYS = {"tier": "gold", "kb": KB_BIG}


def test_a_filtered_search_reaches_the_partial_index_under_a_custom_plan(
    engine, schema, settings, query_vectors
):
    """The control the next two specs need: one filter key loses nothing.

    A custom plan knows a single key's value, estimates it, and still chooses the
    partial index -- so what the generic-plan spec below finds for these two
    filters is about plan caching, not about filtering. Measured: 12 of 12 on the
    partial index at 1.4-1.9 ms with either.

    Both single-key filters, deliberately, and at both ends of the selectivity
    range: 20% of the rows and 100% of them. Two keys is the case a custom plan
    does *not* get right on its own, and it has its own spec below.
    """
    name = _build_big_index(engine, settings)
    for filter_metadata in (FILTER_ONE_IN_FIVE, FILTER_EVERYTHING):
        scans, prepared = _drive_searches(
            engine,
            KB_BIG,
            query_vectors,
            plan_cache_mode="force_custom_plan",
            index_name=name,
            filter_metadata=filter_metadata,
        )
        assert prepared, "the driver never prepared the filtered search statement"
        assert scans[name] == _DRIVEN_EXECUTIONS, (
            f"a custom plan must reach the partial index with {filter_metadata}: {scans}"
        )


def test_a_two_key_filter_reaches_the_partial_index_under_a_custom_plan(
    engine, schema, settings, query_vectors
):
    """Two metadata keys, which a custom plan alone does not save.

    Each key the filter carries cuts the entries of ``meta``'s MCV list the bound
    constant is contained in, and the planner then multiplies that by the
    knowledge-base predicate as if the two were independent. With ``kb`` among
    the keys they are not, so the estimate falls to 217 rows where 2,400 really
    match -- and a sort of 217 rows looks cheaper than the ordered index scan, so
    the index is priced out *even though the planner knows the filter's value*.
    Measured before the fix: 0 of 12 executions on the partial index under
    ``force_custom_plan``, where either single-key filter gets 12 of 12.

    So this is the spec that decided the shape of the fix. Asking for a custom
    plan cannot repair it, because it is not about plan caching; what repairs it
    is pricing the exact sort out, which is also what the 1536-dimension case
    needs. One mechanism, two defects. Measured after: 12 of 12.

    Two keys is an ordinary request -- the search route takes a whole
    ``filter_metadata`` object -- which is why this could not be deferred.
    """
    name = _build_big_index(engine, settings)
    scans, prepared = _drive_searches(
        engine,
        KB_BIG,
        query_vectors,
        plan_cache_mode="force_custom_plan",
        index_name=name,
        filter_metadata=FILTER_TWO_KEYS,
    )
    assert prepared, "the driver never prepared the filtered search statement"
    assert scans[name] == _DRIVEN_EXECUTIONS, (
        f"a two-key filtered search used the partial index for only {scans[name]} of "
        f"{_DRIVEN_EXECUTIONS} executions under a custom plan ({scans})"
    )


@pytest.mark.parametrize(
    "filter_metadata,selectivity",
    [
        (FILTER_ONE_IN_FIVE, "one row in five"),
        (FILTER_EVERYTHING, "every row"),
        (FILTER_TWO_KEYS, "one row in five, through two keys"),
    ],
)
def test_a_filtered_search_keeps_the_index_under_a_forced_generic_plan(
    engine, schema, settings, query_vectors, filter_metadata, selectivity
):
    """A filtered search must not lose the partial index once the plan is cached.

    This is the shape the suite never executed. Measured on this fixture before
    the fix, driving the real store on one pooled connection: 0 of 12 executions
    on the partial index, 5.0 ms for the 20%-selective filter and 15.9 ms for
    the 100%-selective one, against 1.9 ms and 1.4 ms under a custom plan. The
    filter that selects everything is the slower of the two, which is what rules
    out the reading that the filter was simply too selective for an ordered
    scan.

    A connection keeps its plan for the life of the pool entry, so this is not a
    one-search cost: it is every filtered search on that connection from the
    prepare threshold on.
    """
    name = _build_big_index(engine, settings)
    scans, prepared = _drive_searches(
        engine,
        KB_BIG,
        query_vectors,
        plan_cache_mode="force_generic_plan",
        index_name=name,
        filter_metadata=filter_metadata,
    )
    assert prepared, (
        "the driver never prepared the filtered search statement, so this test "
        "could not have seen a generic plan at all"
    )
    assert scans[name] == _DRIVEN_EXECUTIONS, (
        f"a filtered search selecting {selectivity} used the partial index for only "
        f"{scans[name]} of {_DRIVEN_EXECUTIONS} executions under a generic plan ({scans}); "
        "a search with a metadata filter must reach the index the same way an "
        "unfiltered one does"
    )


# ---------------------------------------------------------------------------
# 9. What the thresholds decide
#
# The build threshold decides which knowledge bases never get an index, so the
# question it answers is what a search costs without one. On this fixture, with
# ``dims = 384`` and a median over six query vectors:
#
#   share   rows    old shape        new shape, no index   new shape, indexed
#   30 %   12,000   1.5 ms r=0.41    1.8 ms r=0.41         1.2 ms r=0.41
#   21 %    8,400   4.2 ms r=0.18    4.5 ms r=0.18         1.3 ms r=0.39
#    5 %    2,000   9.9 ms r=0.36    2.2 ms r=1.00         2.2 ms r=1.00
#    1 %      400   1.5 ms r=1.00    1.0 ms r=1.00         0.8 ms r=1.00
#
# ``r`` is recall at 20 against an exact scan of the same knowledge base. Two
# things in that table are worth a spec rather than a comment: below about 5 %
# of the table the new shape stops using any HNSW index and starts answering
# exactly -- slower in principle, correct in fact, and the honest argument for
# this change -- and a partial index built for such a knowledge base is not used
# at all, so building one buys nothing and costs every insert.
# ---------------------------------------------------------------------------


def _build_index_ignoring_thresholds(engine, kb_id: str) -> str:
    """The partial index a knowledge base would get, built past the threshold gate.

    The service will not build one below the build threshold, and rightly; these
    specs need one anyway, to measure what it would be worth.
    """
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"SET maintenance_work_mem = '{_MEM_MB}MB'"))
        conn.execute(text(pvi.per_kb_index_ddl(kb_id, DIMS)))
        conn.execute(text(f"ANALYZE {SCHEMA}.embeddings"))
    return pvi.per_kb_index_name(kb_id, DIMS)


def _recall_against_an_exact_scan(session, sql: str, params: dict, vectors) -> float:
    """Mean fraction of an exact scan's top-20 that ``sql`` returns, over ``vectors``."""
    hits = []
    for vector in vectors:
        call = {**params, "embedding": _literal(vector)}
        session.execute(text("SET LOCAL enable_indexscan = off"))
        exact = {str(r[0]) for r in session.execute(text(sql), call).all()}
        session.rollback()
        got = {str(r[0]) for r in _search(session, sql, call)}
        hits.append(len(got & exact) / max(1, len(exact)))
    return sum(hits) / len(hits)


def test_the_new_shape_answers_exactly_where_the_old_one_was_approximate(
    engine, schema, settings, query_vectors
):
    """The trade this PR actually makes, for a knowledge base that is a small share.

    At 5 % of the table the new predicate lets the planner restrict
    ``ai.embeddings`` by its own ``knowledge_base_id``, and an exact scan of 2,000
    rows is then cheaper than any approximate one -- so the plan stops using an
    HNSW index and starts returning the exact answer. The old shape had to go
    through the shared index and post-filter, which at this share threw most of
    its candidates away: measured 0.36 recall at 20 against 1.00.

    "Slower and correct, where it was fast and quietly wrong" is a real argument
    for this change. It is not the argument the PR body makes, and nothing else
    here pins it.
    """
    sql, params = _capture_search_sql(engine, KB_MID, query_vectors[0])
    old_sql = _without_the_embeddings_predicate(sql)
    with Session(engine) as session:
        new_recall = _recall_against_an_exact_scan(session, sql, params, query_vectors)
        old_recall = _recall_against_an_exact_scan(session, old_sql, params, query_vectors)
        plan = _explain(session, sql, params)
    assert new_recall == 1.0, f"the new shape must return the exact answer here: {new_recall}"
    assert old_recall < 1.0, (
        f"the old shape is supposed to be the approximate one ({old_recall}); if it is now "
        "exact too, this fixture no longer shows the trade and the docstring above is wrong"
    )
    assert "hnsw" not in plan, (
        f"the exact answer above should come from an exact scan, not an index:\n{plan}"
    )


def test_a_partial_index_is_not_used_at_all_for_a_small_share_knowledge_base(
    engine, schema, settings, query_vectors
):
    """Why the build threshold exists, asserted rather than assumed.

    At 5 % of the table an exact scan of 2,000 rows is cheaper than an
    approximate one, and the planner says so: it declines this index when it is
    left to decide.

    It is no longer left to decide, so read this spec for what it says and not for
    what it used to imply. "An index the planner will not choose costs every
    insert and buys no plan" was the argument for the threshold, and
    ``_preferring_this_kbs_partial_index`` retired it -- an index the service has
    built is now used whether the planner prefers it or not. The argument that
    replaces it is about the *answer*: a knowledge base with no index is searched
    exactly, and one with an index is searched approximately. Measured at 1536
    dimensions on a 2,000-row knowledge base, 8.9 ms and recall 1.00 became
    1.3 ms and recall 0.68. So the threshold is what keeps small knowledge bases
    exact, and section 11 pins the half of that this spec does not: that the
    store does not steer at an index that is not there.
    """
    assert MID_ROWS < settings["VECTOR_PER_KB_INDEX_MIN_ROWS"]
    with engine.connect() as conn:
        assert pvi.index_action(conn, KB_MID) is None, "the gate should decline this one"

    name = _build_index_ignoring_thresholds(engine, KB_MID)
    sql, params = _capture_search_sql(engine, KB_MID, query_vectors[0])
    with Session(engine) as session:
        plan = _explain(session, sql, params)
    assert name not in plan, f"the planner is not expected to choose this index:\n{plan}"


def test_the_knowledge_base_the_default_threshold_leaves_out_is_the_measured_one(
    engine, schema, settings
):
    """Pins which side of the threshold each fixture knowledge base falls on.

    The row counts, the thresholds and the measurements in the comment above are
    one argument, and it stops being an argument if a later edit moves a row
    count without moving the table. KB_MED at 8,400 rows is the knowledge base
    the 10,000-row default declines -- the population the threshold decision is
    about -- and KB_BIG at 12,000 is the one it serves.
    """
    build_at = settings["VECTOR_PER_KB_INDEX_MIN_ROWS"]
    assert MED_ROWS < build_at <= BIG_ROWS, (MED_ROWS, build_at, BIG_ROWS)
    with engine.connect() as conn:
        assert pvi.index_action(conn, KB_BIG) == "build"
        assert pvi.index_action(conn, KB_MED) is None
        assert pvi.index_action(conn, KB_MID) is None
        assert pvi.index_action(conn, KB_THIN) is None
    # And the shares the module docstring's regime claim rests on.
    assert 0.28 < BIG_ROWS / TOTAL_ROWS < 0.32, BIG_ROWS / TOTAL_ROWS
    assert 0.19 < MED_ROWS / TOTAL_ROWS < 0.23, MED_ROWS / TOTAL_ROWS
    n_distinct = len({kb for kb, _ in ROW_COUNTS})
    assert n_distinct == 60, n_distinct
    # The one number the bound-item-side-id spec depends on, pinned where the
    # other fixture claims are. A generic plan prices a bound knowledge base id
    # at 1/n_distinct, and the ordered index scan survives that underestimate
    # while share x n_distinct stays below about 12; the regression the spec
    # asserts needs it above about 18. Adding knowledge bases is safe, removing
    # them is not, and this is where that is said out loud.
    assert BIG_ROWS / TOTAL_ROWS * n_distinct >= 18, BIG_ROWS / TOTAL_ROWS * n_distinct


# ---------------------------------------------------------------------------
# 10. Three claims that had no test that could fail
#
# Each of the three below was found by mutation: the guard was removed and the
# whole suite stayed green. A constant asserted to have a value is not the same
# claim as a bound actually applied, a function that returns "dropped" is not
# the same claim as a drop, and a test that reads ``vector_search``'s SQL is not
# the same claim as ``hybrid_search`` calling it.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_the_start_up_sweeps_count_runs_under_the_bound_it_sets(
    engine, schema, settings, monkeypatch
):
    """``SWEEP_TIMEOUT_MS`` is asserted as applied, not as a number.

    The sweep's grouped count reads all of ``ai.embeddings``, and it runs on the
    boot path, so it is bounded: a count that cannot finish must be abandoned and
    the boot must go on. Deleting the ``set_config('statement_timeout', ...)``
    statement so the count runs unbounded left the whole live suite green, because
    every other spec here runs the sweep against a database where the count
    finishes in milliseconds.

    So this one makes the count unable to finish -- a second connection holds an
    ACCESS EXCLUSIVE lock on the table, which the count has to wait for -- and
    asserts that the sweep comes back anyway, within the bound, having abandoned
    it. Without the bound applied the count waits for the lock forever and this
    test fails on its own timeout rather than hanging the run.

    The bound is lowered from its real value for the test's sake; what is being
    pinned is that the value in ``SWEEP_TIMEOUT_MS`` reaches the server as a
    statement bound, whatever it is.
    """
    monkeypatch.setattr(pvi, "SWEEP_TIMEOUT_MS", 400)

    # The control, first: with nothing in the way the count is what finds KB_BIG,
    # since it has no index for the catalog half of the sweep to notice.
    assert pvi.kbs_needing_a_per_kb_index(engine) == [KB_BIG]

    blocker = engine.connect()
    try:
        blocker.execute(text(f"LOCK TABLE {SCHEMA}.embeddings IN ACCESS EXCLUSIVE MODE"))
        started = time.monotonic()
        found = pvi.kbs_needing_a_per_kb_index(engine)
        elapsed = time.monotonic() - started
    finally:
        blocker.rollback()
        blocker.close()

    assert elapsed < 20, (
        f"the sweep took {elapsed:.1f} s against a blocked count; its "
        f"{pvi.SWEEP_TIMEOUT_MS} ms bound is not reaching the server"
    )
    assert found == [], (
        "with the count abandoned the sweep has only its catalog half, and there are "
        f"no per-knowledge-base indexes in this schema, so it should find nothing: {found}"
    )


def test_a_drop_cannot_report_success_while_another_caller_holds_the_lock(engine, schema, settings):
    """A drop that did not happen must not come back as ``dropped``.

    ``drop_per_kb_vector_indexes`` runs on the knowledge-base-delete path, where
    nothing inspects what it returns, so a silent failure there is permanent: the
    knowledge base row is gone and nothing will ever reconcile the index again.
    Turning its ``raise PerKbVectorIndexBuildInProgress`` into ``continue`` left
    both tiers green, and the function then returned
    ``{"status": "dropped", "indexes": []}`` -- success, for an index still on
    disk. The test that looks like it covers this replaces the service with a
    ``MagicMock`` and exercises the task's ``except`` clause instead.

    Asserted on disk as well as on the return value: the index is still there.
    """
    _build_big_index(engine, settings)
    lock = pvi.index_lock_relation(KB_BIG, DIMS)
    holder = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        assert (
            holder.execute(text(pvi.partition_build_lock_sql()), {"relation": lock}).scalar()
            is True
        )
        with pytest.raises(pvi.PerKbVectorIndexBuildInProgress):
            pvi.drop_per_kb_vector_indexes(KB_BIG, engine=engine)
        with engine.connect() as conn:
            assert pvi.existing_per_kb_indexes(conn, KB_BIG) == {DIMS: True}, (
                "the index must still be on disk after a drop that could not take the lock"
            )
    finally:
        holder.execute(text(pvi.partition_build_unlock_sql()), {"relation": lock})
        holder.close()


def test_hybrid_search_has_a_vector_leg_that_reaches_the_partial_index(
    engine, schema, settings, query_vectors
):
    """``hybrid_search`` really runs a vector search, and it lands on the index.

    The unit spec named after this claim never calls ``hybrid_search``: it reads
    ``vector_search``'s SQL and asserts the predicate is in it. Replacing the
    whole ``await self.vector_search(...)`` inside ``hybrid_search`` with
    ``vector_results = []`` left that spec green.

    The honest place to pin it is here rather than in the unit tier, because what
    makes the claim worth anything is not that the call exists in the source but
    that the query it issues reaches this knowledge base's index -- which only a
    real index and a real planner can say. So: call ``hybrid_search``, then read
    the partial index's own scan counter.

    The keyword leg is deliberately not asserted on. It is allowed to come back
    empty (that is its documented degradation), and hybrid search is still
    supposed to answer from its vector leg when it does.
    """
    name = _build_big_index(engine, settings)
    shared = f"idx_ai_embeddings_hnsw_{DIMS}"
    before = _idx_scans(engine, name, shared)

    probe = create_engine(_dsn())
    connection = probe.connect()
    try:
        with Session(bind=connection) as session:
            store = _ChunkStore(db_session=session, knowledge_base_id=KB_BIG, schema=SCHEMA)
            items = asyncio.run(
                store.hybrid_search(query="passage", embedding=list(query_vectors[0]), top_k=10)
            )
            session.commit()
    finally:
        connection.close()
        probe.dispose()

    after = _idx_scans(engine, name, shared)
    assert items, "hybrid search returned nothing at all"
    assert after[name] > before[name], (
        "hybrid search never scanned this knowledge base's partial index, so it has no "
        f"vector leg reaching it (before {before}, after {after})"
    )
    assert all(item.knowledge_base_id == KB_BIG for item in items), items


# ---------------------------------------------------------------------------
# 11. The setting that makes the planner take the index, and its gate
#
# ``_preferring_this_kbs_partial_index`` prices the exact sort out of the search.
# That is a penalty on every sort in the statement, not an instruction naming an
# index, so left ungated it would drive a knowledge base with no index of its own
# onto the shared per-dimension index -- which spans every knowledge base and
# post-filters. Measured at 1536 dimensions with no partial index built, median
# of six query vectors:
#
#   share   rows    the planner's own choice     the same, ungated
#    21 %   8,400   exact scan, 34.4 ms, r=1.00  shared index, 6.3 ms, r=0.04
#     5 %   2,000   exact scan,  9.4 ms, r=1.00  shared index, 31.3 ms, r=0.04
#     1 %     400   exact scan,  1.8 ms, r=1.00  shared index, 39.8 ms, r=0.36
#
# Slower and wrong, so the gate is the fix rather than a refinement on it. These
# specs pin it from the outside: same answers, same index counters, for a
# knowledge base the build threshold left alone and for one whose index is on
# disk but INVALID.
#
# They run at 5 % (KB_MID) because that is where this fixture's planner answers
# exactly without an index, so "the answer did not change" is a claim with
# something in it. At 21 % and above the shared index is already what an
# unindexed search gets, and there is no exact answer to preserve.
# ---------------------------------------------------------------------------


def _exact_answers(engine, kb_id, vectors, **kwargs) -> list[list[str]]:
    """The exact top-20 for each query vector: the same SQL with no index at all."""
    sql, params = _capture_search_sql(engine, kb_id, vectors[0], **kwargs)
    answers = []
    with Session(engine) as session:
        for vector in vectors:
            session.execute(text("SET LOCAL enable_indexscan = off"))
            rows = session.execute(text(sql), {**params, "embedding": _literal(vector)}).all()
            session.rollback()
            answers.append([str(row[0]) for row in rows])
    return answers


def _drive_and_collect(engine, kb_id, vectors, *counted, plan_cache_mode=None, **kwargs):
    """``vector_search`` for real, once per vector, on one connection.

    Returns ``(scans, answers)``: the scans each of ``counted`` served over the
    run, and what each search returned. Same connection discipline as
    ``_drive_searches`` -- an engine of its own, closed before the counters are
    read. ``plan_cache_mode`` is left alone by default, so the default run is
    PostgreSQL deciding for itself when to go generic.
    """
    before = _idx_scans(engine, *counted)
    probe = create_engine(_dsn())
    connection = probe.connect()
    answers = []
    try:
        with Session(bind=connection) as session:
            if plan_cache_mode is not None:
                session.execute(text(f"SET plan_cache_mode = '{plan_cache_mode}'"))
            store = _ChunkStore(db_session=session, knowledge_base_id=kb_id, schema=SCHEMA)
            for vector in vectors:
                items = asyncio.run(store.vector_search(embedding=list(vector), top_k=20, **kwargs))
                answers.append([item.item_id for item in items])
            session.commit()
    finally:
        connection.close()
        probe.dispose()
    after = _idx_scans(engine, *counted)
    return {name: after[name] - before[name] for name in counted}, answers


def test_a_knowledge_base_with_no_index_of_its_own_keeps_the_answer_it_has_today(
    engine, schema, settings, query_vectors
):
    """The gate, from the outside: no index, so nothing changes.

    KB_MID is 5 % of the table and below the build threshold, so the service
    never gives it an index and its search is an exact scan. Ungated, the sort
    penalty would move it onto the shared index and throw most of the answer
    away.
    """
    exact = _exact_answers(engine, KB_MID, query_vectors)
    shared = f"idx_ai_embeddings_hnsw_{DIMS}"
    scans, answers = _drive_and_collect(engine, KB_MID, query_vectors, shared)
    assert scans[shared] == 0, (
        f"a knowledge base with no partial index must not be pushed onto the shared "
        f"one, which post-filters away most of what it returns: {scans}"
    )
    assert answers == exact, (
        "the answer for an unindexed knowledge base must be the exact one, "
        f"unchanged by this fix:\n{answers}\n{exact}"
    )


def test_an_index_that_is_invalid_is_not_steered_at_either(engine, schema, settings, query_vectors):
    """An INVALID index is in the catalog and cannot answer a query.

    A build that was interrupted or ran out of disk leaves one behind, and
    ``ensure_per_kb_vector_index`` repairs it on its next pass -- until then the
    search has to behave as though there were no index, which is what the
    probe's ``indisvalid`` is for. Without it the sort would be priced out for a
    knowledge base with nothing to fall on, and the shared index would take the
    search.
    """
    name = _build_index_ignoring_thresholds(engine, KB_MID)
    _invalidate(engine, name)
    with engine.connect() as conn:
        assert pvi.existing_per_kb_indexes(conn, KB_MID) == {DIMS: False}, "not invalidated"

    exact = _exact_answers(engine, KB_MID, query_vectors)
    shared = f"idx_ai_embeddings_hnsw_{DIMS}"
    scans, answers = _drive_and_collect(engine, KB_MID, query_vectors, name, shared)
    assert scans[name] == 0, f"an INVALID index cannot serve a scan: {scans}"
    assert scans[shared] == 0, (
        f"an INVALID index is not an index to fall on, so the sort must not be "
        f"priced out and the shared index must not take the search: {scans}"
    )
    assert answers == exact, (answers, exact)


@pytest.mark.parametrize("was", ["on", "off"])
def test_a_vector_search_leaves_enable_sort_as_it_found_it(
    engine, schema, settings, query_vectors, was
):
    """The restore, which is not optional.

    ``hybrid_search`` runs its keyword leg on this same session immediately after
    the vector leg, and a keyword ranking is a sort. A penalty left on would
    follow the vector search into it -- and, with a session-level setting, into
    every other statement on that pooled connection.

    Both starting values, because a restore that put back a hardcoded ``on``
    would pass against the default and quietly clear a setting the caller had
    made.
    """
    _build_big_index(engine, settings)
    probe = create_engine(_dsn())
    connection = probe.connect()
    try:
        with Session(bind=connection) as session:
            session.execute(text(f"SET LOCAL enable_sort = {was}"))
            store = _ChunkStore(db_session=session, knowledge_base_id=KB_BIG, schema=SCHEMA)
            items = asyncio.run(store.vector_search(embedding=list(query_vectors[0]), top_k=20))
            assert items, "the search under test returned nothing"
            after = session.execute(text("SELECT current_setting('enable_sort')")).scalar()
            session.commit()
    finally:
        connection.close()
        probe.dispose()
    assert after == was, (
        f"vector_search left enable_sort at {after!r} in a transaction that had it "
        f"at {was!r}; the next statement in the caller's transaction pays for that"
    )


def test_a_filter_matching_no_row_still_answers_nothing(engine, schema, settings, query_vectors):
    """What the forced index scan costs, and that it does not cost correctness.

    With the sort priced out, an extra predicate that matches nothing turns the
    search into a walk of the index looking for rows that are not there. Measured
    at 1536 dimensions on the indexed 30 % knowledge base: a filter matching no
    row went from 3.0 ms to 38.3 ms, and a ``source_ids`` matching no row from
    2.4 ms to 36.7 ms. The answer is the same in both -- none -- and the cost is
    bounded rather than proportional, because pgvector stops an iterative scan at
    ``hnsw.max_scan_tuples`` (20,000 by default).

    That is the trade this fix makes in the direction nobody wants, so it is
    pinned as behaviour: an empty answer, not a wrong one and not an error.
    """
    name = _build_big_index(engine, settings)
    shared = f"idx_ai_embeddings_hnsw_{DIMS}"
    for kwargs in (
        {"filter_metadata": {"tier": "platinum"}},
        {"filter_metadata": {"tier": "gold", "kb": "no such knowledge base"}},
        {"source_ids": ["9f8b1c2e-0000-4000-8000-0000000000bb"]},
        {"item_ids": {"9f8b1c2e-0000-4000-8000-0000000000cc"}},
    ):
        _, answers = _drive_and_collect(engine, KB_BIG, query_vectors[:2], name, shared, **kwargs)
        assert answers == [[], []], f"{kwargs} matches no row, so the answer is none: {answers}"


def test_a_forced_index_scan_still_returns_every_row_that_matches(
    engine, schema, settings, query_vectors
):
    """Fewer rows match than ``top_k``, and all of them come back.

    This is the property that made pricing the sort out acceptable at all. An
    ordered HNSW scan under a ``LIMIT`` produces candidates and the join and the
    extra predicates throw some away, so a search whose predicates are selective
    can be starved: pgvector emits about ``ef_search`` candidates and stops, and
    what survives the filter is whatever happened to be among them.
    ``_apply_iterative_scan`` is what stops that -- it makes pgvector keep going
    until the ``LIMIT`` is filled -- and before this fix an unfiltered search at
    1536 dimensions was not on an index scan at all, so nothing here depended on
    it.

    Now it does. Twelve rows of a 12,000-row knowledge base, asked for as a
    ``top_k`` of 20, on a forced ordered index scan: all twelve, best first.
    """
    name = _build_big_index(engine, settings)
    with engine.connect() as conn:
        wanted = [
            str(row[0])
            for row in conn.execute(
                text(
                    f"SELECT id FROM {SCHEMA}.chunks WHERE knowledge_base_id = :kb "
                    "ORDER BY id LIMIT 12"
                ),
                {"kb": KB_BIG},
            ).all()
        ]
        conn.rollback()
    assert len(wanted) == 12, wanted

    # Twice through the vectors, so the short ones come round again after psycopg
    # has prepared the statement. A cached generic plan built while the sort was
    # priced out is re-used whatever enable_sort says afterwards, so without a
    # custom plan the re-run returns the same short answer -- and a spec that only
    # ever ran unprepared executions would not notice.
    driven = list(query_vectors) * 2
    shared = f"idx_ai_embeddings_hnsw_{DIMS}"
    scans, answers = _drive_and_collect(engine, KB_BIG, driven, name, shared, item_ids=set(wanted))
    assert scans[name] == len(driven), (
        f"this spec is only worth anything on the forced index scan: {scans}"
    )
    for got in answers:
        assert sorted(got) == sorted(wanted), (
            f"an ordered index scan under a LIMIT of 20 dropped rows that matched: "
            f"{len(got)} of {len(wanted)}"
        )

    # And again with the generic plan pinned, which is what makes the re-run's
    # request for a custom plan load-bearing rather than decorative. A generic
    # plan is built once and cached, and PostgreSQL does not rebuild it when a
    # planner GUC changes -- so the plan built while the sort was priced out is
    # the plan the re-run would get, and it would return the same short answer.
    scans, answers = _drive_and_collect(
        engine,
        KB_BIG,
        driven,
        name,
        shared,
        plan_cache_mode="force_generic_plan",
        item_ids=set(wanted),
    )
    assert scans[name] == len(driven), (
        f"the generic plan must still be the forced index scan here: {scans}"
    )
    for got in answers:
        assert sorted(got) == sorted(wanted), (
            "with the generic plan pinned, the re-run has to ask for a custom plan or it "
            f"re-uses the plan that came up short: {len(got)} of {len(wanted)}"
        )
