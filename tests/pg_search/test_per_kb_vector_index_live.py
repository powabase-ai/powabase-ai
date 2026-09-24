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
width: that the sort is priced out only for an *unrestricted* search on a
knowledge base with an index to fall on, that the answer for a knowledge base
without one does not change, and that a search the caller restricted -- named
rows, a source, a metadata filter -- is given the exact plan instead and returns
exactly what an exact scan returns. That last one is the half of the design that
does not depend on the width at all: at 1536 dimensions the planner declines an
HNSW index for a restricted search anyway, so insisting on it changes nothing
there and costs 42.7 ms against 44.8 ms; at 384 the planner would take the index
and answer a full page of the wrong rows.

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
# A second source every fortieth chunk, which is what lets a ``source_ids``
# restriction be a restriction: with one source per knowledge base the only
# source_ids a spec could pass were "all of it" or "none of it", and the shape
# section 12 needs is one that matches many more rows than ``top_k`` and a small
# part of the knowledge base -- which is what a source is, one document among
# many. In the indexed knowledge base that is 300 rows: 2.5% of it, and 15 times
# ``top_k``.
SOURCE_B = "9f8b1c2e-0000-4000-8000-0000000000ab"
SOURCE_B_EVERY = 40

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
                source = SOURCE_B if i % SOURCE_B_EVERY == 0 else SOURCE
                chunks.write(f"{item_id}\t{kb_id}\t{source}\tpassage {i}\t{meta}\n")
                embeddings.write(
                    f"{item_id}\tchunks\t{kb_id}\t{source}\ttest-embed\t{DIMS}\t"
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


def _drop_all_partial_indexes(engine, schema: str = SCHEMA) -> None:
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        names = [
            row[0]
            for row in conn.execute(
                text(
                    "SELECT c.relname FROM pg_class c JOIN pg_namespace n "
                    "ON n.oid = c.relnamespace WHERE n.nspname = :s AND c.relkind = 'i' "
                    "AND c.relname LIKE 'hnsw_kb_%'"
                ),
                {"s": schema},
            ).all()
        ]
        for name in names:
            conn.execute(text(f'DROP INDEX CONCURRENTLY IF EXISTS "{schema}".{name}'))


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


def _capture_search_sql(engine, kb_id, embedding, *, schema=SCHEMA, **kwargs):
    """The SQL the real store issues, and the parameters it binds."""
    with Session(engine) as session:
        recorder = _RecordingSession(session)
        store = _ChunkStore(db_session=recorder, knowledge_base_id=kb_id, schema=schema)
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


def _idx_scans(engine, *names: str, schema: str = SCHEMA) -> dict[str, int]:
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
            {"s": schema, "n": list(names)},
        ).all()
        conn.rollback()
    found = {name: int(count) for name, count in rows}
    missing = [name for name in names if name not in found]
    assert not missing, f"no such index in {schema}: {missing}"
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
# 6. What the index covers: one population, not four
#
# ``ai.embeddings`` holds the vectors of four different item tables, and
# ``item_table`` is NOT NULL and one of those four values. A knowledge base can
# therefore hold more than one population at once -- chunks and whole documents,
# chunks and graph nodes -- and the index this feature builds is named by the
# knowledge base and the dimension only. So an index built for such a knowledge
# base spans every population in it, while the search this feature steers onto
# that index joins ``chunks`` and can only ever return chunk rows.
#
# That is not a tidiness point, it is recall. The ordered scan walks the index by
# distance and the join throws away everything that is not a chunk, so the rows
# the caller gets are drawn from whatever the scan reached before the limit was
# filled -- and the deeper the scan has to go, the more of the real top-k it never
# sees. This fixture is the smallest one that shows it: the same 3,000 chunk
# rows, the same query vectors, once with 7,000 whole-document embeddings beside
# them in the same knowledge base and once without.
#
# The claim is asserted at the answer, not at the DDL. An index definition that
# names ``item_table`` is evidence of an intention; recall against an exact scan
# of the chunk rows is evidence of the outcome, and it is the outcome that moved
# (measured on this fixture over two runs, mean recall at 20 over six query
# vectors, both runs through the real store on an index it built itself: 0.53-0.57
# with the second population in the index against 0.82-0.83 without, and 0.05-0.10
# at the worst vector against 0.60).
# Both runs assert the index served every execution, because an exact scan
# answers with recall 1.00 and a comparison that quietly stopped using the index
# would read as a pass.
#
# The twin knowledge base is what makes the recall number mean something. HNSW
# recall on this fixture's clustered vectors is well below 1.00 even over one
# population -- the vectors within a cluster are near-ties -- so an absolute bar
# would be a fixture constant rather than a claim. The twin holds the chunk
# population and nothing else, so it measures what this search's recall is
# *allowed* to be, from the same rows and the same queries.
#
# Thresholds of its own, deliberately low: the row count that decides whether to
# build is the other half of this question (a knowledge base that crosses the
# threshold on the *sum* over its populations, and would not cross it on chunks
# alone), and pinning that belongs where the counting is unit-testable. These
# specs are about what the index covers once there is one, so both knowledge bases
# here are above the threshold either way and the gate is not the variable.
# ---------------------------------------------------------------------------

MIXED_SCHEMA = f"{SCHEMA}_populations"
# Two knowledge bases with the same chunk population, from the same vectors, so
# the only difference between them is the second population in one of them.
KB_TWO_POPULATIONS = "9f8b1c2e-0000-4000-8000-0000000000f1"
KB_ONE_POPULATION = "9f8b1c2e-0000-4000-8000-0000000000f2"
POPULATION_CHUNK_ROWS = 3_000
# More rows than the chunk population rather than fewer: a knowledge base indexed
# at the page or whole-document level alongside its chunks is the ordinary case,
# not a corner, and the harm scales with how much of the index cannot join.
POPULATION_OTHER_ROWS = 7_000


@pytest.fixture(scope="module")
def mixed_population_schema(engine):
    """A second scratch schema: one knowledge base with two populations, one with one."""
    raw_dsn = engine.url.set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {MIXED_SCHEMA} CASCADE")
        conn.execute(f"CREATE SCHEMA {MIXED_SCHEMA}")
        conn.execute(f"""
            CREATE TABLE {MIXED_SCHEMA}.chunks (
                id uuid PRIMARY KEY,
                knowledge_base_id uuid NOT NULL,
                source_id uuid NOT NULL,
                text text NOT NULL,
                meta jsonb DEFAULT '{{}}'::jsonb
            )
        """)
        conn.execute(f"""
            CREATE TABLE {MIXED_SCHEMA}.embeddings (
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
        conn.execute(f"CREATE INDEX ON {MIXED_SCHEMA}.chunks (knowledge_base_id)")
        conn.execute(f"CREATE INDEX ON {MIXED_SCHEMA}.embeddings (item_id)")
        conn.execute(f"CREATE INDEX ON {MIXED_SCHEMA}.embeddings (knowledge_base_id)")
        # Serial builds here too, for the reason the first fixture gives.
        conn.execute(f"ALTER TABLE {MIXED_SCHEMA}.embeddings SET (parallel_workers = 0)")

        rng = np.random.default_rng(777)
        chunk_vectors = _vectors(rng, POPULATION_CHUNK_ROWS)
        other_vectors = _vectors(rng, POPULATION_OTHER_ROWS)
        chunks = io.StringIO()
        embeddings = io.StringIO()
        for kb_id in (KB_TWO_POPULATIONS, KB_ONE_POPULATION):
            for i in range(POPULATION_CHUNK_ROWS):
                item_id = str(uuid.uuid4())
                meta = json.dumps({"tier": "gold", "kb": kb_id})
                chunks.write(f"{item_id}\t{kb_id}\t{SOURCE}\tpassage {i}\t{meta}\n")
                embeddings.write(
                    f"{item_id}\tchunks\t{kb_id}\t{SOURCE}\ttest-embed\t{DIMS}\t"
                    f"{_literal(chunk_vectors[i])}\n"
                )
        # The second population, in one of the two knowledge bases. No rows in any
        # item table to match them: an embedding of a whole document is not a
        # chunk, so a chunk search's join drops it however it was reached, which
        # is the whole point -- these are index entries that cannot answer.
        for i in range(POPULATION_OTHER_ROWS):
            embeddings.write(
                f"{uuid.uuid4()}\tfull_documents\t{KB_TWO_POPULATIONS}\t{SOURCE}\t"
                f"test-embed\t{DIMS}\t{_literal(other_vectors[i])}\n"
            )
        chunks.seek(0)
        embeddings.seek(0)
        with conn.cursor() as cur:
            with cur.copy(
                f"COPY {MIXED_SCHEMA}.chunks (id, knowledge_base_id, source_id, text, meta) "
                "FROM STDIN"
            ) as copy:
                copy.write(chunks.read())
            with cur.copy(
                f"COPY {MIXED_SCHEMA}.embeddings (item_id, item_table, knowledge_base_id, "
                "source_id, embedding_model, dims, embedding) FROM STDIN"
            ) as copy:
                copy.write(embeddings.read())
        conn.execute(f"VACUUM ANALYZE {MIXED_SCHEMA}.embeddings")
        conn.execute(f"VACUUM ANALYZE {MIXED_SCHEMA}.chunks")
    yield MIXED_SCHEMA
    with psycopg.connect(raw_dsn, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {MIXED_SCHEMA} CASCADE")


@pytest.fixture
def mixed_population(engine, mixed_population_schema, monkeypatch):
    """The service pointed at that schema, with thresholds both populations clear."""
    monkeypatch.setattr(pvi, "AI_SCHEMA", MIXED_SCHEMA)
    values = {
        "VECTOR_PER_KB_INDEX_MIN_ROWS": 1_000,
        "VECTOR_PER_KB_INDEX_DROP_ROWS": 500,
        "VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB": _MEM_MB,
    }
    monkeypatch.setattr(pvi, "get_setting", lambda key: values[key])
    monkeypatch.setattr(pvi, "read_overrides", lambda conn, *keys: dict(values))
    _drop_all_partial_indexes(engine, MIXED_SCHEMA)
    yield MIXED_SCHEMA
    _drop_all_partial_indexes(engine, MIXED_SCHEMA)


def _item_tables_of(engine, kb_id: str, schema: str) -> dict[str, int]:
    """How many embeddings each item table holds for one knowledge base."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT item_table, count(*) FROM {schema}.embeddings "
                "WHERE knowledge_base_id = CAST(:kb AS uuid) GROUP BY item_table"
            ),
            {"kb": kb_id},
        ).all()
        conn.rollback()
    return {str(table): int(count) for table, count in rows}


def _indexed_tuples(engine, name: str, schema: str) -> int:
    """How many tuples the index holds, from the catalog rather than from its DDL.

    ``pg_class.reltuples`` on the index relation is set by the build, so this is a
    measurement of what the index covers and not a reading of the predicate that
    was meant to decide it -- which is the difference the recall numbers above
    turn on.
    """
    with engine.connect() as conn:
        count = conn.execute(
            text(
                "SELECT c.reltuples FROM pg_class c JOIN pg_namespace n "
                "ON n.oid = c.relnamespace WHERE n.nspname = :s AND c.relname = :n"
            ),
            {"s": schema, "n": name},
        ).scalar()
        conn.rollback()
    assert count is not None, f"no such index in {schema}: {name}"
    return int(count)


def _recall_of_the_real_search(engine, kb_id: str, vectors, *, schema: str):
    """``(scans, mean recall, worst recall)`` for the store's own unrestricted search.

    The expectation is PostgreSQL's own answer to the same statement with index
    scans off, once per query vector, so nothing about the right answer is written
    down here.
    """
    name = pvi.per_kb_index_name(kb_id, DIMS)
    exact = _exact_answers(engine, kb_id, vectors, schema=schema)
    assert all(len(answer) == TOP_K for answer in exact), [len(a) for a in exact]
    scans, answers = _drive_and_collect(engine, kb_id, vectors, name, schema=schema)
    hits = [len(set(got) & set(want)) / len(want) for got, want in zip(answers, exact)]
    return scans[name], sum(hits) / len(hits), min(hits)


def test_a_knowledge_base_with_two_item_tables_gets_an_index_of_one_population(
    engine, mixed_population, query_vectors
):
    """The index covers the population the search can join, and the answer shows it.

    Two knowledge bases, the same 3,000 chunk rows built from the same vectors,
    each given its own index by the real service path. One of them also holds
    7,000 whole-document embeddings, which no chunk search can ever return.

    Four assertions, and the order is the argument:

    - the fixture really is what it claims -- two populations in one knowledge base
      and one in the other;
    - both indexes hold the same number of tuples, which is the chunk population.
      An index that spans the knowledge base's other populations holds 10,000
      where its twin holds 3,000;
    - the index served every execution on both, without which the recall
      comparison below would be comparing an exact scan against an index scan and
      would pass for the wrong reason;
    - and the recall the search gets is the recall the same rows and the same
      queries give through an index that holds nothing else. Measured with the
      second population in the index: 0.53-0.57 mean and 0.05-0.10 at the worst
      vector, against 0.82-0.83 and 0.60.

    The margins are there because the two index builds see their rows in a
    different heap order and an HNSW graph is built in the order it reads, so the
    twins are not required to agree to the row; they are required to agree within
    a few rows of a 20-row page.
    """
    two = _item_tables_of(engine, KB_TWO_POPULATIONS, MIXED_SCHEMA)
    one = _item_tables_of(engine, KB_ONE_POPULATION, MIXED_SCHEMA)
    assert two == {"chunks": POPULATION_CHUNK_ROWS, "full_documents": POPULATION_OTHER_ROWS}, two
    assert one == {"chunks": POPULATION_CHUNK_ROWS}, one

    for kb_id in (KB_TWO_POPULATIONS, KB_ONE_POPULATION):
        outcome = pvi.ensure_per_kb_vector_index(kb_id, engine=engine)
        assert outcome["built"] == [pvi.per_kb_index_name(kb_id, DIMS)], outcome
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"ANALYZE {MIXED_SCHEMA}.embeddings"))

    covered = {
        kb_id: _indexed_tuples(engine, pvi.per_kb_index_name(kb_id, DIMS), MIXED_SCHEMA)
        for kb_id in (KB_TWO_POPULATIONS, KB_ONE_POPULATION)
    }
    # Everything is measured before anything is asserted, so a failure of the
    # first claim still reports what it cost at the answer -- which is the half of
    # this that an index definition cannot tell anyone.
    scans_two, mean_two, worst_two = _recall_of_the_real_search(
        engine, KB_TWO_POPULATIONS, query_vectors, schema=MIXED_SCHEMA
    )
    scans_one, mean_one, worst_one = _recall_of_the_real_search(
        engine, KB_ONE_POPULATION, query_vectors, schema=MIXED_SCHEMA
    )
    measured = (
        f"(two populations: {covered[KB_TWO_POPULATIONS]} tuples indexed, recall "
        f"{mean_two:.3f} mean / {worst_two:.3f} worst over {scans_two} index scans; "
        f"one population: {covered[KB_ONE_POPULATION]} tuples indexed, recall "
        f"{mean_one:.3f} mean / {worst_one:.3f} worst over {scans_one} index scans)"
    )

    assert covered[KB_ONE_POPULATION] == POPULATION_CHUNK_ROWS, (covered, measured)
    assert covered[KB_TWO_POPULATIONS] == covered[KB_ONE_POPULATION], (
        f"the index built for a knowledge base with two populations holds "
        f"{covered[KB_TWO_POPULATIONS]} tuples where the one built for the same chunk "
        f"population alone holds {covered[KB_ONE_POPULATION]}; the difference is "
        f"{POPULATION_OTHER_ROWS} entries a chunk search can reach and can never "
        f"return {measured}"
    )
    assert scans_two == len(query_vectors) and scans_one == len(query_vectors), (
        "both searches have to go through the knowledge base's own index for the "
        f"recall below to be about the index at all: {scans_two} and {scans_one} of "
        f"{len(query_vectors)} executions did"
    )
    assert mean_two >= mean_one - 0.05, (
        f"a chunk search on a knowledge base that also holds {POPULATION_OTHER_ROWS} "
        f"whole-document embeddings returned {mean_two:.3f} of the exact answer, where "
        f"the same {POPULATION_CHUNK_ROWS} chunk rows and the same queries return "
        f"{mean_one:.3f} through an index holding only them; the scan is walking "
        "entries that cannot join"
    )
    assert worst_two >= worst_one - 0.20, (
        f"the worst query vector returned {worst_two:.3f} of the exact answer against "
        f"{worst_one:.3f} on the single-population twin; the tail is where a scan that "
        "has to walk past rows it cannot return runs out of budget first"
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
# runtime and attached-knowledge-base configuration paths. It adds a bound value
# to the statement -- ``c.meta @> $n`` -- and it is the one value the planner
# cannot be given as a literal, because it is caller data.
#
# The planner's estimates for it are wrong in two different ways, and the service
# no longer tries to win the cost race they decide. A *generic* plan has no value
# to estimate at all and falls back to a fixed guess. A *custom* plan does better,
# but not always well: it estimates ``meta @> const`` by testing the constant
# against the column's MCV list, so the more keys the filter carries the fewer MCV
# entries it matches, and it then multiplies that by ``c.knowledge_base_id = ...``
# as if the two were independent -- which they are not, when one of the filter's
# own keys is the knowledge base. Measured on this fixture, ``top_k`` 20:
#
#   filter                          estimate   rows that really match
#   {"tier": "gold"}                     722                    2,400
#   {"kb": KB_BIG}                     1,076                   12,000
#   {"tier": "gold", "kb": KB_BIG}       217                    2,400
#
# Which way that race comes out decided what a filtered search returned, and it
# came out differently at different vector widths -- so the answer a caller got
# depended on the width of their embeddings. The service now settles it in the
# same direction at every width: a search the caller restricted gets the exact
# plan insisted on (``_insisting_on_an_exact_search``), because an ordered ANN
# scan cannot promise the rows the caller asked for and a full page of ``top_k``
# rows has no signal in it to notice that with. So these specs assert the
# opposite of what they used to: a filtered search reaches neither index and
# answers exactly, at both ends of the selectivity range and with the correlated
# two-key estimate as well.
#
# Each carries the positive control beside it -- the *unrestricted* search on the
# same knowledge base still being steered onto the partial index -- because
# without it this whole set would also pass against a build with the feature
# removed, which is the failure mode this module has been caught by twice.
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


@pytest.mark.parametrize(
    "filter_metadata,selectivity",
    [
        (FILTER_ONE_IN_FIVE, "one row in five"),
        (FILTER_EVERYTHING, "every row"),
        (FILTER_TWO_KEYS, "one row in five, through two correlated keys"),
    ],
)
def test_a_filtered_search_keeps_the_exact_plan_and_answers_exactly(
    engine, schema, settings, query_vectors, filter_metadata, selectivity
):
    """A metadata filter takes neither index, and returns the exact answer.

    Three filters, and what used to be three different stories about them is one:
    the 20%-selective one and the 100%-selective one kept the partial index when
    the planner was asked for a custom plan, the correlated two-key one did not,
    and none of them kept it once the plan went generic. Every one of those
    outcomes was a cost race between an estimate that is wrong (see the section
    header) and a sort, decided differently at different vector widths -- so what
    a caller got back depended on how wide their embeddings were.

    Now all three are the same and the width does not enter: the search is given
    the exact plan, it scans neither the knowledge base's partial index nor the
    shared per-dimension one, and it answers exactly what an exact scan answers.
    Measured at 1536 dimensions, the width production stores, the largest
    restricted shape costs 42.7 ms this way against 44.8 ms unaided -- the plan
    the planner would have chosen there anyway -- and what replaces the ordered
    scans is a bitmap scan of the same indexes, not a sequential scan.

    Both sides of the plan cache, twelve executions each so the second pass runs
    on a prepared statement, with ``prepared`` as the negative control and the
    unrestricted search as the positive one.
    """
    name = _build_big_index(engine, settings)
    shared = f"idx_ai_embeddings_hnsw_{DIMS}"
    kwargs = {"filter_metadata": filter_metadata}
    exact = _exact_answers(engine, KB_BIG, query_vectors, **kwargs)
    assert all(len(answer) == TOP_K for answer in exact), [len(a) for a in exact]

    driven = list(query_vectors) * 2
    for plan_cache_mode in ("force_custom_plan", "force_generic_plan"):
        prepared: list[str] = []
        scans, answers = _drive_and_collect(
            engine,
            KB_BIG,
            driven,
            name,
            shared,
            plan_cache_mode=plan_cache_mode,
            prepared_out=prepared,
            **kwargs,
        )
        assert prepared, (
            "the driver never prepared the filtered search statement, so this leg "
            f"could not have seen a {plan_cache_mode} plan at all"
        )
        assert scans[name] == 0 and scans[shared] == 0, (
            f"a filtered search selecting {selectivity} must keep the exact plan, and "
            f"an approximate index cannot answer it: {scans[name]} executions of "
            f"{len(driven)} scanned the partial index and {scans[shared]} the shared one "
            f"under {plan_cache_mode} ({scans})"
        )
        assert answers == exact * 2, (
            f"a filtered search selecting {selectivity} did not return the exact answer "
            f"under {plan_cache_mode} (scans {scans})"
        )

    _assert_the_unrestricted_search_still_reaches_the_index(engine, query_vectors, name, shared)


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
#
# This block is half of a symmetric pair and only applies to an *unrestricted*
# search. Its mirror prices the approximate index out for a search the caller
# restricted; sections 8 and 12 are that side.
# ---------------------------------------------------------------------------


def _exact_answers(engine, kb_id, vectors, *, schema=SCHEMA, **kwargs) -> list[list[str]]:
    """The exact top-20 for each query vector: the same SQL with no index at all."""
    sql, params = _capture_search_sql(engine, kb_id, vectors[0], schema=schema, **kwargs)
    answers = []
    with Session(engine) as session:
        for vector in vectors:
            session.execute(text("SET LOCAL enable_indexscan = off"))
            rows = session.execute(text(sql), {**params, "embedding": _literal(vector)}).all()
            session.rollback()
            answers.append([str(row[0]) for row in rows])
    return answers


def _drive_and_collect(
    engine,
    kb_id,
    vectors,
    *counted,
    plan_cache_mode=None,
    prepared_out=None,
    schema=SCHEMA,
    **kwargs,
):
    """``vector_search`` for real, once per vector, on one connection.

    Returns ``(scans, answers)``: the scans each of ``counted`` served over the
    run, and what each search returned. Same connection discipline as
    ``_drive_searches`` -- an engine of its own, closed before the counters are
    read. ``plan_cache_mode`` is left alone by default, so the default run is
    PostgreSQL deciding for itself when to go generic.

    ``prepared_out``, if given, is extended with the names of the prepared
    statements the driver ended up with. A spec about what a *cached* plan does
    needs that as a negative control: a run in which psycopg never prepared
    anything would pass while proving nothing.
    """
    before = _idx_scans(engine, *counted, schema=schema)
    probe = create_engine(_dsn())
    connection = probe.connect()
    answers = []
    try:
        with Session(bind=connection) as session:
            if plan_cache_mode is not None:
                session.execute(text(f"SET plan_cache_mode = '{plan_cache_mode}'"))
            store = _ChunkStore(db_session=session, knowledge_base_id=kb_id, schema=schema)
            for vector in vectors:
                items = asyncio.run(store.vector_search(embedding=list(vector), top_k=20, **kwargs))
                answers.append([item.item_id for item in items])
            if prepared_out is not None:
                prepared_out.extend(
                    row[0]
                    for row in session.execute(
                        text("SELECT name, statement FROM pg_prepared_statements")
                    ).all()
                    if "e.knowledge_base_id" in row[1]
                )
            session.commit()
    finally:
        connection.close()
        probe.dispose()
    after = _idx_scans(engine, *counted, schema=schema)
    return {name: after[name] - before[name] for name in counted}, answers


def _assert_the_unrestricted_search_still_reaches_the_index(engine, vectors, name, shared):
    """The positive control every "stays off the index" spec needs beside it.

    A restricted search taking zero scans of the partial index is also what a
    build with this feature removed would do, and a suite that only asserted the
    negative would pass against one. Twice through ``vector_search`` unrestricted,
    on the knowledge base whose index is built: both executions on the partial
    index, neither on the shared one, and a full page each time.

    Two executions rather than twelve because the steering is decided per search;
    what needs many executions is a claim about a *cached* plan, and that is the
    spec's own business, not this control's.

    What it proves and what it does not, measured rather than assumed. Taking the
    embeddings-side knowledge base id out of the statement, or not building the
    index, fails the specs that call this -- the first through
    ``_capture_search_sql``, the second through ``_build_big_index``. Removing only
    the ``enable_sort`` steering does *not*: at 384 dimensions the planner chooses
    the partial index for an unrestricted search unaided, which is the premise the
    module docstring opens with, so there is nothing here to observe. That half is
    pinned deterministically in the unit tier, in
    ``tests/unit/test_vector_search_plan_shape.py``.
    """
    scans, answers = _drive_and_collect(engine, KB_BIG, list(vectors)[:2], name, shared)
    assert scans[name] == 2 and scans[shared] == 0, (
        "the unrestricted search is the one this feature exists for, and it must still "
        f"be steered onto the knowledge base's own partial index: {scans}"
    )
    assert all(len(answer) == TOP_K for answer in answers), [len(a) for a in answers]


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


@pytest.mark.parametrize("was", ["on", "off"])
def test_a_vector_search_leaves_enable_indexscan_as_it_found_it(
    engine, schema, settings, query_vectors, was
):
    """The twin of the spec above, for the setting the restricted block writes.

    ``_insisting_on_an_exact_search`` prices out every ordered index scan in the
    statement, and it is the block a *restricted* search gets -- so it is the one
    ``hybrid_search``'s keyword leg meets when the caller passed a filter. A
    keyword ranking wants its index scans back, and so does every other statement
    on that pooled connection.

    Two claims, because the restore can fail in two ways and only one of them is
    visible inside the search's own transaction:

    - the value the caller had is the value that comes back, which a restore
      hardcoding ``on`` gets wrong for ``was="off"``;
    - and it comes back *transaction-locally*, which is what the third argument to
      ``set_config`` decides. A session-scoped restore reads correctly inside the
      transaction and then survives the commit, so the next transaction on the
      same connection -- the next checkout of that pool entry -- starts with the
      caller's old value instead of the server's. That is the leak this module
      already found once on ``enable_sort``, so it is read back on the same
      connection after the commit rather than assumed not to have happened.
    """
    _build_big_index(engine, settings)
    probe = create_engine(_dsn())
    connection = probe.connect()
    try:
        with Session(bind=connection) as session:
            at_checkout = session.execute(
                text("SELECT current_setting('enable_indexscan')")
            ).scalar()
            session.execute(text(f"SET LOCAL enable_indexscan = {was}"))
            store = _ChunkStore(db_session=session, knowledge_base_id=KB_BIG, schema=SCHEMA)
            items = asyncio.run(
                store.vector_search(
                    embedding=list(query_vectors[0]),
                    top_k=20,
                    filter_metadata=FILTER_ONE_IN_FIVE,
                )
            )
            assert items, "the restricted search under test returned nothing"
            after = session.execute(text("SELECT current_setting('enable_indexscan')")).scalar()
            session.commit()
        with Session(bind=connection) as session:
            next_checkout = session.execute(
                text("SELECT current_setting('enable_indexscan')")
            ).scalar()
            session.rollback()
    finally:
        connection.close()
        probe.dispose()
    assert after == was, (
        f"vector_search left enable_indexscan at {after!r} in a transaction that had "
        f"it at {was!r}; the keyword leg of a hybrid search runs next on this session"
    )
    assert next_checkout == at_checkout, (
        f"the restore outlived its transaction: this connection was checked out with "
        f"enable_indexscan={at_checkout!r} and the next transaction on it starts at "
        f"{next_checkout!r}, so the restore was made session-scoped rather than local"
    )


class _ReadingTheSettingBeforeTheSearch:
    """A real session that reads a GUC back in the transaction the search runs in.

    The value has to be read there and not afterwards: the settings the two
    steering blocks write are transaction-local and are put back before
    ``vector_search`` returns, so a read after the call sees the restore rather
    than the search.
    """

    def __init__(self, session, setting: str):
        self._session = session
        self._read = f"SELECT current_setting('{setting}', true)"
        self.readings: list[str | None] = []

    def execute(self, clause, params=None):
        if "ORDER BY" in clause.text:
            value = self._session.execute(text(self._read)).scalar()
            self.readings.append(None if value is None else str(value))
        return self._session.execute(clause, params)

    def __getattr__(self, name):
        return getattr(self._session, name)


def test_the_first_vector_search_on_a_connection_runs_at_the_raised_ef_search(
    engine, schema, settings, query_vectors
):
    """The recall the store asks for is the recall the search gets, read back live.

    pgvector registers its GUCs in ``_PG_init``, and ``_PG_init`` runs on the
    first *use of the vector type* on a backend -- not at ``CREATE EXTENSION``,
    and not at connection start, because the library is not preloaded. Until then
    ``current_setting('hnsw.ef_search', true)`` is NULL. Two things make that the
    case to pin rather than a curiosity:

    - the store's own first statement, ``SET LOCAL hnsw.iterative_scan``, does
      **not** load the library -- an unrecognised ``prefix.name`` is accepted as a
      placeholder -- so it cannot be relied on to have made the GUC real by the
      time the probe reads it;
    - NULL is not a value to defer to. ``set_config`` on an unloaded pgvector GUC
      creates a placeholder and the value survives ``_PG_init``: demonstrated on
      this server, ``set_config('hnsw.ef_search','123',true)`` before any vector
      operation and ``current_setting`` reads 123 back after one.

    So a fresh connection is exactly where the raise matters and exactly where it
    is easiest to skip, and at pgvector's default 40 instead of
    ``PER_KB_HNSW_EF_SEARCH`` the search is not wrong, it is less complete:
    measured on real embeddings at 12,000 rows, recall 0.915 against 0.973. One
    such search per connection per pool lifetime, and the first search on a
    connection is also the one most likely to be cold.

    The assertion before the search is the control that keeps this spec honest: if
    the connection has already done vector work by the time the store probes, the
    NULL path is not the one under test and a green result would say nothing about
    it. The unit tier cannot stand in for this one -- its capture hands the probe
    a value, so the NULL branch is the one branch a real fresh connection takes
    and the one no fake takes.
    """
    _build_big_index(engine, settings)
    probe = create_engine(_dsn())
    connection = probe.connect()
    try:
        with Session(bind=connection) as session:
            assert (
                session.execute(text("SELECT current_setting('hnsw.ef_search', true)")).scalar()
                is None
            ), (
                "this connection has already used the vector type, so pgvector's GUCs "
                "are registered on it and the fresh-connection path this spec is about "
                "cannot happen here"
            )
            watcher = _ReadingTheSettingBeforeTheSearch(session, "hnsw.ef_search")
            store = _ChunkStore(db_session=watcher, knowledge_base_id=KB_BIG, schema=SCHEMA)
            items = asyncio.run(store.vector_search(embedding=list(query_vectors[0]), top_k=20))
            assert items, "the search under test returned nothing"
            session.commit()
    finally:
        connection.close()
        probe.dispose()
    assert watcher.readings == [str(bvs.PER_KB_HNSW_EF_SEARCH)], (
        f"the first vector search on a fresh connection ran at hnsw.ef_search "
        f"{watcher.readings} instead of {bvs.PER_KB_HNSW_EF_SEARCH}; a NULL reading "
        "means the raise was skipped because pgvector's GUC was not registered yet, "
        "which is the state every pooled connection is in before its first vector "
        "operation"
    )


def test_a_filter_matching_no_row_still_answers_nothing(engine, schema, settings, query_vectors):
    """A restriction that matches nothing: an empty answer, not a wrong one.

    This was written when a restricted search was driven onto the index, where a
    predicate matching no row turned the search into a walk of the index looking
    for rows that are not there -- measured at 1536 dimensions on the indexed 30 %
    knowledge base, a filter matching no row went from 3.0 ms to 38.3 ms and a
    ``source_ids`` matching no row from 2.4 ms to 36.7 ms. A restricted search is
    given the exact plan now, so that cost is gone with it; the behaviour it was
    written to pin is not width- or plan-dependent and is worth keeping either
    way. All four restriction shapes, because an empty answer is the one case
    where a wrong one is easiest to produce and hardest to notice.
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


def test_a_restriction_below_top_k_returns_every_row_that_matches(
    engine, schema, settings, query_vectors
):
    """Fewer rows match than ``top_k``, and all of them come back.

    This spec used to pin the forced ordered index scan returning all twelve, and
    the after-the-fact re-run that made it do so. Both are gone: a restricted
    search is given the exact plan instead, so there is no approximate scan to
    starve and nothing to repair afterwards. The claim that was always the point
    survives it -- a row the caller named and did not get is a wrong answer, not a
    recall trade -- and it is now structural rather than rescued.

    So the assertions are inverted rather than deleted: zero scans of this
    knowledge base's partial index and zero of the shared one, because neither can
    promise the rows the caller named, and all twelve rows regardless. Both plan
    cache modes, because a cached plan is what a pooled connection settles on, and
    twice through the query vectors so the second pass is a prepared one. The
    unrestricted search beside it is the positive control.

    Its twin above the limit is
    ``test_a_restriction_the_caller_named_still_answers_exactly``, which names 200
    rows and asks for the nearest 20 of them. The two are not the same spec: this
    one is the corner a row count catches -- a page that comes back short -- and
    that one is the corner it cannot, a full page of rows that do not belong.
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
    assert len(wanted) < TOP_K, (len(wanted), TOP_K)

    driven = list(query_vectors) * 2
    shared = f"idx_ai_embeddings_hnsw_{DIMS}"
    for plan_cache_mode in (None, "force_generic_plan"):
        prepared: list[str] = []
        scans, answers = _drive_and_collect(
            engine,
            KB_BIG,
            driven,
            name,
            shared,
            plan_cache_mode=plan_cache_mode,
            prepared_out=prepared,
            item_ids=set(wanted),
        )
        assert prepared, (
            "the driver never prepared the restricted search statement, so the second "
            "pass proves nothing about a cached plan"
        )
        assert scans[name] == 0 and scans[shared] == 0, (
            f"a search restricted to named rows must keep the exact plan: "
            f"{scans[name]} of {len(driven)} executions were steered onto the partial "
            f"index and {scans[shared]} onto the shared one under plan_cache_mode "
            f"{plan_cache_mode or 'auto'} ({scans})"
        )
        for got in answers:
            assert sorted(got) == sorted(wanted), (
                f"a search restricted to {len(wanted)} named rows under a LIMIT of "
                f"{TOP_K} dropped rows that matched: {len(got)} of {len(wanted)} "
                f"(plan_cache_mode {plan_cache_mode or 'auto'}, scans {scans})"
            )

    _assert_the_unrestricted_search_still_reaches_the_index(engine, query_vectors, name, shared)


# ---------------------------------------------------------------------------
# 12. A restriction with more matching rows than ``top_k``
#
# The safety net this section was written against is gone, and what replaced it is
# why. It re-ran a search whose answer came back short, on the signal
# ``len(items) < top_k``: that catches starvation *below* the limit and cannot
# catch starvation *at* it. A restriction that still matches more rows than
# ``top_k`` fills the LIMIT with whatever the ordered scan happened to reach, and
# a full page of the wrong rows is indistinguishable from a complete answer by row
# count alone -- so there is nothing to trigger on, and where the plan the planner
# prefers is also the approximate index the re-run replays the same scan and
# returns the same answer. A restricted search is now given the exact plan
# instead, and exactness is structural rather than rescued.
#
# So these specs sit on the other side of the limit, and they assert what the
# caller gets rather than how the store got it. Neither looks at which branch of
# the store ran or at whether a particular setting was applied; the index counters
# they read are read to assert that a restricted search was not steered onto an
# approximate index, which is behaviour and not mechanism, and each carries the
# unrestricted search beside it as the positive control -- a suite that only
# asserted the negative would pass against a build with the feature removed.
#
# Both restrictions are held to the same bar, because the caller's claim is the
# same in both: a row you named, or a source you narrowed to, is not a recall
# trade. ``item_ids`` is the one the old re-run's docstring argued for and only
# covered below the limit; ``source_ids`` is the shape a caller reaches for most
# often, one document of a knowledge base, and it is a *set* restriction, so the
# two together cover both ways a caller can narrow a search that still leaves many
# more rows than the limit. The metadata filter is the third way and it is pinned
# in section 8, at both ends of its selectivity range.
#
# Both are driven twice through the query vectors, because the failure this is
# about survives plan caching: a plan a pooled connection settled on is re-used
# for the life of the pool entry.
#
# What exact means is measured on the same statement with no index at all, once
# per query vector, so the expectation is PostgreSQL's own and not a number
# written down here.
# ---------------------------------------------------------------------------

# Ten times ``top_k``, and 1.7% of the indexed knowledge base: selective enough
# that an ordered scan of its index has to work to fill a page of 20, and far
# enough above the limit that a short answer is not the failure mode under test.
NAMED_ITEMS = 200

# The ``top_k`` every driving helper in this module searches with. Not a knob:
# the specs below are about the relation between the number of matching rows and
# the limit, so both ends of it have to be named in one place.
TOP_K = 20


@pytest.fixture(scope="module")
def named_items(engine, fixture_schema):
    """``NAMED_ITEMS`` chunk ids of the indexed knowledge base, by id order.

    By id and not by distance, so the named set is unrelated to the query
    vectors: the rows a search has to find are scattered through the index
    rather than sitting in one neighbourhood of it.
    """
    with engine.connect() as conn:
        ids = [
            str(row[0])
            for row in conn.execute(
                text(
                    f"SELECT id FROM {SCHEMA}.chunks WHERE knowledge_base_id = :kb "
                    f"ORDER BY id LIMIT {NAMED_ITEMS}"
                ),
                {"kb": KB_BIG},
            ).all()
        ]
        conn.rollback()
    assert len(ids) == NAMED_ITEMS, len(ids)
    return ids


def _matching_rows(engine, **kwargs) -> int:
    """How many rows of KB_BIG the restriction really matches, from the database.

    The restriction is spelled out here in SQL rather than taken from the store,
    so the margin the specs below assert -- many times ``top_k`` rows matching -- is
    the question the caller asked and not the store's own answer to it.
    """
    where = [f"knowledge_base_id = '{KB_BIG}'"]
    params: dict = {}
    if "item_ids" in kwargs:
        where.append("id = ANY(CAST(:named AS uuid[]))")
        params["named"] = "{" + ",".join(kwargs["item_ids"]) + "}"
    if "source_ids" in kwargs:
        where.append("source_id = ANY(CAST(:srcs AS uuid[]))")
        params["srcs"] = "{" + ",".join(kwargs["source_ids"]) + "}"
    if "filter_metadata" in kwargs:
        where.append("meta @> CAST(:meta AS jsonb)")
        params["meta"] = json.dumps(kwargs["filter_metadata"])
    with engine.connect() as conn:
        count = conn.execute(
            text(f"SELECT count(*) FROM {SCHEMA}.chunks WHERE " + " AND ".join(where)), params
        ).scalar()
        conn.rollback()
    return int(count)


@pytest.mark.parametrize("restriction", ["item_ids", "source_ids"])
def test_a_restriction_above_top_k_answers_exactly(
    engine, schema, settings, query_vectors, named_items, restriction
):
    """The nearest ``top_k`` rows that match, from many more than ``top_k`` matches.

    200 named rows and a source of 300, against a ``top_k`` of 20: ten and fifteen
    times the limit, so the answer is a full page and nothing about it looks wrong
    from the outside. That is the whole case. The exact answer is a full page too,
    which is asserted for the same reason -- if it were not, a wrong answer could
    pass as a short one.

    One spec per restriction rather than one for both, because the two are
    estimated differently -- named ids against the primary key, a source against
    an ordinary equality -- and a change that repairs one can leave the other
    exactly as it was.

    Measured before the store insisted on an exact plan for a restricted search:
    a full page of 20 with 15 of them not among the nearest 20 named rows, and 15
    of 20 for the source, on 12 of 12 executions of the partial index.
    """
    restrictions = {
        "item_ids": {"item_ids": set(named_items)},
        "source_ids": {"source_ids": [SOURCE_B]},
    }
    kwargs = restrictions[restriction]
    matching = _matching_rows(engine, **kwargs)
    assert matching >= 10 * TOP_K, (
        f"{restriction} matches {matching} rows; this spec is only about the case "
        f"where more than top_k ({TOP_K}) do -- the one a row count cannot detect -- "
        "and it keeps a margin of ten times rather than one row"
    )

    name = _build_big_index(engine, settings)
    shared = f"idx_ai_embeddings_hnsw_{DIMS}"
    exact = _exact_answers(engine, KB_BIG, query_vectors, **kwargs)
    assert all(len(answer) == TOP_K for answer in exact), [len(a) for a in exact]

    driven = list(query_vectors) * 2
    for plan_cache_mode in (None, "force_generic_plan"):
        scans, answers = _drive_and_collect(
            engine, KB_BIG, driven, name, shared, plan_cache_mode=plan_cache_mode, **kwargs
        )
        assert scans[name] == 0 and scans[shared] == 0, (
            f"a search restricted by {restriction} must keep the exact plan; neither "
            f"index can promise the rows the caller asked for: {scans}"
        )
        for i, (got, wanted) in enumerate(zip(answers, exact * 2)):
            assert got == wanted, (
                f"a search restricted to {matching} matching rows with top_k {TOP_K} "
                f"returned a full page of {len(got)} that is not the {TOP_K} nearest "
                f"of them: {len(set(got) - set(wanted))} of them do not belong "
                f"(query vector {i % len(query_vectors)}, plan_cache_mode "
                f"{plan_cache_mode or 'auto'}, scans {scans})"
            )

    _assert_the_unrestricted_search_still_reaches_the_index(engine, query_vectors, name, shared)
