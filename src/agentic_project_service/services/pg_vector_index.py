"""One partial HNSW index per large knowledge base, on ``ai.embeddings``.

``ai.embeddings`` carries a single HNSW index per embedding dimension, over
every knowledge base in the project. A KB-scoped vector search therefore walks
a graph built over rows it will throw away, and stops as soon as the join has
produced ``top_k`` rows of the knowledge base it wanted -- which is both slow
(the graph is as large as the whole table) and approximate over the wrong
population (the true top-k *within* one knowledge base needs far more
candidates than ``hnsw.ef_search`` emits globally).

A partial index fixes both, for the knowledge bases big enough to be worth one:

    CREATE INDEX CONCURRENTLY hnsw_kb_<hex>_<dims> ON ai.embeddings
      USING hnsw ((embedding::vector(<dims>)) vector_cosine_ops)
      WHERE knowledge_base_id = '<kb>' AND dims = <dims>;

Measured on a 56,000-row fixture at 1536 dimensions whose largest knowledge
base held 12,000 rows (21% of the table), ``shared_buffers`` 128 MB,
``hnsw.ef_search`` 40, top-20: the shared index answered in 2.3 ms warm with
recall@20 = 0.825 against an exact scan (one of six query vectors returned none
of the true top-20); the partial index answered in 1.7 ms warm with recall
1.000. On a larger fixture (600,000 rows, a 73,290-row knowledge base) the
same change moved warm p50 from 182 ms to 1.4 ms and cold p50 from 824 ms to
225 ms, and the partial index was 573 MB and built in 15-50 s depending on
``maintenance_work_mem``.

Three properties shape this module:

* **The planner only matches a partial index from a predicate on the indexed
  relation.** Today's ``vector_search`` filters ``knowledge_base_id`` on the
  *item* table and joins to ``ai.embeddings``; nothing constrains
  ``e.knowledge_base_id``, and the planner does not reason through the join to
  get there. So the index is useless without the matching predicate in
  ``base_vector_store`` -- verified by ``EXPLAIN``: with the partial index
  present and the old query shape, the plan still picks the shared index.
* **A generic plan cannot match it either.** The predicate's ``knowledge_base_id``
  is a literal in the index definition, so a plan built for an unknown
  parameter cannot prove it. ``base_vector_store`` interpolates the (validated)
  KB id into the SQL for exactly this reason; see
  ``base_vector_store.kb_sql_literal``. Measured: with the id bound as a
  parameter, psycopg's ``prepare_threshold`` prepares the statement and the
  11th execution onwards falls back to a bitmap scan plus an exact sort --
  1.1 ms to 57 ms, correct but 50x slower.
* **``CREATE INDEX CONCURRENTLY`` cannot run inside a transaction** and can
  leave an ``INVALID`` index behind when it fails, so the build lives here, in
  an out-of-band task, with the same invalid-index repair the pg_search BM25
  path uses -- not in ``base_vector_store.ensure_embedding_index``, which runs
  inside the indexing transaction on purpose.

Nothing here touches the shared per-dimension index. Replacing it with a
residual one (``WHERE dims = N AND knowledge_base_id NOT IN (...)``) is what
turns the transitional write cost of maintaining two graphs into a large write
*gain*, but it may only be done once the query change is deployed everywhere:
with a residual index in place the old query shape matches no HNSW index at all
and degenerates to a sequential scan. That is deliberately a follow-up.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import text

from ..db import AI_SCHEMA
from .pg_bm25_index import (
    first_error_line,
    is_transient_db_error,
    partition_build_lock_sql,
    partition_build_unlock_sql,
)
from .settings_registry import SETTINGS_REGISTRY, get_setting

logger = logging.getLogger(__name__)

# Index names are ``hnsw_kb_<32 hex>_<dims>``: 8 + 32 + 1 + 4 = 45 bytes at
# most, inside Postgres' 63-byte identifier limit, and free of the dashes a
# UUID's canonical form would need quoting for.
INDEX_NAME_PREFIX = "hnsw_kb_"

# pgvector's own bound on a vector's dimensions for an HNSW index; the same
# range ``ensure_embedding_index`` enforces.
MIN_DIMS = 1
MAX_DIMS = 8192

# Ceiling on how many of these a single project may hold. The planner opens and
# locks *every* index of a relation while planning any query on it, so partial
# indexes on one table are not free at scale: measured on a Postgres 15 with
# default ``max_locks_per_transaction``, 500 of them cost 1.9 ms of planning and
# 513 locks per backend (comfortable), while 5,000 cost 25 ms of planning and
# made the seventh concurrent search fail with "out of shared memory". 200 is
# far below the point where either matters, and at the default threshold it
# already means 10 million indexed rows in one project. A project that reaches
# it keeps the shared index for the rest of its knowledge bases and says so at
# WARNING, rather than quietly degrading every query on the table.
MAX_PER_KB_INDEXES = 200

# Bounded counts never read more than this many rows past the threshold, so the
# cost of deciding is bounded by the threshold rather than by the size of the
# knowledge base.
_COUNT_HEADROOM = 1


class PerKbVectorIndexBuildInProgress(RuntimeError):
    """Another caller holds the build lock for this index."""


# ---------------------------------------------------------------------------
# Naming and DDL
# ---------------------------------------------------------------------------


def _validated_kb_id(knowledge_base_id: Any) -> str:
    """Canonical UUID string for a KB id, or ValueError.

    The one gate between a caller's string and a SQL literal.
    """
    try:
        return str(uuid.UUID(str(knowledge_base_id)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"knowledge_base_id is not a UUID: {knowledge_base_id!r}") from exc


def _validated_dims(dims: Any) -> int:
    """An embedding dimension safe to interpolate into a type modifier.

    PostgreSQL does not accept a type modifier from a parameter, so ``dims``
    reaches SQL as a literal in both the index expression and its predicate.
    """
    try:
        value = int(dims)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dims is not an integer: {dims!r}") from exc
    if not (MIN_DIMS <= value <= MAX_DIMS):
        raise ValueError(f"dims must be between {MIN_DIMS} and {MAX_DIMS}, got {value}")
    return value


def per_kb_index_name(knowledge_base_id: Any, dims: Any) -> str:
    """Deterministic index name for one knowledge base and dimension."""
    kb_hex = uuid.UUID(_validated_kb_id(knowledge_base_id)).hex
    return f"{INDEX_NAME_PREFIX}{kb_hex}_{_validated_dims(dims)}"


def per_kb_index_ddl(knowledge_base_id: Any, dims: Any) -> str:
    """CREATE statement for one knowledge base's partial HNSW index.

    ``dims`` is in the predicate as well as the expression. Without it, a
    knowledge base holding rows of two different dimensions would fail the
    build outright -- ``embedding::vector(1536)`` raises on a 768-dimension
    value -- and the index would cover rows the query's own ``e.dims``
    predicate excludes.

    The operator class and the cast match the shared per-dimension index
    exactly. Both are load-bearing: the index is on an *expression*, so a query
    whose ``ORDER BY`` does not contain the same ``::vector(N)`` cast matches no
    HNSW index at all.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    n = _validated_dims(dims)
    return (
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {per_kb_index_name(kb_id, n)} "
        f'ON "{AI_SCHEMA}".embeddings '
        f"USING hnsw ((embedding::vector({n})) vector_cosine_ops) "
        f"WHERE knowledge_base_id = '{kb_id}' AND dims = {n}"
    )


def per_kb_index_drop_ddl(knowledge_base_id: Any, dims: Any) -> str:
    """DROP statement for one knowledge base's partial HNSW index."""
    name = per_kb_index_name(knowledge_base_id, dims)
    return f'DROP INDEX CONCURRENTLY IF EXISTS "{AI_SCHEMA}".{name}'


def index_lock_relation(knowledge_base_id: Any, dims: Any) -> str:
    """Advisory-lock subject for building or dropping one of these indexes."""
    return f"{AI_SCHEMA}.{per_kb_index_name(knowledge_base_id, dims)}"


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def read_overrides(conn, *keys: str) -> dict[str, int]:
    """These settings' stored overrides, read on a caller-supplied connection.

    ``get_setting`` reads through ``db.session``, and a read of
    ``ai.project_settings`` that fails -- the table does not exist yet, which is
    exactly the state at the start-up sweep's first run on a new database --
    leaves that session's transaction aborted and takes the caller's next
    statement with it. The start-up sweep shares a transaction with the boot
    migrations, so it reads its settings here instead, on its own connection,
    and falls back to the registry defaults if that read fails too.
    """
    try:
        rows = conn.execute(
            text(f'SELECT key, value FROM "{AI_SCHEMA}".project_settings WHERE key = ANY(:keys)'),
            {"keys": list(keys)},
        ).all()
        conn.rollback()
    except Exception as exc:
        conn.rollback()
        logger.debug(
            "Could not read the vector index settings (%s); using the defaults",
            first_error_line(exc),
        )
        return {}
    found: dict[str, int] = {}
    for key, value in rows:
        try:
            found[key] = int(value)
        except (TypeError, ValueError):
            logger.warning("Bad stored value for %s=%r, using the default", key, value)
    return found


def _clamped_setting(key: str, overrides: dict[str, int] | None = None) -> int:
    """Read an int setting, clamped to the registry's own bounds.

    ``get_setting`` coerces a stored override but does not range-check it --
    that happens on the settings PUT path only -- so a row written before a
    bound was tightened, or by hand, would otherwise be used as-is.

    ``overrides``, when given, replaces the ``get_setting`` read entirely: a
    caller that cannot use ``db.session`` passes what ``read_overrides`` found.
    """
    defn = SETTINGS_REGISTRY[key]
    if overrides is None:
        value = int(get_setting(key))
    else:
        value = int(overrides.get(key, defn.default))
    clamped = value
    if defn.min is not None:
        clamped = max(clamped, int(defn.min))
    if defn.max is not None:
        clamped = min(clamped, int(defn.max))
    if clamped != value:
        logger.warning(
            "%s=%d is outside the allowed range %s-%s; using %d instead",
            key,
            value,
            defn.min,
            defn.max,
            clamped,
        )
    return clamped


def thresholds(overrides: dict[str, int] | None = None) -> tuple[int, int]:
    """``(build_at_or_above, drop_below)`` rows, with the hysteresis enforced.

    Two thresholds rather than one so a knowledge base sitting at the boundary
    cannot have its index built and dropped over and over: each build costs a
    full index build and each drop throws it away. A stored pair that inverts
    the hysteresis (drop at or above build) would do exactly that, so the drop
    threshold is pulled down to half the build threshold and the override is
    reported.

    The crossover where a partial index starts beating an exact scan was
    bracketed, not bisected: at 2,000 rows the planner does not use a partial
    index at all (an exact bitmap scan and sort is genuinely cheaper, and
    exact), and at 73,290 rows the partial index is two orders of magnitude
    faster. The defaults sit deliberately at the conservative end of that
    bracket, and a project that measures its own crossover can move them.

    ``overrides`` is for a caller that cannot read settings through
    ``db.session``; see ``read_overrides``.
    """
    build_at = _clamped_setting("VECTOR_PER_KB_INDEX_MIN_ROWS", overrides)
    drop_below = _clamped_setting("VECTOR_PER_KB_INDEX_DROP_ROWS", overrides)
    if drop_below >= build_at:
        corrected = max(0, build_at // 2)
        logger.warning(
            "VECTOR_PER_KB_INDEX_DROP_ROWS=%d is not below VECTOR_PER_KB_INDEX_MIN_ROWS=%d, "
            "which would build and drop the same index repeatedly; using %d instead",
            drop_below,
            build_at,
            corrected,
        )
        drop_below = corrected
    return build_at, drop_below


def maintenance_work_mem_mb() -> int:
    """How much memory the builder gives its own session for the build.

    A build that does not fit spills, and says so
    (``NOTICE: hnsw graph no longer fits into maintenance_work_mem``): measured
    3.3x slower at 64 MB than at 1 GB for a 73,290-row, 1536-dimension index.
    Raising it is therefore worth real time -- but it is memory the database
    pod has to have, and the smallest project pods have 512 MiB in total, so
    the default is deliberately modest and the bound is what stops an override
    from turning a slow build into an OOM kill.
    """
    return _clamped_setting("VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB")


# ---------------------------------------------------------------------------
# Reading the current state
# ---------------------------------------------------------------------------


def existing_per_kb_indexes(conn, knowledge_base_id: Any) -> dict[int, bool]:
    """``{dims: is_valid}`` for this knowledge base's partial HNSW indexes.

    Read by name rather than by parsing predicates: the name is derived from
    the KB's UUID and the dimension, so the catalog lookup is an equality match
    on ``pg_class.relname`` and cannot mistake another KB's index for this
    one's.
    """
    kb_hex = uuid.UUID(_validated_kb_id(knowledge_base_id)).hex
    prefix = f"{INDEX_NAME_PREFIX}{kb_hex}_"
    rows = conn.execute(
        text(
            "SELECT c.relname, i.indisvalid FROM pg_class c "
            "JOIN pg_index i ON i.indexrelid = c.oid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = :schema AND c.relkind = 'i' AND c.relname LIKE :prefix"
        ),
        {"schema": AI_SCHEMA, "prefix": prefix + "%"},
    ).all()
    found: dict[int, bool] = {}
    for relname, valid in rows:
        suffix = relname[len(prefix) :]
        if suffix.isdigit():
            found[int(suffix)] = bool(valid)
    return found


def per_kb_index_count(conn) -> int:
    """How many of these indexes ``ai.embeddings`` already carries."""
    return (
        conn.execute(
            text(
                "SELECT count(*) FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = :schema AND c.relkind = 'i' AND c.relname LIKE :prefix"
            ),
            {"schema": AI_SCHEMA, "prefix": INDEX_NAME_PREFIX + "%"},
        ).scalar()
        or 0
    )


def bounded_row_count(conn, knowledge_base_id: Any, dims: Any, cap: int) -> int:
    """Rows this knowledge base has at this dimension, counted no further than ``cap``.

    The decision only needs to know which side of a threshold the count falls
    on, so the count stops there. Without the bound this would read every
    embedding of the largest knowledge base in the project on every source that
    finishes indexing.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    return int(
        conn.execute(
            text(
                "SELECT count(*) FROM (SELECT 1 FROM "
                f'"{AI_SCHEMA}".embeddings '
                "WHERE knowledge_base_id = CAST(:kb AS uuid) AND dims = :dims "
                "LIMIT :cap) s"
            ),
            {"kb": kb_id, "dims": _validated_dims(dims), "cap": max(1, int(cap))},
        ).scalar()
        or 0
    )


def candidate_dims(conn, knowledge_base_id: Any, cap: int) -> list[int]:
    """Dimensions this knowledge base has enough rows at to be worth looking at.

    Also bounded: the subquery reads at most ``cap`` rows, so a knowledge base
    that holds two dimensions splits that budget between them and is measured
    conservatively -- it may look smaller than it is and keep the shared index,
    which is the safe direction. In practice a knowledge base holds one
    embedding model at a time, so it has one dimension.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    rows = conn.execute(
        text(
            "SELECT dims FROM (SELECT dims FROM "
            f'"{AI_SCHEMA}".embeddings '
            "WHERE knowledge_base_id = CAST(:kb AS uuid) LIMIT :cap) s "
            "GROUP BY dims ORDER BY count(*) DESC"
        ),
        {"kb": kb_id, "cap": max(1, int(cap))},
    ).all()
    return [int(r[0]) for r in rows if MIN_DIMS <= int(r[0]) <= MAX_DIMS]


def index_action(conn, knowledge_base_id: Any) -> str | None:
    """``"build"``, ``"drop"`` or None -- is there anything to reconcile here?

    Cheap enough for the indexing path to call once per source: one catalog
    lookup plus one bounded count per dimension in play. Never raises for a
    knowledge base that has no embeddings at all.
    """
    build_at, drop_below = thresholds()
    existing = existing_per_kb_indexes(conn, knowledge_base_id)
    if any(not valid for valid in existing.values()):
        # An INVALID index is repaired whatever the row count says: it serves no
        # query and Postgres maintains it on every write.
        return "build"
    cap = build_at + _COUNT_HEADROOM
    for dims in sorted(set(candidate_dims(conn, knowledge_base_id, cap)) | set(existing)):
        rows = bounded_row_count(conn, knowledge_base_id, dims, cap)
        if dims in existing:
            if rows < drop_below:
                return "drop"
        elif rows >= build_at:
            return "build"
    return None


# ---------------------------------------------------------------------------
# Connections and locks
# ---------------------------------------------------------------------------


def _autocommit_connection(engine):
    """A connection outside any transaction: CONCURRENTLY refuses one."""
    return engine.connect().execution_options(isolation_level="AUTOCOMMIT")


def _engine(engine=None):
    if engine is not None:
        return engine
    from ..db import db

    return db.engine


def _try_lock(conn, relation: str) -> bool:
    return bool(conn.execute(text(partition_build_lock_sql()), {"relation": relation}).scalar())


def _release_lock(conn, relation: str) -> None:
    """Give the session-scoped lock back, or throw the session away.

    The lock outlives a transaction on purpose (a concurrent build is not one),
    so a pooled connection handed back still holding it would make every later
    build of this index skip. Ending the backend releases it for certain.
    """
    try:
        conn.execute(text(partition_build_unlock_sql()), {"relation": relation})
    except Exception:
        logger.warning(
            "Could not release the advisory lock on %s; discarding the connection so the "
            "lock cannot outlive it",
            relation,
            exc_info=True,
        )
        try:
            conn.invalidate()
        except Exception:
            logger.debug("Could not invalidate the connection either", exc_info=True)


def _reset_session_setting(conn, name: str) -> None:
    """RESET one session setting without hiding the error a build already raised.

    A server that restarts mid-build ends the session, and SQLAlchemy then
    refuses any further statement on the connection. Raised from a ``finally``,
    that would replace a lost connection -- which the task retries -- with an
    error it does not.
    """
    try:
        conn.execute(text(f"RESET {name}"))
    except Exception as exc:
        logger.warning(
            "Could not reset %s after a build (%s); discarding the connection",
            name,
            first_error_line(exc),
        )
        try:
            conn.invalidate()
        except Exception:
            logger.debug("Could not invalidate the connection either", exc_info=True)


def _build_in_progress(conn) -> bool:
    """Is another backend running CREATE INDEX or REINDEX on ai.embeddings?"""
    row = conn.execute(
        text(
            "SELECT 1 FROM pg_stat_progress_create_index "
            "WHERE relid = to_regclass(:relation) AND pid <> pg_backend_pid()"
        ),
        {"relation": f'"{AI_SCHEMA}".embeddings'},
    ).first()
    return row is not None


# ---------------------------------------------------------------------------
# Build and drop
# ---------------------------------------------------------------------------


def _create_index(conn, kb_id: str, dims: int) -> None:
    """Build one index online, with room to do it in memory.

    Session-level rather than ``SET LOCAL``: this connection is in AUTOCOMMIT
    because ``CREATE INDEX CONCURRENTLY`` refuses a transaction block, and
    ``SET LOCAL`` outside a transaction affects nothing at all. Both settings
    are put back before the connection can return to the pool -- a pooled
    connection left with no statement timeout, or with a large
    ``maintenance_work_mem``, would carry them into unrelated work.
    """
    conn.execute(text("SET statement_timeout = 0"))
    conn.execute(text(f"SET maintenance_work_mem = '{maintenance_work_mem_mb()}MB'"))
    try:
        conn.execute(text(per_kb_index_ddl(kb_id, dims)))
    finally:
        _reset_session_setting(conn, "maintenance_work_mem")
        _reset_session_setting(conn, "statement_timeout")


def _repair_invalid(conn, kb_id: str, dims: int) -> bool:
    """Drop an INVALID index left behind by a failed concurrent build.

    ``CREATE INDEX CONCURRENTLY`` that is cancelled, killed or fails leaves the
    index in place and marked invalid: it answers no query, Postgres still
    maintains it on every write, and ``IF NOT EXISTS`` makes a re-run a no-op,
    so without this the knowledge base would never get a usable index. Skipped
    while a build is actually running on the table, which is the other reason
    an index can be invalid.
    """
    if _build_in_progress(conn):
        return False
    logger.warning(
        "Partial HNSW index %s.%s is INVALID and no build is running on %s.embeddings (an "
        "earlier CREATE INDEX CONCURRENTLY failed or was cancelled); dropping and rebuilding it",
        AI_SCHEMA,
        per_kb_index_name(kb_id, dims),
        AI_SCHEMA,
    )
    conn.execute(text(per_kb_index_drop_ddl(kb_id, dims)))
    return True


def ensure_per_kb_vector_index(knowledge_base_id: Any, engine=None, on_progress=None) -> dict:
    """Give this knowledge base a partial HNSW index per dimension it is big enough for.

    Idempotent, and a no-op whenever a partial index is not the right answer:
    too few rows, one already there and valid, or the project already at
    ``MAX_PER_KB_INDEXES``. Below ``VECTOR_PER_KB_INDEX_DROP_ROWS`` an index
    that exists is dropped, so a knowledge base that shrinks -- sources deleted,
    a reindex to a different embedding model -- does not keep paying for one.
    An ``INVALID`` index is dropped and rebuilt.

    ``on_progress(status)`` is called with ``"building"`` before each build and
    ``"dropping"`` before each drop.

    Returns a dict with ``status`` in ``ready`` (nothing left to do),
    ``building`` (another caller holds this index's lock), or ``skipped`` with
    a ``reason``, plus the names built and dropped.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    engine = _engine(engine)

    def progress(status: str) -> None:
        if on_progress is not None:
            on_progress(status)

    build_at, drop_below = thresholds()
    cap = build_at + _COUNT_HEADROOM
    built: list[str] = []
    dropped: list[str] = []
    repaired: list[str] = []

    with _autocommit_connection(engine) as conn:
        existing = existing_per_kb_indexes(conn, kb_id)
        dims_in_play = sorted(set(candidate_dims(conn, kb_id, cap)) | set(existing))
        if not dims_in_play:
            return {"status": "ready", "reason": "no_embeddings", "built": [], "dropped": []}

        for dims in dims_in_play:
            name = per_kb_index_name(kb_id, dims)
            lock = index_lock_relation(kb_id, dims)
            if not _try_lock(conn, lock):
                # Whoever holds it is building or dropping this very index.
                return {"status": "building", "index": name, "built": built, "dropped": dropped}
            try:
                # Re-read under the lock: another caller may have finished
                # between the survey above and this point.
                valid = existing_per_kb_indexes(conn, kb_id).get(dims)
                if valid is False and _repair_invalid(conn, kb_id, dims):
                    repaired.append(name)
                    valid = None

                rows = bounded_row_count(conn, kb_id, dims, cap)
                if valid is True:
                    if rows < drop_below:
                        progress("dropping")
                        logger.info(
                            "Dropping partial HNSW index %s.%s: knowledge base %s now has "
                            "fewer than %d rows at %d dimensions",
                            AI_SCHEMA,
                            name,
                            kb_id,
                            drop_below,
                            dims,
                        )
                        conn.execute(text(per_kb_index_drop_ddl(kb_id, dims)))
                        dropped.append(name)
                    continue

                if rows < build_at:
                    continue
                total = per_kb_index_count(conn)
                if total >= MAX_PER_KB_INDEXES:
                    logger.warning(
                        "Not building partial HNSW index %s.%s: %s.embeddings already carries "
                        "%d per-knowledge-base indexes (the cap is %d), and every index on a "
                        "relation is opened and locked while planning any query on it. This "
                        "knowledge base keeps the shared index",
                        AI_SCHEMA,
                        name,
                        AI_SCHEMA,
                        total,
                        MAX_PER_KB_INDEXES,
                    )
                    return {
                        "status": "skipped",
                        "reason": "index_cap_reached",
                        "index_count": total,
                        "built": built,
                        "dropped": dropped,
                    }
                progress("building")
                logger.info(
                    "Building partial HNSW index %s.%s for knowledge base %s (%s%d rows at %d "
                    "dimensions, threshold %d)",
                    AI_SCHEMA,
                    name,
                    kb_id,
                    "at least " if rows >= cap else "",
                    rows,
                    dims,
                    build_at,
                )
                _create_index(conn, kb_id, dims)
                built.append(name)
            finally:
                _release_lock(conn, lock)

    outcome: dict = {"status": "ready", "built": built, "dropped": dropped}
    if repaired:
        outcome["repaired_invalid_indexes"] = repaired
    return outcome


def drop_per_kb_vector_indexes(knowledge_base_id: Any, engine=None) -> dict:
    """Drop every partial HNSW index this knowledge base owns.

    What a deleted knowledge base needs: the row is gone, so nothing will ever
    reconcile the index again, and Postgres would keep maintaining an index
    named after a knowledge base that no longer exists. Every dimension it has
    one at, not only the current one: the embedding model may have changed
    since.

    A transient database error is re-raised so the task retries rather than
    reporting a drop that did not happen.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    engine = _engine(engine)

    dropped: list[str] = []
    failed: list[str] = []
    with _autocommit_connection(engine) as conn:
        for dims in sorted(existing_per_kb_indexes(conn, kb_id)):
            name = per_kb_index_name(kb_id, dims)
            lock = index_lock_relation(kb_id, dims)
            if not _try_lock(conn, lock):
                raise PerKbVectorIndexBuildInProgress(
                    f"{AI_SCHEMA}.{name} is being built or dropped by another caller"
                )
            try:
                conn.execute(text(per_kb_index_drop_ddl(kb_id, dims)))
                dropped.append(name)
            except Exception as exc:
                if is_transient_db_error(exc):
                    raise
                failed.append(name)
                logger.warning(
                    "Could not drop partial HNSW index %s.%s", AI_SCHEMA, name, exc_info=True
                )
            finally:
                _release_lock(conn, lock)

    if failed:
        return {"status": "partial", "indexes": dropped, "failed_indexes": failed}
    return {"status": "dropped", "indexes": dropped}


# ---------------------------------------------------------------------------
# Start-up sweep
# ---------------------------------------------------------------------------

# Upper bound on the one grouped count the start-up sweep runs. It reads
# ``ai.embeddings`` once, which on a large project is seconds rather than
# milliseconds; past this it is abandoned, because a start-up must not wait on
# it. Nothing is lost by abandoning it: the next source to finish indexing in
# each knowledge base dispatches the same reconcile.
SWEEP_TIMEOUT_MS = 30_000

_THRESHOLD_KEYS = ("VECTOR_PER_KB_INDEX_MIN_ROWS", "VECTOR_PER_KB_INDEX_DROP_ROWS")


def kbs_needing_a_per_kb_index(engine=None) -> list[str]:
    """Knowledge bases whose partial HNSW indexes are out of step, for the start-up sweep.

    Three cases, in one pass: a knowledge base at or above the build threshold
    with no index, one below the drop threshold that has one, and one whose
    index is ``INVALID``. The last two are read from the catalog, which is
    cheap; the first needs the grouped count, which is not, so it runs under
    ``SWEEP_TIMEOUT_MS`` and an abandoned count leaves the catalog cases to be
    dispatched on their own.

    Never raises, and never touches ``db.session``: this runs inside the boot's
    migration transaction, where a failing statement would abort the boot's own
    work (see ``read_overrides``).
    """
    engine = _engine(engine)
    needing: dict[str, None] = {}

    try:
        with engine.connect() as conn:
            build_at, drop_below = thresholds(read_overrides(conn, *_THRESHOLD_KEYS))
            rows = conn.execute(
                text(
                    "SELECT c.relname, i.indisvalid FROM pg_class c "
                    "JOIN pg_index i ON i.indexrelid = c.oid "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = :schema AND c.relkind = 'i' AND c.relname LIKE :prefix"
                ),
                {"schema": AI_SCHEMA, "prefix": INDEX_NAME_PREFIX + "%"},
            ).all()
            indexed: dict[str, list[int]] = {}
            for relname, valid in rows:
                body = relname[len(INDEX_NAME_PREFIX) :]
                kb_hex, _, dims_part = body.rpartition("_")
                if len(kb_hex) != 32 or not dims_part.isdigit():
                    continue
                try:
                    kb_id = str(uuid.UUID(hex=kb_hex))
                except ValueError:
                    continue
                indexed.setdefault(kb_id, []).append(int(dims_part))
                if not valid:
                    needing[kb_id] = None
            conn.rollback()
    except Exception:
        logger.warning(
            "Could not read the per-knowledge-base HNSW indexes at start-up", exc_info=True
        )
        return []

    try:
        with engine.connect() as conn:
            conn.execute(
                text("SELECT set_config('statement_timeout', :ms, true)"),
                {"ms": str(SWEEP_TIMEOUT_MS)},
            )
            counted = conn.execute(
                text(
                    "SELECT knowledge_base_id::text, dims, count(*) FROM "
                    f'"{AI_SCHEMA}".embeddings GROUP BY 1, 2'
                )
            ).all()
            conn.rollback()
    except Exception as exc:
        logger.warning(
            "Could not count embeddings per knowledge base at start-up (%s); the knowledge "
            "bases that need an index are picked up as their sources finish indexing",
            first_error_line(exc),
        )
        counted = []

    for kb_id, dims, count in counted:
        has_index = int(dims) in indexed.get(kb_id, ())
        if (not has_index and count >= build_at) or (has_index and count < drop_below):
            needing[kb_id] = None
    # A knowledge base whose rows are all gone still has its index, and the
    # grouped count above cannot see it (no rows, no group).
    for kb_id in indexed:
        if not any(r[0] == kb_id for r in counted):
            needing[kb_id] = None
    return list(needing)
