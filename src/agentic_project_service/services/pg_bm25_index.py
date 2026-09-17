"""Postgres-native BM25 keyword search via the ParadeDB ``pg_search`` extension.

One ``USING bm25`` index per knowledge base, so a keyword query is answered by
Tantivy inside Postgres instead of by the bm25s file index or the tsvector
fallback. Everything here degrades: when the extension is not installed, or
this KB has no ready index, callers keep today's behaviour.

Three properties of the extension shape this module, and the first is why the
item tables are partitioned at all:

* **One bm25 index per relation.** So a per-knowledge-base index needs a
  per-knowledge-base relation: ``chunks``, ``full_documents`` and
  ``graph_index_nodes`` are partitioned ``BY LIST (knowledge_base_id)`` with a
  DEFAULT partition for every KB that has not been given one of its own, and
  each partition carries its own index -- with its own stemmer. A partial index
  on the shared table cannot do this: building a second one makes the first
  unscorable.
* **A scored query must name a relation that carries the index.** A partitioned
  parent never does; ``SELECT ... FROM ai.chunks WHERE ... ||| ...`` is refused
  with "`chunks` does not contain a `USING bm25` index" even with a
  ``knowledge_base_id`` predicate that would prune to exactly one indexed
  partition. So the search path names the partition, and the KB id reaches SQL
  only as part of a relation name built from a ``uuid.UUID()``-validated value.
* **A missing index is not always a loud failure.** Readiness is therefore
  checked per knowledge base before this path is used, and a KB with no
  partition or no valid index keeps the existing keyword path.

Requirement: on Postgres 15 and 16, a pg_search build that contains ParadeDB's
fix paradedb/paradedb#6211 (no 0.25.x release does). Indexes here are built
with ``CREATE INDEX CONCURRENTLY`` while the item table keeps being written to,
and without the fix that build fails inside pg_search -- XX000 "buffer ... is
not owned by resource owner", leaving the index INVALID -- or crashes the
server, ending every session on it. Postgres 17 and 18 are not affected.
Nothing pg_search reports tells a build with the fix from one without, so no
bm25 index is built -- and no rows are moved for one -- unless the server shows
it (``concurrent_build_safety``): the image sets ``powabase.pg_search_cic_safe
= on``, the server is Postgres 17+, pg_search is 0.26.0+, or the
``BM25_PG_SEARCH_CONCURRENT_BUILD_SAFE`` setting vouches for it. Otherwise the
knowledge base keeps the keyword path it has and reports ``unavailable``.
``ci/pg_search/Dockerfile`` builds 0.25.9 with the fix and sets the marker.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Iterable
from typing import Any

from sqlalchemy import text

from ..db import AI_SCHEMA
from .sparse_retrieval import STRATEGY_TO_BM25_ITEM_TABLE

logger = logging.getLogger(__name__)

# Item tables that can carry a BM25 index, and the text each one indexes.
# A plain column is indexed directly; an expression needs an ``alias=`` in its
# tokenizer cast (pg_search: "indexed expression requires a tokenizer cast with
# an alias").
_TEXT_EXPRESSIONS: dict[str, str] = {
    "chunks": "text",
    "full_documents": "summary",
    "graph_index_nodes": "(COALESCE({a}title, '') || ' ' || COALESCE({a}text, ''))",
}

BM25_ITEM_TABLES: frozenset[str] = frozenset(_TEXT_EXPRESSIONS)

# Item tables partitioned ``BY LIST (knowledge_base_id)``, so that each
# knowledge base owns a relation of its own and can therefore own a bm25 index
# of its own. ``doc2json_documents`` is deliberately absent from both sets: it
# has no BM25 item table and stays on the tsvector keyword fallback.
PARTITIONED_ITEM_TABLES: frozenset[str] = frozenset(
    {"chunks", "full_documents", "graph_index_nodes"}
)
# A bm25 index needs a relation of its own per knowledge base, so every table
# that can carry one is partitioned (pinned by a unit test).

# How long to wait for another caller's partition build on the same item table
# before giving up and reporting a retryable outcome.
PARTITION_BUILD_LOCK_WAIT_SECONDS = 30.0
_PARTITION_BUILD_LOCK_POLL_SECONDS = 0.25

# Ceiling on how long the move waits for each lock it needs. While it waits for
# SHARE on the parent, writers queue behind it (readers do not), so this is also
# the worst-case extra write stall a *failed* attempt costs. A timeout rolls the
# whole move back -- nothing has moved -- and the task retries later.
MOVE_LOCK_TIMEOUT_MS = 5_000

# How long the move keeps trying for ACCESS EXCLUSIVE on the DEFAULT partition
# (for its temporary check, for the ATTACH, and to drop the check). It does not
# simply wait for that lock in Postgres' queue, for two reasons. A queued ACCESS
# EXCLUSIVE request makes every new reader of DEFAULT queue behind it for as
# long as it waits. And a transaction that read DEFAULT and then writes through
# the parent is already waiting on the move's SHARE lock, so a queued request
# closes a lock cycle that Postgres breaks by aborting whichever side runs its
# deadlock check first -- the application's, whenever it began waiting less than
# ``deadlock_timeout`` before the request (a short lock_timeout on the request
# does not avoid that: with 2 s the application still lost every time).
#
# So the tries are ``NOWAIT``, with a short backoff, plus at most one queued try
# of ``DEFAULT_EXCLUSIVE_QUEUED_TRY_MS`` (capped at half of the server's
# ``deadlock_timeout``) -- without it, overlapping short reads that never leave
# DEFAULT free starve the move. That try is skipped while any holder of a lock
# on DEFAULT is itself waiting for a lock, the one state in which queueing could
# close a cycle; a transaction that only starts waiting after the request starts
# its deadlock check after the queued try has already timed out. A move that
# cannot get the lock in time rolls back (SQLSTATE 55P03) and its task retries.
# A queued request by anyone else for a conflicting lock on DEFAULT makes every
# ``NOWAIT`` try fail for as long as it waits, so the move then has only its
# single queued try.
#
# The cycle above involves the connection holding the parent lock, so it bears
# on the ATTACH and on dropping the check. The check itself is added from a
# second connection that holds no lock: its queued try cannot close a cycle
# through the move that Postgres could see.
#
# One cycle is left open, and has been reproduced. A session already blocked on
# the move (a write through the parent) runs its one-time deadlock check
# ``deadlock_timeout`` after it began waiting. If a holder of DEFAULT starts
# waiting on that session -- for a row it has locked, say -- after the holder
# check but within the queued try's window, the check finds move -> holder ->
# session -> move, and Postgres aborts the session whose check found it -- the
# one already blocked on the move -- with SQLSTATE 40P01. No row is lost: the
# aborted transaction rolls back whole, ``index_source`` requeues, and an API
# writer receives the error. (``index_source`` itself no longer takes part:
# the move gate keeps its transactions out of the move.) Closing it precisely would mean skipping the queued try whenever
# any session waits on the move, and under steady writes one always does --
# for the whole copy of a large move -- so large moves would starve.
#
# The bound is short because each of these steps but the last cleanup runs
# while writers are held off the parent: it is what a failed attempt costs
# them on top of the copy. Measured on the production schema (a 40 000-row
# knowledge base over a 2.5 million-row DEFAULT, 4 attempts per value): with
# 6-8 threads of 50-150 ms reads of DEFAULT every attempt succeeded at 0.4 s,
# as at 2.0 s; at 0.3 s the queued try no longer fits and every attempt
# failed. A long reader arriving mid-move cost writers 2.4-2.6 s per failed
# attempt at 2.0 s and 1.0-1.3 s at 0.4 s (the rest is the copy and the
# VALIDATE scan of DEFAULT).
DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS = 0.4
DEFAULT_EXCLUSIVE_QUEUED_TRY_MS = 200
_EXCLUSIVE_LOCK_QUEUED_TRY_AFTER_SECONDS = 0.1
_EXCLUSIVE_LOCK_FIRST_SLEEP_SECONDS = 0.01
_EXCLUSIVE_LOCK_MAX_SLEEP_SECONDS = 0.25

# Before it takes the parent lock, the move checks that DEFAULT can be locked
# at all: ``NOWAIT`` tries of ACCESS EXCLUSIVE, each released at once, for up to
# this long. If every try is refused and a transaction that has been open for
# longer than ``BM25_MOVE_LONG_HOLDER_SECONDS`` (a project setting, 5 s by
# default) holds DEFAULT -- a reader idle in its transaction, typically -- it
# would refuse every lock try the move makes on DEFAULT too, so the move gives
# up there (SQLSTATE 55P03) instead of holding writers off the parent for those
# tries first. Ordinary requests and overlapping short transactions refuse the
# tries as well, but are younger than that, so the move goes ahead and its
# bounded tries wait them out. Only a heuristic: a long reader can still arrive
# after the probe, and the bounded tries remain the guard for that.
DEFAULT_PREFLIGHT_WAIT_SECONDS = 0.25

# How long a failed move keeps trying to drop the temporary check it put on
# DEFAULT. Whatever broke the move is usually a reader still holding DEFAULT,
# and until the check is gone every write of that knowledge base routed to
# DEFAULT fails its check. A check that outlives this is cleared by the next
# indexing write it refuses, the next ``ensure_bm25_index`` on the item table,
# or at start-up.
MOVE_CHECK_CLEANUP_WAIT_SECONDS = 5.0

# Alias pg_search records for an indexed expression, so the query's expression
# can be matched back to the indexed one.
_EXPRESSION_ALIAS = "bm25_text"

# Snowball stemmers pg_search accepts, intersected with the ts_language values
# this service already allows. Determined by creating one index per language
# against pg_search 0.25.9: the rest ("simple", armenian, basque, catalan,
# hindi, indonesian, irish, lithuanian, nepali, serbian, yiddish) are rejected
# with `unknown stemmer: <name>`, so those KBs get an unstemmed index rather
# than no index.
PG_SEARCH_STEMMERS: frozenset[str] = frozenset(
    {
        "arabic",
        "danish",
        "dutch",
        "english",
        "finnish",
        "french",
        "german",
        "greek",
        "hungarian",
        "italian",
        "norwegian",
        "portuguese",
        "romanian",
        "russian",
        "spanish",
        "swedish",
        "tamil",
        "turkish",
    }
)

# Upper bound on the text handed to the match operator. Keyword queries carry
# conversation context, and nothing upstream bounds their length.
MAX_BM25_QUERY_CHARS = 8192

# Availability and readiness are read on the search path, so they are cached.
# Short enough that a freshly built index is picked up on its own.
_CACHE_TTL_SECONDS = 30.0
_READY_CACHE_MAX_ENTRIES = 1024

_extension_cache: tuple[float, bool] | None = None
_ready_cache: dict[tuple[str, str], tuple[float, bool]] = {}


# ---------------------------------------------------------------------------
# Naming, strategy and language mapping
# ---------------------------------------------------------------------------


def _validated_kb_id(knowledge_base_id: Any) -> str:
    """Canonical UUID string for a KB id, or ValueError.

    The one gate between a caller's string and a SQL literal.
    """
    try:
        return str(uuid.UUID(str(knowledge_base_id)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"knowledge_base_id is not a UUID: {knowledge_base_id!r}") from exc


def _validated_item_table(item_table: str) -> str:
    if item_table not in BM25_ITEM_TABLES:
        raise ValueError(
            f"Unknown BM25 item table {item_table!r}; expected one of {sorted(BM25_ITEM_TABLES)}"
        )
    return item_table


def _validated_partitioned_table(item_table: str) -> str:
    if item_table not in PARTITIONED_ITEM_TABLES:
        raise ValueError(
            f"{item_table!r} is not partitioned by knowledge base; expected one of "
            f"{sorted(PARTITIONED_ITEM_TABLES)}"
        )
    return item_table


def partition_name(knowledge_base_id: Any, item_table: str) -> str:
    """Relation name of one knowledge base's partition of an item table.

    Derived from the UUID's hex form, so the name is a bare lowercase
    identifier that needs no quoting and fits Postgres' 63-byte limit for
    every partitioned table (the longest is
    ``graph_index_nodes_kb_`` + 32 hex characters = 53 bytes).
    """
    kb_hex = uuid.UUID(_validated_kb_id(knowledge_base_id)).hex
    return f"{_validated_partitioned_table(item_table)}_kb_{kb_hex}"


def default_partition_name(item_table: str) -> str:
    """Relation name of the DEFAULT partition -- every KB without one of its own."""
    return f"{_validated_partitioned_table(item_table)}_default"


def bm25_index_name(knowledge_base_id: str, item_table: str) -> str:
    """Deterministic index name for one KB and item table.

    The hex form of the UUID keeps the name inside Postgres' 63-byte identifier
    limit for every item table, and free of the dashes that would need quoting.
    """
    kb_hex = uuid.UUID(_validated_kb_id(knowledge_base_id)).hex
    return f"bm25_{_validated_item_table(item_table)}_{kb_hex}"


def pg_bm25_item_table(strategy: str | None) -> str | None:
    """Item table a KB's indexing strategy keeps BM25-searchable text in."""
    item_table = STRATEGY_TO_BM25_ITEM_TABLE.get(strategy or "")
    if item_table is None or item_table not in BM25_ITEM_TABLES:
        return None
    return item_table


def pg_search_stemmer(ts_language: str | None) -> str | None:
    """pg_search stemmer for a KB's ``retrieval_config.ts_language``.

    None means "index this KB unstemmed": either the language is unknown, or
    pg_search has no Snowball stemmer for it.
    """
    if not ts_language:
        return None
    candidate = str(ts_language).strip().lower()
    return candidate if candidate in PG_SEARCH_STEMMERS else None


# ---------------------------------------------------------------------------
# Tokenizer cast and text expression
# ---------------------------------------------------------------------------


def _is_expression_index(item_table: str) -> bool:
    return "{a}" in _TEXT_EXPRESSIONS[_validated_item_table(item_table)]


def bm25_tokenizer_cast(item_table: str, ts_language: str | None) -> str:
    """The ``::pdb.simple(...)`` cast that tokenizes this table's text."""
    args: list[str] = []
    if _is_expression_index(item_table):
        args.append(f"'alias={_EXPRESSION_ALIAS}'")
    stemmer = pg_search_stemmer(ts_language)
    if stemmer:
        args.append(f"'stemmer={stemmer}'")
    if not args:
        return "::pdb.simple"
    return f"::pdb.simple({', '.join(args)})"


def bm25_text_expression(item_table: str, alias: str | None = None) -> str:
    """The indexed text for this table, optionally qualified by a table alias."""
    template = _TEXT_EXPRESSIONS[_validated_item_table(item_table)]
    prefix = f"{alias}." if alias else ""
    return template.format(a=prefix) if "{a}" in template else f"{prefix}{template}"


def indexdef_matches_tokenizer(indexdef: str | None, tokenizer_cast: str) -> bool:
    """Does an existing ``pg_get_indexdef`` still use this tokenizer?

    A ``ts_language`` change rewrites the cast, and a BM25 index tokenized for
    the wrong language has to be rebuilt rather than reused. The bare
    tokenizer needs the negative lookahead: ``::pdb.simple`` is a prefix of
    ``::pdb.simple('stemmer=german')``, so a plain substring test would call a
    German index a match for an unstemmed one.
    """
    if not indexdef:
        return False
    if tokenizer_cast.endswith(")"):
        return tokenizer_cast in indexdef
    return re.search(re.escape(tokenizer_cast) + r"(?!\s*\()", indexdef) is not None


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------


def bm25_index_ddl(knowledge_base_id: str, item_table: str, ts_language: str | None) -> str:
    """CREATE statement for one KB's BM25 index, on that KB's partition.

    No ``WHERE`` predicate: the partition's LIST bound already restricts the
    index to this knowledge base's rows, and a non-partial index is the shape
    that plans as ``Custom Scan (ParadeDB Base Scan) / TopKScanExecState``.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    name = bm25_index_name(kb_id, item_table)
    partition = partition_name(kb_id, item_table)
    expression = bm25_text_expression(item_table)
    cast = bm25_tokenizer_cast(item_table, ts_language)
    return (
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} "
        f'ON "{AI_SCHEMA}".{partition} '
        f"USING bm25 (id, ({expression}{cast}), source_id, meta) "
        f"WITH (key_field = 'id')"
    )


def bm25_drop_ddl(knowledge_base_id: str, item_table: str) -> str:
    """DROP statement for one KB's BM25 index."""
    name = bm25_index_name(knowledge_base_id, item_table)
    return f'DROP INDEX CONCURRENTLY IF EXISTS "{AI_SCHEMA}".{name}'


# ---------------------------------------------------------------------------
# Partition DDL
# ---------------------------------------------------------------------------


def _qualified(relation: str) -> str:
    return f'"{AI_SCHEMA}".{relation}'


def partition_create_ddl(knowledge_base_id: Any, item_table: str) -> str:
    """Create the KB's partition as an unattached, bare clone of DEFAULT.

    Cloning the DEFAULT partition rather than the partitioned parent is what
    carries the column defaults, NOT NULLs and CHECK constraints across. No
    index and no foreign key: the move copies the knowledge base's rows into a
    bare heap, because maintaining those row by row during the copy is most of
    what a move used to cost (see ``create_partition``). The indexes and keys
    are recreated from DEFAULT's catalog entries -- the ones correctness rests
    on inside the move, the rest after it.
    """
    partition = partition_name(knowledge_base_id, item_table)
    return (
        f"CREATE TABLE IF NOT EXISTS {_qualified(partition)} "
        f"(LIKE {_qualified(default_partition_name(item_table))} "
        "INCLUDING DEFAULTS INCLUDING CONSTRAINTS "
        "INCLUDING STORAGE INCLUDING COMMENTS)"
    )


def partition_attach_ddl(knowledge_base_id: Any, item_table: str) -> str:
    """Attach the KB's partition, bound to exactly that one knowledge base."""
    kb_id = _validated_kb_id(knowledge_base_id)
    partition = partition_name(kb_id, item_table)
    return (
        f"ALTER TABLE {_qualified(_validated_partitioned_table(item_table))} "
        f"ATTACH PARTITION {_qualified(partition)} FOR VALUES IN ('{kb_id}')"
    )


def partition_detach_ddl(knowledge_base_id: Any, item_table: str) -> str:
    """Detach the KB's partition.

    Not ``CONCURRENTLY``: Postgres refuses a concurrent detach while a DEFAULT
    partition exists, and this schema always has one.
    """
    partition = partition_name(knowledge_base_id, item_table)
    return (
        f"ALTER TABLE {_qualified(_validated_partitioned_table(item_table))} "
        f"DETACH PARTITION {_qualified(partition)}"
    )


def partition_drop_ddl(knowledge_base_id: Any, item_table: str) -> str:
    """Drop the KB's (already detached) partition."""
    return f"DROP TABLE IF EXISTS {_qualified(partition_name(knowledge_base_id, item_table))}"


def partition_check_ddl(knowledge_base_id: Any, item_table: str) -> str:
    """A CHECK on the clone that already implies the partition bound.

    This is what keeps the ATTACH short. Postgres skips ATTACH's validation
    scan of the table being attached when that table carries a constraint
    implying the partition constraint; without it, ATTACH reads every row of the
    partition while holding ACCESS EXCLUSIVE -- which is the one window this
    design exists to keep small.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    partition = partition_name(kb_id, item_table)
    return (
        f"ALTER TABLE {_qualified(partition)} "
        f"ADD CONSTRAINT {partition}_kb_check "
        f"CHECK (knowledge_base_id = '{kb_id}')"
    )


# Prefix of the temporary CHECK a move puts on the DEFAULT partition. Named per
# knowledge base (``bm25_move_`` + 32 hex = 42 bytes) so a leftover from a
# crashed move says whose it was.
_DEFAULT_MOVE_CHECK_PREFIX = "bm25_move_"


def default_move_check_name(knowledge_base_id: Any) -> str:
    return f"{_DEFAULT_MOVE_CHECK_PREFIX}{uuid.UUID(_validated_kb_id(knowledge_base_id)).hex}"


def default_move_check_add_ddl(knowledge_base_id: Any, item_table: str) -> str:
    """``CHECK (knowledge_base_id <> kb) NOT VALID`` on the DEFAULT partition.

    The other half of keeping ATTACH short. Besides the new partition, ATTACH
    has to prove that DEFAULT holds no row of the new bound, and it does that
    with a full scan of DEFAULT under ACCESS EXCLUSIVE -- blocking every reader
    and writer of the item table for a time that grows with the whole DEFAULT
    partition, not with this knowledge base -- unless a *validated* constraint
    on DEFAULT already implies it. This is that constraint.

    NOT VALID, so adding it is a catalog change that reads no rows; but from
    the moment it commits Postgres enforces it on new writes, so an INSERT or
    UPDATE of this knowledge base's rows routed to DEFAULT would fail with
    SQLSTATE 23514. ``create_partition`` therefore adds it only while it already
    holds writers off the parent.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    return (
        f"ALTER TABLE {_qualified(default_partition_name(item_table))} "
        f"ADD CONSTRAINT {default_move_check_name(kb_id)} "
        f"CHECK (knowledge_base_id <> '{kb_id}') NOT VALID"
    )


def default_move_check_validate_ddl(knowledge_base_id: Any, item_table: str) -> str:
    """Validate the DEFAULT check: one scan of DEFAULT, not blocking readers.

    VALIDATE CONSTRAINT takes SHARE UPDATE EXCLUSIVE, which does not conflict
    with the ACCESS SHARE readers hold. It can only succeed once this knowledge
    base's rows have left DEFAULT, so it runs inside the move, after the
    DELETE, where the move's SHARE locks are already holding writers off.
    """
    return (
        f"ALTER TABLE {_qualified(default_partition_name(item_table))} "
        f"VALIDATE CONSTRAINT {default_move_check_name(knowledge_base_id)}"
    )


def default_move_check_drop_ddl(item_table: str, constraint_name: str) -> str:
    """Drop a temporary DEFAULT check by name (its own or a crashed move's)."""
    if not re.fullmatch(rf"{_DEFAULT_MOVE_CHECK_PREFIX}[0-9a-f]{{32}}", constraint_name):
        raise ValueError(f"not a move check constraint name: {constraint_name!r}")
    return (
        f"ALTER TABLE {_qualified(default_partition_name(item_table))} "
        f"DROP CONSTRAINT IF EXISTS {constraint_name}"
    )


class PartitionBuildInProgress(RuntimeError):
    """Another caller is already building a partition of this item table."""


def partition_build_lock_relation(item_table: str) -> str:
    """The advisory lock's subject: one item table, schema-qualified."""
    return f"{AI_SCHEMA}.{_validated_partitioned_table(item_table)}"


def partition_build_lock_sql() -> str:
    """Try to claim the right to move rows out of this table's DEFAULT partition.

    Two concurrent moves out of one DEFAULT partition cannot both finish: each
    takes SHARE on it and then needs the ACCESS EXCLUSIVE its own ATTACH takes,
    which the other's SHARE refuses. Before the ACCESS EXCLUSIVE tries were
    made ``NOWAIT`` this was observed on a real project as ``deadlock
    detected`` with both tasks failing; now both would merely give up.
    Serialising the moves per item table avoids the conflict altogether.

    Session-scoped, not transaction-scoped: preparing the clone and moving the
    rows are separate transactions, so a ``pg_advisory_xact_lock`` would be
    released after the first one. The
    session scope also means closing the connection releases it, so a crashed
    build does not leave the table locked. The *try* form is what bounds the
    wait -- a caller that cannot get it reports a retryable outcome.
    """
    return "SELECT pg_try_advisory_lock(hashtextextended(:relation, 0))"


def partition_build_unlock_sql() -> str:
    return "SELECT pg_advisory_unlock(hashtextextended(:relation, 0))"


# How long a move waits, queued, for the move gate (below) before giving up.
MOVE_GATE_WAIT_SECONDS = 30.0


def move_gate_relation(item_table: str) -> str:
    """The move gate's subject for one item table.

    Two advisory locks guard a move, on distinct keys:

    * the **build lock** (``partition_build_lock_relation``: ``ai.chunks``),
      session-scoped and only ever *tried*, serialises moves -- and partition
      drops -- on one item table with each other;
    * the **move gate** (this: ``ai.chunks#move``) serialises a move with
      indexing. Indexing takes it shared, transaction-scoped
      (``hold_move_gate_shared``), for each item table a transaction writes,
      before it first reads or writes that table; the move -- and the empty
      knowledge base's attach -- takes it exclusively (``_acquire_move_gate``)
      before it checks DEFAULT or takes any table lock, and holds it until it
      commits. So an indexing transaction on that table runs entirely before
      a move or entirely after it, and never holds DEFAULT into the move's
      lock tries. A queued exclusive request makes later shared requests for
      the same table queue behind it, so a stream of overlapping indexing
      transactions cannot starve a move once it is waiting.

    One gate per item table, so a transaction that writes only one table waits
    only for moves on that table: a chunks move neither waits for a graph_index
    run nor stalls it. The accepted trade-off is on ``graph_index_nodes``
    itself: a graph_index run writes its nodes in one transaction that stays
    open through its LLM enrichment and embedding stages (minutes), so a move on
    that table gives up (SQLSTATE 55P03 after ``MOVE_GATE_WAIT_SECONDS``) while
    a graph_index source is indexing -- stalling graph_index indexing for that
    wait -- and its task retries later. Nothing else waits for it: a re-index
    clears its old rows one item table per transaction, taking only that
    table's gate and only when the table holds rows of the source
    (``tasks.indexing._clear_source_item_rows``). A transaction that did take
    the gates of several tables would have to take them in one call, in name
    order (``move_gate_shared_sql``) -- and would queue for every one of them
    while holding the first.

    Other writers (API writes, enrichment, graph updates) do not take the gate;
    the parent SHARE lock and the ``NOWAIT`` tries remain their protection.
    """
    return f"{partition_build_lock_relation(item_table)}#move"


def _gate_tables(item_tables: str | Iterable[str]) -> list[str]:
    tables = [item_tables] if isinstance(item_tables, str) else list(item_tables)
    validated = sorted({_validated_partitioned_table(t) for t in tables})
    if not validated:
        raise ValueError("the move gate needs at least one item table")
    return validated


def move_gate_shared_sql(item_tables: str | Iterable[str]) -> tuple[str, dict]:
    """The statement that takes the move gates of these item tables shared, in name order."""
    return (
        "SELECT pg_advisory_xact_lock_shared(hashtextextended(r.relation, 0)) "
        "FROM unnest(CAST(:relations AS text[])) WITH ORDINALITY AS r(relation, position) "
        "ORDER BY r.position",
        {"relations": [move_gate_relation(t) for t in _gate_tables(item_tables)]},
    )


def hold_move_gate_shared(session, item_tables: str | Iterable[str]) -> None:
    """Take the move gate of each item table shared until the caller's transaction ends.

    For indexing: call it before the first statement of a transaction that
    reads or writes one of ``item_tables`` -- only the tables it writes, so it
    waits for no move on any other (taking a gate again in the same
    transaction is harmless). It waits while a move holds a gate -- holding no
    lock on these tables meanwhile -- and raises whatever the wait raises (a
    lock timeout the caller set, typically), which indexing requeues as a lock
    conflict.
    """
    sql, params = move_gate_shared_sql(item_tables)
    session.execute(text(sql), params)


def partition_lock_parent_ddl(item_table: str) -> str:
    """Hold writers off the partitioned parent for the length of a move.

    This is the lock that makes the move safe for UPDATE and DELETE, not just
    INSERT. A statement through the parent fixes its list of partitions when
    it is planned, and it plans after taking its own lock on the parent. A
    writer that waited only on the DEFAULT partition's lock would already have
    planned against the pre-ATTACH list, and once the move committed it would
    find the rows gone from DEFAULT and match nothing -- verified against
    Postgres 15 as updates silently reporting 0 rows and deleted rows coming
    back. Waiting here instead, it plans after the ATTACH is visible.

    ``ONLY``: without it, LOCK on a partitioned table recurses into every
    partition. SHARE conflicts with the ROW EXCLUSIVE every INSERT, UPDATE and
    DELETE takes, and not with the ACCESS SHARE of a reader, so readers do not
    block on this lock. (They can still wait on the move's ACCESS EXCLUSIVE
    steps on DEFAULT -- see ``create_partition``.) The cost is that writes to
    *every* knowledge base on this item table wait for the move, including
    those that already have partitions of their own.
    """
    return f"LOCK TABLE ONLY {_qualified(_validated_partitioned_table(item_table))} IN SHARE MODE"


def partition_lock_default_ddl(item_table: str) -> str:
    """Hold writers off the DEFAULT partition for the length of a move.

    Belt and braces behind the parent lock, for a writer that names the DEFAULT
    partition directly: a row of this knowledge base inserted there after the
    move but before the ATTACH would make Postgres refuse the attach ("updated
    partition constraint for default partition would be violated by some
    row"). Upgrading to the ACCESS EXCLUSIVE that ATTACH needs cannot
    self-deadlock: Postgres lets a request past waiters whose locks the
    requester already conflicts with.
    """
    return f"LOCK TABLE {_qualified(default_partition_name(item_table))} IN SHARE MODE"


def move_rows_sql(knowledge_base_id: Any, item_table: str, columns: list[str]) -> tuple[str, str]:
    """Copy every row of a KB from DEFAULT into its partition, then delete them.

    A partition cannot be attached while the DEFAULT partition still holds a
    row that belongs to it, so the rows move first. Two statements rather than
    one ``DELETE ... RETURNING`` feeding an INSERT: measured at 40 000 rows over
    a 500 000-row DEFAULT, the pair took 0.14 s and the CTE 0.29 s. Both run in
    the move's single transaction, under the locks above, so nothing can change
    the rows between the copy and the delete and no reader ever sees a row
    twice or not at all.

    No foreign key anywhere references these three tables (``embeddings``
    points at items polymorphically by ``item_id``), which is what makes the
    DELETE safe: an FK to the parent is impossible without a parent primary
    key, and one to the DEFAULT partition would cascade on this delete.

    Foreign keys *from* these tables (to ``knowledge_bases``, ``sources``,
    ``indexed_sources``) do cascade into them while a move runs: Postgres'
    referential triggers delete from DEFAULT and from the new partition by
    name, never through the parent, so they wait on the move's SHARE lock on
    DEFAULT (or the move waits on them) instead of on the parent. Such a
    cascade and a move can deadlock; Postgres detects it and rolls one side
    back whole, so no row is lost -- the move is one transaction and its task
    retries, a re-index that loses re-queues its source. The move's ACCESS
    EXCLUSIVE tries on DEFAULT never queue, but its SHARE lock on DEFAULT and
    the ATTACH's lock on the new partition still do, so the cascade is not
    guaranteed to be the side that survives. In the measured case (a source
    deleted while the move was about to ATTACH) the delete waited 1.0 s and
    both committed with no orphaned rows.

    ``columns`` are DEFAULT's insertable columns, already quoted
    (``_insertable_columns``), and are named on both sides: a column added to
    the parent after the clone was created would otherwise shift every value.
    """
    partition = partition_name(knowledge_base_id, item_table)
    default = default_partition_name(item_table)
    if not columns:
        raise ValueError("move_rows_sql needs the column list")
    names = ", ".join(columns)
    predicate = "WHERE knowledge_base_id = CAST(:kb AS uuid)"
    return (
        f"INSERT INTO {_qualified(partition)} ({names}) "
        f"SELECT {names} FROM {_qualified(default)} {predicate}",
        f"DELETE FROM {_qualified(default)} {predicate}",
    )


def copy_policies_sql(source: str, target: str) -> str:
    """Recreate every row-level security policy of ``source`` on ``target``.

    ``source`` and ``target`` are schema-qualified relation names as they
    appear in SQL (``'"ai".chunks'``), built by this module from validated
    parts, never from caller input. Each policy keeps its name, PERMISSIVE or
    RESTRICTIVE, command, roles, USING and WITH CHECK expressions. A policy
    whose name already exists on ``target`` is left alone. Same semantics as
    ``_copy_policies`` in migration 0031, which does this for the parents.

    Why it is needed: Postgres applies only the policies of the relation a
    query names. A new partition or partitioned parent starts with none, so
    once RLS is enabled on it a role without BYPASSRLS reads nothing -- and
    the search path reads each knowledge base's partition by name. The
    self-hosted schema grants ``FOR SELECT TO authenticated`` on these tables.
    """
    return f"""
        DO $$
        DECLARE
            src oid := '{source}'::regclass;
            tgt oid := '{target}'::regclass;
            p record;
            roles text;
            statement text;
        BEGIN
            FOR p IN
                SELECT pol.polname, pol.polpermissive, pol.polcmd, pol.polroles,
                       pg_get_expr(pol.polqual, pol.polrelid) AS qual,
                       pg_get_expr(pol.polwithcheck, pol.polrelid) AS with_check
                FROM pg_policy pol
                WHERE pol.polrelid = src
                  AND NOT EXISTS (
                      SELECT 1 FROM pg_policy existing
                      WHERE existing.polrelid = tgt AND existing.polname = pol.polname
                  )
                ORDER BY pol.polname
            LOOP
                SELECT string_agg(
                           CASE WHEN r = 0 THEN 'PUBLIC'
                                ELSE quote_ident(pg_get_userbyid(r)) END,
                           ', ')
                  INTO roles
                  FROM unnest(p.polroles) AS r;
                statement := format(
                    'CREATE POLICY %I ON {target} AS %s FOR %s TO %s',
                    p.polname,
                    CASE WHEN p.polpermissive THEN 'PERMISSIVE' ELSE 'RESTRICTIVE' END,
                    CASE p.polcmd WHEN 'r' THEN 'SELECT' WHEN 'a' THEN 'INSERT'
                                  WHEN 'w' THEN 'UPDATE' WHEN 'd' THEN 'DELETE'
                                  ELSE 'ALL' END,
                    roles
                );
                IF p.qual IS NOT NULL THEN
                    statement := statement || ' USING (' || p.qual || ')';
                END IF;
                IF p.with_check IS NOT NULL THEN
                    statement := statement || ' WITH CHECK (' || p.with_check || ')';
                END IF;
                EXECUTE statement;
            END LOOP;
        END $$;
    """


def mirror_relation_settings_sql(source: str, target: str) -> str:
    """Copy ownership, GRANTs, the RLS flag and RLS policies from one relation.

    A new partition starts with no privileges, RLS off and no policies, so
    without this a partition is either unreachable by the roles that can read
    the parent, or (if it were granted blindly) readable past the parent's
    row-level rules. Emitted as server-side blocks so every identifier is
    quoted by ``format(%I/%s)`` rather than by string building here. The
    policies matter because the search path reads a partition *by name*, and a
    query is filtered by the policies of the relation it names -- see
    ``copy_policies_sql``.
    """
    return f"""
        DO $$
        DECLARE
            src oid := '{source}'::regclass;
            owner text := (SELECT pg_get_userbyid(relowner) FROM pg_class WHERE oid = src);
            rls boolean := (SELECT relrowsecurity FROM pg_class WHERE oid = src);
            g record;
        BEGIN
            EXECUTE format('ALTER TABLE {target} OWNER TO %I', owner);
            IF rls THEN
                EXECUTE 'ALTER TABLE {target} ENABLE ROW LEVEL SECURITY';
            END IF;
            FOR g IN
                SELECT a.grantee::regrole AS role,
                       string_agg(a.privilege_type, ', ') AS privileges
                FROM pg_class c, aclexplode(c.relacl) a
                WHERE c.oid = src AND a.grantee <> 0
                GROUP BY a.grantee
            LOOP
                EXECUTE format('GRANT %s ON TABLE {target} TO %s',
                               g.privileges, g.role);
            END LOOP;
        END $$;
    """ + copy_policies_sql(source, target)


# ---------------------------------------------------------------------------
# Query normalisation
# ---------------------------------------------------------------------------


def normalize_bm25_query(raw: Any) -> str:
    """Make user text safe to hand to the ``|||`` match operator.

    ``|||`` tokenizes its right-hand side; it does not parse query syntax.
    Verified against pg_search 0.25.9 with quotes, ``:``, ``\\``, ``+ - ~ ^``,
    unbalanced ``( ) [ ] { }``, ``/``, bare AND/OR/NOT, a 50 000-character
    term and ``' OR 1=1; DROP TABLE ...; --``: every one of them returned rows
    or an empty result, none raised. So nothing is escaped -- escaping would
    silently mangle terms like ``C++`` or ``re:invent`` -- and the query is
    bound as a parameter, never interpolated.

    What does have to be removed is what never reaches Postgres intact:
    psycopg refuses a text parameter containing NUL ("PostgreSQL text fields
    cannot contain NUL (0x00) bytes"), so control characters collapse to
    spaces, and the result is length-bounded.
    """
    if raw is None:
        return ""
    collapsed = re.sub(r"[\x00-\x1f\x7f]+", " ", str(raw))
    normalized = re.sub(r"\s+", " ", collapsed).strip()
    if len(normalized) > MAX_BM25_QUERY_CHARS:
        normalized = normalized[:MAX_BM25_QUERY_CHARS].strip()
    return normalized


# ---------------------------------------------------------------------------
# Availability and readiness (cached; never raises)
# ---------------------------------------------------------------------------


def reset_pg_bm25_caches() -> None:
    """Forget extension availability and every index-readiness answer."""
    global _extension_cache
    _extension_cache = None
    _ready_cache.clear()


def invalidate_bm25_index_cache(knowledge_base_id: str) -> None:
    """Forget cached readiness for one KB, after its index changed."""
    kb_id = str(knowledge_base_id)
    for key in [k for k in _ready_cache if k[0] == kb_id]:
        _ready_cache.pop(key, None)


def _probe(session, sql: str, params: dict | None = None):
    """Run a read-only catalog probe without risking the caller's transaction.

    On a Session or a transactional Connection the probe runs in a savepoint,
    so a failure (a stale pooled connection, a cancelled statement) is rolled
    back to it and the caller's transaction -- and the keyword fallback that
    runs in it next -- stays usable. An AUTOCOMMIT connection has no
    transaction to protect and would refuse the SAVEPOINT.
    """
    get_options = getattr(session, "get_execution_options", None)
    autocommit = (
        callable(get_options) and (get_options() or {}).get("isolation_level") == "AUTOCOMMIT"
    )
    begin_nested = getattr(session, "begin_nested", None)
    if autocommit or begin_nested is None:
        return session.execute(text(sql), params or {}).first()
    with begin_nested():
        return session.execute(text(sql), params or {}).first()


def pg_search_installed(session, *, use_cache: bool = True) -> bool:
    """Is the pg_search extension created in this database?

    Cached for ``_CACHE_TTL_SECONDS`` and never raises: this is read on the
    search path, where the honest answer to "can't tell" is "use the old
    path" -- for this call only. A failed probe is logged and not cached, so
    one stale connection cannot hide the extension from every request for the
    TTL. ``use_cache=False`` is for the build and drop paths, which must not
    act on an answer that may be a TTL old.
    """
    global _extension_cache
    now = time.monotonic()
    if (
        use_cache
        and _extension_cache is not None
        and now - _extension_cache[0] < _CACHE_TTL_SECONDS
    ):
        return _extension_cache[1]
    try:
        installed = (
            _probe(session, "SELECT 1 FROM pg_extension WHERE extname = 'pg_search'") is not None
        )
    except Exception as exc:
        logger.warning(
            "Could not determine whether pg_search is installed (%s); using the "
            "existing keyword path for this request",
            first_error_line(exc),
        )
        return False
    _extension_cache = (now, installed)
    return installed


def _read_index_state(session, knowledge_base_id: str, item_table: str) -> str:
    """``absent`` | ``building`` | ``ready``; raises if the catalog cannot be read."""
    if item_table not in PARTITIONED_ITEM_TABLES:
        return "absent"
    name = bm25_index_name(knowledge_base_id, item_table)
    partition = partition_name(knowledge_base_id, item_table)
    row = _probe(
        session,
        "SELECT i.indisvalid FROM pg_index i "
        "JOIN pg_class c ON c.oid = i.indexrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_class t ON t.oid = i.indrelid "
        "WHERE n.nspname = :schema AND c.relname = :name "
        "AND t.relname = :partition",
        {"schema": AI_SCHEMA, "name": name, "partition": partition},
    )
    if row is None:
        return "absent"
    return "ready" if row[0] else "building"


def bm25_index_state(session, knowledge_base_id: str, item_table: str) -> str:
    """``absent`` | ``building`` | ``ready`` for one KB's BM25 index.

    The index has to be the one on *this KB's partition*: that is the relation
    the search path names, so an index of the same name sitting anywhere else
    (a leftover from the unpartitioned design, say) must not read as ready.

    ``building`` is an index row with ``indisvalid = false`` -- what a
    CREATE INDEX CONCURRENTLY still in flight leaves behind, and also one that
    failed until the next ``ensure_bm25_index`` drops and rebuilds it. Such an
    index cannot answer a query, so it is not ready.

    Never raises; an unreadable catalog reads as ``absent``.
    """
    try:
        return _read_index_state(session, knowledge_base_id, item_table)
    except Exception as exc:
        logger.warning(
            "Could not read BM25 index state for KB %s on %s: %s",
            knowledge_base_id,
            item_table,
            first_error_line(exc),
        )
        return "absent"


def bm25_index_ready(session, knowledge_base_id: str, item_table: str) -> bool:
    """Cached "can this KB's BM25 index answer a query right now?".

    A failed probe answers False for this call and is not cached.
    """
    key = (str(knowledge_base_id), item_table)
    now = time.monotonic()
    cached = _ready_cache.get(key)
    if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    try:
        ready = _read_index_state(session, knowledge_base_id, item_table) == "ready"
    except Exception as exc:
        logger.warning(
            "Could not read BM25 index readiness for KB %s on %s: %s; using the "
            "existing keyword path for this request",
            knowledge_base_id,
            item_table,
            first_error_line(exc),
        )
        return False
    if len(_ready_cache) >= _READY_CACHE_MAX_ENTRIES:
        _ready_cache.clear()
    _ready_cache[key] = (now, ready)
    return ready


def _item_table_is_partitioned(session, item_table: str) -> bool:
    """``table_is_partitioned`` for the read paths: the probe runs in a savepoint.

    Raises on a failed probe; ``keyword_index_backend`` decides what that means.
    """
    row = _probe(
        session,
        "SELECT c.relkind FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = :schema AND c.relname = :relname",
        {"schema": AI_SCHEMA, "relname": item_table},
    )
    return row is not None and row[0] == "p"


def keyword_index_backend(session, strategy: str | None) -> str | None:
    """Which keyword index serves a KB with this indexing strategy.

    ``"pg_search"`` -- its own ``USING bm25`` index on its partition: the
    extension is installed, the strategy maps to an item table, and that table
    is partitioned by knowledge base (the checks ``ensure_bm25_index`` makes
    before doing anything). ``"bm25s"`` -- the bm25s file index: every other
    strategy with an item table, including when a probe fails. ``None`` -- the
    strategy has no BM25 item table. A missing strategy means chunk_embed.

    This is a rule about the *item table*, and it decides which build to
    start: the knowledge-base routes (create, PATCH, ``/build-bm25``) ask it,
    so they cannot disagree about which index a KB should get. Whether a given
    KB is *already* served by its pg_search index is a per-KB question --
    ``pg_search_serves_kb`` -- because a KB that predates the extension keeps
    its rows in DEFAULT, and search keeps reading its bm25s file index, until
    it is given its own index.

    Never raises, and probes in a savepoint, so it is safe mid-transaction.
    """
    strategy = strategy or "chunk_embed"
    if STRATEGY_TO_BM25_ITEM_TABLE.get(strategy) is None:
        return None
    if not pg_search_installed(session):
        return "bm25s"
    item_table = pg_bm25_item_table(strategy)
    if item_table is None or item_table not in PARTITIONED_ITEM_TABLES:
        return "bm25s"
    try:
        partitioned = _item_table_is_partitioned(session, item_table)
    except Exception as exc:
        logger.warning(
            "Could not tell whether %s.%s is partitioned (%s); treating its keyword index "
            "as the bm25s file index",
            AI_SCHEMA,
            item_table,
            first_error_line(exc),
        )
        return "bm25s"
    return "pg_search" if partitioned else "bm25s"


def pg_search_serves_kb(session, knowledge_base_id: str, strategy: str | None) -> bool:
    """Is this KB's keyword leg answered by its own pg_search index right now?

    True only when the item table's backend is pg_search *and* this KB's
    partition carries a ready index -- the same test the search path makes
    before it reads the index. Until then search reads the KB's bm25s file
    index, so per-source maintenance of that file index must go on: a KB that
    existed before the extension, whose rows are still in DEFAULT, keeps it
    until an operator builds its index (``POST /build-bm25``).

    Not cached, unlike ``bm25_index_ready``: a cached "ready" can outlive an
    index dropped by another process for the cache's TTL, and stopping
    maintenance on it would silently freeze the file index search falls back
    to. Two or three catalog lookups per call (the extension, whether the table
    is partitioned, the index). Never raises; "can't tell" is False.
    """
    try:
        if keyword_index_backend(session, strategy) != "pg_search":
            return False
        item_table = pg_bm25_item_table(strategy or "chunk_embed")
        if item_table is None:
            return False
        return _read_index_state(session, knowledge_base_id, item_table) == "ready"
    except Exception as exc:
        logger.warning(
            "Could not tell whether pg_search serves KB %s (%s); treating its keyword index "
            "as the bm25s file index",
            knowledge_base_id,
            first_error_line(exc),
        )
        return False


def pg_bm25_status(knowledge_base_id: str, strategy: str | None, session=None) -> str | None:
    """Index state for the KB detail response, or None when not applicable.

    None means "this KB has no pg_search index to report on" -- no extension,
    or a strategy with no keyword item table -- and the caller should fall
    back to whatever it reported before.
    """
    try:
        if session is None:
            from ..db import db

            session = db.session
        if not pg_search_installed(session):
            return None
        item_table = pg_bm25_item_table(strategy)
        if item_table is None:
            return None
        return bm25_index_state(session, knowledge_base_id, item_table)
    except Exception as exc:
        logger.debug(
            "Could not compute pg_search BM25 status for KB %s: %s", knowledge_base_id, exc
        )
        return None


# ---------------------------------------------------------------------------
# Partition and index lifecycle
# ---------------------------------------------------------------------------


def _kb_config_sql() -> str:
    """The three KB fields that decide whether and how to index it.

    A config with no ``strategy`` key is searched as chunk_embed, so it is
    indexed as one.
    """
    return (
        "SELECT COALESCE(indexing_config->>'strategy', 'chunk_embed'), "
        "retrieval_config->>'method', "
        "retrieval_config->>'ts_language' "
        f'FROM "{AI_SCHEMA}".knowledge_bases WHERE id = :id'
    )


def _relkind(conn, relname: str) -> str | None:
    """``pg_class.relkind`` for a relation in the ai schema, or None if absent.

    ``'p'`` is a partitioned parent, ``'r'`` an ordinary table (which a
    partition is).
    """
    row = conn.execute(
        text(
            "SELECT c.relkind FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = :schema AND c.relname = :relname"
        ),
        {"schema": AI_SCHEMA, "relname": relname},
    ).first()
    return row[0] if row else None


def table_is_partitioned(conn, item_table: str) -> bool:
    """Has the conversion migration reached this item table yet?"""
    return _relkind(conn, _validated_partitioned_table(item_table)) == "p"


def _clone_exists(conn, knowledge_base_id: Any, item_table: str) -> bool:
    """Is there a relation of the partition's name, attached or not?"""
    return _relkind(conn, partition_name(knowledge_base_id, item_table)) is not None


def _run_probe(bind, fn):
    """Run ``fn(conn)`` on a connection derived from ``bind``, safely.

    An Engine gets a connection of its own; a Session or Connection runs the
    probe in a savepoint (``_probe``), so a failure cannot abort the caller's
    transaction.
    """
    from sqlalchemy.engine import Engine

    if isinstance(bind, Engine):
        with bind.connect() as conn:
            try:
                return fn(conn)
            finally:
                conn.rollback()
    return fn(bind)


def partition_exists(bind, kb_id: str, item_table: str) -> bool:
    """Does this knowledge base have its own *attached* partition of the item table?

    An unattached clone of that name is a move that did not finish, and does
    not count. ``bind`` is an Engine, Connection or Session. Never raises: an
    invalid id, a table that is never partitioned, or a failed probe is False.
    """
    try:
        partition = partition_name(kb_id, item_table)
        row = _run_probe(
            bind,
            lambda conn: _probe(
                conn,
                "SELECT 1 FROM pg_inherits i "
                "JOIN pg_class c ON c.oid = i.inhrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "JOIN pg_class p ON p.oid = i.inhparent "
                "WHERE n.nspname = :schema AND c.relname = :partition AND p.relname = :parent",
                {"schema": AI_SCHEMA, "partition": partition, "parent": item_table},
            ),
        )
        return row is not None
    except Exception as exc:
        logger.debug(
            "Could not tell whether KB %s has a partition of %s: %s", kb_id, item_table, exc
        )
        return False


def kb_has_rows_in_default(bind, kb_id: str, item_table: str) -> bool | None:
    """Does the item table's DEFAULT partition still hold rows of this knowledge base?

    ``None`` means "cannot tell": an invalid id, a table that is never
    partitioned, no DEFAULT partition, or a failed probe. Reads one row at
    most through the DEFAULT partition's ``knowledge_base_id`` index. The read
    takes ACCESS SHARE on DEFAULT until the caller's transaction ends, so a
    request path should end its transaction promptly. Never raises.
    """
    try:
        kb = _validated_kb_id(kb_id)
        default = _qualified(default_partition_name(item_table))
        row = _run_probe(
            bind,
            lambda conn: _probe(
                conn,
                f"SELECT EXISTS (SELECT 1 FROM {default} "
                "WHERE knowledge_base_id = CAST(:kb AS uuid))",
                {"kb": kb},
            ),
        )
        return bool(row[0]) if row is not None else None
    except Exception as exc:
        logger.debug(
            "Could not tell whether KB %s has rows in the DEFAULT partition of %s: %s",
            kb_id,
            item_table,
            exc,
        )
        return None


def _partition_is_attached(conn, knowledge_base_id: Any, item_table: str) -> bool:
    row = conn.execute(
        text(
            "SELECT 1 FROM pg_inherits i "
            "JOIN pg_class c ON c.oid = i.inhrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_class p ON p.oid = i.inhparent "
            "WHERE n.nspname = :schema AND c.relname = :partition AND p.relname = :parent"
        ),
        {
            "schema": AI_SCHEMA,
            "partition": partition_name(knowledge_base_id, item_table),
            "parent": item_table,
        },
    ).first()
    return row is not None


def _foreign_key_defs(conn, relname: str) -> list[str]:
    """``FOREIGN KEY ...`` clauses of a relation, in constraint-name order."""
    return [
        row[0]
        for row in conn.execute(
            text(
                "SELECT pg_get_constraintdef(c.oid) FROM pg_constraint c "
                "JOIN pg_class t ON t.oid = c.conrelid "
                "JOIN pg_namespace n ON n.oid = t.relnamespace "
                "WHERE n.nspname = :schema AND t.relname = :relname AND c.contype = 'f' "
                "ORDER BY c.conname"
            ),
            {"schema": AI_SCHEMA, "relname": relname},
        ).all()
    ]


def _check_constraint_exists(conn, partition: str) -> bool:
    """Does the clone already carry *its own* kb check, by name?

    Not "any CHECK": ``LIKE ... INCLUDING CONSTRAINTS`` copies every CHECK on
    DEFAULT onto the clone, so the first unrelated CHECK on the item table
    would otherwise stop this one being added, and ATTACH would quietly go back
    to scanning the new partition.
    """
    row = conn.execute(
        text(
            "SELECT 1 FROM pg_constraint c "
            "JOIN pg_class t ON t.oid = c.conrelid "
            "JOIN pg_namespace n ON n.oid = t.relnamespace "
            "WHERE n.nspname = :schema AND t.relname = :relname AND c.contype = 'c' "
            "AND c.conname = :conname"
        ),
        {"schema": AI_SCHEMA, "relname": partition, "conname": f"{partition}_kb_check"},
    ).first()
    return row is not None


# SQLSTATEs worth retrying a partition or index build for: nothing about the
# request is wrong. Another transaction was in the way, a server-side timeout
# cancelled the statement, or the connection went away. The move is one
# transaction and the index build is re-entrant, so a retry is always safe.
_TRANSIENT_SQLSTATES = frozenset(
    {
        "55P03",  # lock_not_available (lock_timeout)
        "40P01",  # deadlock_detected
        "40001",  # serialization_failure
        "57014",  # query_canceled (a role's or database's statement_timeout)
        "57P01",  # admin_shutdown
        "57P02",  # crash_shutdown
        "57P03",  # cannot_connect_now (the server is starting up)
    }
)
# Class 08: connection exceptions.
_TRANSIENT_SQLSTATE_CLASSES = frozenset({"08"})
_LOCK_CONFLICT_SQLSTATES = frozenset({"40P01", "55P03"})


class Bm25IndexBuildFailed(RuntimeError):
    """pg_search's own ``CREATE INDEX CONCURRENTLY`` failed with an internal error.

    Seen with stock pg_search 0.25.9 on Postgres 15/16 under concurrent writes
    (``XX000: buffer ... is not owned by resource owner``), fixed by
    paradedb/paradedb#6211. It leaves an INVALID index that the next
    ``ensure_bm25_index`` drops and rebuilds, so it is retried. The same bug can
    crash the server instead; the build then fails with a lost connection,
    which is retried as well (see ``_reset_statement_timeout``).
    """


# ---------------------------------------------------------------------------
# Is a concurrent bm25 build safe on this server?
# ---------------------------------------------------------------------------

#: A placeholder setting a Postgres image whose pg_search contains
#: paradedb/paradedb#6211 sets in its ``postgresql.conf``. Any name with a dot
#: is accepted by Postgres without an extension defining it, and
#: ``current_setting(name, true)`` answers NULL on a server that does not set it.
PG_SEARCH_CIC_SAFE_MARKER = "powabase.pg_search_cic_safe"
#: The project setting a self-hoster turns on for a pg_search they built with
#: the fix themselves, which nothing on the server can show.
CONCURRENT_BUILD_SAFE_SETTING = "BM25_PG_SEARCH_CONCURRENT_BUILD_SAFE"
#: Postgres 17 and later are not affected.
UNAFFECTED_SERVER_VERSION_NUM = 170000
#: The first pg_search release line that contains the fix: v0.26.0-rc.1 is a
#: descendant of the fix's merge commit, and no 0.25.x release contains it.
FIXED_PG_SEARCH_VERSION = (0, 26)

_CONCURRENT_BUILD_SAFETY_SQL = (
    f"SELECT current_setting('{PG_SEARCH_CIC_SAFE_MARKER}', true), "
    "current_setting('server_version_num')::int, "
    "(SELECT extversion FROM pg_extension WHERE extname = 'pg_search')"
)

#: Why a build was not started, for logs, the recorded outcome and the API.
CONCURRENT_BUILD_UNSAFE_REASON = (
    "not built: nothing shows that this server's pg_search contains the fix for "
    "building a bm25 index while the table takes writes (paradedb/paradedb#6211). "
    "Without it, on Postgres 15 and 16 that build fails or crashes the database "
    "server, so no index was built, no rows were moved and no existing index was "
    "dropped; keyword search keeps the path it uses now. To enable it, run a "
    f"Postgres image that sets {PG_SEARCH_CIC_SAFE_MARKER} = on in postgresql.conf, "
    "Postgres 17 or later, or pg_search 0.26.0 or later -- or, if your pg_search "
    f"build contains the fix, turn on the {CONCURRENT_BUILD_SAFE_SETTING} setting -- "
    "then POST /build-bm25"
)

_unsafe_build_warning_logged = False


def _read_concurrent_build_override() -> bool:
    """``BM25_PG_SEARCH_CONCURRENT_BUILD_SAFE``; off outside an application context."""
    if not _has_app_context():
        return False
    try:
        from . import settings_registry

        return bool(settings_registry.get_setting(CONCURRENT_BUILD_SAFE_SETTING))
    except Exception:
        logger.warning(
            "Could not read %s; treating it as off", CONCURRENT_BUILD_SAFE_SETTING, exc_info=True
        )
        return False


def _concurrent_build_override() -> bool:
    return _read_concurrent_build_override()


def _parse_safety_row(row) -> tuple[bool, str] | None:
    """``(safe, basis)`` from the probe's row, or None when it cannot be read."""
    if row is None or len(row) != 3:
        return None
    marker, server_version_num, extversion = row
    if marker is not None and not isinstance(marker, str):
        return None
    if not isinstance(server_version_num, int) or isinstance(server_version_num, bool):
        return None
    if extversion is not None and not isinstance(extversion, str):
        return None
    if (marker or "").strip().lower() == "on":
        return True, f"marker {PG_SEARCH_CIC_SAFE_MARKER}"
    if server_version_num >= UNAFFECTED_SERVER_VERSION_NUM:
        return True, f"postgres {server_version_num}"
    match = re.match(r"(\d+)\.(\d+)", extversion or "")
    if match and (int(match.group(1)), int(match.group(2))) >= FIXED_PG_SEARCH_VERSION:
        return True, f"pg_search {extversion}"
    return False, "unverified"


def concurrent_build_safety(bind) -> tuple[bool, str]:
    """Can a bm25 index be built concurrently here without the pg_search bug?

    Safe when any of these holds: the server sets ``PG_SEARCH_CIC_SAFE_MARKER``
    to ``on``; it is Postgres 17 or later; its pg_search is 0.26.0 or later; or
    the project setting ``CONCURRENT_BUILD_SAFE_SETTING`` vouches for it.
    Returns ``(safe, basis)``, the basis being what made it safe or
    ``"unverified"``. Raises if the server cannot be asked (the build path's
    caller retries a transient error).
    """
    parsed = _parse_safety_row(_probe(bind, _CONCURRENT_BUILD_SAFETY_SQL))
    if parsed is None:
        raise RuntimeError("could not read whether a concurrent bm25 build is safe")
    if parsed[0]:
        return parsed
    if _concurrent_build_override():
        return True, "setting"
    return parsed


def concurrent_build_known_unsafe(bind) -> bool:
    """For a request path: True only on a clear "not safe" answer. Never raises."""
    try:
        parsed = _parse_safety_row(_probe(bind, _CONCURRENT_BUILD_SAFETY_SQL))
        if parsed is None or parsed[0]:
            return False
        return not _concurrent_build_override()
    except Exception:
        logger.debug("Could not tell whether a concurrent bm25 build is safe", exc_info=True)
        return False


def _warn_unsafe_build_once(kb_id: str) -> None:
    """One WARNING per process: every knowledge base's ensure would repeat it."""
    global _unsafe_build_warning_logged
    if _unsafe_build_warning_logged:
        logger.info(
            "Not building the BM25 index of KB %s: concurrent build not verified safe", kb_id
        )
        return
    _unsafe_build_warning_logged = True
    logger.warning(
        "Not building BM25 indexes (first: KB %s): %s. Logged once per process",
        kb_id,
        CONCURRENT_BUILD_UNSAFE_REASON,
    )


def first_error_line(exc: BaseException) -> str:
    """The first line of an error's message, or its type name when it has none."""
    orig = getattr(exc, "orig", None)
    for candidate in (orig, exc):
        if candidate is None:
            continue
        lines = [line for line in str(candidate).splitlines() if line.strip()]
        if lines:
            return lines[0]
    return type(orig if orig is not None else exc).__name__


def _sqlstate(exc: BaseException) -> str | None:
    """The SQLSTATE of a SQLAlchemy-wrapped or bare driver error, if any."""
    return getattr(getattr(exc, "orig", exc), "sqlstate", None)


def is_transient_db_error(exc: BaseException) -> bool:
    """Did this fail for a reason a retry of the same request can get past?

    Contention, a cancelled statement, a lost connection (reported with no
    SQLSTATE by the driver, which SQLAlchemy marks ``connection_invalidated``),
    or a failed concurrent bm25 build.
    """
    if isinstance(exc, Bm25IndexBuildFailed):
        return True
    if getattr(exc, "connection_invalidated", False):
        return True
    sqlstate = _sqlstate(exc)
    if sqlstate is None:
        return False
    return sqlstate in _TRANSIENT_SQLSTATES or sqlstate[:2] in _TRANSIENT_SQLSTATE_CLASSES


def is_lock_conflict(exc: BaseException) -> bool:
    """Did a statement lose a deadlock or a lock timeout (SQLSTATE 40P01, 55P03)?

    Either means another transaction held what this one needed -- a partition
    move holding the item table, typically -- not that the statement was wrong.
    Unwraps a SQLAlchemy error the same way ``is_transient_db_error`` does.
    """
    return _sqlstate(exc) in _LOCK_CONFLICT_SQLSTATES


def is_partition_move_race(exc: BaseException) -> bool:
    """Did a write fail only because it raced a knowledge base's partition move?

    SQLSTATE 23514 from either of the two checks a move can trip: Postgres'
    own "violates partition constraint" (a row routed to DEFAULT by a statement
    planned before the ATTACH), or the move's temporary ``bm25_move_<kb>``
    check on DEFAULT. An ordinary CHECK violation is a real error and is not
    matched.

    Read from the error's diagnostics when the server sends them, so a
    translated message classifies the same: a CHECK violation always names its
    constraint, and a partition constraint violation names none. Only a server
    that sends no fields at all is read by its (untranslated) message.
    """
    orig = getattr(exc, "orig", None)
    if getattr(orig, "sqlstate", None) != "23514":
        return False
    diag = getattr(orig, "diag", None)
    constraint = getattr(diag, "constraint_name", None)
    table = getattr(diag, "table_name", None)
    if constraint is not None or table is not None:
        if constraint is None:
            return True
        return re.fullmatch(rf"{_DEFAULT_MOVE_CHECK_PREFIX}[0-9a-f]{{32}}", constraint) is not None
    message = first_error_line(exc)
    return "violates partition constraint" in message or (
        f'violates check constraint "{_DEFAULT_MOVE_CHECK_PREFIX}' in message
    )


def _lock_holders_sql() -> str:
    """Locks on the parent, on DEFAULT, and on the item table's move gate.

    An advisory lock on a bigint key appears in ``pg_locks`` split into
    ``classid`` (high 32 bits) and ``objid`` (low 32 bits), with ``objsubid`` 1.
    """
    return (
        "WITH gate AS (SELECT hashtextextended(:gate, 0) AS k) "
        "SELECT l.pid, "
        "CASE WHEN l.locktype = 'advisory' THEN 'move gate' "
        "ELSE l.relation::regclass::text END AS lock_on, "
        "l.mode, l.granted, pg_blocking_pids(l.pid) AS blocked_by, "
        "a.state, round(extract(epoch FROM now() - a.xact_start)::numeric, 1) AS xact_seconds, "
        "left(a.query, 200) AS query "
        "FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid CROSS JOIN gate "
        "WHERE (l.relation IN (to_regclass(:parent), to_regclass(:default)) "
        "OR (l.locktype = 'advisory' AND l.objsubid = 1 AND l.database = ("
        "SELECT oid FROM pg_database WHERE datname = current_database()) "
        "AND l.classid = ((gate.k >> 32) & 4294967295)::oid "
        "AND l.objid = (gate.k & 4294967295)::oid)) "
        "AND l.pid <> pg_backend_pid() AND NOT (l.pid = ANY(CAST(:exclude AS int[]))) "
        "ORDER BY l.granted DESC, l.pid LIMIT 20"
    )


def _backend_pid(conn) -> int | None:
    try:
        return int(conn.connection.dbapi_connection.info.backend_pid)
    except Exception:
        return None


def _lock_holders(engine, item_table: str, exclude: list[int | None]) -> list[dict]:
    """Sessions holding or awaiting locks on the parent, DEFAULT or the move gate. Never raises.

    Read on a connection of its own, so it can run while the failed move's
    transaction is still open: its locks -- and whoever refused them -- are
    still there to be seen. After the rollback, the list is the writers the
    move had just released, never the reader that made it give up.
    """
    try:
        with engine.connect() as probe:
            try:
                return [
                    dict(row._mapping)
                    for row in probe.execute(
                        text(_lock_holders_sql()),
                        {
                            "parent": _qualified(item_table),
                            "default": _qualified(default_partition_name(item_table)),
                            "gate": move_gate_relation(item_table),
                            "exclude": [pid for pid in exclude if pid is not None],
                        },
                    ).all()
                ]
            finally:
                probe.rollback()
    except Exception:
        logger.debug("Could not list lock holders", exc_info=True)
        return []


def _fail_move(
    engine,
    conn,
    knowledge_base_id: str,
    item_table: str,
    exc: BaseException,
    *,
    step: str,
    action: str = "Moving the rows into the partition of",
) -> None:
    """Roll a failed move back and say why, and for contention, who was in the way.

    The lock holders are read *before* the rollback (``_lock_holders``) and
    attached to the exception as ``bm25_lock_holders``, with the step as
    ``bm25_move_step``, so the task that gives up can name them.
    """
    transient = is_transient_db_error(exc)
    holders = _lock_holders(engine, item_table, [_backend_pid(conn)]) if transient else []
    try:
        conn.rollback()
    except Exception:
        logger.debug("Rollback after a failed move raised", exc_info=True)
    try:
        exc.bm25_lock_holders = holders
        exc.bm25_move_step = step
        exc.bm25_item_table = item_table
    except Exception:
        pass
    if not transient:
        logger.warning(
            "%s KB %s on %s.%s failed at step %r (SQLSTATE %s): %s; rolled back",
            action,
            knowledge_base_id,
            AI_SCHEMA,
            item_table,
            step,
            _sqlstate(exc),
            first_error_line(exc),
            exc_info=exc,
        )
        return
    logger.warning(
        "%s KB %s on %s.%s gave up at step %r (SQLSTATE %s: %s); rolled back, retryable. "
        "Sessions holding or awaiting locks on the table: %s",
        action,
        knowledge_base_id,
        AI_SCHEMA,
        item_table,
        step,
        _sqlstate(exc),
        first_error_line(exc),
        holders,
    )


def _acquire_partition_build_lock(conn, item_table: str) -> None:
    """Claim the right to move rows out of this item table's DEFAULT partition.

    Bounded: after ``PARTITION_BUILD_LOCK_WAIT_SECONDS`` of another build
    holding it, this raises ``PartitionBuildInProgress`` rather than waiting on,
    so the caller can report a retryable outcome instead of hanging.
    """
    relation = partition_build_lock_relation(item_table)
    deadline = time.monotonic() + PARTITION_BUILD_LOCK_WAIT_SECONDS
    while True:
        acquired = conn.execute(text(partition_build_lock_sql()), {"relation": relation}).scalar()
        conn.commit()
        if acquired:
            return
        if time.monotonic() >= deadline:
            raise PartitionBuildInProgress(
                f"another partition build is in progress for {relation}; retry"
            )
        time.sleep(_PARTITION_BUILD_LOCK_POLL_SECONDS)


def _acquire_move_gate(conn, item_table: str) -> None:
    """Take the move gate exclusively, queued, for at most ``MOVE_GATE_WAIT_SECONDS``.

    Session-scoped, like the build lock: it has to outlive the transactions
    before the move's. A timeout raises SQLSTATE 55P03 (retried) having taken
    no table lock at all. The failed transaction is left for the caller's
    ``_fail_move``, which names the gate's holders before rolling it back.
    """
    conn.execute(text(f"SET LOCAL lock_timeout = '{int(MOVE_GATE_WAIT_SECONDS * 1000)}ms'"))
    try:
        conn.execute(
            text("SELECT pg_advisory_lock(hashtextextended(:relation, 0))"),
            {"relation": move_gate_relation(item_table)},
        )
    except Exception:
        logger.warning(
            "A move on %s.%s could not take the move gate within %.0f s: indexing "
            "transactions on that table kept it%s; retryable",
            AI_SCHEMA,
            item_table,
            MOVE_GATE_WAIT_SECONDS,
            " (a graph_index run holds it for the length of its run)"
            if item_table == "graph_index_nodes"
            else "",
        )
        raise
    conn.commit()


class _MoveGate:
    """The move gate held on one connection, released once, however the move ends."""

    def __init__(self, conn, item_table: str):
        self.conn = conn
        self.item_table = item_table
        self.held = False

    def acquire(self) -> None:
        _acquire_move_gate(self.conn, self.item_table)
        self.held = True

    def release(self) -> None:
        if self.held:
            self.held = False
            _release_advisory_lock(self.conn, move_gate_relation(self.item_table))


def _release_partition_build_lock(conn, item_table: str) -> None:
    _release_advisory_lock(conn, partition_build_lock_relation(item_table))


def _release_advisory_lock(conn, relation: str) -> None:
    """Give the lock back, and if that cannot be done, throw the session away.

    The lock is session-scoped, and a pooled connection handed back to the pool
    keeps its session -- so a lock left behind would keep every later build on
    this item table (or of this index) waiting. Invalidating the connection ends the backend, which
    releases it for certain.

    Rolls back first. The unlock is committed, and on a connection whose move
    failed part-way that commit would otherwise make the half-done move durable
    -- rows copied into the unattached clone and deleted from DEFAULT, invisible
    through the parent -- or, on an aborted transaction, fail and bury the real
    error under a "could not release" traceback.
    """
    try:
        conn.rollback()
        conn.execute(text(partition_build_unlock_sql()), {"relation": relation})
        conn.commit()
    except Exception:
        logger.warning(
            "Could not release the advisory lock on %s; discarding the "
            "connection so the lock cannot outlive it",
            relation,
            exc_info=True,
        )
        try:
            conn.invalidate()
        except Exception:
            logger.debug("Could not invalidate the connection either", exc_info=True)


def _reset_statement_timeout(conn) -> None:
    """Undo a build's ``SET statement_timeout = 0`` without hiding the build's error.

    Called from the ``finally`` of a build whose connection may already be gone:
    a server that crashes or restarts mid-build ends every session, and
    SQLAlchemy then refuses any further statement on the connection
    (``PendingRollbackError``). Raised from here, that would replace the build's
    own error -- a lost connection, which the task retries -- with one it does
    not retry. So a failed reset is logged, and the connection is discarded
    rather than returned to the pool with no statement timeout.
    """
    try:
        conn.execute(text("RESET statement_timeout"))
    except Exception as exc:
        logger.warning(
            "Could not reset statement_timeout after a build (%s); discarding the connection",
            first_error_line(exc),
        )
        try:
            conn.invalidate()
        except Exception:
            logger.debug("Could not invalidate the connection either", exc_info=True)


def _insertable_columns(conn, relname: str) -> list[str]:
    """A relation's columns in order, quoted, leaving out generated ones."""
    return [
        row[0]
        for row in conn.execute(
            text(
                "SELECT quote_ident(a.attname) FROM pg_attribute a "
                "WHERE a.attrelid = to_regclass(:relation) AND a.attnum > 0 "
                "AND NOT a.attisdropped AND a.attgenerated = '' ORDER BY a.attnum"
            ),
            {"relation": _qualified(relname)},
        ).all()
    ]


def _column_signature(conn, relname: str) -> list[tuple]:
    """(name, type, not null, generated) per column, for comparing two relations."""
    return [
        tuple(row)
        for row in conn.execute(
            text(
                "SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull, "
                "a.attgenerated FROM pg_attribute a "
                "WHERE a.attrelid = to_regclass(:relation) AND a.attnum > 0 "
                "AND NOT a.attisdropped ORDER BY a.attname"
            ),
            {"relation": _qualified(relname)},
        ).all()
    ]


def _check_constraint_signature(conn, relname: str, ignore: str) -> list[str]:
    """A relation's CHECK constraint definitions, sorted, leaving out the move's own.

    ``ignore`` is the name of the one check the move itself adds to that
    relation (the clone's partition-bound check); the temporary move checks on
    DEFAULT are left out by their prefix. ``NOT VALID`` is not compared: a clone
    made with ``LIKE ... INCLUDING CONSTRAINTS`` carries a copied check as valid.
    """
    return sorted(
        row[0].removesuffix(" NOT VALID")
        for row in conn.execute(
            text(
                "SELECT pg_get_constraintdef(c.oid) FROM pg_constraint c "
                "WHERE c.conrelid = to_regclass(:relation) AND c.contype = 'c' "
                "AND c.conname <> :ignore AND c.conname NOT LIKE :prefix"
            ),
            {
                "relation": _qualified(relname),
                "ignore": ignore,
                "prefix": f"{_DEFAULT_MOVE_CHECK_PREFIX}%",
            },
        ).all()
    )


# A stale clone's foreign keys are dropped one at a time before the clone, each
# with this short lock_timeout and a pause between tries (see
# ``_drop_stale_clone``).
_STALE_CLONE_KEY_DROP_TRY_MS = 50
_STALE_CLONE_KEY_DROP_WAIT_SECONDS = 2.0


def _drop_clone_foreign_keys(conn, partition: str) -> None:
    """Drop an unattached clone's foreign keys, without queueing for long on what they reference.

    Dropping a foreign key takes ACCESS EXCLUSIVE on the table it references --
    ``knowledge_bases``, ``sources``, ``indexed_sources`` -- and a request queued
    for that lock makes every new reader of the table wait behind it. So each
    try waits at most ``_STALE_CLONE_KEY_DROP_TRY_MS``, and a key still refused
    after ``_STALE_CLONE_KEY_DROP_WAIT_SECONDS`` raises the last try's 55P03,
    which is retried.
    """
    names = [
        row[0]
        for row in conn.execute(
            text(
                "SELECT quote_ident(conname) FROM pg_constraint "
                "WHERE conrelid = to_regclass(:relation) AND contype = 'f' ORDER BY conname"
            ),
            {"relation": _qualified(partition)},
        ).all()
    ]
    conn.commit()
    for name in names:
        deadline = time.monotonic() + _STALE_CLONE_KEY_DROP_WAIT_SECONDS
        while True:
            try:
                conn.execute(text(f"SET LOCAL lock_timeout = '{_STALE_CLONE_KEY_DROP_TRY_MS}ms'"))
                conn.execute(text(f"ALTER TABLE {_qualified(partition)} DROP CONSTRAINT {name}"))
                conn.commit()
                break
            except Exception as exc:
                conn.rollback()
                if not is_lock_conflict(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)


def _drop_stale_clone(conn, kb_id: str, item_table: str) -> None:
    """Drop an unattached clone that no longer matches what the move expects.

    An unattached clone is not a partition, so a column or CHECK constraint
    added to (or changed on) the parent after a failed move left it behind
    never reaches it, and moving into it -- or attaching it -- would fail for
    good. A clone is also stale when it carries an index or a foreign key: the
    move builds those itself, inside its transaction, so one already there was
    left by an earlier layout and would make that build fail. The move is one
    transaction, so a
    clone is empty whenever no move is in flight -- and the caller holds the
    item table's build lock, so none is. A clone that holds rows anyway is
    never dropped: the move stops with an error naming it instead.

    A clone's foreign keys are dropped first, on their own and each with a
    short lock wait (``_drop_clone_foreign_keys``): dropped with the table,
    they would queue for ACCESS EXCLUSIVE on every table they reference for as
    long as the move's lock timeout, holding up that table's readers
    meanwhile. No attempt of the current design leaves such a clone -- the keys
    are only ever added in the transaction that attaches it -- so this is for
    one an earlier layout left.
    """
    partition = partition_name(kb_id, item_table)
    if _relkind(conn, partition) is None:
        return
    default = default_partition_name(item_table)
    same_shape = _column_signature(conn, partition) == _column_signature(
        conn, default
    ) and _check_constraint_signature(
        conn, partition, f"{partition}_kb_check"
    ) == _check_constraint_signature(conn, default, "")
    if same_shape and not _has_indexes_or_foreign_keys(conn, partition):
        return
    if conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM {_qualified(partition)})")).scalar():
        raise RuntimeError(
            f"{AI_SCHEMA}.{partition} is an unattached clone that no longer matches "
            f"{AI_SCHEMA}.{default_partition_name(item_table)}, and it holds rows; not "
            "dropping it. Move its rows back or drop it, then retry"
        )
    logger.warning(
        "Dropping the stale clone %s.%s: its columns, checks, indexes or keys no longer match "
        "what a move into it expects",
        AI_SCHEMA,
        partition,
    )
    _drop_clone_foreign_keys(conn, partition)
    conn.execute(text(f"SET LOCAL lock_timeout = '{MOVE_LOCK_TIMEOUT_MS}ms'"))
    conn.execute(text(partition_drop_ddl(kb_id, item_table)))


def _has_indexes_or_foreign_keys(conn, relname: str) -> bool:
    return bool(
        conn.execute(
            text(
                "SELECT EXISTS (SELECT 1 FROM pg_index WHERE indrelid = to_regclass(:relation)) "
                "OR EXISTS (SELECT 1 FROM pg_constraint "
                "WHERE conrelid = to_regclass(:relation) AND contype = 'f')"
            ),
            {"relation": _qualified(relname)},
        ).scalar()
    )


def _prepare_partition(conn, kb_id: str, item_table: str) -> None:
    """Create the bare clone and its CHECK constraint. Idempotent.

    Bounded by ``MOVE_LOCK_TIMEOUT_MS`` from its first statement: every lock
    here is on a relation other sessions use (cloning reads DEFAULT's
    definition), and this caller holds the item table's build lock meanwhile.
    A timeout raises SQLSTATE 55P03, which is retried.
    """
    partition = partition_name(kb_id, item_table)
    conn.execute(text(f"SET LOCAL lock_timeout = '{MOVE_LOCK_TIMEOUT_MS}ms'"))
    _drop_stale_clone(conn, kb_id, item_table)
    conn.execute(text(partition_create_ddl(kb_id, item_table)))
    if not _check_constraint_exists(conn, partition):
        conn.execute(text(partition_check_ddl(kb_id, item_table)))
    conn.commit()


def _default_index_definitions(conn, item_table: str) -> list[dict]:
    """How to recreate each of DEFAULT's indexes on a partition.

    One entry per valid index that is not a bm25 index, in index-name order:
    ``constraint`` (``PRIMARY KEY (id)``, ``UNIQUE ...``, ``EXCLUDE ...``) for an
    index that backs a constraint, else ``unique`` and ``tail`` -- the part of
    ``pg_get_indexdef`` after the relation name (``USING btree (source_id)``),
    which does not depend on the index's or the table's name.
    """
    default = _qualified(default_partition_name(item_table))
    rows = conn.execute(
        text(
            "SELECT pg_get_indexdef(i.indexrelid), i.indrelid::regclass::text, i.indisunique, "
            "(SELECT pg_get_constraintdef(c.oid) FROM pg_constraint c "
            " WHERE c.conindid = i.indexrelid AND c.conrelid = i.indrelid "
            " AND c.contype IN ('p', 'u', 'x')) "
            "FROM pg_index i JOIN pg_class ic ON ic.oid = i.indexrelid "
            "JOIN pg_am am ON am.oid = ic.relam "
            "WHERE i.indrelid = to_regclass(:default) AND i.indisvalid AND am.amname <> 'bm25' "
            "ORDER BY ic.relname"
        ),
        {"default": default},
    ).all()
    definitions = []
    for indexdef, regclass, unique, constraint in rows:
        if constraint is not None:
            definitions.append({"constraint": constraint})
            continue
        marker = f" ON {regclass} "
        if marker not in indexdef:
            raise RuntimeError(f"cannot read the definition of an index on {default}: {indexdef}")
        definitions.append({"unique": bool(unique), "tail": indexdef.split(marker, 1)[1]})
    return definitions


def _partition_index_state(conn, partition: str) -> tuple[set, set, set]:
    """(constraint definitions, valid index shapes, invalid index names) of a relation."""
    rows = conn.execute(
        text(
            "SELECT pg_get_indexdef(i.indexrelid), i.indrelid::regclass::text, i.indisunique, "
            "i.indisvalid, quote_ident(ic.relname), "
            "(SELECT pg_get_constraintdef(c.oid) FROM pg_constraint c "
            " WHERE c.conindid = i.indexrelid AND c.conrelid = i.indrelid "
            " AND c.contype IN ('p', 'u', 'x')) "
            "FROM pg_index i JOIN pg_class ic ON ic.oid = i.indexrelid "
            "WHERE i.indrelid = to_regclass(:partition)"
        ),
        {"partition": _qualified(partition)},
    ).all()
    constraints, shapes, invalid = set(), set(), set()
    for indexdef, regclass, unique, valid, name, constraint in rows:
        if constraint is not None:
            constraints.add(constraint)
        elif not valid:
            invalid.add(name)
        else:
            shapes.add((bool(unique), indexdef.split(f" ON {regclass} ", 1)[-1]))
    return constraints, shapes, invalid


def _build_move_indexes_and_keys(
    conn, kb_id: str, item_table: str, *, every_index: bool = False
) -> None:
    """Inside the move, after the copy: what correctness rests on, in bulk.

    Constraint-backed indexes (the local primary key) and every UNIQUE index,
    built once over the copied rows rather than maintained row by row; and the
    foreign keys, added ``NOT VALID`` so no row is re-checked -- the rows came
    from DEFAULT, where the same keys held, and the move's locks keep them
    from changing. ``_complete_partition`` validates the keys and builds the
    plain secondary indexes after the commit, without blocking writes.

    Adding a foreign key takes SHARE ROW EXCLUSIVE on the table it references
    until the move commits, which is why this runs as late as it can.

    ``every_index`` is for an empty clone (``_attach_empty_partition``): every
    index costs nothing to build there, and the keys nothing to validate.
    """
    partition = _qualified(partition_name(kb_id, item_table))
    for definition in _default_index_definitions(conn, item_table):
        if "constraint" in definition:
            conn.execute(text(f"ALTER TABLE {partition} ADD {definition['constraint']}"))
        elif definition["unique"] or every_index:
            unique = "UNIQUE " if definition["unique"] else ""
            conn.execute(text(f"CREATE {unique}INDEX ON {partition} {definition['tail']}"))
    for definition in _foreign_key_defs(conn, default_partition_name(item_table)):
        suffix = "" if definition.endswith(" NOT VALID") or every_index else " NOT VALID"
        conn.execute(text(f"ALTER TABLE {partition} ADD {definition}{suffix}"))


def _complete_partition(conn, kb_id: str, item_table: str) -> dict:
    """Finish an attached partition: validate its keys, build its plain indexes.

    Runs on an AUTOCOMMIT connection, after the move has committed, and never
    blocks writes: ``VALIDATE CONSTRAINT`` and ``CREATE INDEX CONCURRENTLY`` take
    SHARE UPDATE EXCLUSIVE on the partition. Re-entrant -- it compares the
    partition with DEFAULT each time, so a worker killed half-way leaves work
    the next ``ensure_bm25_index`` picks up. An INVALID index left by a failed
    concurrent build is dropped and rebuilt. Returns what it did.
    """
    partition = partition_name(kb_id, item_table)
    qualified = _qualified(partition)
    done: dict = {}
    conn.execute(text("SET statement_timeout = 0"))
    try:
        not_valid = conn.execute(
            text(
                "SELECT quote_ident(conname) FROM pg_constraint "
                "WHERE conrelid = to_regclass(:partition) AND contype = 'f' "
                "AND NOT convalidated ORDER BY conname"
            ),
            {"partition": qualified},
        ).all()
        for (name,) in not_valid:
            conn.execute(text(f"ALTER TABLE {qualified} VALIDATE CONSTRAINT {name}"))
            done.setdefault("validated_foreign_keys", []).append(name)

        _, shapes, invalid = _partition_index_state(conn, partition)
        if invalid and not _index_build_in_progress(conn, partition):
            for name in sorted(invalid):
                if name.startswith("bm25_"):
                    continue  # the bm25 index has its own repair
                conn.execute(text(f'DROP INDEX CONCURRENTLY IF EXISTS "{AI_SCHEMA}".{name}'))
                done.setdefault("dropped_invalid_indexes", []).append(name)
        for definition in _default_index_definitions(conn, item_table):
            if "constraint" in definition:
                continue
            if (definition["unique"], definition["tail"]) in shapes:
                continue
            unique = "UNIQUE " if definition["unique"] else ""
            conn.execute(
                text(f"CREATE {unique}INDEX CONCURRENTLY ON {qualified} {definition['tail']}")
            )
            done.setdefault("built_indexes", []).append(definition["tail"])
    finally:
        _reset_statement_timeout(conn)
    if done:
        logger.info("Completed partition %s.%s: %s", AI_SCHEMA, partition, done)
    return done


def _move_check_names(conn, item_table: str) -> list[str]:
    """Temporary move checks currently on this item table's DEFAULT partition."""
    return [
        row[0]
        for row in conn.execute(
            text(
                "SELECT c.conname FROM pg_constraint c "
                "JOIN pg_class t ON t.oid = c.conrelid "
                "JOIN pg_namespace n ON n.oid = t.relnamespace "
                "WHERE n.nspname = :schema AND t.relname = :relname AND c.contype = 'c' "
                "AND c.conname LIKE :prefix ORDER BY c.conname"
            ),
            {
                "schema": AI_SCHEMA,
                "relname": default_partition_name(item_table),
                "prefix": _DEFAULT_MOVE_CHECK_PREFIX.replace("_", "\\_") + "%",
            },
        ).all()
    ]


def partition_lock_default_exclusive_ddl(item_table: str) -> str:
    """One try for ACCESS EXCLUSIVE on the DEFAULT partition, never queueing."""
    return (
        f"LOCK TABLE {_qualified(default_partition_name(item_table))} "
        "IN ACCESS EXCLUSIVE MODE NOWAIT"
    )


def _default_holder_is_waiting(conn, item_table: str) -> bool:
    """Does any other session holding a lock on DEFAULT wait for a lock itself?

    Such a session may be waiting -- directly or through others -- on the
    caller, and then a queued request for DEFAULT would close a lock cycle.

    ``pg_locks`` covers the whole cluster, and a relation OID is only unique
    within one database (a database copied from a template keeps the
    template's), so the DEFAULT lock must be a relation lock of this database.
    The waiting lock belongs to the same backend; it is matched on the database
    too where it has one -- this database, or 0 for a shared catalog -- while a
    lock with no database (a transaction id, a virtual transaction id) is
    matched by the backend alone.
    """
    this_database = "(SELECT oid FROM pg_database WHERE datname = current_database())"
    return bool(
        conn.execute(
            text(
                "SELECT EXISTS (SELECT 1 FROM pg_locks held "
                "JOIN pg_locks waiting ON waiting.pid = held.pid AND NOT waiting.granted "
                f"AND (waiting.database IS NULL OR waiting.database IN (0, {this_database})) "
                "WHERE held.locktype = 'relation' "
                f"AND held.database = {this_database} "
                "AND held.relation = to_regclass(:default) AND held.granted "
                "AND held.pid <> pg_backend_pid())"
            ),
            {"default": _qualified(default_partition_name(item_table))},
        ).scalar()
    )


def _has_app_context() -> bool:
    try:
        from flask import has_app_context

        return has_app_context()
    except Exception:
        return False


def _long_holder_seconds() -> int:
    """``BM25_MOVE_LONG_HOLDER_SECONDS``, or its registry default without an app.

    Read per move. Outside an application context there is no settings table
    to read, so the registry default applies.
    """
    from . import settings_registry

    definition = settings_registry.SETTINGS_REGISTRY["BM25_MOVE_LONG_HOLDER_SECONDS"]
    if not _has_app_context():
        return int(definition.default)
    try:
        value = int(settings_registry.get_setting("BM25_MOVE_LONG_HOLDER_SECONDS"))
    except Exception:
        return int(definition.default)
    return max(int(definition.min), min(int(definition.max), value))


def _default_has_a_long_holder(conn, item_table: str, held_for_seconds: float) -> bool:
    """Does a transaction that began at least ``held_for_seconds`` ago hold DEFAULT?

    The age is absolute -- how long the holding transaction has been open --
    not how long the caller has been probing. A lock of a prepared transaction
    has no backend, and counts as long.
    """
    this_database = "(SELECT oid FROM pg_database WHERE datname = current_database())"
    return bool(
        conn.execute(
            text(
                "SELECT EXISTS (SELECT 1 FROM pg_locks l "
                "LEFT JOIN pg_stat_activity a ON a.pid = l.pid "
                "WHERE l.locktype = 'relation' "
                f"AND l.database = {this_database} "
                "AND l.relation = to_regclass(:default) AND l.granted "
                "AND l.pid IS DISTINCT FROM pg_backend_pid() "
                "AND (l.pid IS NULL OR a.xact_start <= "
                "clock_timestamp() - make_interval(secs => :seconds)))"
            ),
            {
                "default": _qualified(default_partition_name(item_table)),
                "seconds": held_for_seconds,
            },
        ).scalar()
    )


def _probe_default_before_moving(conn, item_table: str) -> None:
    """Raise SQLSTATE 55P03 if a long transaction holds DEFAULT; else return.

    See ``DEFAULT_PREFLIGHT_WAIT_SECONDS``. Runs on a connection holding no
    lock, each try in a transaction of its own that is rolled back straight
    away, so a granted try blocks nobody for longer than the round trip and a
    refused one blocks nobody at all.
    """
    started = time.monotonic()
    deadline = started + DEFAULT_PREFLIGHT_WAIT_SECONDS
    sleep = _EXCLUSIVE_LOCK_FIRST_SLEEP_SECONDS
    while True:
        try:
            conn.execute(text(partition_lock_default_exclusive_ddl(item_table)))
        except Exception as exc:
            conn.rollback()
            if not is_lock_conflict(exc):
                raise
            refusal = exc
        else:
            conn.rollback()
            return
        if time.monotonic() + sleep > deadline:
            break
        time.sleep(sleep)
        sleep = min(sleep * 2, _EXCLUSIVE_LOCK_MAX_SLEEP_SECONDS)
    long_holder = _default_has_a_long_holder(conn, item_table, _long_holder_seconds())
    conn.rollback()
    if long_holder:
        raise refusal


def _queued_try_ms(conn) -> int:
    """The queued try's lock_timeout: the constant, capped at deadlock_timeout / 2."""
    deadlock_ms = conn.execute(
        text("SELECT setting::int FROM pg_settings WHERE name = 'deadlock_timeout'")
    ).scalar()
    cap = int(deadlock_ms) // 2 if deadlock_ms else DEFAULT_EXCLUSIVE_QUEUED_TRY_MS
    return max(1, min(DEFAULT_EXCLUSIVE_QUEUED_TRY_MS, cap))


def _lock_default_exclusively(conn, item_table: str, wait_seconds: float) -> None:
    """Take ACCESS EXCLUSIVE on DEFAULT without stalling readers or closing a cycle.

    Tries ``NOWAIT`` inside a savepoint, so a refusal leaves the caller's
    transaction usable, and sleeps with a doubling backoff between tries. Once
    ``_EXCLUSIVE_LOCK_QUEUED_TRY_AFTER_SECONDS`` of refusals have passed, it
    makes one queued try bounded by ``_queued_try_ms`` -- unless a holder of
    DEFAULT is waiting for a lock (``_default_holder_is_waiting``), or the
    remaining time is too short. After ``wait_seconds`` the last refusal
    (SQLSTATE 55P03) is raised. See ``DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS``.
    """
    started = time.monotonic()
    deadline = started + wait_seconds
    sleep = _EXCLUSIVE_LOCK_FIRST_SLEEP_SECONDS
    queued_try_done = False
    while True:
        conn.execute(text("SAVEPOINT bm25_default_lock"))
        try:
            conn.execute(text(partition_lock_default_exclusive_ddl(item_table)))
        except Exception as exc:
            conn.execute(text("ROLLBACK TO SAVEPOINT bm25_default_lock"))
            conn.execute(text("RELEASE SAVEPOINT bm25_default_lock"))
            if not is_lock_conflict(exc):
                raise
            refusal = exc
        else:
            conn.execute(text("RELEASE SAVEPOINT bm25_default_lock"))
            return
        now = time.monotonic()
        if not queued_try_done and now - started >= _EXCLUSIVE_LOCK_QUEUED_TRY_AFTER_SECONDS:
            queued_try_done = True
            queued_ms = _queued_try_ms(conn)
            if now + queued_ms / 1000 <= deadline and not _default_holder_is_waiting(
                conn, item_table
            ):
                if _queued_lock_try(conn, item_table, queued_ms):
                    return
                continue
        if time.monotonic() + sleep > deadline:
            raise refusal
        time.sleep(sleep)
        sleep = min(sleep * 2, _EXCLUSIVE_LOCK_MAX_SLEEP_SECONDS)


def _queued_lock_try(conn, item_table: str, lock_timeout_ms: int) -> bool:
    """One queued request for ACCESS EXCLUSIVE on DEFAULT, bounded by a lock_timeout.

    The timeout is set in a savepoint and put back afterwards, so the caller's
    transaction keeps its own. Returns whether the lock was taken.

    Not deadlock-free. A session blocked on the caller's parent lock can have
    its one-time deadlock check fire while this request waits, and a holder of
    DEFAULT that began waiting on that session after ``_default_holder_is_waiting``
    looked closes a three-party cycle within this window: Postgres aborts the
    blocked session, whose deadlock check is the one that finds it, with
    SQLSTATE 40P01. No row is lost -- ``index_source`` requeues, an API writer
    receives the error.
    It is left open because the precise guard, no queued try while anything
    waits on the move, would starve large moves under steady writes (see
    ``DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS``).
    """
    previous = conn.execute(text("SELECT current_setting('lock_timeout')")).scalar()
    conn.execute(text("SAVEPOINT bm25_default_lock"))
    try:
        conn.execute(text(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms'"))
        conn.execute(
            text(
                f"LOCK TABLE {_qualified(default_partition_name(item_table))} IN ACCESS EXCLUSIVE MODE"
            )
        )
    except Exception as exc:
        conn.execute(text("ROLLBACK TO SAVEPOINT bm25_default_lock"))
        conn.execute(text("RELEASE SAVEPOINT bm25_default_lock"))
        if not is_lock_conflict(exc):
            raise
        return False
    conn.execute(text("RELEASE SAVEPOINT bm25_default_lock"))
    conn.execute(text("SELECT set_config('lock_timeout', :value, true)"), {"value": previous})
    return True


def _drop_move_checks(
    conn, item_table: str, names: list[str], wait_seconds: float | None = None
) -> None:
    """Drop temporary DEFAULT checks, each in its own short transaction.

    DROP CONSTRAINT takes ACCESS EXCLUSIVE on DEFAULT for a catalog change
    only; the lock is taken without queueing, within ``wait_seconds``
    (default ``DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS``).
    """
    if wait_seconds is None:
        wait_seconds = DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS
    for name in names:
        _lock_default_exclusively(conn, item_table, wait_seconds)
        conn.execute(text(default_move_check_drop_ddl(item_table, name)))
        conn.commit()


def clear_leftover_move_checks(engine, item_table: str, wait_seconds: float) -> list[str]:
    """Drop every temporary move check on this table's DEFAULT that no move is using.

    A move's check is only in use while that move holds the item table's
    build lock, so this takes the lock -- without waiting: if a move holds it,
    its check is live and nothing is dropped. A check found under the lock was
    left by a move that failed to drop it (a reader outlived its cleanup) or
    never got the chance (its worker was killed), and it refuses every write of
    its knowledge base routed to DEFAULT until it is gone.

    Returns the names dropped. Raises SQLSTATE 55P03 if DEFAULT's lock is not
    free within ``wait_seconds``.
    """
    relation = partition_build_lock_relation(item_table)
    with engine.connect() as conn:
        acquired = conn.execute(text(partition_build_lock_sql()), {"relation": relation}).scalar()
        conn.commit()
        if not acquired:
            logger.info(
                "Leftover move checks on %s.%s not cleared: a move holds the table's build "
                "lock, so its check is live",
                AI_SCHEMA,
                default_partition_name(item_table),
            )
            return []
        try:
            names = _move_check_names(conn, item_table)
            conn.commit()
            _drop_move_checks(conn, item_table, names, wait_seconds)
            return names
        finally:
            _release_partition_build_lock(conn, item_table)


def clear_leftover_move_checks_at_start(engine) -> dict[str, list[str] | str]:
    """Start-up sweep of leftover move checks, for every partitioned item table.

    Cannot block start-up: the build lock is only tried, DEFAULT's lock gets a
    single ``NOWAIT`` try (no queued try), and everything else is a catalog
    read or the DROP CONSTRAINT made under that lock. Never raises. Per table the outcome is the list of checks
    dropped, ``"busy"`` (a move or a reader was in the way; the next
    ``ensure_bm25_index`` on the table clears it), ``"not_partitioned"``, or
    ``"error"``.
    """
    outcomes: dict[str, list[str] | str] = {}
    for item_table in sorted(PARTITIONED_ITEM_TABLES):
        try:
            with engine.connect() as conn:
                partitioned = table_is_partitioned(conn, item_table) and (
                    _relkind(conn, default_partition_name(item_table)) is not None
                )
                conn.rollback()
            if not partitioned:
                outcomes[item_table] = "not_partitioned"
                continue
            outcomes[item_table] = clear_leftover_move_checks(engine, item_table, 0.0)
        except Exception as exc:
            if is_lock_conflict(exc):
                outcomes[item_table] = "busy"
                logger.info(
                    "A leftover move check on %s.%s could not be dropped at start-up: DEFAULT "
                    "is in use. The next BM25 index build on the table will drop it",
                    AI_SCHEMA,
                    default_partition_name(item_table),
                )
            else:
                outcomes[item_table] = "error"
                logger.warning(
                    "Could not check %s.%s for leftover move checks at start-up: %s",
                    AI_SCHEMA,
                    item_table,
                    first_error_line(exc),
                )
            continue
        if outcomes[item_table]:
            logger.warning(
                "Dropped leftover move checks %s from %s.%s at start-up",
                outcomes[item_table],
                AI_SCHEMA,
                default_partition_name(item_table),
            )
    return outcomes


def move_check_refusal_item_table(exc: BaseException) -> str | None:
    """The item table whose DEFAULT refused a write with a move check, or None.

    Read from the error's diagnostics -- the constraint, table and schema names
    Postgres reports -- so only a write refused by a ``bm25_move_<kb>`` check on
    one of this schema's DEFAULT partitions matches. Not every server sends
    those fields: a build with ``pg_search`` first in
    ``shared_preload_libraries`` was seen to send none of them. Then the names
    are read from the message instead, where Postgres quotes them verbatim (an
    untranslated message is needed for that; a translated one is not traced).
    """
    orig = getattr(exc, "orig", None)
    if getattr(orig, "sqlstate", None) != "23514":
        return None
    diag = getattr(orig, "diag", None)
    constraint = getattr(diag, "constraint_name", None)
    table = getattr(diag, "table_name", None)
    schema = getattr(diag, "schema_name", None)
    if constraint is None and table is None and schema is None:
        match = re.search(
            r'relation "([^"]+)" violates check constraint "([^"]+)"', first_error_line(exc)
        )
        if match is None:
            return None
        table, constraint = match.groups()
        schema = AI_SCHEMA
    if not constraint or not re.fullmatch(
        rf"{_DEFAULT_MOVE_CHECK_PREFIX}[0-9a-f]{{32}}", constraint
    ):
        return None
    if schema != AI_SCHEMA:
        return None
    for item_table in sorted(PARTITIONED_ITEM_TABLES):
        if table == default_partition_name(item_table):
            return item_table
    return None


def clear_move_check_after_refusal(engine, exc: BaseException) -> list[str]:
    """After a write was refused by a move check, try once to clear leftover checks.

    For the indexing path, so a check a failed move could not drop does not
    refuse every retry of that knowledge base's indexing until the next index
    build. Cannot block and never raises: the build lock is only tried (a move
    holding it owns a live check, which is left alone), DEFAULT's lock gets a
    single ``NOWAIT`` try, and any failure is logged and swallowed -- the
    caller requeues either way. Returns the names dropped.
    """
    item_table = move_check_refusal_item_table(exc)
    if item_table is None:
        logger.debug(
            "A refused write was not traced to a move check on a DEFAULT partition; "
            "nothing to clear: %s",
            first_error_line(exc),
        )
        return []
    try:
        dropped = clear_leftover_move_checks(engine, item_table, 0.0)
    except Exception as clear_exc:
        logger.info(
            "Could not clear the move check that refused a write on %s.%s (%s); the "
            "write is retried later",
            AI_SCHEMA,
            default_partition_name(item_table),
            first_error_line(clear_exc),
        )
        return []
    if dropped:
        logger.warning(
            "Dropped leftover move checks %s from %s.%s after they refused a write",
            dropped,
            AI_SCHEMA,
            default_partition_name(item_table),
        )
    else:
        logger.info(
            "No move check dropped from %s.%s after one refused a write; the write is "
            "retried later",
            AI_SCHEMA,
            default_partition_name(item_table),
        )
    return dropped


def _drop_failed_move_check(
    conn, kb_id: str, item_table: str, gate: _MoveGate | None = None
) -> None:
    """After a failed move, keep trying to drop its check for a bounded time.

    Never raises: the move's own error is the one the caller needs. A check
    that cannot be dropped refuses this knowledge base's writes routed to
    DEFAULT until ``clear_leftover_move_checks`` runs -- after the next
    indexing write it refuses, at the next ``ensure_bm25_index`` on the item
    table, or at start-up.

    With the move ``gate`` still held, the first tries (up to
    ``DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS``) run under it, when no indexing
    transaction on the table can be holding DEFAULT, so they usually succeed at
    once. If they do not -- an ungated reader is in the way, the usual reason
    the move failed -- the gate is released before the rest of the wait, so
    the table's indexing is not held off for it. That is safe: the gate only
    keeps indexing out of the move, and the move is over. An indexing write of
    this knowledge base routed to DEFAULT meanwhile is refused by the check
    (SQLSTATE 23514) and requeued, as it would be after the gate.
    """
    fence = default_move_check_name(kb_id)
    try:
        if gate is not None and gate.held:
            try:
                _drop_move_checks(conn, item_table, [fence], DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS)
                return
            except Exception:
                conn.rollback()
                gate.release()
        _drop_move_checks(conn, item_table, [fence], MOVE_CHECK_CLEANUP_WAIT_SECONDS)
    except Exception as exc:
        conn.rollback()
        logger.warning(
            "Could not drop the temporary check %s from %s.%s within %.1f s (%s). Writes of "
            "KB %s routed to it fail until it is dropped: after the next indexing write it "
            "refuses, by the next BM25 index build on this table, or at start-up",
            fence,
            AI_SCHEMA,
            default_partition_name(item_table),
            MOVE_CHECK_CLEANUP_WAIT_SECONDS,
            first_error_line(exc),
            kb_id,
        )


def _kb_rows_in_default(conn, kb_id: str, item_table: str) -> bool:
    return bool(
        conn.execute(
            text(
                f"SELECT EXISTS (SELECT 1 FROM {_qualified(default_partition_name(item_table))} "
                "WHERE knowledge_base_id = CAST(:kb AS uuid))"
            ),
            {"kb": kb_id},
        ).scalar()
    )


class RowMoveNotAllowed(RuntimeError):
    """The knowledge base has rows in DEFAULT, and this caller may not move them.

    Moving rows holds writes to the whole item table for the length of the
    move, so it is only done when an operator asked for it (``POST
    /build-bm25``). An ensure dispatched automatically -- at knowledge base
    creation, or by a PATCH -- only ever attaches an empty knowledge base's
    partition, and raises this when rows turned up in the meantime (its
    sources were indexed while it waited or retried).
    """


def _attach_empty_partition(
    engine, conn, kb_id: str, item_table: str, gate: _MoveGate
) -> dict | None:
    """Attach the partition of a knowledge base with no rows in DEFAULT.

    Nothing has to move, so nothing needs writers held off the parent: the
    check that lets ATTACH skip its scan of DEFAULT is added ``NOT VALID`` and
    committed (a brief ACCESS EXCLUSIVE try on DEFAULT), validated in a
    transaction of its own (SHARE UPDATE EXCLUSIVE: readers and writers carry
    on while DEFAULT is scanned), and then the attaching transaction builds
    every index on the still-empty clone and adds its foreign keys (both free
    there), takes a second brief ACCESS EXCLUSIVE try on DEFAULT, attaches,
    drops the check and commits. A write of this knowledge base routed to
    DEFAULT in the meantime is refused by the check (SQLSTATE 23514, which
    indexing requeues).

    Runs under the move gate (``gate``, held by the caller), like a move: the
    lock tries on DEFAULT would otherwise have to find a gap between the
    table's indexing transactions, and under steady indexing they find none.

    The indexes and keys are added only in the attaching transaction, so an
    attempt that gives up at any step leaves a bare clone behind: dropping a
    clone with foreign keys would take ACCESS EXCLUSIVE on the tables they
    reference, which holds up their readers.

    Returns ``None`` when a row of the knowledge base reached DEFAULT before
    the check went up (it then fails to validate): the check is dropped, and
    the caller moves the rows the ordinary way. Called under the item table's
    build lock, with the bare clone prepared.
    """
    partition = partition_name(kb_id, item_table)
    fence = default_move_check_name(kb_id)
    fence_sql = default_move_check_add_ddl(kb_id, item_table)
    fence_committed = False
    blocked = 0.0
    step = "fence"
    try:
        started = time.monotonic()
        _lock_default_exclusively(conn, item_table, DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS)
        conn.execute(text(fence_sql))
        fence_committed = True
        conn.commit()
        blocked += time.monotonic() - started

        step = "validate"
        conn.execute(text(f"SET LOCAL lock_timeout = '{MOVE_LOCK_TIMEOUT_MS}ms'"))
        conn.execute(text("SET LOCAL statement_timeout = 0"))
        try:
            conn.execute(text(default_move_check_validate_ddl(kb_id, item_table)))
        except Exception as exc:
            if getattr(getattr(exc, "orig", None), "sqlstate", None) != "23514":
                raise
            conn.rollback()
            logger.info(
                "A row of KB %s reached %s.%s before its check went up; not attaching an "
                "empty partition",
                kb_id,
                AI_SCHEMA,
                default_partition_name(item_table),
            )
            _drop_move_checks(conn, item_table, [fence])
            return None
        conn.commit()

        step = "indexes, keys and attach"
        attach_sql = partition_attach_ddl(kb_id, item_table)
        conn.execute(text(f"SET LOCAL lock_timeout = '{MOVE_LOCK_TIMEOUT_MS}ms'"))
        # Before DEFAULT's lock, not after: adding the keys waits (bounded) for
        # SHARE ROW EXCLUSIVE on the tables they reference, and must not do that
        # while holding every reader off DEFAULT.
        _build_move_indexes_and_keys(conn, kb_id, item_table, every_index=True)
        started = time.monotonic()
        _lock_default_exclusively(conn, item_table, DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS)
        conn.execute(text(attach_sql))
        conn.execute(text(default_move_check_drop_ddl(item_table, fence)))
        conn.execute(
            text(mirror_relation_settings_sql(_qualified(item_table), _qualified(partition)))
        )
        conn.commit()
        blocked += time.monotonic() - started
    except Exception as exc:
        _fail_move(
            engine,
            conn,
            kb_id,
            item_table,
            exc,
            step=step,
            action="Attaching the empty partition of",
        )
        if fence_committed:
            _drop_failed_move_check(conn, kb_id, item_table, gate)
        raise
    logger.info(
        "Attached empty partition %s.%s without moving rows; writes to its DEFAULT partition "
        "waited at most %.3f s",
        AI_SCHEMA,
        partition,
        blocked,
    )
    return {"rows_moved": 0, "writes_blocked_seconds": blocked}


def create_partition(
    engine, knowledge_base_id: Any, item_table: str, *, allow_row_move: bool = True
) -> dict:
    """Move one knowledge base into a partition of its own, atomically.

    Returns ``{"rows_moved": int, "writes_blocked_seconds": float}``.

    A knowledge base with no rows in DEFAULT is attached without a move
    (``_attach_empty_partition``). With ``allow_row_move=False`` that is all
    this does: a knowledge base with rows in DEFAULT raises
    ``RowMoveNotAllowed`` instead of being moved. Both paths start by taking
    the item table's move gate (see ``move_gate_relation``).

    1. **prepare** (own transaction) -- clone the DEFAULT partition into an
       unattached bare heap (no index, no foreign key) with a CHECK matching
       the partition bound (so ATTACH skips its scan of the new partition).
       Any temporary DEFAULT check a failed or killed move left is dropped,
       and so is a stale clone (``_drop_stale_clone``).
    2. **move** (one transaction) -- SHARE on the parent; meanwhile, on a
       second connection, add ``CHECK (knowledge_base_id <> kb) NOT VALID`` to
       DEFAULT (a catalog change) and commit it; SHARE on DEFAULT; copy the
       KB's rows into the clone and delete them from DEFAULT; VALIDATE the
       DEFAULT check (one scan of DEFAULT, which does not block readers);
       build the primary key and any UNIQUE index in bulk and add the foreign
       keys ``NOT VALID`` (``_build_move_indexes_and_keys``); ATTACH, which
       now needs neither of its scans; drop the DEFAULT check under the lock
       the ATTACH already holds; mirror ownership, grants, RLS and policies;
       commit.
    3. **complete** (``ensure_bm25_index``, after the bm25 index) -- validate
       the foreign keys and build the plain secondary indexes ``CONCURRENTLY``
       (``_complete_partition``); neither blocks writes.

    Every ACCESS EXCLUSIVE lock on DEFAULT (adding the check, the ATTACH) is
    taken by ``_lock_default_exclusively``: ``NOWAIT`` tries for up to
    ``DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS`` and at most one short queued try,
    never while a holder of DEFAULT is waiting for a lock. A reader that stays
    in the way makes the *move* give up (SQLSTATE 55P03), instead of stalling
    new readers for long or deadlocking a transaction that read DEFAULT and
    then writes through the parent. And before step 2 takes the parent lock,
    ``_probe_default_before_moving`` checks for a transaction that already
    holds DEFAULT and would refuse all of those tries: with one there, the move
    gives up the same way without holding writers off at all.

    The bm25 index is built afterwards with CREATE INDEX CONCURRENTLY, outside
    any of this.

    Who waits, measured on the production schema (the item table as the
    migrations leave it: primary key, three foreign keys, three btree indexes
    and a full-text GIN index) on Postgres 15 with pg_search in Docker (128 MB
    shared_buffers, a 2.54 million-row, 2.0 GB DEFAULT in a warm OS page
    cache, ~700-byte rows). Cold caches, slower disks and wider rows are
    slower.

    * **writers** through the parent, for every knowledge base on this item
      table, for all of step 2. A 1 million-row knowledge base: 4.2-8.8 s,
      typically 5.5 s (copy 2.9 s, delete from DEFAULT 2.0 s, VALIDATE 0.3 s,
      primary key 0.3-0.5 s) -- against 59.5-62.6 s when the clone carried
      its indexes and foreign keys during the copy, almost all of it the GIN
      index, and 48-63 s with every index built in bulk inside the move. A
      40 000-row knowledge base: 0.44-0.75 s (was 2.8-3.0 s), most of it the
      VALIDATE scan, which grows with DEFAULT. A knowledge base with no rows
      in DEFAULT takes no parent lock at all (``_attach_empty_partition``):
      writers waited at most 8 ms while one was attached, against 0.34 s
      before. A move that gives up still held writers for as long as it ran:
      its copy time plus at most ``DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS`` of
      lock tries per step (1.0-1.3 s in all for the 40 000-row move when a
      long reader arrives mid-move). A long reader already there when the
      move starts costs them nothing: the move gives up before taking the
      parent lock.
    * **after the commit**, nothing blocks writers: the bm25 index is built
      concurrently (6.5 s for a million rows), then the plain secondary
      indexes (the btrees 1.8 s, the GIN index 47 s) and the foreign keys are
      validated (0.5 s). See the first accepted trade-off below for what
      keyword search does meanwhile.
    * **readers** do not block on the SHARE locks. They can wait on the
      ACCESS EXCLUSIVE steps on DEFAULT (the check going up, the ATTACH, the
      check's drop after a failed move): new readers of DEFAULT, and queries
      through the parent that cannot prune DEFAULT, queue behind the one
      queued try each step may make, so they wait at most
      ``DEFAULT_EXCLUSIVE_QUEUED_TRY_MS`` per step. Measured: 4-8 ms with no
      long reader (the ATTACH itself takes about 1 ms), 0.20 s with a long
      reader present, 44-95 ms under 8-32 overlapping readers.

    Two trade-offs are accepted with this design:

    * **No keyword index right after the move.** Once step 2 commits, the
      knowledge base's rows are in a partition with no bm25 index yet and none
      of the plain secondary indexes, the full-text GIN index among them. Until
      the bm25 index's concurrent build finishes -- roughly 8 s per million
      rows moved (6.5 s of it the build itself) -- its keyword leg falls back to
      the bounded tsvector path, which scans the partition and, at that size,
      runs out of its time budget: hybrid search answers from vectors only, and
      full_text search returns the keyword-timeout 503. Nothing is wrong with
      the data; the answers come back once the index is ready.
    * **A deadlock with an ungated writer of a referenced table.** Adding the
      foreign keys ``NOT VALID`` takes SHARE ROW EXCLUSIVE on
      ``knowledge_bases``, ``sources`` and ``indexed_sources`` while the move
      holds SHARE on the parent and DEFAULT. A transaction that does not take
      the move gate and writes one of those tables and then an item table --
      a cascade delete of a knowledge base or source is exactly that -- closes
      a lock cycle with the move. Postgres aborts one side with SQLSTATE 40P01:
      the side whose one-time deadlock check (``deadlock_timeout`` after it
      began waiting) runs first once the cycle exists. That is the move when
      it was the first to wait, or when the writer began waiting more than
      ``deadlock_timeout`` before the move reached its keys; it is the writer
      when the writer began waiting less than ``deadlock_timeout`` before
      that (measured: a writer waiting 0.3 s before the keys lost; 1.5 s
      before, the move lost). Either way the loser rolls back whole and nothing
      is lost: the move's task retries it, and the writer's caller receives the
      error (a re-index requeues its source).

    Why this order. The DEFAULT check cannot be validated while any of the
    knowledge base's rows are still in DEFAULT, so VALIDATE has to follow the
    DELETE inside the move. Adding the check inside the move's own transaction
    would take ACCESS EXCLUSIVE on DEFAULT there and block readers for the rest
    of the move. Adding it in a transaction of its own *before* taking the
    parent lock would leave a gap in which a write of this knowledge base
    routed to DEFAULT fails the check (SQLSTATE 23514) instead of waiting;
    adding it on a second connection while the parent lock is already held
    closes that gap, because no writer can reach DEFAULT through the parent
    until the move commits.

    Because step 2 is one transaction, every reader sees the knowledge base's
    rows either all in DEFAULT or all in the partition, and writers through the
    parent wait for the commit and then plan against the new partition list
    (see ``partition_lock_parent_ddl``). That is the deliberate choice over the
    earlier online design, which kept writers going but silently lost their
    UPDATEs and DELETEs.

    Serialised per item table by a session-scoped advisory lock: two
    concurrent moves out of one DEFAULT partition deadlock each other.

    Re-entrant: a failure in step 2 rolls the move back, leaving the clone
    empty. If the check had already been committed, the move keeps trying to
    drop it for up to ``MOVE_CHECK_CLEANUP_WAIT_SECONDS`` -- until then this
    KB's writes routed to DEFAULT fail it (SQLSTATE 23514, which indexing
    re-queues). The check is *not* always gone when the call returns: a reader
    that outlives that wait, or a worker killed between the check's commit and
    the move's, leaves it behind, and ``clear_leftover_move_checks`` drops it
    after the next indexing write it refuses, at the next ``ensure_bm25_index``
    on this item table, the next move, or start-up.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    partition = partition_name(kb_id, item_table)
    default = default_partition_name(item_table)
    fence = default_move_check_name(kb_id)

    with engine.connect() as conn:
        _acquire_partition_build_lock(conn, item_table)
        gate = _MoveGate(conn, item_table)
        try:
            # Under the build lock no other move is in flight, so every move
            # check still on DEFAULT is a crashed move's leftover -- and one
            # that refuses some knowledge base's writes until it is gone.
            _drop_move_checks(conn, item_table, _move_check_names(conn, item_table))
            if _partition_is_attached(conn, kb_id, item_table):
                return {"rows_moved": 0, "writes_blocked_seconds": 0.0}

            step = "prepare"
            try:
                _prepare_partition(conn, kb_id, item_table)
                step = "move gate"
                gate.acquire()
            except Exception as exc:
                _fail_move(engine, conn, kb_id, item_table, exc, step=step)
                raise
            # Under the gate, so no indexing transaction on the table adds rows
            # of this knowledge base between this check and the attach.
            has_rows = _kb_rows_in_default(conn, kb_id, item_table)
            conn.commit()
            if not has_rows:
                attached = _attach_empty_partition(engine, conn, kb_id, item_table, gate)
                if attached is not None:
                    return attached
            if not allow_row_move:
                raise RowMoveNotAllowed(
                    f"knowledge base {kb_id} has rows in {AI_SCHEMA}.{default}; moving them "
                    "blocks writes to the whole item table, and this build was not asked to"
                )
            insert_sql, delete_sql = move_rows_sql(
                kb_id, item_table, _insertable_columns(conn, default)
            )
            conn.commit()

            fence_committed = False
            step = "pre-flight check of DEFAULT"
            try:
                _probe_default_before_moving(conn, item_table)
                step = "parent lock"
                conn.execute(text(f"SET LOCAL lock_timeout = '{MOVE_LOCK_TIMEOUT_MS}ms'"))
                # A role's or database's statement_timeout must not cancel a
                # large copy half-way; every wait in here is bounded already.
                conn.execute(text("SET LOCAL statement_timeout = 0"))
                conn.execute(text(partition_lock_parent_ddl(item_table)))
                started = time.monotonic()
                step = "check on DEFAULT"
                # The fence goes up on a second connection while this one holds
                # writers off the parent: it has to commit before the move can
                # validate it, and committing it here would release the lock and
                # open a gap in which this KB's writes hit it and fail.
                with engine.connect() as fencer:
                    _lock_default_exclusively(
                        fencer, item_table, DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS
                    )
                    fencer.execute(text(default_move_check_add_ddl(kb_id, item_table)))
                    # Before the commit, not after: a commit that reaches the
                    # server can still raise on the client, and then the check
                    # is up. A false positive costs one try of an IF EXISTS drop.
                    fence_committed = True
                    fencer.commit()
                step = "DEFAULT lock"
                conn.execute(text(partition_lock_default_ddl(item_table)))
                step = "copy"
                moved = conn.execute(text(insert_sql), {"kb": kb_id}).rowcount
                step = "delete from DEFAULT"
                conn.execute(text(delete_sql), {"kb": kb_id})
                step = "validate the check on DEFAULT"
                conn.execute(text(default_move_check_validate_ddl(kb_id, item_table)))
                step = "key, unique indexes and foreign keys"
                _build_move_indexes_and_keys(conn, kb_id, item_table)
                step = "attach"
                attach_sql = partition_attach_ddl(kb_id, item_table)
                _lock_default_exclusively(conn, item_table, DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS)
                conn.execute(text(attach_sql))
                # Inside the move, under the ACCESS EXCLUSIVE the ATTACH already
                # holds: dropping the check costs no further lock wait, and it
                # commits with the move, so a success never leaves it behind.
                conn.execute(text(default_move_check_drop_ddl(item_table, fence)))
                step = "mirror settings"
                conn.execute(
                    text(
                        mirror_relation_settings_sql(_qualified(item_table), _qualified(partition))
                    )
                )
                step = "commit"
                conn.commit()
                blocked = time.monotonic() - started
            except Exception as exc:
                _fail_move(engine, conn, kb_id, item_table, exc, step=step)
                # Only a check that was committed needs dropping; trying anyway
                # would take DEFAULT's lock again for nothing.
                if fence_committed:
                    _drop_failed_move_check(conn, kb_id, item_table, gate)
                raise
        finally:
            gate.release()
            _release_partition_build_lock(conn, item_table)

        # Statistics for the new partition straight away, outside the build
        # lock: autovacuum would get there, but the first searches would plan
        # without them. Best effort -- the move itself is already done.
        try:
            conn.execute(text(f"ANALYZE {_qualified(partition)}"))
            conn.commit()
        except Exception:
            conn.rollback()
            logger.warning("Could not ANALYZE %s.%s", AI_SCHEMA, partition, exc_info=True)

    logger.info(
        "Moved %d rows from %s into partition %s.%s; writes were blocked for %.3f s",
        moved,
        default,
        AI_SCHEMA,
        partition,
        blocked,
    )
    return {"rows_moved": moved, "writes_blocked_seconds": blocked}


def drop_partition(engine, knowledge_base_id: Any, item_table: str) -> bool:
    """Detach and drop one KB's partition, returning any rows to DEFAULT first.

    Meant for a deleted knowledge base, whose partition is already empty (its
    rows went with the cascade), so the copy back costs nothing. It is safe on
    a knowledge base that still exists -- no row is lost, the KB just goes back
    to the fallback keyword path -- but not cheap: DETACH holds ACCESS
    EXCLUSIVE on the parent until the commit, so every read and write of the
    item table waits while the rows are copied back.

    The wait for that lock is bounded by ``MOVE_LOCK_TIMEOUT_MS``; a timeout
    rolls back, logs who held the table, and raises a transient error for the
    caller to retry.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    partition = partition_name(kb_id, item_table)
    default = default_partition_name(item_table)

    with engine.connect() as conn:
        # Same lock as the build: a DETACH and a concurrent move both want
        # ACCESS EXCLUSIVE on the DEFAULT partition.
        _acquire_partition_build_lock(conn, item_table)
        try:
            if _relkind(conn, partition) is None:
                return False
            try:
                conn.execute(text(f"SET LOCAL lock_timeout = '{MOVE_LOCK_TIMEOUT_MS}ms'"))
                if _partition_is_attached(conn, kb_id, item_table):
                    conn.execute(text(partition_detach_ddl(kb_id, item_table)))
                names = ", ".join(_insertable_columns(conn, default))
                survivors = conn.execute(
                    text(
                        f"INSERT INTO {_qualified(default)} ({names}) "
                        f"SELECT {names} FROM {_qualified(partition)}"
                    )
                ).rowcount
                conn.execute(text(partition_drop_ddl(kb_id, item_table)))
                conn.commit()
            except Exception as exc:
                _fail_move(
                    engine,
                    conn,
                    kb_id,
                    item_table,
                    exc,
                    step="detach",
                    action="Dropping the partition of",
                )
                raise
        finally:
            _release_partition_build_lock(conn, item_table)

    if survivors:
        logger.warning(
            "Dropped partition %s.%s of a knowledge base that still had %d rows; they are "
            "back in %s",
            AI_SCHEMA,
            partition,
            survivors,
            default,
        )
    else:
        logger.info("Dropped partition %s.%s", AI_SCHEMA, partition)
    return True


def _autocommit_connection(engine):
    """A connection outside any transaction: CONCURRENTLY refuses one."""
    return engine.connect().execution_options(isolation_level="AUTOCOMMIT")


def _engine(engine=None):
    if engine is not None:
        return engine
    from ..db import db

    return db.engine


def keyword_item_table(bind, knowledge_base_id: str) -> str | None:
    """The item table a KB's pg_search index would live on, or None. Never raises.

    None for a knowledge base that does not exist, whose retrieval method runs
    no keyword leg, or whose strategy has no keyword item table.
    """
    try:
        kb_id = _validated_kb_id(knowledge_base_id)
        row = _run_probe(bind, lambda conn: _probe(conn, _kb_config_sql(), {"id": kb_id}))
    except Exception as exc:
        logger.debug("Could not read the keyword item table of KB %s: %s", knowledge_base_id, exc)
        return None
    if row is None or row[1] not in ("hybrid", "full_text"):
        return None
    item_table = pg_bm25_item_table(row[0])
    return item_table if item_table in PARTITIONED_ITEM_TABLES else None


def ensure_bm25_index(
    knowledge_base_id: str, engine=None, on_progress=None, *, allow_row_move: bool = True
) -> dict:
    """Give this KB a partition and a BM25 index on it, reporting what happened.

    Idempotent, and a no-op whenever a BM25 index is not the right answer: no
    extension, no such KB, a retrieval method that never runs a keyword leg, a
    strategy with no keyword item table, or an item table the conversion
    migration has not reached. A tokenizer
    that no longer matches the KB's ``ts_language`` is dropped and recreated --
    the tokenizer is baked into the index, so a language change cannot be
    applied in place. This KB's index on any *other* item table -- left by a
    strategy change -- is dropped.

    A knowledge base that existed before the extension (or before its item
    table was partitioned) is not given its partition by anything automatic:
    its rows sit in DEFAULT, search keeps reading its bm25s file index, and
    indexing keeps that file index current (``pg_search_serves_kb``). Moving
    it blocks writes to the whole item table for the length of the move (see
    ``create_partition``), so it is an operator step, scheduled per knowledge
    base: ``POST /knowledge-bases/<id>/build-bm25`` dispatches this.

    ``allow_row_move=False`` is for every other caller. Such an ensure attaches
    the partition of a knowledge base with no rows in DEFAULT and builds or
    repairs the index on an existing partition, but a knowledge base with rows
    in DEFAULT -- including one whose own sources were indexed while this
    waited or retried -- is returned as ``skipped`` with reason
    ``row_move_not_allowed``, and nothing is moved.

    ``on_progress(status)`` is called with ``"moving"`` before a partition is
    created or its rows are moved, and ``"building"`` before the bm25 index
    is built (including on a partition whose move committed but whose index
    never got built); the ensure task persists these.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    engine = _engine(engine)

    def progress(status: str) -> None:
        if on_progress is not None:
            on_progress(status)

    # Two connections, deliberately: this AUTOCOMMIT one, because CREATE and
    # DROP INDEX CONCURRENTLY refuse to run inside a transaction; and the one
    # ``create_partition`` opens for itself, because the move is transactional
    # and its session-scoped build lock has to live and die on that connection.
    with _autocommit_connection(engine) as conn:
        if not pg_search_installed(conn, use_cache=False):
            return {"status": "skipped", "reason": "extension_absent"}

        row = conn.execute(text(_kb_config_sql()), {"id": kb_id}).first()
        if row is None:
            return {"status": "skipped", "reason": "kb_not_found"}
        strategy, method, ts_language = row[0], row[1], row[2]

        if method not in ("hybrid", "full_text"):
            return {"status": "skipped", "reason": "retrieval_method"}
        item_table = pg_bm25_item_table(strategy)
        if item_table is None:
            return {"status": "skipped", "reason": "strategy"}
        # Every BM25 item table is partitioned (doc2json has none), so there is
        # no "not partitionable" case left to skip (pinned by a unit test).
        if not table_is_partitioned(conn, item_table):
            logger.warning(
                "Not building a BM25 index for KB %s: %s.%s is not partitioned by "
                "knowledge base yet, and a scored query cannot be answered by a "
                "partitioned parent. This KB keeps the existing keyword path",
                kb_id,
                AI_SCHEMA,
                item_table,
            )
            return {
                "status": "skipped",
                "reason": "table_not_partitioned",
                "item_table": item_table,
            }

        name = bm25_index_name(kb_id, item_table)
        partition = partition_name(kb_id, item_table)
        outcome: dict = {"index": name, "item_table": item_table, "partition": partition}
        dropped = _drop_indexes_on_other_item_tables(conn, kb_id, item_table)
        if dropped:
            outcome["dropped_indexes"] = dropped
        build_safe, _ = concurrent_build_safety(conn)
        unavailable = {**outcome, "status": "unavailable", "reason": "concurrent_build_unsafe"}

        # An unattached partition is a move that did not finish -- a crash, or a
        # move that timed out waiting for its locks. Resuming it is the same call.
        needs_move = not _clone_exists(conn, kb_id, item_table) or not _partition_is_attached(
            conn, kb_id, item_table
        )
        if not needs_move:
            # A move clears leftover checks itself; with no move to run, this
            # is where a check a failed or killed move left on DEFAULT goes.
            try:
                clear_leftover_move_checks(engine, item_table, DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS)
            except Exception as exc:
                logger.warning(
                    "Could not clear leftover move checks from %s.%s (%s); retrying at the "
                    "next BM25 index build on this table",
                    AI_SCHEMA,
                    default_partition_name(item_table),
                    first_error_line(exc),
                )
        else:
            if not build_safe:
                # A move only exists to be followed by a bm25 build: without
                # one, the moved knowledge base loses its file index to the
                # slow tsvector fallback and gains nothing.
                _warn_unsafe_build_once(kb_id)
                return unavailable
            if _relkind(conn, default_partition_name(item_table)) is None:
                logger.warning(
                    "Not building a BM25 index for KB %s: %s.%s has no DEFAULT partition "
                    "to take its rows from",
                    kb_id,
                    AI_SCHEMA,
                    item_table,
                )
                return {
                    **outcome,
                    "status": "skipped",
                    "reason": "default_partition_absent",
                }
            if not allow_row_move and _kb_rows_in_default(conn, kb_id, item_table):
                return {**outcome, "status": "skipped", "reason": "row_move_not_allowed"}
            progress("moving")
            try:
                move = create_partition(engine, kb_id, item_table, allow_row_move=allow_row_move)
            except RowMoveNotAllowed as exc:
                logger.info("Not moving the rows of KB %s: %s", kb_id, exc)
                return {**outcome, "status": "skipped", "reason": "row_move_not_allowed"}
            except PartitionBuildInProgress as exc:
                logger.info("Deferring the BM25 index build for KB %s: %s", kb_id, exc)
                return {
                    **outcome,
                    "status": "skipped",
                    "reason": "partition_build_in_progress",
                }
            outcome["rows_moved"] = move["rows_moved"]
            outcome["writes_blocked_seconds"] = move["writes_blocked_seconds"]
            outcome["partition_created"] = True

        # Everything from reading the index's state to replacing it runs under
        # a lock on this one index, so two ensures for the same KB cannot drop
        # each other's in-flight build. Not waited for: whoever holds it is
        # building this index right now.
        index_lock = bm25_index_lock_relation(kb_id, item_table)
        if not conn.execute(text(partition_build_lock_sql()), {"relation": index_lock}).scalar():
            return {**outcome, "status": "building"}
        try:
            outcome = _ensure_index_locked(
                conn, outcome, kb_id, item_table, ts_language, progress, build_safe=build_safe
            )
            if outcome.get("status") == "unavailable":
                _warn_unsafe_build_once(kb_id)
            # After the bm25 index, which is what serves this knowledge base's
            # keyword search: until then the plain indexes (the full-text GIN
            # above all, the one slow build) would only delay it. A partition
            # whose bm25 build is not safe here is still completed: its plain
            # indexes are what keep its source-scoped deletes fast.
            if outcome.get("status") in ("ready", "unavailable"):
                completed = _complete_partition(conn, kb_id, item_table)
                if completed:
                    outcome["completed"] = completed
            return outcome
        finally:
            _release_advisory_lock(conn, index_lock)


def _drop_indexes_on_other_item_tables(conn, kb_id: str, item_table: str) -> list[str]:
    """Drop this KB's bm25 indexes on every item table but ``item_table``.

    What a strategy change leaves behind: nothing reads that index any more,
    and Postgres would keep maintaining it on every write. The partition stays
    (moving its rows back is real work, and a switch back needs it).
    """
    others = {
        bm25_index_name(kb_id, other): other for other in sorted(BM25_ITEM_TABLES - {item_table})
    }
    present = [
        row[0]
        for row in conn.execute(
            text(
                "SELECT c.relname FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = :schema AND c.relkind = 'i' AND c.relname = ANY(:names) "
                "ORDER BY c.relname"
            ),
            {"schema": AI_SCHEMA, "names": list(others)},
        ).all()
    ]
    for name in present:
        logger.info(
            "Dropping BM25 index %s.%s: KB %s now keeps its keyword text in %s",
            AI_SCHEMA,
            name,
            kb_id,
            item_table,
        )
        conn.execute(text(bm25_drop_ddl(kb_id, others[name])))
    if present:
        invalidate_bm25_index_cache(kb_id)
    return present


def bm25_index_lock_relation(knowledge_base_id: Any, item_table: str) -> str:
    """Advisory-lock subject for building one KB's index (not the partition move)."""
    return f"{AI_SCHEMA}.{bm25_index_name(knowledge_base_id, item_table)}"


def _index_build_in_progress(conn, partition: str) -> bool:
    """Is some other backend running CREATE INDEX (or REINDEX) on this partition?"""
    row = conn.execute(
        text(
            "SELECT 1 FROM pg_stat_progress_create_index "
            "WHERE relid = to_regclass(:partition) AND pid <> pg_backend_pid()"
        ),
        {"partition": _qualified(partition)},
    ).first()
    return row is not None


def _ensure_index_locked(
    conn,
    outcome: dict,
    kb_id: str,
    item_table: str,
    ts_language,
    progress=lambda status: None,
    *,
    build_safe: bool = True,
) -> dict:
    """Build, repair or keep this KB's bm25 index, under the index's own lock.

    Without ``build_safe`` (``concurrent_build_safety``) nothing is built or
    dropped: an index that matches still reports its state, and every other
    case -- no index, an INVALID one, one tokenized for another language, which
    keeps serving -- is ``unavailable``.
    """
    name = bm25_index_name(kb_id, item_table)
    partition = partition_name(kb_id, item_table)
    cast = bm25_tokenizer_cast(item_table, ts_language)
    unavailable = {**outcome, "status": "unavailable", "reason": "concurrent_build_unsafe"}
    existing = conn.execute(
        text(
            "SELECT pg_get_indexdef(c.oid), i.indisvalid FROM pg_class c "
            "JOIN pg_index i ON i.indexrelid = c.oid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = :schema AND c.relname = :name AND c.relkind = 'i'"
        ),
        {"schema": AI_SCHEMA, "name": name},
    ).first()
    existing_def = existing[0] if existing else None

    if existing_def and not existing[1]:
        if _index_build_in_progress(conn, partition):
            return {**outcome, "status": "building"}
        if not build_safe:
            return unavailable
        # INVALID with nothing building it: what a cancelled, killed or failed
        # CREATE INDEX CONCURRENTLY leaves behind. Its definition still matches,
        # and ``IF NOT EXISTS`` would make a re-run a no-op, so without this the
        # KB would report ``building`` for ever and never be searchable by it.
        logger.warning(
            "BM25 index %s.%s is INVALID and no build is running on %s (an earlier "
            "CREATE INDEX CONCURRENTLY failed or was cancelled); dropping and rebuilding it",
            AI_SCHEMA,
            name,
            partition,
        )
        conn.execute(text(bm25_drop_ddl(kb_id, item_table)))
        existing_def = None
        outcome["repaired_invalid_index"] = True

    if existing_def and indexdef_matches_tokenizer(existing_def, cast):
        return {**outcome, "status": bm25_index_state(conn, kb_id, item_table)}

    if not build_safe:
        return unavailable
    if existing_def:
        logger.info("Rebuilding BM25 index %s: tokenizer changed to %s", name, cast)
        conn.execute(text(bm25_drop_ddl(kb_id, item_table)))
    progress("building")
    # Session-level on this AUTOCOMMIT connection (CONCURRENTLY refuses a
    # transaction), so put back before the connection returns to the pool.
    conn.execute(text("SET statement_timeout = 0"))
    try:
        conn.execute(text(bm25_index_ddl(kb_id, item_table, ts_language)))
    except Exception as exc:
        if _sqlstate(exc) == "XX000":
            raise Bm25IndexBuildFailed(
                f"the concurrent build of {AI_SCHEMA}.{name} failed inside pg_search "
                f"({first_error_line(exc)}); the next ensure rebuilds it"
            ) from exc
        raise
    finally:
        _reset_statement_timeout(conn)

    invalidate_bm25_index_cache(kb_id)
    return {**outcome, "status": bm25_index_state(conn, kb_id, item_table)}


def drop_bm25_index(knowledge_base_id: str, engine=None, drop_partitions: bool = False) -> dict:
    """Drop every BM25 index this KB could own, and optionally its partitions.

    Every candidate table, not just the one its current strategy uses: the
    strategy may have changed since the index was built, and by the time a KB
    is deleted its row is gone anyway. ``drop_partitions`` is what a deleted KB
    needs -- leaving a partition behind would leave a relation named after a
    knowledge base that no longer exists. Partitions are dropped even when the
    extension is gone, because they outlive it.

    A contended or timed-out partition drop raises (``PartitionBuildInProgress``
    or a transient database error) so the task retries; reporting ``dropped``
    there would orphan the partition with nothing left to reconcile it.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    engine = _engine(engine)

    dropped: list[str] = []
    failed: list[str] = []
    with _autocommit_connection(engine) as conn:
        installed = pg_search_installed(conn, use_cache=False)
        if installed:
            for item_table in sorted(BM25_ITEM_TABLES):
                try:
                    conn.execute(text(bm25_drop_ddl(kb_id, item_table)))
                    dropped.append(bm25_index_name(kb_id, item_table))
                except Exception as exc:
                    if is_transient_db_error(exc):
                        raise
                    failed.append(bm25_index_name(kb_id, item_table))
                    logger.warning(
                        "Could not drop BM25 index for KB %s on %s",
                        kb_id,
                        item_table,
                        exc_info=True,
                    )

    removed: list[str] = []
    if drop_partitions:
        for item_table in sorted(PARTITIONED_ITEM_TABLES):
            if drop_partition(engine, kb_id, item_table):
                removed.append(partition_name(kb_id, item_table))

    invalidate_bm25_index_cache(kb_id)
    if not installed and not removed:
        return {"status": "skipped", "reason": "extension_absent"}
    if failed:
        return {
            "status": "partial",
            "indexes": dropped,
            "failed_indexes": failed,
            "partitions": removed,
        }
    return {"status": "dropped", "indexes": dropped, "partitions": removed}
