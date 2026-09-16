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
"""

from __future__ import annotations

import logging
import re
import time
import uuid
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
    """Create the KB's partition as an unattached clone of DEFAULT.

    Cloning the DEFAULT partition rather than the partitioned parent is what
    carries the column defaults, the CHECK/UNIQUE constraints and the ordinary
    indexes across -- including the local ``PRIMARY KEY (id)``, which only the
    partitions have (the parent deliberately declares none, so a bare ``id``
    key stays legal). ``LIKE`` never copies foreign keys; those are added
    separately from the DEFAULT partition's catalog entries.
    """
    partition = partition_name(knowledge_base_id, item_table)
    return (
        f"CREATE TABLE IF NOT EXISTS {_qualified(partition)} "
        f"(LIKE {_qualified(default_partition_name(item_table))} "
        "INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING INDEXES "
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
    """Validate the DEFAULT check: one scan of DEFAULT, readers unaffected.

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

    Two concurrent moves out of one DEFAULT partition deadlock each other: each
    takes SHARE on it and then asks to upgrade to the ACCESS EXCLUSIVE its own
    ATTACH needs, so each waits for the other's SHARE. Observed on a real
    project as ``deadlock detected`` with both tasks failing. Serialising the
    moves per item table removes the cycle entirely.

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
    DELETE takes, and not with the ACCESS SHARE of a reader, so reads through
    the parent carry on throughout. The cost is that writes to *every*
    knowledge base on this item table wait for the move, including those that
    already have partitions of their own.
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


def move_rows_sql(knowledge_base_id: Any, item_table: str) -> tuple[str, str]:
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
    """
    partition = partition_name(knowledge_base_id, item_table)
    default = default_partition_name(item_table)
    predicate = "WHERE knowledge_base_id = CAST(:kb AS uuid)"
    return (
        f"INSERT INTO {_qualified(partition)} SELECT * FROM {_qualified(default)} {predicate}",
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
            str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
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
            str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
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
            str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
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

    This is the one rule: the knowledge-base routes (create, PATCH,
    ``/build-bm25``, ``bm25_status``) and the per-source indexing gate all ask
    it, so they cannot disagree about which index a KB reads.

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
            str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
        )
        return "bm25s"
    return "pg_search" if partitioned else "bm25s"


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


def partition_exists(conn, knowledge_base_id: Any, item_table: str) -> bool:
    return _relkind(conn, partition_name(knowledge_base_id, item_table)) is not None


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
# request is wrong, another transaction was simply in the way.
_TRANSIENT_SQLSTATES = frozenset(
    {
        "55P03",  # lock_not_available (lock_timeout)
        "40P01",  # deadlock_detected
        "40001",  # serialization_failure
    }
)


def is_transient_db_error(exc: BaseException) -> bool:
    """Did this fail only because another transaction was in the way?"""
    orig = getattr(exc, "orig", exc)
    return getattr(orig, "sqlstate", None) in _TRANSIENT_SQLSTATES


def is_partition_move_race(exc: BaseException) -> bool:
    """Did a write fail only because it raced a knowledge base's partition move?

    SQLSTATE 23514 from either of the two checks a move can trip: Postgres'
    own "violates partition constraint" (a row routed to DEFAULT by a statement
    planned before the ATTACH), or the move's temporary ``bm25_move_<kb>``
    check on DEFAULT. An ordinary CHECK violation is a real error and is not
    matched.
    """
    orig = getattr(exc, "orig", None)
    if getattr(orig, "sqlstate", None) != "23514":
        return False
    message = str(orig)
    return "violates partition constraint" in message or (
        f'violates check constraint "{_DEFAULT_MOVE_CHECK_PREFIX}' in message
    )


def _lock_holders_sql() -> str:
    return (
        "SELECT l.pid, l.mode, l.granted, pg_blocking_pids(l.pid) AS blocked_by, "
        "a.state, now() - a.xact_start AS xact_age, left(a.query, 200) AS query "
        "FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid "
        "WHERE l.relation IN (to_regclass(:parent), to_regclass(:default)) "
        "AND l.pid <> pg_backend_pid() ORDER BY l.granted DESC, l.pid"
    )


def _log_move_failure(
    conn,
    knowledge_base_id: str,
    item_table: str,
    exc: BaseException,
    *,
    action: str = "Moving the rows into the partition of",
) -> None:
    """Say why a move failed, and for a lock timeout, who was in the way.

    Runs after the rollback. A lock timeout has already ended the wait, so
    there is no blocked backend left to ask ``pg_blocking_pids`` about; the
    sessions still holding or waiting for locks on the parent and DEFAULT are
    the nearest evidence, each with whatever blocks *it*.
    """
    if not is_transient_db_error(exc):
        logger.warning(
            "%s KB %s on %s.%s failed; rolled back",
            action,
            knowledge_base_id,
            AI_SCHEMA,
            item_table,
        )
        return
    holders: list = []
    try:
        holders = [
            dict(row._mapping)
            for row in conn.execute(
                text(_lock_holders_sql()),
                {
                    "parent": _qualified(item_table),
                    "default": _qualified(default_partition_name(item_table)),
                },
            ).all()
        ]
        conn.rollback()
    except Exception:
        conn.rollback()
        logger.debug("Could not list lock holders", exc_info=True)
    logger.warning(
        "%s KB %s on %s.%s gave up (lock_timeout %d ms, or a deadlock); rolled back, "
        "retryable. Sessions holding or awaiting locks on the table: %s",
        action,
        knowledge_base_id,
        AI_SCHEMA,
        item_table,
        MOVE_LOCK_TIMEOUT_MS,
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


def _prepare_partition(conn, kb_id: str, item_table: str) -> None:
    """Create the clone, its CHECK constraint and its foreign keys. Idempotent."""
    partition = partition_name(kb_id, item_table)
    default = default_partition_name(item_table)
    conn.execute(text(partition_create_ddl(kb_id, item_table)))
    if not _check_constraint_exists(conn, partition):
        conn.execute(text(partition_check_ddl(kb_id, item_table)))
    if not _foreign_key_defs(conn, partition):
        for definition in _foreign_key_defs(conn, default):
            conn.execute(text(f"ALTER TABLE {_qualified(partition)} ADD {definition}"))
    conn.commit()


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


def _drop_move_checks(conn, item_table: str, names: list[str]) -> None:
    """Drop temporary DEFAULT checks, each in its own short transaction.

    DROP CONSTRAINT takes ACCESS EXCLUSIVE on DEFAULT for a catalog change
    only, bounded by the same lock timeout as the move.
    """
    for name in names:
        conn.execute(text(f"SET LOCAL lock_timeout = '{MOVE_LOCK_TIMEOUT_MS}ms'"))
        conn.execute(text(default_move_check_drop_ddl(item_table, name)))
        conn.commit()


def create_partition(engine, knowledge_base_id: Any, item_table: str) -> dict:
    """Move one knowledge base into a partition of its own, atomically.

    Returns ``{"rows_moved": int, "writes_blocked_seconds": float}``.

    1. **prepare** (own transaction) -- clone the DEFAULT partition into an
       unattached table with a CHECK matching the partition bound (so ATTACH
       skips its scan of the new partition) and the foreign keys ``LIKE`` does
       not copy. Any temporary DEFAULT check a crashed move left is dropped.
    2. **move** (one transaction) -- SHARE on the parent; meanwhile, on a
       second connection, add ``CHECK (knowledge_base_id <> kb) NOT VALID`` to
       DEFAULT (a catalog change) and commit it; SHARE on DEFAULT; copy the
       KB's rows into the clone and delete them from DEFAULT; VALIDATE the
       DEFAULT check (one scan of DEFAULT, readers unaffected); ATTACH, which
       now needs neither of its scans; mirror ownership, grants, RLS and
       policies; commit.
    3. **unfence** (own transaction) -- drop the DEFAULT check.

    The bm25 index is built afterwards with CREATE INDEX CONCURRENTLY, outside
    any of this.

    Who waits, measured on Postgres 15 (warm cache, 128 MB shared_buffers)
    moving a 40 000-row knowledge base:

    * **writers** through the parent, for every knowledge base on this item
      table, for all of step 2: 0.23-0.31 s with a 540 000-row (360 MB)
      DEFAULT -- copy ~0.14-0.20 s, delete ~0.02 s, VALIDATE ~0.06 s -- and
      0.48 s with a 2 040 000-row (1.4 GB) DEFAULT, where VALIDATE alone took
      0.26 s. It grows with the rows moved and with the size of DEFAULT.
      Without the DEFAULT check the ATTACH ran the same scan instead (0.06 s
      and 0.30 s), so writers wait about as long either way.
    * **readers** only for the ATTACH (1-8 ms) and the catalog-only ADD and
      DROP of the check. Without the check the ATTACH held ACCESS EXCLUSIVE
      for its whole DEFAULT scan: a reader waited up to 0.07 s at 540 000 rows
      and 0.64 s at 2 040 000, and on a cold cache that scan is disk-bound.

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

    Because step 3 is one transaction, every reader sees the knowledge base's
    rows either all in DEFAULT or all in the partition, and writers through the
    parent wait for the commit and then plan against the new partition list
    (see ``partition_lock_parent_ddl``). That is the deliberate choice over the
    earlier online design, which kept writers going but silently lost their
    UPDATEs and DELETEs.

    Serialised per item table by a session-scoped advisory lock: two
    concurrent moves out of one DEFAULT partition deadlock each other.

    Re-entrant: a failure in step 2 rolls the move back and the check is
    dropped, so a retry starts from a clean DEFAULT and an empty clone.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    partition = partition_name(kb_id, item_table)
    default = default_partition_name(item_table)
    insert_sql, delete_sql = move_rows_sql(kb_id, item_table)
    fence = default_move_check_name(kb_id)

    with engine.connect() as conn:
        _acquire_partition_build_lock(conn, item_table)
        try:
            # Under the build lock no other move is in flight, so every move
            # check still on DEFAULT is a crashed move's leftover -- and one
            # that refuses some knowledge base's writes until it is gone.
            _drop_move_checks(conn, item_table, _move_check_names(conn, item_table))
            if _partition_is_attached(conn, kb_id, item_table):
                return {"rows_moved": 0, "writes_blocked_seconds": 0.0}

            _prepare_partition(conn, kb_id, item_table)

            try:
                conn.execute(text(f"SET LOCAL lock_timeout = '{MOVE_LOCK_TIMEOUT_MS}ms'"))
                conn.execute(text(partition_lock_parent_ddl(item_table)))
                started = time.monotonic()
                # The fence goes up on a second connection while this one holds
                # writers off the parent: it has to commit before the move can
                # validate it, and committing it here would release the lock and
                # open a gap in which this KB's writes hit it and fail.
                with engine.connect() as fencer:
                    fencer.execute(text(f"SET LOCAL lock_timeout = '{MOVE_LOCK_TIMEOUT_MS}ms'"))
                    fencer.execute(text(default_move_check_add_ddl(kb_id, item_table)))
                    fencer.commit()
                conn.execute(text(partition_lock_default_ddl(item_table)))
                moved = conn.execute(text(insert_sql), {"kb": kb_id}).rowcount
                conn.execute(text(delete_sql), {"kb": kb_id})
                conn.execute(text(default_move_check_validate_ddl(kb_id, item_table)))
                conn.execute(text(partition_attach_ddl(kb_id, item_table)))
                conn.execute(
                    text(
                        mirror_relation_settings_sql(_qualified(item_table), _qualified(partition))
                    )
                )
                conn.commit()
                blocked = time.monotonic() - started
            except Exception as exc:
                conn.rollback()
                _log_move_failure(conn, kb_id, item_table, exc)
                raise
            finally:
                conn.rollback()
                try:
                    _drop_move_checks(conn, item_table, [fence])
                except Exception:
                    # Harmless once the move committed (the KB's rows no longer
                    # route to DEFAULT); after a failed move it refuses the KB's
                    # writes until the next move on this table clears it.
                    conn.rollback()
                    logger.warning(
                        "Could not drop the temporary check %s from %s.%s; the next "
                        "partition build on this table will",
                        fence,
                        AI_SCHEMA,
                        default,
                        exc_info=True,
                    )
        finally:
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
                survivors = conn.execute(
                    text(f"INSERT INTO {_qualified(default)} SELECT * FROM {_qualified(partition)}")
                ).rowcount
                conn.execute(text(partition_drop_ddl(kb_id, item_table)))
                conn.commit()
            except Exception as exc:
                conn.rollback()
                _log_move_failure(conn, kb_id, item_table, exc, action="Dropping the partition of")
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


def ensure_bm25_index(knowledge_base_id: str, engine=None) -> dict:
    """Give this KB a partition and a BM25 index on it, reporting what happened.

    Idempotent, and a no-op whenever a BM25 index is not the right answer: no
    extension, no such KB, a retrieval method that never runs a keyword leg, a
    strategy with no keyword item table, an item table that is never
    partitioned, or one the conversion migration has not reached. A tokenizer
    that no longer matches the KB's ``ts_language`` is dropped and recreated --
    the tokenizer is baked into the index, so a language change cannot be
    applied in place.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    engine = _engine(engine)

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

        # An unattached partition is a move that did not finish -- a crash, or a
        # move that timed out waiting for its locks. Resuming it is the same call.
        if not partition_exists(conn, kb_id, item_table) or not _partition_is_attached(
            conn, kb_id, item_table
        ):
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
            try:
                move = create_partition(engine, kb_id, item_table)
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
            return _ensure_index_locked(conn, outcome, kb_id, item_table, ts_language)
        finally:
            _release_advisory_lock(conn, index_lock)


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


def _ensure_index_locked(conn, outcome: dict, kb_id: str, item_table: str, ts_language) -> dict:
    name = bm25_index_name(kb_id, item_table)
    partition = partition_name(kb_id, item_table)
    cast = bm25_tokenizer_cast(item_table, ts_language)
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

    if existing_def:
        logger.info("Rebuilding BM25 index %s: tokenizer changed to %s", name, cast)
        conn.execute(text(bm25_drop_ddl(kb_id, item_table)))
    conn.execute(text(bm25_index_ddl(kb_id, item_table, ts_language)))

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
    return {"status": "dropped", "indexes": dropped, "partitions": removed}
