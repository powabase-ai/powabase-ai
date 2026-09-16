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
    "doc2json_documents": "summary",
    "graph_index_nodes": "(COALESCE({a}title, '') || ' ' || COALESCE({a}text, ''))",
}

BM25_ITEM_TABLES: frozenset[str] = frozenset(_TEXT_EXPRESSIONS)

# Item tables partitioned ``BY LIST (knowledge_base_id)``, so that each
# knowledge base owns a relation of its own and can therefore own a bm25 index
# of its own. ``doc2json_documents`` is deliberately left unpartitioned: it
# keeps the bm25s/tsvector keyword path.
PARTITIONED_ITEM_TABLES: frozenset[str] = frozenset(
    {"chunks", "full_documents", "graph_index_nodes"}
)

# Rows moved out of the DEFAULT partition per statement, so one huge knowledge
# base cannot turn the move into a single unbounded DELETE ... RETURNING.
EVACUATION_BATCH_ROWS = 10_000

# Guard against an evacuation loop that never drains (a concurrent writer
# inserting into DEFAULT faster than the batches move rows out).
_MAX_EVACUATION_BATCHES = 10_000

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


def partition_lock_default_ddl(item_table: str) -> str:
    """Hold writers off the DEFAULT partition for the length of a move.

    Verified against Postgres 15: a row inserted into DEFAULT after the
    evacuation has drained but before the ATTACH makes Postgres refuse the
    attach outright ("updated partition constraint for default partition would
    be violated by some row"), rolling the whole move back. SHARE conflicts
    with ROW EXCLUSIVE, so it stops INSERT/UPDATE/DELETE on the DEFAULT
    partition -- including writes routed there through the parent -- while
    leaving every reader untouched. Upgrading it to the ACCESS EXCLUSIVE that
    ATTACH needs cannot self-deadlock: Postgres lets a request past waiters
    whose locks the requester already conflicts with.
    """
    return f"LOCK TABLE {_qualified(default_partition_name(item_table))} IN SHARE MODE"


def evacuate_batch_sql(knowledge_base_id: Any, item_table: str) -> str:
    """Move one bounded batch of a KB's rows out of DEFAULT into its partition.

    A partition cannot be attached while the DEFAULT partition still holds a
    row that belongs to it, so the rows have to move first. One statement, so
    a row is never missing from both relations, and ``LIMIT :batch`` keeps each
    statement's WAL and memory bounded however large the knowledge base is.
    """
    partition = partition_name(knowledge_base_id, item_table)
    default = default_partition_name(item_table)
    return (
        "WITH moved AS ("
        f" DELETE FROM {_qualified(default)}"
        f" WHERE id IN (SELECT id FROM {_qualified(default)}"
        " WHERE knowledge_base_id = CAST(:kb AS uuid) LIMIT :batch)"
        " RETURNING *"
        f") INSERT INTO {_qualified(partition)} SELECT * FROM moved"
    )


def mirror_relation_settings_sql(source: str, target: str) -> str:
    """Copy ownership, GRANTs and the RLS flag from one relation to another.

    A new partition starts with no privileges and RLS off, so without this a
    partition is either unreachable by the roles that can read the parent, or
    (if it were granted blindly) readable past the parent's row-level rules.
    Emitted as a server-side block so every identifier is quoted by
    ``format(%I/%s)`` rather than by string building here. Policies are
    deliberately *not* copied: a direct read of a partition by a policy-gated
    role stays denied, while reads through the parent keep applying the
    parent's policies unchanged.
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
    """


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


def pg_search_installed(session) -> bool:
    """Is the pg_search extension created in this database?

    Cached for a few seconds and never raises: this is read on the search
    path, where the honest answer to "can't tell" is "use the old path".
    """
    global _extension_cache
    now = time.monotonic()
    if _extension_cache is not None and now - _extension_cache[0] < _CACHE_TTL_SECONDS:
        return _extension_cache[1]
    try:
        row = session.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'pg_search'")
        ).first()
        installed = row is not None
    except Exception as exc:
        logger.debug("Could not determine whether pg_search is installed: %s", exc)
        installed = False
    _extension_cache = (now, installed)
    return installed


def bm25_index_state(session, knowledge_base_id: str, item_table: str) -> str:
    """``absent`` | ``building`` | ``ready`` for one KB's BM25 index.

    The index has to be the one on *this KB's partition*: that is the relation
    the search path names, so an index of the same name sitting anywhere else
    (a leftover from the unpartitioned design, say) must not read as ready.

    ``building`` is an index row with ``indisvalid = false`` -- what a
    CREATE INDEX CONCURRENTLY still in flight (or one that failed) leaves
    behind. Such an index cannot answer a query, so it is not ready.
    """
    if item_table not in PARTITIONED_ITEM_TABLES:
        return "absent"
    name = bm25_index_name(knowledge_base_id, item_table)
    try:
        partition = partition_name(knowledge_base_id, item_table)
        row = session.execute(
            text(
                "SELECT i.indisvalid FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indexrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "JOIN pg_class t ON t.oid = i.indrelid "
                "WHERE n.nspname = :schema AND c.relname = :name "
                "AND t.relname = :partition"
            ),
            {"schema": AI_SCHEMA, "name": name, "partition": partition},
        ).first()
    except Exception as exc:
        logger.debug("Could not read BM25 index state for %s: %s", name, exc)
        return "absent"
    if row is None:
        return "absent"
    return "ready" if row[0] else "building"


def bm25_index_ready(session, knowledge_base_id: str, item_table: str) -> bool:
    """Cached "can this KB's BM25 index answer a query right now?"."""
    key = (str(knowledge_base_id), item_table)
    now = time.monotonic()
    cached = _ready_cache.get(key)
    if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    ready = bm25_index_state(session, knowledge_base_id, item_table) == "ready"
    if len(_ready_cache) >= _READY_CACHE_MAX_ENTRIES:
        _ready_cache.clear()
    _ready_cache[key] = (now, ready)
    return ready


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
    """The three KB fields that decide whether and how to index it."""
    return (
        "SELECT indexing_config->>'strategy', "
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


def create_partition(engine, knowledge_base_id: Any, item_table: str) -> int:
    """Give one knowledge base a partition of its own, and return rows moved.

    One transaction, because a reader must never see this KB's rows missing:
    the rows leave the DEFAULT partition and arrive in the new one within the
    same statement, and the whole sequence commits or none of it does. The
    evacuation is batched so a large knowledge base does not become one
    unbounded ``DELETE ... RETURNING``.

    The order is forced by Postgres: a partition cannot be attached while the
    DEFAULT partition still holds a row that would belong to it, so the rows
    move into an unattached clone first and the ATTACH comes last -- and the
    DEFAULT partition is locked against writers throughout, or a row arriving
    after the last batch would make that ATTACH fail. Readers are never blocked
    by the lock; they are blocked only for the moment the ATTACH itself holds
    ACCESS EXCLUSIVE.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    partition = partition_name(kb_id, item_table)
    default = default_partition_name(item_table)
    moved = 0

    with engine.begin() as tx:
        tx.execute(text(partition_create_ddl(kb_id, item_table)))
        tx.execute(text(partition_lock_default_ddl(item_table)))
        for definition in _foreign_key_defs(tx, default):
            tx.execute(text(f'ALTER TABLE "{AI_SCHEMA}".{partition} ADD {definition}'))

        evacuate = text(evacuate_batch_sql(kb_id, item_table))
        for _ in range(_MAX_EVACUATION_BATCHES):
            result = tx.execute(evacuate, {"kb": kb_id, "batch": EVACUATION_BATCH_ROWS})
            if not result.rowcount:
                break
            moved += result.rowcount
        else:
            raise RuntimeError(
                f'the DEFAULT partition of "{AI_SCHEMA}".{item_table} did not drain for '
                f"knowledge base {kb_id} after {_MAX_EVACUATION_BATCHES} batches"
            )

        tx.execute(text(partition_attach_ddl(kb_id, item_table)))
        tx.execute(
            text(mirror_relation_settings_sql(_qualified(item_table), _qualified(partition)))
        )

    logger.info(
        "Created partition %s.%s and moved %d rows into it from %s",
        AI_SCHEMA,
        partition,
        moved,
        default,
    )
    return moved


def drop_partition(engine, knowledge_base_id: Any, item_table: str) -> bool:
    """Detach and drop one KB's partition, returning any rows to DEFAULT first.

    Rescuing the rows makes this safe to call on a knowledge base that still
    exists: the worst case is that the KB goes back to the fallback keyword
    path, never that a row is lost. After a KB delete the partition is already
    empty (its rows went with the cascade), so the rescue costs nothing.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    partition = partition_name(kb_id, item_table)
    default = default_partition_name(item_table)

    with engine.begin() as tx:
        if _relkind(tx, partition) is None:
            return False
        if _partition_is_attached(tx, kb_id, item_table):
            tx.execute(text(partition_detach_ddl(kb_id, item_table)))
        tx.execute(text(f"INSERT INTO {_qualified(default)} SELECT * FROM {_qualified(partition)}"))
        tx.execute(text(partition_drop_ddl(kb_id, item_table)))

    logger.info("Dropped partition %s.%s; its rows are back in %s", AI_SCHEMA, partition, default)
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

    with _autocommit_connection(engine) as conn:
        if not pg_search_installed(conn):
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
        if item_table not in PARTITIONED_ITEM_TABLES:
            return {
                "status": "skipped",
                "reason": "table_not_partitionable",
                "item_table": item_table,
            }
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
        cast = bm25_tokenizer_cast(item_table, ts_language)
        outcome: dict = {"index": name, "item_table": item_table, "partition": partition}

        if not partition_exists(conn, kb_id, item_table):
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
            outcome["rows_moved"] = create_partition(engine, kb_id, item_table)
            outcome["partition_created"] = True

        existing = conn.execute(
            text(
                "SELECT pg_get_indexdef(c.oid) FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = :schema AND c.relname = :name AND c.relkind = 'i'"
            ),
            {"schema": AI_SCHEMA, "name": name},
        ).first()
        existing_def = existing[0] if existing else None

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
    knowledge base that no longer exists.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    engine = _engine(engine)

    dropped: list[str] = []
    with _autocommit_connection(engine) as conn:
        if not pg_search_installed(conn):
            return {"status": "skipped", "reason": "extension_absent"}

        for item_table in sorted(BM25_ITEM_TABLES):
            try:
                conn.execute(text(bm25_drop_ddl(kb_id, item_table)))
                dropped.append(bm25_index_name(kb_id, item_table))
            except Exception:
                logger.warning(
                    "Could not drop BM25 index for KB %s on %s",
                    kb_id,
                    item_table,
                    exc_info=True,
                )

    removed: list[str] = []
    if drop_partitions:
        for item_table in sorted(PARTITIONED_ITEM_TABLES):
            try:
                if drop_partition(engine, kb_id, item_table):
                    removed.append(partition_name(kb_id, item_table))
            except Exception:
                logger.warning(
                    "Could not drop the partition of %s for KB %s",
                    item_table,
                    kb_id,
                    exc_info=True,
                )

    invalidate_bm25_index_cache(kb_id)
    return {"status": "dropped", "indexes": dropped, "partitions": removed}
