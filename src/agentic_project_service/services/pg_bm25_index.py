"""Postgres-native BM25 keyword search via the ParadeDB ``pg_search`` extension.

One partial ``USING bm25`` index per knowledge base and item table, so a
keyword query is answered by Tantivy inside Postgres instead of by the bm25s
file index or the tsvector fallback. Everything here degrades: when the
extension is not installed, or this KB has no ready index, callers keep
today's behaviour.

Two properties of the extension shape this module and are easy to get wrong:

* **The partial-index predicate is only matched by a literal.** A bound
  parameter (``knowledge_base_id = :kb``) does not match
  ``WHERE knowledge_base_id = '<uuid>'``, so the planner cannot use the index.
  Every KB id that reaches SQL therefore goes through ``uuid.UUID()`` first and
  is interpolated as a canonical literal -- never a caller's raw string.
* **A missing index is not a loud failure.** Querying with ``|||`` against a
  table with no bm25 index at all raises ("does not contain a `USING bm25`
  index"), but querying a KB whose *partial* index does not exist while another
  KB's does returns **zero rows, silently**. Readiness therefore has to be
  checked per knowledge base before this path is used, or a search quietly
  answers nothing.
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
    """CREATE statement for one KB's partial BM25 index."""
    kb_id = _validated_kb_id(knowledge_base_id)
    name = bm25_index_name(kb_id, item_table)
    expression = bm25_text_expression(item_table)
    cast = bm25_tokenizer_cast(item_table, ts_language)
    return (
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} "
        f'ON "{AI_SCHEMA}".{item_table} '
        f"USING bm25 (id, ({expression}{cast}), source_id, meta) "
        f"WITH (key_field = 'id') "
        f"WHERE knowledge_base_id = '{kb_id}'"
    )


def bm25_drop_ddl(knowledge_base_id: str, item_table: str) -> str:
    """DROP statement for one KB's partial BM25 index."""
    name = bm25_index_name(knowledge_base_id, item_table)
    return f'DROP INDEX CONCURRENTLY IF EXISTS "{AI_SCHEMA}".{name}'


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

    ``building`` is an index row with ``indisvalid = false`` -- what a
    CREATE INDEX CONCURRENTLY still in flight (or one that failed) leaves
    behind. Such an index cannot answer a query, so it is not ready.
    """
    name = bm25_index_name(knowledge_base_id, item_table)
    try:
        row = session.execute(
            text(
                "SELECT i.indisvalid FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indexrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = :schema AND c.relname = :name"
            ),
            {"schema": AI_SCHEMA, "name": name},
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
# Index lifecycle
# ---------------------------------------------------------------------------

# pg_search supports exactly one bm25 index per relation, so only the first
# knowledge base to claim an item table gets one; every other KB on that table
# keeps the existing keyword path.
#
# The limit has to be enforced here because pg_search only half-enforces it,
# and the half it misses is the damaging one. Verified against 0.25.9: a plain
# CREATE INDEX on a table that already has a bm25 index is refused with the
# message below, but CREATE INDEX CONCURRENTLY -- which is what a concurrent
# build has to use -- creates it. With two present, unscored matching still
# works for both KBs while a *scored* query against the older index fails with
# "Unsupported query shape". So building a second index does not just fail to
# help the second KB, it breaks keyword search for the first one.
_ONE_INDEX_PER_RELATION = "only have one ParadeDB index"


def _other_bm25_index_on_table(conn, item_table: str, own_index: str) -> str | None:
    """Name of a bm25 index on this item table that is not ``own_index``."""
    row = conn.execute(
        text(
            "SELECT ic.relname FROM pg_class ic "
            "JOIN pg_index i ON i.indexrelid = ic.oid "
            "JOIN pg_am am ON am.oid = ic.relam "
            "JOIN pg_class tc ON tc.oid = i.indrelid "
            "JOIN pg_namespace tn ON tn.oid = tc.relnamespace "
            "WHERE tn.nspname = :schema AND tc.relname = :item_table "
            "AND am.amname = 'bm25' AND ic.relname <> :own LIMIT 1"
        ),
        {"schema": AI_SCHEMA, "item_table": item_table, "own": own_index},
    ).first()
    return row[0] if row else None


def _kb_config_sql() -> str:
    """The three KB fields that decide whether and how to index it."""
    return (
        "SELECT indexing_config->>'strategy', "
        "retrieval_config->>'method', "
        "retrieval_config->>'ts_language' "
        f'FROM "{AI_SCHEMA}".knowledge_bases WHERE id = :id'
    )


def _autocommit_connection(engine):
    """A connection outside any transaction: CONCURRENTLY refuses one."""
    return engine.connect().execution_options(isolation_level="AUTOCOMMIT")


def _engine(engine=None):
    if engine is not None:
        return engine
    from ..db import db

    return db.engine


def ensure_bm25_index(knowledge_base_id: str, engine=None) -> dict:
    """Create (or rebuild) this KB's BM25 index, reporting what happened.

    Idempotent, and a no-op whenever a BM25 index is not the right answer:
    no extension, no such KB, a retrieval method that never runs a keyword
    leg, or a strategy with no keyword item table. A tokenizer that no longer
    matches the KB's ``ts_language`` is dropped and recreated -- the tokenizer
    is baked into the index, so a language change cannot be applied in place.
    """
    kb_id = _validated_kb_id(knowledge_base_id)

    with _autocommit_connection(_engine(engine)) as conn:
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

        name = bm25_index_name(kb_id, item_table)
        cast = bm25_tokenizer_cast(item_table, ts_language)
        outcome = {"index": name, "item_table": item_table}

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

        occupant = _other_bm25_index_on_table(conn, item_table, name)
        if occupant:
            logger.warning(
                "Not building BM25 index %s: %s.%s already carries %s and pg_search "
                "supports one bm25 index per table; building a second one would break "
                "scored queries against the first. This KB keeps the existing keyword path",
                name,
                AI_SCHEMA,
                item_table,
                occupant,
            )
            return {**outcome, "status": "skipped", "reason": "table_index_conflict"}

        try:
            if existing_def:
                logger.info("Rebuilding BM25 index %s: tokenizer changed to %s", name, cast)
                conn.execute(text(bm25_drop_ddl(kb_id, item_table)))
            conn.execute(text(bm25_index_ddl(kb_id, item_table, ts_language)))
        except Exception as exc:
            invalidate_bm25_index_cache(kb_id)
            if _ONE_INDEX_PER_RELATION in str(exc):
                logger.warning(
                    "Not building BM25 index %s: %s.%s already carries another "
                    "knowledge base's pg_search index; this KB keeps the existing "
                    "keyword path",
                    name,
                    AI_SCHEMA,
                    item_table,
                )
                return {**outcome, "status": "skipped", "reason": "table_index_conflict"}
            raise

        invalidate_bm25_index_cache(kb_id)
        return {**outcome, "status": bm25_index_state(conn, kb_id, item_table)}


def drop_bm25_index(knowledge_base_id: str, engine=None) -> dict:
    """Drop every BM25 index this KB could own.

    Every candidate table, not just the one its current strategy uses: the
    strategy may have changed since the index was built, and by the time a KB
    is deleted its row is gone anyway.
    """
    kb_id = _validated_kb_id(knowledge_base_id)

    with _autocommit_connection(_engine(engine)) as conn:
        if not pg_search_installed(conn):
            return {"status": "skipped", "reason": "extension_absent"}

        dropped: list[str] = []
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

    invalidate_bm25_index_cache(kb_id)
    return {"status": "dropped", "indexes": dropped}
