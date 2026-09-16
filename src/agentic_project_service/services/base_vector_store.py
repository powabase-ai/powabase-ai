"""BasePgVectorStore - shared search logic for pgvector-backed stores.

Provides vector similarity search, full-text BM25 search, and hybrid
(RRF-fused) search.  Subclasses configure table/column names via class
attributes and add their own storage methods.
"""

import json
import logging
import time
from typing import Any

from agentic.knowledge.model_config import HYBRID_DEFAULT_VECTOR_WEIGHT
from agentic.knowledge.models import RetrievedItem
from flask import g, has_request_context
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from ..db import AI_SCHEMA
from . import pg_bm25_index
from .kb_search_config import HNSW_ITERATIVE_SCAN_MODE
from .settings_registry import SETTINGS_REGISTRY, get_setting

logger = logging.getLogger(__name__)

# Allowed pgvector hnsw.iterative_scan values; the mode is interpolated into a
# SET LOCAL statement, so it must be validated against this set (never raw input).
_VALID_ITERATIVE_SCAN_MODES = frozenset({"strict_order", "relaxed_order"})

VALID_TS_LANGUAGES = frozenset(
    {
        "simple",
        "arabic",
        "armenian",
        "basque",
        "catalan",
        "danish",
        "dutch",
        "english",
        "finnish",
        "french",
        "german",
        "greek",
        "hindi",
        "hungarian",
        "indonesian",
        "irish",
        "italian",
        "lithuanian",
        "nepali",
        "norwegian",
        "portuguese",
        "romanian",
        "russian",
        "serbian",
        "spanish",
        "swedish",
        "tamil",
        "turkish",
        "yiddish",
    }
)


def ensure_embedding_index(session: Session, schema: str, dims: int) -> None:
    """Create partial hnsw index for this embedding dimension if it doesn't exist.

    Short-circuits via pg_indexes when the index is already present: issuing
    CREATE INDEX IF NOT EXISTS per source takes ShareLock on the embeddings
    table, which conflicts with the RowExclusiveLock the same transaction
    already holds from INSERTing. Under concurrent workers this routinely
    deadlocks. The pg_indexes read takes only AccessShareLock on system
    catalogs and never contends with user-table DML.
    """
    dims = int(dims)
    if not (1 <= dims <= 8192):
        raise ValueError(f"dims must be between 1 and 8192, got {dims}")
    idx_name = f"idx_ai_embeddings_hnsw_{dims}"

    exists_row = session.execute(
        text("SELECT 1 FROM pg_indexes WHERE schemaname = :schema AND indexname = :idx"),
        {"schema": schema, "idx": idx_name},
    ).first()
    if exists_row:
        return

    try:
        with session.begin_nested():
            session.execute(
                text(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name}
                    ON "{schema}".embeddings
                    USING hnsw ((embedding::vector({dims})) vector_cosine_ops)
                    WHERE dims = {dims}
                """),
            )
    except Exception as exc:
        logger.warning(
            "Could not create HNSW index for %d dims: %s; queries will use sequential scan",
            dims,
            exc,
            exc_info=True,
        )


_QUERY_CANCELED = "57014"


class KeywordSearchTimeout(RuntimeError):
    """The SQL keyword-search fallback outran the BM25_FALLBACK_TIMEOUT_MS setting.

    Nothing builds an index in response, so the message names the remedies a
    caller actually has rather than letting it read as "retry in a moment". It
    hedges the build remedy because this layer cannot see the KB's strategy: a
    strategy with no BM25 item table cannot be helped by a build at all. The
    search route resolves the strategy and states one definite remedy.
    """

    def __init__(self, knowledge_base_id: str, timeout_ms: int):
        super().__init__(
            f"Keyword search on knowledge base {knowledge_base_id} exceeded "
            f"{timeout_ms} ms because the knowledge base has no BM25 index. "
            f"Switch the knowledge base's retrieval method to vector_search, or, "
            f"if its indexing strategy supports a BM25 index, build one with "
            f"POST /api/knowledge-bases/{knowledge_base_id}/build-bm25."
        )
        self.knowledge_base_id = knowledge_base_id
        self.timeout_ms = timeout_ms


# Bad BM25_FALLBACK_TIMEOUT_MS values already reported at WARNING by this
# process. The read helper runs on every keyword search, so without this a
# single bad row would emit one warning per search for as long as it sits in
# project_settings — burying the first one. Bounded by the number of distinct
# bad values, which is bounded by how often someone writes the setting. Also
# keys the pg_search keyword-search failure warning (one per KB, table and
# exception type).
_WARNED_TIMEOUT_OVERRIDES: set[str] = set()


def _warn_once_per_bad_value(key: str, message: str, *args: Any) -> None:
    """WARNING the first time this exact bad value is seen, DEBUG afterwards."""
    if key in _WARNED_TIMEOUT_OVERRIDES:
        logger.debug(message, *args)
        return
    _WARNED_TIMEOUT_OVERRIDES.add(key)
    logger.warning(message, *args)


def _bm25_fallback_timeout_ms() -> int:
    """Read the keyword-fallback budget, clamped to the registry's bounds.

    get_setting coerces a stored override but does not range-check it — bounds
    are enforced by validate_setting, i.e. on the settings PUT path only. For
    this one setting an out-of-range value is not merely odd: Postgres reads
    statement_timeout 0 as "no timeout", so a stored 0 would disarm the bound
    this whole path exists to provide. Clamping at read time makes the bound
    hold whatever is in ai.project_settings.

    There is no environment-variable fallback, matching every other registry
    setting: the value is the project setting or the registry default. (The
    only env-read settings in this service are operator-provided platform
    secrets, which are deliberately not tenant-managed.)
    """
    defn = SETTINGS_REGISTRY["BM25_FALLBACK_TIMEOUT_MS"]
    # get_setting has already coerced the stored override to an int, or logged
    # its own warning and returned the registry default, so only the range can
    # still be wrong here.
    value = int(get_setting("BM25_FALLBACK_TIMEOUT_MS"))

    clamped = value
    if defn.min is not None:
        clamped = max(clamped, defn.min)
    if defn.max is not None:
        clamped = min(clamped, defn.max)
    if clamped != value:
        _warn_once_per_bad_value(
            f"range:{value}",
            "BM25_FALLBACK_TIMEOUT_MS=%d is outside the allowed range %s-%s; using %d ms instead",
            value,
            defn.min,
            defn.max,
            clamped,
        )
    return clamped


# Reasons a retrieval answered with less than it was asked for.
KEYWORD_SEARCH_TIMEOUT = "keyword_search_timeout"

_DEGRADED_ATTR = "retrieval_degraded"


def record_retrieval_degradation(reason: str) -> None:
    """Note that this request's retrieval dropped a leg.

    A hybrid search that loses its keyword leg still returns items stamped
    ``retrieval_method="hybrid"``, so without this the caller cannot tell a
    degraded answer from a healthy one. Recorded on the Flask request context,
    read back by the search route; celery tasks and bare threads have no
    request to write to and get the log line only.

    Deliberately appends without checking for duplicates. Retrieval can run in
    a ThreadPoolExecutor over a copied context, so several worker threads share
    one ``g`` and neither the getattr/setattr pair nor a membership test and an
    append are atomic. Reads deduplicate instead, so a duplicated append cannot
    change what a caller sees.

    The first write is still check-then-act: two threads can each find no list,
    build one, and have the later ``setattr`` drop the earlier thread's list and
    its reason with it. Harmless while ``KEYWORD_SEARCH_TIMEOUT`` is the only
    reason — the surviving list holds the same string — but a second reason
    would make the loss observable. Give this a lock or a context-local
    structure before adding one.
    """
    if not has_request_context():
        return
    reasons = getattr(g, _DEGRADED_ATTR, None)
    if reasons is None:
        reasons = []
        setattr(g, _DEGRADED_ATTR, reasons)
    reasons.append(reason)


def get_retrieval_degradations() -> list[str]:
    """Distinct reasons recorded for the current request, sorted."""
    if not has_request_context():
        return []
    return sorted(set(getattr(g, _DEGRADED_ATTR, ())))


def reset_retrieval_degradations() -> None:
    """Drop any reasons carried over from earlier work on this context.

    ``flask.g`` is scoped to the *app* context, not the request, so under a
    long-lived outer app context one request would otherwise read the previous
    request's degradations. The search route calls this before dispatching.
    """
    if has_request_context():
        g.pop(_DEGRADED_ATTR, None)


class BasePgVectorStore:
    """Base class for pgvector-backed stores.

    Subclasses must set:
        TABLE           - SQL table name (e.g. "chunks", "full_documents")
        TEXT_COL        - column returned in RetrievedItem.text (e.g. "text", "full_text")
        SEARCH_TEXT_COL - column used for BM25/tsvector (e.g. "text", "summary")
    """

    TABLE: str
    TEXT_COL: str
    SEARCH_TEXT_COL: str

    def __init__(
        self,
        db_session: Session,
        knowledge_base_id: str,
        schema: str = AI_SCHEMA,
    ):
        self.session = db_session
        self.schema = schema
        self.kb_id = knowledge_base_id

    # ------------------------------------------------------------------
    # Embedding index management
    # ------------------------------------------------------------------

    def _ensure_embedding_index(self, dims: int) -> None:
        """Create partial hnsw index for this dimension if it doesn't exist."""
        ensure_embedding_index(self.session, self.schema, dims)

    # ------------------------------------------------------------------
    # Text resolution hook
    # ------------------------------------------------------------------

    def _resolve_text(self, raw_value: str) -> str:
        """Resolve text column value. Override for storage-backed text."""
        return raw_value

    def _resolve_results(self, items: list[RetrievedItem]) -> list[RetrievedItem]:
        """Resolve text for all items. Override _resolve_text for storage-backed text."""
        for item in items:
            item.text = self._resolve_text(item.text)
        return items

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def _apply_iterative_scan(self) -> None:
        """Enable pgvector HNSW iterative scan for this transaction.

        The HNSW index on ai.embeddings is global (spans all KBs) and the
        vector query filters `knowledge_base_id` AFTER the approximate scan.
        Without iterative scanning, pgvector emits only ~ef_search global
        candidates before that filter, starving KB-scoped queries (often 0
        rows). SET LOCAL keeps this scoped to the current transaction so it
        can't leak across pooled connections. The mode is a validated constant,
        safe to interpolate. No-op (and logged) if pgvector is too old to know
        the GUC — search still works, just without the fix.
        """
        mode = HNSW_ITERATIVE_SCAN_MODE
        if mode not in _VALID_ITERATIVE_SCAN_MODES:
            return
        try:
            self.session.execute(text(f"SET LOCAL hnsw.iterative_scan = '{mode}'"))
        except Exception as e:  # pragma: no cover - depends on pgvector version
            logger.warning(
                "Could not set hnsw.iterative_scan=%s (pgvector < 0.8?): %s; "
                "KB-scoped vector search may under-return results",
                mode,
                e,
            )

    def _fetch_with_timeout(
        self, sql: str, params: dict[str, Any], timeout_ms: int, *, query: str
    ) -> list:
        """Run one query under a statement_timeout scoped to a savepoint.

        Rolling back to the savepoint on cancellation reverts the timeout and
        clears the aborted-transaction state, so the caller's session stays
        usable.

        Two details are load-bearing:

        - The timeout is set with ``set_config('statement_timeout', :ms, true)``
          rather than ``SET LOCAL``, because ``SET LOCAL`` cannot take a bind
          parameter; the third argument ``true`` is what makes it
          transaction-local.
        - On success the previous value is restored explicitly before the
          savepoint is released. ``RELEASE SAVEPOINT`` does not revert a
          transaction-local setting made inside the savepoint, so without that
          restore the budget would go on bounding every later statement in the
          caller's transaction.

        ``query`` is the user's search text, used only for the log line.

        Raises:
            KeywordSearchTimeout: the statement was cancelled (SQLSTATE 57014)
                and at least ``timeout_ms`` had elapsed on the client clock. A
                57014 that arrives sooner cannot be this bound firing, so it is
                re-raised as the original error.
        """
        started = time.monotonic()
        try:
            with self.session.begin_nested():
                previous = self.session.execute(
                    text("SELECT current_setting('statement_timeout')")
                ).scalar()
                self.session.execute(
                    text("SELECT set_config('statement_timeout', :ms, true)"),
                    {"ms": str(timeout_ms)},
                )
                rows = self.session.execute(text(sql), params).fetchall()
                self.session.execute(
                    text("SELECT set_config('statement_timeout', :ms, true)"),
                    {"ms": previous},
                )
                return rows
        except OperationalError as e:
            if getattr(e.orig, "sqlstate", None) != _QUERY_CANCELED:
                raise

            # 57014 is "query canceled" — our statement_timeout, but equally a
            # pg_cancel_backend from anywhere else. Our own bound cannot fire
            # before the budget is spent, so a cancellation that arrives inside
            # it belongs to someone else and must keep its identity rather than
            # be reported as "exceeded N ms".
            #
            # The comparison is exact. The clock starts before begin_nested and
            # two further round trips, and stops after the error has travelled
            # back, so client-measured elapsed strictly exceeds the server's own
            # statement time — a genuine statement_timeout always satisfies
            # this. A tolerance factor would only widen the window in which a
            # foreign cancellation gets mislabelled.
            #
            # Never match on message text: it is localised by lc_messages.
            elapsed_ms = (time.monotonic() - started) * 1000
            if elapsed_ms < timeout_ms:
                raise

            # The one log line for this event. full_text_search re-raises the
            # KeywordSearchTimeout past its generic handler and the hybrid leg
            # logs at debug, so a designed degradation never pages as ERROR and
            # is counted once.
            logger.warning(
                "Keyword search fallback cancelled after %.0f ms of a %d ms budget "
                "(kb=%s table=%s query_len=%d); no BM25 index for this knowledge base",
                elapsed_ms,
                timeout_ms,
                self.kb_id,
                self.TABLE,
                len(query),
            )
            raise KeywordSearchTimeout(self.kb_id, timeout_ms) from e

    async def vector_search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
        item_ids: set[str] | None = None,
        dims: int | None = None,
        _resolve: bool = True,
        source_ids: list[str] | None = None,
    ) -> list[RetrievedItem]:
        """Search using vector cosine similarity via JOIN with ai.embeddings."""
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        effective_dims = int(dims or len(embedding))
        # The partial HNSW index is on (embedding::vector(N)). For the planner
        # to use it, our distance expression must contain that exact cast.
        # dims is interpolated into SQL (not bound) because PostgreSQL does not
        # allow type modifiers to come from a parameter; range-check guards it.
        if not (1 <= effective_dims <= 8192):
            raise ValueError(f"dims must be between 1 and 8192, got {effective_dims}")

        query = f"""
            SELECT
                c.id,
                c.{self.TEXT_COL},
                1 - ((e.embedding::vector({effective_dims})) <=> CAST(:embedding AS vector({effective_dims}))) AS similarity,
                c.source_id,
                c.meta
            FROM "{self.schema}".{self.TABLE} c
            JOIN "{self.schema}".embeddings e ON e.item_id = c.id
            WHERE c.knowledge_base_id = :kb_id
              AND e.dims = :dims
        """

        params: dict[str, Any] = {
            "embedding": embedding_str,
            "kb_id": self.kb_id,
            "dims": effective_dims,
        }

        if item_ids is not None:
            query += " AND c.id = ANY(CAST(:item_ids AS uuid[]))"
            params["item_ids"] = "{" + ",".join(item_ids) + "}"

        if source_ids is not None:
            query += " AND c.source_id = ANY(CAST(:source_ids AS uuid[]))"
            params["source_ids"] = "{" + ",".join(source_ids) + "}"

        if filter_metadata:
            for key, value in filter_metadata.items():
                query += f" AND c.meta @> CAST(:filter_{key} AS jsonb)"
                params[f"filter_{key}"] = json.dumps({key: value})

        query += f"""
            ORDER BY (e.embedding::vector({effective_dims})) <=> CAST(:embedding AS vector({effective_dims}))
            LIMIT :top_k
        """
        params["top_k"] = top_k

        try:
            self._apply_iterative_scan()
            result = self.session.execute(text(query), params)
            items = []
            for row in result:
                items.append(
                    RetrievedItem(
                        item_id=str(row[0]),
                        text=row[1],
                        score=float(row[2]) if row[2] is not None else 0.0,
                        source_id=str(row[3]) if row[3] else None,
                        knowledge_base_id=self.kb_id,
                        meta=row[4] or {},
                    )
                )
            return self._resolve_results(items) if _resolve else items
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

    async def vector_search_per_source(
        self,
        embedding: list[float],
        per_source_k: int,
        source_cap: int,
        similarity_threshold: float = 0.0,
        dims: int | None = None,
        source_ids: list[str] | None = None,
        _resolve: bool = True,
    ) -> list[RetrievedItem]:
        """Return the top ``per_source_k`` chunks for each matched source.

        Used to back-fill the ``min_per_source`` diversity floor: a global
        top-N retrieval is dominated by large sources, so small sources never
        enter the candidate pool. This guarantees every *matched* source — one
        whose best chunk is at/above ``similarity_threshold`` — contributes up to
        ``per_source_k`` of its (also at/above-threshold) best chunks. The
        ``source_cap`` most-relevant *matched* sources are kept, to bound cost;
        the threshold is applied before that cap so a below-threshold source
        cannot consume a slot.
        """
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        effective_dims = int(dims or len(embedding))
        if not (1 <= effective_dims <= 8192):
            raise ValueError(f"dims must be between 1 and 8192, got {effective_dims}")

        source_filter = ""
        params: dict[str, Any] = {
            "embedding": embedding_str,
            "kb_id": self.kb_id,
            "dims": effective_dims,
            "per_source_k": per_source_k,
            "source_cap": source_cap,
            "threshold": similarity_threshold,
        }
        if source_ids is not None:
            source_filter = " AND c.source_id = ANY(CAST(:source_ids AS uuid[]))"
            params["source_ids"] = "{" + ",".join(source_ids) + "}"

        dist_expr = (
            f"(e.embedding::vector({effective_dims})) "
            f"<=> CAST(:embedding AS vector({effective_dims}))"
        )
        query = f"""
            WITH scored AS (
                SELECT
                    c.id AS id,
                    c.{self.TEXT_COL} AS text,
                    c.source_id AS source_id,
                    c.meta AS meta,
                    {dist_expr} AS dist
                FROM "{self.schema}".{self.TABLE} c
                JOIN "{self.schema}".embeddings e ON e.item_id = c.id
                WHERE c.knowledge_base_id = :kb_id
                  AND e.knowledge_base_id = :kb_id
                  AND e.dims = :dims
                  {source_filter}
            ),
            ranked AS (
                SELECT
                    id, text, source_id, meta, dist,
                    ROW_NUMBER() OVER (PARTITION BY source_id ORDER BY dist) AS rn,
                    MIN(dist) OVER (PARTITION BY source_id) AS src_best
                FROM scored
            ),
            top_sources AS (
                -- Rank/cap only sources whose BEST chunk clears the threshold,
                -- so a below-threshold source can't waste a source_cap slot.
                SELECT source_id
                FROM (SELECT DISTINCT source_id, src_best FROM ranked) d
                WHERE (1 - src_best) >= :threshold
                ORDER BY src_best ASC
                LIMIT :source_cap
            )
            SELECT id, text, 1 - dist AS similarity, source_id, meta
            FROM ranked
            WHERE rn <= :per_source_k
              AND (1 - dist) >= :threshold
              AND source_id IN (SELECT source_id FROM top_sources)
            ORDER BY dist ASC
        """

        try:
            self._apply_iterative_scan()
            result = self.session.execute(text(query), params)
            items = [
                RetrievedItem(
                    item_id=str(row[0]),
                    text=row[1],
                    score=float(row[2]) if row[2] is not None else 0.0,
                    source_id=str(row[3]) if row[3] else None,
                    knowledge_base_id=self.kb_id,
                    meta=row[4] or {},
                )
                for row in result
            ]
            return self._resolve_results(items) if _resolve else items
        except Exception as e:
            logger.error(f"Per-source vector search failed: {e}")
            raise

    async def full_text_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict | None = None,
        item_ids: set[str] | None = None,
        ts_language: str = "english",
        _resolve: bool = True,
        source_ids: list[str] | None = None,
    ) -> list[RetrievedItem]:
        """BM25 full-text search on SEARCH_TEXT_COL, returns TEXT_COL."""
        from agentic.knowledge.retrieval.bm25 import bm25_score, parse_tsvector

        if ts_language not in VALID_TS_LANGUAGES:
            raise ValueError(
                f"Invalid ts_language '{ts_language}'. Must be one of: {sorted(VALID_TS_LANGUAGES)}"
            )

        # corpus_stats is MATERIALIZED on purpose. Inlined (the default for a
        # CTE referenced once), a multi-term query gets a one-row estimate and
        # the planner puts this whole-KB aggregate on the inner side of the
        # per-row Nested Loop, re-running it once per matching row: quadratic,
        # about 39 s at 5,000 rows versus about 1 s materialized. doc_freqs needs
        # no hint -- it is read through a scalar subquery and runs once.
        search_query = f"""
            WITH corpus_stats AS MATERIALIZED (
                SELECT
                    COUNT(*) AS total_docs,
                    COALESCE(AVG(LENGTH({self.SEARCH_TEXT_COL})), 0) AS avg_doc_len
                FROM "{self.schema}".{self.TABLE}
                WHERE knowledge_base_id = :kb_id
            ),
            query_lexemes AS (
                SELECT word FROM ts_stat(
                    'SELECT to_tsvector(''{ts_language}'', ' || quote_literal(:query) || ')'
                )
            ),
            doc_freqs AS (
                SELECT
                    ql.word AS term,
                    (SELECT COUNT(*) FROM "{self.schema}".{self.TABLE} c2
                     WHERE c2.knowledge_base_id = :kb_id
                       AND to_tsvector(CAST(:ts_language AS regconfig), c2.{self.SEARCH_TEXT_COL}) @@ to_tsquery(CAST(:ts_language AS regconfig), ql.word)
                    ) AS df
                FROM query_lexemes ql
            )
            SELECT
                c.id,
                c.{self.TEXT_COL},
                c.source_id,
                c.meta,
                to_tsvector(CAST(:ts_language AS regconfig), c.{self.SEARCH_TEXT_COL})::text AS tsvector_text,
                LENGTH(c.{self.SEARCH_TEXT_COL}) AS doc_len,
                cs.total_docs,
                cs.avg_doc_len,
                (SELECT json_object_agg(term, df) FROM doc_freqs) AS doc_freqs_json
            FROM "{self.schema}".{self.TABLE} c
            CROSS JOIN corpus_stats cs
            WHERE c.knowledge_base_id = :kb_id
              AND to_tsvector(CAST(:ts_language AS regconfig), c.{self.SEARCH_TEXT_COL}) @@ websearch_to_tsquery(CAST(:ts_language AS regconfig), :query)
        """

        params: dict[str, Any] = {
            "query": query,
            "kb_id": self.kb_id,
            "ts_language": ts_language,
        }

        if item_ids is not None:
            search_query += " AND c.id = ANY(CAST(:item_ids AS uuid[]))"
            params["item_ids"] = "{" + ",".join(item_ids) + "}"

        if source_ids is not None:
            search_query += " AND c.source_id = ANY(CAST(:source_ids AS uuid[]))"
            params["source_ids"] = "{" + ",".join(source_ids) + "}"

        if filter_metadata:
            for key, value in filter_metadata.items():
                search_query += f" AND c.meta @> CAST(:filter_{key} AS jsonb)"
                params[f"filter_{key}"] = json.dumps({key: value})

        search_query += f"""
            ORDER BY ts_rank(to_tsvector(CAST(:ts_language AS regconfig), c.{self.SEARCH_TEXT_COL}), websearch_to_tsquery(CAST(:ts_language AS regconfig), :query)) DESC
            LIMIT :safety_limit"""
        params["safety_limit"] = top_k * 10

        timeout_ms = _bm25_fallback_timeout_ms()
        try:
            rows = self._fetch_with_timeout(search_query, params, timeout_ms, query=query)

            if not rows:
                return []

            total_docs = int(rows[0][6]) if rows[0][6] else 0
            avg_doc_len = float(rows[0][7]) if rows[0][7] else 0.0
            doc_freqs_json = rows[0][8]
            doc_freqs: dict[str, int] = {}
            if doc_freqs_json:
                if isinstance(doc_freqs_json, str):
                    doc_freqs = {k: int(v) for k, v in json.loads(doc_freqs_json).items()}
                else:
                    doc_freqs = {k: int(v) for k, v in doc_freqs_json.items()}

            query_terms = list(doc_freqs.keys())

            scored_items = []
            for row in rows:
                tsvector_text = row[4] or ""
                doc_len = int(row[5]) if row[5] else 0
                doc_term_freqs = parse_tsvector(tsvector_text)

                score = bm25_score(
                    query_terms=query_terms,
                    doc_term_freqs=doc_term_freqs,
                    doc_len=doc_len,
                    avg_doc_len=avg_doc_len,
                    total_docs=total_docs,
                    doc_freqs=doc_freqs,
                )

                scored_items.append(
                    RetrievedItem(
                        item_id=str(row[0]),
                        text=row[1],
                        score=score,
                        source_id=str(row[2]) if row[2] else None,
                        knowledge_base_id=self.kb_id,
                        meta=row[3] or {},
                    )
                )

            scored_items.sort(key=lambda x: x.score, reverse=True)
            top_items = scored_items[:top_k]
            return self._resolve_results(top_items) if _resolve else top_items

        except KeywordSearchTimeout:
            # A bounded, expected degradation, already warned about once in
            # _fetch_with_timeout. Falling into the handler below would log it
            # as ERROR on every request an un-indexed KB serves.
            raise
        except Exception as e:
            logger.error(f"Full-text search failed: {e}")
            raise

    async def pg_bm25_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict | None = None,
        item_ids: set[str] | None = None,
        _resolve: bool = True,
        source_ids: list[str] | None = None,
    ) -> list[RetrievedItem]:
        """BM25 search answered by this KB's pg_search index.

        The query names the knowledge base's **partition** of the item table,
        not the table. pg_search refuses a scored query against a partitioned
        parent outright ("does not contain a `USING bm25` index"), whatever
        predicate would have pruned it to one indexed partition — so the
        partition is the only relation that can answer, and its LIST bound is
        what restricts the result to this knowledge base. No
        ``knowledge_base_id`` predicate is therefore needed.

        The relation name is the one interpolated value, and it is built from a
        UUID that has been through ``uuid.UUID()``, so nothing a caller supplies
        reaches SQL as an identifier. Every other value, the query text
        included, is bound. Requires the partition's index to exist and be valid
        — see ``pg_bm25_index.bm25_index_ready``.
        """
        partition = pg_bm25_index.partition_name(self.kb_id, self.TABLE)
        normalized = pg_bm25_index.normalize_bm25_query(query)
        if not normalized:
            return []

        match_expression = pg_bm25_index.bm25_text_expression(self.TABLE, alias="c")
        search_query = f"""
            SELECT
                c.id,
                c.{self.TEXT_COL},
                pdb.score(c.id) AS score,
                c.source_id,
                c.meta
            FROM "{self.schema}".{partition} c
            WHERE {match_expression} ||| :bm25_query
        """
        params: dict[str, Any] = {"bm25_query": normalized}

        if item_ids is not None:
            search_query += " AND c.id = ANY(CAST(:item_ids AS uuid[]))"
            params["item_ids"] = "{" + ",".join(item_ids) + "}"

        if source_ids is not None:
            search_query += " AND c.source_id = ANY(CAST(:source_ids AS uuid[]))"
            params["source_ids"] = "{" + ",".join(source_ids) + "}"

        if filter_metadata:
            for key, value in filter_metadata.items():
                search_query += f" AND c.meta @> CAST(:filter_{key} AS jsonb)"
                params[f"filter_{key}"] = json.dumps({key: value})

        search_query += """
            ORDER BY pdb.score(c.id) DESC
            LIMIT :top_k
        """
        params["top_k"] = top_k

        # Inside a savepoint so a failure leaves the caller's session usable.
        # Without it, a rejected query aborts the whole transaction and the
        # keyword fallback dies too with "current transaction is aborted" —
        # which is exactly the case this path has to degrade through.
        with self.session.begin_nested():
            rows = self.session.execute(text(search_query), params).fetchall()

        items = [
            RetrievedItem(
                item_id=str(row[0]),
                text=row[1],
                score=float(row[2]) if row[2] is not None else 0.0,
                source_id=str(row[3]) if row[3] else None,
                knowledge_base_id=self.kb_id,
                meta=row[4] or {},
            )
            for row in rows
        ]
        return self._resolve_results(items) if _resolve else items

    def _pg_bm25_is_usable(self) -> bool:
        """Can this KB's keyword leg be served by pg_search right now?

        Never raises: this runs on the search path, where an unanswerable
        question has to mean "keep the old path". Readiness is only probed
        once the extension is known to be installed, and both answers are
        cached, so the common case costs nothing.
        """
        try:
            if not pg_bm25_index.pg_search_installed(self.session):
                return False
            return pg_bm25_index.bm25_index_ready(self.session, self.kb_id, self.TABLE)
        except Exception as exc:
            logger.debug(
                "Could not determine pg_search availability for KB %s: %s; "
                "using the existing keyword path",
                self.kb_id,
                exc,
            )
            return False

    async def bm25s_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict | None = None,
        item_ids: set[str] | None = None,
        _resolve: bool = True,
        source_ids: list[str] | None = None,
    ) -> list[RetrievedItem]:
        """BM25 keyword search: pg_search index, else bm25s file index, else SQL.

        Prefers this KB's pg_search index when the extension is installed and
        that index is ready. Otherwise the pre-built bm25s file index, and
        failing that the bounded tsvector fallback.

        Args:
            query: Search query (may include conversation context).
            top_k: Maximum results to return.
            filter_metadata: Metadata filters to apply.
            item_ids: Restrict search to these item IDs.
            _resolve: Whether to resolve text (for storage-backed text).
            source_ids: Optional list of source UUIDs to restrict results to.

        Returns:
            List of RetrievedItem ordered by BM25 score.
        """
        from .sparse_retrieval import SparseIndexStore

        if self._pg_bm25_is_usable():
            try:
                return await self.pg_bm25_search(
                    query,
                    top_k=top_k,
                    filter_metadata=filter_metadata,
                    item_ids=item_ids,
                    _resolve=_resolve,
                    source_ids=source_ids,
                )
            except Exception as exc:
                # Once per KB, table and cause at WARNING, without a traceback:
                # this runs on every search for as long as the cause lasts (a
                # readiness answer cached past a dropped index, say).
                cause = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                _warn_once_per_bad_value(
                    f"pg_bm25:{self.kb_id}:{self.TABLE}:{type(exc).__name__}",
                    "pg_search keyword search failed for KB %s table %s (%s); "
                    "falling back to the existing keyword path",
                    self.kb_id,
                    self.TABLE,
                    cause,
                )

        sparse_store = SparseIndexStore(knowledge_base_id=self.kb_id)

        # Fallback to legacy if no index exists
        if not sparse_store.index_exists(item_table=self.TABLE):
            logger.debug(
                "No bm25s index for KB %s table %s, falling back to tsvector",
                self.kb_id,
                self.TABLE,
            )
            return await self.full_text_search(
                query,
                top_k,
                filter_metadata,
                item_ids,
                _resolve=_resolve,
                source_ids=source_ids,
            )

        manager = sparse_store.get_or_load_manager(item_table=self.TABLE)
        retriever = manager.get_retriever()

        if not retriever.is_ready():
            logger.warning(
                "BM25s retriever not ready for KB %s, falling back to tsvector",
                self.kb_id,
            )
            return await self.full_text_search(
                query,
                top_k,
                filter_metadata,
                item_ids,
                _resolve=_resolve,
                source_ids=source_ids,
            )

        # Fetch more results to allow for post-retrieval filtering
        fetch_k = top_k * 3 if (filter_metadata or item_ids or source_ids) else top_k
        results = retriever.search(query, top_k=fetch_k)

        if not results:
            return []

        # Apply item_id filter
        if item_ids:
            results = [r for r in results if r.item_id in item_ids]

        # Fetch full records from DB
        result_ids = [r.item_id for r in results[: top_k * 2]]
        items = await self._fetch_items_by_ids(result_ids)

        # Map bm25s scores to items
        score_map = {r.item_id: r.score for r in results}
        for item in items:
            item.score = score_map.get(item.item_id, 0.0)

        # Apply source_ids filter in Python (post-retrieval)
        if source_ids:
            source_ids_set = set(source_ids)
            items = [item for item in items if item.source_id in source_ids_set]

        # Apply metadata filter in Python (post-retrieval)
        if filter_metadata:
            items = [
                item
                for item in items
                if all((item.meta or {}).get(k) == v for k, v in filter_metadata.items())
            ]

        # Sort by score and limit
        items.sort(key=lambda x: x.score, reverse=True)
        top_items = items[:top_k]

        return self._resolve_results(top_items) if _resolve else top_items

    async def _fetch_items_by_ids(self, item_ids: list[str]) -> list[RetrievedItem]:
        """Fetch full item records by ID.

        Args:
            item_ids: List of item UUIDs to fetch.

        Returns:
            List of RetrievedItem (unordered).
        """
        if not item_ids:
            return []

        # Build parameterized query
        placeholders = ", ".join(f":id_{i}" for i in range(len(item_ids)))
        query = f"""
            SELECT id, {self.TEXT_COL}, source_id, meta
            FROM "{self.schema}".{self.TABLE}
            WHERE knowledge_base_id = :kb_id AND id IN ({placeholders})
        """

        # The KB predicate prunes to one partition, and ids are only unique per
        # partition once the table is partitioned by knowledge base.
        params: dict[str, Any] = {f"id_{i}": id for i, id in enumerate(item_ids)}
        params["kb_id"] = self.kb_id

        try:
            result = self.session.execute(text(query), params)
            items = []
            for row in result:
                items.append(
                    RetrievedItem(
                        item_id=str(row[0]),
                        text=row[1],
                        score=0.0,  # Will be set from bm25s scores
                        source_id=str(row[2]) if row[2] else None,
                        knowledge_base_id=self.kb_id,
                        meta=row[3] or {},
                    )
                )
            return items
        except Exception as e:
            logger.error(f"Failed to fetch items by ID: {e}")
            raise

    async def keyword_search_for_hybrid(
        self,
        query: str,
        top_k: int,
        filter_metadata: dict | None = None,
        item_ids: set[str] | None = None,
        source_ids: list[str] | None = None,
        ts_language: str = "english",
        use_bm25s: bool = True,
    ) -> list[RetrievedItem]:
        """Keyword leg of a hybrid search; empty when the SQL fallback times out.

        Hybrid still has its vector leg, so a slow keyword fallback should cost
        that search its keyword signal, not the whole answer.
        """
        try:
            if use_bm25s:
                return await self.bm25s_search(
                    query=query,
                    top_k=top_k,
                    filter_metadata=filter_metadata,
                    item_ids=item_ids,
                    _resolve=False,
                    source_ids=source_ids,
                )
            return await self.full_text_search(
                query,
                top_k=top_k,
                filter_metadata=filter_metadata,
                item_ids=item_ids,
                ts_language=ts_language,
                _resolve=False,
                source_ids=source_ids,
            )
        except KeywordSearchTimeout:
            record_retrieval_degradation(KEYWORD_SEARCH_TIMEOUT)
            # _fetch_with_timeout already warned once, with the budget and the
            # table; debug here keeps the vector-only answer traceable without
            # logging one event twice.
            logger.debug(
                "Hybrid search on KB %s is returning vector results only: keyword fallback timed out",
                self.kb_id,
            )
            return []

    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        top_k: int = 5,
        vector_weight: float = HYBRID_DEFAULT_VECTOR_WEIGHT,
        filter_metadata: dict | None = None,
        item_ids: set[str] | None = None,
        ts_language: str = "english",
        dims: int | None = None,
        source_ids: list[str] | None = None,
    ) -> list[RetrievedItem]:
        """Combine vector and full-text search using Reciprocal Rank Fusion."""
        from agentic.knowledge.retrieval.fusion import reciprocal_rank_fusion

        fetch_count = top_k * 2
        vector_results = await self.vector_search(
            embedding,
            top_k=fetch_count,
            filter_metadata=filter_metadata,
            item_ids=item_ids,
            dims=dims,
            _resolve=False,
            source_ids=source_ids,
        )
        text_results = await self.keyword_search_for_hybrid(
            query,
            top_k=fetch_count,
            filter_metadata=filter_metadata,
            item_ids=item_ids,
            source_ids=source_ids,
            ts_language=ts_language,
            use_bm25s=False,
        )

        keyword_weight = 1.0 - vector_weight
        fused = reciprocal_rank_fusion(
            result_lists=[vector_results, text_results],
            weights=[vector_weight, keyword_weight],
            top_k=top_k,
        )
        return self._resolve_results(fused)
