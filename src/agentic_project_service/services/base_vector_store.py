"""BasePgVectorStore - shared search logic for pgvector-backed stores.

Provides vector similarity search, full-text BM25 search, and hybrid
(RRF-fused) search.  Subclasses configure table/column names via class
attributes and add their own storage methods.
"""

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from agentic.knowledge.model_config import HYBRID_DEFAULT_VECTOR_WEIGHT
from agentic.knowledge.models import RetrievedItem
from flask import g, has_request_context
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from ..db import AI_SCHEMA
from . import pg_bm25_index, pg_vector_index
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


# Upper bound on ``top_k``, which the vector query interpolates rather than
# binds (see ``kb_sql_literal``). Nothing upstream range-checks it -- the search
# route takes it straight from the request body -- so this is where it is
# checked. Generous: the per-source candidate pool is capped at 200 and hybrid
# search doubles the caller's value, so nothing legitimate comes near.
#
# That doubling also means this is not the ceiling a caller sees on every route:
# hybrid search fetches twice the caller's ``top_k`` from each leg, so the largest
# ``top_k`` it accepts is half of this. It checks that itself, against the value
# the caller sent, so the error names a limit the caller can act on rather than
# the doubled one the vector leg would have reported.
MAX_TOP_K = 10_000


# The one item table whose searches may be steered onto a knowledge base's own
# partial HNSW index (see ``_preferring_this_kbs_partial_index``).
#
# ``ai.embeddings`` is polymorphic -- ``item_table`` is a column on it, and four
# stores inherit ``vector_search`` -- while the partial index's predicate and the
# catalog probe that looks for it name only ``(knowledge_base_id, dims)``. So a
# knowledge base whose embeddings are mostly chunks, but whose search routes to
# one of the document-level stores, passes the probe and is driven into an HNSW
# walk of rows that cannot join. Measured at 1536 dimensions on a document-level
# store of 40 rows at ``top_k=5``: 2.2 -> 25.9 ms, recall 1.00 -> 0.33, against a
# plan the planner would have sorted in 2 ms.
#
# The narrow fix is this constant rather than ``item_table`` in the index name
# and predicate, which would touch index naming, the boot sweep and every drop
# path for a benefit only one store has ever been measured to get. The other
# three stores keep the plan the planner chooses for them, which is the plan they
# have today.
PER_KB_INDEX_ITEM_TABLE = "chunks"


# ``hnsw.ef_search`` for a search that is using a knowledge base's own partial
# index. pgvector's default is 40, and recall degrades with the *absolute* size
# of the index rather than with the knowledge base's share of the table: measured
# on real embeddings, 0.997 at 400 rows, 0.982 at 2,000, 0.933 at 8,400 and 0.915
# at 12,000. The build threshold is 10,000 and a knowledge base can be several
# times that, where the trend projects about 0.85 -- too low to ship as the
# answer to a search that used to be exact. 120 measured 0.973 at 12,000 rows for
# 2.96 ms, still 12x faster than the exact scan it replaces.
#
# **The supported band is 80-400, and the upper bound is a planner cliff rather
# than taste.** pgvector's HNSW cost estimate scales with this setting, and past
# roughly 600-800 a knowledge base's own partial index prices *above* the shared
# per-dimension index, so the planner flips to the shared one and the whole
# mechanism inverts -- reproduced here at 12,000 rows, where ``ef_search`` 600
# kept the partial index at 6.9 ms and 800 took the shared index at 18.8 ms with
# *lower* recall. 120 is deliberately well inside the band.
#
# Orthogonal to the forcing beside it: this decides how accurate an index scan is
# once the planner is on an index, not whether it gets one. Which is why it is
# set only when the probe has found an index to be on.
PER_KB_HNSW_EF_SEARCH = 120


def kb_sql_literal(knowledge_base_id: Any) -> str:
    """A knowledge base id as a quoted SQL literal, or ValueError.

    The embeddings-side ``knowledge_base_id`` predicate is interpolated rather
    than bound, and this is the one gate between a caller's value and the SQL.
    ``uuid.UUID`` accepts nothing that could carry a quote or a statement
    separator, and the result is the *canonical* form, so braces, a ``urn:``
    prefix and surrounding whitespace are all normalised away. A value that is
    not a UUID cannot match ``ai.embeddings.knowledge_base_id`` anyway (the
    column is ``uuid``, so before this it reached the server and failed there).

    Interpolated because a bound parameter cannot prove the partial index's
    predicate. PostgreSQL uses a partial index only when the query's own
    restriction clauses *prove* the index predicate, and the per-knowledge-base
    index names one KB id and one ``dims`` value as literals. A plan built
    without knowing a parameter's value proves neither -- which matters because
    psycopg prepares a statement after ``prepare_threshold`` executions on a
    connection, and from the sixth execution of the prepared statement
    PostgreSQL starts weighing its generic plan against the custom ones.

    ``knowledge_base_id`` is not sufficient on its own, and that was measured
    rather than assumed. Under ``plan_cache_mode = force_generic_plan``, on a
    12,000-embedding fixture at 1536 dimensions with both indexes present:

    | query shape | generic plan reaches |
    |---|---|
    | kb literal, ``dims`` and ``LIMIT`` bound | no HNSW index: Sort, cost 999 |
    | kb literal, ``dims`` literal, ``LIMIT`` bound | no HNSW index: Sort, cost 999 |
    | kb literal, ``dims`` bound, ``LIMIT`` literal | no HNSW index: Sort, cost 845 |
    | **kb, ``dims`` and ``LIMIT`` all literal** | **the partial index**, cost 369 |
    | kb bound, ``dims`` and ``LIMIT`` literal | the *shared* index, cost 533 |

    So ``vector_search`` interpolates all three, on both sides of the join for
    the KB id. ``dims`` is range-checked and already interpolated into the
    distance cast, and ``top_k`` is checked against ``MAX_TOP_K``; the last row
    is why the KB id has to be one of them.

    **What this guarantees, and what it does not.** The guarantee is for the
    *unfiltered* search: every value in the index's predicate, and the LIMIT, is
    a literal, so a generic plan can prove the predicate and keep the ordered
    index scan. A ``filter_metadata`` search has a fourth value the planner does
    not know, and it cannot be made a literal -- it is caller data, bound as
    jsonb. A generic plan has no selectivity estimate for ``meta @>`` at all, so
    it prices the ordered index scan out and the partial index is lost. That
    shape is protected differently, by asking for a custom plan for that one
    execution: see ``BasePgVectorStore._forcing_a_custom_plan``.
    """
    try:
        return f"'{uuid.UUID(str(knowledge_base_id))}'"
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"knowledge_base_id is not a UUID: {knowledge_base_id!r}") from exc


def validated_top_k(top_k: Any) -> int:
    """A row limit safe to interpolate into ``LIMIT``, or ValueError.

    ``LIMIT`` is interpolated so a prepared statement's generic plan can still
    reach the partial HNSW index (see ``kb_sql_literal``): with an unknown limit
    the planner assumes it will be asked for a large fraction of the rows, which
    prices an ordered index scan out and leaves an exact sort. Zero is allowed
    because that is what a bound ``LIMIT 0`` did -- an empty answer, not an
    error.

    ``int()`` coerces rather than rejects, so ``True`` becomes 1 and ``1.9``
    becomes 1: surprising to read, safe to emit, and unreachable from the routes,
    which parse ``top_k`` out of JSON as an int.

    ``OverflowError`` is in the handler for one reachable input: Python's JSON
    parser accepts bare ``Infinity`` and ``NaN``, so a request body of
    ``{"top_k": Infinity}`` arrives as a float, and ``int(float('inf'))`` raises
    ``OverflowError`` rather than ``ValueError``. Without it that body is a 500
    where every other unusable ``top_k`` is a 400.
    """
    try:
        value = int(top_k)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"top_k is not an integer: {top_k!r}") from exc
    if not (0 <= value <= MAX_TOP_K):
        raise ValueError(f"top_k must be between 0 and {MAX_TOP_K}, got {value}")
    return value


def ensure_embedding_index(session: Session, schema: str, dims: int) -> None:
    """Create partial hnsw index for this embedding dimension if it doesn't exist.

    Short-circuits via pg_indexes when the index is already present: issuing
    CREATE INDEX IF NOT EXISTS per source takes ShareLock on the embeddings
    table, which conflicts with the RowExclusiveLock the same transaction
    already holds from INSERTing. Under concurrent workers this routinely
    deadlocks. The pg_indexes read takes only AccessShareLock on system
    catalogs and never contends with user-table DML.

    **This index is load-bearing for every knowledge base without a partial one,
    and narrowing it is sequenced.** It is the index a KB-scoped vector search
    falls back to, so it must keep covering every row of its dimension until the
    embeddings-side ``knowledge_base_id`` predicate (see ``kb_sql_literal``) has
    deployed everywhere. Replacing it with a residual index that excludes the
    knowledge bases holding their own partial index is the planned follow-up and
    the reason that ordering matters: the older query shape matches no HNSW
    index at all against a residual one and degenerates to a sequential scan.
    ``pg_vector_index``'s module docstring holds the constraint in full.
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


METADATA_FILTER_PARAM = "filter_metadata"


def metadata_filter_clause(filter_metadata: dict | None) -> tuple[str, dict[str, str]]:
    """SQL fragment and bound parameter for a metadata containment filter.

    The whole filter is bound as ONE jsonb value and compared server-side, so no
    part of it — keys included — reaches the SQL text. This used to be assembled
    one key at a time, with the key interpolated into the statement as the name
    of its own bind parameter (``:filter_{key}``); a key is caller data, so a key
    carrying SQL of its own became part of the WHERE clause.

    One ``@>`` over the whole object says the same thing as one per pair: jsonb
    object containment holds only when every pair on the right is contained on
    the left, so ``meta @> '{"a": 1, "b": 2}'`` matches exactly the rows
    ``meta @> '{"a": 1}' AND meta @> '{"b": 2}'`` matches.

    Returns ``("", {})`` for an absent filter and for an empty object, so the
    caller appends nothing and binds nothing. Anything else that is not an object
    is rejected rather than bound: ``jsonb @> <array|string|number|boolean>`` does
    not error, it is simply false, so binding one would turn a malformed filter
    into an empty result with nothing logged — indistinguishable, on the agent
    path, from "nothing relevant". A ``ValueError`` reaches the search route's
    400 instead.

    **The type check comes before the falsy one, and the order is the fix.** With
    the falsy check first, a truthy non-object (``["a"]``, ``"gold"``, ``1``) was
    a 400 while a *falsy* one (``[]``, ``""``, ``0``, ``False``) silently added no
    clause at all — so the malformed filter a caller most likely typed by mistake
    was the one that answered with the whole knowledge base instead of an error.
    Widening a search is the worse of the two failures, and it was the quiet one.

    ``json.dumps`` is wrapped because it raises ``TypeError`` for a value it
    cannot serialise — a ``set``, a ``datetime``, anything with no encoder — and
    this function's contract, and the route that turns it into a 400, is
    ``ValueError``.

    Two conventions the caller has to match: the filtered item table is aliased
    ``c``, and the bound parameter is named ``filter_metadata``.

    This is containment, not equality — ``{"a": {"b": 1}}`` matches a row whose
    ``meta`` is ``{"a": {"b": 1, "c": 2}}``. Note the bm25s file-index keyword
    leg does not come through here: it filters in Python with ``==`` on each
    key, which is stricter, and the two have never agreed on nested values.
    """
    if filter_metadata is None:
        return "", {}
    if not isinstance(filter_metadata, dict):
        raise ValueError(
            f"filter_metadata must be a JSON object, got {type(filter_metadata).__name__}"
        )
    if not filter_metadata:
        return "", {}
    try:
        encoded = json.dumps(filter_metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"filter_metadata is not valid JSON: {exc}") from exc
    return (
        f" AND c.meta @> CAST(:{METADATA_FILTER_PARAM} AS jsonb)",
        {METADATA_FILTER_PARAM: encoded},
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
# bad values, which is bounded by how often someone writes the setting.
_WARNED_TIMEOUT_OVERRIDES: set[str] = set()

# pg_search keyword-search failures already reported at WARNING, keyed by KB,
# table and the failure's first line, so a different cause of the same
# exception type is reported too. One per knowledge base can add up, so the
# set is cleared when it reaches the bound: a persisting failure then warns
# again, once.
_WARNED_PG_BM25_FAILURES: set[str] = set()
_WARNED_PG_BM25_FAILURES_MAX = 1024


def _warn_once_per_bad_value(key: str, message: str, *args: Any) -> None:
    """WARNING the first time this exact bad value is seen, DEBUG afterwards."""
    if key in _WARNED_TIMEOUT_OVERRIDES:
        logger.debug(message, *args)
        return
    _WARNED_TIMEOUT_OVERRIDES.add(key)
    logger.warning(message, *args)


def _warn_once_per_pg_bm25_failure(key: str, message: str, *args: Any) -> None:
    """WARNING the first time this failure is seen (within the bound), DEBUG afterwards."""
    if key in _WARNED_PG_BM25_FAILURES:
        logger.debug(message, *args)
        return
    if len(_WARNED_PG_BM25_FAILURES) >= _WARNED_PG_BM25_FAILURES_MAX:
        _WARNED_PG_BM25_FAILURES.clear()
    _WARNED_PG_BM25_FAILURES.add(key)
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

        The per-dimension HNSW index on ai.embeddings spans every KB, and for a
        KB without a partial index of its own the vector query filters
        `knowledge_base_id` AFTER the approximate scan. Without iterative
        scanning, pgvector emits only ~ef_search global candidates before that
        filter, starving KB-scoped queries (often 0 rows). A KB that has its own
        partial index (`pg_vector_index`) does not need this -- its index holds
        only its own rows, so nothing is filtered away after the scan -- but it
        costs that KB nothing either, so the GUC is set unconditionally rather
        than made to depend on a catalog lookup per search.

        SET LOCAL keeps this scoped to the current transaction so it
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

    @contextmanager
    def _forcing_a_custom_plan(self) -> Iterator[None]:
        """Plan the statements in this block against their real parameters.

        Two callers: a search carrying a metadata filter, and the exact re-run in
        ``_search_again_exactly``. Everything the partial HNSW index's predicate
        needs is a literal (see ``kb_sql_literal``), so an unfiltered search that
        is not being re-run keeps whatever plan the cache holds. A metadata
        filter cannot be a literal -- it is caller data -- and a plan built
        without its value has no selectivity estimate for ``meta @> $n`` at all,
        so the planner stops believing the ordered index scan will stop early
        and prices an exact sort below it instead. That is the whole partial
        index lost, for the life of a pooled connection, on a documented and
        shipped search parameter.

        Forcing a custom plan is the direct fix: the planner sees the filter's
        actual value, estimates it, and keeps the index. The cost is one replan
        per filtered search, which is cheap against what it buys.

        Measured on a 20,000-embedding knowledge base with its partial index, on
        a connection whose plan cache had gone generic, 20 filtered searches
        through the real driver in one transaction:

        | filtered search | plans on the partial index | latency |
        |---|---|---|
        | without this | 0 of 20 -- bitmap scan and an exact sort | 8.7-10.8 ms |
        | with this | 20 of 20 | 2.5-6.0 ms |

        The unfiltered shape needs none of it -- 20 of 20 on the index on the
        same pinned-generic connection -- which is what the literals are for.
        The gap grows with the knowledge base, because what replaces the index
        is an exact scan of it.

        **Restored on the way out, which is why this is a block and not the bare
        setter it used to be** (``_force_custom_plan``, in case an older comment
        still names it). Transaction-local was never the whole of it: in the
        single-knowledge-base fast path the session is the request's own, so
        everything the request does after the vector leg -- the hybrid keyword
        leg, the metadata fetches, the reranker's reads, the billing writes --
        would keep replanning for the rest of the transaction. The setting has
        to come off when the statement it was asked for is done, the same way
        ``_fetch_with_timeout`` puts its budget back and
        ``_preferring_this_kbs_partial_index`` puts ``enable_sort`` back.

        ``set_config(..., true)`` rather than ``SET LOCAL`` so the previous value
        can be bound. Same caveat as ``_apply_iterative_scan``: it needs the
        session to be in a transaction, which is how this store is used, and a
        failure here degrades latency rather than the answer, so it is logged
        and not raised -- and the restore is expected to fail when the search
        itself did, because the transaction is then aborted and the setting dies
        with it anyway.
        """
        prior: str | None = None
        try:
            prior = str(
                self.session.execute(text("SELECT current_setting('plan_cache_mode')")).scalar()
            )
            self.session.execute(
                text("SELECT set_config('plan_cache_mode', 'force_custom_plan', true)")
            )
        except Exception as e:  # pragma: no cover - depends on server version
            logger.warning(
                "Could not set plan_cache_mode=force_custom_plan: %s; a filtered vector "
                "search may miss this knowledge base's partial HNSW index",
                e,
            )
            yield
            return
        try:
            yield
        finally:
            try:
                self.session.execute(
                    text("SELECT set_config('plan_cache_mode', :prior, true)"), {"prior": prior}
                )
            except Exception as e:
                logger.debug(
                    "Could not restore plan_cache_mode=%s after a vector search on KB %s: %s",
                    prior,
                    self.kb_id,
                    e,
                )

    # One round trip that answers every question the block below needs: whether
    # this knowledge base has a partial HNSW index worth steering the planner
    # towards, and what ``enable_sort`` and ``hnsw.ef_search`` are set to so they
    # can be put back. Independent scalars rather than a nested ``CASE``, so none
    # of them depends on the order the target list is evaluated in.
    #
    # ``indisvalid`` and not merely "the name exists": a CONCURRENTLY-built index
    # is in the catalog from the moment the build starts and cannot answer a
    # query until it finishes, and a failed build leaves one behind for good.
    # Steering the planner at an index in either state is exactly the case
    # requirement 2 of this fix is about.
    #
    # ``hnsw.ef_search`` is read with ``missing_ok`` because it is pgvector's GUC,
    # not PostgreSQL's: on a database without the extension loaded the name does
    # not exist and a plain ``current_setting`` would raise, failing the whole
    # probe. NULL there means "no pgvector", which also means no HNSW index of
    # any kind, so the block simply leaves that setting alone.
    _PARTIAL_INDEX_PROBE = """
        SELECT
            current_setting('enable_sort') AS prior_sort,
            current_setting('hnsw.ef_search', true) AS prior_ef_search,
            EXISTS (
                SELECT 1 FROM pg_index
                WHERE indexrelid = to_regclass(:index) AND indisvalid
            ) AS usable
    """

    @contextmanager
    def _preferring_this_kbs_partial_index(self, dims: int) -> Iterator[bool]:
        """Price an exact sort out of the search, when there is an index to fall on.

        Yields whether the sort was priced out.

        Everything else in this class makes the partial HNSW index *reachable*.
        This is what makes the planner *take* it, and it is needed because at
        1536 dimensions -- the width most embedding models here produce -- the
        planner's arithmetic comes out the wrong way round.

        A 1536-value vector does not fit in a heap tuple, so it is stored out of
        line: the heap stays small (656 pages for 40,000 rows) while the HNSW
        index holds about one tuple per page (12,001 pages for 12,000 tuples).
        PostgreSQL then prices an exact scan as 656 pages plus a sort of narrow
        tuples, and prices detoasting -- 12,000 out-of-line reads and 12,000
        1536-value distance computations -- at nothing at all. So the exact scan
        is systematically underpriced and the ordered index scan overpriced, and
        the gap does not close as the knowledge base grows: on a
        20-knowledge-base, 39,995-row fixture at 1536 dimensions the planner
        declined the index at every share of the table from 21% to 70%. At 384
        dimensions the vector is inline, the two paths cost about the same, and
        the planner takes the index on its own -- which is why this was invisible
        until the suite was measured at a production width.

        Measured through ``vector_search`` itself, 12 executions on one pooled
        connection under ``force_generic_plan``, ``top_k`` 20, on that fixture,
        with the knowledge base's partial index built and valid:

        | knowledge base | search | before | with this |
        |---|---|---|---|
        | 30 % of the table | unfiltered | 0/12 on the index, 50.5 ms | 12/12, 5.2 ms |
        | 30 % | one metadata key | 0/12, 15.2 ms | 12/12, 8.5 ms |
        | 30 % | two metadata keys | 0/12, 15.0 ms | 12/12, 7.4 ms |
        | 21 % of the table | unfiltered | 0/12, 38.0 ms | 12/12, 4.7 ms |

        The filtered rows are the second thing this repairs, and the reason is
        not the one it looks like. ``_forcing_a_custom_plan`` gets the filter's value
        to the planner, which prices ``meta @> const`` by matching that constant
        against ``meta``'s most-common-value list -- so the estimate is real, but
        it is then multiplied by the knowledge-base predicate as though the two
        were independent. They are not, when one of the filter's keys *is* the
        knowledge base. Measured at 384 dimensions on the live fixture:
        ``{"tier": "gold"}`` estimates 722 rows of 2,400 real and keeps the
        index, and filtering on the knowledge base alone estimates 1,076 of
        12,000 and keeps it -- but the two together estimate **217 of the same
        2,400** and lose it, an ordered index scan at 5,060 against a sort at
        3,837. So it is an 11x-low correlated estimate rather than a key count:
        two uncorrelated keys are likely fine. The estimate is wrong, not the
        clause, so the fix belongs here rather than in how the filter is
        compiled.

        **Only with an index of this knowledge base's own.** This is a cost
        penalty on every sort in the statement, not an instruction to use a
        particular index, so with no partial index it drives the query onto the
        *shared* per-dimension index, which spans every knowledge base and
        post-filters. Measured on the same fixture with no partial index built,
        median of six query vectors:

        | knowledge base | planner's own choice | this, ungated |
        |---|---|---|
        | 21 %, 8,400 rows | exact scan, 34.4 ms, recall 1.00 | shared index, 6.3 ms |
        | 5 %, 2,000 rows | exact scan, 9.4 ms, recall 1.00 | shared index, 31.3 ms |
        | 1 %, 400 rows | exact scan, 1.8 ms, recall 1.00 | shared index, 39.8 ms |

        Mostly *slower*, and with a bad tail on the small knowledge bases. The
        recall figures this table used to carry (0.08 at 21 %, 0.04 at 5 %) are
        an artifact of the fixture rather than a property of the shared index:
        measured again on real embeddings the same shapes give 0.927 and 0.964.
        What does survive on real data is the latency inversion -- the shared
        index gets slower as the knowledge base gets smaller, because the
        post-filter discards more, 7.40 ms at 1 % against an exact scan's 1.66 --
        and the tail, where 1 % is the one cell in the whole real-embedding run
        with a minimum recall below 0.5.

        That is what the catalog probe buys: a knowledge base below the build
        threshold, one whose index is INVALID, and one whose index is still being
        built all keep the plan they have today -- measured, 0 of 12 executions on
        the index and recall 1.00 in each case. The probe costs a round trip,
        which for a knowledge base that has no index is the whole of what this
        adds: +0.3 ms, measured over 120 searches each at 400, 2,000 and 8,400
        rows.

        **Only for an unrestricted search, and that is the other half of the
        gate.** The probe answers "does this knowledge base have a valid partial
        index", which is not the same proposition as "is this search better off
        on it". Any extra predicate turns the ordered scan into a walk of the
        index for rows the join or the restriction then throws away, and every
        restricted shape measured at 1536 dimensions through ``vector_search``
        came out both slower and less accurate:

        | search | planner's choice | on the index |
        |---|---|---|
        | unfiltered | 46.0 ms, recall 1.00 | **15.3 ms** |
        | ``source_ids``, one source of six | 22.7 ms, recall 1.00 | 28.5 ms, recall 0.49 |
        | ``item_ids``, 200 named | 11.0 ms, recall 1.00 | 37.8 ms, recall 0.80 |
        | ``filter_metadata`` matching no row | 3.0 ms | 38.3 ms |
        | ``source_ids`` matching no row | 2.4 ms | 36.7 ms |

        Row one is the whole of what this mechanism is for. So a search carrying
        ``item_ids``, ``source_ids`` or a metadata filter does not enter this
        block at all, and keeps the plan the planner chooses for it.

        ``_search_again_exactly`` is not a substitute for that, which is the
        mistake worth not repeating: it fires on ``len(items) < top_k``, so it
        catches a restriction that starves the ``LIMIT`` *below* the limit and
        misses one that starves it *at* the limit -- a restriction with more than
        ``top_k`` matching rows fills the LIMIT with the wrong rows and looks like
        a complete answer. It fired 0 times in all four restricted rows above.
        A row count cannot be the completeness signal for a restricted search.

        The cost of a restricted search that did get here anyway is bounded
        rather than proportional: pgvector stops an iterative scan at
        ``hnsw.max_scan_tuples``, 20,000 by default, so this is a ceiling and not
        something that grows with the knowledge base.

        Recall is the trade even when the index is there, and it is a real trade
        rather than the collapse this docstring used to imply. On real embeddings
        -- 39,995 passages, 300 held-out queries encoded asymmetrically -- an
        unfiltered search on the index returns 0.915 at 30 % of the table and
        0.933 at 21 %, with the worst of 300 queries at 0.70 and nothing below
        half. The 0.22/0.28 this fixture gives at the same selectivities is a
        generator artifact: clustered random noise in 1536 dimensions puts almost
        every pair about equally far apart, so there is barely a nearest
        neighbour to find. Neither number is a production figure; the first is the
        one to reason from.

        Because that trade grows with the *absolute* size of the index, this block
        also raises ``hnsw.ef_search`` to ``PER_KB_HNSW_EF_SEARCH`` -- see that
        constant for the measurement, and for why the supported band has an upper
        bound. It is set here rather than globally for the same reason
        ``enable_sort`` is: it is only the right value while the search is on a
        knowledge base's own index, and the two are otherwise independent.

        ``set_config(..., true)`` rather than ``SET LOCAL`` so the previous values
        can be bound; the third argument is what makes them transaction-local.
        Transaction-local is not enough on its own here, and that is why every
        setting this store touches is now restored rather than left to the
        transaction: ``hybrid_search`` runs its keyword leg on the same session
        immediately after the vector leg, and a keyword ranking is a sort. So the
        previous values go back on before this returns, the same way
        ``_fetch_with_timeout`` restores its budget and
        ``_forcing_a_custom_plan`` restores ``plan_cache_mode``.

        A failure on any side degrades latency, never the answer, so all of them
        are logged rather than raised -- and the restore is expected to fail when
        the search itself did, because the transaction is then aborted and the
        settings die with it anyway.

        The probe runs in a savepoint. Without one, a probe that errors -- a
        catalog lookup can be cancelled like anything else -- would leave the
        caller's transaction aborted, so the search that follows would fail with
        an unrelated error while this warned that it "may fall back to an exact
        scan". Rolling back to the savepoint makes the warning true.
        """
        # Not inside the try below: both arguments have already been validated by
        # the caller, so a failure here is a programming error and should not be
        # logged as a missing index.
        index = f'"{self.schema}".{pg_vector_index.per_kb_index_name(self.kb_id, dims)}'
        prior: str | None = None
        prior_ef_search: str | None = None
        try:
            with self.session.begin_nested():
                rows = list(self.session.execute(text(self._PARTIAL_INDEX_PROBE), {"index": index}))
            if rows and rows[0][2]:
                prior = str(rows[0][0])
                prior_ef_search = None if rows[0][1] is None else str(rows[0][1])
        except Exception as e:  # pragma: no cover - needs a live catalog
            logger.warning(
                "Could not check for KB %s's partial HNSW index at %d dimensions: %s; "
                "this vector search may fall back to an exact scan",
                self.kb_id,
                dims,
                e,
            )
        if prior is None:
            yield False
            return
        try:
            self.session.execute(text("SELECT set_config('enable_sort', 'off', true)"))
        except Exception as e:  # pragma: no cover - needs a live server
            logger.warning(
                "Could not price the exact sort out for KB %s: %s; this vector search "
                "may miss the knowledge base's partial HNSW index",
                self.kb_id,
                e,
            )
            yield False
            return
        if prior_ef_search is not None:
            try:
                self.session.execute(
                    text("SELECT set_config('hnsw.ef_search', :ef, true)"),
                    {"ef": str(PER_KB_HNSW_EF_SEARCH)},
                )
            except Exception as e:  # pragma: no cover - depends on pgvector version
                # The search still runs, on the index, at pgvector's default
                # ef_search -- lower recall than intended, not a wrong answer, so
                # the block goes ahead rather than giving the index up.
                logger.warning(
                    "Could not raise hnsw.ef_search to %d for KB %s: %s; this vector "
                    "search will run at pgvector's default recall",
                    PER_KB_HNSW_EF_SEARCH,
                    self.kb_id,
                    e,
                )
                prior_ef_search = None
        try:
            yield True
        finally:
            # One statement per setting, each naming its own GUC, so the restore
            # is as readable in a captured statement list as the set was.
            if prior_ef_search is not None:
                try:
                    self.session.execute(
                        text("SELECT set_config('hnsw.ef_search', :prior, true)"),
                        {"prior": prior_ef_search},
                    )
                except Exception as e:
                    logger.debug(
                        "Could not restore hnsw.ef_search=%s after a vector search on KB %s: %s",
                        prior_ef_search,
                        self.kb_id,
                        e,
                    )
            try:
                self.session.execute(
                    text("SELECT set_config('enable_sort', :prior, true)"), {"prior": prior}
                )
            except Exception as e:
                logger.debug(
                    "Could not restore enable_sort=%s after a vector search on KB %s: %s",
                    prior,
                    self.kb_id,
                    e,
                )

    def _search_again_exactly(
        self, run_the_search, first_answer: list[RetrievedItem]
    ) -> list[RetrievedItem]:
        """Re-run a short answer with the sort available again, and keep the better one.

        A short answer from an ordered HNSW scan means either that few rows really
        match, or that the scan starved: pgvector's iterative scan keeps going,
        but under ``strict_order`` it does not reach every tuple in the index.
        Measured at 384 dimensions, a ``top_k`` of 20 against 12 matching rows of
        a 12,000-row knowledge base: the forced index scan reached 11,170 of
        12,006 index tuples and returned 11 of the 12 for two of six query
        vectors, where the exact scan the planner chooses on its own returns all
        12 every time.

        The plan the planner prefers with the sort available answers both cases
        exactly, and answers them cheaply, because what makes an exact scan
        expensive is having many rows to sort. So ask again. A full answer, which
        is the ordinary case, never pays it.

        **This is a net for starvation, not a completeness guarantee, and it is
        not what makes a restricted search safe.** It fires on ``len(items) <
        top_k``, which catches a restriction that starves the ``LIMIT`` below the
        limit and misses one that starves it *at* the limit -- more matching rows
        than ``top_k``, the wrong ones returned, nothing short to notice. That is
        why ``item_ids``, ``source_ids`` and ``filter_metadata`` searches no longer
        enter ``_preferring_this_kbs_partial_index`` in the first place; see its
        docstring for the measurements. What reaches here is an unrestricted
        search that came up short.

        **The longer answer wins, rather than the second one.** Both plans read
        the same rows under the same ``LIMIT``, so the re-run is normally at least
        as complete -- but it is a different plan, and a second plan that comes
        back shorter is not evidence of anything except that it came back
        shorter. Replacing a longer first answer with it would turn this net into
        a way to lose rows.

        The custom plan is what makes the re-run a different plan. PostgreSQL
        builds a statement's generic plan once and does not rebuild it when a
        planner GUC changes, so the plan built while the sort was priced out is
        the plan a second execution would get, and it would come up equally
        short. Asking for a custom plan re-plans against the settings now in
        force. With ``plan_cache_mode`` left at ``auto`` it happens not to be
        needed -- the generic plan carries the disabled sort's cost, so PostgreSQL
        keeps preferring a custom plan anyway -- which is exactly the kind of
        thing not to depend on: removing it makes the live spec's generic-plan leg
        fail.
        """
        logger.debug(
            "Vector search on KB %s returned %d rows from the partial HNSW index, fewer "
            "than top_k; re-running it exactly",
            self.kb_id,
            len(first_answer),
        )
        with self._forcing_a_custom_plan():
            again = run_the_search()
        return again if len(again) >= len(first_answer) else first_answer

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
        # int() truncates rather than rejects, so a float 1536.9 would search as
        # 1536 -- surprising, safe to emit, and not reachable from a caller that
        # takes the value off a stored embedding.
        if not (1 <= effective_dims <= 8192):
            raise ValueError(f"dims must be between 1 and 8192, got {effective_dims}")
        effective_top_k = validated_top_k(top_k)

        # The embeddings-side knowledge_base_id predicate is what lets the
        # planner use this KB's partial HNSW index, if it has one: a partial
        # index is only matched from a restriction clause on the relation it is
        # on, and the planner does not reason through `e.item_id = c.id` to
        # reach `c.knowledge_base_id`. Without it the plan picks the shared
        # per-dimension index and post-filters -- verified by EXPLAIN with the
        # partial index present.
        #
        # The item-table filter stays, and not for the reason this comment used
        # to give. It is not a correctness guard: an embedding whose item no
        # longer exists is dropped by the inner join, not by this clause, and
        # nothing in this service ever moves an item between knowledge bases --
        # both rows are written from one value in the same transaction and no
        # statement anywhere updates either column -- so the two columns cannot
        # disagree. It stays because it is the item table's *partition key*. The
        # item tables are partitioned BY LIST (knowledge_base_id), so this is
        # the clause that prunes the scan to one partition; the join alone would
        # read every one of them, as `_iter_items_for_kb_bm25` says for the same
        # reason. Measured at 1536 dimensions on a 20-knowledge-base fixture
        # whose item table was partitioned exactly as deployed, dropping it cost
        # 5.1 -> 9.8 ms at 1% of the table and 18.0 -> 26.9 ms at 5%.
        #
        # It is not free either, and the cost is a row estimate rather than a
        # correctness risk. `c.knowledge_base_id = K` and `e.knowledge_base_id =
        # K` are perfectly correlated, and the planner multiplies them as
        # independent: the join estimate becomes
        # `rows(c matching K) * rows(e matching K) / n_distinct(e.item_id)`,
        # which is the true count times the knowledge base's share of the table.
        # Measured on the fixture above, each shape against 12,000, 8,400, 2,000
        # and 400 real rows:
        #
        # | shape                            | 30 %   | 21 %  | 5 %   | 1 %  |
        # |---|---|---|---|---|
        # | `c.knowledge_base_id = K` alone  | 12,074 | 8,339 | 2,028 | 393  |
        # | `e.knowledge_base_id = K` alone  | 12,076 | 8,394 | 2,002 | 396  |
        # | both, i.e. this statement        |  3,646 | 1,750 |   102 |   4  |
        #
        # So a 1 % knowledge base is estimated 100x low, and that underestimate
        # is part of what `_preferring_this_kbs_partial_index` has to overcome.
        # Dropping the clause corrects the estimate exactly and is still the
        # wrong trade: it loses the pruning above, and it does not let the
        # planner choose for itself. Measured with the estimate corrected and
        # each knowledge base's partial index built and valid, the planner still
        # took the exact scan at 21 %, 5 % and 1 % of the table (41, 13 and
        # 4.8 ms against 1.7, 2.3 and 1.3 ms once the sort was priced out); only
        # at 30 % did it take the index unaided. Correcting the estimate is
        # therefore not an alternative to the gate.
        #
        # The KB id is interpolated on BOTH sides, and the second one was
        # measured rather than reasoned about. The embeddings-side literal is
        # what makes the index *matchable*; it is not what makes the planner
        # choose it. With `c.knowledge_base_id` still bound, a generic plan has
        # no row estimate for the item side of the join, so it prices a hash
        # join plus an exact sort below the ordered index scan the index would
        # drive -- the index is matchable and not chosen. Measured through the
        # real driver on one pooled connection, on a table where the KB is a
        # small fraction of the rows (the regime this feature exists for):
        # ~0.98 ms while the custom plan held, ~135 ms from the execution the
        # generic plan was adopted on, and for the life of that connection.
        # Both literals come from the same validated gate, so the second costs
        # no new injection surface.
        #
        # So four values decide whether a prepared statement keeps the index:
        # the KB id on each side, `dims`, and the LIMIT. All four are
        # interpolated -- a generic plan can only match the index's predicate
        # when it can prove all of it, and the planner's estimate for an unknown
        # LIMIT prices the ordered index scan out. kb_sql_literal has the
        # measured table; all of them are validated above.
        #
        # What cannot be a literal is the metadata filter below: it is caller
        # data, so it is bound as jsonb, and a generic plan has no selectivity
        # estimate for `@>` at all. That is why a filtered search asks for a
        # custom plan instead -- see _forcing_a_custom_plan.
        kb_literal = kb_sql_literal(self.kb_id)
        query = f"""
            SELECT
                c.id,
                c.{self.TEXT_COL},
                1 - ((e.embedding::vector({effective_dims})) <=> CAST(:embedding AS vector({effective_dims}))) AS similarity,
                c.source_id,
                c.meta
            FROM "{self.schema}".{self.TABLE} c
            JOIN "{self.schema}".embeddings e ON e.item_id = c.id
            WHERE c.knowledge_base_id = {kb_literal}
              AND e.knowledge_base_id = {kb_literal}
              AND e.dims = {effective_dims}
        """

        params: dict[str, Any] = {
            "embedding": embedding_str,
        }

        if item_ids is not None:
            query += " AND c.id = ANY(CAST(:item_ids AS uuid[]))"
            params["item_ids"] = "{" + ",".join(item_ids) + "}"

        if source_ids is not None:
            query += " AND c.source_id = ANY(CAST(:source_ids AS uuid[]))"
            params["source_ids"] = "{" + ",".join(source_ids) + "}"

        filter_sql, filter_params = metadata_filter_clause(filter_metadata)
        query += filter_sql
        params.update(filter_params)

        query += f"""
            ORDER BY (e.embedding::vector({effective_dims})) <=> CAST(:embedding AS vector({effective_dims}))
            LIMIT {effective_top_k}
        """

        def run_the_search() -> list[RetrievedItem]:
            result = self.session.execute(text(query), params)
            return [
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

        # Keyed on the arguments, not on the clauses built above: how a
        # restriction is compiled may change, the reason a restricted search must
        # keep the planner's own plan does not. ``is not None`` rather than
        # truthiness for the two id sets, so it matches exactly the condition
        # under which a clause was added -- an empty set still narrows the search
        # to nothing, which is the most starved restriction there is.
        restricted = item_ids is not None or source_ids is not None or bool(filter_metadata)

        try:
            self._apply_iterative_scan()
            with ExitStack() as scope:
                if filter_metadata:
                    scope.enter_context(self._forcing_a_custom_plan())
                priced_out = False
                if not restricted and self.TABLE == PER_KB_INDEX_ITEM_TABLE:
                    # Inside the block, because restoring enable_sort must not
                    # happen until the rows are off the cursor. The plan is fixed
                    # when the statement executes, so this is belt and braces --
                    # but the belt is free and the alternative depends on how the
                    # driver buffers.
                    priced_out = scope.enter_context(
                        self._preferring_this_kbs_partial_index(effective_dims)
                    )
                items = run_the_search()
            if priced_out and len(items) < effective_top_k:
                items = self._search_again_exactly(run_the_search, items)
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

        **This path deliberately keeps the planner's own plan**, where
        ``vector_search`` steers it: no partial-index gate, no custom plan, and
        no raised ``ef_search``. It is not an oversight and it is not the
        inconsistency it looks like from the KB literal it does carry. The query
        has no ``LIMIT`` on the distance order -- it scores every row of the
        knowledge base by design, then ranks within each source -- so there is no
        ordered-index-scan-against-sort race for the gate to win. Measured at 1536
        dimensions on a 12,000-row knowledge base with its partial index built and
        valid: the plan reaches no HNSW index in any configuration, it contains
        six sort nodes that no setting can remove, ``enable_sort = off`` makes it
        **4.8x slower** (50.5 -> 244.7 ms) while changing nothing about which index
        it uses, and ``ef_search`` is inert (48.3 ms, within noise). The gate would
        be a pure regression here.

        The remaining parameters stay bound for the same reason: with no ordered
        index scan there is no unknown row estimate to price one out, so a generic
        plan costs nothing and one shared statement text serves every knowledge
        base. A ``LIMIT`` on the distance order would make this ``vector_search``'s
        shape, and all of that would have to be revisited together.
        """
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        effective_dims = int(dims or len(embedding))
        if not (1 <= effective_dims <= 8192):
            raise ValueError(f"dims must be between 1 and 8192, got {effective_dims}")

        source_filter = ""
        params: dict[str, Any] = {
            "embedding": embedding_str,
            "kb_id": self.kb_id,
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
        # This leg already carried the embeddings-side knowledge_base_id
        # predicate; it and `dims` are literals here for the same reason
        # vector_search's are (see kb_sql_literal), so a prepared statement's
        # generic plan can prove the partial index's whole predicate. Unlike
        # vector_search this query has no outer LIMIT on the distance order -- it
        # scores the whole knowledge base by design, so no HNSW index is used
        # either way, and the remaining parameters stay bound.
        #
        # That is also why the item-table id below is still bound, where
        # vector_search interpolates it on both sides: there is no ordered index
        # scan here for an unknown row estimate to price out, so the bind costs
        # nothing -- and it keeps one statement text shared across knowledge
        # bases instead of one per knowledge base in the driver's
        # prepared-statement cache. A LIMIT on the distance order would make this
        # query vector_search's shape and the bind would then matter.
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
                  AND e.knowledge_base_id = {kb_sql_literal(self.kb_id)}
                  AND e.dims = {effective_dims}
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

        filter_sql, filter_params = metadata_filter_clause(filter_metadata)
        search_query += filter_sql
        params.update(filter_params)

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

        Matching differs from the tsvector fallback: ``|||`` matches a row
        containing ANY of the query's terms and lets the BM25 score rank rows
        with more of them higher (like the bm25s file index), while the
        fallback's ``websearch_to_tsquery`` requires ALL terms. So a multi-word
        query can return rows here that the fallback would not.
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

        filter_sql, filter_params = metadata_filter_clause(filter_metadata)
        search_query += filter_sql
        params.update(filter_params)

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

    def _file_index_retired(self) -> bool:
        """Has this KB's keyword index moved to pg_search for good? Never raises.

        True once the extension is installed and the KB has its own attached
        partition of this table: from then on nothing maintains the bm25s file
        index. "Can't tell" is False, which keeps today's behaviour.
        """
        try:
            if self.TABLE not in pg_bm25_index.PARTITIONED_ITEM_TABLES:
                return False
            if not pg_bm25_index.pg_search_installed(self.session):
                return False
            return pg_bm25_index.partition_exists(self.session, self.kb_id, self.TABLE)
        except Exception as exc:
            _warn_once_per_pg_bm25_failure(
                f"{self.kb_id}:{self.TABLE}:file-index-check:{pg_bm25_index.first_error_line(exc)[:200]}",
                "Could not tell whether KB %s has its own partition of %s (%s); reading its "
                "bm25s file index if it has one",
                self.kb_id,
                self.TABLE,
                pg_bm25_index.first_error_line(exc),
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
        that index is ready. Otherwise the pre-built bm25s file index -- unless
        the KB already has its own partition, whose file index is no longer
        maintained -- and failing that the bounded tsvector fallback.

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
                cause = pg_bm25_index.first_error_line(exc)
                _warn_once_per_pg_bm25_failure(
                    f"{self.kb_id}:{self.TABLE}:{cause[:200]}",
                    "pg_search keyword search failed for KB %s table %s (%s); "
                    "falling back to the existing keyword path",
                    self.kb_id,
                    self.TABLE,
                    cause,
                )

        if self._file_index_retired():
            # Once this KB has its own partition, indexing no longer maintains
            # its bm25s file index, so that file is frozen at the move. While
            # the KB's own index is not usable (being built, rebuilt for a new
            # language, INVALID), answer from the live rows instead.
            # Once per KB and table at WARNING: every search takes this path
            # while the index is missing, and the fallback can time out on a
            # large KB, so an index that never comes back must be visible.
            _warn_once_per_pg_bm25_failure(
                f"{self.kb_id}:{self.TABLE}:tsvector-fallback",
                "pg_search index for KB %s table %s is not usable (building, INVALID, or not "
                "built on this server); keyword search falls back to the slower tsvector scan "
                "until it is. See the KB's bm25_status",
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

        # Checked here, against the caller's own value, because the vector leg
        # below is handed ``top_k * 2`` -- so left to that leg the error message
        # would name a number the caller never sent, and a ceiling reported as
        # twice what it is cannot be acted on.
        if validated_top_k(top_k) * 2 > MAX_TOP_K:
            raise ValueError(
                f"top_k must be between 0 and {MAX_TOP_K // 2} for hybrid search, "
                f"which fetches twice it from each leg, got {top_k}"
            )
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
