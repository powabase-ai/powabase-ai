"""
Knowledge Search Service.

Provides a reusable service function for searching knowledge bases.
Used by both the KB search API endpoint and the agent run endpoint.
"""

import asyncio
import logging
from typing import Any, NamedTuple

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from agentic.knowledge.model_config import (
    CHUNK_EMBED_EMBEDDING_MODEL,
    HYBRID_DEFAULT_VECTOR_WEIGHT,
    PAGEINDEX_RETRIEVAL_MODEL,
    QUERY_ENRICHMENT_DEFAULT_MODEL,
    RERANKER_CANDIDATE_COUNT,
)

from . import billing_port as billing
from .base_vector_store import BasePgVectorStore
from .doc2json_store import Doc2JSONStore
from .full_document_store import FullDocumentStore
from .graph_index_node_store import GraphIndexNodeStore
from .graph_index_store import GraphIndexStore
from .knowledge_store import PgVectorKnowledgeStore, RetrievedItem
from .page_index_store import PageIndexStore
from .run_context import (
    get_run_id,
    new_request_id,
)
from .storage import get_storage
from ..db import AI_SCHEMA
from ..strategies import get_default_retrieval_method, validate_retriever
from ..strategies.graph_defaults import (
    GRAPH_DEFAULT_INCLUDE_CHILDREN,
    GRAPH_DEFAULT_INCLUDE_DOC_TOC,
    GRAPH_DEFAULT_MAX_CHILDREN,
    GRAPH_DEFAULT_MAX_REFERENCED_NODES,
    GRAPH_MAX_CHILDREN_CEILING,
    GRAPH_MAX_REFERENCED_CEILING,
)

from agentic.knowledge.retrieval import TreeSearchAlgorithm, apply_source_limits

logger = logging.getLogger(__name__)


class EmptyKnowledgeBaseError(ValueError):
    """A search hit a KB that has no indexed documents yet.

    Subclasses ValueError so existing ``except ValueError`` callers keep working,
    while callers that need to tell "empty KB" apart from a genuine retrieval /
    config error (bad retrieval_config, embedding-dim mismatch, pgvector parse
    failure — which also raise ValueError) can catch this specifically instead of
    masking a broken index as "no results". See routes/internal_docs.py.
    """


# When per-source limits are active we must fetch a candidate pool larger than
# top_k so there's a surplus to diversify over. Multiply top_k by this factor,
# capped so a pathological top_k can't trigger a huge scan. Factor 5 gives ample
# headroom for typical top_k (5-10); the cap binds only at top_k >= 40, beyond
# which the per-source floor back-fill (not raw over-fetch) does the real work.
PER_SOURCE_CANDIDATE_FACTOR = 5
PER_SOURCE_CANDIDATE_CAP = 200

# A global top-N retrieval is dominated by large sources, so the min_per_source
# diversity floor must explicitly back-fill each matched source's top chunks.
# Bound how many sources we back-fill per search so a KB with thousands of
# small sources can't blow up the candidate pool.
PER_SOURCE_FLOOR_SOURCE_CAP = 50

# Outline lines per document. A 500-section contract is ~5k tokens of pure
# structure otherwise; beyond this the outline truncates with a marker.
GRAPH_TOC_MAX_NODES = 200

# Retrieval methods whose items carry structure rather than source content.
# They have no pages, and must be kept out of the image page-coverage scan.
STRUCTURAL_RETRIEVAL_METHODS = frozenset({"graph_toc"})


def _read_source_limits(
    retrieval_config: dict | None,
) -> tuple[int | None, int | None]:
    """Extract (min_per_source, max_per_source) from a retrieval_config dict.

    Returns (None, None) when the keys are absent or non-positive (disabled).
    """
    if not retrieval_config:
        return None, None

    def _coerce(key: str) -> int | None:
        raw = retrieval_config.get(key)
        if raw is None:
            return None
        try:
            val = int(raw)
        except (TypeError, ValueError):
            # Present but unparseable (e.g. "1O", a list). The Studio form
            # validates these, but API/agent callers bypass it — surface it
            # rather than silently disabling the limit.
            logger.warning("Ignoring non-integer %s in retrieval_config: %r", key, raw)
            return None
        return val if val > 0 else None

    return _coerce("min_per_source"), _coerce("max_per_source")


def _merge_floor_items(
    main: list[RetrievedItem],
    floor: list[RetrievedItem],
    rescore_below_main: bool = False,
) -> list[RetrievedItem]:
    """Union the main candidate pool with floor back-fill items, deduped by
    item_id (the main pool's copy wins on collision). Tags each newly-added
    floor item's ``meta["source_floor"]`` in place (items are freshly fetched
    here, so the mutation is local).

    ``rescore_below_main``: the floor query returns COSINE similarities, which
    are not commensurable with a hybrid (RRF) or full_text (BM25) main pool.
    Without a reranker to re-score the merged pool uniformly, raw cosine values
    would sort floor items above genuine top matches and corrupt the ranking.
    When set, newly-added floor items are re-scored to sit just below the main
    pool's minimum (preserving their relative order). Note this still uses a
    fixed 0.01 step, where ``_expansion_tier_scores`` derives one from the
    pool's own minimum; the two should converge. For pure ``vector_search``
    the pool is already cosine, so this is left off and true scores kept.
    """
    seen = {it.item_id for it in main}
    new_items = [it for it in floor if it.item_id not in seen]

    if rescore_below_main and new_items:
        main_min = min((it.score for it in main if it.score is not None), default=0.0)
        # new_items arrive best-first (floor query orders by distance asc); place
        # them in a descending band just below main_min so they rank as diversity
        # additions, never above the main pool's genuine matches.
        for rank, it in enumerate(new_items):
            it.score = main_min - 0.01 * (rank + 1)

    merged = list(main)
    for it in new_items:
        if it.meta is None:
            it.meta = {}
        it.meta["source_floor"] = True
        merged.append(it)
    return merged


def _embed_query_for_floor(query: str, indexing_config: dict) -> tuple[list[float], int]:
    """Embed a query the same way the chunk retrievers do, for floor back-fill."""
    from litellm import embedding as litellm_embedding

    embedding_model = indexing_config.get("embedding_model", CHUNK_EMBED_EMBEDDING_MODEL)
    resp = litellm_embedding(model=embedding_model, input=query)
    emb = resp.data[0]["embedding"]
    return emb, len(emb)


def _run_source_floor(
    store: BasePgVectorStore,
    query: str,
    indexing_config: dict,
    min_per_source: int,
    similarity_threshold: float,
    source_ids: list[str] | None,
    enriched_query: str | None,
) -> list[RetrievedItem]:
    """Fetch each matched source's top ``min_per_source`` chunks (sync).

    Called only from the chunk-based pipeline, whose stores all expose the
    ai.embeddings join — so a missing-table fault is not a real case here. Only
    the query-embedding call is best-effort: a transient embedding-provider error
    degrades to "no back-fill" rather than failing the search. Store/SQL errors
    are NOT swallowed — they propagate (vector_search_per_source already logs and
    re-raises them) so genuine faults surface instead of silently disabling the
    floor the user explicitly asked for.
    """
    try:
        emb, dims = _embed_query_for_floor(enriched_query or query, indexing_config)
    except Exception:
        logger.warning(
            "min_per_source floor: query embedding failed; skipping back-fill",
            exc_info=True,
        )
        return []
    return asyncio.run(
        store.vector_search_per_source(
            embedding=emb,
            per_source_k=min_per_source,
            source_cap=PER_SOURCE_FLOOR_SOURCE_CAP,
            similarity_threshold=similarity_threshold,
            dims=dims,
            source_ids=source_ids,
        )
    )


async def _arun_source_floor(
    store: BasePgVectorStore,
    query: str,
    indexing_config: dict,
    min_per_source: int,
    similarity_threshold: float,
    source_ids: list[str] | None,
    enriched_query: str | None,
) -> list[RetrievedItem]:
    """Async variant of _run_source_floor (same error policy)."""
    try:
        emb, dims = _embed_query_for_floor(enriched_query or query, indexing_config)
    except Exception:
        logger.warning(
            "min_per_source floor: query embedding failed; skipping back-fill",
            exc_info=True,
        )
        return []
    return await store.vector_search_per_source(
        embedding=emb,
        per_source_k=min_per_source,
        source_cap=PER_SOURCE_FLOOR_SOURCE_CAP,
        similarity_threshold=similarity_threshold,
        dims=dims,
        source_ids=source_ids,
    )


# Map raw retrieval method name -> billing action name.
# vector_search/hybrid keep the original key; "full_text" bills as "bm25_search"
# because the underlying store call is bm25s_search and the spec lists
# bm25_search as the billable action name.
_RETRIEVAL_BILLING_ACTION: dict[str, str] = {
    "vector_search": "vector_search",
    "full_text": "bm25_search",
    "hybrid": "hybrid_search",
    "tree_search": "tree_search",
}


def _bill_retrieval_action(action: str, request_id: str, *, estimated_cost: int = 1) -> None:
    """Pre-op balance check for a retrieval action.

    Caller invokes _post_retrieval_charge() on success. Goes through the
    billing port — a no-op when no cloud billing adapter is registered
    (unit tests, local dev, OSS build). Raises ServiceUnavailable (503) or
    PaymentRequired (402) for free-tier orgs without enough balance —
    caller propagates to the route which surfaces as HTTP 503/402.
    """
    billing.check_balance(estimated_cost=estimated_cost)


def _post_retrieval_charge(action: str, request_id: str, *, quantity: int = 1) -> None:
    """Post a charge after a retrieval sub-op succeeded.

    Goes through the billing port — a no-op when no cloud billing adapter is
    registered. The charge outcome is never raised as an exception — 402 and
    other terminal outcomes are reported on the returned ChargeOutcome and
    are treated as bounded loss per spec line 54.
    """
    billing.charge(
        action=action,
        quantity=quantity,
        ref_type="retrieval",
        ref_id=request_id,
        idempotency_parts=(request_id,),
    )


def _build_reranker_document(item: RetrievedItem) -> str:
    """Build document text for the reranker, prepending metadata if available."""
    parts: list[str] = []
    meta = item.meta or {}

    # Structural metadata
    doc_name = meta.get("doc_name")
    title = meta.get("title")
    if doc_name:
        parts.append(f"Document: {doc_name}")
    if title:
        parts.append(f"Section: {title}")

    # Enrichment metadata
    enrichment = meta.get("enrichment")
    if isinstance(enrichment, dict) and enrichment:
        annotations = ", ".join(f"{k}: {v}" for k, v in enrichment.items())
        parts.append(f"Metadata: {annotations}")

    if parts:
        return "\n".join(parts) + "\n\n" + item.text
    return item.text


def _apply_reranking(
    query: str,
    results: list[RetrievedItem],
    reranker_config: dict[str, Any],
    final_top_k: int,
) -> list[RetrievedItem]:
    """
    Apply reranking to a list of retrieved items.

    Instantiates the appropriate reranker based on model name, re-scores
    all results, and returns the top `final_top_k` items.

    On any failure, logs an error and falls back to the original results
    truncated to `final_top_k` (search never breaks due to reranker issues).

    Args:
        query: The search query (may be enriched/reformulated).
        results: Candidate items from initial retrieval.
        reranker_config: Dict with "model" and optional "api_key", "api_base".
        final_top_k: Number of results to return after reranking.

    Returns:
        Re-scored list of RetrievedItem, truncated to final_top_k.
    """
    if not results:
        return results

    model = reranker_config.get("model", "")
    if not model:
        return results[:final_top_k]

    try:
        # Route to the correct reranker backend
        if model.startswith("zerank"):
            from agentic.knowledge.reranker import ZeroEntropyReranker

            reranker = ZeroEntropyReranker(
                model=model,
                api_key=reranker_config.get("api_key"),
            )
        else:
            from agentic.knowledge.reranker import LiteLLMReranker

            reranker = LiteLLMReranker(
                model=model,
                api_key=reranker_config.get("api_key"),
                api_base=reranker_config.get("api_base"),
            )

        documents = [_build_reranker_document(item) for item in results]
        rerank_results = reranker.rerank(
            query=query,
            documents=documents,
            top_n=final_top_k,
        )

        # Map rerank results back to RetrievedItem objects
        _SCORE_KEY = {
            "hybrid": "hybrid_search_score",
            "full_text": "bm25_score",
        }
        reranked_items: list[RetrievedItem] = []
        for rr in rerank_results:
            original = results[rr.index]
            score_key = _SCORE_KEY.get(
                (original.meta or {}).get("retrieval_method", ""),
                "vector_similarity_score",
            )
            reranked_items.append(
                RetrievedItem(
                    item_id=original.item_id,
                    text=original.text,
                    score=rr.relevance_score,
                    source_id=original.source_id,
                    knowledge_base_id=original.knowledge_base_id,
                    meta={
                        **original.meta,
                        score_key: original.score,
                        "reranker_score": rr.relevance_score,
                        "reranker_config": {
                            "model": model,
                            "candidate_count": reranker_config.get(
                                "candidate_count", RERANKER_CANDIDATE_COUNT
                            ),
                            **(
                                {"api_base": reranker_config["api_base"]}
                                if reranker_config.get("api_base")
                                else {}
                            ),
                        },
                    },
                )
            )

        logger.info(
            f"Reranked {len(results)} candidates → {len(reranked_items)} results using {model}"
        )
        return reranked_items

    except Exception:
        logger.exception(
            f"Reranking failed with model '{model}', falling back to original ordering"
        )
        return results[:final_top_k]


def _run_vector_search(
    store: BasePgVectorStore,
    query: str,
    fetch_count: int,
    indexing_config: dict[str, Any],
    retrieval_config: dict[str, Any],
    filter_metadata: dict[str, Any] | None,
    enriched_query: str | None = None,
    item_ids: set[str] | None = None,
    **kwargs: Any,
) -> list[RetrievedItem]:
    """Run pure vector similarity search."""
    from litellm import embedding as litellm_embedding

    embed_text = enriched_query or query
    embedding_model = indexing_config.get("embedding_model", CHUNK_EMBED_EMBEDDING_MODEL)
    embedding_response = litellm_embedding(model=embedding_model, input=embed_text)
    query_embedding = embedding_response.data[0]["embedding"]

    source_ids = kwargs.get("source_ids")
    return asyncio.run(
        store.vector_search(
            embedding=query_embedding,
            dims=len(query_embedding),
            top_k=fetch_count,
            filter_metadata=filter_metadata,
            item_ids=item_ids,
            source_ids=source_ids,
        )
    )


def _run_full_text_search(
    store: BasePgVectorStore,
    query: str,
    fetch_count: int,
    indexing_config: dict[str, Any],
    retrieval_config: dict[str, Any],
    filter_metadata: dict[str, Any] | None,
    keyword_query: str | None = None,
    item_ids: set[str] | None = None,
    **kwargs: Any,
) -> list[RetrievedItem]:
    """Run BM25-scored full-text keyword search.

    Uses pre-built bm25s index when available, falls back to PostgreSQL
    tsvector search otherwise.
    """
    # Use keyword_query if provided (already includes chat context)
    search_text = keyword_query or query
    source_ids = kwargs.get("source_ids")
    return asyncio.run(
        store.bm25s_search(
            query=search_text,
            top_k=fetch_count,
            filter_metadata=filter_metadata,
            item_ids=item_ids,
            source_ids=source_ids,
        )
    )


def _run_hybrid_search(
    store: BasePgVectorStore,
    query: str,
    fetch_count: int,
    indexing_config: dict[str, Any],
    retrieval_config: dict[str, Any],
    filter_metadata: dict[str, Any] | None,
    enriched_query: str | None = None,
    keyword_query: str | None = None,
    item_ids: set[str] | None = None,
    **kwargs: Any,
) -> list[RetrievedItem]:
    """Run hybrid search (vector + BM25 fused with RRF).

    Uses bm25s pre-indexed search for the keyword component when available.
    """
    from agentic.knowledge.retrieval.fusion import reciprocal_rank_fusion
    from litellm import embedding as litellm_embedding

    embed_text = enriched_query or query
    embedding_model = indexing_config.get("embedding_model", CHUNK_EMBED_EMBEDDING_MODEL)
    embedding_response = litellm_embedding(model=embedding_model, input=embed_text)
    query_embedding = embedding_response.data[0]["embedding"]

    vector_weight = retrieval_config.get("vector_weight", HYBRID_DEFAULT_VECTOR_WEIGHT)
    search_text = keyword_query or query
    source_ids = kwargs.get("source_ids")

    # Run vector search
    vector_results = asyncio.run(
        store.vector_search(
            embedding=query_embedding,
            dims=len(query_embedding),
            top_k=fetch_count,
            filter_metadata=filter_metadata,
            item_ids=item_ids,
            _resolve=False,
            source_ids=source_ids,
        )
    )

    # Run bm25s search (falls back to tsvector if no index)
    text_results = asyncio.run(
        store.bm25s_search(
            query=search_text,
            top_k=fetch_count,
            filter_metadata=filter_metadata,
            item_ids=item_ids,
            _resolve=False,
            source_ids=source_ids,
        )
    )

    # Fuse with Reciprocal Rank Fusion
    keyword_weight = 1.0 - vector_weight
    fused = reciprocal_rank_fusion(
        result_lists=[vector_results, text_results],
        weights=[vector_weight, keyword_weight],
        top_k=fetch_count,
    )
    return store._resolve_results(fused)


# Registry of chunk-based retrievers. Adding a new retriever =
# write a function with the same signature + add one entry here.
CHUNK_RETRIEVER_MAP: dict[str, Any] = {
    "vector_search": _run_vector_search,
    "full_text": _run_full_text_search,
    "hybrid": _run_hybrid_search,
}


# ---------------------------------------------------------------------------
# Async retriever variants — used from the async workflow engine where
# asyncio.run() would fail because an event loop is already running.
# ---------------------------------------------------------------------------


async def _arun_vector_search(
    store: BasePgVectorStore,
    query: str,
    fetch_count: int,
    indexing_config: dict[str, Any],
    retrieval_config: dict[str, Any],
    filter_metadata: dict[str, Any] | None,
    enriched_query: str | None = None,
    item_ids: set[str] | None = None,
    **kwargs: Any,
) -> list[RetrievedItem]:
    """Async variant of _run_vector_search — awaits store methods directly."""
    from litellm import embedding as litellm_embedding

    embed_text = enriched_query or query
    embedding_model = indexing_config.get("embedding_model", CHUNK_EMBED_EMBEDDING_MODEL)
    embedding_response = litellm_embedding(model=embedding_model, input=embed_text)
    query_embedding = embedding_response.data[0]["embedding"]

    source_ids = kwargs.get("source_ids")
    return await store.vector_search(
        embedding=query_embedding,
        dims=len(query_embedding),
        top_k=fetch_count,
        filter_metadata=filter_metadata,
        item_ids=item_ids,
        source_ids=source_ids,
    )


async def _arun_full_text_search(
    store: BasePgVectorStore,
    query: str,
    fetch_count: int,
    indexing_config: dict[str, Any],
    retrieval_config: dict[str, Any],
    filter_metadata: dict[str, Any] | None,
    keyword_query: str | None = None,
    item_ids: set[str] | None = None,
    **kwargs: Any,
) -> list[RetrievedItem]:
    """Async variant of _run_full_text_search — awaits store methods directly."""
    search_text = keyword_query or query
    source_ids = kwargs.get("source_ids")
    return await store.bm25s_search(
        query=search_text,
        top_k=fetch_count,
        filter_metadata=filter_metadata,
        item_ids=item_ids,
        source_ids=source_ids,
    )


async def _arun_hybrid_search(
    store: BasePgVectorStore,
    query: str,
    fetch_count: int,
    indexing_config: dict[str, Any],
    retrieval_config: dict[str, Any],
    filter_metadata: dict[str, Any] | None,
    enriched_query: str | None = None,
    keyword_query: str | None = None,
    item_ids: set[str] | None = None,
    **kwargs: Any,
) -> list[RetrievedItem]:
    """Async variant of _run_hybrid_search — awaits store methods directly."""
    from agentic.knowledge.retrieval.fusion import reciprocal_rank_fusion
    from litellm import embedding as litellm_embedding

    embed_text = enriched_query or query
    embedding_model = indexing_config.get("embedding_model", CHUNK_EMBED_EMBEDDING_MODEL)
    embedding_response = litellm_embedding(model=embedding_model, input=embed_text)
    query_embedding = embedding_response.data[0]["embedding"]

    vector_weight = retrieval_config.get("vector_weight", HYBRID_DEFAULT_VECTOR_WEIGHT)
    search_text = keyword_query or query
    source_ids = kwargs.get("source_ids")

    vector_results = await store.vector_search(
        embedding=query_embedding,
        dims=len(query_embedding),
        top_k=fetch_count,
        filter_metadata=filter_metadata,
        item_ids=item_ids,
        _resolve=False,
        source_ids=source_ids,
    )

    text_results = await store.bm25s_search(
        query=search_text,
        top_k=fetch_count,
        filter_metadata=filter_metadata,
        item_ids=item_ids,
        _resolve=False,
        source_ids=source_ids,
    )

    keyword_weight = 1.0 - vector_weight
    fused = reciprocal_rank_fusion(
        result_lists=[vector_results, text_results],
        weights=[vector_weight, keyword_weight],
        top_k=fetch_count,
    )
    return store._resolve_results(fused)


ASYNC_CHUNK_RETRIEVER_MAP: dict[str, Any] = {
    "vector_search": _arun_vector_search,
    "full_text": _arun_full_text_search,
    "hybrid": _arun_hybrid_search,
}


async def _aexecute_retrieval_pipeline(
    db_session: Session,
    store: BasePgVectorStore,
    query: str,
    top_k: int,
    retrieval_method: str,
    indexing_config: dict,
    retrieval_config: dict | None,
    filter_metadata: dict[str, Any] | None,
    knowledge_base_id: str,
    enriched_query: str | None = None,
    keyword_query: str | None = None,
    similarity_threshold: float = 0.0,
    source_ids: list[str] | None = None,
    request_id: str | None = None,
) -> list[RetrievedItem]:
    """Async variant of _execute_retrieval_pipeline."""
    rid = request_id or get_run_id() or new_request_id()
    reranker_config = retrieval_config.get("reranker") if retrieval_config else None
    use_reranker = bool(reranker_config and reranker_config.get("model"))

    min_per_source, max_per_source = _read_source_limits(retrieval_config)
    has_source_limits = min_per_source is not None or max_per_source is not None

    if use_reranker:
        fetch_count = reranker_config.get("candidate_count", RERANKER_CANDIDATE_COUNT)
    elif has_source_limits:
        # Over-fetch a candidate pool so apply_source_limits has surplus to
        # diversify over (without a reranker, fetch_count would otherwise == top_k
        # and the per-source cap/floor could not be enforced).
        fetch_count = min(top_k * PER_SOURCE_CANDIDATE_FACTOR, PER_SOURCE_CANDIDATE_CAP)
    else:
        fetch_count = top_k

    primary_action = _RETRIEVAL_BILLING_ACTION.get(retrieval_method, "vector_search")
    _bill_retrieval_action(primary_action, rid)

    retriever_fn = ASYNC_CHUNK_RETRIEVER_MAP.get(retrieval_method, _arun_vector_search)
    results = await retriever_fn(
        store=store,
        query=query,
        fetch_count=fetch_count,
        indexing_config=indexing_config,
        retrieval_config=retrieval_config,
        filter_metadata=filter_metadata,
        enriched_query=enriched_query,
        keyword_query=keyword_query,
        source_ids=source_ids,
    )

    _post_retrieval_charge(primary_action, rid)

    for r in results:
        if r.meta is None:
            r.meta = {}
        r.meta["retrieval_method"] = retrieval_method

    if retrieval_method == "vector_search" and similarity_threshold > 0:
        results = [r for r in results if r.score >= similarity_threshold]

    # min_per_source floor: back-fill each matched source's top chunks (see the
    # sync pipeline for rationale). Merge before rerank/enrichment.
    if min_per_source is not None:
        floor_items = await _arun_source_floor(
            store,
            query,
            indexing_config,
            min_per_source,
            similarity_threshold,
            source_ids,
            enriched_query,
        )
        if floor_items:
            results = _merge_floor_items(
                results,
                floor_items,
                # Floor scores are cosine; re-scale them under a hybrid/BM25 pool
                # so they don't outrank genuine matches when there's no reranker.
                rescore_below_main=(retrieval_method != "vector_search"),
            )

    if _attach_enrichment_metadata(db_session, results, knowledge_base_id):
        _bill_retrieval_action("metadata_enrichment", rid)
        _post_retrieval_charge("metadata_enrichment", rid)

    if use_reranker and results:
        _bill_retrieval_action("reranker_call", rid)
        # When per-source limits are active the reranker must only reorder, not
        # truncate — apply_source_limits performs the final cut below, and the
        # merged floor items must survive reranking.
        rerank_top_k = len(results) if has_source_limits else top_k
        results = _apply_reranking(
            query=enriched_query or query,
            results=results,
            reranker_config=reranker_config,
            final_top_k=rerank_top_k,
        )
        _post_retrieval_charge("reranker_call", rid)

    # Final cut to top_k, enforcing per-source diversity limits when configured.
    # With no limits this is a plain top_k truncation (legacy behaviour).
    results = apply_source_limits(
        results,
        top_k=top_k,
        min_per_source=min_per_source,
        max_per_source=max_per_source,
    )

    for r in results:
        meta = r.meta
        if meta and "pages" not in meta:
            start = meta.get("start_page")
            end = meta.get("end_page")
            if start is not None and end is not None:
                meta["pages"] = list(range(int(start), int(end) + 1))

    return results


async def search_knowledge_base_async(
    db_session: Session,
    knowledge_base_id: str,
    query: str,
    top_k: int = 5,
    retrieval_method: str | None = None,
    similarity_threshold: float = 0.0,
    filter_metadata: dict[str, Any] | None = None,
    indexing_config: dict[str, Any] | None = None,
    retrieval_config: dict[str, Any] | None = None,
    session_history: list[dict[str, Any]] | None = None,
    pre_enriched_query: str | None = None,
    pre_keyword_query: str | None = None,
    source_ids: list[str] | None = None,
    request_id: str | None = None,
) -> list[RetrievedItem]:
    """Async variant of search_knowledge_base for use inside a running event loop.

    Mirrors search_knowledge_base but uses async retriever functions that
    await store methods directly instead of wrapping them with asyncio.run().

    The optional ``request_id`` is a stable natural identifier for the
    originating retrieval call; it is used to derive billing idempotency
    keys so retries of the same logical call do not double-charge. When not
    supplied, a UUID4 is generated.
    """
    rid = request_id or get_run_id() or new_request_id()
    # Fetch KB config if not provided
    if indexing_config is None or retrieval_config is None:
        kb_result = db_session.execute(
            text(f"""
                SELECT id, name, indexing_config, retrieval_config
                FROM "{AI_SCHEMA}".knowledge_bases
                WHERE id = :id
            """),
            {"id": knowledge_base_id},
        )
        kb_row = kb_result.fetchone()
        if not kb_row:
            raise ValueError(f"Knowledge base not found: {knowledge_base_id}")

        if indexing_config is None:
            indexing_config = _coerce_kb_config(kb_row[2], "indexing_config", knowledge_base_id)
        if retrieval_config is None:
            retrieval_config = _coerce_kb_config(kb_row[3], "retrieval_config", knowledge_base_id)

    strategy = indexing_config.get("strategy", "chunk_embed")

    if not retrieval_method and retrieval_config:
        retrieval_method = retrieval_config.get("method")

    if not retrieval_method:
        retrieval_method = get_default_retrieval_method(strategy)

    if not validate_retriever(strategy, retrieval_method):
        raise ValueError(
            f"Retrieval method '{retrieval_method}' is not compatible with "
            f"indexing strategy '{strategy}'."
        )

    # Query enrichment
    enriched_query = query
    keyword_query = query

    if pre_enriched_query is not None and pre_keyword_query is not None:
        enriched_query = pre_enriched_query
        keyword_query = pre_keyword_query
    else:
        enrichment_config = retrieval_config.get("query_enrichment") if retrieval_config else None
        use_llm_enrichment = isinstance(enrichment_config, dict) and enrichment_config.get(
            "enabled"
        )
        enrichment_model = (
            enrichment_config.get("model") if isinstance(enrichment_config, dict) else None
        )
        enrichment_reasoning_effort = (
            enrichment_config.get("reasoning_effort")
            if isinstance(enrichment_config, dict)
            else None
        )

        if retrieval_method in {"full_text", "hybrid"}:
            from .sparse_retrieval.query_context import build_search_query

            result = build_search_query(
                query=query,
                session_history=session_history,
                use_llm_enrichment=use_llm_enrichment,
                enrichment_model=enrichment_model,
                enrichment_reasoning_effort=enrichment_reasoning_effort,
            )
            enriched_query = result["enriched_query"]
            keyword_query = result["sparse_query"]
        elif use_llm_enrichment:
            from .query_enrichment import enrich_query

            result = enrich_query(
                query=query,
                retrieval_method=retrieval_method,
                session_history=session_history,
                model=enrichment_model,
                request_id=rid,
                reasoning_effort=enrichment_reasoning_effort,
            )
            enriched_query = result["enriched_query"]
            keyword_query = result["keyword_query"]

    # tree_search uses asyncio internally — await directly
    if retrieval_method == "tree_search":
        pi_store = PageIndexStore(
            db_session=db_session,
            knowledge_base_id=knowledge_base_id,
        )
        toc_records = pi_store.get_tocs()

        if source_ids:
            source_ids_set = set(source_ids)
            toc_records = [t for t in toc_records if t["source_id"] in source_ids_set]

        if not toc_records:
            raise EmptyKnowledgeBaseError(
                "No indexed documents in this knowledge base. Add and index sources first."
            )

        min_per_source, max_per_source = _read_source_limits(retrieval_config)
        has_source_limits = min_per_source is not None or max_per_source is not None
        # Over-fetch nodes when per-source limits are active so the final
        # selection has a surplus to diversify over.
        node_top_k = (
            min(top_k * PER_SOURCE_CANDIDATE_FACTOR, PER_SOURCE_CANDIDATE_CAP)
            if has_source_limits
            else top_k
        )

        retrieval_model = retrieval_config.get("retrieval_model", PAGEINDEX_RETRIEVAL_MODEL)
        algo_config = {
            "retrieval_model": retrieval_model,
            "retrieval_reasoning_effort": retrieval_config.get("retrieval_reasoning_effort"),
            "top_k": node_top_k,
        }
        algorithm = TreeSearchAlgorithm()

        _bill_retrieval_action("tree_search", rid)

        all_selected_nodes = await _run_tree_search_phases(
            algorithm=algorithm,
            query=query,
            toc_records=toc_records,
            algo_config=algo_config,
        )

        _post_retrieval_charge("tree_search", rid)

        if not all_selected_nodes:
            return []

        selections = [(node.toc_id, node.node_id) for node in all_selected_nodes]
        node_map = pi_store.get_nodes_by_ids(selections)

        results: list[RetrievedItem] = []
        for node in all_selected_nodes:
            key = (node.toc_id, node.node_id)
            node_row = node_map.get(key)
            if not node_row:
                continue

            score = max(0.0, 1.0 - (node.doc_rank * 0.02) - (node.rank * 0.05))
            node_meta = node_row.get("meta") or {}
            start_page = node_meta.get("start_page")
            end_page = node_meta.get("end_page")
            pages = (
                list(range(int(start_page), int(end_page) + 1))
                if start_page is not None and end_page is not None
                else []
            )
            results.append(
                RetrievedItem(
                    item_id=node_row["id"],
                    text=node_row["text"],
                    score=score,
                    source_id=node.source_id,
                    knowledge_base_id=node.knowledge_base_id,
                    meta={
                        "node_id": node.node_id,
                        "doc_name": node.doc_name or "",
                        "doc_description": node.doc_description or "",
                        "title": node.title,
                        "retrieval_method": "tree_search",
                        "score_type": "rank_position",
                        "doc_rank": node.doc_rank,
                        "pages": pages,
                    },
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        results = apply_source_limits(
            results,
            top_k=top_k,
            min_per_source=min_per_source,
            max_per_source=max_per_source,
        )
        if _attach_enrichment_metadata(db_session, results, knowledge_base_id):
            _bill_retrieval_action("metadata_enrichment", rid)
            _post_retrieval_charge("metadata_enrichment", rid)
        return results

    # graph_index
    if strategy == "graph_index":
        gi_node_count = (
            db_session.execute(
                text(
                    f'SELECT COUNT(*) FROM "{AI_SCHEMA}".graph_index_nodes WHERE knowledge_base_id = :kb_id'
                ),
                {"kb_id": knowledge_base_id},
            ).scalar()
            or 0
        )
        if gi_node_count == 0:
            raise EmptyKnowledgeBaseError(
                "No nodes indexed in this knowledge base. Add and index sources first."
            )

        gi_search_store = GraphIndexNodeStore(
            db_session=db_session, knowledge_base_id=knowledge_base_id
        )

        results = await _aexecute_retrieval_pipeline(
            db_session=db_session,
            store=gi_search_store,
            query=query,
            top_k=top_k,
            retrieval_method=retrieval_method,
            indexing_config=indexing_config,
            retrieval_config=retrieval_config,
            filter_metadata=filter_metadata,
            knowledge_base_id=knowledge_base_id,
            enriched_query=enriched_query,
            keyword_query=keyword_query,
            similarity_threshold=0.0,
            source_ids=source_ids,
            request_id=rid,
        )

        results = _expand_graph_neighbors(db_session, results, knowledge_base_id, retrieval_config)

        if source_ids:
            source_ids_set = set(source_ids)
            results = [r for r in results if r.source_id in source_ids_set]

        return results

    # Strategy-aware store selection
    if strategy == "full_document":
        ft_count = (
            db_session.execute(
                text(
                    f'SELECT COUNT(*) FROM "{AI_SCHEMA}".full_documents WHERE knowledge_base_id = :kb_id'
                ),
                {"kb_id": knowledge_base_id},
            ).scalar()
            or 0
        )
        if ft_count == 0:
            raise EmptyKnowledgeBaseError(
                "No documents indexed in this knowledge base. Add and index sources first."
            )
        store = FullDocumentStore(
            db_session=db_session,
            knowledge_base_id=knowledge_base_id,
            storage=get_storage(),
        )
    elif strategy == "doc2json":
        d2j_count = (
            db_session.execute(
                text(
                    f'SELECT COUNT(*) FROM "{AI_SCHEMA}".doc2json_documents WHERE knowledge_base_id = :kb_id'
                ),
                {"kb_id": knowledge_base_id},
            ).scalar()
            or 0
        )
        if d2j_count == 0:
            raise EmptyKnowledgeBaseError(
                "No documents indexed in this knowledge base. Add and index sources first."
            )
        store = Doc2JSONStore(
            db_session=db_session,
            knowledge_base_id=knowledge_base_id,
        )
    else:  # chunk_embed (default)
        chunk_count = (
            db_session.execute(
                text(f'SELECT COUNT(*) FROM "{AI_SCHEMA}".chunks WHERE knowledge_base_id = :kb_id'),
                {"kb_id": knowledge_base_id},
            ).scalar()
            or 0
        )
        if chunk_count == 0:
            raise EmptyKnowledgeBaseError(
                "No chunks indexed in this knowledge base. Add and index sources first."
            )
        store = PgVectorKnowledgeStore(db_session=db_session, knowledge_base_id=knowledge_base_id)

    return await _aexecute_retrieval_pipeline(
        db_session=db_session,
        store=store,
        query=query,
        top_k=top_k,
        retrieval_method=retrieval_method,
        indexing_config=indexing_config,
        retrieval_config=retrieval_config,
        filter_metadata=filter_metadata,
        knowledge_base_id=knowledge_base_id,
        enriched_query=enriched_query,
        keyword_query=keyword_query,
        similarity_threshold=similarity_threshold,
        source_ids=source_ids,
        request_id=rid,
    )


async def _run_tree_search_phases(
    algorithm: TreeSearchAlgorithm,
    query: str,
    toc_records: list[dict],
    algo_config: dict,
) -> list:
    """Run the two-phase tree search (document selection + node selection)."""
    # Stage 1: Document selection (skip if single document)
    if len(toc_records) > 1:
        relevant_docs = await algorithm.select_documents(
            query=query,
            toc_records=toc_records,
            config=algo_config,
        )
        logger.info(
            f"Document selection: {len(relevant_docs)} of {len(toc_records)} documents selected"
        )
    else:
        relevant_docs = [(0, toc_records[0])]

    # Stage 2: Per-document node selection
    all_selected_nodes = []
    for doc_rank, (doc_idx, toc_record) in enumerate(relevant_docs):
        nodes = await algorithm.select_nodes(
            query=query,
            toc_records=[toc_record],
            config=algo_config,
            doc_rank=doc_rank,
        )
        logger.info(
            f"Node selection for doc {doc_idx} "
            f"('{toc_record.get('doc_name', '?')}'): "
            f"{len(nodes)} nodes selected"
        )
        all_selected_nodes.extend(nodes)

    return all_selected_nodes


def _attach_enrichment_metadata(
    db_session: Session,
    results: list[RetrievedItem],
    knowledge_base_id: str,
) -> bool:
    """Look up enrichment metadata and set meta["enrichment"] on each item.

    No-op when no enrichment config exists or enrichment is incomplete.

    Returns True when enrichment data was actually attached (i.e., the
    enrichment lookup did real work). Used by the retrieval pipeline to
    decide whether to bill the metadata_enrichment action.
    """
    if not results:
        return False

    from .enrichment_filter import get_enrichment_config, get_enrichment_metadata_for_items

    enrich_cfg = get_enrichment_config(db_session, knowledge_base_id)
    if not enrich_cfg or enrich_cfg.get("status") not in (
        "completed",
        "completed_with_errors",
    ):
        return False
    if not enrich_cfg.get("fields"):
        return False

    meta_by_id = get_enrichment_metadata_for_items(
        db_session, enrich_cfg, [r.item_id for r in results]
    )
    if not meta_by_id:
        return False

    attached = False
    for item in results:
        enrichment = meta_by_id.get(item.item_id)
        if enrichment:
            if item.meta is None:
                item.meta = {}
            item.meta["enrichment"] = enrichment
            attached = True
    return attached


def _execute_retrieval_pipeline(
    db_session: Session,
    store: BasePgVectorStore,
    query: str,
    top_k: int,
    retrieval_method: str,
    indexing_config: dict,
    retrieval_config: dict | None,
    filter_metadata: dict[str, Any] | None,
    knowledge_base_id: str,
    enriched_query: str | None = None,
    keyword_query: str | None = None,
    similarity_threshold: float = 0.0,
    source_ids: list[str] | None = None,
    request_id: str | None = None,
) -> list[RetrievedItem]:
    """Shared retrieval pipeline: fetch -> stamp -> threshold -> enrich -> rerank.

    Billing: each chunk-retrieval method (vector_search, bm25_search,
    hybrid_search) is billed once per pipeline invocation; reranker_call and
    metadata_enrichment are billed when those steps actually run. The
    request_id (generated by the caller and passed in) is the natural key
    used to derive idempotency keys for every sub-charge.
    """
    rid = request_id or get_run_id() or new_request_id()
    reranker_config = retrieval_config.get("reranker") if retrieval_config else None
    use_reranker = bool(reranker_config and reranker_config.get("model"))

    min_per_source, max_per_source = _read_source_limits(retrieval_config)
    has_source_limits = min_per_source is not None or max_per_source is not None

    if use_reranker:
        fetch_count = reranker_config.get("candidate_count", RERANKER_CANDIDATE_COUNT)
    elif has_source_limits:
        # Over-fetch a candidate pool so apply_source_limits has surplus to
        # diversify over (without a reranker, fetch_count would otherwise == top_k
        # and the per-source cap/floor could not be enforced).
        fetch_count = min(top_k * PER_SOURCE_CANDIDATE_FACTOR, PER_SOURCE_CANDIDATE_CAP)
    else:
        fetch_count = top_k

    # Pre-op balance check for the primary retrieval action (free-tier hard cap).
    primary_action = _RETRIEVAL_BILLING_ACTION.get(retrieval_method, "vector_search")
    _bill_retrieval_action(primary_action, rid)

    retriever_fn = CHUNK_RETRIEVER_MAP.get(retrieval_method, _run_vector_search)
    results = retriever_fn(
        store=store,
        query=query,
        fetch_count=fetch_count,
        indexing_config=indexing_config,
        retrieval_config=retrieval_config,
        filter_metadata=filter_metadata,
        enriched_query=enriched_query,
        keyword_query=keyword_query,
        source_ids=source_ids,
    )

    # Charge the primary retrieval action on success.
    _post_retrieval_charge(primary_action, rid)

    # Stamp the resolved method so callers can read it from meta
    for r in results:
        if r.meta is None:
            r.meta = {}
        r.meta["retrieval_method"] = retrieval_method

    # Apply similarity threshold only for pure vector search (cosine 0-1).
    # BM25 and RRF scores use different scales where a cosine threshold
    # would incorrectly filter out all results.
    if retrieval_method == "vector_search" and similarity_threshold > 0:
        results = [r for r in results if r.score >= similarity_threshold]

    # min_per_source floor: a global top-N pool is dominated by large sources,
    # so explicitly back-fill each matched source's top chunks. Merge before
    # rerank/enrichment so floor chunks are scored and enriched uniformly.
    if min_per_source is not None:
        floor_items = _run_source_floor(
            store,
            query,
            indexing_config,
            min_per_source,
            similarity_threshold,
            source_ids,
            enriched_query,
        )
        if floor_items:
            results = _merge_floor_items(
                results,
                floor_items,
                # Floor scores are cosine; re-scale them under a hybrid/BM25 pool
                # so they don't outrank genuine matches when there's no reranker.
                rescore_below_main=(retrieval_method != "vector_search"),
            )

    # Attach enrichment metadata before reranking so the reranker can
    # incorporate it into its relevance scoring. Bill only when enrichment
    # actually attached values (real work done).
    if _attach_enrichment_metadata(db_session, results, knowledge_base_id):
        _bill_retrieval_action("metadata_enrichment", rid)
        _post_retrieval_charge("metadata_enrichment", rid)

    if use_reranker and results:
        _bill_retrieval_action("reranker_call", rid)
        # When per-source limits are active the reranker must only reorder, not
        # truncate — apply_source_limits performs the final cut below, and the
        # merged floor items must survive reranking.
        rerank_top_k = len(results) if has_source_limits else top_k
        results = _apply_reranking(
            query=enriched_query or query,
            results=results,
            reranker_config=reranker_config,
            final_top_k=rerank_top_k,
        )
        _post_retrieval_charge("reranker_call", rid)

    # Final cut to top_k, enforcing per-source diversity limits when configured.
    # With no limits this is a plain top_k truncation (legacy behaviour).
    results = apply_source_limits(
        results,
        top_k=top_k,
        min_per_source=min_per_source,
        max_per_source=max_per_source,
    )

    # Normalize pages list from start_page/end_page so downstream dedup
    # logic can always rely on meta["pages"].
    for r in results:
        meta = r.meta
        if meta and "pages" not in meta:
            start = meta.get("start_page")
            end = meta.get("end_page")
            if start is not None and end is not None:
                meta["pages"] = list(range(int(start), int(end) + 1))

    return results


GRAPH_EXPANSION_KEYS = frozenset(
    {
        "include_children",
        "max_children_per_parent",
        "max_referenced_nodes",
        "include_doc_toc",
    }
)


class GraphExpansionConfig(NamedTuple):
    """The bounded ``graph_expansion`` block, as expansion actually applies it."""

    include_children: bool
    max_children: int
    max_referenced: int
    include_doc_toc: bool


def _coerce_kb_config(raw: Any, field: str, kb_id: str) -> dict:
    """Return a KB config column as a dict, whatever is actually stored.

    The route now rejects a non-object, but rows written before that landed
    persist as valid JSONB of the wrong shape — a JSON string, say. Every
    later ``.get()`` on one raises, and context_handler turns that into an
    empty knowledge base, so one bad write silently zeroes a KB forever.
    """
    if isinstance(raw, dict):
        return raw
    if raw is not None:
        logger.warning(
            "%s on kb=%s is %s, not an object (%r) — using {}",
            field,
            kb_id,
            type(raw).__name__,
            raw,
        )
    return {}


def _read_graph_expansion_int(
    cfg: dict, key: str, default: int, ceiling: int, kb_id: str
) -> int:
    """Read one bounded integer, clamping into ``[0, ceiling]``.

    Only a real ``int`` is accepted: ``bool`` is an ``int`` in Python, and a
    numeric string would disagree with how Studio reads the same value back.
    Both rejection and clamping are logged — silently clamping would let the
    summary line report an effective value that reads as though the config
    had been honoured.
    """
    raw = cfg.get(key, default)
    if not isinstance(raw, int) or isinstance(raw, bool):
        if key in cfg:
            logger.warning(
                "Ignoring non-integer graph_expansion.%s: %r (kb=%s)", key, raw, kb_id
            )
        return default

    clamped = min(max(0, raw), ceiling)
    if clamped != raw:
        logger.warning(
            "graph_expansion.%s %r out of range [0, %d] — clamped to %d (kb=%s)",
            key,
            raw,
            ceiling,
            clamped,
            kb_id,
        )
    return clamped


def _read_graph_expansion_bool(cfg: dict, key: str, default: bool, kb_id: str) -> bool:
    """Read one boolean, rejecting rather than coercing non-booleans.

    ``bool()`` would fail asymmetrically in the expensive direction: every
    non-empty string is truthy, so ``"false"`` would switch children *on*.
    The graph_index registry entry already stores string booleans elsewhere
    (``"if_add_node_summary": "yes"``), so a caller writing ``"no"`` here by
    analogy is a live possibility, and Studio reads these strictly too.
    """
    raw = cfg.get(key, default)
    if isinstance(raw, bool):
        return raw
    if key in cfg:
        logger.warning(
            "graph_expansion.%s must be a boolean, got %s (%r) — using %s (kb=%s)",
            key,
            type(raw).__name__,
            raw,
            default,
            kb_id,
        )
    return default


def _read_graph_expansion_config(
    retrieval_config: dict | None, knowledge_base_id: str
) -> "GraphExpansionConfig":
    """Read the ``graph_expansion`` block, with every field bounded.

    Children fan-out is opt-in: a referenced node's substructure is advertised
    by the document outline rather than paid for in full section bodies.

    ``retrieval_config`` is unvalidated JSONB, and this runs on every
    graph_index search that returns anything, ahead of the no-references
    early-out. A malformed block therefore has to degrade to defaults:
    raising here would be caught by the bare ``except`` in context_handler
    and reported to the agent as an empty knowledge base, discarding a
    search that had already succeeded.

    Every rejection says which KB it came from, because a misconfigured KB
    otherwise emits an identical, unattributable warning on every query.
    """
    raw_cfg = (retrieval_config or {}).get("graph_expansion")
    if raw_cfg is not None and not isinstance(raw_cfg, dict):
        logger.warning(
            "graph_expansion must be an object, got %s (%r) — using defaults (kb=%s)",
            type(raw_cfg).__name__,
            raw_cfg,
            knowledge_base_id,
        )
        raw_cfg = None
    cfg = raw_cfg or {}

    # A camelCase spelling is the natural mistake for a caller writing JSONB
    # from JS, and silently ignoring it looks identical to configuring nothing.
    unknown = set(cfg) - GRAPH_EXPANSION_KEYS
    if unknown:
        logger.warning(
            "graph_expansion has unknown keys %s — ignored (kb=%s). Known keys: %s",
            sorted(unknown),
            knowledge_base_id,
            sorted(GRAPH_EXPANSION_KEYS),
        )

    return GraphExpansionConfig(
        include_children=_read_graph_expansion_bool(
            cfg, "include_children", GRAPH_DEFAULT_INCLUDE_CHILDREN, knowledge_base_id
        ),
        max_children=_read_graph_expansion_int(
            cfg,
            "max_children_per_parent",
            GRAPH_DEFAULT_MAX_CHILDREN,
            GRAPH_MAX_CHILDREN_CEILING,
            knowledge_base_id,
        ),
        max_referenced=_read_graph_expansion_int(
            cfg,
            "max_referenced_nodes",
            GRAPH_DEFAULT_MAX_REFERENCED_NODES,
            GRAPH_MAX_REFERENCED_CEILING,
            knowledge_base_id,
        ),
        include_doc_toc=_read_graph_expansion_bool(
            cfg, "include_doc_toc", GRAPH_DEFAULT_INCLUDE_DOC_TOC, knowledge_base_id
        ),
    )


def _expansion_tier_scores(min_score: float) -> tuple[float, float, float]:
    """Score the expansion band beneath the pool: outline, node, child.

    The step is proportional because a fixed decrement has to assume a score
    scale, and this path has three: cosine on ``[0, 1]``, RRF (also ``[0, 1]``
    — ``reciprocal_rank_fusion`` normalizes so the top hit is exactly 1.0),
    and whatever a configured reranker emits, which for a cross-encoder is
    routinely negative — hence ``abs``.

    It is also floored, because a ``min_score`` of exactly 0.0 is reachable
    and a purely multiplicative step would collapse all three tiers onto it.
    ``bm25s.retrieve`` pads its top-k with zero-score entries when fewer than
    ``k`` documents contain any query term (``sparse_retrieval/bm25_index.py``),
    and an orthogonal embedding scores exactly 0.0. The tsvector path can reach
    it too: ``@@ websearch_to_tsquery`` gates which rows are admitted, not what
    they score — ``ts_rank`` only orders the SQL and the emitted score is a
    Python BM25 over the parsed tsvector — so a row admitted purely by a
    negated term has no positive lexeme left to score and comes back 0.0.

    Ordering holds within one knowledge base. ``search_multiple_knowledge_bases``
    merges pools by raw score, so an expansion item from a high-scoring KB can
    still outrank a genuine hit from a low-scoring one.
    """
    step = max(abs(min_score), 1e-6) * 0.01
    return min_score - step, min_score - 2 * step, min_score - 3 * step


def _is_structural_meta(meta: dict | None) -> bool:
    """``_is_structural_item`` for code holding a serialized item's meta dict."""
    return (meta or {}).get("retrieval_method") in STRUCTURAL_RETRIEVAL_METHODS


def _is_structural_item(item: RetrievedItem) -> bool:
    """True for items carrying document structure rather than source content.

    These have no pages by construction, and an item with no page info makes
    the image path fall back to fetching *every* image for its source
    (``context_handler``'s ``fetch_all_for``, which wins over precise page
    coverage). Graph references are intra-document, so an outline's source is
    always one already represented by real hits — leaving it in that scan
    would discard the page filter for all of them.
    """
    return _is_structural_meta(item.meta)


def _render_toc_outline(nodes: list[dict], total_nodes: int) -> str:
    """Render a document outline: one indented ``[node_id] Title`` per line.

    Same shape the indexing-time enricher builds (``graph_enricher``'s
    ``_build_toc_context``), minus its optional ``— summary`` suffix and plus
    a bound: a long document's full outline is thousands of tokens nobody
    asked for. ``total_nodes`` is the document's real section count, which
    the store knows even though it only returns the first page of rows.

    Titles are searchable on every retrieval path — ``tasks/indexing.py``
    embeds ``title`` into each node's vector, so a model that reads a title
    here and puts it in a follow-up query will match. (The sparse index also
    covers titles, but only when a BM25 index exists: the tsvector fallback
    scores ``SEARCH_TEXT_COL``, which is ``text`` alone for these nodes.)
    The bracketed ids are for a human or a downstream tool: no route or agent
    tool takes a ``node_id``.
    """
    lines = [
        f"{'  ' * int(node.get('depth') or 0)}[{node['node_id']}] {node.get('title') or ''}".rstrip()
        for node in nodes[:GRAPH_TOC_MAX_NODES]
    ]
    remaining = total_nodes - len(lines)
    if remaining > 0:
        lines.append(f"... ({remaining} more sections)")
    return "\n".join(lines)


def _expand_graph_neighbors(
    db_session: Session,
    results: list[RetrievedItem],
    knowledge_base_id: str,
    retrieval_config: dict | None = None,
) -> list[RetrievedItem]:
    """Pull in first-degree referenced nodes for graph_index results.

    Appends, in descending score order beneath the main pool: one outline per
    document where a reference was actually followed (unless
    ``include_doc_toc`` is off), then the referenced nodes, then — only when
    ``include_children`` is set — a capped number of each referenced node's
    direct children.
    """
    if not results:
        return results

    cfg = _read_graph_expansion_config(retrieval_config, knowledge_base_id)

    # Every (toc_id, node_id) already in the pool, collected before any
    # reference is considered — a hit further down the list still counts as
    # "already present" for a reference made by the first one.
    existing_keys: set[tuple[str, str]] = {
        (meta["toc_id"], meta["node_id"])
        for meta in (item.meta or {} for item in results)
        if meta.get("toc_id") and meta.get("node_id")
    }

    # Candidate references, with what is needed to rank them: how many hits
    # point at each, and the best score among those hits.
    ref_hits: dict[tuple[str, str], int] = {}
    ref_best_score: dict[tuple[str, str], float] = {}

    for item in results:
        meta = item.meta or {}
        toc_id = meta.get("toc_id")
        if not toc_id:
            continue
        score = item.score if item.score is not None else 0.0
        # dict.fromkeys de-duplicates within one hit's list while keeping its
        # order: ref_hits counts how many *hits* point at a section, and
        # nothing guarantees a single hit lists a reference only once. Without
        # this, one hit naming a section twice outranks two hits that agree.
        for ref_nid in dict.fromkeys(meta.get("referenced_nodes") or []):
            key = (toc_id, ref_nid)
            if key in existing_keys:
                continue
            ref_hits[key] = ref_hits.get(key, 0) + 1
            ref_best_score[key] = max(ref_best_score.get(key, float("-inf")), score)

    if not ref_hits:
        logger.debug("graph_expansion: no new refs to expand")
        return results

    # Rank before capping: consensus first, because a section two hits both
    # point at is a better bet than one only the top hit mentions; then the
    # best referring hit's score; then node_id so the order is deterministic.
    ranked = sorted(
        ref_hits,
        key=lambda k: (-ref_hits[k], -ref_best_score[k], k),
    )
    selections = ranked[: cfg.max_referenced]
    if len(ranked) > len(selections):
        logger.info(
            "graph_expansion: %d referenced nodes exceed the cap of %d — "
            "keeping the most-referenced (kb=%s)",
            len(ranked),
            cfg.max_referenced,
            knowledge_base_id,
        )

    gi_store = GraphIndexStore(db_session=db_session, knowledge_base_id=knowledge_base_id)
    node_map = gi_store.get_nodes_by_ids(selections)

    pool_scores = [r.score for r in results if r.score is not None]
    min_score = min(pool_scores) if pool_scores else 0.0
    toc_score, parent_score, child_score = _expansion_tier_scores(min_score)
    all_keys = set(existing_keys)

    # One outline per document that actually contributed a referenced node.
    # Best-effort: the outline decorates results that already exist, so a
    # statement timeout here must not throw away a successful search — the
    # same treatment the source floor gets above.
    outlines: dict[str, dict] = {}
    if cfg.include_doc_toc and node_map:
        toc_ids = sorted({tid for tid, _ in node_map})
        try:
            # SAVEPOINT, not a bare try/except: a DBAPI failure aborts the
            # transaction, and swallowing it without unwinding leaves the
            # session deactivated — every later statement then raises
            # PendingRollbackError, surfacing far from here as an empty
            # knowledge base or a hard tool failure. db.py names the same
            # hazard with a coarser remedy (a full rollback). A nested
            # transaction undoes only this statement, so anything else the
            # caller has pending on a shared request session survives.
            with db_session.begin_nested():
                outlines = gi_store.get_toc_outline(toc_ids, GRAPH_TOC_MAX_NODES)
        except SQLAlchemyError:
            logger.error(
                "graph_expansion: outline fetch failed for kb=%s (%d tocs); "
                "results kept without outlines",
                knowledge_base_id,
                len(toc_ids),
                exc_info=True,
            )
            outlines = {}

    for toc_id, outline in outlines.items():
        outline_nodes = outline.get("nodes") or []
        if not outline_nodes:
            continue
        results.append(
            RetrievedItem(
                item_id=toc_id,
                text=_render_toc_outline(outline_nodes, outline.get("total_nodes") or 0),
                score=toc_score,
                source_id=outline.get("source_id"),
                knowledge_base_id=knowledge_base_id,
                meta={
                    "toc_id": toc_id,
                    "doc_name": outline.get("doc_name", ""),
                    "retrieval_method": "graph_toc",
                    "score_type": "graph_doc_outline",
                    "pages": [],
                },
            )
        )

    # Iterate `selections`, not `node_map`: the map is keyed by the rows the
    # store returned, in DB row order (`get_nodes_by_ids` ORs the pairs into
    # one WHERE with no ORDER BY), so walking it discards the ranking above.
    # Every referenced node carries the same score and `format_items_as_context`
    # truncates positionally, so emission order decides which references
    # survive a tight budget — the ranking has to reach the list, not just the
    # fetch.
    for key in selections:
        node_row = node_map.get(key)
        if node_row is None:
            continue
        toc_id, node_id = key
        all_keys.add(key)
        node_meta = node_row.get("meta") or {}
        start_page = node_meta.get("start_page")
        end_page = node_meta.get("end_page")
        pages = list(range(int(start_page), int(end_page) + 1)) if start_page and end_page else []

        results.append(
            RetrievedItem(
                item_id=node_row["id"],
                text=node_row["text"],
                score=parent_score,
                source_id=node_row.get("source_id"),
                knowledge_base_id=knowledge_base_id,
                meta={
                    "node_id": node_id,
                    "toc_id": toc_id,
                    "title": node_row.get("title", ""),
                    "doc_name": node_meta.get("doc_name", ""),
                    "retrieval_method": "graph_expansion",
                    "score_type": "graph_neighbor",
                    "pages": pages,
                    "referenced_nodes": node_meta.get("referenced_nodes", []),
                },
            )
        )

    # Expand children of referenced parent nodes — opt-in, and capped per
    # parent so one heavily-subdivided section can't flood the context.
    parents = [key for key in selections if key in node_map]
    children_map = (
        gi_store.get_children_by_parent_ids(parents)
        if cfg.include_children and cfg.max_children
        else {}
    )
    children_added = 0

    # Ranked parent order, for the same reason the parents themselves are
    # emitted in it: children inherit their parent's standing, so a positional
    # truncation should reach the least-agreed-on parent's subtree first.
    for parent_key in parents:
        child_rows = children_map.get(parent_key)
        if not child_rows:
            continue
        toc_id, parent_node_id = parent_key
        # The cap charges only newly-added children: one already in the pool
        # is skipped without consuming budget, so a parent can contribute up
        # to max_children *beyond* what the search already found.
        kept = 0
        # node_id is zero-padded and sequential (write_node_id zfills to 4 in
        # pre-order), so this is document order — and the children query has
        # no ORDER BY of its own. Past 9999 nodes the padding stops and this
        # sorts lexically, the same way the SQL would.
        for child_row in sorted(child_rows, key=lambda row: row["node_id"]):
            if kept >= cfg.max_children:
                break
            child_key = (toc_id, child_row["node_id"])
            if child_key in all_keys:
                continue
            all_keys.add(child_key)
            kept += 1

            child_meta = child_row.get("meta") or {}
            start_page = child_meta.get("start_page")
            end_page = child_meta.get("end_page")
            pages = (
                list(range(int(start_page), int(end_page) + 1)) if start_page and end_page else []
            )

            results.append(
                RetrievedItem(
                    item_id=child_row["id"],
                    text=child_row["text"],
                    score=child_score,
                    source_id=child_row.get("source_id"),
                    knowledge_base_id=knowledge_base_id,
                    meta={
                        "node_id": child_row["node_id"],
                        "toc_id": toc_id,
                        "title": child_row.get("title", ""),
                        "doc_name": child_meta.get("doc_name", ""),
                        "retrieval_method": "graph_expansion_child",
                        "score_type": "graph_neighbor_child",
                        "pages": pages,
                        "parent_node_id": parent_node_id,
                        "referenced_nodes": child_meta.get("referenced_nodes", []),
                    },
                )
            )
            children_added += 1

    logger.info(
        "graph_expansion: fetched %d neighbors + %d children + %d outlines "
        "from %d candidate refs (children=%s child_cap=%d ref_cap=%d outline=%s)",
        len(node_map),
        children_added,
        len(outlines),
        len(ranked),
        cfg.include_children,
        cfg.max_children,
        cfg.max_referenced,
        cfg.include_doc_toc,
    )

    return results


def search_knowledge_base(
    db_session: Session,
    knowledge_base_id: str,
    query: str,
    top_k: int = 5,
    retrieval_method: str | None = None,
    similarity_threshold: float = 0.0,
    filter_metadata: dict[str, Any] | None = None,
    indexing_config: dict[str, Any] | None = None,
    retrieval_config: dict[str, Any] | None = None,
    session_history: list[dict[str, Any]] | None = None,
    enrichment_output: dict[str, Any] | None = None,
    pre_enriched_query: str | None = None,
    pre_keyword_query: str | None = None,
    source_ids: list[str] | None = None,
    request_id: str | None = None,
) -> list[RetrievedItem]:
    """
    Search a knowledge base and return ranked chunks.

    This is the core search logic, decoupled from HTTP handling.
    Can be called internally by the agent run endpoint.

    Args:
        db_session: SQLAlchemy session for database operations
        knowledge_base_id: UUID of the knowledge base to search
        query: The search query text
        top_k: Maximum number of results to return (default: 5)
        retrieval_method: Search method (auto-detected from strategy if None)
        similarity_threshold: Minimum similarity score for vector results (0-1)
        filter_metadata: Optional metadata filters
        indexing_config: Optional indexing config override (if None, fetched from DB)
        retrieval_config: Optional retrieval config override (if None, fetched from DB)
        session_history: Optional conversation history for query enrichment context
        enrichment_output: Optional dict populated with enrichment info when auto-enabled
        pre_enriched_query: Optional pre-enriched query (skip enrichment if provided)
        pre_keyword_query: Optional pre-computed keyword query (skip enrichment if provided)
        source_ids: Optional list of source UUIDs to restrict retrieval to
        request_id: Optional stable natural identifier for this retrieval call;
            used to derive billing idempotency keys so a retry of the same
            logical request does not double-charge. Generated when omitted.

    Returns:
        List of RetrievedItem objects sorted by relevance

    Raises:
        ValueError: If knowledge base not found or has no indexed chunks
    """
    rid = request_id or get_run_id() or new_request_id()
    # Fetch KB config if not provided
    if indexing_config is None or retrieval_config is None:
        kb_result = db_session.execute(
            text(f"""
                SELECT id, name, indexing_config, retrieval_config
                FROM "{AI_SCHEMA}".knowledge_bases
                WHERE id = :id
            """),
            {"id": knowledge_base_id},
        )
        kb_row = kb_result.fetchone()
        if not kb_row:
            raise ValueError(f"Knowledge base not found: {knowledge_base_id}")

        if indexing_config is None:
            indexing_config = _coerce_kb_config(kb_row[2], "indexing_config", knowledge_base_id)
        if retrieval_config is None:
            retrieval_config = _coerce_kb_config(kb_row[3], "retrieval_config", knowledge_base_id)

    # Determine strategy and validate retrieval method
    strategy = indexing_config.get("strategy", "chunk_embed")

    if not retrieval_method and retrieval_config:
        retrieval_method = retrieval_config.get("method")

    if not retrieval_method:
        retrieval_method = get_default_retrieval_method(strategy)

    if not validate_retriever(strategy, retrieval_method):
        raise ValueError(
            f"Retrieval method '{retrieval_method}' is not compatible with "
            f"indexing strategy '{strategy}'."
        )

    # Query enrichment: use bm25s tokenization by default, LLM enrichment optional
    enriched_query = query
    keyword_query = query
    enrichment_info: dict[str, Any] | None = None

    if pre_enriched_query is not None and pre_keyword_query is not None:
        # Enrichment already performed upstream (hoisted to execute_retrieval)
        enriched_query = pre_enriched_query
        keyword_query = pre_keyword_query
    else:
        # Self-contained enrichment (direct KB search API, standalone callers)
        # Use fast tokenization-based context by default for full_text/hybrid
        # LLM enrichment is opt-in via retrieval_config
        enrichment_config = retrieval_config.get("query_enrichment") if retrieval_config else None

        # Determine if LLM enrichment is explicitly enabled
        use_llm_enrichment = isinstance(enrichment_config, dict) and enrichment_config.get(
            "enabled"
        )
        enrichment_model = (
            enrichment_config.get("model") if isinstance(enrichment_config, dict) else None
        )
        enrichment_reasoning_effort = (
            enrichment_config.get("reasoning_effort")
            if isinstance(enrichment_config, dict)
            else None
        )

        # For full_text/hybrid, always build context-aware queries
        if retrieval_method in {"full_text", "hybrid"}:
            from .sparse_retrieval.query_context import build_search_query

            result = build_search_query(
                query=query,
                session_history=session_history,
                use_llm_enrichment=use_llm_enrichment,
                enrichment_model=enrichment_model,
                enrichment_reasoning_effort=enrichment_reasoning_effort,
            )
            enriched_query = result["enriched_query"]
            keyword_query = result["sparse_query"]
            enrichment_info = {
                "original_query": query,
                "enriched_query": enriched_query,
                "keyword_query": keyword_query,
                "use_llm_enrichment": use_llm_enrichment,
            }
            if use_llm_enrichment:
                enrichment_info["model"] = enrichment_model or QUERY_ENRICHMENT_DEFAULT_MODEL
        elif use_llm_enrichment:
            # LLM enrichment explicitly enabled for vector_search
            from .query_enrichment import enrich_query

            result = enrich_query(
                query=query,
                retrieval_method=retrieval_method,
                session_history=session_history,
                model=enrichment_model,
                request_id=rid,
                reasoning_effort=enrichment_reasoning_effort,
            )
            enriched_query = result["enriched_query"]
            keyword_query = result["keyword_query"]
            enrichment_info = {
                "original_query": query,
                "enriched_query": enriched_query,
                "keyword_query": keyword_query,
                "model": enrichment_model or QUERY_ENRICHMENT_DEFAULT_MODEL,
                "use_llm_enrichment": True,
            }
            if result.get("error"):
                enrichment_info["error"] = result["error"]

        if enrichment_output is not None and enrichment_info is not None:
            enrichment_output.update(enrichment_info)

    results: list[RetrievedItem] = []

    # Route to tree_search for page_index strategy (two-phase)
    if retrieval_method == "tree_search":
        pi_store = PageIndexStore(
            db_session=db_session,
            knowledge_base_id=knowledge_base_id,
        )

        # Phase 1: Load lightweight ToC records (no section text)
        toc_records = pi_store.get_tocs()

        if source_ids:
            source_ids_set = set(source_ids)
            toc_records = [t for t in toc_records if t["source_id"] in source_ids_set]

        if not toc_records:
            raise EmptyKnowledgeBaseError(
                "No indexed documents in this knowledge base. Add and index sources first."
            )

        min_per_source, max_per_source = _read_source_limits(retrieval_config)
        has_source_limits = min_per_source is not None or max_per_source is not None
        # Over-fetch nodes when per-source limits are active so the final
        # selection has a surplus to diversify over.
        effective_top_k = (
            min(top_k * PER_SOURCE_CANDIDATE_FACTOR, PER_SOURCE_CANDIDATE_CAP)
            if has_source_limits
            else top_k
        )

        retrieval_model = retrieval_config.get("retrieval_model", PAGEINDEX_RETRIEVAL_MODEL)
        algo_config = {
            "retrieval_model": retrieval_model,
            "retrieval_reasoning_effort": retrieval_config.get("retrieval_reasoning_effort"),
            "top_k": effective_top_k,
        }
        algorithm = TreeSearchAlgorithm()

        _bill_retrieval_action("tree_search", rid)

        all_selected_nodes = asyncio.run(
            _run_tree_search_phases(
                algorithm=algorithm,
                query=query,
                toc_records=toc_records,
                algo_config=algo_config,
            )
        )

        _post_retrieval_charge("tree_search", rid)

        if not all_selected_nodes:
            return []

        # Phase 3: Fetch ONLY the selected section rows from the DB
        selections = [(node.toc_id, node.node_id) for node in all_selected_nodes]
        node_map = pi_store.get_nodes_by_ids(selections)

        # Phase 4: Build RetrievedItem objects with combined scoring
        for node in all_selected_nodes:
            key = (node.toc_id, node.node_id)
            node_row = node_map.get(key)
            if node_row:
                text_content = node_row["text"]
            else:
                logger.warning(f"Node not found for toc_id={node.toc_id}, node_id={node.node_id}")
                continue

            # Rank-based positional score (NOT a similarity/embedding score).
            # Penalizes lower-ranked documents (-0.02) and nodes (-0.05).
            score = max(0.0, 1.0 - (node.doc_rank * 0.02) - (node.rank * 0.05))
            node_meta = node_row.get("meta") or {}
            start_page = node_meta.get("start_page")
            end_page = node_meta.get("end_page")
            pages = (
                list(range(int(start_page), int(end_page) + 1))
                if start_page is not None and end_page is not None
                else []
            )
            results.append(
                RetrievedItem(
                    item_id=node_row["id"],
                    text=text_content,
                    score=score,
                    source_id=node.source_id,
                    knowledge_base_id=node.knowledge_base_id,
                    meta={
                        "node_id": node.node_id,
                        "doc_name": node.doc_name or "",
                        "doc_description": node.doc_description or "",
                        "title": node.title,
                        "retrieval_method": "tree_search",
                        "score_type": "rank_position",
                        "doc_rank": node.doc_rank,
                        "pages": pages,
                    },
                )
            )

        # Sort by score descending, then cut to top_k with per-source limits.
        results.sort(key=lambda x: x.score, reverse=True)
        results = apply_source_limits(
            results,
            top_k=top_k,
            min_per_source=min_per_source,
            max_per_source=max_per_source,
        )
        if _attach_enrichment_metadata(db_session, results, knowledge_base_id):
            _bill_retrieval_action("metadata_enrichment", rid)
            _post_retrieval_charge("metadata_enrichment", rid)
        return results

    # graph_index: vector/fulltext/hybrid search → rerank → graph expansion
    if strategy == "graph_index":
        gi_node_count = (
            db_session.execute(
                text(
                    f'SELECT COUNT(*) FROM "{AI_SCHEMA}".graph_index_nodes WHERE knowledge_base_id = :kb_id'
                ),
                {"kb_id": knowledge_base_id},
            ).scalar()
            or 0
        )
        if gi_node_count == 0:
            raise EmptyKnowledgeBaseError(
                "No nodes indexed in this knowledge base. Add and index sources first."
            )

        gi_search_store = GraphIndexNodeStore(
            db_session=db_session, knowledge_base_id=knowledge_base_id
        )
        logger.info(f"Using GraphIndexNodeStore ({gi_node_count} nodes) for KB {knowledge_base_id}")

        results = _execute_retrieval_pipeline(
            db_session=db_session,
            store=gi_search_store,
            query=query,
            top_k=top_k,
            retrieval_method=retrieval_method,
            indexing_config=indexing_config,
            retrieval_config=retrieval_config,
            filter_metadata=filter_metadata,
            knowledge_base_id=knowledge_base_id,
            enriched_query=enriched_query,
            keyword_query=keyword_query,
            similarity_threshold=0.0,
            source_ids=source_ids,
            request_id=rid,
        )

        # Graph expansion: pull in first-degree referenced nodes
        results = _expand_graph_neighbors(db_session, results, knowledge_base_id, retrieval_config)

        # Post-filter by source_ids — graph expansion may add nodes from
        # other sources that weren't in the original filter set.
        if source_ids:
            source_ids_set = set(source_ids)
            results = [r for r in results if r.source_id in source_ids_set]

        return results

    # Strategy-aware store selection
    if strategy == "full_document":
        ft_count = (
            db_session.execute(
                text(
                    f'SELECT COUNT(*) FROM "{AI_SCHEMA}".full_documents WHERE knowledge_base_id = :kb_id'
                ),
                {"kb_id": knowledge_base_id},
            ).scalar()
            or 0
        )
        if ft_count == 0:
            raise EmptyKnowledgeBaseError(
                "No documents indexed in this knowledge base. Add and index sources first."
            )
        store = FullDocumentStore(
            db_session=db_session,
            knowledge_base_id=knowledge_base_id,
            storage=get_storage(),
        )
        logger.info(f"Using FullDocumentStore ({ft_count} documents) for KB {knowledge_base_id}")
    elif strategy == "doc2json":
        d2j_count = (
            db_session.execute(
                text(
                    f'SELECT COUNT(*) FROM "{AI_SCHEMA}".doc2json_documents WHERE knowledge_base_id = :kb_id'
                ),
                {"kb_id": knowledge_base_id},
            ).scalar()
            or 0
        )
        if d2j_count == 0:
            raise EmptyKnowledgeBaseError(
                "No documents indexed in this knowledge base. Add and index sources first."
            )
        store = Doc2JSONStore(
            db_session=db_session,
            knowledge_base_id=knowledge_base_id,
        )
        logger.info(f"Using Doc2JSONStore ({d2j_count} documents) for KB {knowledge_base_id}")
    else:  # chunk_embed (default)
        chunk_count_result = db_session.execute(
            text(f"""
                SELECT COUNT(*) FROM "{AI_SCHEMA}".chunks
                WHERE knowledge_base_id = :kb_id
            """),
            {"kb_id": knowledge_base_id},
        )
        chunk_count = chunk_count_result.scalar() or 0
        if chunk_count == 0:
            raise EmptyKnowledgeBaseError(
                "No chunks indexed in this knowledge base. Add and index sources first."
            )
        store = PgVectorKnowledgeStore(db_session=db_session, knowledge_base_id=knowledge_base_id)
        logger.debug(
            f"Using PgVectorKnowledgeStore ({chunk_count} chunks) for KB {knowledge_base_id}"
        )

    return _execute_retrieval_pipeline(
        db_session=db_session,
        store=store,
        query=query,
        top_k=top_k,
        retrieval_method=retrieval_method,
        indexing_config=indexing_config,
        retrieval_config=retrieval_config,
        filter_metadata=filter_metadata,
        knowledge_base_id=knowledge_base_id,
        enriched_query=enriched_query,
        keyword_query=keyword_query,
        similarity_threshold=similarity_threshold,
        source_ids=source_ids,
        request_id=rid,
    )


def search_multiple_knowledge_bases(
    db_session: Session,
    knowledge_base_configs: list[dict[str, Any]],
    query: str,
) -> list[RetrievedItem]:
    """
    Search multiple knowledge bases and merge results.

    Args:
        db_session: SQLAlchemy session
        knowledge_base_configs: List of KB configs, each with:
            - id: UUID of the knowledge base
            - top_k: Optional limit for this KB (default: 5)
            - retrieval_method: Optional method override
            - similarity_threshold: Optional threshold override
            - source_ids: Optional list of source UUIDs to scope retrieval
        query: The search query text

    Returns:
        List of RetrievedItem objects from all KBs, sorted by score

    Example:
        >>> configs = [
        ...     {"id": "kb-1", "top_k": 3},
        ...     {"id": "kb-2", "top_k": 2},
        ... ]
        >>> items = search_multiple_knowledge_bases(db, configs, "query")
    """
    all_results: list[RetrievedItem] = []

    for kb_config in knowledge_base_configs:
        kb_id = kb_config.get("id")
        if not kb_id:
            logger.warning("Skipping KB config without id")
            continue

        try:
            results = search_knowledge_base(
                db_session=db_session,
                knowledge_base_id=kb_id,
                query=query,
                top_k=kb_config.get("top_k", 5),
                retrieval_method=kb_config.get("retrieval_method"),
                similarity_threshold=kb_config.get("similarity_threshold", 0.0),
                filter_metadata=kb_config.get("filter_metadata"),
                source_ids=kb_config.get("source_ids"),
            )
            all_results.extend(results)
        except ValueError as e:
            logger.warning(f"Skipping KB {kb_id}: {e}")
            continue

    # Sort all results by score (descending)
    all_results.sort(key=lambda x: x.score, reverse=True)

    return all_results


TOKENS_PER_IMAGE = 1000  # Conservative estimate for image token budget


def _format_enrichment_annotation(item: RetrievedItem) -> str:
    """Format metadata as compact annotation for LLM context. Empty string if none."""
    meta = item.meta or {}
    parts: list[str] = []

    # Structural metadata
    doc_name = meta.get("doc_name")
    title = meta.get("title")
    pages = meta.get("pages")
    if doc_name:
        parts.append(f"Document: {doc_name}")
    if title:
        parts.append(f"Section: {title}")
    if isinstance(pages, list) and pages:
        if len(pages) > 1:
            parts.append(f"Pages: {pages[0]}-{pages[-1]}")
        else:
            parts.append(f"Page: {pages[0]}")

    # Enrichment metadata
    enrichment = meta.get("enrichment")
    if isinstance(enrichment, dict) and enrichment:
        parts.extend(f"{k}: {v}" for k, v in enrichment.items())

    return " | ".join(parts)


def _format_chunk_annotation(item: RetrievedItem) -> str:
    """Format chunk-level annotation for grouped context (omits doc_name).

    Unlike _format_enrichment_annotation, this skips doc_name since the
    document header already displays it in grouped mode.
    """
    parts: list[str] = []
    meta = item.meta or {}

    if _is_structural_item(item):
        # Structure, not source text. Without this it reaches the model as a
        # bare citation marker with nothing saying it is an outline.
        return "Document outline (section titles only)"

    title = meta.get("title")
    pages = meta.get("pages")
    if title:
        parts.append(f"Section: {title}")
    if isinstance(pages, list) and pages:
        if len(pages) > 1:
            parts.append(f"Pages: {pages[0]}-{pages[-1]}")
        else:
            parts.append(f"Page: {pages[0]}")

    enrichment = meta.get("enrichment")
    if isinstance(enrichment, dict) and enrichment:
        parts.extend(f"{k}: {v}" for k, v in enrichment.items())

    return " | ".join(parts)


def _group_items_by_document(
    items: list[RetrievedItem],
) -> list[tuple[str, list[tuple[int, RetrievedItem]]]]:
    """Group items by source_id, ordered by best chunk score per group.

    Returns:
        List of (source_id, [(orig_index, item), ...]) tuples.
        Groups are sorted by highest item score descending.
        Items within each group keep their original index order.
    """
    from collections import OrderedDict

    groups: dict[str, list[tuple[int, RetrievedItem]]] = OrderedDict()
    best_score: dict[str, float] = {}

    for idx, item in enumerate(items):
        key = item.source_id or f"_unknown_{idx}"
        if key not in groups:
            groups[key] = []
            best_score[key] = item.score
        else:
            best_score[key] = max(best_score[key], item.score)
        groups[key].append((idx, item))

    sorted_keys = sorted(groups.keys(), key=lambda k: best_score[k], reverse=True)
    return [(k, groups[k]) for k in sorted_keys]


def _build_document_header(item: RetrievedItem) -> str:
    """Build a document-level header from the first item in a group."""
    meta = item.meta or {}
    doc_name = meta.get("doc_name") or meta.get("source_name") or item.source_id or "Unknown"
    header = f'--- Document: "{doc_name}" ---'

    description = meta.get("doc_description") or meta.get("doc_summary")
    if description:
        if len(description) > 200:
            description = description[:197] + "..."
        header += f"\nDescription: {description}"

    enrichment = meta.get("enrichment")
    if isinstance(enrichment, dict) and enrichment:
        header += "\n" + " | ".join(f"{k}: {v}" for k, v in enrichment.items())

    return header


def _pages_for_item(item: RetrievedItem) -> set[int]:
    """Extract the set of page numbers a retrieved item covers.

    Two indexing strategies populate page metadata under different keys:
      - chunk_embed: ``meta.pages`` -- list[int]
      - graph_index: ``meta.start_page`` + ``meta.end_page`` -- inclusive range

    Graph-index nodes do NOT populate ``meta.pages``. Reading only ``meta.pages``
    silently mis-categorizes them as having no page info, which historically
    caused them to fall through to the "all source images" branch and
    quietly amplified the per-query image load.

    Returns an empty set when no page info is available, signaling callers
    to fall back to the conservative "all images" behavior for that item.
    """
    meta = item.meta or {}
    pages = meta.get("pages")
    if isinstance(pages, list) and pages:
        out: set[int] = set()
        for p in pages:
            try:
                out.add(int(p))
            except (TypeError, ValueError):
                continue
        if out:
            return out
    sp = meta.get("start_page")
    ep = meta.get("end_page")
    if sp is not None and ep is not None:
        try:
            return set(range(int(sp), int(ep) + 1))
        except (TypeError, ValueError):
            pass
    if sp is not None:
        try:
            return {int(sp)}
        except (TypeError, ValueError):
            pass
    return set()


def format_items_as_context(
    items: list[RetrievedItem],
    max_tokens: int | None = None,
    include_source_info: bool = True,
    per_kb_context_mode: dict[str, str] | None = None,
    source_image_map: dict[str, list[dict]] | None = None,
    image_delivery: str = "base64",
    group_by_document: bool = True,
    citations_enabled: bool = False,
) -> tuple[str | list[dict], dict]:
    """
    Format retrieved items into context for the LLM.

    When all items use text mode, returns a plain string (backward-compatible).
    When any item uses image mode, returns a list of multimodal content blocks.

    Args:
        items: List of retrieved items
        max_tokens: Optional token limit (estimated at ~4 chars per token)
        include_source_info: Whether to include source metadata
        per_kb_context_mode: Map of kb_id -> "text" or "image"
        source_image_map: Map of source_id -> [{"page": N, "content": url_or_b64}, ...]
        image_delivery: "url" or "base64" (only used when context_mode="image")
        group_by_document: Group chunks under document headers (default True)

    Returns:
        Tuple of (formatted context (str or list[dict]), diagnostics dict)
    """
    if not items:
        return "", {"total_items": 0, "items_included": 0, "items_dropped": 0}

    # Determine if any KB uses image mode
    any_image_mode = False
    if per_kb_context_mode and source_image_map:
        any_image_mode = "image" in per_kb_context_mode.values()

    token_limit = max_tokens
    if citations_enabled and token_limit:
        # Reserve ~60 tokens for the citation instruction that will be
        # appended to the system prompt by the caller.
        from agentic_project_service.services.citations import build_citation_instruction

        citation_instruction_tokens = len(build_citation_instruction()) // 4
        token_limit = token_limit - citation_instruction_tokens
    estimated_tokens = 0
    included_indices: set[int] = set()
    # Track (source_id, page_number) pairs already included to skip
    # fully-overlapping items that would repeat the same page content.
    seen_pages: set[tuple[str | None, int]] = set()

    if any_image_mode:
        # Multimodal output path
        content_blocks: list[dict] = []
        image_refs: list[dict] = []

        if group_by_document:
            doc_groups = _group_items_by_document(items)
            budget_exhausted = False

            for _source_id, group_items in doc_groups:
                if budget_exhausted:
                    break

                # Emit document header
                first_item = group_items[0][1]
                doc_header = _build_document_header(first_item)
                header_tokens = len(doc_header) // 4
                if token_limit and estimated_tokens + header_tokens > token_limit:
                    budget_exhausted = True
                    break
                content_blocks.append({"type": "text", "text": doc_header})
                estimated_tokens += header_tokens

                for orig_idx, item in group_items:
                    kb_mode = (per_kb_context_mode or {}).get(item.knowledge_base_id, "text")

                    if (
                        kb_mode == "image"
                        and item.source_id in (source_image_map or {})
                        and not _is_structural_item(item)
                    ):
                        pages = _pages_for_item(item)
                        source_images = (source_image_map or {}).get(item.source_id, [])
                        matched_images = (
                            [img for img in source_images if img.get("page") in pages]
                            if pages
                            else sorted(source_images, key=lambda x: x.get("page", 0))
                        )

                        # Filter to pages not yet seen
                        new_images = [
                            img
                            for img in matched_images
                            if (item.source_id, img.get("page")) not in seen_pages
                        ]

                        if new_images:
                            chunk_ann = _format_chunk_annotation(item)
                            label = f"  [{orig_idx + 1}]"
                            if chunk_ann:
                                label += f" [{chunk_ann}]"
                            item_tokens = len(label) // 4 + len(new_images) * TOKENS_PER_IMAGE
                            if token_limit and estimated_tokens + item_tokens > token_limit:
                                budget_exhausted = True
                                break

                            content_blocks.append({"type": "text", "text": label})
                            for img in new_images:
                                img_content = img.get("content")
                                if not img_content:
                                    continue
                                img_storage_path = img.get("storage_path")
                                if image_delivery == "base64":
                                    fmt = img.get("format", "png").lower()
                                    mime = (
                                        f"image/{fmt}"
                                        if fmt not in ("jpg", "jpeg")
                                        else "image/jpeg"
                                    )
                                    block = {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{mime};base64,{img_content}"},
                                    }
                                else:
                                    block = {
                                        "type": "image_url",
                                        "image_url": {"url": img_content},
                                    }
                                content_blocks.append(block)
                                if img_storage_path:
                                    image_refs.append(
                                        {
                                            "block_index": len(content_blocks) - 1,
                                            "storage_path": img_storage_path,
                                            "format": img.get("format", "png"),
                                        }
                                    )
                                seen_pages.add((item.source_id, img.get("page")))

                            estimated_tokens += item_tokens
                            included_indices.add(orig_idx)
                        elif matched_images:
                            # All pages already shown — emit annotation only
                            chunk_ann = _format_chunk_annotation(item)
                            label = f"  [{orig_idx + 1}]"
                            if chunk_ann:
                                label += f" [{chunk_ann}]"
                            item_text = f"{label} (content already shown above)"
                            item_tokens = len(item_text) // 4
                            if token_limit and estimated_tokens + item_tokens > token_limit:
                                budget_exhausted = True
                                break
                            content_blocks.append({"type": "text", "text": item_text})
                            estimated_tokens += item_tokens
                            included_indices.add(orig_idx)
                        else:
                            # No matching page images — fall back to text content
                            chunk_ann = _format_chunk_annotation(item)
                            label = f"  [{orig_idx + 1}]"
                            if chunk_ann:
                                label += f" [{chunk_ann}]"
                            item_text = f"{label}\n  {item.text}"
                            item_tokens = len(item_text) // 4
                            if token_limit and estimated_tokens + item_tokens > token_limit:
                                budget_exhausted = True
                                break
                            content_blocks.append({"type": "text", "text": item_text})
                            for p in pages:
                                seen_pages.add((item.source_id, p))
                            estimated_tokens += item_tokens
                            included_indices.add(orig_idx)
                    else:
                        # Text mode item within image output
                        item_pages = (item.meta or {}).get("pages", [])
                        new_pages = [p for p in item_pages if (item.source_id, p) not in seen_pages]
                        chunk_ann = _format_chunk_annotation(item)
                        label = f"  [{orig_idx + 1}]"
                        if chunk_ann:
                            label += f" [{chunk_ann}]"

                        if item_pages and not new_pages:
                            # Fully overlapping — annotation only
                            item_text = f"{label} (content already shown above)"
                        else:
                            item_text = f"{label}\n  {item.text}"
                            for p in item_pages:
                                seen_pages.add((item.source_id, p))

                        item_tokens = len(item_text) // 4
                        if token_limit and estimated_tokens + item_tokens > token_limit:
                            budget_exhausted = True
                            break
                        content_blocks.append({"type": "text", "text": item_text})
                        estimated_tokens += item_tokens
                        included_indices.add(orig_idx)
        else:
            # Flat (ungrouped) multimodal path
            for i, item in enumerate(items):
                kb_mode = (per_kb_context_mode or {}).get(item.knowledge_base_id, "text")

                if (
                    kb_mode == "image"
                    and item.source_id in (source_image_map or {})
                    and not _is_structural_item(item)
                ):
                    pages = _pages_for_item(item)
                    source_images = (source_image_map or {}).get(item.source_id, [])
                    matched_images = (
                        [img for img in source_images if img.get("page") in pages]
                        if pages
                        else sorted(source_images, key=lambda x: x.get("page", 0))
                    )

                    # Filter to pages not yet seen
                    new_images = [
                        img
                        for img in matched_images
                        if (item.source_id, img.get("page")) not in seen_pages
                    ]

                    if new_images:
                        annotation = _format_enrichment_annotation(item)
                        label = f"[{i + 1}] (Source: {item.source_id})"
                        if annotation:
                            label += f" [{annotation}]"
                        item_tokens = len(label) // 4 + len(new_images) * TOKENS_PER_IMAGE
                        if token_limit and estimated_tokens + item_tokens > token_limit:
                            logger.warning(
                                f"[format_items_as_context] Dropping {len(items) - i} of "
                                f"{len(items)} items (hit {token_limit} token limit)"
                            )
                            break

                        content_blocks.append({"type": "text", "text": label})
                        for img in new_images:
                            img_content = img.get("content")
                            if not img_content:
                                continue
                            img_storage_path = img.get("storage_path")
                            if image_delivery == "base64":
                                fmt = img.get("format", "png").lower()
                                mime = (
                                    f"image/{fmt}" if fmt not in ("jpg", "jpeg") else "image/jpeg"
                                )
                                block = {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{img_content}"},
                                }
                            else:
                                block = {
                                    "type": "image_url",
                                    "image_url": {"url": img_content},
                                }
                            content_blocks.append(block)
                            if img_storage_path:
                                image_refs.append(
                                    {
                                        "block_index": len(content_blocks) - 1,
                                        "storage_path": img_storage_path,
                                        "format": img.get("format", "png"),
                                    }
                                )
                            seen_pages.add((item.source_id, img.get("page")))

                        estimated_tokens += item_tokens
                        included_indices.add(i)
                    elif matched_images:
                        # All pages already shown — annotation only
                        annotation = _format_enrichment_annotation(item)
                        label = f"[{i + 1}] (Source: {item.source_id})"
                        if annotation:
                            label += f" [{annotation}]"
                        item_text = f"{label} (content already shown above)"
                        item_tokens = len(item_text) // 4
                        if token_limit and estimated_tokens + item_tokens > token_limit:
                            break
                        content_blocks.append({"type": "text", "text": item_text})
                        estimated_tokens += item_tokens
                        included_indices.add(i)
                    else:
                        # No matching page images — fall back to text content
                        annotation = _format_enrichment_annotation(item)
                        label = f"[{i + 1}] (Source: {item.source_id})"
                        if annotation:
                            label += f" [{annotation}]"
                        item_text = f"{label}\n{item.text}"
                        item_tokens = len(item_text) // 4
                        if token_limit and estimated_tokens + item_tokens > token_limit:
                            logger.warning(
                                f"[format_items_as_context] Dropping {len(items) - i} of "
                                f"{len(items)} items (hit {token_limit} token limit)"
                            )
                            break
                        content_blocks.append({"type": "text", "text": item_text})
                        for p in pages:
                            seen_pages.add((item.source_id, p))
                        estimated_tokens += item_tokens
                        included_indices.add(i)
                else:
                    item_pages = (item.meta or {}).get("pages", [])
                    new_pages = [p for p in item_pages if (item.source_id, p) not in seen_pages]
                    annotation = _format_enrichment_annotation(item)

                    if item_pages and not new_pages:
                        # Fully overlapping — annotation only
                        if include_source_info and item.source_id:
                            header = f"[{i + 1}] (Source: {item.source_id})"
                            if annotation:
                                header += f" [{annotation}]"
                            item_text = f"{header} (content already shown above)"
                        elif annotation:
                            item_text = f"[{i + 1}] [{annotation}] (content already shown above)"
                        else:
                            item_text = f"[{i + 1}] (content already shown above)"
                    else:
                        if include_source_info and item.source_id:
                            header = f"[{i + 1}] (Source: {item.source_id})"
                            if annotation:
                                header += f" [{annotation}]"
                            item_text = f"{header}\n{item.text}"
                        elif annotation:
                            item_text = f"[{i + 1}] [{annotation}]\n{item.text}"
                        else:
                            item_text = f"[{i + 1}] {item.text}"
                        for p in item_pages:
                            seen_pages.add((item.source_id, p))

                    item_tokens = len(item_text) // 4
                    if token_limit and estimated_tokens + item_tokens > token_limit:
                        logger.warning(
                            f"[format_items_as_context] Dropping {len(items) - i} of "
                            f"{len(items)} items (hit {token_limit} token limit)"
                        )
                        break

                    content_blocks.append({"type": "text", "text": item_text})
                    estimated_tokens += item_tokens
                    included_indices.add(i)

        diagnostics = {
            "total_items": len(items),
            "items_included": len(included_indices),
            "items_dropped": len(items) - len(included_indices),
            "token_limit": token_limit,
            "estimated_tokens_used": estimated_tokens,
            "included_indices": sorted(included_indices),
            "context_mode": "image",
            "format_mode": "grouped" if group_by_document else "flat",
            "image_refs": image_refs,
        }
        return content_blocks, diagnostics

    # Text-only output path
    context_parts: list[str] = []

    if group_by_document:
        doc_groups = _group_items_by_document(items)
        budget_exhausted = False

        for _source_id, group_items in doc_groups:
            if budget_exhausted:
                break

            # Document header
            first_item = group_items[0][1]
            doc_header = _build_document_header(first_item)
            header_tokens = len(doc_header) // 4
            if token_limit and estimated_tokens + header_tokens > token_limit:
                remaining = sum(
                    len(gi)
                    for _, gi in doc_groups
                    if any(idx not in included_indices for idx, _ in gi)
                )
                logger.warning(
                    f"[format_items_as_context] Budget exhausted at document header, "
                    f"~{remaining} items remaining"
                )
                budget_exhausted = True
                break
            context_parts.append(doc_header)
            estimated_tokens += header_tokens

            for orig_idx, item in group_items:
                item_pages = (item.meta or {}).get("pages", [])
                new_pages = [p for p in item_pages if (item.source_id, p) not in seen_pages]
                chunk_ann = _format_chunk_annotation(item)

                if item_pages and not new_pages:
                    # Fully overlapping — annotation only
                    if chunk_ann:
                        item_text = (
                            f"  [{orig_idx + 1}] [{chunk_ann}] (content already shown above)"
                        )
                    else:
                        item_text = f"  [{orig_idx + 1}] (content already shown above)"
                else:
                    if chunk_ann:
                        item_text = f"  [{orig_idx + 1}] [{chunk_ann}]\n  {item.text}"
                    else:
                        item_text = f"  [{orig_idx + 1}]\n  {item.text}"
                    for p in item_pages:
                        seen_pages.add((item.source_id, p))

                item_tokens = len(item_text) // 4
                if token_limit and estimated_tokens + item_tokens > token_limit:
                    logger.warning(
                        f"[format_items_as_context] Dropping remaining items "
                        f"(hit {token_limit} token limit)"
                    )
                    budget_exhausted = True
                    break

                context_parts.append(item_text)
                estimated_tokens += item_tokens
                included_indices.add(orig_idx)
    else:
        # Flat (ungrouped) text path — original behavior
        for i, item in enumerate(items):
            item_pages = (item.meta or {}).get("pages", [])
            new_pages = [p for p in item_pages if (item.source_id, p) not in seen_pages]
            annotation = _format_enrichment_annotation(item)

            if item_pages and not new_pages:
                # Fully overlapping — annotation only
                if include_source_info and item.source_id:
                    header = f"[{i + 1}] (Source: {item.source_id})"
                    if annotation:
                        header += f" [{annotation}]"
                    item_text = f"{header} (content already shown above)"
                elif annotation:
                    item_text = f"[{i + 1}] [{annotation}] (content already shown above)"
                else:
                    item_text = f"[{i + 1}] (content already shown above)"
            else:
                if include_source_info and item.source_id:
                    header = f"[{i + 1}] (Source: {item.source_id})"
                    if annotation:
                        header += f" [{annotation}]"
                    item_text = f"{header}\n{item.text}"
                elif annotation:
                    item_text = f"[{i + 1}] [{annotation}]\n{item.text}"
                else:
                    item_text = f"[{i + 1}] {item.text}"
                for p in item_pages:
                    seen_pages.add((item.source_id, p))

            item_tokens = len(item_text) // 4
            if token_limit and estimated_tokens + item_tokens > token_limit:
                logger.warning(
                    f"[format_items_as_context] Dropping {len(items) - i} of "
                    f"{len(items)} items (hit {token_limit} token limit)"
                )
                break

            context_parts.append(item_text)
            estimated_tokens += item_tokens
            included_indices.add(i)

    diagnostics = {
        "total_items": len(items),
        "items_included": len(included_indices),
        "items_dropped": len(items) - len(included_indices),
        "token_limit": token_limit,
        "estimated_tokens_used": estimated_tokens,
        "included_indices": sorted(included_indices),
        "format_mode": "grouped" if group_by_document else "flat",
    }
    return "\n\n".join(context_parts), diagnostics


# Legacy class-based interface for backwards compatibility
class KnowledgeSearchService:
    """Service for searching knowledge bases."""

    def __init__(
        self,
        db_session: Session,
        knowledge_base_id: str,
        schema: str = AI_SCHEMA,
    ):
        self.session = db_session
        self.kb_id = knowledge_base_id
        self.store = PgVectorKnowledgeStore(
            db_session=db_session,
            knowledge_base_id=knowledge_base_id,
        )

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        method: str = "hybrid",
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter_metadata: dict | None = None,
    ) -> list[RetrievedItem]:
        """Search the knowledge base using the specified method."""
        dims = len(query_embedding) if query_embedding else None
        if method == "vector":
            results = await self.store.vector_search(
                embedding=query_embedding,
                dims=dims,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )
        elif method == "text":
            results = await self.store.full_text_search(
                query=query,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )
        else:  # hybrid
            results = await self.store.hybrid_search(
                query=query,
                embedding=query_embedding,
                dims=dims,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )

        if similarity_threshold > 0:
            results = [r for r in results if r.score >= similarity_threshold]

        return results
