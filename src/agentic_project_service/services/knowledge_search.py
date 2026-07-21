"""
Knowledge Search Service.

Provides a reusable service function for searching knowledge bases.
Used by both the KB search API endpoint and the agent run endpoint.
"""

import asyncio
import logging
from typing import Any

from sqlalchemy import text
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

from agentic.knowledge.retrieval import TreeSearchAlgorithm, apply_source_limits

logger = logging.getLogger(__name__)

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
    pool's minimum (preserving their relative order) — the same pool-relative
    approach as ``_expand_graph_neighbors``. For pure ``vector_search`` the pool
    is already cosine, so this is left off and true scores are preserved.
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
            indexing_config = kb_row[2] or {}
        if retrieval_config is None:
            retrieval_config = kb_row[3] or {}

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
            raise ValueError(
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
            raise ValueError(
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

        results = _expand_graph_neighbors(db_session, results, knowledge_base_id)

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
            raise ValueError(
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
            raise ValueError(
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
            raise ValueError(
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


def _expand_graph_neighbors(
    db_session: Session,
    results: list[RetrievedItem],
    knowledge_base_id: str,
) -> list[RetrievedItem]:
    """Pull in first-degree referenced nodes for graph_index results."""
    if not results:
        return results

    # Collect existing (toc_id, node_id) pairs for deduplication
    existing_keys: set[tuple[str, str]] = set()
    refs_to_fetch: dict[str, set[str]] = {}  # toc_id -> {node_ids}

    for item in results:
        meta = item.meta or {}
        toc_id = meta.get("toc_id")
        node_id = meta.get("node_id")
        if toc_id and node_id:
            existing_keys.add((toc_id, node_id))

        referenced = meta.get("referenced_nodes") or []
        for ref_nid in referenced:
            if toc_id and (toc_id, ref_nid) not in existing_keys:
                refs_to_fetch.setdefault(toc_id, set()).add(ref_nid)

    # Remove any refs that are already in results
    for toc_id, nids in list(refs_to_fetch.items()):
        nids -= {k[1] for k in existing_keys if k[0] == toc_id}
        if not nids:
            del refs_to_fetch[toc_id]

    if not refs_to_fetch:
        logger.debug("graph_expansion: no new refs to expand")
        return results

    # Fetch referenced nodes from DB
    gi_store = GraphIndexStore(db_session=db_session, knowledge_base_id=knowledge_base_id)
    selections = [(tid, nid) for tid, nids in refs_to_fetch.items() for nid in nids]
    node_map = gi_store.get_nodes_by_ids(selections)

    # Build RetrievedItems for referenced nodes
    min_score = min(r.score for r in results) if results else 0.0
    parent_score = max(0.0, min_score - 0.01)
    all_keys = set(existing_keys)

    for (toc_id, node_id), node_row in node_map.items():
        all_keys.add((toc_id, node_id))
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

    # Expand children of referenced parent nodes
    children_map = gi_store.get_children_by_parent_ids(list(node_map.keys()))
    child_score = max(0.0, parent_score - 0.01)
    children_added = 0

    for (toc_id, parent_node_id), child_rows in children_map.items():
        for child_row in child_rows:
            child_key = (toc_id, child_row["node_id"])
            if child_key in all_keys:
                continue
            all_keys.add(child_key)

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
        "graph_expansion: fetched %d neighbors + %d children from %d refs",
        len(node_map),
        children_added,
        sum(len(nids) for nids in refs_to_fetch.values()),
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
            indexing_config = kb_row[2] or {}
        if retrieval_config is None:
            retrieval_config = kb_row[3] or {}

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
            raise ValueError(
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
            raise ValueError(
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
        results = _expand_graph_neighbors(db_session, results, knowledge_base_id)

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
            raise ValueError(
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
            raise ValueError(
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
            raise ValueError(
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

                    if kb_mode == "image" and item.source_id in (source_image_map or {}):
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

                if kb_mode == "image" and item.source_id in (source_image_map or {}):
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
