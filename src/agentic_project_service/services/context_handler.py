"""
Context Handler Service.

Encapsulates retrieval coordination across knowledge bases.
Replaces the inline retrieval logic previously duplicated in routes/agents.py.
"""

import base64
import copy
import json
import logging
import uuid
import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import AI_SCHEMA
from .settings_registry import get_setting
from ..models.tenant import ContextHandlerStatus
from .knowledge_search import (
    _pages_for_item,
    search_knowledge_base,
    search_knowledge_base_async,
    format_items_as_context,
)
from .knowledge_store import RetrievedItem
from .storage import get_storage
from ..strategies import get_default_retrieval_method

logger = logging.getLogger(__name__)


def make_lightweight_retrieved_context(
    retrieved_context: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace full_document text with truncated preview + storage path pointer.

    Keeps DB rows small while preserving enough text for UI previews.
    """
    lightweight = []
    for item in retrieved_context:
        if not isinstance(item, dict):
            lightweight.append(item)
            continue
        full_text_path = (item.get("meta") or {}).get("full_text_path")
        if item.get("_type") == "full_document" and full_text_path:
            item_copy = {**item}
            original_text = item_copy.get("text", "")
            drop_limit = get_setting("DROPPED_ITEM_TEXT_LIMIT")
            if len(original_text) > drop_limit:
                item_copy["text"] = original_text[:drop_limit] + "\n\n[... full text in storage]"
            item_copy["full_text_path"] = full_text_path
            lightweight.append(item_copy)
        else:
            lightweight.append(item)
    # Strip base64 content and stale signed URLs from image arrays — images are
    # already in Supabase Storage and will be re-resolved to base64 on read.
    for i, item in enumerate(lightweight):
        if isinstance(item, dict) and item.get("images"):
            item_copy = {**item}
            item_copy["images"] = [
                {k: v for k, v in img.items() if k not in ("content", "url")}
                for img in item["images"]
            ]
            lightweight[i] = item_copy
    return lightweight


def _resolve_full_text_refs(
    retrieved_context: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Resolve full_text_path pointers back to full document text."""
    if not retrieved_context:
        return retrieved_context
    storage = None  # lazy-init: only fetch storage when needed
    resolved = []
    for item in retrieved_context:
        if isinstance(item, dict) and item.get("full_text_path"):
            if storage is None:
                storage = get_storage()
            item_copy = {**item}
            try:
                text_bytes = storage.download_from_path(item_copy["full_text_path"])
                item_copy["text"] = text_bytes.decode("utf-8")
            except Exception:
                logger.warning(
                    "Failed to resolve full_text_path %s",
                    item_copy["full_text_path"],
                )
            item_copy.pop("full_text_path", None)
            resolved.append(item_copy)
        else:
            resolved.append(item)
    return resolved


def _resolve_image_refs(
    retrieved_context: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Resolve image storage_path refs to inline base64 content."""
    if not retrieved_context:
        return retrieved_context
    storage = None  # lazy-init
    result = []
    for item in retrieved_context:
        if not isinstance(item, dict) or not item.get("images"):
            result.append(item)
            continue
        needs_resolution = any(
            "storage_path" in img and "content" not in img for img in item["images"]
        )
        if not needs_resolution:
            result.append(item)
            continue
        if storage is None:
            storage = get_storage()
        item_copy = {**item}
        resolved_images = []
        for img in item_copy["images"]:
            img_copy = {**img}
            if "storage_path" in img_copy and "content" not in img_copy:
                try:
                    image_bytes = storage.download_from_path(img_copy["storage_path"])
                    img_copy["content"] = base64.b64encode(image_bytes).decode("ascii")
                    img_copy.pop("url", None)  # clear stale signed URL
                    resolved_images.append(img_copy)
                except Exception:
                    logger.warning(
                        "Failed to resolve image ref %s",
                        img_copy.get("storage_path"),
                    )
                    resolved_images.append(img_copy)
            else:
                resolved_images.append(img_copy)
        item_copy["images"] = resolved_images
        result.append(item_copy)
    return result


def _resolve_page_images(
    db_session: Session,
    items: list[RetrievedItem],
    delivery_mode: str = "base64",
) -> dict[str, list[dict]]:
    """
    Resolve page images for retrieved items from image derivatives.

    For each source referenced by the items, fetches its image derivatives
    from the DB and either generates signed URLs or downloads + base64-encodes.

    Args:
        db_session: SQLAlchemy session
        items: Retrieved items that need image resolution
        delivery_mode: "url" (signed URLs) or "base64" (inline data)

    Returns:
        Mapping of source_id -> [{"page": N, "content": url_or_base64}, ...]
    """
    if not items:
        return {}

    # Collect unique source_ids
    source_ids = list({item.source_id for item in items if item.source_id})
    if not source_ids:
        return {}

    # Compute the set of pages each source actually needs from its retrieved
    # items. Two cases per source:
    #   - All items expose page metadata (chunk_embed `meta.pages` or graph_index
    #     `meta.start_page`/`meta.end_page`): only fetch those pages.
    #   - At least one item lacks page metadata: conservatively fetch every
    #     image record for that source (preserves the previous behavior for
    #     unforeseen item shapes).
    needed_pages: dict[str, set[int]] = {}
    fetch_all_for: set[str] = set()
    for item in items:
        sid = item.source_id
        if not sid:
            continue
        item_pages = _pages_for_item(item)
        if item_pages:
            needed_pages.setdefault(sid, set()).update(item_pages)
        else:
            fetch_all_for.add(sid)

    # Batch-fetch source derivatives
    placeholders = ", ".join(f":sid_{i}" for i in range(len(source_ids)))
    params = {f"sid_{i}": sid for i, sid in enumerate(source_ids)}
    rows = db_session.execute(
        text(f'SELECT id, derivatives FROM "{AI_SCHEMA}".sources WHERE id IN ({placeholders})'),
        params,
    ).fetchall()

    # Build source_id -> image derivative records, filtered to the pages each
    # source actually needs (unless we're falling back to "all" for that source).
    source_derivs: dict[str, list[dict]] = {}
    for row in rows:
        sid = str(row[0])
        derivs = row[1] or {}
        image_records = derivs.get("image", [])
        if not image_records:
            continue
        if sid in fetch_all_for or sid not in needed_pages:
            # Fall back to fetching every image for this source.
            source_derivs[sid] = image_records
        else:
            wanted = needed_pages[sid]
            source_derivs[sid] = [rec for rec in image_records if rec.get("page") in wanted]

    if not source_derivs:
        return {}

    # Resolve content (URL or base64) for each image
    try:
        storage = get_storage()
    except Exception as e:
        logger.warning(f"Failed to initialize storage for image resolution: {e}")
        return {}

    result: dict[str, list[dict]] = {}
    for sid, image_records in source_derivs.items():
        resolved = []
        for img_rec in image_records:
            storage_path = img_rec.get("storage_path")
            page = img_rec.get("page")
            fmt = img_rec.get("format", "png")
            if not storage_path or not page:
                logger.debug(
                    "Skipping image derivative for source %s: missing %s",
                    sid,
                    "storage_path" if not storage_path else "page number",
                )
                continue

            try:
                if delivery_mode == "url":
                    # Generate publicly-reachable signed URL (1 hour expiry)
                    parts = storage_path.split("/", 1)
                    if len(parts) == 2:
                        signed_url = storage.create_signed_url(
                            bucket_id=parts[0],
                            path=parts[1],
                            expires_in=3600,
                            public=True,
                        )
                        resolved.append(
                            {
                                "page": page,
                                "content": signed_url,
                                "format": fmt,
                                "storage_path": storage_path,
                            }
                        )
                    else:
                        logger.warning(
                            "Invalid storage_path %r for source %s page %s: "
                            "expected 'bucket/path' format",
                            storage_path,
                            sid,
                            page,
                        )
                else:
                    # base64 mode (default)
                    image_bytes = storage.download_from_path(storage_path)
                    b64_data = base64.b64encode(image_bytes).decode("ascii")
                    resolved.append(
                        {
                            "page": page,
                            "content": b64_data,
                            "format": fmt,
                            "storage_path": storage_path,
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to resolve image for source {sid} page {page}: {e}")

        if resolved:
            result[sid] = resolved

    return result


def _batch_fetch_source_metadata(
    db_session: Session,
    all_items: list[RetrievedItem],
    kb_display: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Batch-fetch source-level metadata for retrieved items.

    Returns a map of source_id -> {source_name, source_metadata, source_auto_metadata,
    doc_summary} used to populate document-level headers when formatting context.
    """
    source_ids = list({item.source_id for item in all_items if item.source_id})
    if not source_ids:
        return {}

    # Fetch source name + metadata
    placeholders = ", ".join(f":sid_{i}" for i in range(len(source_ids)))
    params = {f"sid_{i}": sid for i, sid in enumerate(source_ids)}
    rows = db_session.execute(
        text(
            f'SELECT id, name, metadata, auto_metadata FROM "{AI_SCHEMA}".sources '
            f"WHERE id IN ({placeholders})"
        ),
        params,
    ).fetchall()

    meta_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = str(row[0])
        meta_map[sid] = {
            "source_name": row[1] or "",
            "source_metadata": row[2] or {},
            "source_auto_metadata": row[3] or {},
            "doc_summary": None,
        }

    # For full_document KBs, also fetch summaries
    full_doc_kb_ids = [
        kb_id
        for kb_id, info in kb_display.items()
        if info.get("indexing_strategy") == "full_document"
    ]
    if full_doc_kb_ids and source_ids:
        kb_placeholders = ", ".join(f":fkb_{i}" for i in range(len(full_doc_kb_ids)))
        src_placeholders = ", ".join(f":fsid_{i}" for i in range(len(source_ids)))
        fd_params = {
            **{f"fkb_{i}": kid for i, kid in enumerate(full_doc_kb_ids)},
            **{f"fsid_{i}": sid for i, sid in enumerate(source_ids)},
        }
        fd_rows = db_session.execute(
            text(
                f'SELECT source_id, summary FROM "{AI_SCHEMA}".full_documents '
                f"WHERE source_id IN ({src_placeholders}) "
                f"AND knowledge_base_id IN ({kb_placeholders})"
            ),
            fd_params,
        ).fetchall()
        for row in fd_rows:
            sid = str(row[0])
            if sid in meta_map:
                meta_map[sid]["doc_summary"] = row[1]

    return meta_map


def _should_enrich_query(
    valid_kb_configs: list[dict[str, Any]],
    kb_retrieval_configs: dict[str, dict],
    kb_display: dict[str, dict[str, str]],
) -> tuple[bool, str | None]:
    """Determine whether query enrichment should run for any KB.

    Returns (should_enrich, model) where model is the first explicit
    query_enrichment model found across KBs, or None for the default.

    Tree-search KBs are skipped (they use their own LLM-based selection).
    """
    enrichment_model: str | None = None
    for kb_config in valid_kb_configs:
        kb_id = kb_config.get("id")
        retrieval_method = kb_config.get("retrieval_method")
        db_ret_cfg = kb_retrieval_configs.get(kb_id, {})
        if not retrieval_method:
            retrieval_method = db_ret_cfg.get("method")

        # Resolve default from indexing strategy when no explicit method set
        if not retrieval_method:
            strategy = kb_display.get(kb_id, {}).get("indexing_strategy", "chunk_embed")
            try:
                retrieval_method = get_default_retrieval_method(strategy)
            except ValueError:
                continue

        # tree_search uses its own LLM selection — skip
        if retrieval_method == "tree_search":
            continue

        # Check if query enrichment is explicitly enabled
        qe_cfg = db_ret_cfg.get("query_enrichment")
        if isinstance(qe_cfg, dict) and qe_cfg.get("enabled"):
            if qe_cfg.get("model") and not enrichment_model:
                enrichment_model = qe_cfg["model"]
            return True, enrichment_model

    return False, None


def _search_single_kb(
    engine,
    kb_config: dict[str, Any],
    query: str,
    kb_retrieval_configs: dict[str, dict],
    session_history: list[dict[str, Any]] | None,
    pre_enriched_query: str | None = None,
    pre_keyword_query: str | None = None,
) -> dict[str, Any]:
    """Search a single KB in a dedicated thread with its own DB session.

    Never raises — errors are captured in the return value so the caller
    can collect results from all threads without losing partial successes.
    """
    kb_id = kb_config.get("id")
    retrieval_method = kb_config.get("retrieval_method")
    thread_session = None
    try:
        thread_session = Session(bind=engine)
        results = search_knowledge_base(
            db_session=thread_session,
            knowledge_base_id=kb_id,
            query=query,
            top_k=kb_config.get("top_k")
            or kb_retrieval_configs.get(kb_id, {}).get("top_k", get_setting("KB_DEFAULT_TOP_K")),
            retrieval_method=retrieval_method,
            similarity_threshold=kb_config.get("similarity_threshold", 0.0),
            filter_metadata=kb_config.get("filter_metadata"),
            session_history=session_history,
            pre_enriched_query=pre_enriched_query,
            pre_keyword_query=pre_keyword_query,
            source_ids=kb_config.get("source_ids"),
        )
        # Determine the method actually used
        db_method = kb_retrieval_configs.get(kb_id, {}).get("method", "vector_search")
        if results:
            method = (results[0].meta or {}).get("retrieval_method", retrieval_method or db_method)
        else:
            method = retrieval_method or db_method
        return {
            "kb_id": kb_id,
            "results": results,
            "method": method,
            "error": None,
        }
    except Exception as e:
        logger.warning(f"KB retrieval error for {kb_id}: {e}")
        return {
            "kb_id": kb_id,
            "results": [],
            "method": None,
            "error": e,
        }
    finally:
        if thread_session is not None:
            thread_session.close()


def execute_retrieval(
    db_session: Session,
    query: str,
    knowledge_base_configs: list[dict[str, Any]],
    max_context_tokens: int | None = None,
    session_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Core retrieval orchestration over one or more knowledge bases.

    Searches each KB, merges results, formats context with token limiting,
    and builds rich metadata and error records.

    Args:
        db_session: SQLAlchemy session
        query: The search query text
        knowledge_base_configs: List of KB configs (each with 'id' and optional params)
        max_context_tokens: Maximum tokens for the formatted context
        session_history: Optional conversation history for query enrichment context

    Returns:
        Dict with keys: formatted_context, retrieved_context, metadata, errors, status
    """
    if max_context_tokens is None:
        max_context_tokens = get_setting("KB_DEFAULT_MAX_CONTEXT_TOKENS")
    errors: list[dict[str, Any]] = []
    per_kb_results: dict[str, list[RetrievedItem]] = {}
    per_kb_methods: dict[str, str] = {}
    all_items: list[RetrievedItem] = []

    # Fetch KB display metadata (name, strategy, retrieval_config) for enriching items
    kb_ids = [c.get("id") for c in knowledge_base_configs if c.get("id")]
    kb_display: dict[str, dict[str, str]] = {}
    kb_retrieval_configs: dict[str, dict] = {}
    if kb_ids:
        placeholders = ", ".join(f":kb_{i}" for i in range(len(kb_ids)))
        params = {f"kb_{i}": kid for i, kid in enumerate(kb_ids)}
        rows = db_session.execute(
            text(
                f'SELECT id, name, indexing_config, retrieval_config FROM "{AI_SCHEMA}".knowledge_bases WHERE id IN ({placeholders})'
            ),
            params,
        ).fetchall()
        for row in rows:
            cfg = row[2] or {}
            kb_display[str(row[0])] = {
                "kb_name": row[1] or "",
                "indexing_strategy": cfg.get("strategy", "chunk_embed"),
                "chunk_size": cfg.get("chunk_size"),
                "overlap": cfg.get("overlap"),
            }
            kb_retrieval_configs[str(row[0])] = row[3] or {}

    # 1. Search each KB individually, capturing per-KB errors
    valid_kb_configs = [c for c in knowledge_base_configs if c.get("id")]
    if len(knowledge_base_configs) != len(valid_kb_configs):
        logger.warning(
            "Skipping %d KB config(s) without id",
            len(knowledge_base_configs) - len(valid_kb_configs),
        )

    # Hoist query enrichment: run once, reuse across all KBs
    query_enrichment: dict[str, Any] | None = None
    pre_enriched_query: str | None = None
    pre_keyword_query: str | None = None

    should_enrich, enrichment_model = _should_enrich_query(
        valid_kb_configs,
        kb_retrieval_configs,
        kb_display,
    )
    if should_enrich:
        from .query_enrichment import enrich_query

        from agentic.knowledge.model_config import QUERY_ENRICHMENT_DEFAULT_MODEL

        result = enrich_query(
            query=query,
            retrieval_method="hybrid",
            session_history=session_history,
            model=enrichment_model,
        )
        pre_enriched_query = result["enriched_query"]
        pre_keyword_query = result["keyword_query"]
        query_enrichment = {
            "original_query": query,
            "enriched_query": pre_enriched_query,
            "keyword_query": pre_keyword_query,
            "model": enrichment_model or QUERY_ENRICHMENT_DEFAULT_MODEL,
            "method": "llm_enrichment",
        }
        if result.get("error"):
            query_enrichment["error"] = result["error"]
    else:
        # Extract tokens for debug display when using tokenization-based search
        # Check if any KB uses full_text or hybrid (which benefit from token display)
        uses_sparse_search = any(
            (
                kb_config.get("retrieval_method")
                or kb_retrieval_configs.get(kb_config.get("id"), {}).get("method")
                or "vector_search"
            )
            in ("full_text", "hybrid")
            for kb_config in valid_kb_configs
        )
        if uses_sparse_search:
            from .sparse_retrieval.query_context import QueryContextBuilder

            builder = QueryContextBuilder()
            extracted_tokens = builder.extract_context_terms(query, session_history)
            if extracted_tokens:
                query_enrichment = {
                    "original_query": query,
                    "extracted_tokens": extracted_tokens,
                    "token_count": len(extracted_tokens),
                    "method": "tokenization",
                }

    def _collect_outcome(outcome: dict[str, Any]) -> None:
        """Accumulate a single KB search outcome into shared result lists."""
        kb_id = outcome["kb_id"]
        if outcome["error"] is not None:
            errors.append(
                {
                    "type": "kb_retrieval_error",
                    "knowledge_base_id": kb_id,
                    "message": str(outcome["error"]),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            return
        per_kb_results[kb_id] = outcome["results"]
        per_kb_methods[kb_id] = outcome["method"]
        all_items.extend(outcome["results"])

    if len(valid_kb_configs) == 1:
        # Fast path: single KB — use the caller's session directly, no threading
        kb_config = valid_kb_configs[0]
        kb_id = kb_config["id"]
        retrieval_method = kb_config.get("retrieval_method")
        try:
            results = search_knowledge_base(
                db_session=db_session,
                knowledge_base_id=kb_id,
                query=query,
                top_k=kb_config.get("top_k")
                or kb_retrieval_configs.get(kb_id, {}).get(
                    "top_k", get_setting("KB_DEFAULT_TOP_K")
                ),
                retrieval_method=retrieval_method,
                similarity_threshold=kb_config.get("similarity_threshold", 0.0),
                filter_metadata=kb_config.get("filter_metadata"),
                session_history=session_history,
                pre_enriched_query=pre_enriched_query,
                pre_keyword_query=pre_keyword_query,
                source_ids=kb_config.get("source_ids"),
            )
            db_method = kb_retrieval_configs.get(kb_id, {}).get("method", "vector_search")
            if results:
                method = (results[0].meta or {}).get(
                    "retrieval_method", retrieval_method or db_method
                )
            else:
                method = retrieval_method or db_method
            _collect_outcome(
                {
                    "kb_id": kb_id,
                    "results": results,
                    "method": method,
                    "error": None,
                }
            )
        except Exception as e:
            logger.warning(f"KB retrieval error for {kb_id}: {e}")
            _collect_outcome(
                {
                    "kb_id": kb_id,
                    "results": [],
                    "method": None,
                    "error": e,
                }
            )
    elif len(valid_kb_configs) > 1:
        # Parallel path: multiple KBs — each thread gets its own DB session
        engine = db_session.get_bind()
        max_workers = min(len(valid_kb_configs), get_setting("MAX_SEARCH_WORKERS"))

        # ThreadPoolExecutor.submit does NOT propagate contextvars to
        # workers (see agentic.agent.agent + services/run_context for
        # the full pattern + rationale). Capture a FRESH copy of the
        # parent context PER submission and submit ``ctx.run`` so each
        # _search_single_kb worker sees the bound billing run_id —
        # otherwise its inner search_knowledge_base call falls back to
        # uuid4 idempotency keys and multi-KB retries double-charge for
        # vector_search / metadata_enrichment / reranker_call. A single
        # shared parent_ctx would fail with "cannot enter context: ...
        # is already entered" because Context.run accepts only one entry.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_kb = {
                executor.submit(
                    contextvars.copy_context().run,
                    _search_single_kb,
                    engine,
                    kb_config,
                    query,
                    kb_retrieval_configs,
                    session_history,
                    pre_enriched_query,
                    pre_keyword_query,
                ): kb_config
                for kb_config in valid_kb_configs
            }
            for future in as_completed(future_to_kb):
                outcome = future.result()
                _collect_outcome(outcome)

    # 2. Sort all items by score descending
    all_items.sort(key=lambda x: x.score, reverse=True)

    # 2a. Batch-fetch source metadata and inject into items
    source_meta_map = _batch_fetch_source_metadata(db_session, all_items, kb_display)
    for item in all_items:
        sm = source_meta_map.get(item.source_id, {})
        if item.meta is None:
            item.meta = {}
        item.meta.setdefault("source_name", sm.get("source_name", ""))
        item.meta.setdefault("doc_name", sm.get("source_name", ""))
        if sm.get("doc_summary"):
            item.meta.setdefault("doc_summary", sm["doc_summary"])

    # 2b. Determine per-KB context mode and resolve images if needed
    per_kb_context_mode: dict[str, str] = {}
    for kb_id in kb_ids:
        ret_cfg = kb_retrieval_configs.get(kb_id, {})
        per_kb_context_mode[kb_id] = ret_cfg.get("context_mode", "text")

    logger.info(
        "Context mode per KB: %s",
        {kb_id: mode for kb_id, mode in per_kb_context_mode.items()},
    )

    any_image_mode = "image" in per_kb_context_mode.values()
    logger.info("any_image_mode=%s", any_image_mode)

    source_image_map: dict[str, list[dict]] = {}
    image_delivery = get_setting("DEFAULT_IMAGE_DELIVERY")
    if any_image_mode:
        # Only resolve for items from image-mode KBs
        image_items = [
            item for item in all_items if per_kb_context_mode.get(item.knowledge_base_id) == "image"
        ]
        logger.info("Image mode active, resolving page images for %d items", len(image_items))
        # Use image_delivery from the first image-mode KB's config
        for kb_id, mode in per_kb_context_mode.items():
            if mode == "image":
                image_delivery = kb_retrieval_configs.get(kb_id, {}).get(
                    "image_delivery", get_setting("DEFAULT_IMAGE_DELIVERY")
                )
                break
        # Check if URL mode is viable (requires a public storage URL)
        if image_delivery == "url":
            try:
                _storage_check = get_storage()
                if not _storage_check.has_public_url():
                    logger.warning(
                        "image_delivery='url' requested but STORAGE_PUBLIC_URL is not set; "
                        "falling back to base64"
                    )
                    image_delivery = "base64"
            except Exception as e:
                logger.warning(f"URL delivery viability check failed, falling back to base64: {e}")
                image_delivery = "base64"
        source_image_map = _resolve_page_images(
            db_session, image_items, delivery_mode=image_delivery
        )
        source_ids_needed = list({item.source_id for item in image_items if item.source_id})
        logger.info(
            "Resolved images for %d/%d sources, total image records: %d",
            len(source_image_map),
            len(source_ids_needed),
            sum(len(v) for v in source_image_map.values()),
        )

    # 3. Format context with token limiting
    formatted_context, diagnostics = format_items_as_context(
        all_items,
        max_tokens=max_context_tokens,
        per_kb_context_mode=per_kb_context_mode if any_image_mode else None,
        source_image_map=source_image_map if any_image_mode else None,
        image_delivery=image_delivery,
        group_by_document=True,
    )

    logger.info(
        "format_items_as_context returned type=%s (list=%s, len=%s)",
        type(formatted_context).__name__,
        isinstance(formatted_context, list),
        len(formatted_context) if isinstance(formatted_context, (str, list)) else "N/A",
    )

    # 4. Track which items were included vs dropped
    included_set = set(diagnostics.get("included_indices", []))
    items_dropped = diagnostics.get("items_dropped", 0)

    if items_dropped > 0:
        # Build both a flat list (backward-compat) and a per-KB breakdown
        # so callers can see which knowledge base each dropped item belongs to.
        dropped_item_ids = []
        dropped_items_by_kb: dict[str, dict] = {}
        for i in range(len(all_items)):
            if i not in included_set:
                item = all_items[i]
                dropped_item_ids.append(item.item_id)
                kb_id = item.knowledge_base_id or "unknown"
                if kb_id not in dropped_items_by_kb:
                    dropped_items_by_kb[kb_id] = {
                        "kb_name": kb_display.get(kb_id, {}).get("kb_name", ""),
                        "dropped_item_ids": [],
                    }
                dropped_items_by_kb[kb_id]["dropped_item_ids"].append(item.item_id)
        errors.append(
            {
                "type": "context_truncation",
                "items_dropped": items_dropped,
                "dropped_item_ids": dropped_item_ids,  # flat list (backward-compat)
                "dropped_items_by_kb": dropped_items_by_kb,  # keyed by kb_id with name + ids
                "reason": f"exceeded max_context_tokens ({max_context_tokens})",
                "token_limit": diagnostics.get("token_limit"),
                "estimated_tokens_at_drop": diagnostics.get("estimated_tokens_used"),
            }
        )

    # 5. Build retrieved_context (diagnostics + item dicts)
    def _get_retrieval_score(c):
        meta = c.meta or {}
        for key in ("vector_similarity_score", "hybrid_search_score", "bm25_score"):
            if key in meta:
                return meta[key]
        return c.score

    drop_text_limit = get_setting("DROPPED_ITEM_TEXT_LIMIT")
    retrieved_items_list = [
        {
            "_type": (
                "page_index_node"
                if (c.meta or {}).get("retrieval_method") == "tree_search"
                else (
                    "full_document"
                    if kb_display.get(c.knowledge_base_id, {}).get("indexing_strategy")
                    == "full_document"
                    else "text_embedding_chunk"
                )
            ),
            "id": c.item_id,
            "text": (
                c.text
                if idx in included_set
                else (
                    c.text[:drop_text_limit] + "\n\n[... truncated]"
                    if len(c.text) > drop_text_limit
                    else c.text
                )
            ),
            "score": c.score,
            "retrieval_score": _get_retrieval_score(c),
            "reranker_score": (c.meta or {}).get("reranker_score"),
            "source_id": c.source_id,
            "source_name": (c.meta or {}).get("source_name", ""),
            "knowledge_base_id": c.knowledge_base_id,
            "meta": c.meta or {},
            "included_in_context": idx in included_set,
            "kb_name": kb_display.get(c.knowledge_base_id, {}).get("kb_name", ""),
            "indexing_strategy": kb_display.get(c.knowledge_base_id, {}).get(
                "indexing_strategy", ""
            ),
            "chunk_size": kb_display.get(c.knowledge_base_id, {}).get("chunk_size"),
            "overlap": kb_display.get(c.knowledge_base_id, {}).get("overlap"),
            "retrieval_method": per_kb_methods.get(
                c.knowledge_base_id, (c.meta or {}).get("retrieval_method", "")
            ),
        }
        for idx, c in enumerate(all_items)
    ]

    # Attach matched page images to items for frontend debugging
    if source_image_map:
        for item_dict in retrieved_items_list:
            pages = (item_dict.get("meta") or {}).get("pages", [])
            source_id = item_dict.get("source_id")
            if source_id and source_id in source_image_map:
                source_images = source_image_map[source_id]
                if pages:
                    matched = [img for img in source_images if img.get("page") in pages]
                else:
                    # Full-document items — attach all images
                    matched = sorted(source_images, key=lambda x: x.get("page", 0))
                if matched:
                    item_dict["images"] = matched

    # Transfer enrichment metadata from RetrievedItem.meta to retrieved_items_list
    for idx, item_dict in enumerate(retrieved_items_list):
        enrichment = (all_items[idx].meta or {}).get("enrichment")
        if enrichment:
            item_dict["enrichment_metadata"] = enrichment

    retrieved_context = [{"_type": "retrieval_diagnostics", **diagnostics}] + retrieved_items_list

    # 6. Build per-KB metadata
    total_tokens_retrieved = 0
    per_kb_meta: list[dict[str, Any]] = []
    for kb_id, kb_items in per_kb_results.items():
        kb_item_ids = [c.item_id for c in kb_items]
        kb_included = sum(
            1 for i, c in enumerate(all_items) if c.knowledge_base_id == kb_id and i in included_set
        )
        kb_dropped = len(kb_items) - kb_included
        estimated_tokens = sum(len(c.text) // 4 for c in kb_items)
        total_tokens_retrieved += estimated_tokens
        # Extract reranker info from first reranked item in this KB (if any)
        reranked_item = next(
            (c for c in kb_items if (c.meta or {}).get("reranker_score") is not None),
            None,
        )
        kb_meta = {
            "knowledge_base_id": kb_id,
            "items_retrieved": len(kb_items),
            "items_included": kb_included,
            "items_dropped": kb_dropped,
            "estimated_tokens": estimated_tokens,
            "item_ids": kb_item_ids,
            "retrieval_method": per_kb_methods.get(kb_id, "vector_search"),
        }
        if reranked_item:
            rr_cfg = (reranked_item.meta or {}).get("reranker_config", {})
            kb_meta["reranker"] = {
                "model": rr_cfg.get("model", ""),
                "config": rr_cfg,
            }
        per_kb_meta.append(kb_meta)

    metadata = {
        "total_tokens_retrieved": total_tokens_retrieved,
        "total_items_retrieved": len(all_items),
        "total_items_included": diagnostics.get("items_included", 0),
        "total_items_dropped": items_dropped,
        "token_limit": diagnostics.get("token_limit"),
        "estimated_tokens_used": diagnostics.get("estimated_tokens_used", 0),
        "per_kb": per_kb_meta,
    }

    if not all_items and errors and all(e["type"] == "kb_retrieval_error" for e in errors):
        status = ContextHandlerStatus.FAILED
    else:
        status = ContextHandlerStatus.COMPLETED

    return {
        "formatted_context": formatted_context,
        "retrieved_context": retrieved_context,
        "metadata": metadata,
        "errors": errors,
        "status": status.value,
        "query_enrichment": query_enrichment,
    }


async def execute_retrieval_async(
    db_session: Session,
    query: str,
    knowledge_base_configs: list[dict[str, Any]],
    max_context_tokens: int | None = None,
) -> dict[str, Any]:
    """Async variant of execute_retrieval for use inside a running event loop.

    Simplified single-KB path used by the workflow engine's async agent blocks.
    Uses search_knowledge_base_async() which awaits store methods directly
    instead of wrapping them with asyncio.run().
    """
    if max_context_tokens is None:
        max_context_tokens = get_setting("KB_DEFAULT_MAX_CONTEXT_TOKENS")
    errors: list[dict[str, Any]] = []
    all_items: list[RetrievedItem] = []
    per_kb_results: dict[str, list[RetrievedItem]] = {}
    per_kb_methods: dict[str, str] = {}

    valid_kb_configs = [c for c in knowledge_base_configs if c.get("id")]
    if not valid_kb_configs:
        return {
            "formatted_context": "",
            "retrieved_context": [],
            "metadata": {},
            "errors": [],
            "status": ContextHandlerStatus.COMPLETED.value,
            "query_enrichment": None,
        }

    # Fetch KB display metadata
    kb_ids = [c.get("id") for c in valid_kb_configs]
    kb_display: dict[str, dict[str, str]] = {}
    kb_retrieval_configs: dict[str, dict] = {}
    if kb_ids:
        placeholders = ", ".join(f":kb_{i}" for i in range(len(kb_ids)))
        params = {f"kb_{i}": kid for i, kid in enumerate(kb_ids)}
        rows = db_session.execute(
            text(
                f'SELECT id, name, indexing_config, retrieval_config FROM "{AI_SCHEMA}".knowledge_bases WHERE id IN ({placeholders})'
            ),
            params,
        ).fetchall()
        for row in rows:
            cfg = row[2] or {}
            kb_display[str(row[0])] = {
                "kb_name": row[1] or "",
                "indexing_strategy": cfg.get("strategy", "chunk_embed"),
            }
            kb_retrieval_configs[str(row[0])] = row[3] or {}

    # Search each KB
    for kb_config in valid_kb_configs:
        kb_id = kb_config["id"]
        retrieval_method = kb_config.get("retrieval_method")
        try:
            results = await search_knowledge_base_async(
                db_session=db_session,
                knowledge_base_id=kb_id,
                query=query,
                top_k=kb_config.get("top_k")
                or kb_retrieval_configs.get(kb_id, {}).get(
                    "top_k", get_setting("KB_DEFAULT_TOP_K")
                ),
                retrieval_method=retrieval_method,
                similarity_threshold=kb_config.get("similarity_threshold", 0.0),
                filter_metadata=kb_config.get("filter_metadata"),
                source_ids=kb_config.get("source_ids"),
            )
            db_method = kb_retrieval_configs.get(kb_id, {}).get("method", "vector_search")
            if results:
                method = (results[0].meta or {}).get(
                    "retrieval_method", retrieval_method or db_method
                )
            else:
                method = retrieval_method or db_method
            per_kb_results[kb_id] = results
            per_kb_methods[kb_id] = method
            all_items.extend(results)
        except Exception as e:
            logger.warning(f"KB retrieval error for {kb_id}: {e}")
            errors.append(
                {
                    "type": "kb_retrieval_error",
                    "knowledge_base_id": kb_id,
                    "message": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

    all_items.sort(key=lambda x: x.score, reverse=True)

    # Inject source metadata
    source_meta_map = _batch_fetch_source_metadata(db_session, all_items, kb_display)
    for item in all_items:
        sm = source_meta_map.get(item.source_id, {})
        if item.meta is None:
            item.meta = {}
        item.meta.setdefault("source_name", sm.get("source_name", ""))
        item.meta.setdefault("doc_name", sm.get("source_name", ""))

    # Format context
    formatted_context, diagnostics = format_items_as_context(
        all_items,
        max_tokens=max_context_tokens,
        group_by_document=True,
    )

    if not all_items and errors:
        status = ContextHandlerStatus.FAILED
    else:
        status = ContextHandlerStatus.COMPLETED

    return {
        "formatted_context": formatted_context,
        "retrieved_context": [],
        "metadata": {
            "total_items_retrieved": len(all_items),
            "total_items_included": diagnostics.get("items_included", 0),
            "total_items_dropped": diagnostics.get("items_dropped", 0),
        },
        "errors": errors,
        "status": status.value,
        "query_enrichment": None,
    }


def strip_tool_call_images(
    tool_calls: list[dict[str, Any]],
    events: list[dict[str, Any]],
    db_session: Session,
) -> list[dict[str, Any]]:
    """Replace base64 image_url blocks in tool_call results with image_ref placeholders.

    Uses context handler diagnostics (image_refs) to map each image block to its
    storage_path, following the same pattern as formatted_context persistence.
    """
    if not tool_calls:
        return tool_calls

    # Build tool_name → [context_handler_id, ...] from events (order of appearance)
    tool_handler_map: dict[str, list[str]] = {}
    for e in events:
        if e.get("type") == "context_handler_created":
            name = e.get("tool_name", "")
            tool_handler_map.setdefault(name, []).append(e["context_handler_id"])

    if not tool_handler_map:
        return tool_calls

    # Build tool_name → [image_refs, ...] by fetching each context handler's diagnostics
    tool_image_refs: dict[str, list[list[dict]]] = {}
    for tool_name, handler_ids in tool_handler_map.items():
        refs_list = []
        for hid in handler_ids:
            try:
                row = db_session.execute(
                    text(f"""
                        SELECT retrieved_context
                        FROM "{AI_SCHEMA}".context_handlers
                        WHERE id = :id
                    """),
                    {"id": hid},
                ).fetchone()
                if row and row[0]:
                    ctx = row[0] if isinstance(row[0], list) else json.loads(row[0])
                    diag = ctx[0] if ctx else {}
                    refs_list.append(diag.get("image_refs", []))
                else:
                    refs_list.append([])
            except Exception:
                logger.warning("Failed to fetch image_refs for handler %s", hid)
                refs_list.append([])
        tool_image_refs[tool_name] = refs_list

    # Deep-copy and strip base64 from tool_call results
    stripped = copy.deepcopy(tool_calls)
    # Track consumption index per tool_name
    tool_consume_idx: dict[str, int] = {}

    for tc in stripped:
        result = tc.get("result")
        if not isinstance(result, list):
            continue
        tool_name = tc.get("tool_name", "")
        if tool_name not in tool_image_refs:
            continue

        idx = tool_consume_idx.get(tool_name, 0)
        refs_for_call = tool_image_refs[tool_name]
        if idx >= len(refs_for_call):
            continue
        image_refs = refs_for_call[idx]
        tool_consume_idx[tool_name] = idx + 1

        if not image_refs:
            continue

        ref_by_index = {r["block_index"]: r for r in image_refs}

        # The tool result has a header block prepended by _make_search_handler,
        # so formatted_context block at index N is at result index N + 1.
        for i, block in enumerate(result):
            if not isinstance(block, dict):
                continue
            fc_index = i - 1  # offset for header block
            if fc_index in ref_by_index:
                ref = ref_by_index[fc_index]
                result[i] = {
                    "type": "image_ref",
                    "storage_path": ref["storage_path"],
                    "format": ref.get("format", "png"),
                }

    return stripped


def resolve_tool_call_image_refs(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve image_ref blocks in tool_call results to image_url with data URLs.

    Downloads image bytes from Supabase Storage and encodes as base64 data URLs,
    matching the pattern in _parse_formatted_context. Signed URLs are not used
    because storage_url may be a Docker-internal address unreachable by browsers.
    """
    if not tool_calls:
        return tool_calls

    # Fast path: check if any resolution needed
    needs_resolution = False
    for tc in tool_calls:
        result = tc.get("result")
        if isinstance(result, list):
            for block in result:
                if isinstance(block, dict) and block.get("type") == "image_ref":
                    needs_resolution = True
                    break
        if needs_resolution:
            break

    if not needs_resolution:
        return tool_calls

    storage = get_storage()

    for tc in tool_calls:
        result = tc.get("result")
        if not isinstance(result, list):
            continue
        for i, block in enumerate(result):
            if not isinstance(block, dict) or block.get("type") != "image_ref":
                continue
            path = block.get("storage_path", "")
            fmt = block.get("format", "png")
            try:
                image_bytes = storage.download_from_path(path)
                b64 = base64.b64encode(image_bytes).decode("ascii")
                mime = "image/jpeg" if fmt in ("jpg", "jpeg") else f"image/{fmt}"
                result[i] = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            except Exception:
                logger.warning("Failed to resolve tool_call image_ref: %s", path)
                result[i] = {
                    "type": "text",
                    "text": f"[Image unavailable: {path}]",
                }

    return tool_calls


def persist_context_handler(
    db_session: Session,
    query: str,
    knowledge_base_configs: list[dict[str, Any]],
    max_context_tokens: int,
    retrieval_result: dict[str, Any],
) -> str:
    """
    Persist a context handler record to the database.

    Args:
        db_session: SQLAlchemy session
        query: The original search query
        knowledge_base_configs: KB configs used
        max_context_tokens: Token limit used
        retrieval_result: Result dict from execute_retrieval()

    Returns:
        The handler's UUID string
    """
    handler_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    # formatted_context can be str (text mode) or list[dict] (image mode)
    formatted_ctx = retrieval_result["formatted_context"]
    if isinstance(formatted_ctx, list):
        # Strip base64 image data for storage — keep only storage path references.
        # Images are already stored in Supabase Storage; re-resolved on retrieval.
        # Image ref metadata lives in retrieval diagnostics (not in the content
        # blocks themselves, which must stay OpenAI-schema-clean).
        retrieved_ctx = retrieval_result.get("retrieved_context", [])
        diag = retrieved_ctx[0] if retrieved_ctx else {}
        image_refs = diag.get("image_refs", [])
        ref_by_index = {r["block_index"]: r for r in image_refs}

        lightweight_ctx = []
        for idx, block in enumerate(formatted_ctx):
            if idx in ref_by_index:
                ref = ref_by_index[idx]
                lightweight_ctx.append(
                    {
                        "type": "image_ref",
                        "storage_path": ref["storage_path"],
                        "format": ref.get("format", "png"),
                    }
                )
            else:
                lightweight_ctx.append(block)
        formatted_ctx_for_db = json.dumps(lightweight_ctx)
    else:
        formatted_ctx_for_db = formatted_ctx

    db_session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".context_handlers
            (id, query, status, knowledge_base_configs, max_context_tokens,
             retrieved_context, metadata, errors, formatted_context,
             created_at, completed_at)
            VALUES (:id, :query, :status, CAST(:knowledge_base_configs AS jsonb),
                    :max_context_tokens, CAST(:retrieved_context AS jsonb),
                    CAST(:metadata AS jsonb), CAST(:errors AS jsonb),
                    :formatted_context, :created_at, :completed_at)
        """),
        {
            "id": handler_id,
            "query": query,
            "status": retrieval_result["status"],
            "knowledge_base_configs": json.dumps(knowledge_base_configs),
            "max_context_tokens": max_context_tokens,
            "retrieved_context": json.dumps(
                make_lightweight_retrieved_context(retrieval_result["retrieved_context"])
            ),
            "metadata": json.dumps(retrieval_result["metadata"]),
            "errors": json.dumps(retrieval_result["errors"]),
            "formatted_context": formatted_ctx_for_db,
            "created_at": now,
            "completed_at": now,
        },
    )

    return handler_id


def _parse_formatted_context(raw: Any) -> Any:
    """Parse formatted_context back from DB: JSON arrays become list[dict].

    Resolves lightweight image_ref blocks (persisted without base64 data) back
    into image_url blocks with signed URLs from Supabase Storage.
    """
    if isinstance(raw, str) and raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
        # Re-resolve image_ref blocks to signed URLs
        if isinstance(parsed, list):
            needs_resolution = any(
                isinstance(b, dict) and b.get("type") == "image_ref" for b in parsed
            )
            if needs_resolution:
                try:
                    storage = get_storage()
                    for block in parsed:
                        if isinstance(block, dict) and block.get("type") == "image_ref":
                            path = block.get("storage_path", "")
                            fmt = block.get("format", "png")
                            try:
                                image_bytes = storage.download_from_path(path)
                                b64 = base64.b64encode(image_bytes).decode("ascii")
                                mime = "image/jpeg" if fmt in ("jpg", "jpeg") else f"image/{fmt}"
                                block["type"] = "image_url"
                                block["image_url"] = {"url": f"data:{mime};base64,{b64}"}
                                block.pop("storage_path", None)
                                block.pop("format", None)
                            except Exception:
                                logger.warning(
                                    "Failed to resolve image_ref block: %s",
                                    path,
                                )
                                block["type"] = "text"
                                block["text"] = f"[Image unavailable: {path}]"
                                block.pop("storage_path", None)
                                block.pop("format", None)
                except Exception as e:
                    logger.warning(
                        "Failed to resolve image_ref blocks: %s",
                        e,
                    )
                    for block in parsed:
                        if isinstance(block, dict) and block.get("type") == "image_ref":
                            path = block.get("storage_path", "")
                            block["type"] = "text"
                            block["text"] = f"[Image unavailable: {path}]"
                            block.pop("storage_path", None)
                            block.pop("format", None)
        return parsed
    return raw


def list_context_handlers(
    db_session: Session,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """List context handlers with pagination. Returns (handlers, total_count).

    Omits retrieved_context and formatted_context from the list view
    since these can be large (base64 images). Use get_context_handler()
    for full details.
    """
    result = db_session.execute(
        text(f"""
            SELECT id, query, status, knowledge_base_configs, max_context_tokens,
                   metadata, errors, created_at, completed_at
            FROM "{AI_SCHEMA}".context_handlers
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        {"limit": limit, "offset": offset},
    )

    handlers = []
    for row in result:
        handlers.append(
            {
                "id": str(row[0]),
                "query": row[1],
                "status": row[2],
                "knowledge_base_configs": row[3] or [],
                "max_context_tokens": row[4],
                "metadata": row[5] or {},
                "errors": row[6] or [],
                "created_at": row[7].isoformat() if row[7] else None,
                "completed_at": row[8].isoformat() if row[8] else None,
            }
        )

    count_result = db_session.execute(text(f'SELECT COUNT(*) FROM "{AI_SCHEMA}".context_handlers'))
    total = count_result.scalar()

    return handlers, total


def get_context_handler(
    db_session: Session,
    handler_id: str,
    resolve_text: bool = True,
) -> dict[str, Any] | None:
    """
    Fetch a context handler by ID.

    Args:
        db_session: SQLAlchemy session
        handler_id: The context handler UUID
        resolve_text: If True, resolve full_text_path pointers back to full
            document text (for API responses). If False, return pointer-based
            format (for storing in agent_runs).

    Returns:
        Dict representation of the handler, or None if not found
    """
    result = db_session.execute(
        text(f"""
            SELECT id, query, status, knowledge_base_configs, max_context_tokens,
                   retrieved_context, metadata, errors, formatted_context,
                   created_at, completed_at
            FROM "{AI_SCHEMA}".context_handlers
            WHERE id = :id
        """),
        {"id": handler_id},
    )
    row = result.fetchone()
    if not row:
        return None

    retrieved_context = row[5]
    if resolve_text and retrieved_context:
        retrieved_context = _resolve_full_text_refs(retrieved_context)
        retrieved_context = _resolve_image_refs(retrieved_context)

    return {
        "id": str(row[0]),
        "query": row[1],
        "status": row[2],
        "knowledge_base_configs": row[3] or [],
        "max_context_tokens": row[4],
        "retrieved_context": retrieved_context,
        "metadata": row[6] or {},
        "errors": row[7] or [],
        "formatted_context": _parse_formatted_context(row[8]),
        "created_at": row[9].isoformat() if row[9] else None,
        "completed_at": row[10].isoformat() if row[10] else None,
    }


def create_and_execute(
    db_session: Session,
    query: str,
    knowledge_base_configs: list[dict[str, Any]],
    max_context_tokens: int | None = None,
    session_history: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Convenience function: execute retrieval then persist the result.

    Returns:
        Tuple of (handler_id, retrieval_result)
    """
    result = execute_retrieval(
        db_session=db_session,
        query=query,
        knowledge_base_configs=knowledge_base_configs,
        max_context_tokens=max_context_tokens,
        session_history=session_history,
    )

    handler_id = persist_context_handler(
        db_session=db_session,
        query=query,
        knowledge_base_configs=knowledge_base_configs,
        max_context_tokens=max_context_tokens,
        retrieval_result=result,
    )

    return handler_id, result
