"""Source indexing Celery task.

Takes extracted source content and indexes it into a knowledge base
by chunking, embedding, and storing in pgvector.
"""

import asyncio
import json
import logging
import traceback

from ..celery import celery_app
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import text

from agentic.llm.cost_accumulator import init_accumulator, install

from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
from ..services.llm_call import (
    cached_byok_resolver,
    with_llm_key,
)
from ..services.run_context import run_scope
from ..services.knowledge_store import PgVectorKnowledgeStore
from ..services.doc2json_store import Doc2JSONStore
from ..services.full_document_store import FullDocumentStore
from ..services.graph_index_store import GraphIndexStore
from ..services.page_index_store import PageIndexStore
from ..services.storage import StorageError, SupabaseStorage, get_storage
from ..services.settings_registry import get_setting
from ..services.sparse_retrieval import (
    SparseIndexStore,
    STRATEGY_TO_BM25_ITEM_TABLE as _STRATEGY_TO_ITEM_TABLE,
)

logger = logging.getLogger(__name__)

# Indexing strategy -> billing action, per the credits catalog seeded in
# migration 0006_credit_ledger. ``full_document`` is not in the catalog as a
# distinct action; it bills as ``indexing_chunkembed`` (the closest match —
# it produces summary + embedding, the same compute family as chunk_embed).
_INDEXING_STRATEGY_TO_ACTION: dict[str, str] = {
    "chunk_embed": "indexing_chunkembed",
    "page_index": "indexing_pageindex",
    "graph_index": "indexing_graphindex",
    "doc2json": "indexing_doc2json",
    "full_document": "indexing_chunkembed",
}


def _resolve_indexing_action(strategy: str) -> str:
    """Map an indexing strategy to its billing action; falls back to chunkembed."""
    return _INDEXING_STRATEGY_TO_ACTION.get(strategy, "indexing_chunkembed")


def _quantity_from_stats(stats: dict) -> int:
    """Convert indexing stats into a 1k-token quantity for billing.

    Prefers explicit token counts from the strategy when available, falling
    back to total_chars / 4 (a stable rough conversion that mirrors
    OpenAI's tokenizer heuristic). Returns at least 1 so a successful index
    never bills 0.
    """
    # Explicit token counts win when the strategy reported them.
    for key in ("total_tokens", "input_tokens", "full_text_tokens"):
        v = stats.get(key)
        if isinstance(v, int) and v > 0:
            tokens = v
            break
    else:
        chars = stats.get("total_chars") or 0
        tokens = max(1, int(chars) // 4)
    return max(1, (tokens + 999) // 1000)


def _get_kb_retrieval_method(kb_id: str) -> str | None:
    """Read retrieval_config.method for a KB; returns None if KB not found."""
    row = db.session.execute(
        text(
            f"SELECT retrieval_config->>'method' "
            f'FROM "{AI_SCHEMA}".knowledge_bases WHERE id = :id'
        ),
        {"id": kb_id},
    ).fetchone()
    return row[0] if row else None


def _should_build_bm25_now(kb_id: str) -> bool:
    """Decide whether `sparse_store.add_and_save(...)` should run for this KB.

    Returns False when:
      - the KB's retrieval method does not use BM25 (vector_search, None, unknown), OR
      - the project-level BM25_AUTO_INDEXING setting is disabled.
    """
    method = _get_kb_retrieval_method(kb_id)
    if method not in ("hybrid", "full_text"):
        return False
    if not get_setting("BM25_AUTO_INDEXING"):
        return False
    return True


def _fetch_kb_for_bm25_build(kb_id: str) -> dict:
    """Read KB id + indexing_config for the build task."""
    row = db.session.execute(
        text(f'SELECT id::text, indexing_config FROM "{AI_SCHEMA}".knowledge_bases WHERE id = :id'),
        {"id": kb_id},
    ).fetchone()
    if row is None:
        raise ValueError(f"KB not found: {kb_id}")
    return {"id": row[0], "indexing_config": row[1] or {}}


def _iter_items_for_kb_bm25(kb_id: str, item_table: str, batch_size: int = 10_000):
    """Yield batches of {id, text} for the right item_table for this KB."""
    if item_table == "chunks":
        sql = text(
            f"SELECT c.id::text, c.text "
            f'FROM "{AI_SCHEMA}".chunks c '
            f'JOIN "{AI_SCHEMA}".indexed_sources i ON i.id = c.indexed_source_id '
            f"WHERE i.knowledge_base_id = :kb "
            f"ORDER BY c.id"
        )
    elif item_table == "full_documents":
        sql = text(
            f"SELECT d.id::text, d.summary "
            f'FROM "{AI_SCHEMA}".full_documents d '
            f'JOIN "{AI_SCHEMA}".indexed_sources i ON i.id = d.indexed_source_id '
            f"WHERE i.knowledge_base_id = :kb "
            f"ORDER BY d.id"
        )
    elif item_table == "graph_index_nodes":
        sql = text(
            f"SELECT n.id::text, COALESCE(n.title, '') || ' ' || COALESCE(n.text, '') "
            f'FROM "{AI_SCHEMA}".graph_index_nodes n '
            f'JOIN "{AI_SCHEMA}".indexed_sources i ON i.id = n.indexed_source_id '
            f"WHERE i.knowledge_base_id = :kb "
            f"ORDER BY n.id"
        )
    else:
        raise ValueError(f"Unsupported item_table for BM25 rebuild: {item_table}")

    result = db.session.execute(sql, {"kb": kb_id})
    batch: list[dict] = []
    for row in result:
        batch.append({"id": row[0], "text": row[1] or ""})
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# Register the LiteLLM cost-tracking callback exactly once per worker process.
# Subsequent init_accumulator() calls scope an accumulator to each task's thread.
install()


def get_source(source_id: str) -> dict | None:
    """Get a source record from the database."""
    result = db.session.execute(
        text(f"""
            SELECT id, name, file_type, storage_path, extraction_status,
                   derivatives, metadata, auto_metadata
            FROM "{AI_SCHEMA}".sources
            WHERE id = :id
        """),
        {"id": source_id},
    )

    row = result.fetchone()
    if not row:
        return None

    return {
        "id": str(row[0]),
        "name": row[1],
        "file_type": row[2],
        "storage_path": row[3],
        "extraction_status": row[4],
        "derivatives": row[5] or {},
        "metadata": row[6] or {},
        "auto_metadata": row[7] or {},
    }


def get_knowledge_base(kb_id: str) -> dict | None:
    """Get a knowledge base record."""
    result = db.session.execute(
        text(f"""
            SELECT id, name, description, indexing_config, retrieval_config
            FROM "{AI_SCHEMA}".knowledge_bases
            WHERE id = :id
        """),
        {"id": kb_id},
    )

    row = result.fetchone()
    if not row:
        return None

    return {
        "id": str(row[0]),
        "name": row[1],
        "description": row[2],
        "indexing_config": row[3] or {},
        "retrieval_config": row[4] or {},
    }


def _mark_indexed_source_failed(indexed_source_id: str, error_message: str) -> None:
    """Mark an indexed_source as failed with a descriptive error message."""
    try:
        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".indexed_sources
                SET index_status = 'failed', error_message = :error_message
                WHERE id = :id
            """),
            {"id": indexed_source_id, "error_message": error_message},
        )
        db.session.commit()
    except Exception:
        logger.warning(
            "Failed to mark indexed_source %s as failed", indexed_source_id, exc_info=True
        )
        db.session.rollback()


def update_indexed_source_status(
    indexed_source_id: str,
    status: str,
    error_message: str | None = None,
    celery_task_id: str | None = None,
) -> None:
    """Update indexed source status."""
    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".indexed_sources
            SET index_status = :status,
                error_message = :error_message,
                celery_task_id = :celery_task_id
            WHERE id = :id
        """),
        {
            "id": indexed_source_id,
            "status": status,
            "error_message": error_message,
            "celery_task_id": celery_task_id,
        },
    )
    db.session.commit()


def _get_indexed_source_snapshot(indexed_source_id: str) -> dict | None:
    """Fetch the current indexing_config_snapshot for an indexed source."""
    result = db.session.execute(
        text(f"""
            SELECT indexing_config_snapshot
            FROM "{AI_SCHEMA}".indexed_sources
            WHERE id = :id
        """),
        {"id": indexed_source_id},
    )
    row = result.fetchone()
    return row[0] if row else None


def update_indexed_source_config_snapshot(
    indexed_source_id: str,
    indexing_config: dict,
) -> None:
    """Update indexing_config_snapshot to the config used for this run (e.g. on reindex)."""
    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".indexed_sources
            SET indexing_config_snapshot = CAST(:config AS jsonb)
            WHERE id = :id
        """),
        {
            "id": indexed_source_id,
            "config": json.dumps(indexing_config),
        },
    )
    db.session.commit()


def update_indexed_source_result(
    indexed_source_id: str,
    stats: dict,
) -> None:
    """Update indexed source with indexing results."""
    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".indexed_sources
            SET index_status = 'indexed',
                indexed_at = NOW(),
                stats = CAST(:stats AS jsonb),
                error_message = NULL
            WHERE id = :id
        """),
        {
            "id": indexed_source_id,
            "stats": json.dumps(stats),
        },
    )
    db.session.commit()


def get_page_texts_from_derivative(
    source: dict,
    storage: SupabaseStorage | None = None,
) -> list[str] | None:
    """Load per-page texts from page_text derivatives.

    Returns:
        List of per-page text strings (sorted by page number), or None.
    """
    derivatives = source.get("derivatives", {})
    pt_derivs = derivatives.get("page_text", [])
    if not pt_derivs or not storage:
        return None

    pt_derivs = sorted(pt_derivs, key=lambda d: d.get("page", 0))

    # Validate all paths upfront before downloading
    paths: list[tuple[str, str]] = []
    for deriv in pt_derivs:
        sp = deriv.get("storage_path")
        if not sp:
            logger.warning(
                "Missing storage_path for page_text derivative (page=%s)",
                deriv.get("page", "?"),
            )
            return None
        parts = sp.split("/", 1)
        if len(parts) != 2:
            logger.warning(
                "Malformed storage_path '%s' (page=%s)",
                sp,
                deriv.get("page", "?"),
            )
            return None
        paths.append((parts[0], parts[1]))

    # Download in parallel
    def _download(bucket_and_path: tuple[str, str]) -> str:
        raw = storage.download(bucket_and_path[0], bucket_and_path[1])
        return raw.decode("utf-8")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(len(paths), 10)) as pool:
        try:
            page_texts = list(pool.map(_download, paths))
        except Exception as e:
            logger.warning("Failed to download page texts in parallel: %s", e)
            return None

    return page_texts if page_texts else None


def get_page_images_from_derivative(
    source: dict,
    storage: SupabaseStorage | None = None,
) -> list[dict] | None:
    """Fetch and base64-encode page images from source derivatives.

    Returns:
        List of dicts with keys: content (base64), format, page
        Returns None if no image derivatives or download fails.
    """
    import base64
    from concurrent.futures import ThreadPoolExecutor

    derivatives = source.get("derivatives", {})
    img_derivs = derivatives.get("image", [])
    if not img_derivs or not storage:
        return None

    # Sort by page number
    img_derivs = sorted(img_derivs, key=lambda d: d.get("page", 0))

    # Validate all paths upfront
    paths_info: list[tuple[str, int, str]] = []
    for deriv in img_derivs:
        sp = deriv.get("storage_path")
        page = deriv.get("page", 0)
        if not sp:
            logger.warning("Missing storage_path for image derivative (page=%s)", page)
            continue
        # Determine format from path
        fmt = "png"
        if sp.lower().endswith(".jpg") or sp.lower().endswith(".jpeg"):
            fmt = "jpeg"
        paths_info.append((sp, page, fmt))

    if not paths_info:
        return None

    def _download_and_encode(info: tuple[str, int, str]) -> dict | None:
        sp, page, fmt = info
        try:
            raw = storage.download_from_path(sp)
            b64 = base64.b64encode(raw).decode("ascii")
            return {"content": b64, "format": fmt, "page": page}
        except Exception as e:
            logger.warning("Failed to download image %s: %s", sp, e)
            return None

    with ThreadPoolExecutor(max_workers=min(len(paths_info), 10)) as pool:
        results = list(pool.map(_download_and_encode, paths_info))

    # Filter out failed downloads
    images = [r for r in results if r is not None]
    return images if images else None


def get_text_derivative_content(storage: SupabaseStorage, source: dict) -> str | None:
    """Get the text content from source derivatives."""
    derivatives = source.get("derivatives", {})

    for deriv_type in ["markdown", "text", "html"]:
        if deriv_type in derivatives:
            deriv_list = derivatives[deriv_type]
            if deriv_list and len(deriv_list) > 0:
                deriv = deriv_list[0]
                storage_path = deriv.get("storage_path")
                if storage_path:
                    try:
                        content_bytes = storage.download_from_path(storage_path)
                        return content_bytes.decode("utf-8")
                    except Exception as e:
                        logger.warning(f"Failed to download {deriv_type} derivative: {e}")

    return None


def compute_page_boundaries(page_texts: list[str], separator: str = "\n") -> list[int]:
    """Compute cumulative character boundaries for each page.

    Returns list of length len(page_texts)+1 where boundaries[i] is the
    start char offset of page i, and boundaries[-1] is the total length.
    """
    boundaries = [0]
    for i, pt in enumerate(page_texts):
        next_offset = boundaries[-1] + len(pt)
        if i < len(page_texts) - 1:
            next_offset += len(separator)
        boundaries.append(next_offset)
    return boundaries


def map_chunk_to_pages(start_char: int, end_char: int, boundaries: list[int]) -> list[int]:
    """Return 1-indexed page numbers that a chunk spans."""
    pages = []
    for i in range(len(boundaries) - 1):
        page_start = boundaries[i]
        page_end = boundaries[i + 1]
        if start_char < page_end and end_char > page_start:
            pages.append(i + 1)
    return pages


async def run_indexing(
    kb_id: str,
    indexed_source_id: str,
    source_id: str,
    content: str,
    indexing_config: dict,
    page_texts: list[str] | None = None,
    extraction_method: str | None = None,
    provider_keys: dict[str, str] | None = None,  # unused after #437
) -> dict:
    """Run the actual indexing process asynchronously."""
    import time

    from agentic.knowledge.chunking import MarkdownHeaderChunking
    from agentic.knowledge.embedder import LiteLLMEmbedder
    from agentic.knowledge.indexing import ChunkAndEmbedAlgorithm
    from agentic.knowledge.model_config import (
        CHUNK_EMBED_DEFAULT_CHUNK_SIZE,
        CHUNK_EMBED_DEFAULT_OVERLAP,
        CHUNK_EMBED_EMBEDDING_MODEL,
    )
    from agentic.knowledge.models import IndexingConfig

    chunk_size = indexing_config.get("chunk_size", CHUNK_EMBED_DEFAULT_CHUNK_SIZE)
    chunk_overlap = indexing_config.get("overlap", CHUNK_EMBED_DEFAULT_OVERLAP)
    embedding_model = indexing_config.get("embedding_model", CHUNK_EMBED_EMBEDDING_MODEL)

    logger.info(
        f"Indexing with chunk_size={chunk_size}, overlap={chunk_overlap}, model={embedding_model}"
    )
    _run_indexing_t0 = time.monotonic()

    chunker = MarkdownHeaderChunking(
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )
    # chunk_embed is platform-billed (indexing_chunkembed); use env key, not BYOK
    embedder = LiteLLMEmbedder(model=embedding_model)

    algorithm = ChunkAndEmbedAlgorithm(
        chunker=chunker,
        embedder=embedder,
    )

    config = IndexingConfig(
        strategy="chunk_embed",
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )

    _t_aindex = time.monotonic()
    result = await algorithm.aindex(
        content=content,
        config=config,
        source_id=source_id,
    )
    _aindex_secs = time.monotonic() - _t_aindex

    logger.info(f"Generated {len(result.chunks)} chunks")

    if not result.chunks:
        return {"artifact_count": 0, "total_chars": len(content)}

    chunk_dicts = []
    total_tokens = 0

    # Compute page boundaries for chunk-to-page mapping
    page_boundaries = None
    if page_texts:
        # Separator must match what the extractor used to join pages
        sep = (
            "\n\n"
            if extraction_method
            in (
                "pdfplumber",
                "mistral_ocr",
                "paddleocr_vl",
                "lighton_ocr",
                "opendataloader",
                "txt-native",
                "markdown-native",
            )
            else "\n"
        )
        page_boundaries = compute_page_boundaries(page_texts, separator=sep)

    for chunk in result.chunks:
        meta = dict(chunk.metadata) if hasattr(chunk, "metadata") and chunk.metadata else {}

        # Map chunk to source pages if boundaries are available
        if page_boundaries and chunk.start_char is not None and chunk.end_char is not None:
            meta["pages"] = map_chunk_to_pages(chunk.start_char, chunk.end_char, page_boundaries)

        chunk_dict = {
            "text": chunk.text,
            "embedding": chunk.embedding,
            "source_id": source_id,
            "chunk_index": chunk.index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "tokens": getattr(chunk, "tokens", None),
            "meta": meta,
        }
        chunk_dicts.append(chunk_dict)
        tokens = getattr(chunk, "tokens", None)
        if tokens:
            total_tokens += tokens

    store = PgVectorKnowledgeStore(
        db_session=db.session,
        knowledge_base_id=kb_id,
    )

    _t_store = time.monotonic()
    stored_count, chunk_ids = await store.store_chunks(
        indexed_source_id, chunk_dicts, embedding_model=embedding_model
    )
    _store_secs = time.monotonic() - _t_store

    # Build/update BM25s sparse index for fast keyword search
    from ..services.sparse_retrieval import SparseIndexStore

    _t_bm25 = time.monotonic()
    if _should_build_bm25_now(kb_id):
        sparse_store = SparseIndexStore(knowledge_base_id=kb_id)
        sparse_store.add_and_save(
            documents=[c["text"] for c in chunk_dicts],
            item_ids=chunk_ids,
            item_table="chunks",
        )
        _bm25_secs = time.monotonic() - _t_bm25
        logger.info(f"Updated BM25s sparse index: {len(chunk_ids)} chunks added")
    else:
        _bm25_secs = 0.0
        logger.info("Skipped BM25 update (KB does not require BM25 or auto-indexing is off)")

    _run_indexing_total = time.monotonic() - _run_indexing_t0
    logger.info(
        "run_indexing_timing aindex=%.2f store_chunks=%.2f bm25=%.2f total=%.2f chunks=%d",
        _aindex_secs,
        _store_secs,
        _bm25_secs,
        _run_indexing_total,
        len(chunk_ids),
    )

    # Ensure hnsw index exists for this embedding dimension
    embedding_dim = len(result.chunks[0].embedding) if result.chunks[0].embedding else 0
    if embedding_dim:
        from ..services.base_vector_store import ensure_embedding_index

        ensure_embedding_index(db.session, AI_SCHEMA, embedding_dim)

    stats = {
        "artifact_count": stored_count,
        "total_chars": len(content),
        "total_tokens": total_tokens,
        "embedding_dim": embedding_dim,
        "embedding_model": embedding_model,
    }

    return stats


async def run_page_index_indexing(
    kb_id: str,
    indexed_source_id: str,
    source_id: str,
    content: str,
    indexing_config: dict,
    source_name: str | None = None,
    page_texts: list[str] | None = None,
) -> dict:
    """Run PageIndex tree-building indexing process.

    Takes markdown content and builds a hierarchical tree structure
    using LLM-powered analysis, then stores the ToC in page_index_toc
    and individual sections in page_index_nodes.
    """
    with run_scope(f"indexed_source:{indexed_source_id}"), cached_byok_resolver():
        return await _do_run_page_index_indexing(
            kb_id,
            indexed_source_id,
            source_id,
            content,
            indexing_config,
            source_name,
            page_texts,
        )


async def _do_run_page_index_indexing(
    kb_id,
    indexed_source_id,
    source_id,
    content,
    indexing_config,
    source_name,
    page_texts,
):
    """Original ``run_page_index_indexing`` body, separated so the
    ``run_scope`` contextmanager in the public wrapper resets the
    contextvar after this function returns — and the body keeps its
    original indentation (no re-indent of ~100 lines)."""
    from agentic.knowledge.indexing import PageIndexAlgorithm
    from agentic.knowledge.model_config import (
        PAGEINDEX_INDEXING_MODEL,
        PAGEINDEX_MIN_TOKEN_THRESHOLD,
        PAGEINDEX_SUMMARY_TOKEN_THRESHOLD,
    )
    from agentic.knowledge.models import IndexingConfig

    model = indexing_config.get("model", PAGEINDEX_INDEXING_MODEL)
    if_add_node_summary = indexing_config.get("if_add_node_summary", "yes")

    logger.info(
        f"PageIndex indexing with model={model}, "
        f"summary={if_add_node_summary}, source={source_name}"
    )

    extra = {
        "source_name": source_name,
        "model": model,
        "if_add_node_summary": if_add_node_summary,
        "if_thinning": indexing_config.get("if_thinning", False),
        "min_token_threshold": indexing_config.get(
            "min_token_threshold", PAGEINDEX_MIN_TOKEN_THRESHOLD
        ),
        "summary_token_threshold": indexing_config.get(
            "summary_token_threshold", PAGEINDEX_SUMMARY_TOKEN_THRESHOLD
        ),
    }
    # Forward the tree-building reasoning effort to PageIndexAlgorithm, which
    # threads it through to _llm_completion (see _pageindex_lib/utils.py).
    if "reasoning_effort" in indexing_config:
        extra["reasoning_effort"] = indexing_config["reasoning_effort"]
    llm_max_concurrent = indexing_config.get("llm_max_concurrent")
    if llm_max_concurrent is not None:
        extra["llm_max_concurrent"] = llm_max_concurrent
    if page_texts is not None:
        extra["page_texts"] = page_texts

    config = IndexingConfig(
        strategy="page_index",
        extra=extra,
    )

    algorithm = PageIndexAlgorithm()
    result = await algorithm.aindex(
        content=content,
        config=config,
        source_id=source_id,
    )

    node_count = result.stats.get("node_count", 0)
    section_count = result.stats.get("section_count", 0)
    logger.info(f"PageIndex generated tree with {node_count} nodes, {section_count} sections")

    # Store ToC (lightweight metadata-only tree)
    store = PageIndexStore(
        db_session=db.session,
        knowledge_base_id=kb_id,
    )

    toc_id = store.store_toc(
        indexed_source_id=indexed_source_id,
        source_id=source_id,
        structure=result.toc_structure,
        doc_name=result.doc_name,
        doc_description=result.doc_description,
    )

    # Store individual section rows
    from dataclasses import asdict

    section_dicts = [asdict(s) for s in result.sections]

    stored_sections = store.store_nodes(
        toc_id=toc_id,
        indexed_source_id=indexed_source_id,
        source_id=source_id,
        nodes=section_dicts,
    )

    db.session.commit()

    stats = {
        "artifact_count": 1,  # One ToC per source
        "toc_count": 1,
        "section_count": stored_sections,
        "node_count": node_count,
        "total_chars": len(content),
        "model": model,
        "strategy": "page_index",
    }

    return stats


async def run_full_document_indexing(
    kb_id: str,
    indexed_source_id: str,
    source_id: str,
    content: str,
    indexing_config: dict,
    provider_keys: dict[str, str] | None = None,  # unused after #437
) -> dict:
    """Generate an LLM summary, embed it, and store the full document.

    At search time, retrieval operates on the summary + summary_embedding
    but the entire full_text is returned to the caller.
    """
    import litellm
    from agentic.knowledge.chunking.token_utils import build_token_char_map, count_tokens
    from agentic.knowledge.embedder import LiteLLMEmbedder
    from agentic.knowledge.model_config import (
        FULLDOC_EMBEDDING_MODEL,
        FULLDOC_SUMMARY_INPUT_CHARS,
        FULLDOC_SUMMARY_MAX_TOKENS,
        FULLDOC_SUMMARY_MODEL,
    )
    from agentic.llm.routing import maybe_route_through_responses, reasoning_call_kwargs

    summary_model = indexing_config.get("summary_model", FULLDOC_SUMMARY_MODEL)
    embedding_model = indexing_config.get("embedding_model", FULLDOC_EMBEDDING_MODEL)
    reasoning_effort = indexing_config.get("reasoning_effort")

    logger.info(
        f"Full-document indexing: summary_model={summary_model}, "
        f"embedding_model={embedding_model}, content_len={len(content)}"
    )

    # 1. Truncate content to first ~32K tokens for summarisation input
    max_input_tokens = 32_000
    try:
        char_map = build_token_char_map(content)
        if len(char_map) - 1 > max_input_tokens:
            truncated_content = content[: char_map[max_input_tokens]]
        else:
            truncated_content = content
    except Exception as e:
        logger.warning(f"Token-based truncation failed, falling back to char limit: {e}")
        truncated_content = content[:FULLDOC_SUMMARY_INPUT_CHARS]

    # 2. Generate summary via LLM
    system_prompt = (
        "You are a document summarizer. Produce a comprehensive summary of the "
        "following document. The summary should capture all key topics, entities, "
        "facts, and relationships so that it can be used for semantic search. "
        "Be thorough but concise."
    )
    # Route OpenAI reasoning models through the Responses bridge and pack the
    # effort per the verified per-provider shape (see agentic.llm.routing).
    routed_summary_model = maybe_route_through_responses(summary_model, reasoning_effort)
    reasoning_kwargs = reasoning_call_kwargs(reasoning_effort, routed_summary_model)
    with (
        run_scope(f"indexed_source:{indexed_source_id}"),
        with_llm_key(summary_model) as api_key,
    ):
        response = await litellm.acompletion(
            model=routed_summary_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": truncated_content},
            ],
            max_tokens=FULLDOC_SUMMARY_MAX_TOKENS,
            api_key=api_key,
            drop_params=True,
            **reasoning_kwargs,
        )
    raw_content = response.choices[0].message.content
    if not raw_content or not raw_content.strip():
        raise ValueError(
            f"LLM returned empty summary for source {source_id} "
            f"(model={summary_model}, input_len={len(truncated_content)} chars)"
        )
    summary = raw_content.strip()
    summary_tokens = count_tokens(summary)
    logger.info(f"Generated summary: {summary_tokens} tokens, {len(summary)} chars")

    # 3. Embed the summary
    # full_document_index is platform-billed (indexing_full); use env key, not BYOK
    embedder = LiteLLMEmbedder(model=embedding_model)
    summary_embedding = await embedder.aembed(summary)
    logger.info(f"Embedded summary: dim={len(summary_embedding)}, model={embedding_model}")

    # 4. Count full-text tokens
    full_text_tokens = count_tokens(content)
    logger.info(f"Full text: {full_text_tokens} tokens, {len(content)} chars")

    # 5. Store in full_documents table (text uploaded to storage, path stored in DB)
    store = FullDocumentStore(
        db_session=db.session,
        knowledge_base_id=kb_id,
        storage=get_storage(),
    )
    doc_id = store.store_full_document(
        indexed_source_id=indexed_source_id,
        source_id=source_id,
        summary=summary,
        summary_embedding=summary_embedding,
        full_text=content,
        summary_model=summary_model,
        embedding_model=embedding_model,
        summary_tokens=summary_tokens,
        full_text_tokens=full_text_tokens,
    )

    # Build/update BM25s sparse index for full_documents (search on summary)
    from ..services.sparse_retrieval import SparseIndexStore

    if _should_build_bm25_now(kb_id):
        sparse_store = SparseIndexStore(knowledge_base_id=kb_id)
        sparse_store.add_and_save(
            documents=[summary],
            item_ids=[doc_id],
            item_table="full_documents",
        )
        logger.info("Updated BM25s sparse index: 1 full_document added")
    else:
        logger.info("Skipped BM25 update (KB does not require BM25 or auto-indexing is off)")

    # Ensure hnsw index exists for this embedding dimension
    if summary_embedding:
        from ..services.base_vector_store import ensure_embedding_index

        ensure_embedding_index(db.session, AI_SCHEMA, len(summary_embedding))

    stats = {
        "artifact_count": 1,
        "total_chars": len(content),
        "summary_chars": len(summary),
        "summary_tokens": summary_tokens,
        "full_text_tokens": full_text_tokens,
        "embedding_dim": len(summary_embedding),
        "summary_model": summary_model,
        "embedding_model": embedding_model,
        "strategy": "full_document",
    }

    return stats


async def run_graph_index_indexing(
    kb_id: str,
    indexed_source_id: str,
    source_id: str,
    content: str,
    indexing_config: dict,
    source_name: str | None = None,
    page_texts: list[str] | None = None,
    source: dict | None = None,
    provider_keys: dict[str, str] | None = None,  # unused after #437
) -> dict:
    """Run GraphIndex three-stage indexing process.

    Stage 1: Build ToC using PageIndexAlgorithm (reuse page_index pipeline)
    Stage 2: Enrich nodes with cross-section references via LLM
    Stage 3: Compute embeddings on node summaries + metadata
    """
    with run_scope(f"indexed_source:{indexed_source_id}"), cached_byok_resolver():
        return await _do_run_graph_index_indexing(
            kb_id,
            indexed_source_id,
            source_id,
            content,
            indexing_config,
            source_name,
            page_texts,
            source,
            provider_keys,
        )


async def _do_run_graph_index_indexing(
    kb_id,
    indexed_source_id,
    source_id,
    content,
    indexing_config,
    source_name,
    page_texts,
    source,
    provider_keys,
):
    """Inner body of ``run_graph_index_indexing``. Split so the outer
    wrapper can hold the ``run_scope`` contextmanager open across
    all inner LLM calls (stage 2 graph enrichment fires LLM calls via
    ``graph_enricher.enrich_referenced_nodes``)."""
    from agentic.knowledge.embedder import LiteLLMEmbedder
    from agentic.knowledge.indexing import PageIndexAlgorithm
    from agentic.knowledge.model_config import (
        GRAPHINDEX_EMBEDDING_MODEL,
        GRAPHINDEX_ENRICHMENT_MODEL,
        GRAPHINDEX_INDEXING_MODEL,
        PAGEINDEX_MIN_TOKEN_THRESHOLD,
        PAGEINDEX_SUMMARY_TOKEN_THRESHOLD,
    )
    from agentic.knowledge.models import IndexingConfig

    from ..services.graph_enricher import enrich_referenced_nodes

    model = indexing_config.get("model", GRAPHINDEX_INDEXING_MODEL)
    enrichment_model = indexing_config.get("enrichment_model", GRAPHINDEX_ENRICHMENT_MODEL)
    embedding_model = indexing_config.get("embedding_model", GRAPHINDEX_EMBEDDING_MODEL)
    if_add_node_summary = indexing_config.get("if_add_node_summary", "yes")

    logger.info(
        f"GraphIndex indexing with model={model}, enrichment={enrichment_model}, "
        f"embedding={embedding_model}, source={source_name}"
    )

    # ===== Stage 1: ToC building (reuse PageIndexAlgorithm) =====
    extra = {
        "source_name": source_name,
        "model": model,
        "if_add_node_summary": if_add_node_summary,
        "if_thinning": indexing_config.get("if_thinning", False),
        "min_token_threshold": indexing_config.get(
            "min_token_threshold", PAGEINDEX_MIN_TOKEN_THRESHOLD
        ),
        "summary_token_threshold": indexing_config.get(
            "summary_token_threshold", PAGEINDEX_SUMMARY_TOKEN_THRESHOLD
        ),
    }
    if "reasoning_effort" in indexing_config:
        extra["reasoning_effort"] = indexing_config["reasoning_effort"]
    llm_max_concurrent = indexing_config.get("llm_max_concurrent")
    if llm_max_concurrent is not None:
        extra["llm_max_concurrent"] = llm_max_concurrent
    if page_texts is not None:
        extra["page_texts"] = page_texts

    config = IndexingConfig(
        strategy="page_index",
        extra=extra,
    )

    algorithm = PageIndexAlgorithm()
    result = await algorithm.aindex(
        content=content,
        config=config,
        source_id=source_id,
    )

    node_count = result.stats.get("node_count", 0)
    section_count = result.stats.get("section_count", 0)
    logger.info(f"GraphIndex Stage 1: tree with {node_count} nodes, {section_count} sections")

    # Store ToC + nodes (no embeddings yet)
    store = GraphIndexStore(
        db_session=db.session,
        knowledge_base_id=kb_id,
    )

    toc_id = store.store_toc(
        indexed_source_id=indexed_source_id,
        source_id=source_id,
        structure=result.toc_structure,
        doc_name=result.doc_name,
        doc_description=result.doc_description,
    )

    # Build section dicts with toc_id in meta
    from dataclasses import asdict

    section_dicts = []
    for s in result.sections:
        sd = asdict(s)
        meta = sd.get("meta", {}) or {}
        meta["toc_id"] = toc_id
        meta["node_id"] = sd["node_id"]
        sd["meta"] = meta
        section_dicts.append(sd)

    stored_sections, node_row_ids = store.store_nodes(
        toc_id=toc_id,
        indexed_source_id=indexed_source_id,
        source_id=source_id,
        nodes=section_dicts,
        embedding_model=embedding_model,
    )

    # Build/update BM25s sparse index for graph_index_nodes (search on title + text)
    from ..services.sparse_retrieval import SparseIndexStore

    if _should_build_bm25_now(kb_id):
        sparse_store = SparseIndexStore(knowledge_base_id=kb_id)
        node_texts = [f"{s.get('title', '')} {s.get('text', '')}" for s in section_dicts]
        sparse_store.add_and_save(
            documents=node_texts,
            item_ids=node_row_ids,
            item_table="graph_index_nodes",
        )
        logger.info(f"Updated BM25s sparse index: {len(node_row_ids)} graph_index_nodes added")
    else:
        logger.info("Skipped BM25 update (KB does not require BM25 or auto-indexing is off)")

    logger.info(
        "GraphIndex Stage 1 complete: stored %d nodes for toc %s",
        stored_sections,
        toc_id,
    )

    # ===== Stage 2: Referenced nodes enrichment =====
    logger.info(
        "GraphIndex Stage 2: enriching references for toc %s (%d nodes)",
        toc_id,
        stored_sections,
    )
    ref_results, enrichment_errors = await enrich_referenced_nodes(
        db_session=db.session,
        knowledge_base_id=kb_id,
        toc_id=toc_id,
        model=enrichment_model,
        api_key=None,
        reasoning_effort=indexing_config.get("enrichment_reasoning_effort"),
    )

    ref_count = sum(len(v) for v in ref_results.values())
    logger.info(
        "GraphIndex Stage 2 complete: %d total references found for toc %s",
        ref_count,
        toc_id,
    )

    # ===== Stage 3: Embedding computation =====
    # Re-fetch nodes to get updated meta (with referenced_nodes)
    all_nodes = store.get_all_nodes_for_toc(toc_id)

    # Build a node_id -> title lookup for reference title resolution
    node_title_map = {n["node_id"]: n.get("title", "") for n in all_nodes}

    embedder = LiteLLMEmbedder(model=embedding_model)

    # Build embedding input texts
    embed_inputs = []
    node_ids_ordered = []
    for node in all_nodes:
        meta = node.get("meta") or {}
        title = node.get("title", "")
        summary = meta.get("summary", "")
        referenced = meta.get("referenced_nodes", [])

        parts = []
        if title:
            parts.append(title)
        if summary:
            parts.append(summary)
        if referenced:
            ref_titles = [node_title_map.get(r, r) for r in referenced]
            parts.append(f"References: {', '.join(ref_titles)}")

        embed_text = "\n".join(parts) if parts else (node.get("text") or "")[:500]
        embed_inputs.append(embed_text)
        node_ids_ordered.append(node["node_id"])

    logger.info(
        "GraphIndex Stage 3: embedding %d nodes for toc %s (model=%s)",
        len(embed_inputs),
        toc_id,
        embedding_model,
    )

    # Batch embed
    embeddings = await embedder.aembed_batch(embed_inputs)

    if len(embeddings) != len(node_ids_ordered):
        raise ValueError(
            f"Embedding count mismatch: got {len(embeddings)} embeddings "
            f"for {len(node_ids_ordered)} nodes"
        )

    # Store embeddings
    for nid, emb in zip(node_ids_ordered, embeddings):
        store.update_node_embedding(toc_id, nid, emb, embedding_model=embedding_model)

    # Ensure hnsw index exists for this embedding dimension
    if embeddings:
        from ..services.base_vector_store import ensure_embedding_index

        ensure_embedding_index(db.session, AI_SCHEMA, len(embeddings[0]))

    # Single commit after all three stages succeed
    db.session.commit()

    logger.info(
        "GraphIndex Stage 3 complete: embedded %d nodes (dim=%d) for toc %s",
        len(embeddings),
        len(embeddings[0]) if embeddings else 0,
        toc_id,
    )

    stats = {
        "artifact_count": 1,
        "toc_count": 1,
        "section_count": stored_sections,
        "node_count": node_count,
        "total_chars": len(content),
        "model": model,
        "enrichment_model": enrichment_model,
        "embedding_model": embedding_model,
        "embedding_dim": len(embeddings[0]) if embeddings else 0,
        "reference_count": ref_count,
        "strategy": "graph_index",
    }

    return stats


async def run_doc2json_indexing(
    kb_id: str,
    indexed_source_id: str,
    source_id: str,
    content: str,
    indexing_config: dict,
    page_images: list[dict] | None = None,
) -> dict:
    """Run Doc2JSON sliding window extraction and indexing.

    Scans document with overlapping windows, extracting structured JSON
    according to a user-defined schema. Generates summaries at each step
    and produces a combined summary for retrieval.

    Supports two modes:
    - Text mode (default): Process extracted text with token-based windows
    - Image mode (use_images=True): Process page images with page-based windows
    """
    from agentic.knowledge.indexing import Doc2JSONAlgorithm
    from agentic.knowledge.model_config import (
        DOC2JSON_DEFAULT_PAGES_PER_WINDOW,
        DOC2JSON_DEFAULT_WINDOW_OVERLAP,
        DOC2JSON_DEFAULT_WINDOW_SIZE,
        DOC2JSON_EMBEDDING_MODEL,
        DOC2JSON_EXTRACTION_MODEL,
        DOC2JSON_USE_IMAGES,
    )
    from agentic.knowledge.models import IndexingConfig

    extraction_model = indexing_config.get("extraction_model", DOC2JSON_EXTRACTION_MODEL)
    embedding_model = indexing_config.get("embedding_model", DOC2JSON_EMBEDDING_MODEL)
    window_size = indexing_config.get("window_size", DOC2JSON_DEFAULT_WINDOW_SIZE)
    window_overlap = indexing_config.get("window_overlap", DOC2JSON_DEFAULT_WINDOW_OVERLAP)
    json_schema = indexing_config.get("json_schema", {})
    use_images = indexing_config.get("use_images", DOC2JSON_USE_IMAGES)
    pages_per_window = indexing_config.get("pages_per_window", DOC2JSON_DEFAULT_PAGES_PER_WINDOW)

    if use_images and page_images:
        logger.info(
            f"Doc2JSON indexing (IMAGE MODE): model={extraction_model}, "
            f"embedding={embedding_model}, pages_per_window={pages_per_window}, "
            f"total_pages={len(page_images)}"
        )
    else:
        logger.info(
            f"Doc2JSON indexing (TEXT MODE): model={extraction_model}, "
            f"embedding={embedding_model}, window_size={window_size}, "
            f"window_overlap={window_overlap}, content_len={len(content)}"
        )

    if not json_schema:
        raise ValueError("doc2json strategy requires a json_schema in indexing_config")

    # Build config extra - include page_images if using image mode
    extra = {
        "extraction_model": extraction_model,
        "embedding_model": embedding_model,
        "json_schema": json_schema,
        "use_images": use_images,
    }
    if "reasoning_effort" in indexing_config:
        extra["reasoning_effort"] = indexing_config["reasoning_effort"]

    if use_images and page_images:
        extra["page_images"] = page_images
        extra["pages_per_window"] = pages_per_window
    else:
        extra["window_size"] = window_size
        extra["window_overlap"] = window_overlap

    config = IndexingConfig(
        strategy="doc2json",
        extra=extra,
    )

    algorithm = Doc2JSONAlgorithm()
    result = await algorithm.aindex(
        content=content,
        config=config,
        source_id=source_id,
    )

    logger.info(
        f"Doc2JSON extraction complete: {result.stats.get('window_count', 0)} windows, "
        f"{result.stats.get('input_tokens', 0)} input tokens"
    )

    # Store in doc2json_documents table
    store = Doc2JSONStore(
        db_session=db.session,
        knowledge_base_id=kb_id,
    )

    doc_id = store.store_doc2json_document(
        indexed_source_id=indexed_source_id,
        source_id=source_id,
        summary=result.combined_summary,
        summary_embedding=result.combined_summary_embedding,
        extracted_json=result.extracted_json,
        json_schema=result.json_schema,
        window_summaries=result.window_summaries,
        extraction_model=extraction_model,
        embedding_model=embedding_model,
        summary_tokens=result.stats.get("summary_tokens"),
        input_tokens=result.stats.get("input_tokens"),
        window_size=window_size,
        window_overlap=window_overlap,
        window_count=result.stats.get("window_count"),
    )

    # Ensure hnsw index exists for this embedding dimension
    if result.combined_summary_embedding:
        from ..services.base_vector_store import ensure_embedding_index

        ensure_embedding_index(db.session, AI_SCHEMA, len(result.combined_summary_embedding))

    stats = {
        "artifact_count": 1,
        "total_chars": len(content) if content else 0,
        "summary_chars": len(result.combined_summary),
        "summary_tokens": result.stats.get("summary_tokens", 0),
        "input_tokens": result.stats.get("input_tokens", 0),
        "window_count": result.stats.get("window_count", 0),
        "embedding_dim": len(result.combined_summary_embedding)
        if result.combined_summary_embedding
        else 0,
        "extraction_model": extraction_model,
        "embedding_model": embedding_model,
        "strategy": "doc2json",
        "doc_id": doc_id,
    }

    return stats


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
@billing.task_context
def index_source(
    self,
    knowledge_base_id: str,
    source_id: str,
    indexed_source_id: str | None = None,
    provider_keys: dict[str, str] | None = None,
    billing_idempotency_key: str | None = None,
    billing_org_id: str | None = None,
    billing_project_id: str | None = None,
    idempotency_action: str | None = None,
    idempotency_parts: list | None = None,
):
    """
    Index a source into a knowledge base.

    Args:
        knowledge_base_id: The knowledge base UUID
        source_id: The source UUID
        indexed_source_id: Optional existing indexed_source ID (for re-indexing)
        provider_keys: Optional dict of provider→api_key for LLM/embedding calls
        billing_idempotency_key, billing_org_id, billing_project_id: VESTIGIAL —
            retained for deploy-compat only. Billing now flows through the
            billing port, which derives identity (org/project) from the adapter
            and builds the idempotency key from the threaded inputs below. An
            in-flight task enqueued before the port migration still carries
            these; keeping them avoids a TypeError on a cross-deploy retry.
        idempotency_action, idempotency_parts: caller-supplied inputs for the
            indexing charge's idempotency KEY. The two caller families use
            DIFFERENT schemes and index_source cannot recompute the key from
            its own args, so they are threaded here (org is added by the
            adapter): the single-index route keys on (indexing_action,
            indexed_source_id); the reindex fan-out / batch / watchdog paths key
            on the literal "indexing" namespace (+ indexed_source_id [+
            source_id]). The BILLED action is always the strategy-resolved
            indexing_<strategy>; only the KEY action may differ (the split).
            NEVER derived from self.request.id (which changes per retry →
            double-charge).

    Returns:
        Dict with indexing results or error info
    """
    import time as _time

    task_id = self.request.id
    _task_t0 = _time.monotonic()
    logger.info(
        f"Starting indexing task {task_id} for source {source_id} into KB {knowledge_base_id}"
    )

    try:
        kb = get_knowledge_base(knowledge_base_id)
        if not kb:
            error_msg = f"Knowledge base {knowledge_base_id} not found — it may have been deleted from ai.knowledge_bases. Verify it exists before re-indexing."
            logger.error(error_msg)
            if indexed_source_id:
                _mark_indexed_source_failed(indexed_source_id, error_msg)
            return {"status": "error", "error": error_msg}

        indexing_config = kb["indexing_config"]

        source = get_source(source_id)
        if not source:
            error_msg = f"Source {source_id} not found — it may have been deleted from ai.sources. Remove it from the knowledge base and re-add it."
            logger.error(error_msg)
            if indexed_source_id:
                _mark_indexed_source_failed(indexed_source_id, error_msg)
            return {"status": "error", "error": error_msg}

        if source["extraction_status"] != "extracted":
            error_msg = (
                f"Source must be extracted first. Current status: {source['extraction_status']}"
            )
            logger.error(
                f"Source {source_id} not extracted yet (status: {source['extraction_status']})"
            )
            if indexed_source_id:
                _mark_indexed_source_failed(indexed_source_id, error_msg)
            return {
                "status": "error",
                "error": error_msg,
            }

        strategy = indexing_config.get("strategy", "chunk_embed")
        strategy_changed = False

        # Early exit if already cancelled (handles race where cancel arrives before task starts)
        if indexed_source_id:
            current_status = db.session.execute(
                text(f'SELECT index_status FROM "{AI_SCHEMA}".indexed_sources WHERE id = :id'),
                {"id": indexed_source_id},
            ).scalar()
            if current_status == "cancelled":
                logger.info(f"Indexed source {indexed_source_id} already cancelled, skipping")
                return {"status": "cancelled", "source_id": source_id}

        if indexed_source_id:
            # Read old snapshot BEFORE overwriting to detect strategy changes
            old_snapshot = _get_indexed_source_snapshot(indexed_source_id)
            old_strategy = (old_snapshot or {}).get("strategy", "chunk_embed")
            strategy_changed = old_strategy != strategy

            update_indexed_source_config_snapshot(indexed_source_id, indexing_config)

            # Mark as indexing BEFORE deleting old artifacts so the UI
            # doesn't show "indexed" while content is being cleared.
            update_indexed_source_status(indexed_source_id, "indexing", celery_task_id=task_id)

            # Clean up all embeddings for this indexed source FIRST
            # (prevents search queries from finding embeddings with missing content)
            db.session.execute(
                text(f"""
                    DELETE FROM "{AI_SCHEMA}".embeddings
                    WHERE indexed_source_id = :indexed_source_id
                """),
                {"indexed_source_id": indexed_source_id},
            )
            db.session.commit()

            # Delete artifacts from ALL strategy tables unconditionally.
            # Each call is a no-op (0 rows deleted) if no artifacts exist for that type.

            # First, query IDs before deletion for sparse index cleanup
            from ..services.sparse_retrieval import SparseIndexStore

            chunk_ids_to_remove = [
                str(r[0])
                for r in db.session.execute(
                    text(f'SELECT id FROM "{AI_SCHEMA}".chunks WHERE indexed_source_id = :is_id'),
                    {"is_id": indexed_source_id},
                ).fetchall()
            ]
            fd_ids_to_remove = [
                str(r[0])
                for r in db.session.execute(
                    text(
                        f'SELECT id FROM "{AI_SCHEMA}".full_documents WHERE indexed_source_id = :is_id'
                    ),
                    {"is_id": indexed_source_id},
                ).fetchall()
            ]
            gi_ids_to_remove = [
                str(r[0])
                for r in db.session.execute(
                    text(
                        f'SELECT id FROM "{AI_SCHEMA}".graph_index_nodes WHERE indexed_source_id = :is_id'
                    ),
                    {"is_id": indexed_source_id},
                ).fetchall()
            ]

            store = PgVectorKnowledgeStore(
                db_session=db.session,
                knowledge_base_id=knowledge_base_id,
            )
            asyncio.run(store.delete_chunks(indexed_source_id))

            pi_store = PageIndexStore(
                db_session=db.session,
                knowledge_base_id=knowledge_base_id,
            )
            pi_store.delete_by_indexed_source(indexed_source_id)

            fd_store = FullDocumentStore(
                db_session=db.session,
                knowledge_base_id=knowledge_base_id,
                storage=get_storage(),
            )
            fd_store.delete_by_indexed_source(indexed_source_id)

            gi_store = GraphIndexStore(
                db_session=db.session,
                knowledge_base_id=knowledge_base_id,
            )
            gi_store.delete_by_indexed_source(indexed_source_id)

            d2j_store = Doc2JSONStore(
                db_session=db.session,
                knowledge_base_id=knowledge_base_id,
            )
            d2j_store.delete_by_indexed_source(indexed_source_id)
            # Remove deleted items from sparse indexes
            if _should_build_bm25_now(knowledge_base_id):
                sparse_store = SparseIndexStore(knowledge_base_id=knowledge_base_id)
                if chunk_ids_to_remove:
                    sparse_store.remove_and_save(chunk_ids_to_remove, item_table="chunks")
                if fd_ids_to_remove:
                    sparse_store.remove_and_save(fd_ids_to_remove, item_table="full_documents")
                if gi_ids_to_remove:
                    sparse_store.remove_and_save(gi_ids_to_remove, item_table="graph_index_nodes")

            logger.info(
                "Cleared all artifact tables for re-indexing (old_strategy=%s, new_strategy=%s)",
                old_strategy,
                strategy,
            )
        else:
            # Get or create indexed_source record
            result = db.session.execute(
                text(f"""
                    SELECT id FROM "{AI_SCHEMA}".indexed_sources
                    WHERE knowledge_base_id = :kb_id AND source_id = :source_id
                """),
                {"kb_id": knowledge_base_id, "source_id": source_id},
            )
            row = result.fetchone()
            if row:
                indexed_source_id = str(row[0])
            else:
                logger.error("indexed_source_id not provided and record not found")
                return {"status": "error", "error": "indexed_source record not found"}

        update_indexed_source_status(indexed_source_id, "indexing", celery_task_id=task_id)

        storage = get_storage()
        content = get_text_derivative_content(storage, source)

        use_images = strategy == "doc2json" and indexing_config.get("use_images", False)

        if not content:
            if use_images:
                logger.info(
                    "No text derivative found — proceeding in image-only mode "
                    "for doc2json strategy (source_id=%s)",
                    source_id,
                )
                content = ""
            else:
                error_msg = "No text derivative found for source"
                logger.error(error_msg)
                update_indexed_source_status(indexed_source_id, "failed", error_msg)
                return {"status": "error", "error": error_msg}

        if content:
            logger.info(f"Loaded text content: {len(content)} chars")

        # Extract page_texts from derivative metadata (available for PDF sources)
        page_texts = get_page_texts_from_derivative(source, storage=storage)

        # Set the cost accumulator on this thread's context BEFORE the
        # strategy dispatch so each asyncio.run() snapshot below inherits
        # the same accumulator object.
        acc = init_accumulator()

        if strategy == "page_index":
            stats = asyncio.run(
                run_page_index_indexing(
                    kb_id=knowledge_base_id,
                    indexed_source_id=indexed_source_id,
                    source_id=source_id,
                    content=content,
                    indexing_config=indexing_config,
                    source_name=source["name"],
                    page_texts=page_texts,
                )
            )
        elif strategy == "full_document":
            stats = asyncio.run(
                run_full_document_indexing(
                    kb_id=knowledge_base_id,
                    indexed_source_id=indexed_source_id,
                    source_id=source_id,
                    content=content,
                    indexing_config=indexing_config,
                    provider_keys=provider_keys,
                )
            )
        elif strategy == "graph_index":
            stats = asyncio.run(
                run_graph_index_indexing(
                    kb_id=knowledge_base_id,
                    indexed_source_id=indexed_source_id,
                    source_id=source_id,
                    content=content,
                    indexing_config=indexing_config,
                    source_name=source["name"],
                    page_texts=page_texts,
                    source=source,
                    provider_keys=provider_keys,
                )
            )
        elif strategy == "doc2json":
            # Fetch page images if use_images mode is enabled
            page_images = None
            use_images = indexing_config.get("use_images", False)
            if use_images:
                page_images = get_page_images_from_derivative(source, storage=storage)
                if not page_images:
                    logger.warning(
                        "use_images=True but no page images found for source %s, "
                        "falling back to text mode",
                        source_id,
                    )

            stats = asyncio.run(
                run_doc2json_indexing(
                    kb_id=knowledge_base_id,
                    indexed_source_id=indexed_source_id,
                    source_id=source_id,
                    content=content,
                    indexing_config=indexing_config,
                    page_images=page_images,
                )
            )
        else:  # chunk_embed (default)
            auto_meta = source.get("auto_metadata", {})
            extraction_method = auto_meta.get("extraction_method")
            # For -via-pdf wrappers (txt, docx), use the underlying PDF
            # extraction method so the separator matches what was used to
            # join pages during extraction.
            if extraction_method and extraction_method.endswith("-via-pdf"):
                extraction_method = auto_meta.get("pdf_extraction_method", extraction_method)
            stats = asyncio.run(
                run_indexing(
                    kb_id=knowledge_base_id,
                    indexed_source_id=indexed_source_id,
                    source_id=source_id,
                    content=content,
                    indexing_config=indexing_config,
                    page_texts=page_texts,
                    extraction_method=extraction_method,
                    provider_keys=provider_keys,
                )
            )

        stats["llm_costs"] = acc.to_dict()
        update_indexed_source_result(indexed_source_id, stats)

        # Bill the indexing action with the actual 1k-token quantity. The
        # billing-configured check now lives in the adapter (no-op when
        # unconfigured), so this is unconditional. The BILLED action is the
        # strategy-resolved indexing_<strategy>; the KEY action is whatever the
        # caller threaded (== billed action for single-index, literal "indexing"
        # for reindex/batch/watchdog — the split), reproducing the pre-port key.
        indexing_action = _resolve_indexing_action(strategy)
        actual_quantity = _quantity_from_stats(stats)
        # billing.charge never raises; ChargeOutcome reports outcome. A
        # post-success 402 is bounded over-serve per spec line 54.
        billing.charge(
            action=indexing_action,
            idempotency_action=idempotency_action,
            idempotency_parts=tuple(idempotency_parts or ()),
            ref_type="indexing",
            ref_id=str(indexed_source_id),
            quantity=actual_quantity,
            metadata={"strategy": strategy, "source_id": source_id},
        )

        # Auto-enrich if KB has an enrichment config
        try:
            from .enrichment import enrich_knowledge_base as _enrich_kb

            _enrich_config = db.session.execute(
                text(
                    f'SELECT id FROM "{AI_SCHEMA}".enrichment_configs '
                    f"WHERE knowledge_base_id = :kb_id AND status != 'enriching'"
                ),
                {"kb_id": knowledge_base_id},
            ).fetchone()
            if _enrich_config:
                # Full enrichment when strategy changed — TRUNCATEs stale metadata rows.
                # Incremental otherwise — only enriches newly created items.
                use_incremental = not strategy_changed
                # No idempotency key threaded: enrichment's per-batch charging is
                # gated by the billing port's ctx (per_batch_callback enabled by
                # default, ctx-gated when billing is off) — the old
                # child_enrich_idem sentinel was only a proxy for "billing
                # active", which the adapter now checks directly.
                _enrich_kb.delay(
                    knowledge_base_id,
                    incremental=use_incremental,
                )
                logger.info(
                    "Triggered %s enrichment for KB %s",
                    "incremental" if use_incremental else "full",
                    knowledge_base_id,
                )
        except Exception:
            logger.warning(
                "Failed to trigger auto-enrichment for KB %s",
                knowledge_base_id,
                exc_info=True,
            )

        if strategy == "page_index":
            artifact_desc = (
                f"{stats.get('node_count', 0)} nodes, {stats.get('section_count', 0)} sections"
            )
        elif strategy == "graph_index":
            artifact_desc = (
                f"{stats.get('node_count', 0)} nodes, {stats.get('section_count', 0)} sections"
            )
        elif strategy == "full_document":
            artifact_desc = f"1 full document ({stats.get('summary_tokens', 0)} summary tokens)"
        elif strategy == "doc2json":
            artifact_desc = f"1 doc2json ({stats.get('window_count', 0)} windows, {stats.get('summary_tokens', 0)} summary tokens)"
        else:
            artifact_desc = f"{stats['artifact_count']} chunks"
        _task_total = _time.monotonic() - _task_t0
        logger.info(
            "index_source_timing task_id=%s total=%.2f strategy=%s",
            task_id,
            _task_total,
            strategy,
        )
        logger.info(f"Indexing complete for source {source_id}: {artifact_desc} created")

        return {
            "status": "success",
            "indexed_source_id": indexed_source_id,
            "source_id": source_id,
            "knowledge_base_id": knowledge_base_id,
            "stats": stats,
        }

    except SoftTimeLimitExceeded:
        logger.warning(
            f"Indexing cancelled/timed out for source {source_id} in knowledge base {knowledge_base_id}"
        )
        db.session.rollback()
        if indexed_source_id:
            # Check if cancelled by user (cancel endpoint sets status before revoking)
            current = db.session.execute(
                text(f'SELECT index_status FROM "{AI_SCHEMA}".indexed_sources WHERE id = :id'),
                {"id": indexed_source_id},
            ).scalar()
            if current == "cancelled":
                return {"status": "cancelled", "source_id": source_id}
            update_indexed_source_status(indexed_source_id, "failed", "Indexing timed out", task_id)
        return {
            "status": "error",
            "source_id": source_id,
            "knowledge_base_id": knowledge_base_id,
            "error": "Indexing timed out",
        }

    except StorageError as e:
        logger.error(f"Storage error during indexing: {e}")
        db.session.rollback()  # Discard partial indexing data
        if indexed_source_id:
            update_indexed_source_status(
                indexed_source_id, "failed", traceback.format_exc(), task_id
            )
        raise self.retry(exc=e) from e

    except Exception:
        logger.error(f"Indexing failed for source {source_id}", exc_info=True)
        db.session.rollback()  # Discard partial indexing data
        if indexed_source_id:
            update_indexed_source_status(
                indexed_source_id, "failed", traceback.format_exc(), task_id
            )

        return {
            "status": "error",
            "source_id": source_id,
            "knowledge_base_id": knowledge_base_id,
            "error": traceback.format_exc(),
        }


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
@billing.no_billing_context
def reindex_knowledge_base(
    self,
    knowledge_base_id: str,
    provider_keys: dict[str, str] | None = None,
    billing_org_id: str | None = None,
    billing_project_id: str | None = None,
):
    """
    Re-index all sources in a knowledge base.

    Args:
        knowledge_base_id: The knowledge base UUID
        provider_keys: Optional dict of provider→api_key to forward to sub-tasks
        billing_org_id, billing_project_id: VESTIGIAL — retained for deploy-compat
            only. Each child ``index_source`` now receives the key inputs
            (``idempotency_action`` / ``idempotency_parts``) directly; identity
            comes from the billing adapter, so these are no longer used.

    Returns:
        Dict with summary of indexing results
    """
    task_id = self.request.id
    logger.info(f"Starting full re-index task {task_id} for KB {knowledge_base_id}")

    try:
        result = db.session.execute(
            text(f"""
                SELECT id, source_id
                FROM "{AI_SCHEMA}".indexed_sources
                WHERE knowledge_base_id = :kb_id
            """),
            {"kb_id": knowledge_base_id},
        )

        indexed_sources = [(str(row[0]), str(row[1])) for row in result]

        if not indexed_sources:
            logger.info(f"No sources to re-index for KB {knowledge_base_id}")
            return {"status": "success", "reindexed": 0}

        logger.info(f"Re-indexing {len(indexed_sources)} sources")

        for idx_source_id, source_id in indexed_sources:
            # Key each child on the literal "indexing" namespace + the
            # (idx_source_id, source_id) pair — deterministic per indexed_source
            # so retries of THIS child share the same key (idempotent), while a
            # fresh reindex run bills again. The adapter adds org_id; the child
            # BILLS the strategy-resolved indexing_<strategy> (the split).
            index_source.delay(
                knowledge_base_id=knowledge_base_id,
                source_id=source_id,
                indexed_source_id=idx_source_id,
                provider_keys=provider_keys,
                idempotency_action="indexing",
                idempotency_parts=[idx_source_id, source_id],
            )

        return {
            "status": "success",
            "queued": len(indexed_sources),
            "knowledge_base_id": knowledge_base_id,
        }

    except Exception:
        logger.error(f"Re-index failed for KB {knowledge_base_id}", exc_info=True)
        return {
            "status": "error",
            "knowledge_base_id": knowledge_base_id,
            "error": traceback.format_exc(),
        }


def _restore_indexed_status(knowledge_base_id: str, indexed_source_id: str | None) -> None:
    """Restore index_status to 'indexed' for sources set to 'indexing'."""
    if indexed_source_id:
        update_indexed_source_status(indexed_source_id, "indexed")
    else:
        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".indexed_sources
                SET index_status = 'indexed', celery_task_id = NULL
                WHERE knowledge_base_id = :kb_id AND index_status = 'indexing'
            """),
            {"kb_id": knowledge_base_id},
        )
        db.session.commit()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)
@billing.task_context
def reenrich_graph_references(
    self,
    knowledge_base_id: str,
    retry_failed: bool = False,
    indexed_source_id: str | None = None,
    provider_keys: dict[str, str] | None = None,
    billing_idempotency_key: str | None = None,
    billing_org_id: str | None = None,
    billing_project_id: str | None = None,
):
    """Re-run graph reference enrichment (Stage 2) and re-embed (Stage 3) for a graph_index KB.

    This is a lighter-weight alternative to full reindex when only enrichment
    partially failed. Skips Stage 1 (ToC building) entirely.

    Args:
        knowledge_base_id: The knowledge base UUID.
        retry_failed: When True, only re-enrich nodes with enrichment_error set.
        indexed_source_id: When set, only process nodes belonging to this source.
        provider_keys: Unused after #437 (kept for backward compatibility).
        billing_idempotency_key, billing_org_id, billing_project_id: VESTIGIAL —
            retained for deploy-compat only. Per-batch graph charging now flows
            through the billing port (``billing.per_batch_callback``), which is
            enabled by default and ctx-gated by the adapter; these params are no
            longer read.
    """
    task_id = self.request.id
    logger.info(
        "Starting graph re-enrichment task %s for KB %s (retry_failed=%s)",
        task_id,
        knowledge_base_id,
        retry_failed,
    )

    try:
        kb = get_knowledge_base(knowledge_base_id)
        if not kb:
            _restore_indexed_status(knowledge_base_id, indexed_source_id)
            return {"status": "error", "error": "Knowledge base not found"}

        indexing_config = kb["indexing_config"]
        strategy = indexing_config.get("strategy", "chunk_embed")
        if strategy != "graph_index":
            _restore_indexed_status(knowledge_base_id, indexed_source_id)
            return {
                "status": "error",
                "error": f"Re-enrichment is only supported for graph_index strategy, got '{strategy}'",
            }

        from agentic.knowledge.embedder import LiteLLMEmbedder
        from agentic.knowledge.model_config import (
            GRAPHINDEX_EMBEDDING_MODEL,
            GRAPHINDEX_ENRICHMENT_MODEL,
        )

        from ..services.base_vector_store import ensure_embedding_index
        from ..services.graph_enricher import enrich_referenced_nodes

        enrichment_model = indexing_config.get("enrichment_model", GRAPHINDEX_ENRICHMENT_MODEL)
        embedding_model = indexing_config.get("embedding_model", GRAPHINDEX_EMBEDDING_MODEL)

        store = GraphIndexStore(
            db_session=db.session,
            knowledge_base_id=knowledge_base_id,
        )

        tocs = store.get_tocs()
        if not tocs:
            _restore_indexed_status(knowledge_base_id, indexed_source_id)
            return {
                "status": "success",
                "toc_count": 0,
                "reference_count": 0,
                "nodes_embedded": 0,
                "errors": [],
            }

        graph_batch_cb = billing.per_batch_callback(
            config_id=str(indexed_source_id) if indexed_source_id else knowledge_base_id,
            action="indexing_graphindex",
        )

        # C1 (#445): collapse per-ToC asyncio.run calls into ONE event loop.
        # The old pattern (2 asyncio.run per ToC) created a new event loop per
        # iteration, which churns LiteLLM's global LoggingWorker queue — the
        # inferred OOM vector for healthy-balance large runs.
        async def _run_all_tocs() -> tuple[int, int, list[str]]:
            refs_total, embedded_total, errs = 0, 0, []

            # C2 (#445): propagate 402-abort across ToCs. When graph_batch_cb
            # returns "abort" (per-batch 402), enrich_referenced_nodes breaks its
            # own internal batch loop but returns normally — _run_all_tocs would
            # otherwise continue into Stage 3 re-embed for the current ToC and
            # then proceed through all remaining ToCs, each issuing another 402.
            # Wrap graph_batch_cb in a closure that records the abort decision so
            # we can break after Stage 2 completes for the triggering ToC.
            aborted = {"flag": False}
            _on_batch = None
            if graph_batch_cb is not None:

                def _on_batch(batch_ok, batch_item_ids):
                    decision = graph_batch_cb(batch_ok, batch_item_ids)
                    if decision == "abort":
                        aborted["flag"] = True
                    return decision

            for toc in tocs:
                toc_id = toc["id"]

                # === Stage 2: Reference enrichment ===
                ref_results, enrichment_errors = await enrich_referenced_nodes(
                    db_session=db.session,
                    knowledge_base_id=knowledge_base_id,
                    toc_id=toc_id,
                    model=enrichment_model,
                    retry_failed=retry_failed,
                    indexed_source_id=indexed_source_id,
                    api_key=None,
                    reasoning_effort=indexing_config.get("enrichment_reasoning_effort"),
                    on_batch_complete=_on_batch,
                )
                refs_total += sum(len(v) for v in ref_results.values())
                errs.extend(enrichment_errors)

                # If billing returned 402, stop the job: skip Stage 3 for this
                # ToC and do not process any remaining ToCs. Already-committed
                # enrichment results for this ToC remain committed.
                if aborted["flag"]:
                    logger.warning(
                        "Graph re-enrichment aborted (billing 402) — stopping after toc %s; "
                        "skipping re-embed and remaining ToCs for kb=%s",
                        toc_id,
                        knowledge_base_id,
                    )
                    break

                # === Stage 3: Re-embed nodes (changed references affect embedding text) ===
                all_nodes = store.get_all_nodes_for_toc(toc_id)
                if indexed_source_id:
                    all_nodes = [
                        n for n in all_nodes if str(n.get("indexed_source_id")) == indexed_source_id
                    ]
                if not all_nodes:
                    continue

                node_title_map = {n["node_id"]: n.get("title", "") for n in all_nodes}
                embedder = LiteLLMEmbedder(model=embedding_model)

                embed_inputs = []
                node_ids_ordered = []
                for node in all_nodes:
                    meta = node.get("meta") or {}
                    title = node.get("title", "")
                    summary = meta.get("summary", "")
                    referenced = meta.get("referenced_nodes", [])

                    parts = []
                    if title:
                        parts.append(title)
                    if summary:
                        parts.append(summary)
                    if referenced:
                        ref_titles = [node_title_map.get(r, r) for r in referenced]
                        parts.append(f"References: {', '.join(ref_titles)}")

                    embed_text = "\n".join(parts) if parts else (node.get("text") or "")[:500]
                    embed_inputs.append(embed_text)
                    node_ids_ordered.append(node["node_id"])

                embeddings = await embedder.aembed_batch(embed_inputs)

                if len(embeddings) != len(node_ids_ordered):
                    raise ValueError(
                        f"Embedding count mismatch: got {len(embeddings)} embeddings "
                        f"for {len(node_ids_ordered)} nodes"
                    )

                for nid, emb in zip(node_ids_ordered, embeddings):
                    store.update_node_embedding(toc_id, nid, emb, embedding_model=embedding_model)

                if embeddings:
                    ensure_embedding_index(db.session, AI_SCHEMA, len(embeddings[0]))

                embedded_total += len(embeddings)

            return refs_total, embedded_total, errs

        total_refs, total_embedded, all_errors = asyncio.run(_run_all_tocs())

        # C4 (#445): read-only probe of LiteLLM's global LoggingWorker queue —
        # the inferred OOM vector. Converts "inferred" -> "observable" at ~0 risk.
        # Fully guarded: a renamed/missing internal must never fail the task.
        try:
            from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

            _q = getattr(GLOBAL_LOGGING_WORKER, "_queue", None)
            _max = getattr(GLOBAL_LOGGING_WORKER, "max_queue_size", 0)
            if _q is not None and _max and _q.qsize() > 0.5 * _max:
                logger.warning(
                    "LITELLM_LOGGING_WORKER_QUEUE_HIGH qsize=%d maxsize=%d kb=%s — "
                    "candidate #445 OOM vector; if this fires on a healthy-balance run, "
                    "escalate to the D2 LoggingWorker patch.",
                    _q.qsize(),
                    _max,
                    knowledge_base_id,
                    extra={"alert": "litellm_logging_worker_queue_high"},
                )
        except Exception:
            pass

        db.session.commit()

        # Update config snapshot and restore status to "indexed"
        if indexed_source_id:
            update_indexed_source_config_snapshot(indexed_source_id, indexing_config)
            update_indexed_source_status(indexed_source_id, "indexed")
        else:
            # All sources were re-enriched — update all indexed_sources for this KB
            source_rows = db.session.execute(
                text(
                    f'SELECT id FROM "{AI_SCHEMA}".indexed_sources WHERE knowledge_base_id = :kb_id'
                ),
                {"kb_id": knowledge_base_id},
            ).fetchall()
            for row in source_rows:
                update_indexed_source_config_snapshot(str(row[0]), indexing_config)
                update_indexed_source_status(str(row[0]), "indexed")

        logger.info(
            "Graph re-enrichment complete for KB %s: %d tocs, %d refs, %d embedded, %d errors",
            knowledge_base_id,
            len(tocs),
            total_refs,
            total_embedded,
            len(all_errors),
        )

        return {
            "status": "success",
            "knowledge_base_id": knowledge_base_id,
            "toc_count": len(tocs),
            "reference_count": total_refs,
            "nodes_embedded": total_embedded,
            "errors": all_errors,
        }

    except StorageError as e:
        logger.warning(
            "Storage error during re-enrichment for KB %s, retrying",
            knowledge_base_id,
            exc_info=True,
        )
        db.session.rollback()
        raise self.retry(exc=e) from e

    except Exception:
        logger.error(
            "Graph re-enrichment failed for KB %s",
            knowledge_base_id,
            exc_info=True,
        )
        db.session.rollback()
        # Restore status so sources don't stay stuck at "indexing"
        try:
            _restore_indexed_status(knowledge_base_id, indexed_source_id)
        except Exception:
            logger.warning(
                "Failed to restore index_status after re-enrichment error for KB %s",
                knowledge_base_id,
                exc_info=True,
            )
        return {
            "status": "error",
            "knowledge_base_id": knowledge_base_id,
            "error": traceback.format_exc(),
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=300)
@billing.no_billing_context
def build_bm25_for_kb(self, kb_id: str) -> dict:
    """One-shot BM25 rebuild for a KB.

    Walks the appropriate item table for the KB's strategy, accumulates
    documents in-memory, and atomically replaces the on-disk BM25 index
    via SparseIndexStore.rebuild_from_scratch.

    Note: all documents are accumulated in-memory before the rebuild
    call. At current corpus sizes (~200k items per KB) this is
    acceptable; for million-item KBs a streaming rebuild path will
    be needed (tracked separately).
    """
    kb = _fetch_kb_for_bm25_build(kb_id)
    strategy = kb["indexing_config"].get("strategy")
    item_table = _STRATEGY_TO_ITEM_TABLE.get(strategy)
    if item_table is None:
        raise ValueError(
            f"Strategy '{strategy}' for KB {kb_id} does not support BM25 rebuild "
            f"(supported: {sorted(_STRATEGY_TO_ITEM_TABLE)})"
        )

    documents: list[str] = []
    item_ids: list[str] = []
    for batch in _iter_items_for_kb_bm25(kb_id, item_table, batch_size=10_000):
        for item in batch:
            documents.append(item["text"])
            item_ids.append(item["id"])

    sparse_store = SparseIndexStore(knowledge_base_id=kb_id)
    sparse_store.rebuild_from_scratch(
        documents=documents,
        item_ids=item_ids,
        item_table=item_table,
    )
    logger.info(
        "build_bm25_for_kb done: kb=%s item_table=%s item_count=%d",
        kb_id,
        item_table,
        len(item_ids),
    )
    return {"item_table": item_table, "item_count": len(item_ids)}
