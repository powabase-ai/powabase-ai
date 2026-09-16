"""Sparse retrieval package for BM25s-based keyword search."""

from .base import SparseIndexManager, SparseRetriever, SparseSearchResult
from .bm25_index import BM25IndexManager, BM25Retriever
from .config import (
    BM25_STEMMER_LANGUAGE,
    BM25_STOPWORDS,
    BM25_VARIANT,
    SPARSE_INDEX_BASE_PATH,
    USE_LLM_ENRICHMENT_DEFAULT,
)
from .query_context import QueryContextBuilder, build_search_query
from .sparse_index_store import SparseIndexStore

# Maps KB indexing strategy → the item_table that holds BM25-searchable text.
# Read by the whole-KB rebuild (tasks/indexing.py's build_bm25_for_kb) and by
# the bm25_status computation (routes/knowledge_bases.py). The per-source
# incremental paths in tasks/indexing.py do NOT read this map — each hardcodes
# the item table it writes — so a strategy is only safe to add here once it has
# those per-source add/remove callsites too.
#
# A strategy absent from this map cannot get an index at all: the rebuild
# refuses it and keyword search stays on the tsvector fallback in
# full_text_search. Deliberately absent:
#   page_index — compatible with tree_search only, and its text lives in
#     page_index_toc/page_index_nodes, so there is no item table to index. The
#     table it used to be mapped to, full_documents, belongs to full_document.
#   doc2json — has no sparse-index maintenance: run_doc2json_indexing never
#     calls SparseIndexStore and index_source's removal block does not cover
#     doc2json_documents. An index built here would freeze at build time —
#     later documents invisible to keyword search, deleted ones still scoring —
#     so doc2json stays on the slow-but-correct fallback until that lands.
STRATEGY_TO_BM25_ITEM_TABLE: dict[str, str] = {
    "chunk_embed": "chunks",
    "full_document": "full_documents",
    "graph_index": "graph_index_nodes",
}

__all__ = [
    # Base abstractions
    "SparseSearchResult",
    "SparseRetriever",
    "SparseIndexManager",
    # BM25 implementation
    "BM25IndexManager",
    "BM25Retriever",
    # Storage
    "SparseIndexStore",
    # Query context
    "QueryContextBuilder",
    "build_search_query",
    # Config
    "SPARSE_INDEX_BASE_PATH",
    "BM25_VARIANT",
    "BM25_STEMMER_LANGUAGE",
    "BM25_STOPWORDS",
    "USE_LLM_ENRICHMENT_DEFAULT",
    # Strategy mapping
    "STRATEGY_TO_BM25_ITEM_TABLE",
]
