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
# Source of truth for both the per-source build path (tasks/indexing.py) and
# the bm25_status computation (routes/knowledge_bases.py).
STRATEGY_TO_BM25_ITEM_TABLE: dict[str, str] = {
    "chunk_embed": "chunks",
    "page_index": "full_documents",
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
