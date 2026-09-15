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
# Source of truth for the per-source build path (tasks/indexing.py), the
# bm25_status computation (routes/knowledge_bases.py) and the pg_search index
# helpers (services/pg_bm25_index.py).
#
# ``page_index`` is deliberately absent: it is compatible with tree_search
# only, and its text lives in page_index_toc/page_index_nodes, so it has no
# keyword index to build. The table it used to be mapped to, full_documents,
# belongs to the ``full_document`` strategy.
STRATEGY_TO_BM25_ITEM_TABLE: dict[str, str] = {
    "chunk_embed": "chunks",
    "full_document": "full_documents",
    "graph_index": "graph_index_nodes",
    "doc2json": "doc2json_documents",
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
