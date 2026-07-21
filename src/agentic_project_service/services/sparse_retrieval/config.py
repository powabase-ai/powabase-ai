"""Configuration for sparse retrieval (BM25s)."""

import os

# Storage paths
SPARSE_INDEX_BASE_PATH = os.environ.get("SPARSE_INDEX_PATH", "/data/sparse_indexes")

# Cache settings (LRU eviction to prevent unbounded memory growth)
SPARSE_INDEX_CACHE_SIZE = int(os.environ.get("SPARSE_INDEX_CACHE_SIZE", "100"))

# BM25 settings
BM25_VARIANT = os.environ.get("BM25_VARIANT", "robertson")
BM25_STEMMER_LANGUAGE = os.environ.get("BM25_STEMMER_LANG", "english")
BM25_STOPWORDS = os.environ.get("BM25_STOPWORDS", "en")

# Query context settings
QUERY_CONTEXT_MAX_HISTORY = int(os.environ.get("QUERY_CONTEXT_MAX_HISTORY", "6"))
QUERY_TERM_WEIGHT = float(os.environ.get("QUERY_TERM_WEIGHT", "2.0"))
USE_LLM_ENRICHMENT_DEFAULT = os.environ.get("USE_LLM_ENRICHMENT", "false").lower() == "true"

# Message truncation (avoid noise from very long messages)
MAX_MESSAGE_CHARS = int(os.environ.get("SPARSE_MAX_MESSAGE_CHARS", "500"))
