"""KB search configuration — centralized runtime constants.

All tunable parameters for knowledge base search tools live here so they're
easy to find, review, and adjust without digging through tool_registry,
context_handler, or the core tools module.
"""

# ---------------------------------------------------------------------------
# Tool-level defaults (used by KnowledgeSearchTool / tool_registry)
# ---------------------------------------------------------------------------

# Default number of top results to retrieve per knowledge base.
DEFAULT_TOP_K = 10

# Maximum context tokens for formatted search results returned to the LLM.
# This is the budget for the knowledge_search tool's output.
DEFAULT_MAX_CONTEXT_TOKENS = 16_000

# ---------------------------------------------------------------------------
# Context handler retrieval defaults
# ---------------------------------------------------------------------------

# Maximum concurrent threads when searching multiple KBs in parallel.
MAX_SEARCH_WORKERS = 5

# Default image delivery mode for multimodal KB content ("base64" or "url").
DEFAULT_IMAGE_DELIVERY = "base64"

# Maximum characters for dropped-item preview text stored in DB diagnostics.
DROPPED_ITEM_TEXT_LIMIT = 500

# ---------------------------------------------------------------------------
# Vector index scan behavior (pgvector HNSW)
# ---------------------------------------------------------------------------

# pgvector HNSW iterative-scan mode applied per vector_search transaction.
#
# The HNSW index on ai.embeddings is global (one partial index spanning ALL
# KBs); vector_search applies `knowledge_base_id = <kb>` as a POST-scan filter.
# With pgvector's default (off), the approximate scan emits only ~ef_search
# global candidates BEFORE that filter, so a KB-scoped query is starved —
# returning far fewer than top_k, often zero (a regression introduced when the
# HNSW index was first made usable, #284). iterative_scan makes pgvector
# re-enter the graph in batches until top_k filtered rows are found.
#
#   "strict_order"  — return in exact distance order (conservative; default)
#   "relaxed_order" — may return slightly out of order; can be faster
#   "" / "off"      — disable (revert to single-batch, starvation-prone behavior)
#
# Per-KB partial HNSW indexes (issue #527) would remove the need for iterative
# scanning and its latency cost; until then this is the fix.
HNSW_ITERATIVE_SCAN_MODE = "strict_order"
