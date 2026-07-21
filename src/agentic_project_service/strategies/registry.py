"""
Strategy Registry.

Couples each indexing strategy to its compatible retrieval methods,
default configs, and DB artifacts. This makes it easy to add new
strategy pairs and prevents incompatible indexer/retriever combinations.

Usage:
    from agentic_project_service.strategies import get_strategy, validate_retriever

    strategy = get_strategy("page_index")
    assert validate_retriever("page_index", "tree_search")
    assert not validate_retriever("page_index", "vector_search")
"""

import copy

from agentic.knowledge.model_config import (
    CHUNK_EMBED_DEFAULT_CHUNK_SIZE,
    CHUNK_EMBED_DEFAULT_OVERLAP,
    CHUNK_EMBED_EMBEDDING_MODEL,
    DOC2JSON_DEFAULT_PAGES_PER_WINDOW,
    DOC2JSON_DEFAULT_WINDOW_OVERLAP,
    DOC2JSON_DEFAULT_WINDOW_SIZE,
    DOC2JSON_EMBEDDING_MODEL,
    DOC2JSON_EXTRACTION_MODEL,
    DOC2JSON_USE_IMAGES,
    FULLDOC_EMBEDDING_MODEL,
    FULLDOC_SUMMARY_MODEL,
    GRAPHINDEX_EMBEDDING_MODEL,
    GRAPHINDEX_ENRICHMENT_MODEL,
    GRAPHINDEX_INDEXING_MODEL,
    PAGEINDEX_INDEXING_MODEL,
    PAGEINDEX_RETRIEVAL_MODEL,
)

RETRIEVER_LABELS: dict[str, str] = {
    "vector_search": "Vector search",
    "full_text": "Full-text search",
    "hybrid": "Hybrid (vector + full-text)",
    "tree_search": "Tree search (LLM reasoning)",
}

STRATEGY_REGISTRY: dict[str, dict] = {
    "chunk_embed": {
        "label": "Chunk + Embed",
        "compatible_retrievers": ["vector_search", "full_text", "hybrid"],
        "default_retrieval_method": "hybrid",
        "supports_reranker": True,
        "default_indexing_config": {
            "strategy": "chunk_embed",
            "chunk_size": CHUNK_EMBED_DEFAULT_CHUNK_SIZE,
            "overlap": CHUNK_EMBED_DEFAULT_OVERLAP,
            "embedding_model": CHUNK_EMBED_EMBEDDING_MODEL,
        },
        "default_retrieval_config": {
            "method": "hybrid",
            "top_k": 5,
            "context_mode": "text",
            "ts_language": "english",
        },
    },
    "page_index": {
        "label": "PageIndex (tree-based)",
        "compatible_retrievers": ["tree_search"],
        "default_retrieval_method": "tree_search",
        "supports_reranker": False,
        "default_indexing_config": {
            "strategy": "page_index",
            "model": PAGEINDEX_INDEXING_MODEL,
            "if_add_node_summary": "yes",
        },
        "default_retrieval_config": {
            "method": "tree_search",
            "top_k": 5,
            "retrieval_model": PAGEINDEX_RETRIEVAL_MODEL,
            "context_mode": "text",
        },
    },
    "full_document": {
        "label": "Full Document (document-level)",
        "compatible_retrievers": ["vector_search", "full_text", "hybrid"],
        "default_retrieval_method": "hybrid",
        "supports_reranker": True,
        "default_indexing_config": {
            "strategy": "full_document",
            "summary_model": FULLDOC_SUMMARY_MODEL,
            "embedding_model": FULLDOC_EMBEDDING_MODEL,
        },
        "default_retrieval_config": {
            "method": "hybrid",
            "top_k": 5,
            "context_mode": "text",
            "ts_language": "english",
        },
    },
    "graph_index": {
        "label": "GraphIndex (graph-based)",
        "compatible_retrievers": ["vector_search", "full_text", "hybrid"],
        "default_retrieval_method": "hybrid",
        "supports_reranker": True,
        "default_indexing_config": {
            "strategy": "graph_index",
            "model": GRAPHINDEX_INDEXING_MODEL,
            "enrichment_model": GRAPHINDEX_ENRICHMENT_MODEL,
            "embedding_model": GRAPHINDEX_EMBEDDING_MODEL,
            "if_add_node_summary": "yes",
        },
        "default_retrieval_config": {
            "method": "hybrid",
            "top_k": 5,
            "context_mode": "text",
            "ts_language": "english",
        },
    },
    "doc2json": {
        "label": "Doc2JSON (structured extraction)",
        "compatible_retrievers": ["vector_search", "full_text", "hybrid"],
        "default_retrieval_method": "hybrid",
        "supports_reranker": True,
        "default_indexing_config": {
            "strategy": "doc2json",
            "extraction_model": DOC2JSON_EXTRACTION_MODEL,
            "embedding_model": DOC2JSON_EMBEDDING_MODEL,
            "window_size": DOC2JSON_DEFAULT_WINDOW_SIZE,
            "window_overlap": DOC2JSON_DEFAULT_WINDOW_OVERLAP,
            "use_images": DOC2JSON_USE_IMAGES,
            "pages_per_window": DOC2JSON_DEFAULT_PAGES_PER_WINDOW,
            "json_schema": {},  # User must provide schema
        },
        "default_retrieval_config": {
            "method": "hybrid",
            "top_k": 5,
            "context_mode": "text",
            "ts_language": "english",
        },
    },
}


def get_strategy(name: str) -> dict:
    """Get strategy configuration by name.

    Args:
        name: Strategy name (e.g., "chunk_embed", "page_index")

    Returns:
        Strategy configuration dict

    Raises:
        ValueError: If strategy name is not registered
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: '{name}'. Available strategies: {list(STRATEGY_REGISTRY.keys())}"
        )
    return copy.deepcopy(STRATEGY_REGISTRY[name])


def validate_retriever(strategy: str, retriever: str) -> bool:
    """Check if a retrieval method is compatible with a strategy.

    Args:
        strategy: Indexing strategy name
        retriever: Retrieval method name

    Returns:
        True if compatible, False otherwise
    """
    if strategy not in STRATEGY_REGISTRY:
        return False
    return retriever in STRATEGY_REGISTRY[strategy]["compatible_retrievers"]


def get_default_retrieval_method(strategy: str) -> str:
    """Get the default retrieval method for a strategy.

    Args:
        strategy: Indexing strategy name

    Returns:
        Default retrieval method name

    Raises:
        ValueError: If strategy name is not registered
    """
    config = get_strategy(strategy)
    return config["default_retrieval_method"]
