"""Strategy registry for indexing and retrieval."""

from .registry import (
    RETRIEVER_LABELS,
    STRATEGY_REGISTRY,
    get_default_retrieval_method,
    get_strategy,
    validate_retriever,
)

__all__ = [
    "RETRIEVER_LABELS",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "validate_retriever",
    "get_default_retrieval_method",
]
