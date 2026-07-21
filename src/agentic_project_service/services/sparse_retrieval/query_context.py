"""Query context builder for BM25 search.

Builds search context by tokenizing the query alongside recent conversation
history, eliminating the need for LLM-powered query enrichment in most cases.
"""

from __future__ import annotations

import logging
from typing import Any

import bm25s
from Stemmer import Stemmer

from .config import (
    BM25_STEMMER_LANGUAGE,
    BM25_STOPWORDS,
    MAX_MESSAGE_CHARS,
    QUERY_CONTEXT_MAX_HISTORY,
    QUERY_TERM_WEIGHT,
    USE_LLM_ENRICHMENT_DEFAULT,
)

logger = logging.getLogger(__name__)


class QueryContextBuilder:
    """Builds search context by tokenizing query + conversation history.

    This is the fast alternative to LLM-powered query enrichment. It extracts
    meaningful terms from recent chat history to resolve conversational
    references like "what about pricing?" or "tell me more".

    The approach:
    1. Weight the current query higher (repeat it N times)
    2. Include recent messages (both user and assistant) truncated
    3. Concatenate all text for BM25 tokenization
    """

    def __init__(
        self,
        stemmer_language: str = BM25_STEMMER_LANGUAGE,
        stopwords: str | list[str] = BM25_STOPWORDS,
        max_history_messages: int = QUERY_CONTEXT_MAX_HISTORY,
        query_weight: float = QUERY_TERM_WEIGHT,
        max_message_chars: int = MAX_MESSAGE_CHARS,
    ):
        """Initialize query context builder.

        Args:
            stemmer_language: Language for stemming (e.g., "english").
            stopwords: Stopwords to filter ("en" or custom list).
            max_history_messages: Maximum number of recent messages to include.
            query_weight: Weight multiplier for current query terms (1.0 = no boost).
            max_message_chars: Maximum characters per message (truncate longer).
        """
        self.stemmer = Stemmer(stemmer_language)
        self.stopwords = stopwords
        self.max_history = max_history_messages
        self.query_weight = query_weight
        self.max_message_chars = max_message_chars

    def build_search_text(
        self,
        query: str,
        session_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build combined search text from query and conversation history.

        Args:
            query: Current user query.
            session_history: List of {"role": "user"|"assistant", "content": "..."}.

        Returns:
            Combined text for BM25 search with query terms emphasized.
        """
        parts = []

        # Add query multiple times to boost its terms (simple term weighting)
        # This ensures the current query has more influence than history
        weight_count = max(1, int(self.query_weight))
        for _ in range(weight_count):
            parts.append(query)

        # Add recent conversation history
        if session_history:
            recent = session_history[-self.max_history :]
            for msg in recent:
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    # Truncate long messages to avoid noise
                    truncated = content[: self.max_message_chars]
                    parts.append(truncated)

        return " ".join(parts)

    def extract_context_terms(
        self,
        query: str,
        session_history: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Extract tokenized terms from query + history for debugging/logging.

        Args:
            query: Current user query.
            session_history: Conversation history.

        Returns:
            List of stemmed tokens extracted from the combined context.
            Returns empty list if tokenization fails.
        """
        text = self.build_search_text(query, session_history)
        try:
            tokens = bm25s.tokenize(
                [text],
                stopwords=self.stopwords,
                stemmer=self.stemmer,
                return_ids=False,  # Return actual strings, not vocab indices
            )
            # tokens is a list of token arrays, get first (only) one
            return list(tokens[0]) if tokens and len(tokens) > 0 else []
        except Exception as e:
            logger.warning("Failed to extract context terms: %s", e)
            return []


def build_search_query(
    query: str,
    session_history: list[dict[str, Any]] | None = None,
    use_llm_enrichment: bool = USE_LLM_ENRICHMENT_DEFAULT,
    enrichment_model: str | None = None,
    enrichment_reasoning_effort: str | None = None,
) -> dict[str, str]:
    """Build search queries for retrieval.

    This is the main entry point for query processing. It supports two modes:

    1. Fast tokenization (default): Uses QueryContextBuilder to combine
       query with chat history. No LLM call required.

    2. LLM enrichment (optional): Falls back to existing LLM-powered
       query enrichment for higher quality but slower results.

    Args:
        query: Current user query.
        session_history: Conversation history for context resolution.
        use_llm_enrichment: If True, use LLM for keyword generation.
        enrichment_model: Model to use for LLM enrichment (if enabled).

    Returns:
        Dict with:
            - "sparse_query": Query text for BM25 search (includes context)
            - "enriched_query": Query for vector embedding
    """
    if use_llm_enrichment:
        # Use existing LLM enrichment for high-quality results
        # Import here to avoid circular dependency
        from ..query_enrichment import enrich_query

        logger.debug("Using LLM enrichment for query: %s", query[:50])

        result = enrich_query(
            query=query,
            retrieval_method="hybrid",
            session_history=session_history,
            model=enrichment_model,
            reasoning_effort=enrichment_reasoning_effort,
        )

        return {
            "sparse_query": result.get("keyword_query", query),
            "enriched_query": result.get("enriched_query", query),
        }

    # Fast tokenization-based context building (default)
    builder = QueryContextBuilder()
    sparse_query = builder.build_search_text(query, session_history)

    # Log context terms for debugging
    if session_history:
        context_terms = builder.extract_context_terms(query, session_history)
        logger.debug(
            "Built sparse query with %d context terms from %d history messages",
            len(context_terms),
            len(session_history),
        )

    return {
        "sparse_query": sparse_query,
        # For vector search, use original query (not context-expanded)
        # The embedding model handles semantic similarity
        "enriched_query": query,
    }
