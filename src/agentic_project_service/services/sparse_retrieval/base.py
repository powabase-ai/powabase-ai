"""Base abstractions for sparse retrieval.

Defines the SparseRetriever protocol and SparseIndexManager ABC to enable
extensibility for future sparse retrieval methods (SPLADE, TF-IDF, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class SparseSearchResult:
    """Result from sparse retrieval."""

    item_id: str
    score: float
    text: str | None = None  # Optional corpus text if stored


@runtime_checkable
class SparseRetriever(Protocol):
    """Protocol for sparse retrieval implementations.

    Enables duck-typing for any retriever that implements search() and is_ready().
    """

    def search(self, query: str, top_k: int = 10) -> list[SparseSearchResult]:
        """Search the sparse index.

        Args:
            query: The search query (may include conversation context).
            top_k: Maximum number of results to return.

        Returns:
            List of SparseSearchResult ordered by score descending.
        """
        ...

    def is_ready(self) -> bool:
        """Check if the index is loaded and ready for queries."""
        ...


class SparseIndexManager(ABC):
    """Abstract base for sparse index lifecycle management.

    Implementations handle building, updating, persisting, and loading
    sparse indexes for different retrieval algorithms.
    """

    @abstractmethod
    def build_index(self, documents: list[str], item_ids: list[str]) -> dict:
        """Build the index from documents.

        Args:
            documents: List of document texts to index.
            item_ids: Corresponding item IDs (same length as documents).

        Returns:
            Stats dict with keys like "doc_count", "vocab_size".
        """
        ...

    @abstractmethod
    def add_documents(self, documents: list[str], item_ids: list[str]) -> None:
        """Incrementally add documents to existing index.

        Args:
            documents: New document texts to add.
            item_ids: Corresponding item IDs.
        """
        ...

    @abstractmethod
    def remove_documents(self, item_ids: list[str]) -> None:
        """Remove documents from the index by ID.

        Args:
            item_ids: IDs of documents to remove.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist index to storage.

        Args:
            path: Directory path to save index files.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from storage.

        Args:
            path: Directory path containing index files.
        """
        ...

    @abstractmethod
    def get_retriever(self) -> SparseRetriever:
        """Get a retriever instance for this index.

        Returns:
            A SparseRetriever that can search the loaded index.
        """
        ...

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if the index is empty (no documents indexed)."""
        ...
