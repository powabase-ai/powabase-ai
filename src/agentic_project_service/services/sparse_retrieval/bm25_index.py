"""BM25s-based sparse index implementation.

Uses the bm25s library for fast, pre-computed BM25 scoring with
memory-mapped index loading.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import bm25s
from Stemmer import Stemmer

from .base import SparseIndexManager, SparseSearchResult
from .config import BM25_STEMMER_LANGUAGE, BM25_STOPWORDS, BM25_VARIANT

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Implements SparseRetriever protocol for BM25s.

    Provides search functionality over a pre-built BM25 index.
    """

    def __init__(
        self,
        bm25: bm25s.BM25 | None,
        item_ids: list[str],
        stemmer: Stemmer,
        stopwords: str | list[str],
    ):
        self._bm25 = bm25
        self._item_ids = item_ids
        self._stemmer = stemmer
        self._stopwords = stopwords

    def search(self, query: str, top_k: int = 10) -> list[SparseSearchResult]:
        """Search the BM25 index.

        Args:
            query: Search query (may include conversation context).
            top_k: Maximum results to return.

        Returns:
            List of SparseSearchResult ordered by BM25 score descending.
        """
        if not self.is_ready():
            return []

        # Tokenize query using same stemmer/stopwords as index
        query_tokens = bm25s.tokenize(
            [query],
            stopwords=self._stopwords,
            stemmer=self._stemmer,
        )

        # Retrieve top-k results
        effective_k = min(top_k, len(self._item_ids))
        if effective_k == 0:
            return []

        results, scores = self._bm25.retrieve(query_tokens, k=effective_k)

        # Validate results/scores alignment
        if len(results[0]) != len(scores[0]):
            logger.error(
                "BM25 results/scores length mismatch: %d results vs %d scores",
                len(results[0]),
                len(scores[0]),
            )
            # Truncate to minimum length
            min_len = min(len(results[0]), len(scores[0]))
            result_indices = results[0][:min_len]
            result_scores = scores[0][:min_len]
        else:
            result_indices = results[0]
            result_scores = scores[0]

        # Convert to SparseSearchResult with bounds checking
        search_results = []
        for idx, score in zip(result_indices, result_scores):
            idx_int = int(idx)  # Convert numpy int to Python int
            if idx_int < 0 or idx_int >= len(self._item_ids):
                logger.warning(
                    "BM25 returned out-of-bounds index %d (size: %d), skipping",
                    idx_int,
                    len(self._item_ids),
                )
                continue
            search_results.append(
                SparseSearchResult(
                    item_id=self._item_ids[idx_int],
                    score=float(score),
                )
            )

        return search_results

    def is_ready(self) -> bool:
        """Check if retriever is ready for queries."""
        return self._bm25 is not None and len(self._item_ids) > 0


class BM25IndexManager(SparseIndexManager):
    """BM25s-based sparse index manager with incremental support.

    Manages the lifecycle of a BM25 index including building, updating,
    persisting, and loading. Uses bm25s library for fast sparse retrieval.
    """

    def __init__(
        self,
        variant: str = BM25_VARIANT,
        stemmer_language: str = BM25_STEMMER_LANGUAGE,
        stopwords: str | list[str] = BM25_STOPWORDS,
    ):
        """Initialize BM25 index manager.

        Args:
            variant: BM25 variant ("robertson", "lucene", "atire", "bm25l", "bm25+").
            stemmer_language: Language for stemming (e.g., "english").
            stopwords: Stopwords to filter ("en" for English, or custom list).
        """
        self.variant = variant
        self.stemmer = Stemmer(stemmer_language)
        self.stopwords = stopwords
        self._bm25: bm25s.BM25 | None = None
        self._item_ids: list[str] = []
        self._corpus: list[str] = []

    def build_index(self, documents: list[str], item_ids: list[str]) -> dict:
        """Build BM25 index from documents.

        Args:
            documents: Document texts to index.
            item_ids: Corresponding item IDs (must match documents length).

        Returns:
            Stats dict with doc_count and vocab_size.
        """
        if len(documents) != len(item_ids):
            raise ValueError(
                f"documents ({len(documents)}) and item_ids ({len(item_ids)}) must have same length"
            )

        if not documents:
            self._bm25 = None
            self._item_ids = []
            self._corpus = []
            return {"doc_count": 0, "vocab_size": 0}

        # Tokenize all documents
        tokens = bm25s.tokenize(
            documents,
            stopwords=self.stopwords,
            stemmer=self.stemmer,
        )

        # Build BM25 index
        self._bm25 = bm25s.BM25(method=self.variant)
        self._bm25.index(tokens)
        self._item_ids = list(item_ids)
        self._corpus = list(documents)

        vocab_size = len(self._bm25.vocab) if hasattr(self._bm25, "vocab") else 0

        logger.info(
            "Built BM25 index: %d docs, %d vocab, variant=%s",
            len(documents),
            vocab_size,
            self.variant,
        )

        return {"doc_count": len(documents), "vocab_size": vocab_size}

    def add_documents(self, documents: list[str], item_ids: list[str]) -> None:
        """Incrementally add documents to existing index.

        Note: bm25s doesn't support true incremental indexing, so we
        merge with existing corpus and rebuild. This is still fast
        due to bm25s's efficiency.

        Args:
            documents: New document texts to add.
            item_ids: Corresponding item IDs.

        Raises:
            ValueError: If documents and item_ids have different lengths.
        """
        if not documents:
            return

        if len(documents) != len(item_ids):
            raise ValueError(
                f"documents ({len(documents)}) and item_ids ({len(item_ids)}) must have same length"
            )

        # Merge with existing corpus
        self._corpus.extend(documents)
        self._item_ids.extend(item_ids)

        # Rebuild index
        self.build_index(self._corpus, self._item_ids)

    def remove_documents(self, item_ids: list[str]) -> None:
        """Remove documents from the index by ID.

        Args:
            item_ids: IDs of documents to remove.
        """
        if not item_ids:
            return

        remove_set = set(item_ids)

        # Filter out removed documents
        filtered = [
            (doc, id) for doc, id in zip(self._corpus, self._item_ids) if id not in remove_set
        ]

        if filtered:
            docs, ids = zip(*filtered)
            self.build_index(list(docs), list(ids))
        else:
            # All documents removed
            self._corpus = []
            self._item_ids = []
            self._bm25 = None

    def save(self, path: str) -> None:
        """Save index to disk.

        Args:
            path: Directory path to save index files.
        """
        if self._bm25 is None:
            logger.warning("Cannot save empty BM25 index")
            return

        os.makedirs(path, exist_ok=True)

        # Save BM25 index with corpus
        self._bm25.save(path, corpus=self._corpus)

        # Save item IDs mapping
        item_ids_path = os.path.join(path, "item_ids.json")
        with open(item_ids_path, "w") as f:
            json.dump(self._item_ids, f)

        logger.info("Saved BM25 index to %s (%d docs)", path, len(self._item_ids))

    def load(self, path: str) -> None:
        """Load index from disk with memory mapping.

        Args:
            path: Directory path containing index files.
        """
        item_ids_path = os.path.join(path, "item_ids.json")

        if not os.path.exists(item_ids_path):
            raise FileNotFoundError(f"Item IDs file not found: {item_ids_path}")

        # Load BM25 index with memory mapping for efficiency.
        # load_corpus=True keeps _corpus and _item_ids in sync so subsequent
        # add_documents() calls extend both lists symmetrically. With mmap=True,
        # bm25s memory-maps the corpus via JsonlCorpus — no full-RAM cost.
        self._bm25 = bm25s.BM25.load(path, load_corpus=True, mmap=True)

        # Load item IDs mapping
        with open(item_ids_path) as f:
            self._item_ids = json.load(f)

        # Load corpus from bm25s. With load_corpus=True, self._bm25.corpus is
        # either a list[dict] or a memory-mapped JsonlCorpus — iterate to
        # materialize a plain list for in-memory mutation by add_documents.
        if hasattr(self._bm25, "corpus") and self._bm25.corpus is not None:
            self._corpus = [
                doc["text"] if isinstance(doc, dict) and "text" in doc else doc
                for doc in self._bm25.corpus
            ]
        else:
            self._corpus = []

        # Clear bm25s's internal corpus reference. When self._bm25.corpus is
        # set, bm25s.BM25.retrieve() returns document objects (dicts) instead
        # of integer indices, breaking BM25Retriever.search() which expects
        # to int(idx) into self._item_ids. We've already copied the corpus
        # into self._corpus above; subsequent re-saves explicitly pass
        # corpus=self._corpus to bm25s.save, so dropping this reference here
        # is safe.
        self._bm25.corpus = None

        logger.info("Loaded BM25 index from %s (%d docs)", path, len(self._item_ids))

    def get_retriever(self) -> BM25Retriever:
        """Get a retriever instance for this index."""
        return BM25Retriever(
            self._bm25,
            self._item_ids,
            self.stemmer,
            self.stopwords,
        )

    def is_empty(self) -> bool:
        """Check if the index is empty."""
        return self._bm25 is None or len(self._item_ids) == 0
