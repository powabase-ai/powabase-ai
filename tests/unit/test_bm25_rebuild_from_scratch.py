"""Tests for SparseIndexStore.rebuild_from_scratch — atomic replace of the
BM25 index for one (kb, item_table)."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_project_service.services.sparse_retrieval.sparse_index_store import (
    SparseIndexStore,
)


def test_rebuild_from_scratch_creates_files(tmp_path):
    """Builds index, writes files including metadata.json sidecar."""
    store = SparseIndexStore(knowledge_base_id="test-kb-4", base_path=str(tmp_path))
    docs = ["hello world", "foo bar baz", "the quick brown fox"]
    ids = ["a", "b", "c"]
    store.rebuild_from_scratch(documents=docs, item_ids=ids, item_table="chunks")
    index_path = Path(store.get_index_path("chunks"))
    assert index_path.exists()
    assert (index_path / "metadata.json").exists()
    metadata = json.loads((index_path / "metadata.json").read_text())
    assert metadata["item_count"] == 3


def test_rebuild_from_scratch_replaces_existing(tmp_path):
    """Rebuilding overrides any previously-saved index for the same key."""
    store = SparseIndexStore(knowledge_base_id="test-kb-5", base_path=str(tmp_path))
    # bm25s filters single-char tokens as stopwords; use multi-word strings to keep vocab non-empty
    store.rebuild_from_scratch(
        documents=["apple orange grape", "banana mango kiwi", "cherry plum peach"],
        item_ids=["1", "2", "3"],
        item_table="chunks",
    )
    assert store.read_metadata("chunks")["item_count"] == 3

    store.rebuild_from_scratch(
        documents=[
            "apple orange grape",
            "banana mango kiwi",
            "cherry plum peach",
            "lemon lime melon",
            "watermelon papaya guava",
        ],
        item_ids=["1", "2", "3", "4", "5"],
        item_table="chunks",
    )
    assert store.read_metadata("chunks")["item_count"] == 5


def test_rebuild_from_scratch_clears_cache(tmp_path):
    """After rebuild, the in-memory manager cache for this (kb, item_table)
    must be invalidated so future reads see the new index, not the stale one."""
    store = SparseIndexStore(knowledge_base_id="test-kb-6", base_path=str(tmp_path))
    store.add_and_save(documents=["alpha"], item_ids=["a1"], item_table="chunks")
    cache_key = "test-kb-6:chunks"
    assert cache_key in store._managers

    store.rebuild_from_scratch(
        documents=["beta", "gamma"], item_ids=["b1", "b2"], item_table="chunks"
    )
    assert cache_key not in store._managers


def test_rebuild_from_scratch_empty_documents_writes_nothing(tmp_path):
    """If documents is empty, no index files are created."""
    store = SparseIndexStore(knowledge_base_id="test-kb-7", base_path=str(tmp_path))
    store.rebuild_from_scratch(documents=[], item_ids=[], item_table="chunks")
    assert store.read_metadata("chunks") is None
