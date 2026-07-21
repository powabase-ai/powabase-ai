"""Tests for the metadata.json sidecar that records when/how big the
BM25 index was last rebuilt."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_project_service.services.sparse_retrieval.sparse_index_store import (
    SparseIndexStore,
)


def test_write_and_read_metadata_sidecar_roundtrip(tmp_path):
    """write_metadata then read_metadata returns the same item_count."""
    store = SparseIndexStore(knowledge_base_id="test-kb-1", base_path=str(tmp_path))
    store.write_metadata(item_table="chunks", item_count=1234)
    metadata = store.read_metadata(item_table="chunks")
    assert metadata is not None
    assert metadata["item_count"] == 1234
    assert "built_at" in metadata


def test_read_metadata_missing_returns_none(tmp_path):
    """If metadata.json doesn't exist, read_metadata returns None."""
    store = SparseIndexStore(knowledge_base_id="test-kb-2", base_path=str(tmp_path))
    assert store.read_metadata(item_table="chunks") is None


def test_read_metadata_malformed_returns_none(tmp_path):
    """Malformed JSON in the sidecar must not crash the caller; return None."""
    store = SparseIndexStore(knowledge_base_id="test-kb-3", base_path=str(tmp_path))
    path = Path(store.get_index_path("chunks"))
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text("not valid json {{{")
    assert store.read_metadata(item_table="chunks") is None
