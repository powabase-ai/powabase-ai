"""Regression tests for BM25IndexManager save/load round-trip.

The bug this pins: BM25IndexManager.load() called bm25s.BM25.load with
load_corpus=False. That left self._bm25.corpus = None, so the fallback
branch set self._corpus = []. Subsequent add_documents() then extended
_item_ids with the prior on-disk count + new count, but extended _corpus
with only the new count — making the lengths diverge before build_index()
hit its length-equality check and raised ValueError.

Verified empirically against a real production failure where indexing a
new source after a worker restart failed with:
  "documents (96) and item_ids (228) must have same length"

These tests reproduce the conditions: build → save → fresh-manager-load →
add_documents, and assert the lengths stay aligned across the round-trip.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agentic_project_service.services.sparse_retrieval.bm25_index import (
    BM25IndexManager,
)


def _initial_docs() -> tuple[list[str], list[str]]:
    docs = [
        "the quick brown fox jumps over the lazy dog",
        "graph indexing for retrieval augmented generation",
        "lexical search complements vector embeddings",
        "bm25 is a classical information retrieval algorithm",
    ]
    ids = [f"item-a-{i}" for i in range(len(docs))]
    return docs, ids


def _additional_docs() -> tuple[list[str], list[str]]:
    docs = [
        "incremental indexing requires a consistent corpus state",
        "memory mapped corpora avoid loading the full file into ram",
    ]
    ids = [f"item-b-{i}" for i in range(len(docs))]
    return docs, ids


def test_corpus_persists_across_save_load_cycle():
    """After save → fresh-manager → load, _corpus must have the same length
    as _item_ids. A fresh manager simulates a worker restart."""
    initial_docs, initial_ids = _initial_docs()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "bm25")

        # Build and persist an index — what happens during the first source's indexing.
        manager_a = BM25IndexManager()
        manager_a.build_index(initial_docs, initial_ids)
        manager_a.save(path)

        # Fresh manager (cache miss after worker restart) — must reload the corpus.
        manager_b = BM25IndexManager()
        manager_b.load(path)

        # The bug: _item_ids loaded but _corpus did not.
        assert len(manager_b._item_ids) == len(initial_ids)
        assert len(manager_b._corpus) == len(initial_docs), (
            f"_corpus was not loaded from disk: "
            f"len(_corpus)={len(manager_b._corpus)}, "
            f"len(_item_ids)={len(manager_b._item_ids)}"
        )


def test_add_documents_after_load_does_not_raise():
    """After loading a saved index, add_documents must NOT raise the
    'documents and item_ids must have same length' error. This is the
    user-visible failure mode."""
    initial_docs, initial_ids = _initial_docs()
    new_docs, new_ids = _additional_docs()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "bm25")

        manager_a = BM25IndexManager()
        manager_a.build_index(initial_docs, initial_ids)
        manager_a.save(path)

        manager_b = BM25IndexManager()
        manager_b.load(path)

        # This call raised ValueError prior to the fix.
        manager_b.add_documents(new_docs, new_ids)

        # And the post-add state should be the union of both batches.
        expected_count = len(initial_ids) + len(new_ids)
        assert len(manager_b._corpus) == expected_count
        assert len(manager_b._item_ids) == expected_count


def test_load_then_add_then_save_then_load_again():
    """Two save/load cycles to make sure the corpus persists through
    repeated round-trips, not just the first one. This catches partial
    fixes where load() only partially restores state."""
    initial_docs, initial_ids = _initial_docs()
    new_docs, new_ids = _additional_docs()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "bm25")

        m1 = BM25IndexManager()
        m1.build_index(initial_docs, initial_ids)
        m1.save(path)

        m2 = BM25IndexManager()
        m2.load(path)
        m2.add_documents(new_docs, new_ids)
        m2.save(path)

        # Second worker restart cycle.
        m3 = BM25IndexManager()
        m3.load(path)

        expected_count = len(initial_ids) + len(new_ids)
        assert len(m3._corpus) == expected_count
        assert len(m3._item_ids) == expected_count


def test_search_after_load_returns_item_ids_not_dicts():
    """After load(), the retriever's search() must return SparseSearchResult
    whose item_id matches one of the loaded ids — proving retrieve() came
    back as indices, not corpus dicts.

    Regression: a prior fix changed load_corpus=False → load_corpus=True to
    persist _corpus across worker restarts. That populated self._bm25.corpus,
    which makes bm25s.BM25.retrieve() return document objects (dicts like
    {'id': ..., 'text': ...}) instead of integer indices. The wrapper's
    `int(idx)` call then raised:

        int() argument must be a string, a bytes-like object or a real
        number, not 'dict'

    The fix preserves _corpus (so add_documents still works) AND ensures
    retrieve() falls back to integer indices."""
    initial_docs, initial_ids = _initial_docs()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "bm25")

        m1 = BM25IndexManager()
        m1.build_index(initial_docs, initial_ids)
        m1.save(path)

        m2 = BM25IndexManager()
        m2.load(path)

        retriever = m2.get_retriever()
        results = retriever.search("quick brown fox", top_k=3)

        assert len(results) > 0, "expected at least one BM25 match for 'quick brown fox'"
        for r in results:
            assert isinstance(r.item_id, str), (
                f"item_id must be a string from _item_ids, got {type(r.item_id).__name__}: {r.item_id!r}"
            )
            assert r.item_id in initial_ids, (
                f"item_id {r.item_id!r} not in loaded ids {initial_ids}"
            )
