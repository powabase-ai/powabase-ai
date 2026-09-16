"""Tests for the build_bm25_for_kb Celery task — chunks-only rebuild
that walks the right item_table per the KB's indexing strategy."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def mock_kb_lookup():
    with patch("agentic_project_service.tasks.indexing._fetch_kb_for_bm25_build") as m:
        yield m


@pytest.fixture
def mock_iter_items():
    with patch("agentic_project_service.tasks.indexing._iter_items_for_kb_bm25") as m:
        yield m


@pytest.fixture
def mock_sparse_store_cls():
    with patch("agentic_project_service.tasks.indexing.SparseIndexStore") as cls:
        yield cls


def test_build_picks_chunks_table_for_chunk_embed(
    mock_kb_lookup, mock_iter_items, mock_sparse_store_cls
):
    from agentic_project_service.tasks.indexing import build_bm25_for_kb

    mock_kb_lookup.return_value = {
        "id": "kb-1",
        "indexing_config": {"strategy": "chunk_embed"},
    }
    mock_iter_items.return_value = iter(
        [[{"id": "c1", "text": "hello"}, {"id": "c2", "text": "world"}]]
    )
    sparse_store = mock_sparse_store_cls.return_value

    result = build_bm25_for_kb.run("kb-1")

    mock_iter_items.assert_called_once_with("kb-1", "chunks", batch_size=10_000)
    sparse_store.rebuild_from_scratch.assert_called_once_with(
        documents=["hello", "world"],
        item_ids=["c1", "c2"],
        item_table="chunks",
    )
    assert result == {"item_table": "chunks", "item_count": 2}


def test_build_picks_full_documents_for_full_document(
    mock_kb_lookup, mock_iter_items, mock_sparse_store_cls
):
    """full_documents belongs to the full_document strategy.

    page_index is compatible with tree_search only and stores its text in
    page_index_toc/page_index_nodes, so it has no BM25 item table at all.
    """
    from agentic_project_service.tasks.indexing import build_bm25_for_kb

    mock_kb_lookup.return_value = {
        "id": "kb-2",
        "indexing_config": {"strategy": "full_document"},
    }
    mock_iter_items.return_value = iter([[{"id": "d1", "text": "summary one"}]])
    sparse_store = mock_sparse_store_cls.return_value

    build_bm25_for_kb.run("kb-2")
    mock_iter_items.assert_called_once_with("kb-2", "full_documents", batch_size=10_000)
    sparse_store.rebuild_from_scratch.assert_called_once_with(
        documents=["summary one"],
        item_ids=["d1"],
        item_table="full_documents",
    )


def test_build_picks_graph_index_nodes_for_graph_index(
    mock_kb_lookup, mock_iter_items, mock_sparse_store_cls
):
    from agentic_project_service.tasks.indexing import build_bm25_for_kb

    mock_kb_lookup.return_value = {
        "id": "kb-3",
        "indexing_config": {"strategy": "graph_index"},
    }
    mock_iter_items.return_value = iter([[{"id": "n1", "text": "Title One Body One"}]])
    sparse_store = mock_sparse_store_cls.return_value

    build_bm25_for_kb.run("kb-3")
    mock_iter_items.assert_called_once_with("kb-3", "graph_index_nodes", batch_size=10_000)


def test_build_raises_for_unknown_strategy(mock_kb_lookup, mock_iter_items, mock_sparse_store_cls):
    from agentic_project_service.tasks.indexing import build_bm25_for_kb

    mock_kb_lookup.return_value = {
        "id": "kb-4",
        "indexing_config": {"strategy": "weird-thing"},
    }

    with pytest.raises(ValueError, match="weird-thing"):
        build_bm25_for_kb.run("kb-4")


def test_build_rejects_page_index(mock_kb_lookup, mock_iter_items, mock_sparse_store_cls):
    """page_index is tree_search-only and keeps its text in page_index_* tables.

    It has no BM25 item table, so the build must refuse rather than index the
    wrong table.
    """
    from agentic_project_service.tasks.indexing import build_bm25_for_kb

    mock_kb_lookup.return_value = {
        "id": "kb-6",
        "indexing_config": {"strategy": "page_index"},
    }

    with pytest.raises(ValueError, match="page_index"):
        build_bm25_for_kb.run("kb-6")
    mock_iter_items.assert_not_called()


def test_build_rejects_doc2json(mock_kb_lookup, mock_iter_items, mock_sparse_store_cls):
    """doc2json has no sparse-index maintenance, so it must stay on the fallback.

    run_doc2json_indexing never calls SparseIndexStore and the removal block in
    index_source does not cover doc2json_documents. Letting a build succeed
    would freeze the index at build time: documents added afterwards would be
    invisible to keyword search and deleted ones would keep scoring, with
    bm25_status showing nothing. Until that maintenance exists, a doc2json KB
    deliberately uses the slow-but-correct tsvector fallback.
    """
    from agentic_project_service.tasks.indexing import build_bm25_for_kb

    mock_kb_lookup.return_value = {
        "id": "kb-7",
        "indexing_config": {"strategy": "doc2json"},
    }

    with pytest.raises(ValueError, match="doc2json"):
        build_bm25_for_kb.run("kb-7")
    mock_iter_items.assert_not_called()


def test_build_with_multiple_batches_accumulates(
    mock_kb_lookup, mock_iter_items, mock_sparse_store_cls
):
    """Iterator may yield multiple batches; the task accumulates before rebuild."""
    from agentic_project_service.tasks.indexing import build_bm25_for_kb

    mock_kb_lookup.return_value = {
        "id": "kb-5",
        "indexing_config": {"strategy": "chunk_embed"},
    }
    mock_iter_items.return_value = iter(
        [
            [{"id": f"c{i}", "text": f"text-{i}"} for i in range(3)],
            [{"id": f"c{i}", "text": f"text-{i}"} for i in range(3, 5)],
        ]
    )
    sparse_store = mock_sparse_store_cls.return_value

    result = build_bm25_for_kb.run("kb-5")
    sparse_store.rebuild_from_scratch.assert_called_once_with(
        documents=[f"text-{i}" for i in range(5)],
        item_ids=[f"c{i}" for i in range(5)],
        item_table="chunks",
    )
    assert result["item_count"] == 5
