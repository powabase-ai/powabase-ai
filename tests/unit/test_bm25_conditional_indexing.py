"""Tests for BM25 conditional indexing — setting registration and helper logic."""

from __future__ import annotations

from unittest.mock import patch

from agentic_project_service.services.settings_registry import (
    SETTINGS_REGISTRY,
    get_setting,
)
from agentic_project_service.tasks.indexing import _should_build_bm25_now


def test_bm25_auto_indexing_setting_registered():
    """The BM25_AUTO_INDEXING setting must be in the registry under
    knowledge-indexing, with type=bool, default=True, advanced=True."""
    setting = SETTINGS_REGISTRY.get("BM25_AUTO_INDEXING")
    assert setting is not None, "BM25_AUTO_INDEXING not registered"
    assert setting.category == "knowledge-indexing"
    assert setting.type == "bool"
    assert setting.default is True
    assert setting.advanced is True


def test_bm25_auto_indexing_default_value():
    """get_setting() returns True when no override is configured (default)."""
    assert get_setting("BM25_AUTO_INDEXING") is True


@patch("agentic_project_service.tasks.indexing._get_kb_retrieval_method")
def test_should_build_bm25_skips_vector_search(mock_method):
    """method=vector_search → skip (False), regardless of setting."""
    mock_method.return_value = "vector_search"
    with patch("agentic_project_service.tasks.indexing.get_setting", return_value=True):
        assert _should_build_bm25_now("kb-id") is False


@patch("agentic_project_service.tasks.indexing._get_kb_retrieval_method")
def test_should_build_bm25_skips_when_setting_disabled(mock_method):
    """method=hybrid but BM25_AUTO_INDEXING=false → skip."""
    mock_method.return_value = "hybrid"
    with patch("agentic_project_service.tasks.indexing.get_setting", return_value=False):
        assert _should_build_bm25_now("kb-id") is False


@patch("agentic_project_service.tasks.indexing._get_kb_retrieval_method")
def test_should_build_bm25_runs_for_hybrid_with_setting_on(mock_method):
    mock_method.return_value = "hybrid"
    with patch("agentic_project_service.tasks.indexing.get_setting", return_value=True):
        assert _should_build_bm25_now("kb-id") is True


@patch("agentic_project_service.tasks.indexing._get_kb_retrieval_method")
def test_should_build_bm25_runs_for_full_text_with_setting_on(mock_method):
    mock_method.return_value = "full_text"
    with patch("agentic_project_service.tasks.indexing.get_setting", return_value=True):
        assert _should_build_bm25_now("kb-id") is True


@patch("agentic_project_service.tasks.indexing._get_kb_retrieval_method")
def test_should_build_bm25_skips_when_method_unknown(mock_method):
    """Unknown method (None or otherwise) is treated as not-BM25."""
    mock_method.return_value = None
    with patch("agentic_project_service.tasks.indexing.get_setting", return_value=True):
        assert _should_build_bm25_now("kb-id") is False
