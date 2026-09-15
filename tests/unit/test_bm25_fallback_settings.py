"""BM25 fallback timeout setting, rebuild debounce config, auto-indexing wording."""

from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY
from agentic_project_service.services.sparse_retrieval import config


def test_bm25_fallback_timeout_setting():
    d = SETTINGS_REGISTRY.get("BM25_FALLBACK_TIMEOUT_MS")
    assert d is not None, "BM25_FALLBACK_TIMEOUT_MS not registered"
    assert d.type == "int"
    assert d.default == 10000
    assert d.min == 1000
    assert d.max == 120000
    assert d.category == "knowledge-retrieval"


def test_rebuild_debounce_default():
    assert config.BM25_REBUILD_DEBOUNCE_SECONDS == 120


def test_auto_indexing_description_matches_coalesced_rebuilds():
    text = SETTINGS_REGISTRY.get("BM25_AUTO_INDEXING").description.lower()
    assert "rebuilt" in text
    assert "removed" in text
    assert "absent" in text
