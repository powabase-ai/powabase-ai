"""BM25 fallback timeout setting definition and its range validation."""

from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY


def test_bm25_fallback_timeout_setting():
    d = SETTINGS_REGISTRY.get("BM25_FALLBACK_TIMEOUT_MS")
    assert d is not None, "BM25_FALLBACK_TIMEOUT_MS not registered"
    assert d.type == "int"
    assert d.default == 10000
    assert d.min == 1000
    assert d.max == 120000
    assert d.category == "knowledge-retrieval"
