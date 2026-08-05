"""The external-API rate-limit settings exist with sane defaults."""

from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY


def test_firecrawl_rate_limit_setting():
    d = SETTINGS_REGISTRY.get("FIRECRAWL_RATE_LIMIT_PER_MINUTE")
    assert d is not None, "FIRECRAWL_RATE_LIMIT_PER_MINUTE not registered"
    assert d.type == "int"
    assert d.default == 30
    assert d.category == "tools"


def test_exa_rate_limit_setting():
    d = SETTINGS_REGISTRY.get("EXA_RATE_LIMIT_PER_MINUTE")
    assert d is not None, "EXA_RATE_LIMIT_PER_MINUTE not registered"
    assert d.type == "int"
    assert d.default == 60
    assert d.category == "tools"
