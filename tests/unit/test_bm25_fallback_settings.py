"""BM25 fallback timeout setting definition and its range validation."""

import pytest

from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY, validate_setting


def test_bm25_fallback_timeout_setting():
    d = SETTINGS_REGISTRY.get("BM25_FALLBACK_TIMEOUT_MS")
    assert d is not None, "BM25_FALLBACK_TIMEOUT_MS not registered"
    assert d.type == "int"
    assert d.default == 10000
    assert d.min == 1000
    # The budget has to stay under the HTTP timeout of whoever is waiting on
    # the search, or the caller gives up while the query keeps burning the
    # database -- the pile-up this bound exists to prevent.
    assert d.max == 30000
    assert d.category == "knowledge-retrieval"


def test_bm25_fallback_timeout_description_warns_about_the_caller_timeout():
    d = SETTINGS_REGISTRY["BM25_FALLBACK_TIMEOUT_MS"]
    assert "timeout" in d.description.lower()
    assert "caller" in d.description.lower()


@pytest.mark.parametrize(
    "value,ok",
    [
        (999, False),
        (1000, True),
        (10000, True),
        (30000, True),
        (30001, False),
        (0, False),
        (-1, False),
        ("not-a-number", False),
    ],
)
def test_validate_setting_enforces_the_range(value, ok):
    accepted, message = validate_setting("BM25_FALLBACK_TIMEOUT_MS", value)
    assert accepted is ok, message
    if not ok:
        assert message
