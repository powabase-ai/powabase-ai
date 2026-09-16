"""BM25 move long-holder threshold setting definition and its range validation."""

import pytest

from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY, validate_setting


def test_bm25_move_long_holder_seconds_setting():
    d = SETTINGS_REGISTRY.get("BM25_MOVE_LONG_HOLDER_SECONDS")
    assert d is not None, "BM25_MOVE_LONG_HOLDER_SECONDS not registered"
    assert d.type == "int"
    assert d.default == 5
    assert d.min == 1
    assert d.max == 300
    assert d.category == "knowledge-retrieval"
    assert d.advanced is True


def test_bm25_move_long_holder_seconds_label_and_description():
    d = SETTINGS_REGISTRY["BM25_MOVE_LONG_HOLDER_SECONDS"]
    assert d.label == "BM25 Move Long-Holder Timeout (s)"
    assert "partition" in d.description.lower()
    assert "retry" in d.description.lower()


@pytest.mark.parametrize(
    "value,ok",
    [
        (0, False),
        (1, True),
        (5, True),
        (300, True),
        (301, False),
        (-1, False),
        ("not-a-number", False),
    ],
)
def test_validate_setting_enforces_the_range(value, ok):
    accepted, message = validate_setting("BM25_MOVE_LONG_HOLDER_SECONDS", value)
    assert accepted is ok, message
    if not ok:
        assert message
