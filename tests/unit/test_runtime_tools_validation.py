"""Validation for the per-request runtime_tools field."""

from unittest.mock import MagicMock

import pytest

from agentic_project_service.routes._runtime_tools import (
    RUNTIME_TOOLS_MAX_ENTRIES,
    validate_runtime_tools,
)


def test_absent_field_is_valid_and_empty():
    configs, err = validate_runtime_tools({}, MagicMock())
    assert configs == [] and err is None


def test_non_list_rejected():
    _, err = validate_runtime_tools({"runtime_tools": "web_search"}, MagicMock())
    assert err is not None and "list" in err


def test_empty_list_rejected():
    _, err = validate_runtime_tools({"runtime_tools": []}, MagicMock())
    assert err is not None and "list" in err


def test_entry_without_type_rejected():
    _, err = validate_runtime_tools({"runtime_tools": [{"name": "web_search"}]}, MagicMock())
    assert err is not None and "type" in err


def test_unknown_type_rejected():
    _, err = validate_runtime_tools({"runtime_tools": [{"type": "plugin"}]}, MagicMock())
    assert err is not None and "type" in err


def test_cap_enforced():
    """Uses VALID builtin entries so this pins the cap itself, not an earlier
    per-entry rejection."""
    entries = [{"type": "builtin", "name": "web_search"}] * (RUNTIME_TOOLS_MAX_ENTRIES + 1)
    _, err = validate_runtime_tools({"runtime_tools": entries}, MagicMock())
    assert err is not None and str(RUNTIME_TOOLS_MAX_ENTRIES) in err


def test_builtin_unknown_key_rejected():
    """Typo'd keys 400 (strict), unlike the attach endpoints which ignore them."""
    data = {"runtime_tools": [{"type": "builtin", "name": "web_search", "Config_Override": {}}]}
    _, err = validate_runtime_tools(data, MagicMock())
    assert err is not None and "Config_Override" in err


def test_builtin_unknown_name_rejected():
    data = {"runtime_tools": [{"type": "builtin", "name": "telepathy"}]}
    _, err = validate_runtime_tools(data, MagicMock())
    assert err is not None and "telepathy" in err


def test_builtin_blocked_names_rejected_distinctly():
    """database_query / database_write / code_execute are blocked at runtime —
    with a message that says blocked, not unknown."""
    for name in ("database_query", "database_write", "code_execute"):
        data = {"runtime_tools": [{"type": "builtin", "name": name}]}
        _, err = validate_runtime_tools(data, MagicMock())
        assert err is not None and name in err
        assert "unknown" not in err.lower()


def test_builtin_config_override_must_be_object():
    data = {"runtime_tools": [{"type": "builtin", "name": "web_search", "config_override": "deep"}]}
    _, err = validate_runtime_tools(data, MagicMock())
    assert err is not None and "config_override" in err


def test_builtin_config_override_keys_must_be_in_input_schema():
    """The attach path silently filters non-schema keys; runtime rejects them
    so callers see typos."""
    data = {
        "runtime_tools": [
            {"type": "builtin", "name": "web_search", "config_override": {"serch_type": "deep"}}
        ]
    }
    _, err = validate_runtime_tools(data, MagicMock())
    assert err is not None and "serch_type" in err


def test_builtin_valid_config_override_passes():
    data = {
        "runtime_tools": [
            {"type": "builtin", "name": "web_search", "config_override": {"search_type": "deep"}}
        ]
    }
    configs, err = validate_runtime_tools(data, MagicMock())
    assert err is None
    assert configs == data["runtime_tools"]
    assert configs[0] is not data["runtime_tools"][0]  # copies, never the caller's objects


@pytest.mark.xfail(reason="final-name dedup lands with the mcp task", strict=True)
def test_duplicate_builtin_names_rejected():
    data = {
        "runtime_tools": [
            {"type": "builtin", "name": "web_search"},
            {"type": "builtin", "name": "web_search"},
        ]
    }
    _, err = validate_runtime_tools(data, MagicMock())
    assert err is not None and "web_search" in err
