"""Validation for the per-request runtime_tools field."""

from unittest.mock import MagicMock


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


def test_unhashable_type_rejected_not_raised():
    """An object/array 'type' must 400, not raise TypeError from an unhashable
    dict-key lookup."""
    for bad_type in ([], {}):
        _, err = validate_runtime_tools({"runtime_tools": [{"type": bad_type}]}, MagicMock())
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


def test_duplicate_builtin_names_rejected():
    data = {
        "runtime_tools": [
            {"type": "builtin", "name": "web_search"},
            {"type": "builtin", "name": "web_search"},
        ]
    }
    _, err = validate_runtime_tools(data, MagicMock())
    assert err is not None and "web_search" in err


_TOOL_1 = "11111111-1111-1111-1111-111111111111"
_TOOL_2 = "22222222-2222-2222-2222-222222222222"


def _db_returning_tools(rows):
    """A db_session whose execute() yields (id, name) rows from ai.tools."""
    db = MagicMock()
    db.execute.return_value = rows
    return db


def _definition(**overrides):
    d = {
        "name": "case_lookup",
        "description": "Look up a case by docket number",
        "input_schema": {"type": "object", "properties": {"docket": {"type": "string"}}},
        "config": {"endpoint": "https://api.example.com/cases"},
    }
    d.update(overrides)
    return d


def test_custom_requires_exactly_one_of_tool_id_or_definition():
    for entry in (
        {"type": "custom"},
        {"type": "custom", "tool_id": _TOOL_1, "definition": _definition()},
    ):
        _, err = validate_runtime_tools({"runtime_tools": [entry]}, MagicMock())
        assert err is not None and "tool_id" in err and "definition" in err


def test_custom_null_tool_id_with_valid_definition_is_accepted_as_definition_entry():
    """An explicit 'tool_id': null alongside a valid 'definition' is treated
    the same as an omitted 'tool_id' (key present, value None isn't a real
    reference) — it must validate as a definition entry, not fall through to
    a DB query with ids=[None]."""
    db = MagicMock()
    db.execute.return_value = []
    _, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "tool_id": None, "definition": _definition()}]},
        db,
    )
    assert err is None


def test_custom_null_definition_with_valid_tool_id_rejected_not_raised():
    """An explicit 'definition': null alongside a valid 'tool_id' must not
    raise when the final-name dedup pass looks up definition['name']."""
    configs, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "definition": None, "tool_id": _TOOL_1}]},
        _db_returning_tools([(_TOOL_1, "case_lookup")]),
    )
    assert err is None
    assert configs[0]["_resolved_name"] == "case_lookup"


def test_custom_malformed_tool_id_rejected_without_db():
    db = MagicMock()
    _, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "tool_id": "not-a-uuid"}]}, db
    )
    assert err is not None and "not-a-uuid" in err
    db.execute.assert_not_called()


def test_custom_tool_id_normalized_and_resolved():
    """Uppercase input still matches the DB row and passes through canonical."""
    upper = _TOOL_1.upper()
    configs, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "tool_id": upper}]},
        _db_returning_tools([(_TOOL_1, "case_lookup")]),
    )
    assert err is None
    assert configs[0]["tool_id"] == _TOOL_1
    assert configs[0]["_resolved_name"] == "case_lookup"


def test_custom_tool_id_resolved_name_shadowing_builtin_rejected():
    """An ai.tools row named like a builtin (e.g. web_search), referenced by
    tool_id, must be held to the same shadow rule as an inline definition —
    otherwise it silently replaces the attached builtin and its billing
    action at build time."""
    _, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "tool_id": _TOOL_1}]},
        _db_returning_tools([(_TOOL_1, "web_search")]),
    )
    assert err is not None and "web_search" in err


def test_custom_tool_id_resolved_name_shadowing_knowledge_search_rejected():
    _, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "tool_id": _TOOL_1}]},
        _db_returning_tools([(_TOOL_1, "knowledge_search")]),
    )
    assert err is not None and "knowledge_search" in err


def test_custom_resolved_name_key_cannot_be_injected_by_caller():
    """'_resolved_name' is set internally after DB resolution. A caller
    supplying it is already rejected by the strict per-type key allowlist —
    pin that."""
    data = {
        "runtime_tools": [{"type": "custom", "tool_id": _TOOL_1, "_resolved_name": "web_search"}]
    }
    _, err = validate_runtime_tools(data, _db_returning_tools([(_TOOL_1, "case_lookup")]))
    assert err is not None and "_resolved_name" in err


def test_custom_unknown_tool_id_rejected():
    _, err = validate_runtime_tools(
        {
            "runtime_tools": [
                {"type": "custom", "tool_id": _TOOL_1},
                {"type": "custom", "tool_id": _TOOL_2},
            ]
        },
        _db_returning_tools([(_TOOL_1, "case_lookup")]),
    )
    assert err is not None and _TOOL_2 in err


def test_custom_duplicate_tool_ids_rejected_after_normalization():
    _, err = validate_runtime_tools(
        {
            "runtime_tools": [
                {"type": "custom", "tool_id": _TOOL_1},
                {"type": "custom", "tool_id": _TOOL_1.upper()},
            ]
        },
        MagicMock(),
    )
    assert err is not None and _TOOL_1 in err


def test_definition_valid_passes():
    configs, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "definition": _definition()}]}, MagicMock()
    )
    assert err is None and configs[0]["definition"]["name"] == "case_lookup"


def test_definition_unknown_keys_rejected():
    d = _definition()
    d["is_concurrency_safe"] = True
    _, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "definition": d}]}, MagicMock()
    )
    assert err is not None and "is_concurrency_safe" in err


def test_definition_requires_name_description_schema_config():
    for missing in ("name", "description", "input_schema", "config"):
        d = _definition()
        del d[missing]
        _, err = validate_runtime_tools(
            {"runtime_tools": [{"type": "custom", "definition": d}]}, MagicMock()
        )
        assert err is not None and missing in err


def test_definition_name_shadowing_builtin_rejected():
    d = _definition(name="web_search")
    _, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "definition": d}]}, MagicMock()
    )
    assert err is not None and "web_search" in err


def test_definition_name_shadowing_knowledge_search_rejected():
    """knowledge_search is never touched by this feature — an inline
    definition claiming that name must 400, not silently displace the real
    KnowledgeSearchTool at build time."""
    d = _definition(name="knowledge_search")
    _, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "definition": d}]}, MagicMock()
    )
    assert err is not None and "knowledge_search" in err


def test_definition_input_schema_must_be_object_typed():
    d = _definition(input_schema={"type": "string"})
    _, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "definition": d}]}, MagicMock()
    )
    assert err is not None and "input_schema" in err


def test_definition_endpoint_must_be_http_url():
    for bad in ("ftp://x", "file:///etc/passwd", "api.example.com/cases", "", None):
        d = _definition(config={"endpoint": bad})
        _, err = validate_runtime_tools(
            {"runtime_tools": [{"type": "custom", "definition": d}]}, MagicMock()
        )
        assert err is not None and "endpoint" in err


def test_definition_unknown_config_keys_rejected():
    d = _definition(config={"endpoint": "https://x.example.com", "retries": 3})
    _, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "definition": d}]}, MagicMock()
    )
    assert err is not None and "retries" in err


def test_definition_method_whitelisted():
    d = _definition(config={"endpoint": "https://x.example.com", "method": "TRACE"})
    _, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "definition": d}]}, MagicMock()
    )
    assert err is not None and "TRACE" in err


def test_definition_unhashable_method_rejected_not_raised():
    """An unhashable 'method' (list/dict) must 400 like any other invalid
    value, not raise TypeError from `method not in _HTTP_METHODS` (a set)."""
    for bad in ([], {}, ["GET"]):
        d = _definition(config={"endpoint": "https://x.example.com", "method": bad})
        _, err = validate_runtime_tools(
            {"runtime_tools": [{"type": "custom", "definition": d}]}, MagicMock()
        )
        assert err is not None and "method" in err


def test_definition_timeout_bounds():
    for bad in (0, 601, True, "30"):
        d = _definition(config={"endpoint": "https://x.example.com", "timeout_seconds": bad})
        _, err = validate_runtime_tools(
            {"runtime_tools": [{"type": "custom", "definition": d}]}, MagicMock()
        )
        assert err is not None and "timeout_seconds" in err


def test_definition_header_value_errors_do_not_echo_the_value():
    """Headers carry secrets. A rejected header value must be named by KEY
    only — the message must not contain the value itself."""
    d = _definition(
        config={"endpoint": "https://x.example.com", "headers": {"Authorization": ["sk-oops"]}}
    )
    _, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "definition": d}]}, MagicMock()
    )
    assert err is not None and "Authorization" in err
    assert "sk-oops" not in err


def _mcp(**overrides):
    e = {"type": "mcp", "name": "github", "url": "https://mcp.example.com/mcp"}
    e.update(overrides)
    return e


def test_mcp_valid_entry_passes():
    """Transport is optional; the builder ignores it (matching attached
    servers, where only 'http' is honored today)."""
    configs, err = validate_runtime_tools({"runtime_tools": [_mcp()]}, MagicMock())
    assert err is None
    assert configs[0]["name"] == "github"


def test_mcp_name_regex_enforced():
    for bad in ("", "has space", "a" * 65, "semi;colon", None, 7):
        _, err = validate_runtime_tools({"runtime_tools": [_mcp(name=bad)]}, MagicMock())
        assert err is not None and "name" in err


def test_mcp_url_must_be_http():
    _, err = validate_runtime_tools({"runtime_tools": [_mcp(url="ws://x")]}, MagicMock())
    assert err is not None and "url" in err


def test_mcp_transport_values():
    _, err = validate_runtime_tools({"runtime_tools": [_mcp(transport="grpc")]}, MagicMock())
    assert err is not None and "transport" in err
    _, err = validate_runtime_tools({"runtime_tools": [_mcp(transport="http")]}, MagicMock())
    assert err is None


def test_mcp_header_value_errors_do_not_echo_the_value():
    _, err = validate_runtime_tools(
        {"runtime_tools": [_mcp(headers={"Authorization": 12345})]}, MagicMock()
    )
    assert err is not None and "Authorization" in err
    assert "12345" not in err


def test_duplicate_mcp_names_rejected():
    _, err = validate_runtime_tools({"runtime_tools": [_mcp(), _mcp()]}, MagicMock())
    assert err is not None and "github" in err


def test_final_name_collision_across_shapes_rejected():
    """An inline definition and a referenced ai.tools row resolving to the
    same name is ambiguous — reject, mirroring dup-id rejection."""
    entry_ref = {"type": "custom", "tool_id": _TOOL_1}
    entry_inline = {"type": "custom", "definition": _definition(name="case_lookup")}
    _, err = validate_runtime_tools(
        {"runtime_tools": [entry_ref, entry_inline]},
        _db_returning_tools([(_TOOL_1, "case_lookup")]),
    )
    assert err is not None and "case_lookup" in err


def test_collision_with_attached_tools_is_not_checked_here():
    """Validation has no attached-tool knowledge — a runtime tool named like
    an ATTACHED tool is the documented override case and passes validation."""
    configs, err = validate_runtime_tools(
        {"runtime_tools": [{"type": "custom", "definition": _definition(name="anything")}]},
        MagicMock(),
    )
    assert err is None
