"""build_runtime_tools + the loader's runtime_tool_configs merge."""

from unittest.mock import MagicMock

import pytest

from agentic_project_service.services import tool_registry


@pytest.fixture
def registry_env(monkeypatch):
    """No Flask app, no app-context wrapping, inert settings."""
    monkeypatch.setattr(tool_registry, "_get_flask_app", lambda: None)
    monkeypatch.setattr(tool_registry, "_ensure_app_context", lambda h, app: h)
    monkeypatch.setattr(
        tool_registry,
        "get_setting",
        lambda key: {"CUSTOM_TOOL_TIMEOUT": 30}.get(key, 1000),
    )


def test_runtime_builtin_builds_with_forced_args(registry_env, monkeypatch):
    """config_override must actually reach the handler as a forced argument,
    not just be accepted and ignored."""
    seen_args = []

    def stub_handler(arguments, context):
        seen_args.append(dict(arguments))
        return "ok"

    monkeypatch.setitem(tool_registry.BUILTIN_HANDLERS, "web_search", stub_handler)

    tools = tool_registry.build_runtime_tools(
        "agent-1",
        MagicMock(),
        [{"type": "builtin", "name": "web_search", "config_override": {"search_type": "deep"}}],
    )
    assert set(tools) == {"web_search"}
    assert tools["web_search"].input_schema  # real builtin schema came through

    tools["web_search"].execute({"query": "test"}, None)
    assert seen_args and seen_args[0]["search_type"] == "deep"


def test_runtime_custom_by_reference_builds_from_db_row(registry_env):
    tool_row = MagicMock()
    tool_row.name = "case_lookup"
    tool_row.description = "desc"
    tool_row.input_schema = {"type": "object", "properties": {}}
    tool_row.config = {"endpoint": "https://api.example.com/cases", "method": "GET"}
    db = MagicMock()
    db.get.return_value = tool_row
    tools = tool_registry.build_runtime_tools(
        "agent-1",
        db,
        [
            {
                "type": "custom",
                "tool_id": "11111111-1111-1111-1111-111111111111",
                "_resolved_name": "case_lookup",
            }
        ],
    )
    assert set(tools) == {"case_lookup"}
    assert tools["case_lookup"].endpoint == "https://api.example.com/cases"
    assert tools["case_lookup"].method == "GET"


def test_runtime_custom_reference_missing_at_build_time_logs_and_drops(registry_env, caplog):
    """TOCTOU: the tool row was deleted between validation and build."""
    import logging

    db = MagicMock()
    db.get.return_value = None
    tool_id = "11111111-1111-1111-1111-111111111111"
    with caplog.at_level(logging.WARNING, logger=tool_registry.__name__):
        tools = tool_registry.build_runtime_tools(
            "agent-1", db, [{"type": "custom", "tool_id": tool_id, "_resolved_name": "x"}]
        )
    assert tools == {}
    assert any(tool_id in r.getMessage() for r in caplog.records)


def test_runtime_custom_inline_definition_builds(registry_env):
    definition = {
        "name": "case_lookup",
        "description": "desc",
        "input_schema": {"type": "object", "properties": {"docket": {"type": "string"}}},
        "config": {
            "endpoint": "https://api.example.com/cases",
            "headers": {"Authorization": "Bearer sk-test"},
            "timeout_seconds": 12,
        },
    }
    tools = tool_registry.build_runtime_tools(
        "agent-1", MagicMock(), [{"type": "custom", "definition": definition}]
    )
    tool = tools["case_lookup"]
    assert tool.endpoint == "https://api.example.com/cases"
    assert tool.headers == {"Authorization": "Bearer sk-test"}
    assert tool.timeout == 12


def _loader_env(monkeypatch, assignments):
    class _Query:
        def filter_by(self, **_):
            return self

        def all(self):
            return assignments

    class _FakeAgentTool:
        query = _Query()

    monkeypatch.setattr(tool_registry, "AgentTool", _FakeAgentTool)
    monkeypatch.setattr(tool_registry, "_get_flask_app", lambda: None)
    monkeypatch.setattr(tool_registry, "_ensure_app_context", lambda h, app: h)
    monkeypatch.setattr(tool_registry, "build_kb_tools_for_agent", lambda *a, **k: {})
    monkeypatch.setattr(tool_registry, "build_mcp_tools_for_agent", lambda *a, **k: {})


def test_loader_threads_runtime_tool_configs(monkeypatch):
    _loader_env(monkeypatch, [])
    seen = {}

    def fake_build_runtime_tools(agent_id, db_session, configs, **kwargs):
        seen["configs"] = configs
        return {}

    monkeypatch.setattr(tool_registry, "build_runtime_tools", fake_build_runtime_tools)
    cfg = [{"type": "builtin", "name": "web_search"}]
    tool_registry.load_all_tools_for_agent("agent-1", None, runtime_tool_configs=cfg)
    assert seen["configs"] == cfg


def test_loader_skips_runtime_build_when_unset(monkeypatch):
    _loader_env(monkeypatch, [])

    def boom(*a, **k):
        raise AssertionError("build_runtime_tools must not run without configs")

    monkeypatch.setattr(tool_registry, "build_runtime_tools", boom)
    assert tool_registry.load_all_tools_for_agent("agent-1", None) == {}


def test_runtime_tool_overrides_attached_tool_with_same_name(monkeypatch):
    """Override-on-duplicate by final tool name: an attached web_search loses
    to the runtime web_search for this run."""
    import types

    assignment = types.SimpleNamespace(
        tool_type="builtin", tool_name="web_search", config_override=None
    )
    _loader_env(monkeypatch, [assignment])

    runtime_marker = object()

    def fake_build_runtime_tools(agent_id, db_session, configs, **kwargs):
        return {"web_search": runtime_marker}

    monkeypatch.setattr(tool_registry, "build_runtime_tools", fake_build_runtime_tools)
    tools = tool_registry.load_all_tools_for_agent(
        "agent-1",
        None,
        runtime_tool_configs=[{"type": "builtin", "name": "web_search"}],
    )
    assert tools["web_search"] is runtime_marker


def _fake_discovered(name):
    t = MagicMock()
    t.name = name
    t.description = "d"
    t.input_schema = {"type": "object", "properties": {}}
    t.read_only_hint = True
    t.destructive_hint = False
    return t


def test_runtime_mcp_entry_discovers_and_namespaces_tools(registry_env, monkeypatch):
    monkeypatch.setattr(
        "agentic.mcp.client.discover_mcp_tools",
        lambda url, headers: [_fake_discovered("list_issues")],
    )
    tools = tool_registry.build_runtime_tools(
        "agent-1",
        MagicMock(),
        [
            {
                "type": "mcp",
                "name": "github",
                "url": "https://mcp.example.com/mcp",
                "headers": {"Authorization": "Bearer sk-mcp-secret"},
            }
        ],
    )
    assert set(tools) == {"mcp__github__list_issues"}
    assert tools["mcp__github__list_issues"].server_headers == {
        "Authorization": "Bearer sk-mcp-secret"
    }


def test_runtime_mcp_discovery_failure_skips_and_never_logs_headers(
    registry_env, monkeypatch, caplog
):
    """Soft-fail like attached servers — and the warning must not leak the
    Authorization header value."""
    import logging

    def boom(url, headers):
        raise ConnectionError("no route to host")

    monkeypatch.setattr("agentic.mcp.client.discover_mcp_tools", boom)
    with caplog.at_level(logging.WARNING, logger=tool_registry.__name__):
        tools = tool_registry.build_runtime_tools(
            "agent-1",
            MagicMock(),
            [
                {
                    "type": "mcp",
                    "name": "github",
                    "url": "https://mcp.example.com/mcp",
                    "headers": {"Authorization": "Bearer sk-mcp-secret"},
                }
            ],
        )
    assert tools == {}
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "github" in joined
    assert "sk-mcp-secret" not in joined
