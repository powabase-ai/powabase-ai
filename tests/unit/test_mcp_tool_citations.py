"""MCP tools built for an agent key their results only when the server opted in
(citation_mapping not null) and the run collects citations (a registry is bound)."""

import json
import logging
from types import SimpleNamespace

import pytest
from agentic.mcp.types import McpToolInfo

from agentic_project_service.services import tool_registry
from agentic_project_service.services.citation_registry import (
    CitationRegistry,
    citation_registry_var,
    current_tool_call_id_var,
)

SEARCH = json.dumps({"results": [{"name": "Item A", "link": "https://example.org/a"}]})
RESULTS = {"search_items": SEARCH, "lookup": "Plain answer text."}


class _Query:
    def __init__(self, servers):
        self._servers = servers

    def filter_by(self, **_):
        return self

    def all(self):
        return self._servers


def _build(monkeypatch, citation_mapping):
    server = SimpleNamespace(
        name="docs", url="https://mcp.example.com", headers={}, citation_mapping=citation_mapping
    )
    monkeypatch.setattr(tool_registry, "AgentMcpServer", SimpleNamespace(query=_Query([server])))
    schema = {"type": "object", "properties": {}}
    monkeypatch.setattr(
        "agentic.mcp.client.discover_mcp_tools",
        lambda url, headers, **kw: [
            McpToolInfo(name="search_items", description="Search.", input_schema=schema),
            McpToolInfo(name="lookup", description="Look up.", input_schema=schema),
        ],
    )
    monkeypatch.setattr(
        "agentic.mcp.client.call_mcp_tool", lambda **kw: RESULTS[kw["tool_name"]]
    )
    return tool_registry.build_mcp_tools_for_agent("agent-x", db_session=None)


@pytest.fixture
def registry():
    reg = CitationRegistry()
    reg_token = citation_registry_var.set(reg)
    call_token = current_tool_call_id_var.set("call_7")
    yield reg
    current_tool_call_id_var.reset(call_token)
    citation_registry_var.reset(reg_token)


def test_a_server_without_a_mapping_is_never_keyed(monkeypatch, registry):
    tools = _build(monkeypatch, None)
    assert tools["mcp__docs__lookup"].execute({}, None) == "Plain answer text."
    assert len(registry) == 0


def test_a_mapped_server_is_not_keyed_when_the_run_has_no_registry(monkeypatch):
    tools = _build(monkeypatch, {"search_items": {"items": "$.results[*]"}})
    assert tools["mcp__docs__search_items"].execute({}, None) == SEARCH


def test_mapped_and_unmapped_tools_of_an_opted_in_server(monkeypatch, registry):
    tools = _build(monkeypatch, {"search_items": {"items": "$.results[*]", "title": "name"}})
    keyed = json.loads(tools["mcp__docs__search_items"].execute({"q": "a"}, None))
    assert keyed["results"][0]["cite"] == "[1]"
    assert tools["mcp__docs__lookup"].execute({}, None) == "[2] Plain answer text."
    units = registry.citation_map()
    assert (units["1"]["tool_name"], units["1"]["title"], units["1"]["call_id"]) == (
        "mcp__docs__search_items",
        "Item A",
        "call_7",
    )
    assert (units["2"]["kind"], units["2"]["tool_name"]) == ("tool_call", "mcp__docs__lookup")


def test_an_empty_mapping_keys_every_tool_by_call(monkeypatch, registry):
    tools = _build(monkeypatch, {})
    assert tools["mcp__docs__search_items"].execute({}, None) == f"[1] {SEARCH}"


def test_an_invalid_stored_mapping_is_ignored_loudly(monkeypatch, registry, caplog):
    with caplog.at_level(logging.WARNING):
        tools = _build(monkeypatch, {"search_items": {"items": "$..x"}})
    assert tools["mcp__docs__search_items"].execute({}, None) == SEARCH
    assert len(registry) == 0
    assert "citation_mapping" in caplog.text and "docs" in caplog.text


def test_a_whole_call_unit_caps_raw_at_the_tools_result_limit(monkeypatch, registry):
    tools = _build(monkeypatch, {})
    lookup = tools["mcp__docs__lookup"]
    lookup.max_result_chars = 5
    assert lookup.execute({}, None) == "[1] Plain answer text."
    meta = registry.citation_map()["1"]["meta"]
    assert (meta["raw"], meta["raw_truncated"]) == ("Plain", True)
