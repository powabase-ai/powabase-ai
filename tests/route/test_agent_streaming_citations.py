"""Run-wide citation keys through /run/stream (ReAct branch).

``Agent.run`` is patched with a fake that calls the run's real tools:
* ``knowledge_search``: retrieval is replaced by a fake that uses the real
  registry callback and the real context formatter.
* MCP tools: the MCP client is replaced by canned results.
The fake emits the engine's ``tool_call`` events itself, the way the agent loop
does before running each tool.
"""

import json
import uuid

import pytest
from agentic.knowledge.models import RetrievedItem
from agentic.mcp.types import McpToolInfo
from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.services.knowledge_search import format_items_as_context
from tests.route.test_agent_streaming import _make_fake_agent_output, _parse_sse_events

_NS = uuid.UUID("6f1c2d3e-0000-4000-8000-000000000000")
SEARCH_RESULT = json.dumps(
    {
        "results": [
            {"name": "Item A", "link": "https://example.org/a"},
            {"name": "Item B", "link": "https://example.org/b"},
        ]
    }
)
MAPPING = {"search_items": {"items": "$.results[*]", "title": "name", "url": "link"}}


def _chunk_id(query: str, n: int) -> str:
    return str(uuid.uuid5(_NS, f"{query}-{n}"))


def _tool_call(context, tool_name: str, call_id: str, arguments: dict) -> None:
    context.emit_event(
        {
            "type": "tool_call",
            "step": 1,
            "tool_name": tool_name,
            "arguments": arguments,
            "call_id": call_id,
        }
    )


def _events(resp, name: str) -> list[dict]:
    return [e for e in _parse_sse_events(resp.get_data()) if e.get("event") == name]


def _rows(run_id: str) -> list[tuple]:
    return db.session.execute(
        text(
            "SELECT c.citation_key, c.cited, c.kind, c.knowledge_base_id::text "
            'FROM "ai".message_citations c JOIN "ai".agent_runs r ON r.id = c.run_id '
            "WHERE r.run_id = :rid ORDER BY c.citation_key"
        ),
        {"rid": run_id},
    ).all()


@pytest.fixture(autouse=True)
def _llm_ready(monkeypatch):
    # Only citation handling is under test, not whether an LLM key is configured.
    monkeypatch.setattr(
        "agentic_project_service.routes.agents.check_model_available", lambda model: None
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Use output.content as the final answer so the fake's markers survive.
    monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "false")


@pytest.fixture
def kb_agent(client, mock_auth, auth_headers, test_agent, test_knowledge_base):
    resp = client.post(
        f"/api/agents/{test_agent['id']}/knowledge-bases",
        json={"knowledge_base_id": test_knowledge_base["id"]},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    return {"agent_id": test_agent["id"], "kb_id": test_knowledge_base["id"]}


@pytest.fixture
def fake_retrieval(mocker):
    """Two chunks per query, labelled through the real registry callback."""

    def fake(
        db_session,
        query,
        knowledge_base_configs,
        max_context_tokens=None,
        session_history=None,
        register_items=None,
    ):
        kb_id = knowledge_base_configs[0]["id"]
        items = [
            RetrievedItem(
                item_id=_chunk_id(query, n),
                text=f"{query} fact {n}",
                score=1.0 - n / 10,
                source_id=None,
                knowledge_base_id=kb_id,
                meta={"source_name": f"{query}.txt"},
            )
            for n in (1, 2)
        ]
        labels = register_items(items) if register_items else None
        formatted, _ = format_items_as_context(items, group_by_document=False, labels=labels)
        handler_id = str(uuid.uuid5(_NS, f"handler-{query}"))
        db_session.execute(
            text(
                'INSERT INTO "ai".context_handlers (id, query, knowledge_base_configs) '
                "VALUES (:id, :q, '[]') ON CONFLICT (id) DO NOTHING"
            ),
            {"id": handler_id, "q": query},
        )
        return handler_id, {"formatted_context": formatted, "retrieved_context": [], "errors": []}

    mocker.patch(
        "agentic_project_service.services.tool_registry.create_and_execute", side_effect=fake
    )


@pytest.fixture
def fake_mcp(mocker):
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    mocker.patch(
        "agentic.mcp.client.discover_mcp_tools",
        return_value=[
            McpToolInfo(
                name="search_items", description="Search.", input_schema=schema, read_only_hint=True
            ),
            McpToolInfo(
                name="lookup", description="Look up.", input_schema=schema, read_only_hint=True
            ),
        ],
    )
    results = {"search_items": SEARCH_RESULT, "lookup": "Plain answer text."}
    mocker.patch(
        "agentic.mcp.client.call_mcp_tool", side_effect=lambda **kw: results[kw["tool_name"]]
    )


def _add_mcp_server(client, auth_headers, agent_id, mapping):
    resp = client.post(
        f"/api/agents/{agent_id}/mcp-servers",
        json={"name": "docs", "url": "https://mcp.example.com", "citation_mapping": mapping},
        headers=auth_headers,
    )
    assert resp.status_code == 201


def _stream(client, auth_headers, agent_id, **body):
    resp = client.post(
        f"/api/agents/{agent_id}/run/stream",
        json={"message": "question", **body},
        headers=auth_headers,
        buffered=True,
    )
    assert resp.status_code == 200
    return resp


def test_a_marker_on_the_second_search_resolves_to_the_second_search(
    client, kb_agent, fake_retrieval, auth_headers, mocker
):
    seen = {}

    def fake_run(messages, *, context=None, tools=None, **kwargs):
        search = tools["knowledge_search"]
        _tool_call(context, "knowledge_search", "call_a", {"query": "alpha"})
        seen["first"] = search.execute({"query": "alpha"}, context)
        _tool_call(context, "knowledge_search", "call_b", {"query": "beta"})
        seen["second"] = search.execute({"query": "beta"}, context)
        return _make_fake_agent_output(content="Beta holds [3]; nothing holds [99].")

    mocker.patch("agentic.agent.agent.Agent.run", side_effect=fake_run)
    resp = _stream(client, auth_headers, kb_agent["agent_id"], citations_enabled=True)

    assert "[1]" in seen["first"] and "[2]" in seen["first"]
    assert "[3]" in seen["second"] and "[4]" in seen["second"]
    assert "[1]" not in seen["second"]

    registered = _events(resp, "citation_registered")
    assert [e["key"] for e in registered] == [1, 2, 3, 4]
    assert {e["kind"] for e in registered} == {"kb_chunk"}
    assert registered[2]["title"] == "beta.txt"

    [complete] = _events(resp, "complete")
    assert complete["content"] == "Beta holds [3]; nothing holds ."
    [cite] = complete["citations"]
    assert (cite["key"], cite["item_id"], cite["kind"]) == ("3", _chunk_id("beta", 1), "kb_chunk")
    assert (cite["tool_name"], cite["call_id"], cite["cited"]) == (
        "knowledge_search",
        "call_b",
        True,
    )
    assert cite["knowledge_base_id"] == kb_agent["kb_id"]

    run_id = _events(resp, "start")[0]["run_id"]
    assert _rows(run_id) == [
        (1, False, "kb_chunk", kb_agent["kb_id"]),
        (2, False, "kb_chunk", kb_agent["kb_id"]),
        (3, True, "kb_chunk", kb_agent["kb_id"]),
        (4, False, "kb_chunk", kb_agent["kb_id"]),
    ]


def test_prefetched_context_keeps_its_labels_and_tools_continue_after_it(
    client, kb_agent, fake_retrieval, auth_headers, mocker
):
    seen = {}

    def fake_run(messages, *, context=None, tools=None, **kwargs):
        _tool_call(context, "knowledge_search", "call_a", {"query": "alpha"})
        seen["search"] = tools["knowledge_search"].execute({"query": "alpha"}, context)
        return _make_fake_agent_output(content="Seed [1], alpha [3].")

    mocker.patch("agentic.agent.agent.Agent.run", side_effect=fake_run)
    resp = _stream(
        client,
        auth_headers,
        kb_agent["agent_id"],
        citations_enabled=True,
        context_items=[{"text": "Seed fact."}],
    )

    assert "[2]" in seen["search"] and "[3]" in seen["search"]
    assert "[1]" not in seen["search"]
    [complete] = _events(resp, "complete")
    seed, chunk = complete["citations"]
    assert (seed["key"], seed["text_excerpt"], seed["tool_name"]) == ("1", "Seed fact.", None)
    assert (chunk["key"], chunk["item_id"]) == ("3", _chunk_id("alpha", 2))


def test_mapped_mcp_items_and_unmapped_calls_get_run_keys(
    client, mock_auth, test_agent, fake_mcp, auth_headers, mocker
):
    _add_mcp_server(client, auth_headers, test_agent["id"], MAPPING)
    seen = {}

    def fake_run(messages, *, context=None, tools=None, **kwargs):
        _tool_call(context, "mcp__docs__search_items", "call_s", {"q": "a"})
        seen["search"] = tools["mcp__docs__search_items"].execute({"q": "a"}, context)
        _tool_call(context, "mcp__docs__lookup", "call_l", {})
        seen["lookup"] = tools["mcp__docs__lookup"].execute({}, context)
        return _make_fake_agent_output(content="A is first [1]. Plain [3].")

    mocker.patch("agentic.agent.agent.Agent.run", side_effect=fake_run)
    resp = _stream(client, auth_headers, test_agent["id"], citations_enabled=True)

    assert [r["cite"] for r in json.loads(seen["search"])["results"]] == ["[1]", "[2]"]
    assert seen["lookup"] == "[3] Plain answer text."
    registered = _events(resp, "citation_registered")
    assert [(e["key"], e["kind"], e["title"], e["url"]) for e in registered] == [
        (1, "tool_item", "Item A", "https://example.org/a"),
        (2, "tool_item", "Item B", "https://example.org/b"),
        (3, "tool_call", "mcp__docs__lookup {}", None),
    ]
    [complete] = _events(resp, "complete")
    item, call = complete["citations"]
    assert (item["key"], item["call_id"], item["meta"]) == (
        "1",
        "call_s",
        {"name": "Item A", "link": "https://example.org/a"},
    )
    assert (call["key"], call["meta"]) == ("3", {"raw": "Plain answer text.", "arguments": {}})

    run_id = _events(resp, "start")[0]["run_id"]
    units = client.get(f"/api/agents/runs/{run_id}", headers=auth_headers).get_json()[
        "citation_units"
    ]
    assert [(u["key"], u["cited"]) for u in units] == [("1", True), ("2", False), ("3", True)]


def test_a_server_without_a_mapping_is_not_keyed_and_nothing_is_stripped(
    client, mock_auth, test_agent, fake_mcp, auth_headers, mocker
):
    _add_mcp_server(client, auth_headers, test_agent["id"], None)
    seen = {}

    def fake_run(messages, *, context=None, tools=None, **kwargs):
        _tool_call(context, "mcp__docs__lookup", "call_l", {})
        seen["lookup"] = tools["mcp__docs__lookup"].execute({}, context)
        return _make_fake_agent_output(content="Plain [1].")

    mocker.patch("agentic.agent.agent.Agent.run", side_effect=fake_run)
    resp = _stream(client, auth_headers, test_agent["id"], citations_enabled=True)

    assert seen["lookup"] == "Plain answer text."
    assert _events(resp, "citation_registered") == []
    [complete] = _events(resp, "complete")
    assert complete["content"] == "Plain [1]."
    assert "citations" not in complete
    assert _rows(_events(resp, "start")[0]["run_id"]) == []


def test_citations_off_leaves_mapped_results_alone(
    client, mock_auth, test_agent, fake_mcp, auth_headers, mocker
):
    _add_mcp_server(client, auth_headers, test_agent["id"], MAPPING)
    seen = {}

    def fake_run(messages, *, context=None, tools=None, **kwargs):
        _tool_call(context, "mcp__docs__search_items", "call_s", {"q": "a"})
        seen["search"] = tools["mcp__docs__search_items"].execute({"q": "a"}, context)
        return _make_fake_agent_output(content="A [1].")

    mocker.patch("agentic.agent.agent.Agent.run", side_effect=fake_run)
    resp = _stream(client, auth_headers, test_agent["id"])

    assert seen["search"] == SEARCH_RESULT
    assert _events(resp, "citation_registered") == []
    assert _rows(_events(resp, "start")[0]["run_id"]) == []


def test_a_citation_storage_failure_does_not_fail_a_completed_run(
    client, kb_agent, fake_retrieval, auth_headers, mocker
):
    def fake_run(messages, *, context=None, tools=None, **kwargs):
        _tool_call(context, "knowledge_search", "call_a", {"query": "alpha"})
        tools["knowledge_search"].execute({"query": "alpha"}, context)
        return _make_fake_agent_output(content="Alpha [1].")

    mocker.patch("agentic.agent.agent.Agent.run", side_effect=fake_run)
    mocker.patch(
        "agentic_project_service.routes.agents.persist_citations",
        side_effect=RuntimeError("storage down"),
    )
    resp = _stream(client, auth_headers, kb_agent["agent_id"], citations_enabled=True)

    assert _events(resp, "error") == []
    [complete] = _events(resp, "complete")
    assert complete["status"] == "completed"
    run_id = _events(resp, "start")[0]["run_id"]
    run = client.get(f"/api/agents/runs/{run_id}", headers=auth_headers).get_json()
    assert run["status"] == "completed"
