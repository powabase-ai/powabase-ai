"""build_kb_tools_for_agent with per-request runtime KB configs."""

from unittest.mock import MagicMock

import pytest

from agentic_project_service.services import tool_registry


def _fake_kb(kb_id, name, retrieval_config=None, description=None):
    kb = MagicMock()
    kb.id = kb_id
    kb.name = name
    kb.retrieval_config = retrieval_config or {}
    kb.description = description
    return kb


def _fake_assignment(kb_id, config=None):
    a = MagicMock()
    a.knowledge_base_id = kb_id
    a.config = config or {}
    return a


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def filter_by(self, **_):
        return self

    def filter(self, *_):
        return self

    def all(self):
        return self._rows


@pytest.fixture
def env(monkeypatch):
    """Patch DB models, settings, and the search handler; return a mutable env."""
    state = {"assignments": [], "kbs": []}

    class FakeAssignmentModel:
        query = None

    class FakeKBModel:
        query = None
        id = MagicMock()  # KnowledgeBase.id.in_(...) must not explode

    def install():
        FakeAssignmentModel.query = _Query(state["assignments"])
        FakeKBModel.query = _Query(state["kbs"])

    state["install"] = install
    monkeypatch.setattr(tool_registry, "AgentKnowledgeBase", FakeAssignmentModel)
    monkeypatch.setattr(tool_registry, "KnowledgeBase", FakeKBModel)
    monkeypatch.setattr(tool_registry, "_make_search_handler", lambda db: lambda **kw: "ctx")
    monkeypatch.setattr(
        tool_registry,
        "get_setting",
        lambda key: {"KB_DEFAULT_TOP_K": 10, "KB_DEFAULT_MAX_CONTEXT_TOKENS": 16000}[key],
    )
    return state


def test_runtime_only_builds_tool_without_assignments(env):
    env["kbs"] = [_fake_kb("kb-r", "Contracts")]
    env["install"]()
    tools = tool_registry.build_kb_tools_for_agent(
        "agent-1", MagicMock(), runtime_kb_configs=[{"id": "kb-r", "top_k": 4}]
    )
    tool = tools["knowledge_search"]
    assert len(tool.knowledge_base_configs) == 1
    cfg = tool.knowledge_base_configs[0]
    assert cfg["id"] == "kb-r" and cfg["top_k"] == 4 and cfg["runtime"] is True
    assert tool.include_kb_filter is False  # single KB → no name filter


def test_runtime_merges_with_attached(env):
    env["assignments"] = [_fake_assignment("kb-a", {"top_k": 7})]
    env["kbs"] = [_fake_kb("kb-a", "Docs"), _fake_kb("kb-r", "Contracts")]
    env["install"]()
    tools = tool_registry.build_kb_tools_for_agent(
        "agent-1", MagicMock(), runtime_kb_configs=[{"id": "kb-r"}]
    )
    tool = tools["knowledge_search"]
    ids = {c["id"] for c in tool.knowledge_base_configs}
    assert ids == {"kb-a", "kb-r"}
    assert tool.include_kb_filter is True  # two KBs → name filter appears
    attached = next(c for c in tool.knowledge_base_configs if c["id"] == "kb-a")
    assert "runtime" not in attached


def test_runtime_entry_overrides_attached_config_for_same_kb(env):
    env["assignments"] = [_fake_assignment("kb-a", {"top_k": 7})]
    env["kbs"] = [_fake_kb("kb-a", "Docs")]
    env["install"]()
    tools = tool_registry.build_kb_tools_for_agent(
        "agent-1",
        MagicMock(),
        runtime_kb_configs=[{"id": "kb-a", "top_k": 2, "source_ids": ["s9"]}],
    )
    cfg = tools["knowledge_search"].knowledge_base_configs[0]
    assert cfg["top_k"] == 2 and cfg["source_ids"] == ["s9"] and cfg["runtime"] is True
    assert len(tools["knowledge_search"].knowledge_base_configs) == 1


def test_runtime_resolution_falls_back_to_kb_retrieval_config(env):
    env["kbs"] = [_fake_kb("kb-r", "Contracts", retrieval_config={"top_k": 5, "method": "hybrid"})]
    env["install"]()
    tools = tool_registry.build_kb_tools_for_agent(
        "agent-1", MagicMock(), runtime_kb_configs=[{"id": "kb-r"}]
    )
    cfg = tools["knowledge_search"].knowledge_base_configs[0]
    assert cfg["top_k"] == 5 and cfg["retrieval_method"] == "hybrid"


def test_no_assignments_and_no_runtime_returns_empty(env):
    env["install"]()
    assert tool_registry.build_kb_tools_for_agent("agent-1", MagicMock()) == {}


def test_load_all_tools_threads_runtime_configs(monkeypatch):
    seen = {}

    def fake_build(agent_id, db_session, runtime_kb_configs=None):
        seen["runtime"] = runtime_kb_configs
        return {}

    monkeypatch.setattr(tool_registry, "build_kb_tools_for_agent", fake_build)
    monkeypatch.setattr(tool_registry, "build_mcp_tools_for_agent", lambda *a, **k: {})
    monkeypatch.setattr(tool_registry, "_get_flask_app", lambda: None)

    class _EmptyQuery:
        def filter_by(self, **_):
            return self

        def all(self):
            return []

    class FakeAgentTool:
        query = _EmptyQuery()

    monkeypatch.setattr(tool_registry, "AgentTool", FakeAgentTool)
    tool_registry.load_all_tools_for_agent("agent-1", None, runtime_kb_configs=[{"id": "kb-r"}])
    assert seen["runtime"] == [{"id": "kb-r"}]
