"""Runtime tools flow to every orchestration sub-agent's tool build."""

from tests.unit.test_runtime_kb_orchestration import (
    _install_fakes,
    _two_agent_entities,
)

from agentic_project_service.services import orchestration as orch_service


def test_build_orchestration_threads_runtime_tool_configs_to_both_entities(monkeypatch):
    entities, agent_rows = _two_agent_entities()
    load_tools_calls: list = []
    _install_fakes(monkeypatch, entities, agent_rows, load_tools_calls)

    runtime_tool_configs = [{"type": "builtin", "name": "web_search"}]
    orch_service.build_orchestration("orch-1", runtime_tool_configs=runtime_tool_configs)

    assert len(load_tools_calls) == 2
    for _agent_id, kwargs in load_tools_calls:
        assert kwargs["runtime_tool_configs"] == runtime_tool_configs


def test_build_orchestration_defaults_runtime_tool_configs_to_none(monkeypatch):
    entities, agent_rows = _two_agent_entities()
    load_tools_calls: list = []
    _install_fakes(monkeypatch, entities, agent_rows, load_tools_calls)

    orch_service.build_orchestration("orch-1")

    assert len(load_tools_calls) == 2
    for _agent_id, kwargs in load_tools_calls:
        assert kwargs["runtime_tool_configs"] is None
