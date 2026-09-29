"""The tool loader: who the tools act for, and the agent login's grants.

Two properties the loader owns:

* ``caller`` is required, so a new call site cannot forget to say who a run
  acts for (the tools would refuse, silently to the operator).
* A service-role run's database tools act as the agent's own login, whose
  grants are synced first. If that sync fails the tools refuse, instead of
  running with whatever grants the login had before.
"""

import json
import types
from unittest.mock import MagicMock

import pytest

from agentic_project_service.services import tool_registry
from agentic_project_service.services.tool_caller import ToolCaller

AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"
SERVICE = ToolCaller.service()
USER = ToolCaller(claims={"sub": "8d2f6b4e-1a70-4e11-9a3c-3f9a1c2e5b7d"}, token="user-jwt")

ASSIGNMENTS = [
    types.SimpleNamespace(
        tool_type="builtin",
        tool_name="database_query",
        config_override={"schemas": {"public": ["customers", "orders"]}},
    ),
    types.SimpleNamespace(
        tool_type="builtin",
        tool_name="database_write",
        config_override={"schemas": {"public": ["orders"], "empty": []}},
    ),
    types.SimpleNamespace(tool_type="builtin", tool_name="web_search", config_override={}),
]


class _Ref:
    value = ASSIGNMENTS


ASSIGNMENTS_REF = _Ref()


@pytest.fixture
def loader(monkeypatch):
    class _Query:
        def filter_by(self, **_):
            return self

        def all(self):
            return ASSIGNMENTS_REF.value

    class _FakeAgentTool:
        query = _Query()

    ran: list[str] = []

    def recorder(name):
        def handler(arguments, context):
            ran.append(name)
            return "[]"

        return handler

    monkeypatch.setattr(tool_registry, "AgentTool", _FakeAgentTool)
    monkeypatch.setattr(tool_registry, "_get_flask_app", lambda: None)
    monkeypatch.setattr(tool_registry, "_ensure_app_context", lambda h, app: h)
    monkeypatch.setattr(tool_registry, "_wrap_handler_with_billing", lambda h, name: h)
    monkeypatch.setattr(tool_registry, "_introspect_table_metadata", lambda *a: "tables")
    monkeypatch.setattr(tool_registry, "build_kb_tools_for_agent", lambda *a, **k: {})
    monkeypatch.setattr(tool_registry, "build_mcp_tools_for_agent", lambda *a, **k: {})
    for name in ("database_query", "database_write", "web_search"):
        monkeypatch.setitem(tool_registry.BUILTIN_HANDLERS, name, recorder(name))
    sync = MagicMock()
    monkeypatch.setattr(tool_registry.agent_sql, "sync_agent_role", sync)
    return types.SimpleNamespace(ran=ran, sync=sync)


class TestCallerIsRequired:
    def test_omitting_the_caller_is_a_type_error(self, loader):
        with pytest.raises(TypeError, match="caller"):
            tool_registry.load_all_tools_for_agent(AGENT_ID, db_session=None)

    def test_an_explicit_none_is_allowed_and_syncs_nothing(self, loader):
        tool_registry.load_all_tools_for_agent(AGENT_ID, db_session=None, caller=None)
        loader.sync.assert_not_called()


class TestAgentLoginSync:
    def test_service_run_syncs_exactly_the_configured_tables(self, loader):
        tool_registry.load_all_tools_for_agent(AGENT_ID, db_session=None, caller=SERVICE)
        loader.sync.assert_called_once_with(
            AGENT_ID,
            read_tables={"public": ["customers", "orders"]},
            write_tables={"public": ["orders"]},
        )

    def test_an_agent_without_database_tools_is_never_synced(self, loader, monkeypatch):
        """Only agents with database tools get a login; the rest cost no catalog work."""
        monkeypatch.setattr(ASSIGNMENTS_REF, "value", ASSIGNMENTS[2:])
        tool_registry.load_all_tools_for_agent(AGENT_ID, db_session=None, caller=SERVICE)
        loader.sync.assert_not_called()

    def test_end_user_run_does_not_sync(self, loader):
        tool_registry.load_all_tools_for_agent(AGENT_ID, db_session=None, caller=USER)
        loader.sync.assert_not_called()

    def test_a_failed_sync_makes_the_database_tools_refuse(self, loader, caplog):
        loader.sync.side_effect = RuntimeError("permission denied to create role")
        tools = tool_registry.load_all_tools_for_agent(AGENT_ID, db_session=None, caller=SERVICE)

        query = json.loads(tools["database_query"].handler({"query": "SELECT 1"}, None))
        write = json.loads(tools["database_write"].handler({"table": "orders"}, None))

        assert "permissions could not be prepared" in query["error"]
        assert write["success"] is False and "permissions" in write["message"]
        assert loader.ran == []
        assert AGENT_ID in caplog.text

    def test_a_failed_sync_leaves_other_tools_working(self, loader):
        loader.sync.side_effect = RuntimeError("boom")
        tools = tool_registry.load_all_tools_for_agent(AGENT_ID, db_session=None, caller=SERVICE)
        tools["web_search"].handler({"query": "x"}, None)
        assert loader.ran == ["web_search"]

    def test_sync_agent_database_role_reads_the_assignments(self, loader):
        tool_registry.sync_agent_database_role(AGENT_ID)
        loader.sync.assert_called_once_with(
            AGENT_ID,
            read_tables={"public": ["customers", "orders"]},
            write_tables={"public": ["orders"]},
        )
