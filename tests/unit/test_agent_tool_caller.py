"""Agent tools that touch project data act as the run's caller.

The database tools hand their SQL to ``agent_sql`` together with the caller
(an end user, or the service role) and the agent, instead of running it on
the service's superuser session. The storage tools call Storage with the end
user's own token, and never reach the internal sources bucket. With no caller
at all — a code path that forgot to say who the run is for — every one of
them refuses.

What the database does with the caller is pinned against a real Postgres in
tests/test_agent_sql_store.py.
"""

import json
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from agentic_project_service.services import tool_registry
from agentic_project_service.services.agent_sql import AgentSqlRejected
from agentic_project_service.services.storage import SOURCES_BUCKET
from agentic_project_service.services.tool_caller import ToolCaller
from agentic_project_service.tools import builtin
from agentic_project_service.tools.builtin import (
    database_query_handler,
    database_write_handler,
    storage_read_handler,
    storage_write_handler,
)

AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"
USER = ToolCaller(claims={"sub": "user-1", "role": "authenticated"}, token="user-jwt")
SERVICE = ToolCaller.service()
SCHEMAS = {"public": ["customers"]}


def _query_args(caller, **extra):
    return {
        "query": "SELECT * FROM customers",
        "_caller": caller,
        "_agent_id": AGENT_ID,
        "_schemas_config": SCHEMAS,
        "_allowed_schemas": ["public"],
        "_allowed_tables": {"customers"},
        **extra,
    }


# ---------------------------------------------------------------------------
# database_query
# ---------------------------------------------------------------------------


class TestDatabaseQuery:
    def test_runs_through_agent_sql_as_the_caller(self):
        with patch.object(builtin.agent_sql, "run_query", return_value=[{"id": 1}]) as run:
            result = json.loads(database_query_handler(_query_args(USER), None))
        assert result == [{"id": 1}]
        run.assert_called_once_with(USER, AGENT_ID, "SELECT * FROM customers", SCHEMAS)

    def test_the_tools_module_has_no_handle_on_the_service_session(self):
        """The service's session is a superuser; the tools must not be able to reach it."""
        assert not hasattr(builtin, "db")

    def test_a_rejected_query_is_reported(self):
        with patch.object(
            builtin.agent_sql,
            "run_query",
            side_effect=AgentSqlRejected("Function set_config is not allowed"),
        ):
            result = json.loads(database_query_handler(_query_args(USER), None))
        assert result == {"error": "Function set_config is not allowed"}

    def test_without_a_caller_it_refuses(self):
        with patch.object(builtin.agent_sql, "run_query") as run:
            result = json.loads(database_query_handler(_query_args(None), None))
        assert "error" in result
        run.assert_not_called()

    def test_a_caller_the_model_supplies_is_not_trusted(self):
        """Tool arguments come from the model; only the loader's injection counts."""
        args = _query_args(None)
        args["_caller"] = {"claims": None}  # not a ToolCaller
        with patch.object(builtin.agent_sql, "run_query") as run:
            result = json.loads(database_query_handler(args, None))
        assert "error" in result
        run.assert_not_called()


# ---------------------------------------------------------------------------
# database_write
# ---------------------------------------------------------------------------


def _fake_transaction(conn):
    calls = []

    @contextmanager
    def transaction(caller, agent_id, schemas, *, read_only):
        calls.append((caller, agent_id, list(schemas), read_only))
        yield conn

    return transaction, calls


class TestDatabaseWrite:
    def _args(self, caller):
        return {
            "table": "customers",
            "operation": "insert",
            "data": {"name": "widget"},
            "_caller": caller,
            "_agent_id": AGENT_ID,
            "_schemas_config": SCHEMAS,
            "_allowed_schemas": ["public"],
            "_allowed_tables": {"customers"},
        }

    def test_writes_in_a_caller_transaction(self):
        conn = MagicMock()
        conn.execute.return_value.rowcount = 1
        conn.execute.return_value.__iter__.return_value = iter([])
        transaction, calls = _fake_transaction(conn)
        with patch.object(builtin.agent_sql, "agent_transaction", transaction):
            result = json.loads(database_write_handler(self._args(USER), None))
        assert result["success"] is True
        assert calls == [(USER, AGENT_ID, ["public"], False)]
        assert any("INSERT INTO" in str(c.args[0]) for c in conn.execute.call_args_list)

    def test_a_database_error_is_reported(self):
        @contextmanager
        def failing(*a, **k):
            raise RuntimeError("new row violates row-level security policy")
            yield  # pragma: no cover

        with patch.object(builtin.agent_sql, "agent_transaction", failing):
            result = json.loads(database_write_handler(self._args(USER), None))
        assert result["success"] is False
        assert "row-level security" in result["message"]

    def test_without_a_caller_it_refuses(self):
        with patch.object(builtin.agent_sql, "agent_transaction") as transaction:
            result = json.loads(database_write_handler(self._args(None), None))
        assert result["success"] is False
        transaction.assert_not_called()


# ---------------------------------------------------------------------------
# storage_read / storage_write
# ---------------------------------------------------------------------------


def _list_response():
    response = MagicMock(status_code=200)
    response.json.return_value = [{"name": "a.txt"}]
    return response


class TestStorageTools:
    def test_end_user_reads_with_their_own_token(self):
        user_storage = MagicMock()
        user_storage._request.return_value = _list_response()
        with (
            patch.object(builtin, "get_storage_for_user", return_value=user_storage) as for_user,
            patch.object(builtin, "get_storage") as service_storage,
        ):
            result = json.loads(
                storage_read_handler(
                    {"operation": "list", "bucket": "docs", "path": "", "_caller": USER}, None
                )
            )
        assert result["objects"] == [{"name": "a.txt"}]
        for_user.assert_called_once_with("user-jwt")
        service_storage.assert_not_called()

    def test_end_user_writes_with_their_own_token(self):
        user_storage = MagicMock()
        user_storage.upload.return_value = "docs/a.txt"
        with (
            patch.object(builtin, "get_storage_for_user", return_value=user_storage),
            patch.object(builtin, "get_storage") as service_storage,
        ):
            storage_write_handler(
                {"bucket": "docs", "path": "a.txt", "content": "hi", "_caller": USER}, None
            )
        user_storage.upload.assert_called_once()
        service_storage.assert_not_called()

    def test_service_run_uses_the_service_client(self):
        service_storage = MagicMock()
        service_storage._request.return_value = _list_response()
        with patch.object(builtin, "get_storage", return_value=service_storage):
            result = json.loads(
                storage_read_handler(
                    {"operation": "list", "bucket": "docs", "path": "", "_caller": SERVICE}, None
                )
            )
        assert result["objects"] == [{"name": "a.txt"}]

    @pytest.mark.parametrize("caller", [SERVICE, USER])
    @pytest.mark.parametrize(
        "call",
        [
            lambda c: storage_read_handler(
                {"operation": "list", "bucket": SOURCES_BUCKET, "path": "", "_caller": c}, None
            ),
            lambda c: storage_read_handler(
                {"operation": "download", "bucket": SOURCES_BUCKET, "path": "x.pdf", "_caller": c},
                None,
            ),
            lambda c: storage_write_handler(
                {"bucket": SOURCES_BUCKET, "path": "x.txt", "content": "x", "_caller": c}, None
            ),
        ],
    )
    def test_the_internal_sources_bucket_is_never_reachable(self, caller, call):
        with (
            patch.object(builtin, "get_storage") as service_storage,
            patch.object(builtin, "get_storage_for_user") as user_storage,
        ):
            result = json.loads(call(caller))
        assert "error" in result
        service_storage.assert_not_called()
        user_storage.assert_not_called()

    def test_without_a_caller_storage_refuses(self):
        with patch.object(builtin, "get_storage") as service_storage:
            result = json.loads(
                storage_read_handler({"operation": "list", "bucket": "docs", "path": ""}, None)
            )
        assert "error" in result
        service_storage.assert_not_called()


# ---------------------------------------------------------------------------
# The loader: who the tools act for, and the agent role's grants
# ---------------------------------------------------------------------------


@pytest.fixture
def loader(monkeypatch):
    """load_all_tools_for_agent with one database_query, one database_write and
    one storage_read assignment, the DB-backed builders stubbed out."""
    assignments = [
        types.SimpleNamespace(
            tool_type="builtin",
            tool_name="database_query",
            config_override={"schemas": {"public": ["customers", "orders"]}},
        ),
        types.SimpleNamespace(
            tool_type="builtin",
            tool_name="database_write",
            config_override={"schemas": {"public": ["orders"]}},
        ),
        types.SimpleNamespace(tool_type="builtin", tool_name="storage_read", config_override={}),
    ]

    class _Query:
        def filter_by(self, **_):
            return self

        def all(self):
            return assignments

    class _FakeAgentTool:
        query = _Query()

    seen: dict[str, dict] = {}

    def recorder(name):
        def handler(arguments, context):
            seen[name] = dict(arguments)
            return "[]"

        return handler

    monkeypatch.setattr(tool_registry, "AgentTool", _FakeAgentTool)
    monkeypatch.setattr(tool_registry, "_get_flask_app", lambda: None)
    monkeypatch.setattr(tool_registry, "_ensure_app_context", lambda h, app: h)
    monkeypatch.setattr(tool_registry, "_wrap_handler_with_billing", lambda h, name: h)
    monkeypatch.setattr(tool_registry, "_introspect_table_metadata", lambda *a: "tables")
    monkeypatch.setattr(tool_registry, "build_kb_tools_for_agent", lambda *a, **k: {})
    monkeypatch.setattr(tool_registry, "build_mcp_tools_for_agent", lambda *a, **k: {})
    for name in ("database_query", "database_write", "storage_read"):
        monkeypatch.setitem(tool_registry.BUILTIN_HANDLERS, name, recorder(name))
    sync = MagicMock()
    monkeypatch.setattr(tool_registry.agent_sql, "sync_agent_role", sync)
    return types.SimpleNamespace(seen=seen, sync=sync)


class TestLoaderCarriesTheCaller:
    @pytest.mark.parametrize("caller", [USER, SERVICE])
    def test_every_data_tool_receives_the_caller_and_agent(self, loader, caller):
        tools = tool_registry.load_all_tools_for_agent(AGENT_ID, db_session=None, caller=caller)
        for name in ("database_query", "database_write", "storage_read"):
            tools[name].handler({"_caller": "forged by the model", "_agent_id": "other"}, None)
            assert loader.seen[name]["_caller"] is caller
        for name in ("database_query", "database_write"):
            assert loader.seen[name]["_agent_id"] == AGENT_ID

    def test_service_run_syncs_the_agent_roles_grants(self, loader):
        tool_registry.load_all_tools_for_agent(AGENT_ID, db_session=None, caller=SERVICE)
        loader.sync.assert_called_once_with(
            AGENT_ID,
            read_tables={"public": ["customers", "orders"]},
            write_tables={"public": ["orders"]},
        )

    def test_end_user_run_needs_no_agent_role(self, loader):
        tool_registry.load_all_tools_for_agent(AGENT_ID, db_session=None, caller=USER)
        loader.sync.assert_not_called()

    def test_no_caller_means_tools_that_refuse(self, loader):
        tools = tool_registry.load_all_tools_for_agent(AGENT_ID, db_session=None)
        tools["database_query"].handler({}, None)
        assert loader.seen["database_query"]["_caller"] is None
        loader.sync.assert_not_called()
