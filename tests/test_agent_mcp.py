"""Tests for agent MCP server CRUD endpoints."""

import uuid

from agentic.mcp import McpError
from agentic.mcp.types import McpToolInfo


class TestMcpServerCRUD:
    def test_add_mcp_server(self, client, mock_auth, auth_headers, test_agent):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/mcp-servers",
            json={
                "name": "github",
                "transport": "http",
                "url": "https://mcp-github.example.com",
                "headers": {"Authorization": "Bearer token"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["name"] == "github"
        assert data["transport"] == "http"
        assert data["url"] == "https://mcp-github.example.com"
        assert data["headers"] == {"Authorization": "Bearer token"}
        assert data["enabled"] is True
        assert "id" in data

    def test_add_mcp_server_missing_fields(self, client, mock_auth, auth_headers, test_agent):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/mcp-servers",
            json={"name": "github"},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_list_mcp_servers(self, client, mock_auth, auth_headers, test_agent):
        client.post(
            f"/api/agents/{test_agent['id']}/mcp-servers",
            json={
                "name": "github",
                "transport": "http",
                "url": "https://mcp.example.com",
            },
            headers=auth_headers,
        )
        resp = client.get(
            f"/api/agents/{test_agent['id']}/mcp-servers",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["mcp_servers"]) == 1
        assert data["mcp_servers"][0]["name"] == "github"

    def test_update_mcp_server(self, client, mock_auth, auth_headers, test_agent):
        create_resp = client.post(
            f"/api/agents/{test_agent['id']}/mcp-servers",
            json={
                "name": "github",
                "transport": "http",
                "url": "https://mcp.example.com",
            },
            headers=auth_headers,
        )
        server_id = create_resp.get_json()["id"]

        resp = client.put(
            f"/api/agents/{test_agent['id']}/mcp-servers/{server_id}",
            json={"url": "https://new-url.example.com", "enabled": False},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["url"] == "https://new-url.example.com"
        assert data["enabled"] is False

    def test_delete_mcp_server(self, client, mock_auth, auth_headers, test_agent):
        create_resp = client.post(
            f"/api/agents/{test_agent['id']}/mcp-servers",
            json={
                "name": "github",
                "transport": "http",
                "url": "https://mcp.example.com",
            },
            headers=auth_headers,
        )
        server_id = create_resp.get_json()["id"]

        resp = client.delete(
            f"/api/agents/{test_agent['id']}/mcp-servers/{server_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["deleted"] is True

        # Verify it's gone
        list_resp = client.get(
            f"/api/agents/{test_agent['id']}/mcp-servers",
            headers=auth_headers,
        )
        assert len(list_resp.get_json()["mcp_servers"]) == 0

    def test_duplicate_name_rejected(self, client, mock_auth, auth_headers, test_agent):
        client.post(
            f"/api/agents/{test_agent['id']}/mcp-servers",
            json={
                "name": "github",
                "transport": "http",
                "url": "https://mcp.example.com",
            },
            headers=auth_headers,
        )
        resp = client.post(
            f"/api/agents/{test_agent['id']}/mcp-servers",
            json={
                "name": "github",
                "transport": "http",
                "url": "https://other.example.com",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 409

    def test_delete_nonexistent_server(self, client, mock_auth, auth_headers, test_agent):
        resp = client.delete(
            f"/api/agents/{test_agent['id']}/mcp-servers/00000000-0000-0000-0000-000000000000",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestMcpToolDiscovery:
    def _add_server(self, client, auth_headers, agent_id):
        resp = client.post(
            f"/api/agents/{agent_id}/mcp-servers",
            json={
                "name": "github",
                "transport": "http",
                "url": "https://mcp-github.example.com",
            },
            headers=auth_headers,
        )
        return resp.get_json()["id"]

    def test_discover_returns_tools(self, client, mock_auth, auth_headers, test_agent, mocker):
        server_id = self._add_server(client, auth_headers, test_agent["id"])
        spy = mocker.patch(
            "agentic_project_service.routes.agents.discover_mcp_tools",
            return_value=[
                McpToolInfo(
                    name="create_issue",
                    description="Create an issue",
                    input_schema={"type": "object"},
                )
            ],
        )

        resp = client.get(
            f"/api/agents/{test_agent['id']}/mcp-servers/{server_id}/tools",
            headers=auth_headers,
        )

        assert resp.status_code == 200
        tools = resp.get_json()["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "create_issue"
        assert tools[0]["description"] == "Create an issue"
        assert tools[0]["input_schema"] == {"type": "object"}
        # The stored URL, the stored headers (auth for the MCP server lives
        # there), and the interactive timeout must all reach the client.
        spy.assert_called_once_with("https://mcp-github.example.com", {}, timeout=10)

    def test_discover_unknown_server_returns_404(self, client, mock_auth, auth_headers, test_agent):
        resp = client.get(
            f"/api/agents/{test_agent['id']}/mcp-servers/{uuid.uuid4()}/tools",
            headers=auth_headers,
        )
        assert resp.status_code == 404
        assert resp.get_json()["error"] == "MCP server not found"

    def test_discover_surfaces_server_error_as_502(
        self, client, mock_auth, auth_headers, test_agent, mocker
    ):
        server_id = self._add_server(client, auth_headers, test_agent["id"])
        mocker.patch(
            "agentic_project_service.routes.agents.discover_mcp_tools",
            side_effect=McpError(
                "MCP server returned HTTP 406: Not Acceptable: Client must "
                "accept both application/json and text/event-stream"
            ),
        )

        resp = client.get(
            f"/api/agents/{test_agent['id']}/mcp-servers/{server_id}/tools",
            headers=auth_headers,
        )

        assert resp.status_code == 502
        assert "406" in resp.get_json()["error"]
        assert "Not Acceptable" in resp.get_json()["error"]

    def test_discover_server_of_another_agent_is_404_and_never_contacted(
        self, client, mock_auth, auth_headers, test_agent, mocker
    ):
        server_id = self._add_server(client, auth_headers, test_agent["id"])
        other_agent = client.post(
            "/api/agents",
            json={"name": "Other Agent", "model": "gpt-4o-mini"},
            headers=auth_headers,
        ).get_json()
        spy = mocker.patch("agentic_project_service.routes.agents.discover_mcp_tools")

        resp = client.get(
            f"/api/agents/{other_agent['id']}/mcp-servers/{server_id}/tools",
            headers=auth_headers,
        )

        assert resp.status_code == 404
        assert resp.get_json()["error"] == "MCP server not found"
        # The tenancy check must gate the outbound request, not just the
        # response shape.
        spy.assert_not_called()

    def test_discover_bounds_the_reflected_error_message(
        self, client, mock_auth, auth_headers, test_agent, mocker
    ):
        # Only the client's HTTP-error path truncates upstream text. A server
        # answering 200 with a huge JSON-RPC error.message reaches this route
        # unbounded, so the route must cap what it reflects.
        server_id = self._add_server(client, auth_headers, test_agent["id"])
        mocker.patch(
            "agentic_project_service.routes.agents.discover_mcp_tools",
            side_effect=McpError("MCP tools/list failed: " + "x" * 10_000),
        )

        resp = client.get(
            f"/api/agents/{test_agent['id']}/mcp-servers/{server_id}/tools",
            headers=auth_headers,
        )

        assert resp.status_code == 502
        assert len(resp.get_json()["error"]) <= 500


class TestMcpDegradeLogging:
    def test_discovery_failure_log_names_the_agent_and_keeps_the_traceback(
        self, app, client, mock_auth, auth_headers, test_agent, mocker, caplog
    ):
        client.post(
            f"/api/agents/{test_agent['id']}/mcp-servers",
            json={
                "name": "github",
                "transport": "http",
                "url": "https://mcp-github.example.com",
            },
            headers=auth_headers,
        )
        # build_mcp_tools_for_agent imports discover_mcp_tools inside the
        # function body, so the source module is the only patchable target —
        # a routes-module patch would never be hit.
        mocker.patch(
            "agentic.mcp.client.discover_mcp_tools",
            side_effect=RuntimeError("boom"),
        )

        from agentic_project_service.db import db
        from agentic_project_service.services import tool_registry

        with app.app_context(), caplog.at_level("WARNING"):
            tools = tool_registry.build_mcp_tools_for_agent(test_agent["id"], db.session)

        assert tools == {}
        records = [r for r in caplog.records if "MCP server" in r.message]
        assert records, "expected a discovery-failure warning"
        record = records[0]
        # Server names are only unique per agent, so without the agent id the
        # log cannot answer "whose server failed".
        assert str(test_agent["id"]) in record.getMessage()
        # The broad except exists so agent runs degrade; when it eventually
        # catches a programming error, the traceback must be in the log.
        assert record.exc_info


class TestBuildMcpToolsForAgent:
    def _add_server(self, client, auth_headers, agent_id, name, url, enabled=True):
        resp = client.post(
            f"/api/agents/{agent_id}/mcp-servers",
            json={"name": name, "transport": "http", "url": url},
            headers=auth_headers,
        )
        server_id = resp.get_json()["id"]
        if not enabled:
            resp = client.put(
                f"/api/agents/{agent_id}/mcp-servers/{server_id}",
                json={"enabled": False},
                headers=auth_headers,
            )
            assert resp.status_code == 200
        return server_id

    def test_builds_tools_from_enabled_servers_only(
        self, app, client, mock_auth, auth_headers, test_agent, mocker
    ):
        self._add_server(client, auth_headers, test_agent["id"], "alpha", "https://a.example.com")
        self._add_server(
            client,
            auth_headers,
            test_agent["id"],
            "beta",
            "https://b.example.com",
            enabled=False,
        )
        spy = mocker.patch(
            "agentic.mcp.client.discover_mcp_tools",
            return_value=[
                McpToolInfo(
                    name="read_thing",
                    description="reads",
                    input_schema={"type": "object"},
                    read_only_hint=True,
                ),
                McpToolInfo(
                    name="drop_thing",
                    description="drops",
                    input_schema={"type": "object"},
                    destructive_hint=True,
                ),
            ],
        )

        from agentic_project_service.db import db
        from agentic_project_service.services import tool_registry

        with app.app_context():
            tools = tool_registry.build_mcp_tools_for_agent(test_agent["id"], db.session)

        # The disabled server is never contacted.
        assert [c.args[0] for c in spy.call_args_list] == ["https://a.example.com"]
        assert set(tools) == {"mcp__alpha__read_thing", "mcp__alpha__drop_thing"}
        reader = tools["mcp__alpha__read_thing"]
        assert reader.mcp_tool_name == "read_thing"
        assert reader.server_url == "https://a.example.com"
        assert reader.is_read_only is True
        assert reader.is_destructive is False
        dropper = tools["mcp__alpha__drop_thing"]
        assert dropper.is_destructive is True
        assert dropper.is_read_only is False

    def test_one_failing_server_does_not_stop_the_next(
        self, app, client, mock_auth, auth_headers, test_agent, mocker
    ):
        self._add_server(client, auth_headers, test_agent["id"], "bad", "https://bad.example.com")
        self._add_server(client, auth_headers, test_agent["id"], "good", "https://good.example.com")

        def discover(url, headers, *args, **kwargs):
            if "bad" in url:
                raise RuntimeError("unreachable")
            return [McpToolInfo(name="works", description="", input_schema={"type": "object"})]

        mocker.patch("agentic.mcp.client.discover_mcp_tools", side_effect=discover)

        from agentic_project_service.db import db
        from agentic_project_service.services import tool_registry

        with app.app_context():
            tools = tool_registry.build_mcp_tools_for_agent(test_agent["id"], db.session)

        # Degrade must continue to the next server, not break out of the loop.
        assert set(tools) == {"mcp__good__works"}
