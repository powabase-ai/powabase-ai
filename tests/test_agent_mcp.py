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

    def test_discover_returns_tools(
        self, client, mock_auth, auth_headers, test_agent, mocker
    ):
        server_id = self._add_server(client, auth_headers, test_agent["id"])
        mocker.patch(
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

    def test_discover_unknown_server_returns_404(
        self, client, mock_auth, auth_headers, test_agent
    ):
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
