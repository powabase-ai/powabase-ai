"""Tests for agent MCP server CRUD endpoints."""


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
