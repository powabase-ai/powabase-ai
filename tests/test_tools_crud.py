"""Tests for tool CRUD endpoints."""


class TestCreateTool:
    def test_create_custom_tool(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/tools",
            json={
                "name": "calculate_premium",
                "description": "Calculate insurance premium",
                "type": "custom",
                "input_schema": {
                    "type": "object",
                    "properties": {"age": {"type": "integer"}},
                    "required": ["age"],
                },
                "config": {
                    "endpoint": "https://example.com/api",
                    "method": "POST",
                    "headers": {},
                    "timeout_seconds": 10,
                },
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["name"] == "calculate_premium"
        assert data["type"] == "custom"
        assert data["id"] is not None

    def test_create_tool_missing_name(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/tools",
            json={
                "description": "no name",
                "type": "custom",
                "input_schema": {"type": "object"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400


class TestListTools:
    def test_list_includes_builtins(self, client, mock_auth, auth_headers):
        resp = client.get("/api/tools", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        names = [t["name"] for t in data["tools"]]
        assert "database_query" in names
        assert "http_request" in names

    def test_list_includes_custom(self, client, mock_auth, auth_headers):
        client.post(
            "/api/tools",
            json={
                "name": "my_tool",
                "description": "test",
                "type": "custom",
                "input_schema": {"type": "object"},
                "config": {"endpoint": "https://x.com"},
            },
            headers=auth_headers,
        )
        resp = client.get("/api/tools", headers=auth_headers)
        names = [t["name"] for t in resp.get_json()["tools"]]
        assert "my_tool" in names


class TestGetTool:
    def test_get_tool(self, client, mock_auth, auth_headers):
        create_resp = client.post(
            "/api/tools",
            json={
                "name": "fetch_tool",
                "description": "test get",
                "type": "custom",
                "input_schema": {"type": "object"},
            },
            headers=auth_headers,
        )
        tool_id = create_resp.get_json()["id"]
        resp = client.get(f"/api/tools/{tool_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "fetch_tool"

    def test_get_tool_not_found(self, client, mock_auth, auth_headers):
        resp = client.get(
            "/api/tools/00000000-0000-0000-0000-000000000000",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestUpdateTool:
    def test_update_tool(self, client, mock_auth, auth_headers):
        create_resp = client.post(
            "/api/tools",
            json={
                "name": "old_name",
                "description": "old desc",
                "type": "custom",
                "input_schema": {"type": "object"},
            },
            headers=auth_headers,
        )
        tool_id = create_resp.get_json()["id"]
        resp = client.put(
            f"/api/tools/{tool_id}",
            json={"name": "new_name"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "new_name"


class TestDeleteTool:
    def test_delete_tool(self, client, mock_auth, auth_headers):
        create_resp = client.post(
            "/api/tools",
            json={
                "name": "deleteme",
                "description": "test",
                "type": "custom",
                "input_schema": {"type": "object"},
                "config": {"endpoint": "https://x.com"},
            },
            headers=auth_headers,
        )
        tool_id = create_resp.get_json()["id"]
        resp = client.delete(f"/api/tools/{tool_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["deleted"] is True

        # Verify it's gone
        get_resp = client.get(f"/api/tools/{tool_id}", headers=auth_headers)
        assert get_resp.status_code == 404

    def test_delete_tool_not_found(self, client, mock_auth, auth_headers):
        resp = client.delete(
            "/api/tools/00000000-0000-0000-0000-000000000000",
            headers=auth_headers,
        )
        assert resp.status_code == 404
