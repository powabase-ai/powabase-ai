class TestAssignTool:
    def test_assign_builtin_tool(self, client, mock_auth, auth_headers, test_agent):
        agent_id = test_agent["id"]
        resp = client.post(
            f"/api/agents/{agent_id}/tools",
            json={
                "tool_type": "builtin",
                "tool_name": "database_query",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201

    def test_assign_custom_tool(self, client, mock_auth, auth_headers, test_agent):
        agent_id = test_agent["id"]
        tool_resp = client.post(
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
        tool_id = tool_resp.get_json()["id"]
        resp = client.post(
            f"/api/agents/{agent_id}/tools",
            json={
                "tool_type": "custom",
                "tool_name": "my_tool",
                "tool_id": tool_id,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201

    def test_list_agent_tools(self, client, mock_auth, auth_headers, test_agent):
        agent_id = test_agent["id"]
        client.post(
            f"/api/agents/{agent_id}/tools",
            json={
                "tool_type": "builtin",
                "tool_name": "database_query",
            },
            headers=auth_headers,
        )
        resp = client.get(f"/api/agents/{agent_id}/tools", headers=auth_headers)
        assert resp.status_code == 200
        tools = resp.get_json()["tools"]
        assert len(tools) == 1
        assert tools[0]["tool_name"] == "database_query"

    def test_remove_tool(self, client, mock_auth, auth_headers, test_agent):
        agent_id = test_agent["id"]
        create_resp = client.post(
            f"/api/agents/{agent_id}/tools",
            json={
                "tool_type": "builtin",
                "tool_name": "database_query",
            },
            headers=auth_headers,
        )
        assignment_id = create_resp.get_json()["id"]
        resp = client.delete(f"/api/agents/{agent_id}/tools/{assignment_id}", headers=auth_headers)
        assert resp.status_code == 200

    def test_assign_missing_fields(self, client, mock_auth, auth_headers, test_agent):
        agent_id = test_agent["id"]
        resp = client.post(
            f"/api/agents/{agent_id}/tools",
            json={
                "tool_type": "builtin",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400
