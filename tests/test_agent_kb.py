class TestAssignKB:
    def test_assign_kb_to_agent(
        self, client, mock_auth, auth_headers, test_agent, test_knowledge_base
    ):
        agent_id = test_agent["id"]
        kb_id = test_knowledge_base["id"]
        resp = client.post(
            f"/api/agents/{agent_id}/knowledge-bases",
            json={
                "knowledge_base_id": kb_id,
                "config": {"retrieval_method": "hybrid", "top_k": 10},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["knowledge_base_id"] == kb_id

    def test_assign_duplicate_kb(
        self, client, mock_auth, auth_headers, test_agent, test_knowledge_base
    ):
        agent_id = test_agent["id"]
        kb_id = test_knowledge_base["id"]
        client.post(
            f"/api/agents/{agent_id}/knowledge-bases",
            json={
                "knowledge_base_id": kb_id,
            },
            headers=auth_headers,
        )
        resp = client.post(
            f"/api/agents/{agent_id}/knowledge-bases",
            json={
                "knowledge_base_id": kb_id,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 409

    def test_list_agent_kbs(self, client, mock_auth, auth_headers, test_agent, test_knowledge_base):
        agent_id = test_agent["id"]
        kb_id = test_knowledge_base["id"]
        client.post(
            f"/api/agents/{agent_id}/knowledge-bases",
            json={
                "knowledge_base_id": kb_id,
            },
            headers=auth_headers,
        )
        resp = client.get(f"/api/agents/{agent_id}/knowledge-bases", headers=auth_headers)
        assert resp.status_code == 200
        kbs = resp.get_json()["knowledge_bases"]
        assert len(kbs) == 1
        assert kbs[0]["knowledge_base_id"] == kb_id

    def test_remove_kb_from_agent(
        self, client, mock_auth, auth_headers, test_agent, test_knowledge_base
    ):
        agent_id = test_agent["id"]
        kb_id = test_knowledge_base["id"]
        create_resp = client.post(
            f"/api/agents/{agent_id}/knowledge-bases",
            json={
                "knowledge_base_id": kb_id,
            },
            headers=auth_headers,
        )
        assignment_id = create_resp.get_json()["id"]
        resp = client.delete(
            f"/api/agents/{agent_id}/knowledge-bases/{assignment_id}", headers=auth_headers
        )
        assert resp.status_code == 200

    def test_assign_missing_kb_id(self, client, mock_auth, auth_headers, test_agent):
        agent_id = test_agent["id"]
        resp = client.post(f"/api/agents/{agent_id}/knowledge-bases", json={}, headers=auth_headers)
        assert resp.status_code == 400
