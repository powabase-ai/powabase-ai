"""Tests for orchestration CRUD and entity management endpoints."""


class TestCreateOrchestration:
    def test_create(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/orchestrations",
            json={
                "name": "claims_processor",
                "description": "Processes insurance claims",
                "strategy": "supervisor",
                "settings": {"model": "gpt-4.1-mini", "max_steps": 10},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["name"] == "claims_processor"
        assert data["strategy"] == "supervisor"
        assert data["id"] is not None

    def test_create_missing_name(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/orchestrations",
            json={"description": "no name"},
            headers=auth_headers,
        )
        assert resp.status_code == 400


class TestListOrchestrations:
    def test_list(self, client, mock_auth, auth_headers):
        client.post("/api/orchestrations", json={"name": "orch1"}, headers=auth_headers)
        client.post("/api/orchestrations", json={"name": "orch2"}, headers=auth_headers)
        resp = client.get("/api/orchestrations", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.get_json()["orchestrations"]) == 2


class TestGetOrchestration:
    def test_get_with_entities(self, client, mock_auth, auth_headers, test_agent):
        orch_resp = client.post("/api/orchestrations", json={"name": "test"}, headers=auth_headers)
        orch_id = orch_resp.get_json()["id"]
        client.post(
            f"/api/orchestrations/{orch_id}/entities",
            json={
                "entity_type": "agent",
                "entity_ref_id": test_agent["id"],
                "role_description": "Test agent",
            },
            headers=auth_headers,
        )
        resp = client.get(f"/api/orchestrations/{orch_id}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["entities"]) == 1
        assert data["entities"][0]["entity_type"] == "agent"

    def test_get_not_found(self, client, mock_auth, auth_headers):
        resp = client.get(
            "/api/orchestrations/00000000-0000-0000-0000-000000000000",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestDeleteOrchestration:
    def test_delete(self, client, mock_auth, auth_headers):
        resp = client.post("/api/orchestrations", json={"name": "deleteme"}, headers=auth_headers)
        orch_id = resp.get_json()["id"]
        resp = client.delete(f"/api/orchestrations/{orch_id}", headers=auth_headers)
        assert resp.status_code == 200


class TestEntityManagement:
    def test_add_entity(self, client, mock_auth, auth_headers, test_agent):
        orch_resp = client.post("/api/orchestrations", json={"name": "test"}, headers=auth_headers)
        orch_id = orch_resp.get_json()["id"]
        resp = client.post(
            f"/api/orchestrations/{orch_id}/entities",
            json={
                "entity_type": "agent",
                "entity_ref_id": test_agent["id"],
                "role_description": "Claims specialist",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201

    def test_list_entities(self, client, mock_auth, auth_headers, test_agent):
        orch_resp = client.post("/api/orchestrations", json={"name": "test"}, headers=auth_headers)
        orch_id = orch_resp.get_json()["id"]
        client.post(
            f"/api/orchestrations/{orch_id}/entities",
            json={
                "entity_type": "agent",
                "entity_ref_id": test_agent["id"],
            },
            headers=auth_headers,
        )
        resp = client.get(f"/api/orchestrations/{orch_id}/entities", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.get_json()["entities"]) == 1

    def test_remove_entity(self, client, mock_auth, auth_headers, test_agent):
        orch_resp = client.post("/api/orchestrations", json={"name": "test"}, headers=auth_headers)
        orch_id = orch_resp.get_json()["id"]
        entity_resp = client.post(
            f"/api/orchestrations/{orch_id}/entities",
            json={
                "entity_type": "agent",
                "entity_ref_id": test_agent["id"],
            },
            headers=auth_headers,
        )
        entity_id = entity_resp.get_json()["id"]
        resp = client.delete(
            f"/api/orchestrations/{orch_id}/entities/{entity_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200

    def test_add_entity_missing_fields(self, client, mock_auth, auth_headers):
        orch_resp = client.post("/api/orchestrations", json={"name": "test"}, headers=auth_headers)
        orch_id = orch_resp.get_json()["id"]
        resp = client.post(
            f"/api/orchestrations/{orch_id}/entities",
            json={"entity_type": "agent"},
            headers=auth_headers,
        )
        assert resp.status_code == 400
