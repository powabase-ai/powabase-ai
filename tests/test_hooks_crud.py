"""Tests for hooks CRUD endpoints (agents and orchestrations)."""

import pytest


class TestHookCRUD:
    def test_add_hook(self, client, mock_auth, auth_headers, test_agent):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "rule",
                "matcher": "database_query",
                "config": {
                    "condition": "query CONTAINS 'DROP'",
                    "action": "deny",
                    "message": "No DROP",
                },
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["event"] == "PreToolUse"
        assert data["type"] == "rule"
        assert data["matcher"] == "database_query"
        assert data["config"]["action"] == "deny"
        assert data["enabled"] is True
        assert "id" in data

    def test_list_hooks(self, client, mock_auth, auth_headers, test_agent):
        client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "rule",
                "config": {"condition": "x CONTAINS 'y'"},
            },
            headers=auth_headers,
        )
        resp = client.get(
            f"/api/agents/{test_agent['id']}/hooks",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["hooks"]) == 1
        assert data["hooks"][0]["event"] == "PreToolUse"

    def test_delete_hook(self, client, mock_auth, auth_headers, test_agent):
        create = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "rule",
                "config": {},
            },
            headers=auth_headers,
        )
        hook_id = create.get_json()["id"]
        resp = client.delete(
            f"/api/agents/{test_agent['id']}/hooks/{hook_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["deleted"] is True

        # Confirm it's gone
        list_resp = client.get(
            f"/api/agents/{test_agent['id']}/hooks",
            headers=auth_headers,
        )
        assert len(list_resp.get_json()["hooks"]) == 0

    def test_missing_required_fields(self, client, mock_auth, auth_headers, test_agent):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "type": "rule",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_missing_type(self, client, mock_auth, auth_headers, test_agent):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "config": {},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_missing_config(self, client, mock_auth, auth_headers, test_agent):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "rule",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_list_hooks_empty(self, client, mock_auth, auth_headers, test_agent):
        resp = client.get(
            f"/api/agents/{test_agent['id']}/hooks",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["hooks"] == []

    def test_delete_nonexistent_hook(self, client, mock_auth, auth_headers, test_agent):
        import uuid

        fake_id = str(uuid.uuid4())
        resp = client.delete(
            f"/api/agents/{test_agent['id']}/hooks/{fake_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_hook_position_ordering(self, client, mock_auth, auth_headers, test_agent):
        for pos in [2, 0, 1]:
            client.post(
                f"/api/agents/{test_agent['id']}/hooks",
                json={"event": "PreToolUse", "type": "rule", "config": {}, "position": pos},
                headers=auth_headers,
            )
        resp = client.get(
            f"/api/agents/{test_agent['id']}/hooks",
            headers=auth_headers,
        )
        positions = [h["position"] for h in resp.get_json()["hooks"]]
        assert positions == sorted(positions)


class TestOrchestrationHookCRUD:
    @pytest.fixture
    def test_orchestration(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/orchestrations",
            json={"name": "Test Orchestration"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        return resp.get_json()

    def test_add_hook_to_orchestration(self, client, mock_auth, auth_headers, test_orchestration):
        resp = client.post(
            f"/api/orchestrations/{test_orchestration['id']}/hooks",
            json={
                "event": "PreAgentRun",
                "type": "logger",
                "config": {"level": "info"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["event"] == "PreAgentRun"
        assert data["type"] == "logger"
        assert "id" in data

    def test_list_orchestration_hooks(self, client, mock_auth, auth_headers, test_orchestration):
        client.post(
            f"/api/orchestrations/{test_orchestration['id']}/hooks",
            json={"event": "PreAgentRun", "type": "logger", "config": {}},
            headers=auth_headers,
        )
        resp = client.get(
            f"/api/orchestrations/{test_orchestration['id']}/hooks",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["hooks"]) == 1

    def test_orchestration_hook_missing_fields(
        self, client, mock_auth, auth_headers, test_orchestration
    ):
        resp = client.post(
            f"/api/orchestrations/{test_orchestration['id']}/hooks",
            json={"type": "logger"},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_list_hooks_orchestration_not_found(self, client, mock_auth, auth_headers):
        import uuid

        fake_id = str(uuid.uuid4())
        resp = client.get(
            f"/api/orchestrations/{fake_id}/hooks",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_delete_orchestration_hook(self, client, mock_auth, auth_headers, test_orchestration):
        create = client.post(
            f"/api/orchestrations/{test_orchestration['id']}/hooks",
            json={"event": "PreAgentRun", "type": "logger", "config": {}},
            headers=auth_headers,
        )
        hook_id = create.get_json()["id"]
        resp = client.delete(
            f"/api/orchestrations/{test_orchestration['id']}/hooks/{hook_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["deleted"] is True

        # Confirm it's gone from the list
        list_resp = client.get(
            f"/api/orchestrations/{test_orchestration['id']}/hooks",
            headers=auth_headers,
        )
        assert len(list_resp.get_json()["hooks"]) == 0

    def test_delete_nonexistent_orchestration_hook(
        self, client, mock_auth, auth_headers, test_orchestration
    ):
        import uuid

        fake_id = str(uuid.uuid4())
        resp = client.delete(
            f"/api/orchestrations/{test_orchestration['id']}/hooks/{fake_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_delete_orchestration_hook_bad_uuid(
        self, client, mock_auth, auth_headers, test_orchestration
    ):
        resp = client.delete(
            f"/api/orchestrations/{test_orchestration['id']}/hooks/not-a-uuid",
            headers=auth_headers,
        )
        assert resp.status_code == 404
