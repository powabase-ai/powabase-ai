"""Tests for health check and index endpoints."""


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "project-service"


def test_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["service"] == "agentic-project-service"
    assert data["version"] == "0.1.0"
    assert "/api/sources" in data["endpoints"]
    assert "/api/agents" in data["endpoints"]
    assert "/api/knowledge-bases" in data["endpoints"]
