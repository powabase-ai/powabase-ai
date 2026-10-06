"""citation_mapping on agent MCP servers: set, read back, cleared, rejected."""

from sqlalchemy import text

from agentic_project_service.db import db

MAPPING = {
    "search_items": {"items": "$.results[*]", "title": "name", "url": "link"},
    "stats": {"items": "none"},
}


def _create(client, auth_headers, agent_id, **extra):
    return client.post(
        f"/api/agents/{agent_id}/mcp-servers",
        json={"name": "docs", "url": "https://mcp.example.com", **extra},
        headers=auth_headers,
    )


def _stored(server_id):
    return db.session.execute(
        text(
            "SELECT citation_mapping, citation_mapping IS NULL "
            'FROM "ai".agent_mcp_servers WHERE id = :id'
        ),
        {"id": server_id},
    ).one()


class TestCitationMapping:
    def test_create_with_a_mapping_echoes_and_lists_it(
        self, client, mock_auth, auth_headers, test_agent
    ):
        resp = _create(client, auth_headers, test_agent["id"], citation_mapping=MAPPING)
        assert resp.status_code == 201
        assert resp.get_json()["citation_mapping"] == MAPPING
        listed = client.get(
            f"/api/agents/{test_agent['id']}/mcp-servers", headers=auth_headers
        ).get_json()["mcp_servers"]
        assert listed[0]["citation_mapping"] == MAPPING

    def test_create_without_a_mapping_stores_sql_null(
        self, client, mock_auth, auth_headers, test_agent
    ):
        resp = _create(client, auth_headers, test_agent["id"])
        assert resp.status_code == 201
        assert resp.get_json()["citation_mapping"] is None
        assert _stored(resp.get_json()["id"]) == (None, True)

    def test_an_invalid_mapping_is_rejected_and_nothing_is_created(
        self, client, mock_auth, auth_headers, test_agent
    ):
        resp = _create(
            client,
            auth_headers,
            test_agent["id"],
            citation_mapping={"search_items": {"items": "$..name"}},
        )
        assert resp.status_code == 400
        assert "unsupported JSONPath" in resp.get_json()["error"]
        listed = client.get(
            f"/api/agents/{test_agent['id']}/mcp-servers", headers=auth_headers
        ).get_json()["mcp_servers"]
        assert listed == []

    def test_null_clears_to_sql_null_and_an_empty_object_is_kept(
        self, client, mock_auth, auth_headers, test_agent
    ):
        """{} keys every tool, null keys none; they must not blur."""
        server_id = _create(client, auth_headers, test_agent["id"], citation_mapping=MAPPING)
        server_id = server_id.get_json()["id"]
        url = f"/api/agents/{test_agent['id']}/mcp-servers/{server_id}"

        resp = client.put(url, json={"citation_mapping": {}}, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["citation_mapping"] == {}
        assert _stored(server_id) == ({}, False)

        resp = client.put(url, json={"citation_mapping": None}, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["citation_mapping"] is None
        assert _stored(server_id) == (None, True)

    def test_an_invalid_update_leaves_the_server_unchanged(
        self, client, mock_auth, auth_headers, test_agent
    ):
        server_id = _create(client, auth_headers, test_agent["id"], citation_mapping=MAPPING)
        server_id = server_id.get_json()["id"]
        resp = client.put(
            f"/api/agents/{test_agent['id']}/mcp-servers/{server_id}",
            json={"name": "renamed", "citation_mapping": {"t": {"items": 3}}},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        db.session.expire_all()
        assert _stored(server_id) == (MAPPING, False)
        name = db.session.execute(
            text('SELECT name FROM "ai".agent_mcp_servers WHERE id = :id'), {"id": server_id}
        ).scalar_one()
        assert name == "docs"
