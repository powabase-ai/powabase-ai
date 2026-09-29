"""Tool assignments and the per-agent database role.

A service-role run's database tools act as a role holding grants on exactly
the tables configured on the agent. Every change to an agent's tool
assignments re-syncs those grants, and every assignment is validated the same
way whether it is created or edited. DB-free: the data layer is stubbed.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from agentic_project_service.routes import agents as agents_route
from agentic_project_service.services import tool_registry

AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"
ASSIGNMENT_ID = "7e1f5a6b-9c2d-4e55-9f7a-c16d0f8e5b14"
HEADERS = {"Authorization": "Bearer fake"}


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(agents_route.agents_bp)
    with (
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"role": "service_role", "is_service_role": True},
        ),
        app.test_client() as c,
    ):
        yield c


@pytest.fixture
def db_session():
    session = MagicMock()
    with patch.object(agents_route.db, "session", session):
        yield session


@pytest.fixture
def sync():
    # Added to tool_registry alongside this change; created here so the test
    # does not depend on which lands first.
    with patch.object(tool_registry, "sync_agent_database_role", create=True) as fn:
        yield fn


BAD_SCHEMAS = [
    pytest.param({"schemas": ["public"]}, id="schemas-not-a-dict"),
    pytest.param({"schemas": {"pub lic": ["notes"]}}, id="bad-schema-name"),
    pytest.param({"schemas": {"pg_catalog": ["pg_authid"]}}, id="pg-schema"),
    pytest.param({"schemas": {"auth": ["users"]}}, id="system-schema"),
    pytest.param({"schemas": {"public": "notes"}}, id="tables-not-a-list"),
    pytest.param({"schemas": {"public": ['we"ird']}}, id="bad-table-name"),
    pytest.param({"schemas": {"public": [1]}}, id="table-not-a-string"),
]


class TestAssignToolValidatesLikePatch:
    @pytest.mark.parametrize("config", BAD_SCHEMAS)
    def test_bad_schemas_are_400_and_nothing_is_saved(self, client, db_session, sync, config):
        resp = client.post(
            f"/api/agents/{AGENT_ID}/tools",
            headers=HEADERS,
            json={
                "tool_type": "builtin",
                "tool_name": "database_query",
                "config_override": config,
            },
        )
        assert resp.status_code == 400, resp.get_data(as_text=True)
        db_session.add.assert_not_called()
        db_session.commit.assert_not_called()
        sync.assert_not_called()

    @pytest.mark.parametrize("config", BAD_SCHEMAS)
    def test_patch_gives_the_same_answer(self, client, db_session, sync, config):
        with patch.object(agents_route, "AgentTool") as model:
            model.query.filter_by.return_value.first.return_value = MagicMock()
            patched = client.patch(
                f"/api/agents/{AGENT_ID}/tools/{ASSIGNMENT_ID}",
                headers=HEADERS,
                json={"config_override": config},
            )
        assigned = client.post(
            f"/api/agents/{AGENT_ID}/tools",
            headers=HEADERS,
            json={"tool_type": "builtin", "tool_name": "database_query", "config_override": config},
        )
        assert patched.status_code == assigned.status_code == 400
        assert patched.get_json() == assigned.get_json()


class TestAssignmentChangesResyncTheRole:
    def test_assign_syncs_after_commit(self, client, db_session, sync):
        order = []
        db_session.commit.side_effect = lambda: order.append("commit")
        sync.side_effect = lambda agent_id: order.append(("sync", agent_id))
        resp = client.post(
            f"/api/agents/{AGENT_ID}/tools",
            headers=HEADERS,
            json={
                "tool_type": "builtin",
                "tool_name": "database_query",
                "config_override": {"schemas": {"public": ["notes"]}},
            },
        )
        assert resp.status_code == 201, resp.get_data(as_text=True)
        assert order == ["commit", ("sync", AGENT_ID)]

    def test_update_syncs(self, client, db_session, sync):
        with patch.object(agents_route, "AgentTool") as model:
            model.query.filter_by.return_value.first.return_value = MagicMock(
                id=ASSIGNMENT_ID, tool_type="builtin", tool_name="database_query"
            )
            resp = client.patch(
                f"/api/agents/{AGENT_ID}/tools/{ASSIGNMENT_ID}",
                headers=HEADERS,
                json={"config_override": {"schemas": {"public": ["notes"]}}},
            )
        assert resp.status_code == 200, resp.get_data(as_text=True)
        sync.assert_called_once_with(AGENT_ID)

    def test_remove_syncs(self, client, db_session, sync):
        db_session.get.return_value = MagicMock(agent_id=AGENT_ID)
        resp = client.delete(f"/api/agents/{AGENT_ID}/tools/{ASSIGNMENT_ID}", headers=HEADERS)
        assert resp.status_code == 200
        sync.assert_called_once_with(AGENT_ID)

    def test_remove_of_an_unknown_assignment_does_not_sync(self, client, db_session, sync):
        db_session.get.return_value = None
        resp = client.delete(f"/api/agents/{AGENT_ID}/tools/{ASSIGNMENT_ID}", headers=HEADERS)
        assert resp.status_code == 404
        sync.assert_not_called()

    @pytest.mark.parametrize("method", ["assign", "update", "remove"])
    def test_a_failed_sync_is_logged_and_the_change_still_succeeds(
        self, client, db_session, sync, caplog, method
    ):
        """The change is committed; the next service-role run re-syncs the grants."""
        sync.side_effect = RuntimeError("grant failed")
        with patch.object(agents_route, "AgentTool") as model:
            model.return_value = MagicMock(id=ASSIGNMENT_ID)
            model.query.filter_by.return_value.first.return_value = MagicMock(
                id=ASSIGNMENT_ID, tool_type="builtin", tool_name="database_query"
            )
            db_session.get.return_value = MagicMock(agent_id=AGENT_ID)
            with caplog.at_level(logging.ERROR, logger=agents_route.logger.name):
                if method == "assign":
                    resp = client.post(
                        f"/api/agents/{AGENT_ID}/tools",
                        headers=HEADERS,
                        json={"tool_type": "builtin", "tool_name": "database_query"},
                    )
                elif method == "update":
                    resp = client.patch(
                        f"/api/agents/{AGENT_ID}/tools/{ASSIGNMENT_ID}",
                        headers=HEADERS,
                        json={"config_override": {}},
                    )
                else:
                    resp = client.delete(
                        f"/api/agents/{AGENT_ID}/tools/{ASSIGNMENT_ID}", headers=HEADERS
                    )
        assert resp.status_code in (200, 201), resp.get_data(as_text=True)
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        assert errors[0].exc_info is not None
        assert AGENT_ID in errors[0].getMessage()


class TestDeleteAgentDropsItsRole:
    def test_row_is_deleted_then_the_role_dropped(self, client, db_session):
        order = []
        db_session.commit.side_effect = lambda: order.append("commit")
        with patch.object(agents_route.agent_sql, "drop_agent_role") as drop:
            drop.side_effect = lambda agent_id: order.append(("drop", agent_id))
            resp = client.delete(f"/api/agents/{AGENT_ID}", headers=HEADERS)
        assert resp.status_code == 200
        assert order == ["commit", ("drop", AGENT_ID)]

    def test_a_failed_drop_is_logged_and_the_delete_still_succeeds(
        self, client, db_session, caplog
    ):
        with (
            patch.object(agents_route.agent_sql, "drop_agent_role", side_effect=RuntimeError("x")),
            caplog.at_level(logging.ERROR, logger=agents_route.logger.name),
        ):
            resp = client.delete(f"/api/agents/{AGENT_ID}", headers=HEADERS)
        assert resp.status_code == 200
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1 and errors[0].exc_info is not None
