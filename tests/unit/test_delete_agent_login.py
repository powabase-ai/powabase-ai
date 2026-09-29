"""DELETE /api/agents/<id> removes the agent and its database login together.

Both happen in one transaction, under the lock grant syncs take, so no sync
can recreate the login in between; if the login cannot be dropped, the agent
is not deleted either. That the lock really serialises them is pinned against
Postgres in tests/test_agent_sql_store.py.
"""

from unittest.mock import MagicMock, patch

from flask import Flask

from agentic_project_service.routes import agents as agents_route

AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"


def _app():
    app = Flask(__name__)
    app.register_blueprint(agents_route.agents_bp)
    return app


def _as_service():
    return patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"role": "service_role", "is_service_role": True},
    )


def _delete(agent_id=AGENT_ID):
    with _app().test_client() as client:
        return client.delete(f"/api/agents/{agent_id}", headers={"Authorization": "Bearer k"})


class TestDeleteAgent:
    def test_row_and_login_go_in_one_transaction(self):
        session = MagicMock()
        order: list[str] = []
        session.execute.side_effect = lambda *a, **k: order.append("delete row")
        session.commit.side_effect = lambda: order.append("commit")
        with (
            _as_service(),
            patch.object(agents_route.db, "session", session),
            patch.object(
                agents_route.agent_sql,
                "drop_agent_role_in",
                side_effect=lambda conn, agent: order.append(f"drop login {agent}"),
            ) as drop,
        ):
            resp = _delete()
        assert resp.status_code == 200
        assert order == ["delete row", f"drop login {AGENT_ID}", "commit"]
        assert drop.call_args.args[0] is session.connection.return_value

    def test_a_failed_drop_rolls_back_the_delete(self, caplog):
        session = MagicMock()
        with (
            _as_service(),
            patch.object(agents_route.db, "session", session),
            patch.object(
                agents_route.agent_sql,
                "drop_agent_role_in",
                side_effect=RuntimeError("role is still referenced"),
            ),
        ):
            resp = _delete()
        assert resp.status_code == 500
        session.commit.assert_not_called()
        session.rollback.assert_called_once()
        assert AGENT_ID in caplog.text

    def test_a_malformed_id_is_not_found(self):
        session = MagicMock()
        with _as_service(), patch.object(agents_route.db, "session", session):
            resp = _delete("not-a-uuid")
        assert resp.status_code == 404
        session.execute.assert_not_called()
