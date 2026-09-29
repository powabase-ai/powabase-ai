"""The tool logins' engines never put parameter values in error text."""

from unittest.mock import patch

from sqlalchemy.engine import make_url

from agentic_project_service.services import agent_sql


def test_both_kinds_of_tool_engine_hide_parameters():
    url = make_url("postgresql+psycopg://svc:pw@db.invalid:5432/postgres")
    with (
        patch.object(agent_sql, "db") as db,
        patch.dict(agent_sql._engines, clear=True),
    ):
        db.engine.url = url
        user = agent_sql._engine(agent_sql.USER_LOGIN)
        agent = agent_sql._engine(agent_sql.agent_role_name("3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"))
    assert user.hide_parameters is True
    assert agent.hide_parameters is True
