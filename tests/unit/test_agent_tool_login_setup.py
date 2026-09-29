"""Startup sets up the agent tool logins, and survives when it cannot."""

from unittest.mock import patch

from agentic_project_service import main
from agentic_project_service.services import agent_sql


def test_both_steps_run():
    with (
        patch.object(agent_sql, "ensure_login_roles") as ensure,
        patch.object(agent_sql, "reconcile_agent_roles") as reconcile,
    ):
        main.set_up_agent_tool_logins()
    ensure.assert_called_once_with()
    reconcile.assert_called_once_with()


def test_failures_are_logged_not_raised(caplog):
    with (
        patch.object(agent_sql, "ensure_login_roles", side_effect=RuntimeError("no createrole")),
        patch.object(agent_sql, "reconcile_agent_roles", side_effect=RuntimeError("no table")),
    ):
        main.set_up_agent_tool_logins()
    assert "Could not set up the agent tool database logins" in caplog.text
    assert "Could not reconcile the agent tool database logins" in caplog.text


def test_a_failed_setup_still_reconciles():
    with (
        patch.object(agent_sql, "ensure_login_roles", side_effect=RuntimeError("x")),
        patch.object(agent_sql, "reconcile_agent_roles") as reconcile,
    ):
        main.set_up_agent_tool_logins()
    reconcile.assert_called_once_with()
