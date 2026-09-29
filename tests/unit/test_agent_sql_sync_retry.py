"""Grant sync retries transient catalog conflicts, and nothing else."""

from unittest.mock import patch

import pytest
from sqlalchemy.exc import DBAPIError

from agentic_project_service.services import agent_sql

AGENT = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"


def _error(sqlstate):
    orig = Exception("catalog conflict")
    orig.sqlstate = sqlstate
    return DBAPIError("GRANT ...", {}, orig)


def _sync():
    agent_sql.sync_agent_role(AGENT, {"public": ["orders"]}, {})


@pytest.mark.parametrize("sqlstate", ["XX000", "40P01", "40001"])
def test_a_transient_catalog_conflict_is_retried(sqlstate):
    calls = []

    def once(*args):
        calls.append(args)
        if len(calls) < 3:
            raise _error(sqlstate)

    with (
        patch.object(agent_sql, "_sync_once", side_effect=once),
        patch.object(agent_sql.time, "sleep"),
    ):
        _sync()
    assert len(calls) == 3


def test_retries_are_bounded_and_the_last_error_is_raised():
    with (
        patch.object(agent_sql, "_sync_once", side_effect=_error("XX000")) as once,
        patch.object(agent_sql.time, "sleep"),
        pytest.raises(DBAPIError),
    ):
        _sync()
    assert once.call_count == 3


@pytest.mark.parametrize("sqlstate", ["42501", "42P01", None])
def test_any_other_error_is_raised_at_once(sqlstate):
    with (
        patch.object(agent_sql, "_sync_once", side_effect=_error(sqlstate)) as once,
        patch.object(agent_sql.time, "sleep"),
        pytest.raises(DBAPIError),
    ):
        _sync()
    assert once.call_count == 1
