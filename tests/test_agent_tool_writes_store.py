"""database_write against a real Postgres: JSON values, and what reaches the logs.

An object or array the model sends for a json/jsonb column is written as
JSON; an object for any other column is refused with a clear message. A
failing write logs no value from the row.
"""

import json
import logging

import pytest
from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.tools import builtin
from tests import test_agent_sql_store as store

# The agent SQL store tests' database, fixtures and helpers.
project_db = store.project_db
agent_id = store.agent_id
SERVICE = store.SERVICE
_sync = store._sync

SECRET = "SECRET-123-45-6789"


@pytest.fixture
def docs_table(project_db):
    db.session.execute(
        text(
            """
            DROP TABLE IF EXISTS public.agent_sql_docs;
            CREATE TABLE public.agent_sql_docs (
              id serial PRIMARY KEY, data jsonb, raw json, tags text[], note text, n int);
            """
        )
    )
    db.session.commit()
    yield
    db.session.execute(text("DROP TABLE IF EXISTS public.agent_sql_docs"))
    db.session.commit()


def _write(agent, operation, data=None, where=None):
    arguments = {
        "table": "agent_sql_docs",
        "operation": operation,
        "data": data or {},
        "where": where or {},
        "_caller": SERVICE,
        "_agent_id": agent,
        "_schemas_config": {"public": ["agent_sql_docs"]},
        "_allowed_schemas": ["public"],
        "_allowed_tables": {"agent_sql_docs"},
    }
    return json.loads(builtin.database_write_handler(arguments, None))


def _rows():
    return [
        tuple(r)
        for r in db.session.execute(
            text("SELECT data, raw, tags, note FROM public.agent_sql_docs ORDER BY id")
        )
    ]


class TestJsonValues:
    def test_objects_and_arrays_go_to_json_columns_as_json(self, agent_id, docs_table):
        _sync(agent_id, write=["agent_sql_docs"])
        result = _write(
            agent_id,
            "insert",
            {"data": {"a": 1, "b": [2]}, "raw": [1, "x"], "tags": ["p", "q"], "note": "n"},
        )
        assert result["success"] is True, result
        assert _rows() == [({"a": 1, "b": [2]}, [1, "x"], ["p", "q"], "n")]

    def test_an_update_can_set_and_match_json(self, agent_id, docs_table):
        _sync(agent_id, write=["agent_sql_docs"])
        _write(agent_id, "insert", {"data": {"v": 1}, "note": "x"})
        result = _write(agent_id, "update", {"data": {"v": 2}}, where={"note": "x"})
        assert result["success"] is True and result["rows_affected"] == 1
        assert _rows()[0][0] == {"v": 2}

    def test_an_object_for_a_plain_column_is_refused_clearly(self, agent_id, docs_table):
        _sync(agent_id, write=["agent_sql_docs"])
        result = _write(agent_id, "insert", {"note": {"x": 1}})
        assert result["success"] is False
        assert "note" in result["message"] and "json" in result["message"]


class TestFailedWritesLogNoValues:
    def test_a_rejected_value_is_not_logged(self, agent_id, docs_table, caplog):
        _sync(agent_id, write=["agent_sql_docs"])
        with caplog.at_level(logging.DEBUG, logger=builtin.logger.name):
            result = _write(agent_id, "insert", {"n": SECRET, "data": {"ssn": SECRET}})
        assert result["success"] is False
        assert SECRET not in caplog.text
