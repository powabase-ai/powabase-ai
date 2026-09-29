"""What the database tools tell the model about errors a real database raised.

tests/unit/test_tool_error_messages.py pins the rules with hand-built psycopg
errors. These tests raise the errors for real, through SQLAlchemy and psycopg
against the test database, and check that only the server's primary message
reaches the model: never the SQL text, its parameters, the DETAIL line or
SQLAlchemy's wrapper.

The handlers' transactions are real connections acting as a role with row
level security; how agent_sql chooses that role is pinned in
tests/test_agent_sql_store.py.
"""

import json
import os
import time
from contextlib import contextmanager
from unittest.mock import patch

import psycopg.errors
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError

from agentic_project_service.services.tool_caller import ToolCaller
from agentic_project_service.tools import builtin
from agentic_project_service.tools.builtin import database_query_handler, database_write_handler

SCHEMA = "tool_error_messages"
ROLE = "tool_error_messages_caller"
AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"
OWNER = "aaaaaaaa-0000-4000-8000-000000000001"
SOMEONE_ELSE = "bbbbbbbb-0000-4000-8000-000000000002"
PARAM = "PARAM-VALUE-7731"
TABLES = {SCHEMA: ["notes", "tags"]}
CALLER = ToolCaller(
    claims={"sub": OWNER, "role": "authenticated", "exp": time.time() + 3600},
    token="user-token",
)


@pytest.fixture(scope="module")
def engine():
    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        conn.execute(text(f"DROP ROLE IF EXISTS {ROLE}"))
        conn.execute(text(f"CREATE ROLE {ROLE} NOLOGIN"))
        conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA {SCHEMA} TO {ROLE}"))
        conn.execute(
            text(
                f"CREATE TABLE {SCHEMA}.notes "
                "(id int PRIMARY KEY, owner text NOT NULL, body text, n int)"
            )
        )
        conn.execute(text(f"ALTER TABLE {SCHEMA}.notes ENABLE ROW LEVEL SECURITY"))
        conn.execute(
            text(
                f"CREATE POLICY own ON {SCHEMA}.notes "
                "USING (owner = current_setting('test.sub', true)) "
                "WITH CHECK (owner = current_setting('test.sub', true))"
            )
        )
        conn.execute(text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {SCHEMA}.notes TO {ROLE}"))
        # Without row level security, so a key violation carries its DETAIL.
        conn.execute(text(f"CREATE TABLE {SCHEMA}.tags (id int PRIMARY KEY, name text)"))
        conn.execute(text(f"GRANT SELECT, INSERT ON {SCHEMA}.tags TO {ROLE}"))
        conn.execute(text(f"INSERT INTO {SCHEMA}.tags VALUES (1, 'first')"))
        conn.execute(text(f"CREATE TABLE {SCHEMA}.secrets (id int)"))  # nothing granted
        conn.execute(
            text(f"INSERT INTO {SCHEMA}.notes VALUES (1, :owner, 'mine', 1)"), {"owner": OWNER}
        )
    yield engine
    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        conn.execute(text(f"DROP ROLE IF EXISTS {ROLE}"))
    engine.dispose()


def _open_as_caller(engine, statement_timeout="5s"):
    @contextmanager
    def transaction(*args, **kwargs):
        with engine.connect() as conn, conn.begin():
            conn.execute(text(f"SET LOCAL statement_timeout = '{statement_timeout}'"))
            conn.execute(text(f"SET LOCAL ROLE {ROLE}"))
            conn.execute(text(f"SET LOCAL search_path TO {SCHEMA}"))
            conn.execute(text("SELECT set_config('test.sub', :sub, true)"), {"sub": OWNER})
            yield conn

    return transaction


class _Raised:
    """Runs the tool with real transactions, keeping the error the database raised."""

    def __init__(self, engine, statement_timeout="5s"):
        self.transaction = _open_as_caller(engine, statement_timeout)
        self.error: DBAPIError | None = None

    def query(self, sql):
        def run_query(caller, agent_id, query, schemas_config):
            try:
                with self.transaction() as conn:
                    return [dict(row._mapping) for row in conn.execute(text(query))]
            except DBAPIError as e:
                self.error = e
                raise

        with patch.object(builtin.agent_sql, "run_query", run_query):
            raw = database_query_handler(
                {"query": sql, "_caller": CALLER, "_agent_id": AGENT_ID, "_schemas_config": TABLES},
                None,
            )
        return raw, json.loads(raw)["error"]

    def write(self, table="notes", **arguments):
        @contextmanager
        def transaction(*args, **kwargs):
            try:
                with self.transaction() as conn:
                    yield conn
            except DBAPIError as e:
                self.error = e
                raise

        with (
            patch.object(builtin.agent_sql, "agent_transaction", transaction),
            # A plain table; the relation check is pinned in the agent_sql store tests.
            patch.object(builtin.agent_sql, "check_write_target", create=True),
        ):
            raw = database_write_handler(
                {
                    "table": table,
                    **arguments,
                    "_caller": CALLER,
                    "_agent_id": AGENT_ID,
                    "_schemas_config": TABLES,
                },
                None,
            )
        result = json.loads(raw)
        assert result["success"] is False
        return raw, result["message"]


def _assert_only_the_primary_message(raw, message, error, sqlstate, *internals):
    """The database raised ``sqlstate``, and the model got its primary message only."""
    assert isinstance(error, DBAPIError)
    assert isinstance(error.orig, psycopg.Error)
    assert error.orig.sqlstate == sqlstate
    assert message == error.orig.diag.message_primary
    # The raised error does carry what must not reach the model.
    assert "[SQL:" in str(error)
    for internal in ("[SQL:", "sqlalche.me", *internals):
        assert internal not in raw


def test_a_missing_grant(engine):
    raised = _Raised(engine)
    raw, message = raised.query("SELECT id FROM secrets WHERE id = 7731")
    assert message == "permission denied for table secrets"
    _assert_only_the_primary_message(raw, message, raised.error, "42501", "7731")


def test_a_row_the_callers_policies_refuse(engine):
    raised = _Raised(engine)
    raw, message = raised.write(
        operation="insert", data={"id": 2, "owner": SOMEONE_ELSE, "body": PARAM}
    )
    assert message == 'new row violates row-level security policy for table "notes"'
    _assert_only_the_primary_message(raw, message, raised.error, "42501", PARAM, SOMEONE_ELSE)


def test_a_duplicate_key_leaves_out_the_detail(engine):
    """The DETAIL line quotes the row's key; the primary message does not."""
    raised = _Raised(engine)
    raw, message = raised.write(table="tags", operation="insert", data={"id": 1, "name": PARAM})
    assert message == 'duplicate key value violates unique constraint "tags_pkey"'
    assert raised.error.orig.diag.message_detail == "Key (id)=(1) already exists."
    _assert_only_the_primary_message(raw, message, raised.error, "23505", PARAM, "Key (id)")


def test_a_bad_value_in_an_update(engine):
    raised = _Raised(engine)
    raw, message = raised.write(operation="update", data={"n": "many"}, where={"id": 1})
    assert message == 'invalid input syntax for type integer: "many"'
    _assert_only_the_primary_message(raw, message, raised.error, "22P02", "UPDATE")


def test_a_statement_timeout(engine):
    raised = _Raised(engine, statement_timeout="50ms")
    raw, message = raised.query("SELECT pg_sleep(2), id FROM notes")
    assert message == "canceling statement due to statement timeout"
    _assert_only_the_primary_message(raw, message, raised.error, "57014", "pg_sleep")


def test_a_login_failure_is_generic(engine):
    """A connection error names the server and the login; the model gets neither."""
    url = make_url(os.environ["DATABASE_URL"])
    wrong_password = create_engine(url.set(password="not-the-password"))
    raised = _Raised(wrong_password)
    try:
        raw, message = raised.query("SELECT id FROM notes")
    finally:
        wrong_password.dispose()
    assert message == builtin._DB_TOOL_FAILED
    assert isinstance(raised.error.orig, psycopg.OperationalError)
    assert url.username in str(raised.error)
    assert url.username not in raw
    assert str(url.host) not in raw
