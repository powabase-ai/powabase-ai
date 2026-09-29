"""Startup, drift and statement tracking for the agent tool logins, on a real Postgres.

- Booting the app runs the reconcile: a login whose agent is gone is dropped.
- A login whose attributes drifted is put back when it is next ensured.
- Where pg_stat_statements tracks utility statements, the transaction that
  sets a password turns that off first.
"""

import os
import subprocess
import sys
import uuid

from sqlalchemy import create_engine, text

from agentic_project_service.db import db
from agentic_project_service.services import agent_sql
from tests import test_agent_sql_store as store

# The agent SQL store tests' database, fixtures and helpers.
project_db = store.project_db
agent_id = store.agent_id
_attributes = store._attributes
_role_exists = store._role_exists
_statements_during = store._statements_during
_sync = store._sync


def test_booting_the_app_drops_the_login_of_an_agent_that_is_gone(project_db):
    """A real boot (migrations, then the login setup) on a fresh database."""
    stale = agent_sql.agent_role_name(str(uuid.uuid4()))
    fresh = f"boot_{uuid.uuid4().hex[:12]}"
    db.session.execute(text(f'CREATE ROLE "{stale}" NOLOGIN'))
    db.session.commit()
    with db.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f'CREATE DATABASE "{fresh}"'))
    try:
        url = db.engine.url.set(database=fresh).render_as_string(hide_password=False)
        side = create_engine(url)
        with side.begin() as conn:  # just enough of the ai schema for the reconcile to run
            conn.execute(
                text(
                    "CREATE SCHEMA ai; CREATE TABLE ai.agent_tools "
                    "(agent_id uuid, tool_type text, tool_name text)"
                )
            )
        side.dispose()
        boot = subprocess.run(
            [
                sys.executable,
                "-c",
                "from agentic_project_service.main import create_app; create_app()",
            ],
            env=dict(os.environ, DATABASE_URL=url),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert boot.returncode == 0, boot.stderr[-2000:]
        assert not _role_exists(stale)
    finally:
        db.session.rollback()
        db.session.execute(text(f'DROP ROLE IF EXISTS "{stale}"'))
        db.session.commit()
        with db.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{fresh}" WITH (FORCE)'))


def test_drifted_login_attributes_are_put_back(project_db):
    login = agent_sql.USER_LOGIN
    db.session.execute(text(f'ALTER ROLE "{login}" BYPASSRLS INHERIT'))
    db.session.commit()
    agent_sql.ensure_login_roles()
    assert _attributes(login) == (False, False, False, True)


def test_a_drifted_agent_login_is_put_back_on_the_next_sync(agent_id):
    _sync(agent_id, read=["agent_sql_orders"])
    role = agent_sql.agent_role_name(agent_id)
    db.session.execute(text(f'ALTER ROLE "{role}" INHERIT'))
    db.session.commit()
    _sync(agent_id, read=["agent_sql_orders"])
    assert _attributes(role) == (False, True, False, True)


def test_the_password_statement_is_untracked_where_pg_stat_statements_is(project_db):
    """Without the extension a placeholder setting stands in for it; the
    SET LOCAL must come before the ALTER ROLE."""
    name = db.engine.url.database
    db.session.execute(text(f'ALTER DATABASE "{name}" SET pg_stat_statements.track_utility = on'))
    db.session.commit()
    db.engine.dispose()
    try:
        sent = _statements_during(agent_sql.ensure_login_roles)
        off = [i for i, s in enumerate(sent) if "pg_stat_statements.track_utility = off" in s]
        alter = [
            i
            for i, s in enumerate(sent)
            if s.lstrip().upper().startswith("ALTER ROLE") and "PASSWORD" in s
        ]
        assert off and alter and off[0] < alter[0], sent
    finally:
        db.session.rollback()
        db.session.execute(text(f'ALTER DATABASE "{name}" RESET pg_stat_statements.track_utility'))
        db.session.commit()
        db.engine.dispose()
