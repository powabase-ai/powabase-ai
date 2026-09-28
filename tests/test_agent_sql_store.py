"""Agent database tools run as the caller, against a real Postgres.

What only a database can decide: that the tool logins are not superusers,
that an end user's run sees exactly the rows their RLS policies allow, that a
service-role run's agent role reaches exactly its configured tables, and that
the schemas and functions a superuser could reach are out of reach.

Runs in CI's store tier (plain Postgres + pgvector), so the Supabase pieces a
project database has — the ``authenticated`` role and ``auth.uid()`` — are
created here when missing.
"""

import json
import uuid

import pytest
from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.services import agent_sql
from agentic_project_service.services.agent_sql import AgentSqlRejected
from agentic_project_service.services.tool_caller import ToolCaller

USER_A = "aaaaaaaa-0000-4000-8000-000000000001"
USER_B = "bbbbbbbb-0000-4000-8000-000000000002"


def _user(sub):
    return ToolCaller(claims={"sub": sub, "role": "authenticated", "aud": "authenticated"})


SERVICE = ToolCaller.service()


@pytest.fixture(scope="module")
def project_db(app):
    """Supabase-shaped roles and tables, plus the agent logins."""
    with app.app_context():
        db.session.execute(
            text(
                """
                DO $$ BEGIN
                  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
                    CREATE ROLE authenticated NOLOGIN;
                  END IF;
                END $$;
                CREATE SCHEMA IF NOT EXISTS auth;
                DO $$ BEGIN
                  IF to_regprocedure('auth.uid()') IS NULL THEN
                    CREATE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS $f$
                      SELECT (nullif(current_setting('request.jwt.claims', true), '')::jsonb
                              ->> 'sub')::uuid
                    $f$;
                  END IF;
                END $$;
                GRANT USAGE ON SCHEMA auth TO authenticated;
                GRANT USAGE ON SCHEMA public TO authenticated;

                DROP TABLE IF EXISTS public.agent_sql_notes, public.agent_sql_orders,
                  public.agent_sql_secrets CASCADE;
                CREATE TABLE public.agent_sql_notes (
                  id serial PRIMARY KEY, owner uuid NOT NULL, body text NOT NULL);
                ALTER TABLE public.agent_sql_notes ENABLE ROW LEVEL SECURITY;
                CREATE POLICY own_notes ON public.agent_sql_notes TO authenticated
                  USING (owner = auth.uid()) WITH CHECK (owner = auth.uid());
                GRANT SELECT, INSERT, UPDATE, DELETE ON public.agent_sql_notes TO authenticated;
                GRANT USAGE ON SEQUENCE public.agent_sql_notes_id_seq TO authenticated;
                INSERT INTO public.agent_sql_notes (owner, body) VALUES
                  ('aaaaaaaa-0000-4000-8000-000000000001', 'a1'),
                  ('aaaaaaaa-0000-4000-8000-000000000001', 'a2'),
                  ('bbbbbbbb-0000-4000-8000-000000000002', 'b1');

                CREATE TABLE public.agent_sql_orders (id serial PRIMARY KEY, total int);
                INSERT INTO public.agent_sql_orders (total) VALUES (10), (20);
                GRANT SELECT ON public.agent_sql_orders TO authenticated;

                CREATE TABLE public.agent_sql_secrets (id int, secret text);
                INSERT INTO public.agent_sql_secrets VALUES (1, 'nope');
                GRANT SELECT ON public.agent_sql_secrets TO authenticated;
                """
            )
        )
        db.session.commit()
        agent_sql.ensure_login_roles()
        yield app
        db.session.execute(
            text(
                "DROP TABLE IF EXISTS public.agent_sql_notes, public.agent_sql_orders, "
                "public.agent_sql_secrets CASCADE"
            )
        )
        db.session.commit()


@pytest.fixture
def agent_id(project_db):
    agent = str(uuid.uuid4())
    yield agent
    with project_db.app_context():
        agent_sql.drop_agent_role(agent)


def _query(caller, agent, sql, tables):
    return agent_sql.run_query(caller, agent, sql, {"public": tables})


def _raw(caller, agent, sql, schemas=("public",)):
    """Run SQL under the tool identity, skipping the parse gate — the DB alone."""
    with agent_sql.agent_transaction(caller, agent, list(schemas), read_only=True) as conn:
        return [tuple(r) for r in conn.execute(text(sql))]


# ---------------------------------------------------------------------------
# The logins
# ---------------------------------------------------------------------------


class TestLogins:
    def test_neither_login_is_a_superuser_or_bypasses_rls(self, project_db):
        rows = db.session.execute(
            text(
                "SELECT rolname, rolsuper, rolbypassrls, rolinherit, rolcanlogin "
                "FROM pg_roles WHERE rolname = ANY(:names) ORDER BY rolname"
            ),
            {"names": [agent_sql.BACKEND_LOGIN, agent_sql.USER_LOGIN]},
        ).all()
        assert [tuple(r) for r in rows] == [
            (agent_sql.BACKEND_LOGIN, False, False, False, True),
            (agent_sql.USER_LOGIN, False, False, False, True),
        ]

    def test_the_end_user_login_can_only_become_authenticated(self, project_db):
        granted = (
            db.session.execute(
                text(
                    "SELECT r.rolname FROM pg_auth_members m "
                    "JOIN pg_roles r ON r.oid = m.roleid "
                    "JOIN pg_roles u ON u.oid = m.member WHERE u.rolname = :login"
                ),
                {"login": agent_sql.USER_LOGIN},
            )
            .scalars()
            .all()
        )
        assert granted == ["authenticated"]

    def test_ensuring_twice_is_harmless(self, project_db):
        agent_sql.ensure_login_roles()
        agent_sql.ensure_login_roles()


# ---------------------------------------------------------------------------
# End-user runs: exactly what the user could read themselves
# ---------------------------------------------------------------------------


class TestEndUserRuns:
    def test_rls_limits_rows_to_the_caller(self, agent_id):
        rows = _query(
            _user(USER_A),
            agent_id,
            "SELECT body FROM agent_sql_notes ORDER BY body",
            ["agent_sql_notes"],
        )
        assert rows == [{"body": "a1"}, {"body": "a2"}]
        rows = _query(
            _user(USER_B), agent_id, "SELECT body FROM agent_sql_notes", ["agent_sql_notes"]
        )
        assert rows == [{"body": "b1"}]

    def test_forging_the_caller_is_rejected(self, agent_id):
        sql = (
            "SELECT (SELECT set_config('request.jwt.claims', "
            f"'{json.dumps({'sub': USER_B})}', true)), (SELECT count(*) FROM agent_sql_notes)"
        )
        with pytest.raises(AgentSqlRejected):
            _query(_user(USER_A), agent_id, sql, ["agent_sql_notes"])

    def test_tables_outside_the_agents_list_are_rejected(self, agent_id):
        with pytest.raises(AgentSqlRejected, match="agent_sql_secrets"):
            _query(_user(USER_A), agent_id, "SELECT * FROM agent_sql_secrets", ["agent_sql_notes"])

    def test_system_views_are_not_on_any_list(self, agent_id):
        with pytest.raises(AgentSqlRejected):
            _query(
                _user(USER_A), agent_id, "SELECT query FROM pg_stat_activity", ["agent_sql_notes"]
            )

    def test_the_ai_schema_is_unreachable_even_without_the_gate(self, agent_id):
        with pytest.raises(Exception, match="permission denied"):
            _raw(_user(USER_A), agent_id, "SELECT count(*) FROM ai.sources")

    def test_superuser_functions_fail_even_without_the_gate(self, agent_id):
        with pytest.raises(Exception, match="permission denied"):
            _raw(_user(USER_A), agent_id, "SELECT pg_read_file('/etc/hostname')")

    def test_query_transaction_is_read_only(self, agent_id):
        with pytest.raises(Exception, match="read-only"):
            _query(
                _user(USER_A),
                agent_id,
                "SELECT nextval('agent_sql_notes_id_seq')",
                ["agent_sql_notes"],
            )

    def test_a_shadowing_function_in_an_allowed_schema_is_rejected(self, agent_id):
        db.session.execute(
            text(
                "CREATE OR REPLACE FUNCTION public.lower(text) RETURNS text LANGUAGE sql AS $$ SELECT $1 $$"
            )
        )
        db.session.commit()
        try:
            with pytest.raises(AgentSqlRejected, match="lower"):
                _query(
                    _user(USER_A),
                    agent_id,
                    "SELECT lower(body) FROM agent_sql_notes",
                    ["agent_sql_notes"],
                )
        finally:
            db.session.execute(text("DROP FUNCTION public.lower(text)"))
            db.session.commit()

    def test_writes_obey_rls_with_check(self, agent_id):
        with agent_sql.agent_transaction(
            _user(USER_A), agent_id, ["public"], read_only=False
        ) as conn:
            conn.execute(
                text("INSERT INTO agent_sql_notes (owner, body) VALUES (:o, 'mine')"),
                {"o": USER_A},
            )
        with pytest.raises(Exception, match="row-level security"):
            with agent_sql.agent_transaction(
                _user(USER_A), agent_id, ["public"], read_only=False
            ) as conn:
                conn.execute(
                    text("INSERT INTO agent_sql_notes (owner, body) VALUES (:o, 'theirs')"),
                    {"o": USER_B},
                )
        db.session.execute(text("DELETE FROM public.agent_sql_notes WHERE body = 'mine'"))
        db.session.commit()


# ---------------------------------------------------------------------------
# Service-role runs: exactly the agent's configured tables
# ---------------------------------------------------------------------------


class TestServiceRuns:
    def test_configured_table_is_readable_and_rls_is_bypassed(self, agent_id):
        agent_sql.sync_agent_role(
            agent_id, read_tables={"public": ["agent_sql_notes"]}, write_tables={}
        )
        rows = _query(
            SERVICE, agent_id, "SELECT count(*) AS n FROM agent_sql_notes", ["agent_sql_notes"]
        )
        assert rows == [{"n": 3}]

    def test_unconfigured_table_is_denied_by_postgres(self, agent_id):
        agent_sql.sync_agent_role(
            agent_id, read_tables={"public": ["agent_sql_notes"]}, write_tables={}
        )
        with pytest.raises(Exception, match="permission denied"):
            _raw(SERVICE, agent_id, "SELECT * FROM agent_sql_orders")

    def test_removing_a_table_revokes_it(self, agent_id):
        agent_sql.sync_agent_role(
            agent_id, read_tables={"public": ["agent_sql_orders"]}, write_tables={}
        )
        assert _raw(SERVICE, agent_id, "SELECT count(*) FROM agent_sql_orders") == [(2,)]
        agent_sql.sync_agent_role(agent_id, read_tables={}, write_tables={})
        with pytest.raises(Exception, match="permission denied"):
            _raw(SERVICE, agent_id, "SELECT count(*) FROM agent_sql_orders")

    def test_read_config_grants_no_writes(self, agent_id):
        agent_sql.sync_agent_role(
            agent_id, read_tables={"public": ["agent_sql_orders"]}, write_tables={}
        )
        with pytest.raises(Exception, match="permission denied"):
            with agent_sql.agent_transaction(
                SERVICE, agent_id, ["public"], read_only=False
            ) as conn:
                conn.execute(text("DELETE FROM agent_sql_orders"))

    def test_write_config_can_insert_through_a_serial_column(self, agent_id):
        agent_sql.sync_agent_role(
            agent_id, read_tables={}, write_tables={"public": ["agent_sql_orders"]}
        )
        with agent_sql.agent_transaction(SERVICE, agent_id, ["public"], read_only=False) as conn:
            conn.execute(text("INSERT INTO agent_sql_orders (total) VALUES (99)"))
        db.session.execute(text("DELETE FROM public.agent_sql_orders WHERE total = 99"))
        db.session.commit()

    def test_protected_schemas_are_never_granted(self, agent_id):
        agent_sql.sync_agent_role(agent_id, read_tables={"ai": ["sources"]}, write_tables={})
        with pytest.raises(Exception, match="permission denied"):
            _raw(SERVICE, agent_id, "SELECT count(*) FROM ai.sources", schemas=("ai",))

    def test_one_agent_cannot_use_anothers_grants(self, agent_id):
        other = str(uuid.uuid4())
        try:
            agent_sql.sync_agent_role(
                other, read_tables={"public": ["agent_sql_secrets"]}, write_tables={}
            )
            agent_sql.sync_agent_role(
                agent_id, read_tables={"public": ["agent_sql_orders"]}, write_tables={}
            )
            with pytest.raises(AgentSqlRejected):
                _query(
                    SERVICE,
                    agent_id,
                    f"SELECT set_config('role', '{agent_sql.agent_role_name(other)}', true)",
                    ["agent_sql_orders"],
                )
        finally:
            agent_sql.drop_agent_role(other)

    def test_dropping_the_agent_role_removes_it(self, project_db):
        agent = str(uuid.uuid4())
        agent_sql.sync_agent_role(
            agent, read_tables={"public": ["agent_sql_orders"]}, write_tables={}
        )
        agent_sql.drop_agent_role(agent)
        exists = db.session.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :r"),
            {"r": agent_sql.agent_role_name(agent)},
        ).first()
        assert exists is None
