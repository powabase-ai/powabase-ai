"""Agent database tools run as the caller, against a real Postgres.

What only a database can decide: that the tool logins are not superusers,
that an end user's run sees exactly the rows their RLS policies allow, that a
service-role run's agent login reaches exactly its configured tables, and that
the schemas, functions, operators and casts a superuser could use are out of
reach — including the built-ins that run a SQL string they are handed.

Runs in CI's store tier (plain Postgres + pgvector), so the Supabase pieces a
project database has — the ``authenticated`` role and ``auth.uid()`` — are
created here when missing. The ``auth.uid()`` created is the older form that
reads only ``request.jwt.claim.sub``, so both ways of passing claims are
exercised: that one by RLS, the JSON one by a direct read.
"""

import json
import time
import uuid

import pytest
from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.services import agent_sql
from agentic_project_service.services.agent_sql import AgentSqlRejected, AgentToolsUnavailable
from agentic_project_service.services.session import session_accessible_to
from agentic_project_service.services.tool_caller import ToolCaller

USER_A = "aaaaaaaa-0000-4000-8000-000000000001"
USER_B = "bbbbbbbb-0000-4000-8000-000000000002"


def _user(sub, **claims):
    return ToolCaller(
        claims={"sub": sub, "role": "authenticated", "aud": "authenticated", **claims},
        token="user-token",
    )


SERVICE = ToolCaller.service()


@pytest.fixture(scope="module")
def project_db(app):
    """Supabase-shaped roles and tables, plus the end-user login."""
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
                    -- The older form, which reads only the per-claim setting:
                    -- what supabase/postgres ships before GoTrue replaces it.
                    CREATE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS $f$
                      SELECT nullif(current_setting('request.jwt.claim.sub', true), '')::uuid
                    $f$;
                  END IF;
                END $$;
                GRANT USAGE ON SCHEMA auth TO authenticated;
                GRANT USAGE ON SCHEMA public TO authenticated;

                DROP VIEW IF EXISTS public.agent_sql_all_notes, public.agent_sql_own_notes;
                DROP TABLE IF EXISTS public.agent_sql_notes, public.agent_sql_orders,
                  public.agent_sql_secrets, public."we""ird" CASCADE;
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

                -- Owner-rights view: reads past RLS. Invoker view: does not.
                CREATE VIEW public.agent_sql_all_notes AS SELECT * FROM public.agent_sql_notes;
                CREATE VIEW public.agent_sql_own_notes WITH (security_invoker = true)
                  AS SELECT * FROM public.agent_sql_notes;
                GRANT SELECT ON public.agent_sql_all_notes, public.agent_sql_own_notes
                  TO authenticated;

                CREATE TABLE public.agent_sql_orders (id serial PRIMARY KEY, total int);
                INSERT INTO public.agent_sql_orders (total) VALUES (10), (20);
                GRANT SELECT ON public.agent_sql_orders TO authenticated;

                CREATE TABLE public.agent_sql_secrets (id int, secret text);
                INSERT INTO public.agent_sql_secrets VALUES (1, 'nope');
                GRANT SELECT ON public.agent_sql_secrets TO authenticated;

                CREATE TABLE public."we""ird" (id int);
                """
            )
        )
        db.session.commit()
        agent_sql.ensure_login_roles()
        yield app
        db.session.execute(
            text(
                "DROP VIEW IF EXISTS public.agent_sql_all_notes, public.agent_sql_own_notes; "
                "DROP TABLE IF EXISTS public.agent_sql_notes, public.agent_sql_orders, "
                'public.agent_sql_secrets, public."we""ird" CASCADE'
            )
        )
        db.session.commit()


def _new_agent() -> str:
    agent = str(uuid.uuid4())
    db.session.execute(
        text(
            "INSERT INTO ai.agents (id, name, model) VALUES (:id, 'agent-sql-test', 'gpt-4o-mini')"
        ),
        {"id": agent},
    )
    db.session.commit()
    return agent


def _forget_agent(agent: str) -> None:
    db.session.execute(text("DELETE FROM ai.agents WHERE id = :id"), {"id": agent})
    db.session.commit()
    agent_sql.drop_agent_role(agent)


@pytest.fixture
def agent_id(project_db):
    agent = _new_agent()
    yield agent
    with project_db.app_context():
        _forget_agent(agent)


def _query(caller, agent, sql, tables):
    return agent_sql.run_query(caller, agent, sql, {"public": tables})


def _raw(caller, agent, sql, schemas=("public",)):
    """Run SQL under the tool identity, skipping the parse gate — the DB alone."""
    with agent_sql.agent_transaction(caller, agent, list(schemas), read_only=True) as conn:
        return [tuple(r) for r in conn.execute(text(sql))]


def _sync(agent, read=None, write=None):
    agent_sql.sync_agent_role(
        agent, read_tables={"public": read} if read else {}, write_tables={"public": write} if write else {}
    )


def _role_exists(role):
    return (
        db.session.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role}).first()
        is not None
    )


def _attributes(role):
    return tuple(
        db.session.execute(
            text(
                "SELECT rolsuper, rolbypassrls, rolinherit, rolcanlogin "
                "FROM pg_roles WHERE rolname = :r"
            ),
            {"r": role},
        ).one()
    )


def _memberships(role):
    return (
        db.session.execute(
            text(
                "SELECT r.rolname FROM pg_auth_members m "
                "JOIN pg_roles r ON r.oid = m.roleid "
                "JOIN pg_roles u ON u.oid = m.member WHERE u.rolname = :login"
            ),
            {"login": role},
        )
        .scalars()
        .all()
    )


# ---------------------------------------------------------------------------
# The logins
# ---------------------------------------------------------------------------


class TestLogins:
    def test_the_end_user_login_is_no_superuser_and_cannot_bypass_rls(self, project_db):
        assert _attributes(agent_sql.USER_LOGIN) == (False, False, False, True)

    def test_the_end_user_login_can_only_become_authenticated(self, project_db):
        assert _memberships(agent_sql.USER_LOGIN) == ["authenticated"]

    def test_an_agent_login_bypasses_rls_but_is_no_superuser_and_no_member(self, agent_id):
        _sync(agent_id, read=["agent_sql_orders"])
        role = agent_sql.agent_role_name(agent_id)
        assert _attributes(role) == (False, True, False, True)
        assert _memberships(role) == []

    def test_ensuring_twice_is_harmless(self, project_db):
        agent_sql.ensure_login_roles()
        agent_sql.ensure_login_roles()

    def test_every_allowlisted_function_is_a_builtin_here(self, project_db):
        """Guards the allowlist against a name this Postgres does not have."""
        present = set(
            db.session.execute(
                text(
                    "SELECT DISTINCT proname FROM pg_proc "
                    "WHERE pronamespace = 'pg_catalog'::regnamespace AND proname = ANY(:n)"
                ),
                {"n": sorted(agent_sql.ALLOWED_FUNCTIONS)},
            ).scalars()
        )
        assert sorted(agent_sql.ALLOWED_FUNCTIONS - present) == []


# ---------------------------------------------------------------------------
# End-user runs: the user's own grants and RLS, on the configured tables
# ---------------------------------------------------------------------------


class TestEndUserRuns:
    def test_rls_limits_rows_to_the_caller(self, agent_id):
        sql = "SELECT body FROM agent_sql_notes ORDER BY body"
        assert _query(_user(USER_A), agent_id, sql, ["agent_sql_notes"]) == [
            {"body": "a1"},
            {"body": "a2"},
        ]
        assert _query(_user(USER_B), agent_id, sql, ["agent_sql_notes"]) == [{"body": "b1"}]

    def test_claims_are_also_exposed_as_json(self, agent_id):
        rows = _raw(_user(USER_A), agent_id, "SELECT current_setting('request.jwt.claims')")
        assert json.loads(rows[0][0])["sub"] == USER_A

    def test_forging_the_caller_is_rejected(self, agent_id):
        sql = (
            "SELECT (SELECT set_config('request.jwt.claims', "
            f"'{json.dumps({'sub': USER_B})}', true)), (SELECT count(*) FROM agent_sql_notes)"
        )
        with pytest.raises(AgentSqlRejected):
            _query(_user(USER_A), agent_id, sql, ["agent_sql_notes"])

    def test_forging_the_caller_inside_ts_stat_is_rejected(self, agent_id):
        """ts_stat runs the SQL string it is handed; the parse gate never sees it."""
        sql = (
            "SELECT (SELECT count(*) FROM ts_stat($q$select to_tsvector("
            f"set_config('request.jwt.claim.sub','{USER_B}',true))$q$)), "
            "(SELECT string_agg(body, ',') FROM agent_sql_notes)"
        )
        with pytest.raises(AgentSqlRejected, match="ts_stat"):
            _query(_user(USER_A), agent_id, sql, ["agent_sql_notes"])

    def test_tables_outside_the_agents_list_are_rejected(self, agent_id):
        with pytest.raises(AgentSqlRejected, match="agent_sql_secrets"):
            _query(_user(USER_A), agent_id, "SELECT * FROM agent_sql_secrets", ["agent_sql_notes"])

    def test_system_views_are_not_on_any_list(self, agent_id):
        with pytest.raises(AgentSqlRejected):
            _query(
                _user(USER_A), agent_id, "SELECT query FROM pg_stat_activity", ["agent_sql_notes"]
            )

    def test_an_owner_rights_view_is_rejected_even_when_configured(self, agent_id):
        with pytest.raises(AgentSqlRejected, match="agent_sql_all_notes"):
            _query(
                _user(USER_A),
                agent_id,
                "SELECT body FROM agent_sql_all_notes",
                ["agent_sql_all_notes"],
            )

    def test_a_security_invoker_view_is_allowed_and_rls_still_applies(self, agent_id):
        rows = _query(
            _user(USER_A),
            agent_id,
            "SELECT body FROM agent_sql_own_notes ORDER BY body",
            ["agent_sql_own_notes"],
        )
        assert rows == [{"body": "a1"}, {"body": "a2"}]

    def test_the_ai_schema_is_unreachable_even_without_the_gate(self, agent_id):
        with pytest.raises(Exception, match="permission denied"):
            _raw(_user(USER_A), agent_id, "SELECT count(*) FROM ai.sources")

    def test_superuser_functions_fail_even_without_the_gate(self, agent_id):
        with pytest.raises(Exception, match="permission denied"):
            _raw(_user(USER_A), agent_id, "SELECT pg_read_file('/etc/hostname')")

    def test_query_transaction_is_read_only(self, agent_id):
        with pytest.raises(Exception, match="read-only"):
            _raw(_user(USER_A), agent_id, "SELECT nextval('agent_sql_notes_id_seq')")

    def test_statement_and_lock_timeouts_are_set(self, agent_id):
        rows = _raw(
            _user(USER_A),
            agent_id,
            "SELECT current_setting('statement_timeout'), current_setting('lock_timeout')",
        )
        assert rows == [(agent_sql.STATEMENT_TIMEOUT, agent_sql.LOCK_TIMEOUT)]

    def test_an_expired_session_is_refused(self, agent_id):
        with pytest.raises(AgentToolsUnavailable, match="expired"):
            _query(
                _user(USER_A, exp=int(time.time()) - 60),
                agent_id,
                "SELECT body FROM agent_sql_notes",
                ["agent_sql_notes"],
            )

    def test_a_shadowing_function_in_an_allowed_schema_is_rejected(self, agent_id):
        db.session.execute(
            text(
                "CREATE OR REPLACE FUNCTION public.lower(text) RETURNS text "
                "LANGUAGE sql AS $$ SELECT $1 $$"
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

    def test_a_sql_language_operator_in_an_allowed_schema_is_rejected(self, agent_id):
        db.session.execute(
            text(
                "CREATE FUNCTION public.agent_sql_eq(text, text) RETURNS boolean LANGUAGE sql "
                "AS $$ SELECT $1 = $2 $$; "
                "CREATE OPERATOR public.= (LEFTARG = text, RIGHTARG = text, "
                "FUNCTION = public.agent_sql_eq)"
            )
        )
        db.session.commit()
        try:
            with pytest.raises(AgentSqlRejected, match="Operator ="):
                _query(
                    _user(USER_A),
                    agent_id,
                    "SELECT body FROM agent_sql_notes WHERE body = 'a1'",
                    ["agent_sql_notes"],
                )
        finally:
            db.session.execute(
                text(
                    "DROP OPERATOR public.= (text, text); "
                    "DROP FUNCTION public.agent_sql_eq(text, text)"
                )
            )
            db.session.commit()

    def test_a_cast_to_a_domain_is_rejected(self, agent_id):
        db.session.execute(text("CREATE DOMAIN public.agent_sql_trap AS text CHECK (VALUE <> '')"))
        db.session.commit()
        try:
            with pytest.raises(AgentSqlRejected, match="agent_sql_trap"):
                _query(
                    _user(USER_A),
                    agent_id,
                    "SELECT body::agent_sql_trap FROM agent_sql_notes",
                    ["agent_sql_notes"],
                )
        finally:
            db.session.execute(text("DROP DOMAIN public.agent_sql_trap"))
            db.session.commit()

    def test_writes_obey_rls_with_check(self, agent_id):
        insert = text("INSERT INTO agent_sql_notes (owner, body) VALUES (:o, :b)")
        with agent_sql.agent_transaction(
            _user(USER_A), agent_id, ["public"], read_only=False
        ) as conn:
            conn.execute(insert, {"o": USER_A, "b": "mine"})
        with pytest.raises(Exception, match="row-level security"):
            with agent_sql.agent_transaction(
                _user(USER_A), agent_id, ["public"], read_only=False
            ) as conn:
                conn.execute(insert, {"o": USER_B, "b": "theirs"})
        db.session.execute(text("DELETE FROM public.agent_sql_notes WHERE body = 'mine'"))
        db.session.commit()


# ---------------------------------------------------------------------------
# Service-role runs: exactly the agent's configured tables
# ---------------------------------------------------------------------------


class TestServiceRuns:
    def test_configured_table_is_readable_and_rls_is_bypassed(self, agent_id):
        _sync(agent_id, read=["agent_sql_notes"])
        rows = _query(
            SERVICE, agent_id, "SELECT count(*) AS n FROM agent_sql_notes", ["agent_sql_notes"]
        )
        assert rows == [{"n": 3}]

    def test_unconfigured_table_is_denied_by_postgres(self, agent_id):
        _sync(agent_id, read=["agent_sql_notes"])
        with pytest.raises(Exception, match="permission denied"):
            _raw(SERVICE, agent_id, "SELECT * FROM agent_sql_orders")

    def test_removing_a_table_revokes_it(self, agent_id):
        _sync(agent_id, read=["agent_sql_orders"])
        assert _raw(SERVICE, agent_id, "SELECT count(*) FROM agent_sql_orders") == [(2,)]
        _sync(agent_id)
        with pytest.raises(Exception, match="permission denied"):
            _raw(SERVICE, agent_id, "SELECT count(*) FROM agent_sql_orders")

    def test_removing_a_write_table_revokes_its_sequence_and_schema_too(self, agent_id):
        _sync(agent_id, write=["agent_sql_orders"])
        role = agent_sql.agent_role_name(agent_id)
        granted = text(
            "SELECT has_sequence_privilege(:r, 'public.agent_sql_orders_id_seq', 'USAGE'), "
            "EXISTS (SELECT 1 FROM pg_namespace n CROSS JOIN LATERAL aclexplode(n.nspacl) a "
            "WHERE n.nspname = 'public' AND a.grantee = CAST(:r AS regrole))"
        )
        assert tuple(db.session.execute(granted, {"r": role}).one()) == (True, True)
        _sync(agent_id)
        assert tuple(db.session.execute(granted, {"r": role}).one()) == (False, False)

    def test_read_config_grants_no_writes(self, agent_id):
        _sync(agent_id, read=["agent_sql_orders"])
        with pytest.raises(Exception, match="permission denied"):
            with agent_sql.agent_transaction(SERVICE, agent_id, ["public"], read_only=False) as conn:
                conn.execute(text("DELETE FROM agent_sql_orders"))

    def test_write_config_can_insert_through_a_serial_column(self, agent_id):
        _sync(agent_id, write=["agent_sql_orders"])
        with agent_sql.agent_transaction(SERVICE, agent_id, ["public"], read_only=False) as conn:
            conn.execute(text("INSERT INTO agent_sql_orders (total) VALUES (99)"))
        db.session.execute(text("DELETE FROM public.agent_sql_orders WHERE total = 99"))
        db.session.commit()

    def test_protected_schemas_are_never_granted(self, agent_id):
        agent_sql.sync_agent_role(agent_id, read_tables={"ai": ["sources"]}, write_tables={})
        with pytest.raises(Exception, match="permission denied"):
            _raw(SERVICE, agent_id, "SELECT count(*) FROM ai.sources", schemas=("ai",))

    def test_owner_rights_views_are_never_granted(self, agent_id):
        _sync(agent_id, read=["agent_sql_all_notes"])
        with pytest.raises(Exception, match="permission denied"):
            _raw(SERVICE, agent_id, "SELECT count(*) FROM agent_sql_all_notes")

    def test_odd_names_neither_break_the_sync_nor_keep_old_grants(self, agent_id):
        _sync(agent_id, read=["agent_sql_orders"])
        _sync(agent_id, read=['we"ird', "agent_sql_orders_pkey", "no_such_table"])
        assert _raw(SERVICE, agent_id, 'SELECT count(*) FROM "we""ird"') == [(0,)]
        with pytest.raises(Exception, match="permission denied"):
            _raw(SERVICE, agent_id, "SELECT count(*) FROM agent_sql_orders")

    def test_one_agent_cannot_take_on_anothers_grants_even_without_the_gate(self, agent_id):
        other = _new_agent()
        try:
            _sync(other, read=["agent_sql_secrets"])
            _sync(agent_id, read=["agent_sql_orders"])
            with pytest.raises(Exception, match="permission denied"):
                _raw(SERVICE, agent_id, f'SET ROLE "{agent_sql.agent_role_name(other)}"')
        finally:
            _forget_agent(other)


# ---------------------------------------------------------------------------
# Agent logins over the agent's lifetime
# ---------------------------------------------------------------------------


class TestAgentLoginLifecycle:
    def test_dropping_the_agent_role_removes_it(self, project_db):
        agent = _new_agent()
        _sync(agent, read=["agent_sql_orders"])
        _forget_agent(agent)
        assert not _role_exists(agent_sql.agent_role_name(agent))

    def test_a_sync_for_a_deleted_agent_does_not_recreate_its_login(self, project_db):
        agent = str(uuid.uuid4())  # no ai.agents row: deleted between tool load and sync
        _sync(agent, read=["agent_sql_orders"])
        assert not _role_exists(agent_sql.agent_role_name(agent))

    def test_reconcile_drops_logins_whose_agent_is_gone(self, project_db):
        kept = _new_agent()
        gone = _new_agent()
        try:
            for agent in (kept, gone):
                _sync(agent, read=["agent_sql_orders"])
            db.session.execute(text("DELETE FROM ai.agents WHERE id = :id"), {"id": gone})
            db.session.commit()
            agent_sql.reconcile_agent_roles()
            assert _role_exists(agent_sql.agent_role_name(kept))
            assert not _role_exists(agent_sql.agent_role_name(gone))
        finally:
            _forget_agent(kept)
            agent_sql.drop_agent_role(gone)


# ---------------------------------------------------------------------------
# Which sessions an end user may continue
# ---------------------------------------------------------------------------


class TestSessionAccessibleTo:
    def _session(self, agent, session_id, owner):
        db.session.execute(
            text(
                "INSERT INTO ai.agent_sessions (id, session_id, agent_id, user_id) "
                "VALUES (:id, :sid, :aid, :uid)"
            ),
            {"id": str(uuid.uuid4()), "sid": session_id, "aid": agent, "uid": owner},
        )
        db.session.commit()

    def test_own_other_ownerless_and_unknown_sessions(self, agent_id):
        suffix = uuid.uuid4().hex[:8]
        self._session(agent_id, f"sess_a_{suffix}", USER_A)
        self._session(agent_id, f"sess_b_{suffix}", USER_B)
        self._session(agent_id, f"sess_none_{suffix}", None)
        try:
            assert session_accessible_to(db.session, f"sess_a_{suffix}", USER_A) is True
            assert session_accessible_to(db.session, f"sess_b_{suffix}", USER_A) is False
            assert session_accessible_to(db.session, f"sess_none_{suffix}", USER_A) is False
            assert session_accessible_to(db.session, f"sess_new_{suffix}", USER_A) is True
        finally:
            # End the read, or the between-tests TRUNCATE waits on its lock.
            db.session.rollback()
