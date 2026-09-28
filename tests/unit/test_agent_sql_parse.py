"""The parse-time gate on SQL an agent's database_query tool runs.

The tool runs on a non-superuser login under the caller's role, so Postgres
privileges and RLS are the real boundary. This gate closes what they cannot:
the caller's identity lives in session settings (``request.jwt.claims``,
``role``) that any SQL can overwrite with ``set_config`` in the same
statement, which would let one end user read as another, or one agent assume
another agent's grants. So the gate accepts one plain SELECT, reports every
table it reads (checked against the agent's allowlist later, in the
database, where names resolve exactly as the query will), and rejects any
function that is not a built-in or that can change settings or run a SQL
string.
"""

import pytest

from agentic_project_service.services.agent_sql import AgentSqlRejected, parse_select


def _rels(sql):
    return sorted(parse_select(sql).relations, key=lambda r: (r[0] or "", r[1]))


def _funcs(sql):
    return sorted(parse_select(sql).functions)


class TestAcceptsPlainSelects:
    def test_reports_tables_and_functions(self):
        parsed = parse_select("SELECT count(*), lower(name) FROM customers WHERE id > 1")
        assert parsed.relations == [(None, "customers")]
        assert sorted(parsed.functions) == ["count", "lower"]

    def test_schema_qualified_table_keeps_its_schema(self):
        assert _rels("SELECT * FROM sales.orders o JOIN customers c ON c.id = o.cid") == [
            (None, "customers"),
            ("sales", "orders"),
        ]

    def test_trailing_semicolon_is_fine(self):
        assert _rels("SELECT 1 FROM t;") == [(None, "t")]

    def test_pg_catalog_qualified_builtin_is_fine(self):
        assert _funcs("SELECT pg_catalog.lower('A')") == ["lower"]

    def test_union_and_subqueries_report_every_table(self):
        sql = "SELECT id FROM a UNION SELECT id FROM b WHERE id IN (SELECT id FROM c)"
        assert _rels(sql) == [(None, "a"), (None, "b"), (None, "c")]

    def test_set_returning_function_in_from(self):
        assert _funcs("SELECT * FROM generate_series(1, 3) g") == ["generate_series"]


class TestRejectsAnythingButOneSelect:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1; SELECT 2",
            "INSERT INTO t VALUES (1)",
            "UPDATE t SET a = 1",
            "DELETE FROM t",
            "EXPLAIN SELECT 1",
            "SET ROLE postgres",
            "CALL p()",
            "DO $$ BEGIN END $$",
            "SELECT * INTO copy_of_t FROM t",
            "SELECT * FROM t FOR UPDATE",
            "SELECT * FROM t WHERE id IN (SELECT id FROM u FOR SHARE)",
            "SELECT id FROM a UNION SELECT id INTO x FROM b",
            "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d",
            "",
            "SELEC 1",
        ],
    )
    def test_rejected(self, sql):
        with pytest.raises(AgentSqlRejected):
            parse_select(sql)


class TestRejectsIdentityAndSqlStringFunctions:
    @pytest.mark.parametrize(
        "sql",
        [
            # Forging the end user, or assuming another role.
            "SELECT set_config('request.jwt.claims', '{\"sub\":\"x\"}', true)",
            "SELECT * FROM t WHERE set_config('role', 'other', true) IS NOT NULL",
            "SELECT (SELECT set_config('role', 'other', true)), (SELECT count(*) FROM t)",
            "WITH c AS (SELECT set_config('role', 'x', true)) SELECT * FROM c",
            "SELECT pg_catalog.set_config('role', 'x', true)",
            "SELECT \"set_config\"('role', 'x', true)",
            "SELECT SET_CONFIG('role', 'x', true)",
            # Running a SQL string, which the parse gate cannot see into.
            "SELECT query_to_xml('select 1', true, true, '')",
            "SELECT table_to_xml('t', true, true, '')",
            "SELECT cursor_to_xml('c', 1, true, true, '')",
            # Reading other sessions, files, large objects; sleeping; locking.
            "SELECT * FROM pg_stat_get_activity(NULL)",
            "SELECT pg_read_file('/etc/hostname')",
            "SELECT lo_import('/etc/passwd')",
            "SELECT pg_sleep(100)",
            "SELECT pg_advisory_lock(1)",
            "SELECT pg_terminate_backend(1)",
            # Functions outside pg_catalog could run any SQL inside them.
            "SELECT public.my_function(1)",
            "SELECT * FROM auth.users_view()",
        ],
    )
    def test_rejected(self, sql):
        with pytest.raises(AgentSqlRejected):
            parse_select(sql)


class TestCteScoping:
    """A CTE name only hides a table where Postgres would resolve to the CTE."""

    def test_cte_reference_is_not_a_table(self):
        assert _rels("WITH t AS (SELECT 1 AS x) SELECT * FROM t") == []

    def test_cte_body_of_its_own_name_reads_the_real_table(self):
        """Non-recursive: inside its own body, `t` is the table, not the CTE."""
        assert _rels("WITH t AS (SELECT * FROM t) SELECT * FROM t") == [(None, "t")]

    def test_later_cte_can_use_an_earlier_one(self):
        sql = "WITH a AS (SELECT 1 AS x), b AS (SELECT * FROM a) SELECT * FROM b"
        assert _rels(sql) == []

    def test_earlier_cte_cannot_see_a_later_one(self):
        sql = "WITH a AS (SELECT * FROM b), b AS (SELECT 1 AS x) SELECT * FROM a"
        assert _rels(sql) == [(None, "b")]

    def test_recursive_cte_may_reference_itself(self):
        sql = (
            "WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM r WHERE n < 3) "
            "SELECT * FROM r"
        )
        assert _rels(sql) == []

    def test_cte_does_not_leak_out_of_its_subquery(self):
        sql = "SELECT * FROM secret, (WITH secret AS (SELECT 1 AS x) SELECT * FROM secret) s"
        assert _rels(sql) == [(None, "secret")]

    def test_cte_never_hides_a_schema_qualified_table(self):
        sql = "WITH orders AS (SELECT 1 AS x) SELECT * FROM public.orders"
        assert _rels(sql) == [("public", "orders")]
