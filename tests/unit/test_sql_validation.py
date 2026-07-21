"""Tests for SQL validation in copilot execute_public_sql."""

import pytest

from agentic_project_service.services.copilot import (
    validate_sql_for_public_execution,
)


# ---------------------------------------------------------------------------
# Valid SQL — should return None (no error)
# ---------------------------------------------------------------------------


class TestValidSQL:
    """Commands that copilot users should be allowed to run."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM users",
            "select id, name from users where active = true",
            "INSERT INTO users (name) VALUES ('Alice')",
            "UPDATE users SET name = 'Bob' WHERE id = 1",
            "DELETE FROM users WHERE id = 1",
            "WITH cte AS (SELECT 1) SELECT * FROM cte",
            "EXPLAIN SELECT * FROM users",
            "explain analyze SELECT * FROM users",
            "CREATE TABLE test (id serial primary key, name text)",
            "CREATE INDEX idx_name ON users (name)",
            "CREATE VIEW user_names AS SELECT name FROM users",
            "ALTER TABLE users ADD COLUMN email text",
            "DROP TABLE IF EXISTS temp_data",
            "DROP INDEX idx_name",
            "DROP VIEW user_names",
        ],
    )
    def test_allowed_commands(self, sql):
        assert validate_sql_for_public_execution(sql) is None

    def test_semicolons_inside_string_literals(self):
        sql = "SELECT * FROM t WHERE name = 'O''Brien; DROP TABLE t'"
        assert validate_sql_for_public_execution(sql) is None

    def test_semicolons_inside_double_quoted_identifier(self):
        sql = 'SELECT * FROM "my;table"'
        assert validate_sql_for_public_execution(sql) is None

    def test_schema_name_inside_string_literal(self):
        sql = "SELECT * FROM t WHERE schema_name = 'ai.agents'"
        assert validate_sql_for_public_execution(sql) is None

    def test_leading_whitespace(self):
        sql = "   SELECT 1"
        assert validate_sql_for_public_execution(sql) is None

    def test_trailing_semicolon(self):
        sql = "SELECT 1;"
        assert validate_sql_for_public_execution(sql) is None


# ---------------------------------------------------------------------------
# Multi-statement SQL — should be rejected
# ---------------------------------------------------------------------------


class TestMultiStatement:
    """Multi-statement SQL must always be rejected."""

    def test_two_selects(self):
        assert validate_sql_for_public_execution("SELECT 1; SELECT 2") is not None

    def test_set_then_select(self):
        sql = "SET search_path TO ai; SELECT * FROM agents"
        assert validate_sql_for_public_execution(sql) is not None

    def test_select_then_drop(self):
        sql = "SELECT 1; DROP TABLE users"
        assert validate_sql_for_public_execution(sql) is not None

    def test_semicolon_after_comment(self):
        sql = "SELECT 1 -- comment\n; SELECT 2"
        assert validate_sql_for_public_execution(sql) is not None

    def test_semicolon_in_block_comment(self):
        sql = "SELECT /* ; */ 1; SELECT 2"
        assert validate_sql_for_public_execution(sql) is not None


# ---------------------------------------------------------------------------
# Protected schema references — should be rejected
# ---------------------------------------------------------------------------


class TestProtectedSchemas:
    """References to protected schemas must be blocked."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM ai.agents",
            "SELECT * FROM auth.users",
            "SELECT * FROM storage.objects",
            'SELECT * FROM "ai".agents',
            'SELECT * FROM "auth"."users"',
            "SELECT * FROM AI.agents",
        ],
    )
    def test_schema_references_blocked(self, sql):
        result = validate_sql_for_public_execution(sql)
        assert result is not None
        assert "protected schema" in result.lower() or "schema" in result.lower()


# ---------------------------------------------------------------------------
# Forbidden commands — should be rejected
# ---------------------------------------------------------------------------


class TestForbiddenCommands:
    """Commands not in the allowlist must be rejected."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SET search_path TO ai",
            "SET LOCAL search_path TO ai",
            "GRANT ALL ON TABLE users TO public",
            "REVOKE ALL ON TABLE users FROM public",
            "COPY users TO '/tmp/data.csv'",
            "LOAD 'auto_explain'",
            "TRUNCATE users",
            "CREATE ROLE admin",
            "CREATE SCHEMA private",
            "CREATE DATABASE other",
            "CREATE EXTENSION pg_trgm",
            "ALTER ROLE admin SET search_path TO public",
            "DROP ROLE admin",
            "DROP SCHEMA public CASCADE",
            "DROP DATABASE other",
        ],
    )
    def test_forbidden_commands_rejected(self, sql):
        result = validate_sql_for_public_execution(sql)
        assert result is not None
        assert "not allowed" in result.lower()


# ---------------------------------------------------------------------------
# Comment bypass attempts — should be rejected
# ---------------------------------------------------------------------------


class TestCommentBypass:
    """Validation must strip comments before checking."""

    def test_set_hidden_in_block_comment(self):
        sql = "SET /* hide */ search_path TO ai"
        result = validate_sql_for_public_execution(sql)
        assert result is not None

    def test_line_comment_hiding_command(self):
        sql = "-- innocent\nSET search_path TO ai"
        result = validate_sql_for_public_execution(sql)
        assert result is not None

    def test_block_comment_hiding_schema(self):
        sql = "SELECT * FROM /* public. */ ai.agents"
        result = validate_sql_for_public_execution(sql)
        assert result is not None


# ---------------------------------------------------------------------------
# Dollar quoting — should be rejected
# ---------------------------------------------------------------------------


class TestDollarQuoting:
    """Dollar quoting is rejected outright (no legitimate copilot use)."""

    def test_dollar_quoting_rejected(self):
        sql = "SELECT $$ malicious $$"
        result = validate_sql_for_public_execution(sql)
        assert result is not None

    def test_tagged_dollar_quoting_rejected(self):
        sql = "SELECT $tag$ malicious $tag$"
        result = validate_sql_for_public_execution(sql)
        assert result is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_empty_string(self):
        result = validate_sql_for_public_execution("")
        assert result is not None

    def test_whitespace_only(self):
        result = validate_sql_for_public_execution("   \n\t  ")
        assert result is not None

    def test_case_insensitive_select(self):
        assert validate_sql_for_public_execution("sElEcT 1") is None

    def test_case_insensitive_forbidden(self):
        result = validate_sql_for_public_execution("sEt search_path TO ai")
        assert result is not None

    def test_comment_only(self):
        result = validate_sql_for_public_execution("-- just a comment")
        assert result is not None

    def test_block_comment_only(self):
        result = validate_sql_for_public_execution("/* just a comment */")
        assert result is not None


# ---------------------------------------------------------------------------
# Schema name in string literal — should NOT false-positive
# ---------------------------------------------------------------------------


class TestSchemaInStringLiteral:
    """Schema names inside string literals must not trigger rejection."""

    def test_schema_name_in_string_with_space(self):
        sql = "SELECT * FROM t WHERE x = 'see ai.agents for details'"
        assert validate_sql_for_public_execution(sql) is None

    def test_schema_name_in_string_after_comma(self):
        sql = "INSERT INTO t (a) VALUES ('auth.users reference')"
        assert validate_sql_for_public_execution(sql) is None


# ---------------------------------------------------------------------------
# Extended CREATE/ALTER/DROP syntax — should be allowed
# ---------------------------------------------------------------------------


class TestExtendedDDL:
    """Common DDL variants that must be allowed."""

    def test_create_unique_index(self):
        assert validate_sql_for_public_execution("CREATE UNIQUE INDEX idx ON t(col)") is None

    def test_create_index_concurrently(self):
        assert validate_sql_for_public_execution("CREATE INDEX CONCURRENTLY idx ON t(col)") is None

    def test_create_or_replace_view(self):
        assert validate_sql_for_public_execution("CREATE OR REPLACE VIEW v AS SELECT 1") is None

    def test_create_temporary_table(self):
        assert validate_sql_for_public_execution("CREATE TEMPORARY TABLE tmp (id int)") is None

    def test_create_temp_table(self):
        assert validate_sql_for_public_execution("CREATE TEMP TABLE tmp (id int)") is None

    def test_create_table_if_not_exists(self):
        assert validate_sql_for_public_execution("CREATE TABLE IF NOT EXISTS t (id int)") is None
