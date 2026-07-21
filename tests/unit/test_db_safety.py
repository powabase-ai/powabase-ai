"""Guard for issue #164: refuse to run tests against the dev `postgres` DB.

The autouse `db_cleanup` fixture in `tests/conftest.py` TRUNCATEs ai.* tables
between tests. Pointed at the database that `make up` brings online, that wipes
ai-schema rows the developer cares about. The helper under test refuses early.
"""

import pytest

from tests._db_safety import assert_safe_test_database


class TestRefusesDevDatabase:
    def test_localhost_5432_postgres(self):
        with pytest.raises(pytest.UsageError, match="Refusing"):
            assert_safe_test_database(
                "postgresql+psycopg://supabase_admin:secret@localhost:5432/postgres"
            )

    def test_127_0_0_1(self):
        with pytest.raises(pytest.UsageError, match="Refusing"):
            assert_safe_test_database(
                "postgresql+psycopg://supabase_admin:secret@127.0.0.1:5432/postgres"
            )

    def test_plain_postgresql_scheme(self):
        with pytest.raises(pytest.UsageError, match="Refusing"):
            assert_safe_test_database("postgresql://supabase_admin:secret@db:5432/postgres")


class TestAllowsTestDatabase:
    def test_postgres_test(self):
        assert_safe_test_database(
            "postgresql+psycopg://supabase_admin:secret@localhost:5432/postgres_test"
        )

    def test_arbitrary_test_db_name(self):
        assert_safe_test_database(
            "postgresql+psycopg://supabase_admin:secret@localhost:5432/agentic_test"
        )


class TestErrorMessage:
    def test_mentions_database_url_env_var(self):
        with pytest.raises(pytest.UsageError) as exc_info:
            assert_safe_test_database(
                "postgresql+psycopg://supabase_admin:secret@localhost:5432/postgres"
            )
        assert "DATABASE_URL" in str(exc_info.value)

    def test_suggests_test_db_name(self):
        with pytest.raises(pytest.UsageError) as exc_info:
            assert_safe_test_database(
                "postgresql+psycopg://supabase_admin:secret@localhost:5432/postgres"
            )
        assert "postgres_test" in str(exc_info.value)

    def test_links_to_issue(self):
        with pytest.raises(pytest.UsageError) as exc_info:
            assert_safe_test_database(
                "postgresql+psycopg://supabase_admin:secret@localhost:5432/postgres"
            )
        assert "164" in str(exc_info.value)
