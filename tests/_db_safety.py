"""Safety guard for the test database connection (issue #164).

The autouse `db_cleanup` fixture in `tests/conftest.py` TRUNCATEs ai.*
tables between tests. When `DATABASE_URL` resolves to the database that
`make up` brings online (the default `postgres` database), that wipes
ai-schema rows the developer cares about.

`assert_safe_test_database` refuses to proceed in that case with a clear
message pointing at the fix.
"""

from sqlalchemy.engine import make_url


def assert_safe_test_database(url: str) -> None:
    """Raise pytest.UsageError if `url` targets the dev `postgres` database.

    Lazy-imports pytest so this module stays usable from tooling that doesn't
    have pytest installed (e.g. ad-hoc scripts).
    """
    import pytest

    parsed = make_url(url)
    if parsed.database == "postgres":
        raise pytest.UsageError(
            f"Refusing to run tests against database `postgres` "
            f"(host={parsed.host}, port={parsed.port}). The autouse "
            f"db_cleanup fixture would wipe ai.* rows on the developer's "
            f"running stack.\n\n"
            f"Set DATABASE_URL to a separate test database before running "
            f"pytest, e.g.:\n"
            f"  DATABASE_URL=postgresql+psycopg://supabase_admin:$POSTGRES_PASSWORD"
            f"@localhost:5432/postgres_test\n\n"
            f"`make test` creates and uses "
            f"`postgres_test` automatically. See issue #164."
        )
