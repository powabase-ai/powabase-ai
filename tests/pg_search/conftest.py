# Override parent conftest — these tests drive their own engine against a
# Postgres that has the pg_search extension, and never touch the ai schema the
# parent fixtures bootstrap and truncate.
import pytest


@pytest.fixture(autouse=True)
def db_cleanup():
    """No-op override of parent db_cleanup."""
    yield
