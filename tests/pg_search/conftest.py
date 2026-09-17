# Override parent conftest — these tests drive their own engine against a
# Postgres that has the pg_search extension, and never touch the ai schema the
# parent fixtures bootstrap and truncate.
from pathlib import Path

import pytest

# Per-test limit (pytest-timeout) for every test in this directory that does
# not set its own ``@pytest.mark.timeout``. Many of these tests wait on locks
# on purpose; a regression that turns a bounded wait into an unbounded one must
# fail that test, not hang the run. The slowest test, the move under a live
# re-index mix, takes 27-50 s and every other one under 4 s, so this only ever
# fires on a hang. ``--timeout`` on the command line wins.
PG_SEARCH_TEST_TIMEOUT_SECONDS = 120

_HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    if config.getoption("timeout", default=None):
        return
    for item in items:
        if item.get_closest_marker("timeout") is not None:
            continue
        if _HERE in Path(str(item.path)).resolve().parents:
            item.add_marker(pytest.mark.timeout(PG_SEARCH_TEST_TIMEOUT_SECONDS))


@pytest.fixture(autouse=True)
def db_cleanup():
    """No-op override of parent db_cleanup."""
    yield
