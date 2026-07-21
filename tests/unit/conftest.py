# Override parent conftest — unit tests don't need Flask app or DB
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def db_cleanup():
    """No-op override of parent db_cleanup."""
    yield


@pytest.fixture(autouse=True)
def _no_flask_app():
    """Prevent Celery ContextTask from creating a real Flask app (and DB connection)."""
    mock_app = MagicMock()

    @contextmanager
    def _fake_app_context():
        yield

    mock_app.app_context = _fake_app_context
    with patch("agentic_project_service.celery.get_flask_app", return_value=mock_app):
        yield


# ---------------------------------------------------------------------------
# Shared test fixtures for unit tests (Tasks 14, 15, 18, 19)
# ---------------------------------------------------------------------------


class _FakeDbSession:
    """Minimal SQLAlchemy-Session shape for MetadataEnricher's batch loop.

    The enricher's batch loop calls:
      - db_session.execute(text(...), params)  — for store_result, _ensure_error_column,
        get_enrichable_items, count_total_items, etc.
      - db_session.commit()                    — between batches

    The fake records every execute() call for later assertion and makes
    commit() a no-op. execute() returns a stub result whose fetchone() /
    fetchall() / scalar() return sentinel values caller can patch via
    monkeypatch if it cares about query results.

    Used by Tasks 14 (circuit breaker), 15 (as_completed streaming),
    18 (per-batch charging), 19 (ops alert).
    """

    def __init__(self):
        self.executed: list[tuple] = []
        self.committed = 0

    def execute(self, statement, params=None):
        self.executed.append((str(statement), params))
        return _FakeResult()

    def commit(self):
        self.committed += 1

    def rollback(self):
        pass


class _FakeResult:
    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def scalar(self):
        return 0


@pytest.fixture
def fake_db_session():
    """Per-test instance of _FakeDbSession."""
    return _FakeDbSession()
