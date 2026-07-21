# Override parent conftest — services unit tests don't need Flask app or DB
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
