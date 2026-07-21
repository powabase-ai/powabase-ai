# Override parent conftest — these integration tests exercise the LiteLLM
# callback dispatch path with mock_response and don't need a Flask app or DB.
import pytest


@pytest.fixture(autouse=True)
def db_cleanup():
    """No-op override of parent db_cleanup."""
    yield
