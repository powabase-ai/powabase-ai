"""Test configuration and fixtures for the project service."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root so POSTGRES_PASSWORD is available
# without having to pass it on every pytest invocation.
_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=False)

# Set environment variables BEFORE any app imports
_pg_password = os.environ.get("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault(
    "DATABASE_URL",
    f"postgresql+psycopg://supabase_admin:{_pg_password}@localhost:5432/postgres_test",
)
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("STORAGE_URL", "http://localhost:5000/storage/v1")

# Refuse early if DATABASE_URL points at the dev `postgres` DB (issue #164).
from tests._db_safety import assert_safe_test_database  # noqa: E402

assert_safe_test_database(os.environ["DATABASE_URL"])

import uuid  # noqa: E402

import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402

from agentic_project_service.db import db  # noqa: E402
from agentic_project_service.main import create_app  # noqa: E402
from agentic_project_service.services import billing_port  # noqa: E402
from tests.support.billing import RecordingBillingAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def app():
    """Create the Flask application for testing.

    Bootstraps the ``ai`` schema and ORM tables on session start.
    Requires a running Postgres (``make up`` brings up the full stack).
    """
    application = create_app()
    application.config["TESTING"] = True

    with application.app_context():
        # Ensure schema + pgvector extension exist
        db.session.execute(text("CREATE SCHEMA IF NOT EXISTS ai"))
        db.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        db.session.commit()

        # Create all ORM-defined tables (ai.sources, ai.agents, etc.)
        db.metadata.create_all(db.engine)
        db.session.commit()

    yield application


@pytest.fixture
def client(app):
    """Flask test client.

    Uses the client as a context manager so the application context remains
    active for the full duration of the test. This lets tests call
    ``db.session`` directly after making requests without wrapping every
    DB access in an explicit ``app.app_context()`` block.
    """
    with app.test_client() as c:
        yield c


@pytest.fixture
def auth_headers():
    """Request headers with a fake Bearer token."""
    return {
        "Authorization": "Bearer fake-token",
        "Content-Type": "application/json",
    }


@pytest.fixture
def mock_auth(mocker):
    """Mock JWT auth so every request authenticates as a test user.

    Returns the generated ``user_id`` so tests can assert ownership.
    """
    test_user_id = str(uuid.uuid4())
    mocker.patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": test_user_id, "role": "authenticated"},
    )
    return test_user_id


# ---------------------------------------------------------------------------
# DB cleanup (runs after every test)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def db_cleanup(app):
    """Truncate all ai-schema tables between tests."""
    yield
    with app.app_context():
        db.session.execute(
            text(
                "TRUNCATE "
                '"ai".workflow_block_logs, '
                '"ai".workflow_executions, '
                '"ai".hooks, '
                '"ai".orchestration_runs, '
                '"ai".orchestration_sessions, '
                '"ai".orchestration_entities, '
                '"ai".orchestrations, '
                '"ai".agent_runs, '
                '"ai".agent_sessions, '
                '"ai".agent_mcp_servers, '
                '"ai".agent_tools, '
                '"ai".agent_knowledge_bases, '
                '"ai".context_handlers, '
                '"ai".enrichment_configs, '
                '"ai".embeddings, '
                '"ai".chunks, '
                '"ai".full_documents, '
                '"ai".page_index_nodes, '
                '"ai".page_index_toc, '
                '"ai".graph_index_nodes, '
                '"ai".graph_index_toc, '
                '"ai".doc2json_documents, '
                '"ai".indexed_sources, '
                '"ai".sources, '
                '"ai".knowledge_bases, '
                '"ai".agents, '
                '"ai".tools, '
                '"ai".ai_provider_keys '
                "CASCADE"
            )
        )
        db.session.commit()


# ---------------------------------------------------------------------------
# Convenience record fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_source(app):
    """Insert an already-extracted source directly into the DB."""
    source_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".sources
                    (id, name, file_type, storage_path, extraction_status)
                VALUES
                    (:id, :name, :file_type, :storage_path, 'extracted')
                """
            ),
            {
                "id": source_id,
                "name": "test.pdf",
                "file_type": "application/pdf",
                "storage_path": "sources/test.pdf",
            },
        )
        db.session.commit()
    return {"id": source_id, "name": "test.pdf"}


@pytest.fixture
def test_indexed_source(app, test_source, test_knowledge_base):
    """Insert an indexed_source row linking a source to a knowledge base."""
    indexed_source_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".indexed_sources
                    (id, source_id, knowledge_base_id, index_status)
                VALUES
                    (:id, :source_id, :kb_id, 'completed')
                """
            ),
            {
                "id": indexed_source_id,
                "source_id": test_source["id"],
                "kb_id": test_knowledge_base["id"],
            },
        )
        db.session.commit()
    return {
        "id": indexed_source_id,
        "source_id": test_source["id"],
        "kb_id": test_knowledge_base["id"],
    }


@pytest.fixture
def test_agent(client, mock_auth, auth_headers):
    """Create a test agent via the API."""
    resp = client.post(
        "/api/agents",
        json={"name": "Test Agent", "model": "gpt-4o-mini"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    return resp.get_json()


@pytest.fixture
def test_knowledge_base(client, mock_auth, auth_headers):
    """Create a test knowledge base via the API."""
    resp = client.post(
        "/api/knowledge-bases",
        json={"name": "Test KB"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    return resp.get_json()


# ---------------------------------------------------------------------------
# Billing-port test isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_billing():
    """Install a RecordingBillingAdapter for the test; restore the prior adapter after."""
    saved = billing_port.get_billing_adapter()
    rec = RecordingBillingAdapter()
    billing_port.set_billing_adapter(rec)
    yield rec
    billing_port.set_billing_adapter(saved)


@pytest.fixture(autouse=True)
def _billing_adapter_isolation():
    """Backstop: save/restore the module-global billing adapter around EVERY test.
    test_billing_bootstrap.py calls create_app() -> install_billing() (Task 3),
    which registers the CloudBillingAdapter and does NOT reset it; create_app()
    runs in many tests, so without this any later test that assumes the no-op
    default becomes order-dependent. recording_billing restores too, but it is
    opt-in — this autouse fixture covers the rest."""
    saved = billing_port.get_billing_adapter()
    yield
    billing_port.set_billing_adapter(saved)
