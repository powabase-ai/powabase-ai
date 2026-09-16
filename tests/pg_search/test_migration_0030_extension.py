"""Creating the pg_search extension: revision 0030 and the start-up hook.

``pg_search`` requires ``vector``. A database bootstrapped without ``vector``
used to make the CREATE fail, and the failure was turned into a server-side
WARNING that no application log ever showed, while the revision was stamped
and never ran again. These tests run against a scratch database created from
``template0``, so neither extension is there to begin with.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from tests.pg_search.test_partition_migration import (
    database_url_or_skip,
    load_revision,
    skip_unless_required,
)

PROBE_ROLE = "bm25_ext_probe"


@pytest.fixture(scope="module")
def revision():
    return load_revision("0030_create_pg_search_extension.py", "mig_0030_extension")


def server_engine_or_skip():
    """An engine on the test server, or skip when it has no pg_search."""
    eng = create_engine(database_url_or_skip())
    with eng.connect() as conn:
        available = conn.execute(
            text("SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search'")
        ).first()
    if available is None:
        eng.dispose()
        skip_unless_required("pg_search is not available on this server")
    return eng


@contextmanager
def scratch_database(server):
    """A brand-new database with no extensions at all, dropped afterwards."""
    name = f"{server.url.database}_ext_{uuid.uuid4().hex[:8]}"
    with server.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f'CREATE DATABASE "{name}" TEMPLATE template0'))
    eng = create_engine(server.url.set(database=name))
    try:
        yield eng
    finally:
        eng.dispose()
        with server.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


@pytest.fixture(scope="module")
def server_engine():
    eng = server_engine_or_skip()
    yield eng
    eng.dispose()


@pytest.fixture
def scratch_engine(server_engine):
    with scratch_database(server_engine) as eng:
        yield eng


def _extensions(engine) -> set[str]:
    with engine.connect() as conn:
        return {r[0] for r in conn.execute(text("SELECT extname FROM pg_extension")).all()}


def _upgrade(revision, conn, monkeypatch) -> None:
    monkeypatch.setattr(revision, "op", SimpleNamespace(get_bind=lambda: conn))
    revision.upgrade()


# ---------------------------------------------------------------------------
# Revision 0030
# ---------------------------------------------------------------------------


def test_upgrade_creates_pg_search_and_the_vector_extension_it_requires(
    revision, scratch_engine, monkeypatch
):
    assert not {"vector", "pg_search"} & _extensions(scratch_engine)

    with scratch_engine.begin() as conn:
        _upgrade(revision, conn, monkeypatch)

    assert {"vector", "pg_search"} <= _extensions(scratch_engine)


def test_upgrade_is_a_logged_no_op_where_the_extension_is_not_available(
    revision, scratch_engine, monkeypatch, caplog
):
    monkeypatch.setattr(revision, "EXTENSION", "no_such_extension_for_this_test")

    with caplog.at_level(logging.INFO, logger="alembic.runtime.migration"):
        with scratch_engine.begin() as conn:
            _upgrade(revision, conn, monkeypatch)

    assert "pg_search" not in _extensions(scratch_engine)
    assert any(
        r.levelno == logging.INFO and "not available" in r.getMessage() for r in caplog.records
    )
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_failed_create_is_logged_as_an_error_and_leaves_the_transaction_usable(
    revision, scratch_engine, monkeypatch, caplog
):
    with scratch_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(
            text(
                f"DO $$ BEGIN CREATE ROLE {PROBE_ROLE} NOLOGIN; "
                "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
        )

    with caplog.at_level(logging.INFO, logger="alembic.runtime.migration"):
        with scratch_engine.begin() as conn:
            # pg_search is not a trusted extension: a role without superuser
            # cannot create it, which is exactly the "available but creation
            # fails" case a restricted bootstrap role hits.
            conn.execute(text(f"SET LOCAL ROLE {PROBE_ROLE}"))
            _upgrade(revision, conn, monkeypatch)
            # The next revision runs in this same transaction.
            assert conn.execute(text("SELECT 1")).scalar() == 1

    assert "pg_search" not in _extensions(scratch_engine)
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("pg_search" in m and "permission denied" in m for m in errors), errors


# ---------------------------------------------------------------------------
# Start-up hook
# ---------------------------------------------------------------------------


def test_startup_hook_creates_the_extension_and_is_idempotent(scratch_engine, caplog):
    from agentic_project_service._pg_search_extension import ensure_pg_search_extension

    assert ensure_pg_search_extension(scratch_engine) == "created"
    assert {"vector", "pg_search"} <= _extensions(scratch_engine)

    with caplog.at_level(logging.INFO):
        assert ensure_pg_search_extension(scratch_engine) == "present"
        assert ensure_pg_search_extension(scratch_engine) == "present"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_startup_hook_reports_a_failure_as_an_error_and_never_raises(scratch_engine, caplog):
    from agentic_project_service._pg_search_extension import ensure_pg_search_extension

    with scratch_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(
            text(
                f"DO $$ BEGIN CREATE ROLE {PROBE_ROLE} NOLOGIN; "
                "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
        )
        conn.execute(
            text(f'GRANT CONNECT ON DATABASE "{scratch_engine.url.database}" TO {PROBE_ROLE}')
        )

    # Every connection of this engine runs as the unprivileged role.
    restricted = create_engine(
        scratch_engine.url, connect_args={"options": f"-c role={PROBE_ROLE}"}
    )

    with caplog.at_level(logging.INFO):
        assert ensure_pg_search_extension(restricted) == "failed"

    restricted.dispose()
    assert "pg_search" not in _extensions(scratch_engine)
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("pg_search" in m and "permission denied" in m for m in errors), errors
