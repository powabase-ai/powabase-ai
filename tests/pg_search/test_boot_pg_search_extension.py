"""Application start-up enables pg_search when the server provides it.

A project whose Postgres gains the extension after revision 0030 was stamped
must still get it: that revision never runs again, so start-up does it.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from tests.pg_search.test_migration_0030_extension import scratch_database, server_engine_or_skip


@pytest.fixture
def scratch_engine():
    server = server_engine_or_skip()
    try:
        with scratch_database(server) as eng:
            yield eng
    finally:
        server.dispose()


def _installed(engine) -> set[str]:
    with engine.connect() as conn:
        return {r[0] for r in conn.execute(text("SELECT extname FROM pg_extension")).all()}


def test_create_app_enables_pg_search_and_a_second_start_is_harmless(scratch_engine, monkeypatch):
    from agentic_project_service.db import db
    from agentic_project_service.main import create_app

    url = scratch_engine.url.render_as_string(hide_password=False)
    monkeypatch.setenv("DATABASE_URL", url)
    assert "pg_search" not in _installed(scratch_engine)

    for _ in range(2):
        app = create_app()
        with app.app_context():
            db.engine.dispose()

    assert {"vector", "pg_search"} <= _installed(scratch_engine)


def test_create_app_starts_when_the_extension_cannot_be_created(scratch_engine, monkeypatch):
    from agentic_project_service import _pg_search_extension
    from agentic_project_service.db import db
    from agentic_project_service.main import create_app

    monkeypatch.setattr(_pg_search_extension, "EXTENSION", "no_such_extension_for_this_test")
    monkeypatch.setenv("DATABASE_URL", scratch_engine.url.render_as_string(hide_password=False))

    app = create_app()
    with app.app_context():
        db.engine.dispose()
    assert app is not None
    assert "pg_search" not in _installed(scratch_engine)


def test_create_app_sweeps_leftover_move_checks_and_survives_a_failing_sweep(
    scratch_engine, monkeypatch
):
    """Start-up clears checks a killed partition move left on DEFAULT, and a
    sweep that blows up anyway must not stop the service starting."""
    from agentic_project_service.db import db
    from agentic_project_service.main import create_app
    from agentic_project_service.services import pg_bm25_index

    calls: list = []

    def _sweep(engine):
        calls.append(engine)
        raise RuntimeError("sweep failed")

    monkeypatch.setattr(pg_bm25_index, "clear_leftover_move_checks_at_start", _sweep)
    monkeypatch.setenv("DATABASE_URL", scratch_engine.url.render_as_string(hide_password=False))

    app = create_app()
    with app.app_context():
        db.engine.dispose()

    assert app is not None
    assert len(calls) == 1
