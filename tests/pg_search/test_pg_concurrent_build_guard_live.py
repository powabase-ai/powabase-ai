"""The guard against a pg_search that cannot build a bm25 index under writes.

Stock pg_search 0.25.9 on Postgres 15 and 16 fails, or crashes the server on,
a bm25 ``CREATE INDEX CONCURRENTLY`` while the table takes writes, and nothing
the extension reports tells that build from one with paradedb/paradedb#6211.
A Postgres image with the fix says so with the placeholder setting
``powabase.pg_search_cic_safe = on`` in ``postgresql.conf``.

The CI image sets that marker, so a server without it is simulated per session
(``SET powabase.pg_search_cic_safe = off``), which is what ``current_setting``
then answers. The stock image answers NULL; both are "not safe" on Postgres 15
with pg_search 0.25.9.

Same database and scratch schema as ``test_pg_bm25_live``.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event, text

from agentic_project_service.services import pg_bm25_index as pgb
from tests.pg_search import test_pg_bm25_live as live
from tests.pg_search.test_pg_bm25_live import (
    KB_A,
    _indexdef,
    _rows_in,
    _set_language,
    _set_strategy,
)

migration = live.migration
engine = live.engine
scratch_schema = live.scratch_schema
session = live.session


def _engine_with_marker(engine, value: str):
    """An engine on the same server whose sessions see the marker as ``value``."""
    marked = create_engine(engine.url)

    @event.listens_for(marked, "connect")
    def _set_marker(dbapi_connection, _record):
        with dbapi_connection.cursor() as cursor:
            cursor.execute(f"SET {pgb.PG_SEARCH_CIC_SAFE_MARKER} = '{value}'")
        dbapi_connection.commit()

    return marked


@pytest.fixture
def unmarked(engine, monkeypatch):
    monkeypatch.setattr(pgb, "_unsafe_build_warning_logged", False)
    server = _engine_with_marker(engine, "off")
    with server.connect() as conn:
        version, extversion = conn.execute(
            text(
                "SELECT current_setting('server_version_num')::int, "
                "(SELECT extversion FROM pg_extension WHERE extname = 'pg_search')"
            )
        ).one()
    if version >= pgb.UNAFFECTED_SERVER_VERSION_NUM:
        server.dispose()
        pytest.fail(f"this suite targets Postgres 15/16; the server is {version}")
    assert extversion.startswith("0.25."), extversion
    yield server
    server.dispose()


@pytest.fixture
def marked(engine):
    server = _engine_with_marker(engine, "on")
    yield server
    server.dispose()


def _partition_attached(session, kb_id=KB_A) -> bool:
    try:
        return pgb.partition_exists(session, kb_id, "chunks")
    finally:
        session.rollback()


def test_the_guard_reads_the_marker_and_nothing_else_on_this_server(unmarked, marked):
    with unmarked.connect() as conn:
        assert pgb.concurrent_build_safety(conn) == (False, "unverified")
        assert pgb.concurrent_build_known_unsafe(conn) is True
    with marked.connect() as conn:
        assert pgb.concurrent_build_safety(conn) == (
            True,
            f"marker {pgb.PG_SEARCH_CIC_SAFE_MARKER}",
        )
        assert pgb.concurrent_build_known_unsafe(conn) is False


def test_without_the_marker_nothing_is_moved_or_built(unmarked, session, caplog):
    with caplog.at_level("WARNING", logger=pgb.logger.name):
        outcome = pgb.ensure_bm25_index(KB_A, engine=unmarked, allow_row_move=True)
        again = pgb.ensure_bm25_index(KB_A, engine=unmarked, allow_row_move=True)

    assert outcome["status"] == "unavailable", outcome
    assert again["status"] == "unavailable", again
    assert not _partition_attached(session)
    assert _rows_in(session, "chunks_default", KB_A) == 3
    assert _indexdef(session) is None
    assert len([r for r in caplog.records if "powabase.pg_search_cic_safe" in r.getMessage()]) == 1


def test_without_the_marker_a_working_index_survives_a_language_change(unmarked, marked, session):
    assert pgb.ensure_bm25_index(KB_A, engine=marked, allow_row_move=True)["status"] == "ready"
    before = _indexdef(session)
    assert "stemmer=german" in before

    _set_language(session, KB_A, "english")
    outcome = pgb.ensure_bm25_index(KB_A, engine=unmarked)

    assert outcome["status"] == "unavailable", outcome
    assert _indexdef(session) == before
    assert pgb.bm25_index_state(session, KB_A, "chunks") == "ready"
    session.rollback()


def test_with_the_marker_the_move_and_build_go_ahead(marked, session):
    outcome = pgb.ensure_bm25_index(KB_A, engine=marked, allow_row_move=True)
    assert outcome["status"] == "ready", outcome
    assert _partition_attached(session)
    assert _rows_in(session, pgb.partition_name(KB_A, "chunks")) == 3


def test_the_setting_vouches_for_a_server_without_the_marker(unmarked, session, monkeypatch):
    monkeypatch.setattr(pgb, "_concurrent_build_override", lambda: True)
    outcome = pgb.ensure_bm25_index(KB_A, engine=unmarked, allow_row_move=True)
    assert outcome["status"] == "ready", outcome
    assert "stemmer=german" in _indexdef(session)


def test_a_partition_left_without_an_index_is_not_given_one(unmarked, engine, session):
    """Moved on an earlier image, say: the partition stays, no build starts."""
    pgb.create_partition(engine, KB_A, "chunks")
    outcome = pgb.ensure_bm25_index(KB_A, engine=unmarked)
    assert outcome["status"] == "unavailable", outcome
    assert _indexdef(session) is None
    with engine.connect() as conn:
        building = conn.execute(text("SELECT count(*) FROM pg_stat_progress_create_index")).scalar()
    assert building == 0


def test_without_the_marker_a_strategy_change_keeps_the_index_it_left(unmarked, marked, session):
    """chunk_embed -> full_document -> chunk_embed on a server that cannot build:
    the chunks index must still be there when the strategy comes back."""
    assert pgb.ensure_bm25_index(KB_A, engine=marked, allow_row_move=True)["status"] == "ready"
    before = _indexdef(session)
    assert before

    _set_strategy(session, KB_A, "full_document")
    outcome = pgb.ensure_bm25_index(KB_A, engine=unmarked)
    assert outcome["status"] == "unavailable", outcome
    assert "dropped_indexes" not in outcome
    assert _indexdef(session) == before

    _set_strategy(session, KB_A, "chunk_embed")
    assert pgb.ensure_bm25_index(KB_A, engine=unmarked)["status"] == "ready"
    assert _indexdef(session) == before


def test_a_refused_login_is_not_retried_as_transient(engine):
    wrong = create_engine(engine.url.set(password="not-the-password"))
    try:
        with wrong.connect():
            pass
    except Exception as exc:
        error = exc
    else:
        error = None
    finally:
        wrong.dispose()
    assert error is not None
    assert pgb.is_transient_db_error(error) is False
