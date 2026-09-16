"""Revision 0031 gives up on a held lock instead of hanging the boot.

The conversion renames each item table, which needs ACCESS EXCLUSIVE. Migrations
run at application start, so a long reader holding the table -- a nightly
``pg_dump``, an analytics query -- would otherwise queue the rename, and every
query of the table behind it, for as long as that reader runs. The revision
bounds each lock wait, fails with an error that names the table and says the
next start retries, and rolls back cleanly.
"""

from __future__ import annotations

import logging
import time

import pytest
from sqlalchemy import create_engine, text

from tests.pg_search.test_partition_migration import (
    SCHEMA,
    _create_unpartitioned,
    database_url_or_skip,
    load_revision,
)


@pytest.fixture(scope="module")
def migration():
    return load_revision("0031_partition_bm25_item_tables.py", "mig_0031_lock_timeout")


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(database_url_or_skip())
    yield eng
    eng.dispose()


@pytest.fixture
def prefilled(engine):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        _create_unpartitioned(conn, rows_per_kb=1)
    yield
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))


def _relkind(conn, relname: str) -> str | None:
    row = conn.execute(
        text(
            "SELECT relkind FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = :s AND c.relname = :r"
        ),
        {"s": SCHEMA, "r": relname},
    ).first()
    return row[0] if row else None


def test_a_held_lock_fails_the_migration_quickly_and_leaves_the_table_alone(
    migration, engine, prefilled, monkeypatch, caplog
):
    monkeypatch.setattr(migration, "LOCK_TIMEOUT_MS", 300)

    with engine.connect() as reader:
        # What a pg_dump holds for its whole run: ACCESS SHARE on every table.
        reader.execute(text(f"SELECT count(*) FROM {SCHEMA}.full_documents"))
        try:
            started = time.monotonic()
            with caplog.at_level(logging.ERROR, logger="alembic.runtime.migration"):
                with pytest.raises(migration.MigrationLockTimeout) as excinfo:
                    with engine.begin() as conn:
                        # Safety net only: without the migration's own bound
                        # this statement_timeout is what ends the wait, and
                        # its QueryCanceled is not the error expected above.
                        conn.execute(text("SET LOCAL statement_timeout = '10s'"))
                        migration.partition_item_tables(conn, schema=SCHEMA)
            elapsed = time.monotonic() - started
        finally:
            reader.rollback()

    assert elapsed < 5
    message = str(excinfo.value)
    assert f"{SCHEMA}.full_documents" in message
    assert "next start" in message
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    # The log names the table and the session in the way, so an operator can
    # see what to wait for (or stop).
    assert any("full_documents" in m and "pid " in m for m in errors), errors

    with engine.connect() as conn:
        # The whole transaction rolled back, including the table converted
        # before the one that timed out.
        assert _relkind(conn, "chunks") == "r"
        assert _relkind(conn, "full_documents") == "r"
        assert _relkind(conn, "chunks_default") is None


def test_the_bound_is_scoped_to_the_conversion(migration, engine, prefilled):
    """Later revisions in the same transaction keep the server's lock_timeout."""
    with engine.begin() as conn:
        conn.execute(text("SET LOCAL lock_timeout = '42s'"))
        seen: list[str] = []
        original = migration._partition_one

        def spy(bind, schema, table):
            seen.append(bind.execute(text("SHOW lock_timeout")).scalar())
            original(bind, schema, table)

        migration._partition_one = spy
        try:
            migration.partition_item_tables(conn, schema=SCHEMA)
        finally:
            migration._partition_one = original

        assert seen and all(value != "42s" for value in seen)
        assert conn.execute(text("SHOW lock_timeout")).scalar() == "42s"
        assert _relkind(conn, "chunks") == "p"


def test_downgrade_is_bounded_the_same_way(migration, engine, prefilled, monkeypatch):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        migration.partition_item_tables(conn, schema=SCHEMA)
    monkeypatch.setattr(migration, "LOCK_TIMEOUT_MS", 300)

    with engine.connect() as reader:
        reader.execute(text(f"SELECT count(*) FROM {SCHEMA}.chunks"))
        try:
            with pytest.raises(migration.MigrationLockTimeout):
                with engine.begin() as conn:
                    conn.execute(text("SET LOCAL statement_timeout = '10s'"))
                    migration.unpartition_item_tables(conn, schema=SCHEMA)
        finally:
            reader.rollback()

    with engine.connect() as conn:
        assert _relkind(conn, "chunks") == "p"
