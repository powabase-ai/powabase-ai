"""Database-backed checks for ai.bm25_index_builds (revision 0032).

Runs against ``PG_SEARCH_TEST_DATABASE_URL`` if set, otherwise ``DATABASE_URL``.
The table lives in the (hardcoded) ``ai`` schema, so this uses a brand-new
scratch *database* rather than a scratch schema, the way revision 0030's
tests do -- never the real ``ai`` schema on the target server.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from agentic_project_service.services.bm25_build_outcome import STATUSES
from tests.pg_search.test_migration_0030_extension import scratch_database, server_engine_or_skip
from tests.pg_search.test_partition_migration import load_revision

KB_A = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
KB_B = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"


@pytest.fixture(scope="module")
def revision():
    return load_revision("0032_add_bm25_index_builds_table.py", "mig_0032_bm25_index_builds")


@pytest.fixture(scope="module")
def server_engine():
    eng = server_engine_or_skip()
    yield eng
    eng.dispose()


@pytest.fixture
def scratch_engine(server_engine):
    with scratch_database(server_engine) as eng:
        with eng.begin() as conn:
            conn.execute(text("CREATE SCHEMA ai"))
            conn.execute(text("CREATE TABLE ai.knowledge_bases (id uuid PRIMARY KEY)"))
            conn.execute(
                text("INSERT INTO ai.knowledge_bases VALUES (CAST(:id AS uuid))"), {"id": KB_A}
            )
            conn.execute(
                text("INSERT INTO ai.knowledge_bases VALUES (CAST(:id AS uuid))"), {"id": KB_B}
            )
        yield eng


def _upgrade(revision, conn, monkeypatch) -> None:
    monkeypatch.setattr(revision, "op", SimpleNamespace(get_bind=lambda: conn))
    revision.upgrade()


def _downgrade(revision, conn, monkeypatch) -> None:
    monkeypatch.setattr(revision, "op", SimpleNamespace(get_bind=lambda: conn))
    revision.downgrade()


def _table_exists(conn) -> bool:
    return bool(
        conn.execute(
            text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'ai' AND table_name = 'bm25_index_builds'"
            )
        ).first()
    )


def test_upgrade_creates_the_table(revision, scratch_engine, monkeypatch):
    with scratch_engine.begin() as conn:
        assert not _table_exists(conn)
        _upgrade(revision, conn, monkeypatch)
        assert _table_exists(conn)


def test_upgrade_is_a_no_op_on_a_second_run(revision, scratch_engine, monkeypatch):
    with scratch_engine.begin() as conn:
        _upgrade(revision, conn, monkeypatch)
        _upgrade(revision, conn, monkeypatch)
        assert _table_exists(conn)


def test_downgrade_drops_the_table(revision, scratch_engine, monkeypatch):
    with scratch_engine.begin() as conn:
        _upgrade(revision, conn, monkeypatch)
        _downgrade(revision, conn, monkeypatch)
        assert not _table_exists(conn)


def test_check_constraint_rejects_an_unknown_status(revision, scratch_engine, monkeypatch):
    with scratch_engine.begin() as conn:
        _upgrade(revision, conn, monkeypatch)

    with scratch_engine.connect() as conn:
        with pytest.raises(IntegrityError, match="bm25_index_builds_status_check"):
            conn.execute(
                text(
                    "INSERT INTO ai.bm25_index_builds "
                    "(knowledge_base_id, item_table, status) "
                    "VALUES (CAST(:kb AS uuid), 'chunks', 'not-a-real-status')"
                ),
                {"kb": KB_A},
            )
        conn.rollback()


def test_check_constraint_accepts_every_documented_status(revision, scratch_engine, monkeypatch):
    with scratch_engine.begin() as conn:
        _upgrade(revision, conn, monkeypatch)

    statuses = tuple(sorted(STATUSES))
    assert {"needs_build", "completing", "unavailable"} <= set(statuses)
    with scratch_engine.begin() as conn:
        for status in statuses:
            conn.execute(
                text(
                    "INSERT INTO ai.bm25_index_builds "
                    "(knowledge_base_id, item_table, status) "
                    "VALUES (CAST(:kb AS uuid), :item_table, :status)"
                ),
                {"kb": KB_A, "item_table": f"t_{status}", "status": status},
            )
        rows = conn.execute(text("SELECT status FROM ai.bm25_index_builds")).scalars().all()
        assert sorted(rows) == sorted(statuses)


def test_fk_cascades_when_the_knowledge_base_is_deleted(revision, scratch_engine, monkeypatch):
    with scratch_engine.begin() as conn:
        _upgrade(revision, conn, monkeypatch)
        conn.execute(
            text(
                "INSERT INTO ai.bm25_index_builds (knowledge_base_id, item_table, status) "
                "VALUES (CAST(:kb AS uuid), 'chunks', 'ready')"
            ),
            {"kb": KB_A},
        )
        conn.execute(
            text(
                "INSERT INTO ai.bm25_index_builds (knowledge_base_id, item_table, status) "
                "VALUES (CAST(:kb AS uuid), 'chunks', 'ready')"
            ),
            {"kb": KB_B},
        )

    with scratch_engine.begin() as conn:
        conn.execute(
            text("DELETE FROM ai.knowledge_bases WHERE id = CAST(:kb AS uuid)"), {"kb": KB_A}
        )
        remaining = (
            conn.execute(text("SELECT knowledge_base_id FROM ai.bm25_index_builds")).scalars().all()
        )
        assert [str(r) for r in remaining] == [KB_B]


def test_primary_key_rejects_a_duplicate_kb_and_item_table_pair(
    revision, scratch_engine, monkeypatch
):
    with scratch_engine.begin() as conn:
        _upgrade(revision, conn, monkeypatch)
        conn.execute(
            text(
                "INSERT INTO ai.bm25_index_builds (knowledge_base_id, item_table, status) "
                "VALUES (CAST(:kb AS uuid), 'chunks', 'ready')"
            ),
            {"kb": KB_A},
        )

    with scratch_engine.connect() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                text(
                    "INSERT INTO ai.bm25_index_builds (knowledge_base_id, item_table, status) "
                    "VALUES (CAST(:kb AS uuid), 'chunks', 'queued')"
                ),
                {"kb": KB_A},
            )
        conn.rollback()
