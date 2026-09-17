"""Live checks for bm25_build_outcome: upsert/read ordering and savepoint isolation.

Runs against a brand-new scratch database (same helper as revision 0030's
tests) with the ``ai`` schema and ``ai.bm25_index_builds`` created via
revision 0032, plus a minimal ``ai.knowledge_bases`` for the FK.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from agentic_project_service.services.bm25_build_outcome import (
    read_bm25_build_outcome,
    record_bm25_build_outcome,
)
from tests.pg_search.test_migration_0030_extension import scratch_database, server_engine_or_skip
from tests.pg_search.test_partition_migration import load_revision

KB_A = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
KB_B = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
MISSING_KB = "00000000-0000-4000-8000-000000000000"


@pytest.fixture(scope="module")
def server_engine():
    eng = server_engine_or_skip()
    yield eng
    eng.dispose()


@pytest.fixture
def scratch_engine(server_engine):
    revision = load_revision("0032_add_bm25_index_builds_table.py", "mig_0032_live_outcome")
    with scratch_database(server_engine) as eng:
        with eng.begin() as conn:
            conn.execute(text("CREATE SCHEMA ai"))
            conn.execute(text("CREATE TABLE ai.knowledge_bases (id uuid PRIMARY KEY)"))
            for kb_id in (KB_A, KB_B):
                conn.execute(
                    text("INSERT INTO ai.knowledge_bases VALUES (CAST(:id AS uuid))"),
                    {"id": kb_id},
                )
        with eng.begin() as conn:
            with patch.object(revision, "op", SimpleNamespace(get_bind=lambda: conn)):
                revision.upgrade()
        yield eng


def test_a_later_record_overwrites_the_earlier_one_for_the_same_item_table(scratch_engine):
    record_bm25_build_outcome(scratch_engine, KB_A, "chunks", "moving")
    time.sleep(0.01)
    record_bm25_build_outcome(scratch_engine, KB_A, "chunks", "ready", attempts=2)

    result = read_bm25_build_outcome(scratch_engine, KB_A)

    assert result == {
        "status": "ready",
        "reason": None,
        "item_table": "chunks",
        "attempts": 2,
        "updated_at": result["updated_at"],
    }
    with scratch_engine.connect() as conn:
        count = conn.execute(
            text(
                "SELECT count(*) FROM ai.bm25_index_builds "
                "WHERE knowledge_base_id = CAST(:kb AS uuid)"
            ),
            {"kb": KB_A},
        ).scalar()
    assert count == 1


def test_read_returns_the_most_recently_updated_item_table_for_the_kb(scratch_engine):
    record_bm25_build_outcome(scratch_engine, KB_A, "chunks", "ready")
    time.sleep(0.01)
    record_bm25_build_outcome(scratch_engine, KB_A, "full_documents", "retrying", reason="lock")

    result = read_bm25_build_outcome(scratch_engine, KB_A)

    assert result["item_table"] == "full_documents"
    assert result["status"] == "retrying"
    assert result["reason"] == "lock"


def test_read_does_not_see_another_kbs_rows(scratch_engine):
    record_bm25_build_outcome(scratch_engine, KB_A, "chunks", "ready")
    record_bm25_build_outcome(scratch_engine, KB_B, "chunks", "failed", reason="gave up")

    result = read_bm25_build_outcome(scratch_engine, KB_B)

    assert result["status"] == "failed"
    assert result["reason"] == "gave up"


def test_a_failing_write_inside_a_callers_connection_leaves_it_usable(scratch_engine):
    with scratch_engine.connect() as conn:
        with conn.begin():
            conn.execute(
                text("INSERT INTO ai.knowledge_bases VALUES (CAST(:id AS uuid))"),
                {"id": "22222222-2222-4222-8222-222222222222"},
            )

            # Violates the FK: MISSING_KB has no row in ai.knowledge_bases.
            record_bm25_build_outcome(conn, MISSING_KB, "chunks", "ready")

            # The connection's transaction must still be usable afterwards.
            conn.execute(
                text("INSERT INTO ai.knowledge_bases VALUES (CAST(:id AS uuid))"),
                {"id": "33333333-3333-4333-8333-333333333333"},
            )

    with scratch_engine.connect() as conn:
        ids = {str(r[0]) for r in conn.execute(text("SELECT id FROM ai.knowledge_bases")).all()}
        assert "22222222-2222-4222-8222-222222222222" in ids
        assert "33333333-3333-4333-8333-333333333333" in ids
        builds = conn.execute(text("SELECT count(*) FROM ai.bm25_index_builds")).scalar()
        assert builds == 0


def test_a_failing_write_inside_a_callers_session_leaves_it_usable(scratch_engine):
    session = Session(bind=scratch_engine)
    try:
        session.execute(
            text("INSERT INTO ai.knowledge_bases VALUES (CAST(:id AS uuid))"),
            {"id": "44444444-4444-4444-8444-444444444444"},
        )

        record_bm25_build_outcome(session, MISSING_KB, "chunks", "ready")

        session.execute(
            text("INSERT INTO ai.knowledge_bases VALUES (CAST(:id AS uuid))"),
            {"id": "55555555-5555-4555-8555-555555555555"},
        )
        session.commit()
    finally:
        session.close()

    with scratch_engine.connect() as conn:
        ids = {str(r[0]) for r in conn.execute(text("SELECT id FROM ai.knowledge_bases")).all()}
        assert "44444444-4444-4444-8444-444444444444" in ids
        assert "55555555-5555-4555-8555-555555555555" in ids
