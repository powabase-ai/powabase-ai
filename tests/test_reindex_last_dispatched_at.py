"""Integration test: /reindex sets last_dispatched_at to a fresh NOW()."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from agentic_project_service.db import db


@pytest.fixture
def seed_indexed_source(app, test_knowledge_base):
    """Insert one source + indexed_sources row with an old last_dispatched_at.

    Uses the test_knowledge_base fixture (created via API) to avoid duplicating
    the NOT NULL columns required by ai.knowledge_bases.
    ai.sources requires: id, name, file_type, storage_path, extraction_status.
    """
    with app.app_context():
        src_id = uuid.uuid4()
        is_id = uuid.uuid4()
        old_ts = datetime.now(timezone.utc) - timedelta(hours=2)
        db.session.execute(
            text("""
                INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status)
                VALUES (:id, 'test-src', 'application/pdf', 'sources/test-src.pdf', 'extracted')
            """),
            {"id": src_id},
        )
        db.session.execute(
            text("""
                INSERT INTO ai.indexed_sources (
                    id, knowledge_base_id, source_id, index_status, last_dispatched_at
                ) VALUES (:id, :kb, :src, 'failed', :ts)
            """),
            {
                "id": is_id,
                "kb": test_knowledge_base["id"],
                "src": src_id,
                "ts": old_ts,
            },
        )
        db.session.commit()
        yield {
            "kb_id": test_knowledge_base["id"],
            "src_id": src_id,
            "indexed_source_id": is_id,
            "old_ts": old_ts,
        }


def test_reindex_selective_updates_last_dispatched_at(
    app, client, seed_indexed_source, auth_headers, mock_auth
):
    """POST /reindex with indexed_source_ids must refresh last_dispatched_at."""
    kb_id = seed_indexed_source["kb_id"]
    is_id = seed_indexed_source["indexed_source_id"]
    old_ts = seed_indexed_source["old_ts"]

    resp = client.post(
        f"/api/knowledge-bases/{kb_id}/reindex",
        json={"indexed_source_ids": [str(is_id)]},
        headers=auth_headers,
    )
    assert resp.status_code == 200

    with app.app_context():
        row = db.session.execute(
            text("SELECT last_dispatched_at, index_status FROM ai.indexed_sources WHERE id = :id"),
            {"id": is_id},
        ).fetchone()
        assert row.index_status == "pending"
        assert row.last_dispatched_at > old_ts, (
            f"last_dispatched_at was not refreshed (still {row.last_dispatched_at})"
        )
