"""Integration test: ON CONFLICT path refreshes last_dispatched_at.

When a source is re-added to a KB that already has an indexed_sources row,
the ON CONFLICT DO UPDATE clause must refresh last_dispatched_at so the
watchdog's dispatch-race guard doesn't immediately flag it as orphaned.
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from agentic_project_service.db import db


def test_on_conflict_path_refreshes_last_dispatched_at(
    app, client, auth_headers, mock_auth, test_knowledge_base
):
    """When a source is re-added to a KB, ON CONFLICT must refresh last_dispatched_at
    so the watchdog's dispatch-race guard works."""
    kb_id = test_knowledge_base["id"]
    src_id = uuid.uuid4()
    old_ts = datetime.now(timezone.utc) - timedelta(hours=2)

    with app.app_context():
        # Create the source first
        db.session.execute(
            text("""
                INSERT INTO ai.sources (id, name, extraction_status, file_type, storage_path)
                VALUES (:id, 'test-src', 'extracted', 'application/pdf', 'sources/test.pdf')
            """),
            {"id": src_id},
        )
        # Manually seed the indexed_source with old last_dispatched_at
        db.session.execute(
            text("""
                INSERT INTO ai.indexed_sources (id, knowledge_base_id, source_id, index_status, last_dispatched_at)
                VALUES (:id, :kb, :src, 'failed', :ts)
            """),
            {"id": uuid.uuid4(), "kb": kb_id, "src": src_id, "ts": old_ts},
        )
        db.session.commit()

    # Patch index_source.delay to avoid needing a real Celery worker
    with patch("agentic_project_service.routes.knowledge_bases.index_source") as mock_index_source:
        mock_task = MagicMock()
        mock_task.id = str(uuid.uuid4())
        mock_index_source.delay.return_value = mock_task

        resp = client.post(
            f"/api/knowledge-bases/{kb_id}/sources",
            json={"source_id": str(src_id)},
            headers=auth_headers,
        )

    assert resp.status_code == 201, (
        f"Add-source failed: {resp.status_code} {resp.get_data(as_text=True)}"
    )

    with app.app_context():
        row = db.session.execute(
            text(
                "SELECT index_status, last_dispatched_at"
                " FROM ai.indexed_sources"
                " WHERE knowledge_base_id = :kb AND source_id = :src"
            ),
            {"kb": kb_id, "src": src_id},
        ).fetchone()
        assert row.index_status == "pending"
        assert row.last_dispatched_at > old_ts, (
            f"last_dispatched_at was not refreshed by ON CONFLICT; still {row.last_dispatched_at}"
        )
