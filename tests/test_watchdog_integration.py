"""Integration test: watchdog finds and recovers a real orphan row."""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from agentic_project_service.db import db


@pytest.fixture
def orphan_row(app, test_knowledge_base):
    """Insert source + indexed_sources row that looks orphaned
    (status='indexing', celery_task_id set to something Celery doesn't know
    about, last_dispatched_at old). db_cleanup autouse handles teardown.

    Uses the test_knowledge_base fixture from conftest for KB creation
    (avoids replicating all required ai.knowledge_bases columns inline).
    """
    with app.app_context():
        kb_id = test_knowledge_base["id"]
        src_id = uuid.uuid4()
        is_id = uuid.uuid4()
        old_ts = datetime.now(timezone.utc) - timedelta(hours=1)
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
                    id, knowledge_base_id, source_id, index_status,
                    celery_task_id, last_dispatched_at
                ) VALUES (:id, :kb, :src, 'indexing', :tid, :ts)
            """),
            {
                "id": is_id,
                "kb": kb_id,
                "src": src_id,
                "tid": "task-not-in-celery",
                "ts": old_ts,
            },
        )
        db.session.commit()
        yield {"kb_id": kb_id, "src_id": src_id, "indexed_source_id": is_id}


def test_watchdog_recovers_real_orphan(app, orphan_row):
    """End-to-end: orphan in DB + Celery doesn't know its task_id -> row gets reset."""
    from agentic_project_service.tasks import watchdog

    is_id = orphan_row["indexed_source_id"]

    fake_redis = MagicMock()
    fake_redis.lrange.return_value = []
    fake_inspect = MagicMock()
    fake_inspect.active.return_value = {"worker-a": []}  # responded, no live tasks
    fake_inspect.reserved.return_value = {"worker-a": []}

    captured = []

    def fake_delay(kb_id, src_id, **kwargs):
        captured.append((kb_id, src_id, kwargs.get("indexed_source_id")))

    with (
        app.app_context(),
        patch.object(watchdog, "_get_redis", return_value=(fake_redis, "test:lock")),
        patch.object(watchdog.celery_app.control, "inspect", return_value=fake_inspect),
        patch("agentic_project_service.tasks.indexing.index_source.delay", side_effect=fake_delay),
        patch.object(watchdog, "get_all_user_provider_keys", return_value={}),
    ):
        watchdog._run_one_tick()

    # The orphan should now be 'pending' with a fresh last_dispatched_at
    with app.app_context():
        row = db.session.execute(
            text("SELECT index_status, last_dispatched_at FROM ai.indexed_sources WHERE id = :id"),
            {"id": is_id},
        ).fetchone()
        assert row.index_status == "pending"
        assert row.last_dispatched_at > datetime.now(timezone.utc) - timedelta(minutes=1)

    # And index_source.delay was called for this row
    assert any(is_id == captured_is for _, _, captured_is in captured)
