"""Reconciler bounds retries: attempts>=MAX -> failed (neutral), else pending."""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from agentic_project_service.db import db


@pytest.fixture
def orphan(app, test_knowledge_base):
    """Build an orphan row (status 'indexing', dead owner, stale) at N attempts."""

    def _make(attempts):
        src_id, is_id = uuid.uuid4(), uuid.uuid4()
        old = datetime.now(timezone.utc) - timedelta(hours=1)
        with app.app_context():
            db.session.execute(
                text("""INSERT INTO ai.sources (id,name,file_type,storage_path,extraction_status)
                        VALUES (:id,'s','application/pdf','sources/s.pdf','extracted')"""),
                {"id": src_id},
            )
            db.session.execute(
                text("""INSERT INTO ai.indexed_sources
                        (id,knowledge_base_id,source_id,index_status,celery_task_id,
                         last_dispatched_at,attempts)
                        VALUES (:id,:kb,:src,'indexing','dead-task',:ts,:att)"""),
                {
                    "id": is_id,
                    "kb": test_knowledge_base["id"],
                    "src": src_id,
                    "ts": old,
                    "att": attempts,
                },
            )
            db.session.commit()
        return str(is_id)

    return _make


def _run_watchdog(app):
    from agentic_project_service.tasks import watchdog

    fake_redis = MagicMock()
    fake_redis.lrange.return_value = []
    fake_inspect = MagicMock()
    fake_inspect.active.return_value = {"w": []}
    fake_inspect.reserved.return_value = {"w": []}
    with (
        app.app_context(),
        patch.object(watchdog, "_get_redis", return_value=(fake_redis, "t:lock")),
        patch.object(watchdog.celery_app.control, "inspect", return_value=fake_inspect),
        patch("agentic_project_service.tasks.indexing.index_source.delay"),
        patch.object(watchdog, "get_all_user_provider_keys", return_value={}),
    ):
        watchdog._run_one_tick()


def _row(app, is_id):
    with app.app_context():
        return db.session.execute(
            text(
                "SELECT index_status, error_message, celery_task_id "
                "FROM ai.indexed_sources WHERE id=:id"
            ),
            {"id": is_id},
        ).fetchone()


def test_orphan_under_bound_goes_pending(app, orphan):
    from agentic_project_service.tasks import indexing

    is_id = orphan(attempts=indexing.MAX_ATTEMPTS - 1)
    _run_watchdog(app)
    assert _row(app, is_id).index_status == "pending"


def test_orphan_at_bound_goes_failed_with_neutral_message(app, orphan):
    from agentic_project_service.tasks import indexing

    is_id = orphan(attempts=indexing.MAX_ATTEMPTS)
    _run_watchdog(app)
    row = _row(app, is_id)
    assert row.index_status == "failed"
    msg = (row.error_message or "").lower()
    assert "attempts" in msg
    for banned in ("resource limit", "too big", "raise the tier", "oom", "memory"):
        assert banned not in msg  # cause-neutral


def test_recovered_row_keeps_stale_owner_until_reclaimed(app, orphan):
    """Recovery does NOT clear celery_task_id; the next claim overwrites it.

    Pins the coupling between the reconciler's reset and the claim's ownership
    write — if recovery ever started NULLing celery_task_id, the claim would
    still work, but this assertion documents which component owns that field.
    """
    from agentic_project_service.tasks import indexing

    is_id = orphan(attempts=indexing.MAX_ATTEMPTS - 1)
    _run_watchdog(app)
    row = _row(app, is_id)
    assert row.index_status == "pending"
    assert row.celery_task_id == "dead-task"  # untouched by recovery

    with app.app_context():
        assert indexing._claim_indexed_source(is_id, "fresh-task") is not None
    assert _row(app, is_id).celery_task_id == "fresh-task"  # claim takes ownership
