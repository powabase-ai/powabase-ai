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
                "SELECT index_status, error_message, celery_task_id, attempts "
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


def test_recovered_row_is_reclaimable_and_spends_one_attempt(app, orphan):
    """A recovered row can be claimed again, and that claim costs one attempt.

    The earlier version asserted ``celery_task_id == "dead-task"`` after
    recovery — that the reset leaves the stale id in place. A mutation run
    showed the assertion inverted: making recovery NULL the id (behaviourally
    harmless — the claim overwrites it either way) failed THIS test and nothing
    else, while every mutation that broke real behaviour left it passing. It
    pinned an implementation detail and guarded no requirement, which its own
    docstring conceded.

    What actually has to hold is that recovery hands the row back in a state
    the claim accepts, and that the round trip consumes budget — otherwise a
    row that dies every time would cycle forever, which is the bug this whole
    change exists to stop.
    """
    from agentic_project_service.tasks import indexing

    is_id = orphan(attempts=indexing.MAX_ATTEMPTS - 2)
    before = _row(app, is_id).attempts

    _run_watchdog(app)
    assert _row(app, is_id).index_status == "pending"   # handed back for retry

    with app.app_context():
        claimed = indexing._claim_indexed_source(is_id, "fresh-task")
    assert claimed is not None                          # ... and it IS claimable
    after = _row(app, is_id)
    assert after.index_status == "indexing"
    assert after.attempts == before + 1                 # the retry cost budget
    assert after.celery_task_id == "fresh-task"         # the claimer owns it now


def _run_watchdog_with_broken_broker(app):
    """Same tick, but the re-dispatch publish raises (broker down)."""
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
        patch(
            "agentic_project_service.tasks.indexing.index_source.delay",
            side_effect=OSError("broker unreachable"),
        ),
        patch.object(watchdog, "get_all_user_provider_keys", return_value={}),
    ):
        watchdog._run_one_tick()


def test_unqueueable_recovery_is_terminal_not_stranded(app, orphan):
    """A recovery whose publish never reaches the broker must end terminal.

    The reset to 'pending' is committed before the dispatch. If the publish
    raises and we only log it, the row sits at 'pending' with part of its
    budget spent -- and ORPHAN_QUERY only ever selects 'indexing', so nothing
    finds it again. Stranded, in the one component whose job is to un-strand
    rows; a broker outage would do it to every orphan in a single tick.
    """
    is_id = orphan(attempts=0)

    _run_watchdog_with_broken_broker(app)

    row = _row(app, is_id)
    assert row.index_status == "failed"          # terminal, not left at 'pending'
    assert row.error_message is not None
    # cause-neutral: the reconciler cannot see WHY the broker refused
    assert "could not be re-queued" in row.error_message
