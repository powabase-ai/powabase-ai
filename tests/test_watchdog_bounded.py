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


def test_exhausted_rows_are_never_re_dispatched(app, orphan):
    """A row written 'failed' at the bound must not also be handed to .delay().

    The dispatch loop iterates `recoverable`, not `rows`. Nothing pinned that:
    changing it to `rows` re-dispatches the very rows the bound just retired --
    the exact unbounded-retry behaviour this whole change exists to stop -- and
    the suite stayed green. This asserts the split directly.
    """
    from agentic_project_service.tasks import indexing, watchdog

    at_bound = orphan(attempts=indexing.MAX_ATTEMPTS)
    under_bound = orphan(attempts=indexing.MAX_ATTEMPTS - 1)

    fake_redis = MagicMock()
    fake_redis.lrange.return_value = []
    fake_inspect = MagicMock()
    fake_inspect.active.return_value = {"w": []}
    fake_inspect.reserved.return_value = {"w": []}
    with (
        app.app_context(),
        patch.object(watchdog, "_get_redis", return_value=(fake_redis, "t:lock")),
        patch.object(watchdog.celery_app.control, "inspect", return_value=fake_inspect),
        patch("agentic_project_service.tasks.indexing.index_source.delay") as delay,
        patch.object(watchdog, "get_all_user_provider_keys", return_value={}),
    ):
        watchdog._run_one_tick()

    assert _row(app, at_bound).index_status == "failed"
    assert _row(app, under_bound).index_status == "pending"
    # exactly one dispatch, and it is the under-bound row
    assert delay.call_count == 1
    dispatched_ids = {str(a) for c in delay.call_args_list for a in c.args} | {
        str(v) for c in delay.call_args_list for v in c.kwargs.values()
    }
    assert under_bound in dispatched_ids
    assert at_bound not in dispatched_ids   # the retired row was NOT re-queued
