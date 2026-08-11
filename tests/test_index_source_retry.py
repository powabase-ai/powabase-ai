"""StorageError folds into the single attempts bound; no self.retry."""
import uuid

from sqlalchemy import text

from agentic_project_service.db import db


def _seed_indexing(app, kb_id, owner, attempts):
    src_id, is_id = str(uuid.uuid4()), str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text("""INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status)
                    VALUES (:id, 's', 'application/pdf', 'sources/s.pdf', 'extracted')"""),
            {"id": src_id},
        )
        db.session.execute(
            text("""INSERT INTO ai.indexed_sources
                    (id, knowledge_base_id, source_id, index_status, celery_task_id, attempts)
                    VALUES (:id, :kb, :src, 'indexing', :owner, :att)"""),
            {"id": is_id, "kb": kb_id, "src": src_id, "owner": owner, "att": attempts},
        )
        db.session.commit()
    return src_id, is_id


def test_storage_error_under_bound_resets_to_pending_and_redispatches(
    app, test_knowledge_base, mocker
):
    from agentic_project_service.tasks import indexing

    src_id, is_id = _seed_indexing(app, test_knowledge_base["id"], "owner-B", attempts=1)
    redispatch = mocker.patch.object(indexing.index_source, "delay")

    with app.app_context():
        indexing._handle_storage_error(
            knowledge_base_id=test_knowledge_base["id"], source_id=src_id,
            indexed_source_id=is_id, task_id="owner-B", provider_keys={},
        )
        row = db.session.execute(
            text("SELECT index_status FROM ai.indexed_sources WHERE id = :id"),
            {"id": is_id},
        ).fetchone()

    assert row.index_status == "pending"   # attempts (1) < MAX (3): retry
    redispatch.assert_called_once()


def test_storage_error_at_bound_marks_failed(app, test_knowledge_base, mocker):
    from agentic_project_service.tasks import indexing

    src_id, is_id = _seed_indexing(
        app, test_knowledge_base["id"], "owner-B", attempts=indexing.MAX_ATTEMPTS
    )
    redispatch = mocker.patch.object(indexing.index_source, "delay")

    with app.app_context():
        indexing._handle_storage_error(
            knowledge_base_id=test_knowledge_base["id"], source_id=src_id,
            indexed_source_id=is_id, task_id="owner-B", provider_keys={},
        )
        row = db.session.execute(
            text("SELECT index_status FROM ai.indexed_sources WHERE id = :id"),
            {"id": is_id},
        ).fetchone()

    assert row.index_status == "failed"    # attempts == MAX: terminal
    redispatch.assert_not_called()


def test_storage_error_and_worker_death_share_one_bound(app, test_knowledge_base, mocker):
    """Interleaved StorageError + worker-death consume ONE counter, not two.

    Before this change Celery's own max_retries ran alongside the reconciler's
    re-dispatch, so a row could execute up to max_retries + MAX_ATTEMPTS times.
    With the single `attempts` bound, total executions stay <= MAX_ATTEMPTS
    however the earlier deaths were caused.
    """
    from agentic_project_service.tasks import indexing

    # attempts=1 already consumed by a simulated worker-death (OOM: the row was
    # claimed, the process died, the reconciler reset it to 'pending').
    src_id, is_id = _seed_indexing(app, test_knowledge_base["id"], "owner-B", attempts=1)
    redispatch = mocker.patch.object(indexing.index_source, "delay")

    # Now a *different* failure mode on the same row, MAX_ATTEMPTS-1 more times.
    with app.app_context():
        for _ in range(indexing.MAX_ATTEMPTS - 1):
            db.session.execute(
                text("""UPDATE ai.indexed_sources
                        SET attempts = attempts + 1, celery_task_id = 'owner-B'
                        WHERE id = :id"""),
                {"id": is_id},
            )
            db.session.commit()
            indexing._handle_storage_error(
                knowledge_base_id=test_knowledge_base["id"], source_id=src_id,
                indexed_source_id=is_id, task_id="owner-B", provider_keys={},
            )
        row = db.session.execute(
            text("SELECT index_status, attempts FROM ai.indexed_sources WHERE id = :id"),
            {"id": is_id},
        ).fetchone()

    assert row.index_status == "failed"          # terminal, not looping
    assert row.attempts <= indexing.MAX_ATTEMPTS  # ONE counter, not two
    # The final at-bound call must not re-dispatch; earlier under-bound ones may.
    assert redispatch.call_count < indexing.MAX_ATTEMPTS


def test_storage_error_retries_via_delay_not_celery_retry(app, test_knowledge_base, mocker):
    """The retry goes through .delay(), NOT self.retry — one counter, not two.

    The earlier version of this test asserted only ``retry.assert_not_called()``.
    That cannot fail from anything being broken: it passes just as well when
    _handle_storage_error does nothing at all, which is exactly what a mutation
    run showed (handler gutted -> its three siblings failed, this one passed).
    A negative assertion needs a positive beside it, or it is not evidence.

    So assert the whole disposition: the retry happened, and it happened
    through the path that shares the ``attempts`` budget. self.retry would be a
    second counter Celery increments independently, which is what allowed a row
    to execute max_retries + MAX_ATTEMPTS times before this change.
    """
    from agentic_project_service.tasks import indexing

    src_id, is_id = _seed_indexing(app, test_knowledge_base["id"], "owner-B", attempts=1)
    retry = mocker.patch.object(indexing.index_source, "retry")
    delay = mocker.patch.object(indexing.index_source, "delay")

    with app.app_context():
        indexing._handle_storage_error(
            knowledge_base_id=test_knowledge_base["id"], source_id=src_id,
            indexed_source_id=is_id, task_id="owner-B", provider_keys={},
        )
        row = db.session.execute(
            text("SELECT index_status FROM ai.indexed_sources WHERE id = :id"),
            {"id": is_id},
        ).fetchone()

    delay.assert_called_once()          # a retry DID happen ...
    retry.assert_not_called()           # ... and not through Celery's own counter
    assert row.index_status == "pending"  # ... leaving the row claimable again
