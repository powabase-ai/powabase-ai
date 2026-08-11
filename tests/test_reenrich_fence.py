"""Reenrich stamps its own celery_task_id so the reconciler won't hijack it."""
import uuid

from sqlalchemy import text

from agentic_project_service.db import db


def _seed_indexed(app, kb_id):
    """A previously-indexed row (populated, stale celery_task_id) — the hijack target."""
    src_id, is_id = str(uuid.uuid4()), str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text("""INSERT INTO ai.sources (id,name,file_type,storage_path,extraction_status)
                    VALUES (:id,'s','application/pdf','sources/s.pdf','extracted')"""),
            {"id": src_id},
        )
        db.session.execute(
            text("""INSERT INTO ai.indexed_sources
                    (id,knowledge_base_id,source_id,index_status,celery_task_id)
                    VALUES (:id,:kb,:src,'indexed','old-dead-task')"""),
            {"id": is_id, "kb": kb_id, "src": src_id},
        )
        db.session.commit()
    return is_id


def test_mark_reenriching_stamps_own_task_id(app, test_knowledge_base):
    """After marking, the row is 'indexing' AND owned by the reenrich task id
    (not the stale 'old-dead-task'), so the liveness check skips it."""
    from agentic_project_service.routes.knowledge_bases import _mark_reenriching

    is_id = _seed_indexed(app, test_knowledge_base["id"])
    with app.app_context():
        _mark_reenriching(test_knowledge_base["id"], is_id, "reenrich-task-42")
        row = db.session.execute(
            text("SELECT index_status, celery_task_id, last_dispatched_at "
                 "FROM ai.indexed_sources WHERE id = :id"),
            {"id": is_id},
        ).fetchone()

    assert row.index_status == "indexing"
    assert row.celery_task_id == "reenrich-task-42"   # NOT the stale dead id
    assert row.last_dispatched_at is not None


def _seed(app, kb_id, status, task_id=None):
    """One source + one indexed_source in the given status."""
    src_id, is_id = str(uuid.uuid4()), str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text("""INSERT INTO ai.sources (id,name,file_type,storage_path,extraction_status)
                    VALUES (:id,'s','application/pdf','sources/s.pdf','extracted')"""),
            {"id": src_id},
        )
        db.session.execute(
            text("""INSERT INTO ai.indexed_sources
                    (id,knowledge_base_id,source_id,index_status,celery_task_id)
                    VALUES (:id,:kb,:src,:st,:tid)"""),
            {"id": is_id, "kb": kb_id, "src": src_id, "st": status, "tid": task_id},
        )
        db.session.commit()
    return is_id


def test_kb_wide_mark_leaves_a_pending_row_claimable(app, test_knowledge_base):
    """A queued-but-unclaimed source must survive a KB-wide reenrich mark.

    Without the 'indexed' scope the mark flips 'pending' -> 'indexing', and
    _claim_indexed_source only accepts 'pending' -- so index_source exits
    'skipped' without re-dispatching and the source is never indexed, while the
    reenrich completion stamps the row 'indexed'. Green, and empty.
    """
    from agentic_project_service.routes.knowledge_bases import _mark_reenriching
    from agentic_project_service.tasks import indexing

    is_id = _seed(app, test_knowledge_base["id"], "pending")

    with app.app_context():
        _mark_reenriching(test_knowledge_base["id"], None, "reenrich-task")
        row = db.session.execute(
            text("SELECT index_status FROM ai.indexed_sources WHERE id=:id"), {"id": is_id}
        ).fetchone()
        assert row.index_status == "pending"   # untouched by the mark
        # and therefore still claimable, which is the property that matters
        assert indexing._claim_indexed_source(is_id, "worker-task") is not None


def test_kb_wide_mark_does_not_steal_a_live_task_fence(app, test_knowledge_base):
    """A row an index_source task currently owns must keep its owner.

    Overwriting celery_task_id makes the live task's terminal write match zero
    rows, so its inserts roll back -- while its DELETEs, already committed, do
    not. The source ends up stripped and then stamped 'indexed'.
    """
    from agentic_project_service.routes.knowledge_bases import _mark_reenriching

    is_id = _seed(app, test_knowledge_base["id"], "indexing", task_id="live-worker")

    with app.app_context():
        _mark_reenriching(test_knowledge_base["id"], None, "reenrich-task")
        row = db.session.execute(
            text("SELECT index_status, celery_task_id FROM ai.indexed_sources WHERE id=:id"),
            {"id": is_id},
        ).fetchone()

    assert row.index_status == "indexing"
    assert row.celery_task_id == "live-worker"   # the live owner keeps the fence
