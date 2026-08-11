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
