"""Manual reindex is a fresh intent: it resets attempts to 0."""
import uuid

from sqlalchemy import text

from agentic_project_service.db import db


def test_selective_reindex_resets_attempts(app, client, mock_auth, auth_headers,
                                            test_knowledge_base):
    kb_id = test_knowledge_base["id"]
    src_id, is_id = str(uuid.uuid4()), str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text("""INSERT INTO ai.sources (id,name,file_type,storage_path,extraction_status)
                    VALUES (:id,'s','application/pdf','sources/s.pdf','extracted')"""),
            {"id": src_id},
        )
        db.session.execute(
            text("""INSERT INTO ai.indexed_sources
                    (id,knowledge_base_id,source_id,index_status,attempts)
                    VALUES (:id,:kb,:src,'failed',3)"""),
            {"id": is_id, "kb": kb_id, "src": src_id},
        )
        db.session.commit()

    resp = client.post(
        f"/api/knowledge-bases/{kb_id}/reindex",
        json={"indexed_source_ids": [is_id]},
        headers=auth_headers,
    )
    assert resp.status_code == 200

    with app.app_context():
        row = db.session.execute(
            text("SELECT index_status, attempts FROM ai.indexed_sources WHERE id=:id"),
            {"id": is_id},
        ).fetchone()
    assert row.index_status == "pending"
    assert row.attempts == 0     # fresh intent


def test_reindex_of_an_in_flight_row_supersedes_its_task(app, client, mock_auth,
                                                          auth_headers, test_knowledge_base):
    """Reindex must supersede a task that is still running, not co-exist with it.

    The reset zeroes attempts. Without also dropping ownership, an in-flight
    task keeps its celery_task_id, still WINS its fence, and commits a result
    against a counter that was just reset -- so the row records a budget it did
    not have, and a genuinely-consumed attempt is lost.

    Guarding with `AND index_status <> 'indexing'` would be the wrong fix: it
    makes Reindex a no-op on exactly the row a user is most likely to press it
    on -- one that appears stuck. Clearing celery_task_id supersedes instead,
    through the machinery that already exists: the running task's terminal
    write is fenced on that column, so it matches nothing and rolls back, and
    the freshly dispatched task claims the row cleanly.
    """
    from agentic_project_service.tasks.indexing import _fenced_mark_indexed

    kb_id = test_knowledge_base["id"]
    src_id, is_id = str(uuid.uuid4()), str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text("""INSERT INTO ai.sources (id,name,file_type,storage_path,extraction_status)
                    VALUES (:id,'s','application/pdf','sources/s.pdf','extracted')"""),
            {"id": src_id},
        )
        db.session.execute(
            text("""INSERT INTO ai.indexed_sources
                    (id,knowledge_base_id,source_id,index_status,celery_task_id,attempts)
                    VALUES (:id,:kb,:src,'indexing','live-worker',2)"""),
            {"id": is_id, "kb": kb_id, "src": src_id},
        )
        db.session.commit()

    resp = client.post(f"/api/knowledge-bases/{kb_id}/reindex",
                       json={"indexed_source_ids": [is_id]}, headers=auth_headers)
    assert resp.status_code == 200

    with app.app_context():
        row = db.session.execute(
            text("SELECT index_status, attempts, celery_task_id FROM ai.indexed_sources WHERE id=:id"),
            {"id": is_id},
        ).fetchone()
        assert row.index_status == "pending"
        assert row.attempts == 0
        assert row.celery_task_id is None      # ownership dropped

        # and the still-running task can no longer commit a result
        superseded = _fenced_mark_indexed(is_id, "live-worker", {"artifact_count": 1})
        db.session.rollback()
    assert superseded == 0
