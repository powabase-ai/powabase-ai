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
