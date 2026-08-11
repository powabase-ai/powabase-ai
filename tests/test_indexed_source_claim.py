"""Claim + attempts-column behavior for ai.indexed_sources."""
import uuid

from sqlalchemy import text

from agentic_project_service.db import db


def _seed_pending(app, kb_id):
    """Insert an extracted source + a 'pending' indexed_source; return ids."""
    src_id, is_id = str(uuid.uuid4()), str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text("""
                INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status)
                VALUES (:id, 'src', 'application/pdf', 'sources/src.pdf', 'extracted')
            """),
            {"id": src_id},
        )
        db.session.execute(
            text("""
                INSERT INTO ai.indexed_sources (id, knowledge_base_id, source_id, index_status)
                VALUES (:id, :kb, :src, 'pending')
            """),
            {"id": is_id, "kb": kb_id, "src": src_id},
        )
        db.session.commit()
    return src_id, is_id


def test_attempts_defaults_to_zero(app, test_knowledge_base):
    _, is_id = _seed_pending(app, test_knowledge_base["id"])
    with app.app_context():
        row = db.session.execute(
            text("SELECT attempts FROM ai.indexed_sources WHERE id = :id"),
            {"id": is_id},
        ).fetchone()
        assert row.attempts == 0
