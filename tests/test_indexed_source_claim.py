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


def test_claim_is_mutually_exclusive(app, test_knowledge_base):
    """Two claims on one 'pending' row: exactly one wins; the other gets None."""
    from agentic_project_service.tasks.indexing import _claim_indexed_source

    _, is_id = _seed_pending(app, test_knowledge_base["id"])
    with app.app_context():
        first = _claim_indexed_source(is_id, "task-A")
        second = _claim_indexed_source(is_id, "task-B")

        assert first == 1          # claimed; attempts incremented 0 -> 1
        assert second is None      # row no longer 'pending' -> not claimable

        row = db.session.execute(
            text("SELECT index_status, celery_task_id, attempts "
                 "FROM ai.indexed_sources WHERE id = :id"),
            {"id": is_id},
        ).fetchone()
        assert row.index_status == "indexing"
        assert row.celery_task_id == "task-A"   # first claimer owns the row
        assert row.attempts == 1                # incremented exactly once


def test_index_source_noops_when_not_claimable(app, test_knowledge_base, mocker):
    """A duplicate task on a non-'pending' row returns before loading content."""
    from agentic_project_service.tasks import indexing

    src_id, is_id = _seed_pending(app, test_knowledge_base["id"])
    with app.app_context():
        # Pre-claim the row as a different owner so our task can't claim it.
        indexing._claim_indexed_source(is_id, "other-owner")

    load_spy = mocker.patch.object(
        indexing, "get_text_derivative_content",
        side_effect=AssertionError("must not load"),
    )
    with app.app_context():
        result = indexing.index_source.run(
            test_knowledge_base["id"], src_id,
            indexed_source_id=is_id, provider_keys={},
        )
    assert result["status"] == "skipped"
    load_spy.assert_not_called()
