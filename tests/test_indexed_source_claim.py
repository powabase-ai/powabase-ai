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


def _seed_indexed_with_content(app, kb_id):
    """Like _seed_pending, but also attaches a chunk + embedding to the
    indexed_source, standing in for a row a prior run already indexed.
    Returns (src_id, is_id); the row starts 'pending' same as _seed_pending.
    """
    src_id, is_id = _seed_pending(app, kb_id)
    chunk_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text("""
                INSERT INTO ai.chunks (id, indexed_source_id, knowledge_base_id, source_id, text)
                VALUES (:id, :is_id, :kb_id, :src_id, 'pre-existing chunk')
            """),
            {"id": chunk_id, "is_id": is_id, "kb_id": kb_id, "src_id": src_id},
        )
        db.session.execute(
            text("""
                INSERT INTO ai.embeddings (
                    item_id, item_table, indexed_source_id, knowledge_base_id,
                    source_id, embedding_model, dims, embedding
                )
                VALUES (
                    :item_id, 'chunks', :is_id, :kb_id, :src_id,
                    'test-model', 3, CAST(:embedding AS vector)
                )
            """),
            {
                "item_id": chunk_id,
                "is_id": is_id,
                "kb_id": kb_id,
                "src_id": src_id,
                "embedding": "[0,0,0]",
            },
        )
        db.session.commit()
    return src_id, is_id


def test_claim_before_reindex_cleanup_prevents_loser_deleting_content(app, test_knowledge_base):
    """Regression guard for claim-before-cleanup ordering.

    If the claim is moved back to run AFTER the reindex branch's embedding/chunk
    deletion — the ordering this shipped with before it was corrected — this test
    fails: the loser deletes and commits the winner's already-indexed chunk and
    embedding before discovering it lost the claim. Against the current ordering
    (claim runs before any reindex cleanup) it passes: the loser exits on the
    claim check before touching either row.
    """
    from agentic_project_service.tasks import indexing

    src_id, is_id = _seed_indexed_with_content(app, test_knowledge_base["id"])
    with app.app_context():
        # Pre-claim as the winner so this task's own claim will lose.
        indexing._claim_indexed_source(is_id, "winner")

    with app.app_context():
        result = indexing.index_source.run(
            test_knowledge_base["id"], src_id,
            indexed_source_id=is_id, provider_keys={},
        )
    assert result["status"] == "skipped"

    with app.app_context():
        chunk_count = db.session.execute(
            text("SELECT COUNT(*) FROM ai.chunks WHERE indexed_source_id = :id"),
            {"id": is_id},
        ).scalar()
        embedding_count = db.session.execute(
            text("SELECT COUNT(*) FROM ai.embeddings WHERE indexed_source_id = :id"),
            {"id": is_id},
        ).scalar()
    assert chunk_count == 1
    assert embedding_count == 1
