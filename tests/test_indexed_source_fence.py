"""Ownership fence: a superseded task cannot commit a durable result."""
import uuid

from sqlalchemy import text

from agentic_project_service.db import db


def _seed_indexing(app, kb_id, owner_task_id):
    """Insert source + an 'indexing' row owned by owner_task_id."""
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
                INSERT INTO ai.indexed_sources
                    (id, knowledge_base_id, source_id, index_status, celery_task_id)
                VALUES (:id, :kb, :src, 'indexing', :owner)
            """),
            {"id": is_id, "kb": kb_id, "src": src_id, "owner": owner_task_id},
        )
        db.session.commit()
    return src_id, is_id


def test_fence_blocks_superseded_status_write(app, test_knowledge_base):
    from agentic_project_service.tasks.indexing import _fenced_mark_indexed

    _, is_id = _seed_indexing(app, test_knowledge_base["id"], owner_task_id="owner-B")
    with app.app_context():
        superseded = _fenced_mark_indexed(is_id, "loser-A", {"artifact_count": 1})
        db.session.rollback()   # a real caller rolls back on 0-row fence
        assert superseded == 0  # loser matched no row

        owner = _fenced_mark_indexed(is_id, "owner-B", {"artifact_count": 1})
        db.session.commit()
        assert owner == 1       # owner matched its row

        row = db.session.execute(
            text("SELECT index_status FROM ai.indexed_sources WHERE id = :id"),
            {"id": is_id},
        ).fetchone()
        assert row.index_status == "indexed"


def test_superseded_execution_leaves_zero_chunks(app, test_knowledge_base, mocker):
    """Real run_indexing path: a superseded task's chunk inserts roll back."""
    from agentic_project_service.tasks import indexing

    src_id, is_id = _seed_indexing(app, test_knowledge_base["id"], owner_task_id="owner-B")

    # Deterministic 1-chunk embed so run_indexing does real DB inserts without
    # calling a real embedding API. The loader returns a bare str.
    mocker.patch.object(indexing, "get_text_derivative_content", return_value="hello world")
    mocker.patch.object(indexing, "get_page_texts_from_derivative", return_value=None)
    mocker.patch(
        "agentic.knowledge.embedder.litellm.LiteLLMEmbedder.aembed_batch",
        return_value=[[0.1] * 1536],
    )

    # Run the body as a task ("loser-A") that does NOT own the row (owner is
    # "owner-B"). _run_index_body starts AFTER the claim, so it always performs
    # the inserts — the FENCE (not the claim) must roll them back.
    with app.app_context():
        indexing._run_index_body(
            knowledge_base_id=test_knowledge_base["id"], source_id=src_id,
            indexed_source_id=is_id, task_id="loser-A", provider_keys={},
        )
        n_chunks = db.session.execute(
            text("SELECT COUNT(*) AS c FROM ai.chunks WHERE indexed_source_id = :id"),
            {"id": is_id},
        ).fetchone().c
        n_emb = db.session.execute(
            text("SELECT COUNT(*) AS c FROM ai.embeddings WHERE indexed_source_id = :id"),
            {"id": is_id},
        ).fetchone().c
        assert n_chunks == 0   # superseded -> fence rolled back the chunk inserts
        assert n_emb == 0      # ... and the embedding inserts
