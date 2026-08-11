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


def _seed_stale_chunk(app, kb_id, src_id, is_id):
    """One pre-existing chunk, so the reindex cleanup has an id to remove.

    Without it ``chunk_ids_to_remove`` is empty and the removal call is guarded
    out, which would make the assertions below pass vacuously.
    """
    chunk_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text("""
                INSERT INTO ai.chunks
                    (id, indexed_source_id, knowledge_base_id, source_id, text, chunk_index)
                VALUES (:id, :is_id, :kb, :src, 'stale chunk from the previous run', 0)
            """),
            {"id": chunk_id, "is_id": is_id, "kb": kb_id, "src": src_id},
        )
        db.session.commit()
    return chunk_id


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


def test_superseded_execution_fires_no_side_effects(app, test_knowledge_base, mocker):
    """A superseded task performs NO BM25 build, billing, or enrichment."""
    from agentic_project_service.tasks import indexing

    src_id, is_id = _seed_indexing(app, test_knowledge_base["id"], owner_task_id="owner-B")
    mocker.patch.object(indexing, "get_text_derivative_content", return_value="hello world")
    mocker.patch.object(indexing, "get_page_texts_from_derivative", return_value=None)
    mocker.patch(
        "agentic.knowledge.embedder.litellm.LiteLLMEmbedder.aembed_batch",
        return_value=[[0.1] * 1536],
    )
    bm25 = mocker.patch(
        "agentic_project_service.services.sparse_retrieval."
        "sparse_index_store.SparseIndexStore.add_and_save"
    )
    charge = mocker.patch("agentic_project_service.services.billing_port.charge")

    with app.app_context():
        indexing._run_index_body(
            knowledge_base_id=test_knowledge_base["id"], source_id=src_id,
            indexed_source_id=is_id, task_id="loser-A", provider_keys={},
            idempotency_action="indexing", idempotency_parts=[is_id],
        )

    bm25.assert_not_called()     # superseded -> no BM25 append
    charge.assert_not_called()   # superseded -> no billing


def test_owner_execution_fires_each_side_effect_once(app, test_knowledge_base, mocker):
    """The owner commits, then fires BM25 + billing exactly once each."""
    from agentic_project_service.tasks import indexing

    src_id, is_id = _seed_indexing(app, test_knowledge_base["id"], owner_task_id="owner-B")
    mocker.patch.object(indexing, "get_text_derivative_content", return_value="hello world")
    mocker.patch.object(indexing, "get_page_texts_from_derivative", return_value=None)
    mocker.patch(
        "agentic.knowledge.embedder.litellm.LiteLLMEmbedder.aembed_batch",
        return_value=[[0.1] * 1536],
    )
    bm25 = mocker.patch(
        "agentic_project_service.services.sparse_retrieval."
        "sparse_index_store.SparseIndexStore.add_and_save"
    )
    charge = mocker.patch("agentic_project_service.services.billing_port.charge")

    with app.app_context():
        # Same call as the superseded test, but task_id MATCHES the row owner.
        indexing._run_index_body(
            knowledge_base_id=test_knowledge_base["id"], source_id=src_id,
            indexed_source_id=is_id, task_id="owner-B", provider_keys={},
            idempotency_action="indexing", idempotency_parts=[is_id],
        )
        row = db.session.execute(
            text("SELECT index_status FROM ai.indexed_sources WHERE id = :id"),
            {"id": is_id},
        ).fetchone()
        n = db.session.execute(
            text("SELECT COUNT(*) AS c FROM ai.chunks WHERE indexed_source_id = :id"),
            {"id": is_id},
        ).fetchone().c

    assert row.index_status == "indexed"   # owner committed
    assert n == 1                          # exactly one chunk set persisted
    bm25.assert_called_once()
    charge.assert_called_once()


def _run_reindex_as(app, kb, src_id, is_id, task_id, mocker):
    """Drive _run_index_body over a source that already has a chunk.

    Returns the patched SparseIndexStore.remove_and_save mock.
    """
    from agentic_project_service.tasks import indexing

    mocker.patch.object(indexing, "get_text_derivative_content", return_value="hello world")
    mocker.patch.object(indexing, "get_page_texts_from_derivative", return_value=None)
    mocker.patch(
        "agentic.knowledge.embedder.litellm.LiteLLMEmbedder.aembed_batch",
        return_value=[[0.1] * 1536],
    )
    mocker.patch(
        "agentic_project_service.services.sparse_retrieval."
        "sparse_index_store.SparseIndexStore.add_and_save"
    )
    remove = mocker.patch(
        "agentic_project_service.services.sparse_retrieval."
        "sparse_index_store.SparseIndexStore.remove_and_save"
    )
    with app.app_context():
        indexing._run_index_body(
            knowledge_base_id=kb, source_id=src_id,
            indexed_source_id=is_id, task_id=task_id, provider_keys={},
        )
    return remove


def test_superseded_execution_strips_nothing_from_the_sparse_index(
    app, test_knowledge_base, mocker
):
    """A superseded task must not remove the live run's sparse entries.

    The reindex cleanup deletes the previous run's chunk rows and once dropped
    their sparse entries in the same breath -- before the task had proved it
    owned the row. The row deletes roll back with the fence; a file write does
    not. So a loser stripped the winner's entries and then rolled its own
    writes back, leaving the index holding fewer entries than EITHER task
    intended. The removals now run only after the fenced commit.
    """
    src_id, is_id = _seed_indexing(app, test_knowledge_base["id"], owner_task_id="owner-B")
    _seed_stale_chunk(app, test_knowledge_base["id"], src_id, is_id)

    remove = _run_reindex_as(
        app, test_knowledge_base["id"], src_id, is_id, "loser-A", mocker
    )

    remove.assert_not_called()


def test_owner_execution_strips_the_superseded_sparse_entries(
    app, test_knowledge_base, mocker
):
    """The owner DOES remove them -- so the test above is not vacuous.

    Same fixture, same seeded chunk, same call; only task_id differs. If the
    BM25 gate were off, or the guard swallowed an empty id list, this arm would
    fail and expose the other one as proving nothing.
    """
    src_id, is_id = _seed_indexing(app, test_knowledge_base["id"], owner_task_id="owner-B")
    stale_chunk_id = _seed_stale_chunk(app, test_knowledge_base["id"], src_id, is_id)

    remove = _run_reindex_as(
        app, test_knowledge_base["id"], src_id, is_id, "owner-B", mocker
    )

    remove.assert_any_call([stale_chunk_id], item_table="chunks")
