"""FullDocumentStore - document-level retrieval for the full_document strategy.

Inherits vector, full-text, and hybrid search from BasePgVectorStore.
Searches operate on summary (embeddings via ai.embeddings table) but return full_text
in RetrievedItem.text.  Only document-specific storage methods live here.
"""

import json
import logging
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import AI_SCHEMA
from .base_vector_store import BasePgVectorStore
from .knowledge_store import RetrievedItem
from .storage import SOURCES_BUCKET, SupabaseStorage, get_derivative_storage_path

logger = logging.getLogger(__name__)


class FullDocumentStore(BasePgVectorStore):
    """PostgreSQL store for the full_document indexing strategy."""

    TABLE = "full_documents"
    TEXT_COL = "full_text_path"
    SEARCH_TEXT_COL = "summary"

    def __init__(
        self,
        db_session: Session,
        knowledge_base_id: str,
        schema: str = AI_SCHEMA,
        storage: SupabaseStorage | None = None,
    ):
        super().__init__(db_session, knowledge_base_id, schema)
        self.storage = storage

    # ------------------------------------------------------------------
    # Text resolution
    # ------------------------------------------------------------------

    def _resolve_text(self, raw_value: str) -> str:
        """Download full_text from storage path."""
        if not self.storage:
            raise RuntimeError(
                "FullDocumentStore requires a storage client to resolve text. "
                "Pass storage= to the constructor."
            )
        return self.storage.download_from_path(raw_value).decode("utf-8")

    def _resolve_results(self, items: list[RetrievedItem]) -> list[RetrievedItem]:
        """Resolve text from storage in parallel, preserving paths in meta."""
        if not items:
            return items
        # Save storage paths to meta before resolution
        for item in items:
            item.meta = {**(item.meta or {}), "full_text_path": item.text}
        # Download all files in parallel
        from concurrent.futures import ThreadPoolExecutor

        paths = [item.text for item in items]
        with ThreadPoolExecutor(max_workers=min(len(paths), 10)) as pool:
            texts = list(pool.map(self._resolve_text, paths))
        for item, resolved in zip(items, texts):
            item.text = resolved
        return items

    # ------------------------------------------------------------------
    # Storage methods
    # ------------------------------------------------------------------

    def store_full_document(
        self,
        indexed_source_id: str,
        source_id: str,
        summary: str,
        summary_embedding: list[float],
        full_text: str,
        summary_model: str | None = None,
        embedding_model: str | None = None,
        summary_tokens: int | None = None,
        full_text_tokens: int | None = None,
        meta: dict | None = None,
    ) -> str:
        """Insert a full_documents row and return its id."""
        if not summary_embedding:
            raise ValueError(
                "summary_embedding is empty — cannot store full_document without embedding"
            )
        if not self.storage:
            raise RuntimeError(
                "FullDocumentStore requires a storage client to store documents. "
                "Pass storage= to the constructor."
            )

        doc_id = str(uuid4())
        embedding_str = f"[{','.join(str(x) for x in summary_embedding)}]"

        # Upload full text to storage
        storage_path = get_derivative_storage_path(source_id, "full_documents", f"{doc_id}.txt")
        full_text_path = self.storage.upload(
            bucket_id=SOURCES_BUCKET,
            path=storage_path,
            file_data=full_text.encode("utf-8"),
            content_type="text/plain",
        )

        dims = len(summary_embedding)

        try:
            self.session.execute(
                text(f"""
                    INSERT INTO "{self.schema}".full_documents (
                        id, indexed_source_id, knowledge_base_id, source_id,
                        summary, full_text_path,
                        summary_model,
                        summary_tokens, full_text_tokens, meta
                    ) VALUES (
                        :id, :indexed_source_id, :kb_id, :source_id,
                        :summary, :full_text_path,
                        :summary_model,
                        :summary_tokens, :full_text_tokens, CAST(:meta AS jsonb)
                    )
                """),
                {
                    "id": doc_id,
                    "indexed_source_id": indexed_source_id,
                    "kb_id": self.kb_id,
                    "source_id": source_id,
                    "summary": summary,
                    "full_text_path": full_text_path,
                    "summary_model": summary_model,
                    "summary_tokens": summary_tokens,
                    "full_text_tokens": full_text_tokens,
                    "meta": json.dumps(meta or {}),
                },
            )

            self.session.execute(
                text(f"""
                    INSERT INTO "{self.schema}".embeddings (
                        item_id, item_table, indexed_source_id,
                        knowledge_base_id, source_id,
                        embedding_model, dims, embedding
                    ) VALUES (
                        :item_id, 'full_documents', :indexed_source_id,
                        :kb_id, :source_id,
                        :embedding_model, :dims, CAST(:embedding AS vector)
                    )
                """),
                {
                    "item_id": doc_id,
                    "indexed_source_id": indexed_source_id,
                    "kb_id": self.kb_id,
                    "source_id": source_id,
                    "embedding_model": embedding_model or "unknown",
                    "dims": dims,
                    "embedding": embedding_str,
                },
            )

            self._ensure_embedding_index(dims)
            self.session.commit()
            logger.info(
                f"Stored full_document {doc_id} for source {source_id} in KB {self.kb_id} "
                f"(summary={summary_tokens} tokens, full_text={full_text_tokens} tokens, "
                f"storage_path={full_text_path})"
            )
            return doc_id
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Failed to store full_document for source {source_id} in KB {self.kb_id}: {e}"
            )
            raise

    def delete_by_indexed_source(self, indexed_source_id: str) -> int:
        """Delete all full_documents rows for an indexed source (for re-indexing).

        When a storage client is available, also removes the corresponding
        storage blobs so they don't accumulate as orphans across re-indexes.
        """
        try:
            # Query storage paths before deleting rows
            storage_paths: list[str] = []
            if self.storage:
                paths_result = self.session.execute(
                    text(f"""
                        SELECT full_text_path FROM "{self.schema}".full_documents
                        WHERE indexed_source_id = :indexed_source_id
                    """),
                    {"indexed_source_id": indexed_source_id},
                )
                storage_paths = [row[0] for row in paths_result if row[0]]

            result = self.session.execute(
                text(f"""
                    DELETE FROM "{self.schema}".full_documents
                    WHERE indexed_source_id = :indexed_source_id
                """),
                {"indexed_source_id": indexed_source_id},
            )
            self.session.commit()

            # Clean up storage blobs after successful DB delete
            if self.storage and storage_paths:
                for path in storage_paths:
                    try:
                        parts = path.split("/", 1)
                        if len(parts) == 2:
                            self.storage.delete(parts[0], [parts[1]])
                    except Exception:
                        logger.warning(
                            f"Failed to delete storage object {path} during "
                            f"re-index cleanup for indexed_source {indexed_source_id}"
                        )

            logger.info(
                f"Deleted {result.rowcount} full_document(s) for indexed_source {indexed_source_id}"
            )
            return result.rowcount
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Failed to delete full_documents for indexed_source {indexed_source_id}: {e}"
            )
            raise
