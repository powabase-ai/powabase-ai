"""Doc2JSONStore - document-level retrieval for the doc2json strategy.

Inherits vector, full-text, and hybrid search from BasePgVectorStore.
Searches operate on summary (embeddings via ai.embeddings table) but return
extracted_json in RetrievedItem.meta['extracted_json'].
"""

import json
import logging
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import AI_SCHEMA
from .base_vector_store import BasePgVectorStore
from .knowledge_store import RetrievedItem

logger = logging.getLogger(__name__)


class Doc2JSONStore(BasePgVectorStore):
    """PostgreSQL store for the doc2json indexing strategy."""

    TABLE = "doc2json_documents"
    TEXT_COL = "summary"
    SEARCH_TEXT_COL = "summary"

    def __init__(
        self,
        db_session: Session,
        knowledge_base_id: str,
        schema: str = AI_SCHEMA,
    ):
        super().__init__(db_session, knowledge_base_id, schema)

    # ------------------------------------------------------------------
    # Override result building to include extracted_json
    # ------------------------------------------------------------------

    def _build_retrieved_item(self, row, score: float = 0.0) -> RetrievedItem:
        """Build RetrievedItem with extracted_json in meta."""
        # Row structure from vector_search/full_text_search:
        # (id, text_col, similarity/score, source_id, meta)
        item_id = str(row[0])
        text_val = row[1]
        source_id = str(row[3]) if row[3] else None
        meta = row[4] or {}

        # Fetch extracted_json and json_schema in a single query (BUG 4 fix: avoid N+1)
        extracted_json, json_schema = self._get_doc_json_data(item_id)

        return RetrievedItem(
            item_id=item_id,
            text=text_val,
            score=score,
            source_id=source_id,
            knowledge_base_id=self.kb_id,
            meta={
                **meta,
                "extracted_json": extracted_json,
                "json_schema": json_schema,
            },
        )

    def _get_doc_json_data(self, doc_id: str) -> tuple[dict, dict]:
        """Fetch extracted_json and json_schema for a document in a single query."""
        try:
            result = self.session.execute(
                text(f"""
                    SELECT extracted_json, json_schema FROM "{self.schema}".{self.TABLE}
                    WHERE id = :doc_id
                """),
                {"doc_id": doc_id},
            )
            row = result.fetchone()
            if row:
                return (row[0] or {}, row[1] or {})
            return ({}, {})
        except Exception as e:
            logger.warning(f"Failed to fetch doc json data for {doc_id}: {e}")
            return ({}, {})

    # ------------------------------------------------------------------
    # Storage methods
    # ------------------------------------------------------------------

    def store_doc2json_document(
        self,
        indexed_source_id: str,
        source_id: str,
        summary: str,
        summary_embedding: list[float],
        extracted_json: dict,
        json_schema: dict,
        window_summaries: list[dict] | None = None,
        extraction_model: str | None = None,
        embedding_model: str | None = None,
        summary_tokens: int | None = None,
        input_tokens: int | None = None,
        window_size: int | None = None,
        window_overlap: int | None = None,
        window_count: int | None = None,
        meta: dict | None = None,
    ) -> str:
        """Insert a doc2json_documents row and return its id."""
        if not summary_embedding:
            raise ValueError(
                "summary_embedding is empty — cannot store doc2json_document without embedding"
            )

        doc_id = str(uuid4())
        embedding_str = f"[{','.join(str(x) for x in summary_embedding)}]"
        dims = len(summary_embedding)

        try:
            self.session.execute(
                text(f"""
                    INSERT INTO "{self.schema}".doc2json_documents (
                        id, indexed_source_id, knowledge_base_id, source_id,
                        summary, extracted_json, json_schema, window_summaries,
                        extraction_model, summary_tokens, input_tokens,
                        window_size, window_overlap, window_count, meta
                    ) VALUES (
                        :id, :indexed_source_id, :kb_id, :source_id,
                        :summary, CAST(:extracted_json AS jsonb), CAST(:json_schema AS jsonb),
                        CAST(:window_summaries AS jsonb),
                        :extraction_model, :summary_tokens, :input_tokens,
                        :window_size, :window_overlap, :window_count, CAST(:meta AS jsonb)
                    )
                """),
                {
                    "id": doc_id,
                    "indexed_source_id": indexed_source_id,
                    "kb_id": self.kb_id,
                    "source_id": source_id,
                    "summary": summary,
                    "extracted_json": json.dumps(extracted_json),
                    "json_schema": json.dumps(json_schema),
                    "window_summaries": json.dumps(window_summaries or []),
                    "extraction_model": extraction_model,
                    "summary_tokens": summary_tokens,
                    "input_tokens": input_tokens,
                    "window_size": window_size,
                    "window_overlap": window_overlap,
                    "window_count": window_count,
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
                        :item_id, 'doc2json_documents', :indexed_source_id,
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
                f"Stored doc2json_document {doc_id} for source {source_id} in KB {self.kb_id} "
                f"(windows={window_count}, input_tokens={input_tokens})"
            )
            return doc_id
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Failed to store doc2json_document for source {source_id} in KB {self.kb_id}: {e}"
            )
            raise

    def delete_by_indexed_source(self, indexed_source_id: str) -> int:
        """Delete all doc2json_documents rows for an indexed source (for re-indexing)."""
        try:
            result = self.session.execute(
                text(f"""
                    DELETE FROM "{self.schema}".doc2json_documents
                    WHERE indexed_source_id = :indexed_source_id
                """),
                {"indexed_source_id": indexed_source_id},
            )
            self.session.commit()
            logger.info(
                f"Deleted {result.rowcount} doc2json_document(s) for indexed_source {indexed_source_id}"
            )
            return result.rowcount
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Failed to delete doc2json_documents for indexed_source {indexed_source_id}: {e}"
            )
            raise

    def get_document_details(self, doc_id: str) -> dict | None:
        """Get full document details including extracted JSON and schema."""
        try:
            result = self.session.execute(
                text(f"""
                    SELECT id, summary, extracted_json, json_schema,
                           window_summaries, extraction_model,
                           summary_tokens, input_tokens,
                           window_size, window_overlap, window_count,
                           source_id, created_at
                    FROM "{self.schema}".doc2json_documents
                    WHERE id = :doc_id
                """),
                {"doc_id": doc_id},
            )
            row = result.fetchone()
            if not row:
                return None

            return {
                "id": str(row[0]),
                "summary": row[1],
                "extracted_json": row[2],
                "json_schema": row[3],
                "window_summaries": row[4],
                "extraction_model": row[5],
                "summary_tokens": row[6],
                "input_tokens": row[7],
                "window_size": row[8],
                "window_overlap": row[9],
                "window_count": row[10],
                "source_id": str(row[11]) if row[11] else None,
                "created_at": row[12].isoformat() if row[12] else None,
            }
        except Exception as e:
            logger.error(f"Failed to get doc2json_document details for {doc_id}: {e}")
            return None
