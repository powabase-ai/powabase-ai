"""PgVectorKnowledgeStore - chunk-level retrieval for knowledge bases.

Inherits vector, full-text, and hybrid search from BasePgVectorStore.
Only chunk-specific storage methods live here.
"""

import json
import logging
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import AI_SCHEMA
from agentic.knowledge.models import RetrievedItem  # noqa: F401 – re-export
from .base_vector_store import VALID_TS_LANGUAGES, BasePgVectorStore  # noqa: F401 – re-export

logger = logging.getLogger(__name__)


class PgVectorKnowledgeStore(BasePgVectorStore):
    """PostgreSQL/pgvector store for chunked knowledge bases."""

    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"

    def __init__(
        self,
        db_session: Session,
        knowledge_base_id: str,
        schema: str = AI_SCHEMA,
    ):
        super().__init__(db_session, knowledge_base_id, schema)

    async def store_chunks(
        self,
        indexed_source_id: str,
        chunks: list[dict],
        embedding_model: str | None = None,
    ) -> tuple[int, list[str]]:
        """Store chunks and their embeddings (in ai.embeddings table)."""
        if not chunks:
            return 0, []

        inserted = 0
        chunk_ids: list[str] = []
        dims_seen: int | None = None
        for chunk in chunks:
            chunk_id = str(uuid4())
            chunk_ids.append(chunk_id)
            embedding = chunk.get("embedding")

            self.session.execute(
                text(f"""
                    INSERT INTO "{self.schema}".chunks (
                        id, indexed_source_id, knowledge_base_id, source_id,
                        text, chunk_index, start_char, end_char, tokens, meta
                    ) VALUES (
                        :id, :indexed_source_id, :kb_id, :source_id,
                        :text, :chunk_index,
                        :start_char, :end_char, :tokens, CAST(:meta AS jsonb)
                    )
                """),
                {
                    "id": chunk_id,
                    "indexed_source_id": indexed_source_id,
                    "kb_id": self.kb_id,
                    "source_id": chunk.get("source_id"),
                    "text": chunk["text"],
                    "chunk_index": chunk.get("chunk_index"),
                    "start_char": chunk.get("start_char"),
                    "end_char": chunk.get("end_char"),
                    "tokens": chunk.get("tokens"),
                    "meta": json.dumps(chunk.get("meta", {})),
                },
            )

            if embedding and embedding_model:
                embedding_str = f"[{','.join(str(x) for x in embedding)}]"
                dims = len(embedding)
                dims_seen = dims
                self.session.execute(
                    text(f"""
                        INSERT INTO "{self.schema}".embeddings (
                            item_id, item_table, indexed_source_id,
                            knowledge_base_id, source_id,
                            embedding_model, dims, embedding
                        ) VALUES (
                            :item_id, 'chunks', :indexed_source_id,
                            :kb_id, :source_id,
                            :embedding_model, :dims, CAST(:embedding AS vector)
                        )
                    """),
                    {
                        "item_id": chunk_id,
                        "indexed_source_id": indexed_source_id,
                        "kb_id": self.kb_id,
                        "source_id": chunk.get("source_id"),
                        "embedding_model": embedding_model,
                        "dims": dims,
                        "embedding": embedding_str,
                    },
                )

            inserted += 1

        if dims_seen is not None:
            self._ensure_embedding_index(dims_seen)

        self.session.commit()
        return inserted, chunk_ids

    async def delete_chunks(self, indexed_source_id: str) -> int:
        """Delete all chunks for an indexed source."""
        result = self.session.execute(
            text(f"""
                DELETE FROM "{self.schema}".chunks
                WHERE indexed_source_id = :indexed_source_id
            """),
            {"indexed_source_id": indexed_source_id},
        )
        self.session.commit()
        return result.rowcount

    async def count_chunks(self, indexed_source_id: str | None = None) -> int:
        """Count chunks, optionally filtered by indexed source."""
        query = f'SELECT COUNT(*) FROM "{self.schema}".chunks WHERE knowledge_base_id = :kb_id'
        params = {"kb_id": self.kb_id}

        if indexed_source_id:
            query += " AND indexed_source_id = :indexed_source_id"
            params["indexed_source_id"] = indexed_source_id

        result = self.session.execute(text(query), params)
        return result.scalar() or 0
