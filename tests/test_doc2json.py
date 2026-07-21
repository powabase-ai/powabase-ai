"""Tests for doc2json indexing strategy: store, orchestration, and search."""

import asyncio
import uuid
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.services.doc2json_store import Doc2JSONStore
from agentic_project_service.services.knowledge_search import search_knowledge_base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 8
FAKE_EMBEDDING = [0.1] * EMBEDDING_DIM
FAKE_JSON = {"name": "Acme Corp", "revenue": 1_000_000}
FAKE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "revenue": {"type": "number"},
    },
}


@dataclass
class FakeDoc2JSONResult:
    """Stand-in for agentic.knowledge.indexing.doc2json.Doc2JSONResult."""

    extracted_json: dict
    combined_summary: str
    combined_summary_embedding: list[float]
    window_summaries: list[dict]
    json_schema: dict
    stats: dict = field(default_factory=dict)


def _make_store(kb_id: str) -> Doc2JSONStore:
    return Doc2JSONStore(db_session=db.session, knowledge_base_id=kb_id)


# ===================================================================
# 1. TestDoc2JSONStore — CRUD operations
# ===================================================================


class TestDoc2JSONStore:
    """Integration tests for Doc2JSONStore against real Postgres."""

    def test_store_and_retrieve_document(
        self, app, test_source, test_knowledge_base, test_indexed_source
    ):
        with app.app_context():
            store = _make_store(test_knowledge_base["id"])
            doc_id = store.store_doc2json_document(
                indexed_source_id=test_indexed_source["id"],
                source_id=test_source["id"],
                summary="Test summary about Acme Corp",
                summary_embedding=FAKE_EMBEDDING,
                extracted_json=FAKE_JSON,
                json_schema=FAKE_SCHEMA,
                extraction_model="gpt-4o-mini",
                embedding_model="text-embedding-3-small",
                summary_tokens=42,
                input_tokens=100,
                window_size=2000,
                window_overlap=200,
                window_count=3,
            )

            assert isinstance(doc_id, str)
            uuid.UUID(doc_id)  # validates it's a proper UUID

            details = store.get_document_details(doc_id)
            assert details is not None
            assert details["id"] == doc_id
            assert details["summary"] == "Test summary about Acme Corp"
            assert details["extracted_json"] == FAKE_JSON
            assert details["json_schema"] == FAKE_SCHEMA
            assert details["extraction_model"] == "gpt-4o-mini"
            assert details["summary_tokens"] == 42
            assert details["input_tokens"] == 100
            assert details["window_size"] == 2000
            assert details["window_overlap"] == 200
            assert details["window_count"] == 3
            assert details["source_id"] == test_source["id"]

    def test_store_empty_embedding_raises(
        self, app, test_source, test_knowledge_base, test_indexed_source
    ):
        with app.app_context():
            store = _make_store(test_knowledge_base["id"])
            with pytest.raises(ValueError, match="summary_embedding is empty"):
                store.store_doc2json_document(
                    indexed_source_id=test_indexed_source["id"],
                    source_id=test_source["id"],
                    summary="No embedding",
                    summary_embedding=[],
                    extracted_json=FAKE_JSON,
                    json_schema=FAKE_SCHEMA,
                )

    def test_get_document_details_not_found(self, app, test_knowledge_base):
        with app.app_context():
            store = _make_store(test_knowledge_base["id"])
            result = store.get_document_details(str(uuid.uuid4()))
            assert result is None

    def test_delete_by_indexed_source(
        self, app, test_source, test_knowledge_base, test_indexed_source
    ):
        with app.app_context():
            store = _make_store(test_knowledge_base["id"])
            doc_id = store.store_doc2json_document(
                indexed_source_id=test_indexed_source["id"],
                source_id=test_source["id"],
                summary="To be deleted",
                summary_embedding=FAKE_EMBEDDING,
                extracted_json=FAKE_JSON,
                json_schema=FAKE_SCHEMA,
            )

            deleted = store.delete_by_indexed_source(test_indexed_source["id"])
            assert deleted == 1
            assert store.get_document_details(doc_id) is None

    def test_delete_by_indexed_source_no_match(self, app, test_knowledge_base):
        with app.app_context():
            store = _make_store(test_knowledge_base["id"])
            deleted = store.delete_by_indexed_source(str(uuid.uuid4()))
            assert deleted == 0

    def test_build_retrieved_item(self, app, test_source, test_knowledge_base, test_indexed_source):
        with app.app_context():
            store = _make_store(test_knowledge_base["id"])
            doc_id = store.store_doc2json_document(
                indexed_source_id=test_indexed_source["id"],
                source_id=test_source["id"],
                summary="Build item test",
                summary_embedding=FAKE_EMBEDDING,
                extracted_json=FAKE_JSON,
                json_schema=FAKE_SCHEMA,
            )

            extracted, schema = store._get_doc_json_data(doc_id)
            assert extracted == FAKE_JSON
            assert schema == FAKE_SCHEMA


# ===================================================================
# 2. TestRunDoc2JSONIndexing — async orchestration
# ===================================================================


class TestRunDoc2JSONIndexing:
    """Tests for run_doc2json_indexing with mocked LLM algorithm."""

    def _make_fake_result(self):
        return FakeDoc2JSONResult(
            extracted_json=FAKE_JSON,
            combined_summary="Acme Corp annual report summary",
            combined_summary_embedding=FAKE_EMBEDDING,
            window_summaries=[{"window": 1, "summary": "Window 1 summary"}],
            json_schema=FAKE_SCHEMA,
            stats={
                "window_count": 2,
                "input_tokens": 500,
                "summary_tokens": 80,
            },
        )

    def test_run_doc2json_indexing_text_mode(
        self, app, test_source, test_knowledge_base, test_indexed_source
    ):
        from agentic_project_service.tasks.indexing import run_doc2json_indexing

        fake_result = self._make_fake_result()

        with app.app_context():
            with patch("agentic_project_service.tasks.indexing.Doc2JSONAlgorithm") as MockAlgo:
                instance = MockAlgo.return_value
                instance.aindex = AsyncMock(return_value=fake_result)

                with patch("agentic_project_service.tasks.indexing.ensure_embedding_index"):
                    stats = asyncio.get_event_loop().run_until_complete(
                        run_doc2json_indexing(
                            kb_id=test_knowledge_base["id"],
                            indexed_source_id=test_indexed_source["id"],
                            source_id=test_source["id"],
                            content="Acme Corp earned $1M in 2024.",
                            indexing_config={
                                "strategy": "doc2json",
                                "json_schema": FAKE_SCHEMA,
                            },
                        )
                    )

            assert stats["artifact_count"] == 1
            assert stats["strategy"] == "doc2json"
            assert stats["window_count"] == 2
            assert stats["input_tokens"] == 500
            assert stats["summary_tokens"] == 80
            assert stats["embedding_dim"] == EMBEDDING_DIM
            assert "doc_id" in stats

            # Verify actually stored in DB
            store = _make_store(test_knowledge_base["id"])
            details = store.get_document_details(stats["doc_id"])
            assert details is not None
            assert details["extracted_json"] == FAKE_JSON

    def test_run_doc2json_indexing_image_mode(
        self, app, test_source, test_knowledge_base, test_indexed_source
    ):
        from agentic_project_service.tasks.indexing import run_doc2json_indexing

        fake_result = self._make_fake_result()

        with app.app_context():
            with patch("agentic_project_service.tasks.indexing.Doc2JSONAlgorithm") as MockAlgo:
                instance = MockAlgo.return_value
                instance.aindex = AsyncMock(return_value=fake_result)

                with patch("agentic_project_service.tasks.indexing.ensure_embedding_index"):
                    stats = asyncio.get_event_loop().run_until_complete(
                        run_doc2json_indexing(
                            kb_id=test_knowledge_base["id"],
                            indexed_source_id=test_indexed_source["id"],
                            source_id=test_source["id"],
                            content="",
                            indexing_config={
                                "strategy": "doc2json",
                                "json_schema": FAKE_SCHEMA,
                                "use_images": True,
                                "pages_per_window": 3,
                            },
                            page_images=[
                                {"page": 1, "base64": "abc123"},
                                {"page": 2, "base64": "def456"},
                            ],
                        )
                    )

            # Verify the algorithm was called with image-mode config
            call_kwargs = instance.aindex.call_args
            config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
            if config is None:
                config = call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs.args[1]
            assert config.extra["use_images"] is True
            assert config.extra["pages_per_window"] == 3
            assert "page_images" in config.extra
            # window_size/window_overlap should NOT be in extra for image mode
            assert "window_size" not in config.extra
            assert "window_overlap" not in config.extra

    def test_run_doc2json_indexing_missing_schema_raises(
        self, app, test_source, test_knowledge_base, test_indexed_source
    ):
        from agentic_project_service.tasks.indexing import run_doc2json_indexing

        with app.app_context():
            with pytest.raises(ValueError, match="doc2json strategy requires a json_schema"):
                asyncio.get_event_loop().run_until_complete(
                    run_doc2json_indexing(
                        kb_id=test_knowledge_base["id"],
                        indexed_source_id=test_indexed_source["id"],
                        source_id=test_source["id"],
                        content="Some content",
                        indexing_config={"strategy": "doc2json"},
                    )
                )


# ===================================================================
# 3. TestDoc2JSONSearch — knowledge_search.py doc2json branch
# ===================================================================


class TestDoc2JSONSearch:
    """Tests for the doc2json branch in search_knowledge_base."""

    def test_search_doc2json_no_documents_raises(self, app, test_knowledge_base):
        """KB with doc2json strategy but no indexed documents should raise."""
        with app.app_context():
            with pytest.raises(ValueError, match="No documents indexed"):
                search_knowledge_base(
                    db_session=db.session,
                    knowledge_base_id=test_knowledge_base["id"],
                    query="test query",
                    indexing_config={"strategy": "doc2json"},
                )

    def test_search_doc2json_vector_search(
        self, app, test_source, test_knowledge_base, test_indexed_source
    ):
        """Store a doc2json doc, then search with vector_search."""
        with app.app_context():
            store = _make_store(test_knowledge_base["id"])
            store.store_doc2json_document(
                indexed_source_id=test_indexed_source["id"],
                source_id=test_source["id"],
                summary="Annual report for Acme Corporation showing strong growth",
                summary_embedding=FAKE_EMBEDDING,
                extracted_json=FAKE_JSON,
                json_schema=FAKE_SCHEMA,
                embedding_model="text-embedding-3-small",
            )

            # Mock litellm.embedding to return a query embedding
            with patch("litellm.embedding") as mock_embed:
                mock_embed.return_value.data = [type("Obj", (), {"embedding": FAKE_EMBEDDING})()]

                results = search_knowledge_base(
                    db_session=db.session,
                    knowledge_base_id=test_knowledge_base["id"],
                    query="Acme Corp revenue",
                    retrieval_method="vector_search",
                    indexing_config={
                        "strategy": "doc2json",
                        "embedding_model": "text-embedding-3-small",
                    },
                )

            assert len(results) >= 1
            result = results[0]
            assert result.meta["extracted_json"] == FAKE_JSON
            assert result.meta["json_schema"] == FAKE_SCHEMA
