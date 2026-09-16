"""Database-backed checks for the keyword-search fallback timeout.

A spy session shows the SQL is shaped right; only Postgres shows the statement
is really cancelled, the caller's transaction survives, and the timeout does not
leak into later statements.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services.knowledge_store import PgVectorKnowledgeStore


def _insert_chunks(kb_id: str, source_id: str, n: int) -> None:
    db.session.execute(
        text("""
            INSERT INTO "ai".chunks (knowledge_base_id, source_id, text, chunk_index)
            SELECT CAST(:kb AS uuid), CAST(:sid AS uuid),
                   'weather note for the hiking trip number ' || g
                   || repeat(' filler words for the parser', 40),
                   g
            FROM generate_series(1, :n) AS g
        """),
        {"kb": kb_id, "sid": source_id, "n": n},
    )
    db.session.commit()


def _statement_timeout() -> str:
    return db.session.execute(text("SELECT current_setting('statement_timeout')")).scalar()


def test_fallback_is_cancelled_and_session_survives(app, test_source, test_knowledge_base):
    kb_id = test_knowledge_base["id"]
    with app.app_context():
        _insert_chunks(kb_id, test_source["id"], 20000)
        before = _statement_timeout()
        store = PgVectorKnowledgeStore(db_session=db.session, knowledge_base_id=kb_id)

        with patch.object(bvs, "_bm25_fallback_timeout_ms", return_value=1):
            with pytest.raises(bvs.KeywordSearchTimeout):
                asyncio.run(store.full_text_search("weather", top_k=5))

        assert db.session.execute(text("SELECT 1")).scalar() == 1
        assert _statement_timeout() == before


def test_fallback_within_budget_returns_results_and_restores_timeout(
    app, test_source, test_knowledge_base
):
    kb_id = test_knowledge_base["id"]
    with app.app_context():
        _insert_chunks(kb_id, test_source["id"], 20)
        before = _statement_timeout()
        store = PgVectorKnowledgeStore(db_session=db.session, knowledge_base_id=kb_id)

        with patch.object(bvs, "_bm25_fallback_timeout_ms", return_value=60000):
            results = asyncio.run(store.full_text_search("weather", top_k=5))

        assert len(results) == 5
        assert _statement_timeout() == before
