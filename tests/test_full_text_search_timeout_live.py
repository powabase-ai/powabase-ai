"""Database-backed checks for the keyword-search fallback timeout.

A spy session shows the SQL is shaped right; only Postgres shows the statement
is really cancelled, the caller's transaction survives, and the timeout does not
leak into later statements.

Needs a FRESH database. This is pre-existing and affects every test module that
uses the ``app`` fixture: on a database that fixture has already bootstrapped,
the boot migrations fail part-way (a CREATE POLICY ... TO service_role, then a
missing workflow_id column) and the run dies in setup. CI gets a new service
container each time and never sees it; locally, create a new database (or drop
and recreate the one you point DATABASE_URL at) before each run.
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


def test_cancellation_rolls_back_only_the_savepoint(app, test_source, test_knowledge_base):
    """The cancellation must not take the caller's transaction with it.

    `SELECT 1` afterwards cannot tell the two apart -- SQLAlchemy would autobegin
    a fresh transaction and answer 1 either way. An uncommitted row written
    before the savepoint can: a rollback to the savepoint leaves it in place,
    while a transaction-level rollback discards it and starts a new txid.
    """
    kb_id = test_knowledge_base["id"]
    with app.app_context():
        _insert_chunks(kb_id, test_source["id"], 20000)  # commits

        marker_id = db.session.execute(
            text("""
                INSERT INTO "ai".chunks (knowledge_base_id, source_id, text, chunk_index)
                VALUES (CAST(:kb AS uuid), CAST(:sid AS uuid),
                        'marker row, deliberately left uncommitted', -1)
                RETURNING id
            """),
            {"kb": kb_id, "sid": test_source["id"]},
        ).scalar()
        txid_before = db.session.execute(text("SELECT txid_current()")).scalar()

        store = PgVectorKnowledgeStore(db_session=db.session, knowledge_base_id=kb_id)
        with patch.object(bvs, "_bm25_fallback_timeout_ms", return_value=1):
            with pytest.raises(bvs.KeywordSearchTimeout):
                asyncio.run(store.full_text_search("weather", top_k=5))

        assert db.session.execute(text("SELECT txid_current()")).scalar() == txid_before
        assert (
            db.session.execute(
                text('SELECT COUNT(*) FROM "ai".chunks WHERE id = CAST(:id AS uuid)'),
                {"id": str(marker_id)},
            ).scalar()
            == 1
        )
        db.session.rollback()


def test_a_non_timeout_database_error_also_leaves_the_session_usable(
    app, test_source, test_knowledge_base
):
    """The savepoint has to protect the caller from any failure inside it.

    A missing relation is not 57014, so it never reaches the mapping branch and
    is re-raised as itself — what keeps the session alive is begin_nested's own
    rollback. Only Postgres can show that: against a spy session, every
    statement after the failure answers regardless.
    """
    kb_id = test_knowledge_base["id"]
    with app.app_context():
        marker_id = db.session.execute(
            text("""
                INSERT INTO "ai".chunks (knowledge_base_id, source_id, text, chunk_index)
                VALUES (CAST(:kb AS uuid), CAST(:sid AS uuid),
                        'marker row, deliberately left uncommitted', -2)
                RETURNING id
            """),
            {"kb": kb_id, "sid": test_source["id"]},
        ).scalar()
        txid_before = db.session.execute(text("SELECT txid_current()")).scalar()

        store = PgVectorKnowledgeStore(db_session=db.session, knowledge_base_id=kb_id)
        with pytest.raises(Exception) as exc_info:
            store._fetch_with_timeout(
                'SELECT 1 FROM "ai".no_such_table_here', {}, 5000, query="weather"
            )
        assert not isinstance(exc_info.value, bvs.KeywordSearchTimeout)

        assert db.session.execute(text("SELECT 1")).scalar() == 1
        assert db.session.execute(text("SELECT txid_current()")).scalar() == txid_before
        assert (
            db.session.execute(
                text('SELECT COUNT(*) FROM "ai".chunks WHERE id = CAST(:id AS uuid)'),
                {"id": str(marker_id)},
            ).scalar()
            == 1
        )
        db.session.rollback()


def test_two_term_query_on_thousands_of_rows_fits_the_default_budget(
    app, test_source, test_knowledge_base
):
    """The fallback must not be quadratic in the number of matching rows.

    With corpus_stats inlined, a two-term query over 5,000 matching chunks ran
    for about 39 s and tripped the default 10 s budget on every search; with it
    materialized it takes about 1 s. No wall-clock number is asserted -- the
    property is that the real statement, under the real default budget,
    returns results instead of timing out.
    """
    kb_id = test_knowledge_base["id"]
    with app.app_context():
        _insert_chunks(kb_id, test_source["id"], 5000)
        db.session.execute(text('ANALYZE "ai".chunks'))
        db.session.commit()
        assert bvs._bm25_fallback_timeout_ms() == 10000

        store = PgVectorKnowledgeStore(db_session=db.session, knowledge_base_id=kb_id)
        results = asyncio.run(store.full_text_search("weather weather", top_k=5))

        assert len(results) == 5


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
