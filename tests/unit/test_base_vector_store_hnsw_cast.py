"""Regression tests for BasePgVectorStore.vector_search HNSW index usage.

The partial HNSW index in ai_schema.sql is defined as:

    CREATE INDEX idx_ai_embeddings_hnsw_1536
        ON ai.embeddings USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
        WHERE dims = 1536;

For PostgreSQL to use this index, the query's ORDER BY expression must be
syntactically identical to the index expression: the column must be cast to
vector(N) with the same N as the index. Without the cast, the planner falls
back to sequential scan — on Judocu prod (1.2M embeddings) that turned a
sub-second query into an 85-second one.
"""

import asyncio
from unittest.mock import MagicMock

from agentic_project_service.services.base_vector_store import BasePgVectorStore


class _FakeStore(BasePgVectorStore):
    TABLE = "graph_index_nodes"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


def _capture_statements(embedding):
    """Run vector_search against a spy session, return ALL executed SQL strings
    in order (vector_search issues a session GUC before the search query)."""
    session = MagicMock()
    captured: list[str] = []

    def spy_execute(text_obj, params=None):
        sql = text_obj.text if hasattr(text_obj, "text") else str(text_obj)
        captured.append(sql)
        return iter([])

    session.execute = spy_execute
    store = _FakeStore(db_session=session, knowledge_base_id="kb-test")
    asyncio.run(store.vector_search(embedding=embedding, top_k=10))
    assert captured, "vector_search did not execute any SQL"
    return captured


def _capture_sql(embedding):
    """Return the main search query (the statement with the ORDER BY clause)."""
    stmts = _capture_statements(embedding)
    search = [s for s in stmts if "ORDER BY" in s]
    assert search, f"no search query executed; statements: {stmts}"
    return search[0]


def test_vector_search_enables_hnsw_iterative_scan():
    """A SET LOCAL hnsw.iterative_scan must be issued so the KB post-filter
    doesn't starve the global HNSW index. The index spans all KBs; without
    iterative scanning, pgvector emits only ~ef_search global candidates and
    the `knowledge_base_id` filter then leaves far fewer than top_k (often 0)."""
    stmts = _capture_statements([0.0] * 1536)
    set_stmts = [s for s in stmts if "hnsw.iterative_scan" in s]
    assert set_stmts, f"expected a SET LOCAL hnsw.iterative_scan; statements: {stmts}"
    normalized = "".join(set_stmts[0].split()).lower()
    assert "setlocalhnsw.iterative_scan" in normalized, set_stmts[0]
    assert "strict_order" in normalized, set_stmts[0]


def test_iterative_scan_set_before_search_query():
    """The GUC must be set before the search query runs (same transaction;
    SET LOCAL is transaction-scoped)."""
    stmts = _capture_statements([0.0] * 1536)
    idx_set = next(i for i, s in enumerate(stmts) if "hnsw.iterative_scan" in s)
    idx_search = next(i for i, s in enumerate(stmts) if "ORDER BY" in s)
    assert idx_set < idx_search, f"GUC set after search; statements: {stmts}"


def test_vector_search_orders_by_dim_casted_embedding_to_match_hnsw_index():
    """ORDER BY must cast embedding to vector(N) so the partial HNSW index is used."""
    sql = _capture_sql([0.0] * 1536)
    normalized = "".join(sql.split())
    assert "ORDERBY(e.embedding::vector(1536))" in normalized, (
        "ORDER BY must use (e.embedding::vector(N)) to match the partial HNSW "
        "index on ai.embeddings. Without the cast PostgreSQL cannot use the "
        f"index and falls back to seq scan.\nGenerated SQL:\n{sql}"
    )


def test_vector_search_uses_embedding_dim_not_hardcoded_1536():
    """The cast must use the embedding's actual dimension, not a hardcoded value."""
    sql = _capture_sql([0.0] * 768)
    normalized = "".join(sql.split())
    assert "(e.embedding::vector(768))" in normalized, (
        "Cast must reflect the embedding's actual dimension (768 here), so KBs "
        "using non-1536-dim models also benefit when matching partial indexes "
        f"exist.\nGenerated SQL:\n{sql}"
    )
