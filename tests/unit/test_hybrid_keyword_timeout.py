"""A keyword-leg timeout leaves hybrid search with its vector results."""

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest
from agentic.knowledge.models import RetrievedItem

from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services import knowledge_search as ks


class _Store(bvs.BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


def _item(i: str) -> RetrievedItem:
    return RetrievedItem(
        item_id=i, text=f"t{i}", score=0.9, source_id="s", knowledge_base_id="kb", meta={}
    )


def _store_with_timeout():
    store = _Store(db_session=MagicMock(), knowledge_base_id="kb")

    async def vec(*a, **k):
        return [_item("v1"), _item("v2")]

    async def kw(*a, **k):
        raise bvs.KeywordSearchTimeout("kb", 10000)

    store.vector_search = vec
    store.bm25s_search = kw
    store.full_text_search = kw
    return store


def _fake_embedding():
    resp = MagicMock()
    resp.data = [{"embedding": [0.1, 0.2, 0.3]}]
    return patch("litellm.embedding", return_value=resp)


def test_keyword_search_for_hybrid_swallows_timeout():
    store = _store_with_timeout()
    out = asyncio.run(
        store.keyword_search_for_hybrid(
            "q", top_k=4, filter_metadata=None, item_ids=None, source_ids=None
        )
    )
    assert out == []


def test_keyword_search_for_hybrid_propagates_other_errors():
    """Only the designed timeout may be swallowed.

    Widening the except to Exception would turn any keyword-leg bug -- a bad
    ts_language, a broken connection, a programming error -- into a silently
    vector-only answer that nobody can tell apart from a healthy one.
    """
    store = _store_with_timeout()

    async def boom(*a, **k):
        raise RuntimeError("keyword leg is broken")

    store.bm25s_search = boom
    store.full_text_search = boom

    with pytest.raises(RuntimeError, match="keyword leg is broken"):
        asyncio.run(
            store.keyword_search_for_hybrid(
                "q", top_k=4, filter_metadata=None, item_ids=None, source_ids=None
            )
        )


def test_keyword_search_for_hybrid_records_the_degradation_on_the_request():
    """The dropped leg is recorded where the route can read it back."""
    from flask import Flask

    store = _store_with_timeout()
    app = Flask(__name__)
    with app.test_request_context("/"):
        asyncio.run(
            store.keyword_search_for_hybrid(
                "q", top_k=4, filter_metadata=None, item_ids=None, source_ids=None
            )
        )
        assert bvs.get_retrieval_degradations() == ["keyword_search_timeout"]


def test_keyword_search_for_hybrid_records_each_reason_once():
    from flask import Flask

    store = _store_with_timeout()
    app = Flask(__name__)
    with app.test_request_context("/"):
        for _ in range(3):
            asyncio.run(
                store.keyword_search_for_hybrid(
                    "q", top_k=4, filter_metadata=None, item_ids=None, source_ids=None
                )
            )
        assert bvs.get_retrieval_degradations() == ["keyword_search_timeout"]


def test_keyword_search_for_hybrid_does_not_re_warn(caplog):
    """_fetch_with_timeout already warned; a second WARNING double-counts.

    The hybrid leg keeps a debug line so the vector-only answer is traceable
    without paging twice for one event.
    """
    store = _store_with_timeout()
    with caplog.at_level(logging.DEBUG, logger=bvs.logger.name):
        asyncio.run(
            store.keyword_search_for_hybrid(
                "q", top_k=4, filter_metadata=None, item_ids=None, source_ids=None
            )
        )

    assert [r.levelname for r in caplog.records if r.levelname in ("WARNING", "ERROR")] == []
    assert any("vector results only" in r.getMessage() for r in caplog.records)


def test_sync_hybrid_returns_vector_results_on_keyword_timeout():
    store = _store_with_timeout()
    with _fake_embedding():
        out = ks._run_hybrid_search(store, "q", 4, {}, {}, None)
    assert [r.item_id for r in out] == ["v1", "v2"]


def test_async_hybrid_returns_vector_results_on_keyword_timeout():
    store = _store_with_timeout()
    with _fake_embedding():
        out = asyncio.run(ks._arun_hybrid_search(store, "q", 4, {}, {}, None))
    assert [r.item_id for r in out] == ["v1", "v2"]


def test_store_hybrid_search_returns_vector_results_on_keyword_timeout():
    store = _store_with_timeout()
    out = asyncio.run(store.hybrid_search("q", [0.1, 0.2, 0.3], top_k=4))
    assert [r.item_id for r in out] == ["v1", "v2"]


def test_full_text_retriever_still_raises():
    store = _store_with_timeout()
    with pytest.raises(bvs.KeywordSearchTimeout):
        ks._run_full_text_search(store, "q", 4, {}, {}, None)
