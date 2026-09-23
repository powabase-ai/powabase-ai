"""A knowledge base cannot be configured into returning nothing, silently.

The agent retrieval paths take ``top_k`` from a knowledge base's stored config
and the reranker's ``candidate_count`` from its stored retrieval config, and
neither is range-checked where it is written. The vector store now refuses a row
limit above ``MAX_TOP_K`` -- correctly, because it interpolates it into the SQL
-- and every one of these call sites wraps the search in ``except Exception`` and
returns an empty result list. So a knowledge base with ``candidate_count: 20000``
worked one day and retrieved nothing at all the next, leaving one warning naming
a parameter whoever made the request never sent.

Clamped, loudly, rather than validated at the write boundary: the rows are
already written, and the caller of these paths cannot see or influence the value.
The REST search route keeps its 400, where the value really did come from the
caller.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from agentic_project_service.services import context_handler as ch
from agentic_project_service.services.base_vector_store import MAX_TOP_K, validated_top_k
from agentic_project_service.services.knowledge_search import RERANKER_CANDIDATE_COUNT
from agentic_project_service.services.knowledge_store import RetrievedItem

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


def _item() -> RetrievedItem:
    return RetrievedItem(
        item_id="c1", text="a hit", score=0.9, source_id=None, knowledge_base_id=KB, meta={}
    )


def _store_guard(top_k, method: str | None) -> None:
    """The row-limit guard the vector store really applies, where it applies it.

    Hybrid search asks each leg for twice the caller's limit before the store
    checks it, so the same stored number is twice as likely to be refused for a
    knowledge base configured for hybrid retrieval.
    """
    validated_top_k(top_k * 2 if method == "hybrid" else top_k)


def _search_spy(monkeypatch, *, async_path: bool = False):
    """Stand in for ``search_knowledge_base``, applying the store's real guards.

    Returns the list of call kwargs. Raising the same ValueError the store would
    is what makes these specs fail when the clamp is removed, instead of merely
    recording a large number.
    """
    calls: list[dict] = []
    # These paths read a handful of settings; outside a request context the
    # registry read logs a warning of its own, which these specs would then have
    # to filter out of caplog.
    defaults = {
        "KB_DEFAULT_TOP_K": 5,
        "KB_DEFAULT_MAX_CONTEXT_TOKENS": 4000,
        "MAX_SEARCH_WORKERS": 4,
    }
    monkeypatch.setattr(ch, "get_setting", lambda key: defaults.get(key, 5))

    def fake(**kwargs):
        calls.append(kwargs)
        method = kwargs.get("retrieval_method")
        _store_guard(kwargs["top_k"], method)
        reranker = (kwargs.get("retrieval_config") or {}).get("reranker") or {}
        if reranker.get("model"):
            # knowledge_search hands the store the candidate pool, not top_k.
            _store_guard(reranker.get("candidate_count", RERANKER_CANDIDATE_COUNT), method)
        return [_item()]

    if async_path:

        async def afake(**kwargs):
            return fake(**kwargs)

        monkeypatch.setattr(ch, "search_knowledge_base_async", afake)
    else:
        monkeypatch.setattr(ch, "search_knowledge_base", fake)
    return calls


def _single(monkeypatch, kb_config: dict, retrieval_config: dict | None = None):
    calls = _search_spy(monkeypatch)
    outcome = ch._search_single_kb(
        engine=MagicMock(),
        kb_config={"id": KB, **kb_config},
        query="weather",
        kb_retrieval_configs={KB: retrieval_config} if retrieval_config is not None else {},
        session_history=None,
    )
    return outcome, calls


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------


def test_a_stored_top_k_the_store_would_refuse_still_returns_results(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        outcome, calls = _single(monkeypatch, {"top_k": MAX_TOP_K * 2})
    assert outcome["error"] is None, outcome["error"]
    assert len(outcome["results"]) == 1
    assert calls[0]["top_k"] == ch.SAFE_RETRIEVAL_TOP_K
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert KB in message and str(MAX_TOP_K * 2) in message, message
    assert "top_k" in message, message


def test_the_clamp_allows_for_hybrid_doubling(monkeypatch):
    """A hybrid knowledge base asks for twice the limit, so the ceiling is halved."""
    outcome, calls = _single(monkeypatch, {"top_k": MAX_TOP_K, "retrieval_method": "hybrid"})
    assert outcome["error"] is None, outcome["error"]
    assert calls[0]["top_k"] * 2 <= MAX_TOP_K


def test_a_stored_top_k_within_the_ceiling_is_passed_through_untouched(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        _, calls = _single(monkeypatch, {"top_k": 7})
    assert calls[0]["top_k"] == 7
    assert caplog.records == []


def test_a_stored_top_k_that_is_not_a_number_falls_back_to_the_default(monkeypatch, caplog):
    """Nothing to clamp, and the store's error would name a caller's parameter."""
    with caplog.at_level(logging.WARNING):
        outcome, calls = _single(monkeypatch, {"top_k": "as many as possible"})
    assert outcome["error"] is None, outcome["error"]
    assert isinstance(calls[0]["top_k"], int)
    assert "as many as possible" in "\n".join(r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# reranker candidate_count
# ---------------------------------------------------------------------------


def test_a_candidate_count_the_store_would_refuse_still_returns_results(monkeypatch, caplog):
    stored = {"reranker": {"model": "cohere/rerank-english-v3.0", "candidate_count": 20000}}
    with caplog.at_level(logging.WARNING):
        outcome, calls = _single(monkeypatch, {}, retrieval_config=stored)
    assert outcome["error"] is None, outcome["error"]
    assert len(outcome["results"]) == 1
    passed = calls[0]["retrieval_config"]["reranker"]["candidate_count"]
    assert passed == ch.SAFE_RETRIEVAL_TOP_K
    # The stored dict is not mutated: it is also what the response metadata and
    # the enrichment hoist read.
    assert stored["reranker"]["candidate_count"] == 20000
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert "candidate_count" in message and KB in message, message


def test_an_unusable_candidate_count_falls_back_to_the_services_own_default(monkeypatch):
    stored = {"reranker": {"model": "cohere/rerank-english-v3.0", "candidate_count": "lots"}}
    outcome, calls = _single(monkeypatch, {}, retrieval_config=stored)
    assert outcome["error"] is None, outcome["error"]
    assert "candidate_count" not in calls[0]["retrieval_config"]["reranker"]


def test_a_sane_configuration_is_left_entirely_alone(monkeypatch):
    """No substitute config, so the search reads the stored row as it always did."""
    stored = {"reranker": {"model": "cohere/rerank-english-v3.0", "candidate_count": 100}}
    _, calls = _single(monkeypatch, {}, retrieval_config=stored)
    assert calls[0].get("retrieval_config") is None

    _, calls = _single(monkeypatch, {}, retrieval_config={"method": "hybrid"})
    assert calls[0].get("retrieval_config") is None


# ---------------------------------------------------------------------------
# What is left over is reported as a configuration problem
# ---------------------------------------------------------------------------


def test_a_search_that_rejects_a_stored_value_is_logged_as_a_configuration_error(
    monkeypatch, caplog
):
    """Not every bad stored value can be clamped; none may look like a caller's fault."""

    def boom(**kwargs):
        raise ValueError("top_k must be between 0 and 10000, got 999999")

    _search_spy(monkeypatch)
    monkeypatch.setattr(ch, "search_knowledge_base", boom)
    with caplog.at_level(logging.WARNING):
        outcome = ch._search_single_kb(
            engine=MagicMock(),
            kb_config={"id": KB},
            query="weather",
            kb_retrieval_configs={},
            session_history=None,
        )
    assert outcome["results"] == []
    assert isinstance(outcome["error"], ValueError)
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "a knowledge base that can answer nothing at all is not a warning"
    message = errors[0].getMessage()
    assert KB in message and "configuration" in message.lower(), message


# ---------------------------------------------------------------------------
# The other two retrieval paths
# ---------------------------------------------------------------------------


def _execute_retrieval_rows(retrieval_config: dict) -> MagicMock:
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = [(KB, "a kb", {}, retrieval_config)]
    session.execute.return_value.fetchone.return_value = None
    return session


def test_the_single_knowledge_base_fast_path_clamps_too(monkeypatch):
    """It does not go through ``_search_single_kb``: it searches on the caller's session."""
    calls = _search_spy(monkeypatch)
    out = ch.execute_retrieval(
        db_session=_execute_retrieval_rows({}),
        query="weather",
        knowledge_base_configs=[{"id": KB, "top_k": MAX_TOP_K * 2}],
    )
    assert out["errors"] == [], out["errors"]
    assert calls[0]["top_k"] == ch.SAFE_RETRIEVAL_TOP_K


def test_the_async_path_clamps_too(monkeypatch):
    import asyncio

    calls = _search_spy(monkeypatch, async_path=True)
    out = asyncio.run(
        ch.execute_retrieval_async(
            db_session=_execute_retrieval_rows({}),
            query="weather",
            knowledge_base_configs=[{"id": KB, "top_k": MAX_TOP_K * 2}],
        )
    )
    assert out["errors"] == [], out["errors"]
    assert calls[0]["top_k"] == ch.SAFE_RETRIEVAL_TOP_K


def test_the_ceiling_is_derived_from_the_stores_own_limit():
    """Not a second number to keep in step."""
    assert ch.SAFE_RETRIEVAL_TOP_K == MAX_TOP_K // 2


@pytest.mark.parametrize("path", ["sync", "async"])
def test_the_candidate_pool_is_clamped_on_every_path(monkeypatch, path):
    stored = {"reranker": {"model": "cohere/rerank-english-v3.0", "candidate_count": 20000}}
    if path == "sync":
        calls = _search_spy(monkeypatch)
        out = ch.execute_retrieval(
            db_session=_execute_retrieval_rows(stored),
            query="weather",
            knowledge_base_configs=[{"id": KB}],
        )
    else:
        import asyncio

        calls = _search_spy(monkeypatch, async_path=True)
        out = asyncio.run(
            ch.execute_retrieval_async(
                db_session=_execute_retrieval_rows(stored),
                query="weather",
                knowledge_base_configs=[{"id": KB}],
            )
        )
    assert out["errors"] == [], out["errors"]
    assert calls[0]["retrieval_config"]["reranker"]["candidate_count"] == ch.SAFE_RETRIEVAL_TOP_K
