"""Tests for knowledge_search billing wiring.

Focused on the retrieval pipeline because that's where every chargeable
sub-action (vector_search, bm25_search, hybrid_search, reranker_call,
metadata_enrichment, tree_search) is mapped to a billing call. Tests
inject the retriever via the CHUNK_RETRIEVER_MAP (and the async map) so
they exercise ONLY the wiring, not the underlying retrieval internals.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from werkzeug.exceptions import HTTPException

from agentic_project_service.services import billing_port, knowledge_search
from agentic_project_service.services.knowledge_store import RetrievedItem
from tests.support.billing import RecordingBillingAdapter


@pytest.fixture
def billing_env(monkeypatch):
    monkeypatch.setenv("BILLING_ORG_ID", "org-1")
    monkeypatch.setenv("PROJECT_ID", "proj-1")
    monkeypatch.delenv("BILLING_PLAN_TIER", raising=False)
    yield


def _fake_results(n=2):
    return [
        RetrievedItem(
            item_id=f"item-{i}",
            text=f"text {i}",
            score=0.9 - 0.1 * i,
            source_id=f"src-{i}",
            knowledge_base_id="kb-1",
            meta={},
        )
        for i in range(n)
    ]


def _stub_retriever(results=None):
    """Build a sync stub retriever matching the signature in CHUNK_RETRIEVER_MAP."""
    results = results if results is not None else _fake_results()

    def stub(**kwargs):
        return results

    return stub


def _stub_async_retriever(results=None):
    """Build an async stub for ASYNC_CHUNK_RETRIEVER_MAP."""
    results = results if results is not None else _fake_results()

    async def stub(**kwargs):
        return results

    return stub


def test_execute_retrieval_pipeline_charges_vector_search(
    billing_env, monkeypatch, recording_billing
):
    """vector_search retrieval → check_balance + charge called once."""
    monkeypatch.setitem(knowledge_search.CHUNK_RETRIEVER_MAP, "vector_search", _stub_retriever())
    with patch.object(knowledge_search, "_attach_enrichment_metadata", return_value=False):
        knowledge_search._execute_retrieval_pipeline(
            db_session=MagicMock(),
            store=MagicMock(),
            query="q",
            top_k=5,
            retrieval_method="vector_search",
            indexing_config={},
            retrieval_config={},
            filter_metadata=None,
            knowledge_base_id="kb-1",
            request_id="req-vec",
        )

    # plan_tier/org_id/project_id are no longer visible at this layer — the
    # port's check_balance() takes only estimated_cost; how the cloud adapter
    # maps that onto org/project/plan_tier is pinned by
    # test_billing_cloud_adapter.py::test_check_balance_delegates_with_ctx.
    assert recording_billing.balance_checks == [1]

    assert len(recording_billing.charges) == 1
    charge = recording_billing.charges[0]
    assert charge["action"] == "vector_search"
    assert charge["quantity"] == 1
    assert charge["ref_type"] == "retrieval"
    assert charge["ref_id"] == "req-vec"
    # idempotency_parts is the (request_id,) tail the cloud adapter combines
    # with org_id to build the real key (see T3-addendum in transform-T.md).
    assert charge["idempotency_parts"] == ("req-vec",)


def test_execute_retrieval_pipeline_charges_bm25_for_full_text(
    billing_env, monkeypatch, recording_billing
):
    """full_text retrieval bills as bm25_search (action name from spec)."""
    monkeypatch.setitem(knowledge_search.CHUNK_RETRIEVER_MAP, "full_text", _stub_retriever())
    with patch.object(knowledge_search, "_attach_enrichment_metadata", return_value=False):
        knowledge_search._execute_retrieval_pipeline(
            db_session=MagicMock(),
            store=MagicMock(),
            query="q",
            top_k=5,
            retrieval_method="full_text",
            indexing_config={},
            retrieval_config={},
            filter_metadata=None,
            knowledge_base_id="kb-1",
            request_id="req-bm25",
        )

    actions = [c["action"] for c in recording_billing.charges]
    assert "bm25_search" in actions


def test_execute_retrieval_pipeline_charges_hybrid(billing_env, monkeypatch, recording_billing):
    """hybrid retrieval bills as hybrid_search."""
    monkeypatch.setitem(knowledge_search.CHUNK_RETRIEVER_MAP, "hybrid", _stub_retriever())
    with patch.object(knowledge_search, "_attach_enrichment_metadata", return_value=False):
        knowledge_search._execute_retrieval_pipeline(
            db_session=MagicMock(),
            store=MagicMock(),
            query="q",
            top_k=5,
            retrieval_method="hybrid",
            indexing_config={},
            retrieval_config={},
            filter_metadata=None,
            knowledge_base_id="kb-1",
            request_id="req-hybrid",
        )

    actions = [c["action"] for c in recording_billing.charges]
    assert "hybrid_search" in actions


def test_execute_retrieval_pipeline_bills_reranker_when_invoked(
    billing_env, monkeypatch, recording_billing
):
    """When retrieval_config has a reranker model, reranker_call is billed."""
    fake = _fake_results()
    monkeypatch.setitem(
        knowledge_search.CHUNK_RETRIEVER_MAP, "vector_search", _stub_retriever(fake)
    )
    with (
        patch.object(knowledge_search, "_attach_enrichment_metadata", return_value=False),
        patch.object(knowledge_search, "_apply_reranking", return_value=fake),
    ):
        knowledge_search._execute_retrieval_pipeline(
            db_session=MagicMock(),
            store=MagicMock(),
            query="q",
            top_k=5,
            retrieval_method="vector_search",
            indexing_config={},
            retrieval_config={"reranker": {"model": "zerank-mini"}},
            filter_metadata=None,
            knowledge_base_id="kb-1",
            request_id="req-rerank",
        )

    actions = [c["action"] for c in recording_billing.charges]
    assert "vector_search" in actions
    assert "reranker_call" in actions


def test_execute_retrieval_pipeline_skips_reranker_charge_when_no_model(
    billing_env, monkeypatch, recording_billing
):
    """No reranker model in config → no reranker_call charge."""
    monkeypatch.setitem(knowledge_search.CHUNK_RETRIEVER_MAP, "vector_search", _stub_retriever())
    with patch.object(knowledge_search, "_attach_enrichment_metadata", return_value=False):
        knowledge_search._execute_retrieval_pipeline(
            db_session=MagicMock(),
            store=MagicMock(),
            query="q",
            top_k=5,
            retrieval_method="vector_search",
            indexing_config={},
            retrieval_config={},
            filter_metadata=None,
            knowledge_base_id="kb-1",
            request_id="req-nx",
        )

    actions = [c["action"] for c in recording_billing.charges]
    assert "reranker_call" not in actions


def test_execute_retrieval_pipeline_bills_metadata_enrichment_when_attached(
    billing_env, monkeypatch, recording_billing
):
    """When _attach_enrichment_metadata returns True, metadata_enrichment is billed."""
    monkeypatch.setitem(knowledge_search.CHUNK_RETRIEVER_MAP, "vector_search", _stub_retriever())
    with patch.object(knowledge_search, "_attach_enrichment_metadata", return_value=True):
        knowledge_search._execute_retrieval_pipeline(
            db_session=MagicMock(),
            store=MagicMock(),
            query="q",
            top_k=5,
            retrieval_method="vector_search",
            indexing_config={},
            retrieval_config={},
            filter_metadata=None,
            knowledge_base_id="kb-1",
            request_id="req-meta",
        )

    actions = [c["action"] for c in recording_billing.charges]
    assert "metadata_enrichment" in actions


def test_execute_retrieval_pipeline_skips_metadata_enrichment_when_not_attached(
    billing_env, monkeypatch, recording_billing
):
    """No enrichment attached → no metadata_enrichment charge."""
    monkeypatch.setitem(knowledge_search.CHUNK_RETRIEVER_MAP, "vector_search", _stub_retriever())
    with patch.object(knowledge_search, "_attach_enrichment_metadata", return_value=False):
        knowledge_search._execute_retrieval_pipeline(
            db_session=MagicMock(),
            store=MagicMock(),
            query="q",
            top_k=5,
            retrieval_method="vector_search",
            indexing_config={},
            retrieval_config={},
            filter_metadata=None,
            knowledge_base_id="kb-1",
            request_id="req-nometa",
        )

    actions = [c["action"] for c in recording_billing.charges]
    assert "metadata_enrichment" not in actions


def test_execute_retrieval_pipeline_propagates_payment_required(billing_env, monkeypatch):
    """402 from billing.check_balance bubbles up to the caller."""
    ran = []

    def retriever(**kwargs):
        ran.append(1)
        return _fake_results()

    monkeypatch.setitem(knowledge_search.CHUNK_RETRIEVER_MAP, "vector_search", retriever)

    rec = RecordingBillingAdapter(raise_402=True)
    billing_port.set_billing_adapter(rec)

    with pytest.raises(HTTPException) as exc_info:
        knowledge_search._execute_retrieval_pipeline(
            db_session=MagicMock(),
            store=MagicMock(),
            query="q",
            top_k=5,
            retrieval_method="vector_search",
            indexing_config={},
            retrieval_config={},
            filter_metadata=None,
            knowledge_base_id="kb-1",
        )
    assert exc_info.value.code == 402

    assert ran == []  # retriever never ran
    assert rec.charges == []


def test_execute_retrieval_pipeline_idempotency_key_uses_request_id(
    billing_env, monkeypatch, recording_billing
):
    """Same request_id → same idempotency_parts (so retries don't double-charge)."""
    monkeypatch.setitem(knowledge_search.CHUNK_RETRIEVER_MAP, "vector_search", _stub_retriever())

    with patch.object(knowledge_search, "_attach_enrichment_metadata", return_value=False):
        knowledge_search._execute_retrieval_pipeline(
            db_session=MagicMock(),
            store=MagicMock(),
            query="q",
            top_k=5,
            retrieval_method="vector_search",
            indexing_config={},
            retrieval_config={},
            filter_metadata=None,
            knowledge_base_id="kb-1",
            request_id="req-stable",
        )
        knowledge_search._execute_retrieval_pipeline(
            db_session=MagicMock(),
            store=MagicMock(),
            query="q",
            top_k=5,
            retrieval_method="vector_search",
            indexing_config={},
            retrieval_config={},
            filter_metadata=None,
            knowledge_base_id="kb-1",
            request_id="req-stable",
        )

    # Both invocations charged with the same idempotency_parts.
    parts = [c["idempotency_parts"] for c in recording_billing.charges]
    assert len(parts) == 2
    assert parts[0] == parts[1] == ("req-stable",)


def test_aexecute_retrieval_pipeline_charges_vector_search(
    billing_env, monkeypatch, recording_billing
):
    """Async variant mirrors sync: vector_search retrieval bills correctly."""
    monkeypatch.setitem(
        knowledge_search.ASYNC_CHUNK_RETRIEVER_MAP,
        "vector_search",
        _stub_async_retriever(),
    )
    with patch.object(knowledge_search, "_attach_enrichment_metadata", return_value=False):
        asyncio.run(
            knowledge_search._aexecute_retrieval_pipeline(
                db_session=MagicMock(),
                store=MagicMock(),
                query="q",
                top_k=5,
                retrieval_method="vector_search",
                indexing_config={},
                retrieval_config={},
                filter_metadata=None,
                knowledge_base_id="kb-1",
                request_id="req-async",
            )
        )

    assert recording_billing.balance_checks == [1]
    assert len(recording_billing.charges) == 1
    assert recording_billing.charges[0]["action"] == "vector_search"
    assert recording_billing.charges[0]["ref_id"] == "req-async"


def test_attach_enrichment_metadata_returns_false_for_empty():
    """Empty results → False (no work done, so no enrichment charge)."""
    assert knowledge_search._attach_enrichment_metadata(MagicMock(), [], "kb-1") is False
