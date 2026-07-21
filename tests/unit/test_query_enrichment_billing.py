"""Tests for query_enrichment billing wiring, routed through the billing port.

Verifies:
  * billing.check_balance() is called BEFORE the LLM call
  * billing.charge() is called on success with action=query_enrichment and
    idempotency_parts=(request_id,) — the tail the cloud adapter combines
    with org_id to build the deterministic idempotency key
  * A 402 from check_balance propagates (surfaces as HTTP 402); a 503
    propagates as ServiceUnavailable
  * The short-circuit path for trivial queries never touches billing

"ctx unconfigured -> billing skipped" is no longer this module's concern —
that guard now lives in CloudBillingAdapter (see test_billing_cloud_adapter.py).
"""

from unittest.mock import MagicMock, patch

import pytest
from werkzeug.exceptions import HTTPException, ServiceUnavailable

from agentic_project_service.services import billing_port, query_enrichment
from tests.support.billing import RecordingBillingAdapter


@pytest.fixture(autouse=True)
def _stub_byok_resolver():
    """The BYOK-unification refactor (2026-06-03) wraps the litellm call
    in ``with_llm_key`` which calls ``get_all_user_provider_keys``
    → SQLAlchemy session → needs a Flask app context. Unit tests in this
    file mock litellm directly and don't spin up the app — short-circuit
    the resolver to return ``{}`` so the wrapper yields ``None`` (no
    BYOK injection) and the test path is otherwise unchanged.
    """
    with patch(
        "agentic_project_service.services.llm_call.get_all_user_provider_keys",
        return_value={},
    ):
        yield


@pytest.fixture
def billing_env(monkeypatch):
    """Set the env vars that make get_billing_context() return a context."""
    monkeypatch.setenv("BILLING_ORG_ID", "org-1")
    monkeypatch.setenv("PROJECT_ID", "proj-1")
    monkeypatch.delenv("BILLING_PLAN_TIER", raising=False)  # default free
    yield


def _mock_litellm_completion():
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[
        0
    ].message.content = '{"enriched_query": "enriched", "keywords": "kw OR words"}'
    return response


def test_enrich_query_calls_check_balance_then_charges(billing_env, recording_billing):
    """Happy path: balance check fires before LLM, charge fires on success."""
    with patch("litellm.completion", return_value=_mock_litellm_completion()):
        result = query_enrichment.enrich_query(
            query="how does redis caching work?",
            retrieval_method="vector_search",
            request_id="req-123",
        )

    # billing.check_balance() fired with estimated_cost=1. org_id/project_id/
    # plan_tier are no longer visible at this layer — that mapping is the
    # cloud adapter's job (test_billing_cloud_adapter.py).
    assert recording_billing.balance_checks == [1]

    # billing.charge() fired with action=query_enrichment and
    # idempotency_parts=(request_id,) — the tail the cloud adapter combines
    # with org_id to rebuild the same key as before the port migration.
    assert len(recording_billing.charges) == 1
    charge = recording_billing.charges[0]
    assert charge["action"] == "query_enrichment"
    assert charge["quantity"] == 1
    assert charge["ref_type"] == "retrieval"
    assert charge["ref_id"] == "req-123"
    assert charge["idempotency_parts"] == ("req-123",)

    # Sanity-check the result
    assert result["enriched_query"] == "enriched"
    assert result["keyword_query"] == "kw OR words"


def test_enrich_query_propagates_payment_required(billing_env):
    """A 402 from billing.check_balance propagates to the caller."""
    rec = RecordingBillingAdapter(raise_402=True)
    billing_port.set_billing_adapter(rec)

    with patch("litellm.completion") as mock_llm:
        with pytest.raises(HTTPException) as exc_info:
            query_enrichment.enrich_query(
                query="some query",
                retrieval_method="vector_search",
            )
    assert exc_info.value.code == 402

    # LLM never ran and we never posted a charge
    mock_llm.assert_not_called()
    assert rec.charges == []


def test_enrich_query_propagates_service_unavailable(billing_env):
    """A 503 from billing.check_balance propagates (billing unreachable)."""
    rec = RecordingBillingAdapter(raise_503=True)
    billing_port.set_billing_adapter(rec)

    with patch("litellm.completion") as mock_llm:
        with pytest.raises(ServiceUnavailable):
            query_enrichment.enrich_query(
                query="some query",
                retrieval_method="vector_search",
            )

    mock_llm.assert_not_called()
    assert rec.charges == []


def test_enrich_query_short_circuit_bypasses_billing(billing_env, recording_billing):
    """Short queries with no session history return immediately — no billing."""
    result = query_enrichment.enrich_query(
        query="a",  # < 2 chars
        retrieval_method="vector_search",
    )

    assert recording_billing.balance_checks == []
    assert recording_billing.charges == []
    assert result == {"enriched_query": "a", "keyword_query": "a"}
