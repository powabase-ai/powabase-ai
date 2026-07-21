"""Tests for PS balance_cache — 30s TTL, fail-closed for free tier."""

import json
import time

import pytest
import responses
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from werkzeug.exceptions import ServiceUnavailable

from agentic_project_service.services.billing_cloud import balance_cache
from agentic_project_service.services.billing_cloud.balance_cache import (
    PaymentRequired,
    check_balance_or_503,
    get_balance_cached,
    invalidate,
)


@pytest.fixture(autouse=True)
def billing_env(monkeypatch):
    """Provide BILLING_SERVICE_URL + JWT signing key envs and clear cache between tests."""
    private = ec.generate_private_key(ec.SECP256R1())
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    monkeypatch.setenv("BILLING_SERVICE_URL", "http://billing-service:5100")
    monkeypatch.setenv("BILLING_JWT_PRIVATE_KEY_PEM", private_pem.decode())
    monkeypatch.setenv("BILLING_JWT_KID", "proj-test-v1")
    # Module reads BILLING_URL at import time; override it here too.
    monkeypatch.setattr(balance_cache, "BILLING_URL", "http://billing-service:5100")
    balance_cache.clear_all()  # reset cache between tests
    yield


@responses.activate
def test_balance_cache_hit_returns_cached():
    """First call hits billing, second call within TTL hits cache."""
    responses.add(
        responses.GET,
        "http://billing-service:5100/api/organizations/_/credits/balance",
        json={"balance": 5000, "org_id": "org-1"},
        status=200,
    )
    b1 = get_balance_cached(org_id="org-1", project_id="proj-1")
    b2 = get_balance_cached(org_id="org-1", project_id="proj-1")
    assert b1 == 5000
    assert b2 == 5000
    assert len(responses.calls) == 1  # only fetched once


@responses.activate
def test_balance_cache_miss_after_ttl(monkeypatch):
    """After TTL expires, refetch from billing."""
    responses.add(
        responses.GET,
        "http://billing-service:5100/api/organizations/_/credits/balance",
        json={"balance": 5000, "org_id": "org-1"},
        status=200,
    )
    # First call
    get_balance_cached(org_id="org-1", project_id="proj-1")
    # Shift time by mocking time.time
    real_time = time.time()
    monkeypatch.setattr(balance_cache.time, "time", lambda: real_time + 31)
    get_balance_cached(org_id="org-1", project_id="proj-1")
    assert len(responses.calls) == 2


@responses.activate
def test_check_balance_failclosed_for_free_tier_when_unreachable():
    """Free tier + cache miss + billing unreachable → ServiceUnavailable."""
    # No responses.add — billing is unreachable
    with pytest.raises(ServiceUnavailable):
        check_balance_or_503(
            org_id="org-1",
            project_id="proj-1",
            estimated_cost=10,
            plan_tier="free",
        )


@responses.activate
def test_check_balance_paid_tier_applies_soft_cap(monkeypatch):
    """Paid tier is no longer a no-op (#445): it fetches balance and applies the
    projected-balance soft cap. With the default GRACE=0 it is prepaid — a paid
    org whose balance < estimated_cost is refused, exactly like free tier."""
    balance_cache.clear_all()
    monkeypatch.delenv("BILLING_PAID_TIER_SOFT_CAP_GRACE_CREDITS", raising=False)
    responses.add(
        responses.GET,
        "http://billing-service:5100/api/organizations/_/credits/balance",
        json={"balance": 5, "org_id": "org-1"},
        status=200,
    )
    with pytest.raises(PaymentRequired):
        check_balance_or_503(
            org_id="org-1",
            project_id="proj-1",
            estimated_cost=10,
            plan_tier="pro",
        )


@responses.activate
def test_check_balance_402_when_insufficient():
    """Free tier + balance < estimated_cost → PaymentRequired."""
    responses.add(
        responses.GET,
        "http://billing-service:5100/api/organizations/_/credits/balance",
        json={"balance": 5, "org_id": "org-1"},
        status=200,
    )
    with pytest.raises(PaymentRequired):
        check_balance_or_503(
            org_id="org-1",
            project_id="proj-1",
            estimated_cost=10,
            plan_tier="free",
        )


@responses.activate
def test_invalidate_forces_refetch():
    """invalidate() drops cache; next call refetches."""
    responses.add(
        responses.GET,
        "http://billing-service:5100/api/organizations/_/credits/balance",
        json={"balance": 5000},
        status=200,
    )
    get_balance_cached(org_id="org-1", project_id="proj-1")
    invalidate("org-1")
    get_balance_cached(org_id="org-1", project_id="proj-1")
    assert len(responses.calls) == 2


@responses.activate
def test_fetch_failure_with_stale_cache_still_fail_closed(monkeypatch):
    """Spec line 63: cache miss / cache unavailable → fail-closed with 503.

    Even when a stale entry exists, a fresh-fetch failure must NOT fall back
    to the stale value. Returning stale would let a free-tier org keep
    spending past its cap whenever billing is briefly unreachable.
    """
    # First call: succeed and prime the cache
    responses.add(
        responses.GET,
        "http://billing-service:5100/api/organizations/_/credits/balance",
        json={"balance": 5000},
        status=200,
    )
    assert get_balance_cached(org_id="org-1", project_id="proj-1") == 5000
    responses.reset()
    # Move past TTL — entry is now stale
    real_time = time.time()
    monkeypatch.setattr(balance_cache.time, "time", lambda: real_time + 31)
    # Fresh fetch fails (no response registered → ConnectionError)
    result = get_balance_cached(org_id="org-1", project_id="proj-1")
    assert result is None, "must fail-closed; stale fallback is forbidden"


def test_post_charge_invalidates_balance_cache_on_commit(monkeypatch):
    """credits_client.post_charge must call balance_cache.invalidate on a
    successful charge so the next free-tier check fetches a fresh balance.

    Wiring test: confirms the production call exists, not just that the
    function works in isolation."""
    from agentic_project_service.services.billing_cloud import credits_client

    private = ec.generate_private_key(ec.SECP256R1())
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    monkeypatch.setenv("BILLING_JWT_PRIVATE_KEY_PEM", private_pem.decode())
    monkeypatch.setenv("BILLING_JWT_KID", "proj-test-v1")
    monkeypatch.setattr(credits_client, "BILLING_URL", "http://billing-service:5100")
    monkeypatch.setattr(credits_client, "RETRY_BACKOFF_SECONDS", [0.001])

    invalidated: list[str] = []

    def _spy(org_id: str) -> None:
        invalidated.append(org_id)

    monkeypatch.setattr(credits_client.balance_cache, "invalidate", _spy)

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "http://billing-service:5100/internal/charges",
            json={"id": "c-1", "credits": -1, "status": "committed", "balance": 99},
            status=201,
        )
        result = credits_client.post_charge(
            org_id="org-cache-1",
            project_id="proj-1",
            action="vector_search",
            quantity=1,
            idempotency_key="cache-test-1",
        )
    assert result.success is True
    assert invalidated == ["org-cache-1"]


def test_post_charge_invalidates_balance_cache_on_402(monkeypatch):
    """A 402 means the balance is exhausted right now — invalidate so the
    next free-tier check fetches the fresh (likely zero) value and short-
    circuits without a wasted post_charge round trip."""
    from agentic_project_service.services.billing_cloud import credits_client

    private = ec.generate_private_key(ec.SECP256R1())
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    monkeypatch.setenv("BILLING_JWT_PRIVATE_KEY_PEM", private_pem.decode())
    monkeypatch.setenv("BILLING_JWT_KID", "proj-test-v1")
    monkeypatch.setattr(credits_client, "BILLING_URL", "http://billing-service:5100")
    monkeypatch.setattr(credits_client, "RETRY_BACKOFF_SECONDS", [0.001])

    invalidated: list[str] = []

    def _spy(org_id: str) -> None:
        invalidated.append(org_id)

    monkeypatch.setattr(credits_client.balance_cache, "invalidate", _spy)

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "http://billing-service:5100/internal/charges",
            json={"error": "insufficient credits", "balance": 0, "required": 1},
            status=402,
        )
        result = credits_client.post_charge(
            org_id="org-cache-2",
            project_id="proj-1",
            action="vector_search",
            quantity=1,
            idempotency_key="cache-test-2",
        )
    assert result.success is False
    assert result.failure_mode == "insufficient_credits"
    assert invalidated == ["org-cache-2"]


def test_payment_required_response_is_json_with_renews_at(monkeypatch):
    """402 response body is JSON with error, balance, estimated_cost, renews_at.

    Note: this test calls check_balance_or_503 + raised.get_response()
    directly without a Flask app context. The implementation must therefore
    render the body without calling flask.jsonify (which requires
    current_app); use werkzeug.Response + json.dumps instead.
    """
    monkeypatch.setattr(
        "agentic_project_service.services.billing_cloud.balance_cache.get_balance_cached",
        lambda **kw: 5,
    )
    raised = None
    try:
        check_balance_or_503(
            org_id="00000000-0000-0000-0000-000000000001",
            project_id="00000000-0000-0000-0000-000000000002",
            estimated_cost=10,
            plan_tier="free",
        )
    except PaymentRequired as exc:
        raised = exc
    assert raised is not None
    response = raised.get_response()
    assert response.status_code == 402
    assert response.content_type == "application/json"
    body = json.loads(response.get_data(as_text=True))
    assert body["error"] == "insufficient_credits"
    assert body["balance"] == 5
    assert body["estimated_cost"] == 10
    # Pin to the helper output rather than just shape (endswith would
    # accept 1970-01-01). NOTE: this is a self-comparison — both sides
    # call PS's _compute_renews_at — so it does NOT verify cross-package
    # parity with billing-service's compute_renews_at. A formal parity
    # test would need a new top-level integration test dir importing
    # both packages; tracked as a follow-up.
    from agentic_project_service.services.billing_cloud.balance_cache import _compute_renews_at

    assert body["renews_at"] == _compute_renews_at()
