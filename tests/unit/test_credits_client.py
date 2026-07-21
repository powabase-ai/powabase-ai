"""Tests for PS credits_client.post_charge."""

import pytest
import requests
import responses
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from agentic_project_service.services.billing_cloud.credits_client import (
    post_charge,
)


@pytest.fixture(autouse=True)
def billing_env(monkeypatch):
    """Provide BILLING_SERVICE_URL + JWT signing key envs for every test."""
    private = ec.generate_private_key(ec.SECP256R1())
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    monkeypatch.setenv("BILLING_SERVICE_URL", "http://billing-service:5100")
    monkeypatch.setenv("BILLING_JWT_PRIVATE_KEY_PEM", private_pem.decode())
    monkeypatch.setenv("BILLING_JWT_KID", "proj-test-v1")
    # Force fast retries in tests
    monkeypatch.setattr(
        "agentic_project_service.services.billing_cloud.credits_client.RETRY_BACKOFF_SECONDS",
        [0.001, 0.001, 0.001],
    )
    # The module reads BILLING_URL at import time; override it here too.
    monkeypatch.setattr(
        "agentic_project_service.services.billing_cloud.credits_client.BILLING_URL",
        "http://billing-service:5100",
    )


@responses.activate
def test_post_charge_happy_path():
    """post_charge calls /internal/charges with correct payload and returns the result."""
    responses.add(
        responses.POST,
        "http://billing-service:5100/internal/charges",
        json={"id": "charge-1", "credits": -1, "status": "committed", "balance": 4999},
        status=201,
    )
    result = post_charge(
        org_id="org-uuid",
        project_id="proj-uuid",
        action="vector_search",
        quantity=1,
        ref_type="retrieval",
        ref_id="abc-123",
        idempotency_key="test-1",
    )
    assert result.success is True
    assert result.charge_id == "charge-1"
    assert result.balance == 4999
    assert responses.calls[0].request.headers["Authorization"].startswith("Bearer ")


@responses.activate
def test_post_charge_402_returns_insufficient_credits_result():
    """402 must NOT raise — would clobber op status on the 11 non-Celery surfaces.

    Returns ChargeResult(success=False, failure_mode='insufficient_credits') so
    callers can check success without try/except boilerplate.
    """
    responses.add(
        responses.POST,
        "http://billing-service:5100/internal/charges",
        json={"error": "insufficient credits", "balance": 0, "required": 1},
        status=402,
    )
    result = post_charge(
        org_id="org-uuid",
        project_id="proj-uuid",
        action="vector_search",
        quantity=1,
        idempotency_key="test-2",
    )
    assert result.success is False
    assert result.failure_mode == "insufficient_credits"
    assert result.balance == 0


@responses.activate
def test_post_charge_5xx_retries_then_returns_terminal_failure():
    responses.add(
        responses.POST,
        "http://billing-service:5100/internal/charges",
        json={"error": "internal"},
        status=500,
    )
    result = post_charge(
        org_id="org-uuid",
        project_id="proj-uuid",
        action="vector_search",
        quantity=1,
        idempotency_key="test-3",
    )
    assert result.success is False
    assert result.failure_mode in ("billing_5xx", "retry_exhausted")
    # 3 attempts via RETRY_BACKOFF_SECONDS list of length 3
    assert len(responses.calls) == 3


@responses.activate
def test_post_charge_network_timeout_retries():
    """Simulate connection error → retries → terminal failure."""
    responses.add(
        responses.POST,
        "http://billing-service:5100/internal/charges",
        body=requests.ConnectionError("connection refused"),
    )
    result = post_charge(
        org_id="org-uuid",
        project_id="proj-uuid",
        action="vector_search",
        quantity=1,
        idempotency_key="test-4",
    )
    assert result.success is False
    assert result.failure_mode == "network_timeout"


@responses.activate
def test_post_charge_4xx_other_than_402_does_not_retry():
    """4xx (not 402) is a permanent failure; don't retry."""
    responses.add(
        responses.POST,
        "http://billing-service:5100/internal/charges",
        json={"error": "bad request"},
        status=400,
    )
    result = post_charge(
        org_id="org-uuid",
        project_id="proj-uuid",
        action="vector_search",
        quantity=1,
        idempotency_key="test-5",
    )
    assert result.success is False
    assert result.failure_mode == "client_error"
    # 1 attempt only — no retries for non-transient 4xx
    assert len(responses.calls) == 1
