"""Tests for BillingLogger._build_charge_args 4-row gate table.

Verifies the post-fix semantic: skip llm_call iff this call is inside an
agent/orch/workflow path (recoupable=True) AND the user has BYOK for the
provider (user paid OpenAI). Else charge llm_call (platform paid OpenAI).
"""

from types import SimpleNamespace

import pytest

from agentic_project_service.services.billing_cloud.identity import (
    current_byok_providers,
    recoupable_llm_call,
)
from agentic_project_service.services.billing_cloud.billing_litellm import _build_charge_args


@pytest.fixture
def env_billing_on(monkeypatch):
    monkeypatch.setenv("BILLING_AI_ON_US_ENABLED", "true")
    monkeypatch.setenv("BILLING_ORG_ID", "org-test")
    monkeypatch.setenv("PROJECT_ID", "proj-test")


@pytest.fixture
def mock_response():
    """Build a minimal LiteLLM-like response object with the fields _build_charge_args reads."""
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )


@pytest.fixture
def base_kwargs():
    return {
        "call_type": "acompletion",
        "model": "gpt-4o-mini",
        "response_cost": 0.0001,
        "litellm_call_id": "call_test_123",
    }


def test_agent_with_byok_skips(env_billing_on, mock_response, base_kwargs):
    """Agent path + user has BYOK → user paid → skip llm_call."""
    byok_token = current_byok_providers.set(frozenset({"openai"}))
    try:
        with recoupable_llm_call():
            args = _build_charge_args(base_kwargs, mock_response)
        assert args is None
    finally:
        current_byok_providers.reset(byok_token)


def test_agent_without_byok_charges(env_billing_on, mock_response, base_kwargs):
    """Agent path + no BYOK → platform paid → charge llm_call."""
    byok_token = current_byok_providers.set(frozenset())
    try:
        with recoupable_llm_call():
            args = _build_charge_args(base_kwargs, mock_response)
        assert args is not None
        assert args["action"] == "llm_call"
        assert args["unit_credits"] > 0
    finally:
        current_byok_providers.reset(byok_token)


def test_platform_internal_with_byok_charges(env_billing_on, mock_response, base_kwargs):
    """Platform-internal path (recoupable=False default) + user has BYOK → platform paid
    (user's key was NOT injected for this path) → MUST charge llm_call.

    This is the bug the fix closes: before, this returned None (silent platform absorption).
    """
    byok_token = current_byok_providers.set(frozenset({"openai"}))
    try:
        # No recoupable_llm_call() — simulates metadata_enricher / chunk_embed / etc.
        args = _build_charge_args(base_kwargs, mock_response)
        assert args is not None
        assert args["action"] == "llm_call"
    finally:
        current_byok_providers.reset(byok_token)


def test_platform_internal_without_byok_charges(env_billing_on, mock_response, base_kwargs):
    """Platform-internal path + no BYOK → platform paid → charge llm_call (already correct pre-fix)."""
    byok_token = current_byok_providers.set(frozenset())
    try:
        args = _build_charge_args(base_kwargs, mock_response)
        assert args is not None
        assert args["action"] == "llm_call"
    finally:
        current_byok_providers.reset(byok_token)
