"""End-to-end: AI-on-us agent run → llm_call ledger row written.
Uses litellm.acompletion(mock_response=...) so BillingLogger callback dispatches."""

import asyncio
from unittest.mock import patch, AsyncMock
import litellm
import pytest

from agentic_project_service.services.billing_cloud.identity import recoupable_llm_call


@pytest.fixture
def ai_on_us_enabled(monkeypatch):
    monkeypatch.setenv("BILLING_ENABLED", "true")
    monkeypatch.setenv("BILLING_AI_ON_US_ENABLED", "true")
    monkeypatch.setenv("BILLING_LLM_MARKUP_MULTIPLIER", "1.25")
    monkeypatch.setenv("BILLING_ORG_ID", "test-org")
    monkeypatch.setenv("PROJECT_ID", "test-project")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")


@pytest.fixture(autouse=True)
def reset_litellm_callbacks():
    saved = list(litellm.callbacks)
    litellm.callbacks = []
    yield
    litellm.callbacks = saved


@pytest.mark.asyncio
async def test_ai_on_us_call_fires_billing_logger_with_charge_call(ai_on_us_enabled):
    """litellm.acompletion(mock_response=...) keeps real callback dispatch,
    so BillingLogger fires and (when no BYOK provider) calls post_charge."""
    from agentic_project_service.services.billing_cloud.billing_litellm import (
        register_billing_logger,
    )
    from agentic_project_service.services.billing_cloud.identity import current_byok_providers

    register_billing_logger()
    # No BYOK key for anthropic in current_byok_providers — should fire charge
    current_byok_providers.set(frozenset())

    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
        new=AsyncMock(),
    ) as mock_post:
        response = await litellm.acompletion(
            model="anthropic/claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="ok",
        )
        await asyncio.sleep(0.1)  # allow callback to complete

    assert response is not None
    # The callback should have fired exactly once (one completion call).
    # Strict ``== 1`` is intentional — symmetric with the BYOK-skip test
    # below (``== 0``) and tight enough to catch a double-charge regression
    # if the same call ever ends up wired into the callback twice.
    assert (
        mock_post.call_count == 1
    ), "BillingLogger did not call post_charge exactly once — dispatch may have been bypassed or double-fired"
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["action"] == "llm_call"
    assert call_kwargs["org_id"] == "test-org"
    assert call_kwargs["project_id"] == "test-project"
    # LiteLLM strips the provider prefix from `model` in callback kwargs
    # (verified empirically with litellm==1.83.14) — the BillingLogger stores
    # the stripped form. The provider is still resolved correctly via
    # litellm.get_llm_provider() for the BYOK skip check.
    assert call_kwargs["metadata"]["model"] == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_byok_call_does_not_fire_charge(ai_on_us_enabled):
    """With anthropic in current_byok_providers and inside recoupable_llm_call(),
    the callback should skip (agent/orch/workflow path — user paid via their BYOK key)."""
    from agentic_project_service.services.billing_cloud.billing_litellm import (
        register_billing_logger,
    )
    from agentic_project_service.services.billing_cloud.identity import current_byok_providers

    register_billing_logger()
    current_byok_providers.set(frozenset({"anthropic"}))
    try:
        with patch(
            "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
            new=AsyncMock(),
        ) as mock_post:
            with recoupable_llm_call():
                await litellm.acompletion(
                    model="anthropic/claude-haiku-4-5",
                    messages=[{"role": "user", "content": "hi"}],
                    mock_response="ok",
                )
                await asyncio.sleep(0.1)
        assert mock_post.call_count == 0, "BYOK skip failed — post_charge was called"
    finally:
        current_byok_providers.set(frozenset())
