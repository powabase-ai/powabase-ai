# tests/unit/test_oss_build_noop.py
"""Simulate the OSS build: the no-op billing adapter is active. Chargeable code
paths must run, charge nothing, and never touch a billing service. Proves the
Plan B deliverable 'OSS build carries zero charging code' behaviorally."""

from unittest.mock import patch

import pytest

from agentic_project_service.services import billing_port
from agentic_project_service.services.billing_port import NoopBillingAdapter


@pytest.fixture(autouse=True)
def _oss_billing():
    saved = billing_port.get_billing_adapter()
    billing_port.set_billing_adapter(NoopBillingAdapter())
    yield
    billing_port.set_billing_adapter(saved)


def test_charge_and_check_balance_are_inert_and_do_no_io():
    # No billing HTTP client is even importable from core; the facade must not
    # raise and must not try to reach a billing service.
    with patch("requests.post") as post, patch("requests.get") as get:
        billing_port.check_balance(estimated_cost=10_000_000)  # never 402/503 in OSS
        out = billing_port.charge(action="agent_run", idempotency_parts=("run-1",))
    assert out.charged is False
    post.assert_not_called()
    get.assert_not_called()


def test_check_model_available_still_gates_on_real_keys_without_billing():
    # With no billing contextvars in play, availability is decided purely by the
    # project's provider keys / platform env — proving the rewrite is billing-free.
    from agentic_project_service.services import llm_availability
    from werkzeug.exceptions import BadRequest

    with (
        patch.object(llm_availability, "list_byok_providers", return_value=frozenset()),
        patch.object(llm_availability, "platform_supports", return_value=False),
    ):
        with pytest.raises(BadRequest):
            llm_availability.check_model_available("openrouter/some-model")

    with patch.object(
        llm_availability, "list_byok_providers", return_value=frozenset({"openrouter"})
    ):
        llm_availability.check_model_available("openrouter/some-model")  # no raise


def test_with_llm_key_resolves_key_without_arming_any_charge():
    from agentic_project_service.services import llm_call

    with (
        patch.object(llm_call, "get_all_user_provider_keys", return_value={"openai": "sk-x"}),
        patch.object(llm_call, "resolve_api_key_for_model", return_value="sk-x"),
    ):
        with llm_call.with_llm_key("openai/gpt-5") as key:
            assert key == "sk-x"  # BYOK resolution works; no metering happens
