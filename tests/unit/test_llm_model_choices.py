"""Pin every _LLM_MODEL_CHOICES entry to the deployed LiteLLM's model
registry. _LLM_MODEL_CHOICES is the shared picker behind the agent and
copilot model settings (AGENT_DEFAULT_MODEL, copilot_model, …), so the same
guards the copilot picker gets (see test_copilot_picker_models.py) must cover
this surface too. Failure modes caught at CI time (not at boot):

1. A picker model whose ID doesn't resolve in litellm at all (typo or an
   upstream OpenRouter slug rename — the exact class of bug this PR fixes,
   e.g. the old ``openrouter/mistralai/mistral-small-3.1`` → cost-map miss).
2. A picker model that resolves but lacks ``input_cost_per_token`` — every
   AI-on-us charge for it would silently free-ride.
3. A picker model that LiteLLM does not flag as ``supports_function_calling``
   — the agent and copilot use tools, so a non-tool model would fail every
   run on the first tool call (this is what excludes
   ``openrouter/mistralai/mistral-small-3.1-24b-instruct``, which resolves
   and has cost but reports function-calling = False).

These run as ordinary unit tests (no network, no DB) — litellm reads its
model_cost JSON locally.
"""

from __future__ import annotations

import pytest

import litellm

from agentic_project_service.services.settings_registry import _LLM_MODEL_CHOICES


@pytest.mark.parametrize("model_id", _LLM_MODEL_CHOICES)
def test_choice_resolves_in_litellm(model_id: str) -> None:
    """``litellm.get_model_info`` must recognize every picker entry."""
    info = litellm.get_model_info(model_id)
    assert info is not None, f"model not found in litellm registry: {model_id}"


@pytest.mark.parametrize("model_id", _LLM_MODEL_CHOICES)
def test_choice_has_cost_data(model_id: str) -> None:
    """Without ``input_cost_per_token`` BillingLogger would drop every
    AI-on-us charge for the model. Hard fail at PR time so we never ship a
    free-LLM model into the picker."""
    info = litellm.get_model_info(model_id)
    cost = info.get("input_cost_per_token")
    assert cost and cost > 0, (
        f"{model_id} has no input_cost_per_token — AI-on-us charges would "
        f"silently drop. Either add a local cost override or remove it from "
        f"_LLM_MODEL_CHOICES."
    )


@pytest.mark.parametrize("model_id", _LLM_MODEL_CHOICES)
def test_choice_supports_function_calling(model_id: str) -> None:
    """The agent and copilot use tools. A model LiteLLM doesn't flag as
    function-calling-capable would fail every run as soon as a tool fires,
    so it must not be selectable here."""
    assert litellm.supports_function_calling(model=model_id), (
        f"{model_id} does not support function calling per litellm — the "
        f"agent/copilot would fail on first tool call. Remove from "
        f"_LLM_MODEL_CHOICES."
    )
