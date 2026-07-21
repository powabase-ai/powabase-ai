"""Pin every COPILOT_MODEL_OPTIONS entry to the deployed LiteLLM's model
registry. Failure modes this catches at CI time (not at boot):

1. A new picker model whose ID doesn't resolve in litellm at all (typo
   or model name change upstream) — every charge for that model would
   fail the response_cost > 0 gate and silently free-ride.
2. A picker model that resolves but lacks ``input_cost_per_token`` —
   same silent revenue leak.
3. A picker model that LiteLLM does not flag as
   ``supports_function_calling`` — the copilot uses tools, so a non-tool
   model would fail every run on the first tool call.

These run as ordinary unit tests (no network, no DB) — litellm reads its
model_cost JSON locally — and add ~50ms to the suite.
"""

from __future__ import annotations

import pytest

import litellm

from agentic_project_service.services.copilot_config import COPILOT_MODEL_OPTIONS


_PICKER_MODEL_IDS = [model_id for _label, model_id in COPILOT_MODEL_OPTIONS]


@pytest.mark.parametrize("model_id", _PICKER_MODEL_IDS)
def test_picker_model_resolves_in_litellm(model_id: str) -> None:
    """``litellm.get_model_info`` must recognize every picker entry."""
    info = litellm.get_model_info(model_id)
    assert info is not None, f"model not found in litellm registry: {model_id}"


@pytest.mark.parametrize("model_id", _PICKER_MODEL_IDS)
def test_picker_model_has_cost_data(model_id: str) -> None:
    """Without ``input_cost_per_token`` BillingLogger would drop every
    AI-on-us charge for the model with the ``llm_call_unknown_cost``
    error path. Hard fail at PR time so we never ship a free-LLM model."""
    info = litellm.get_model_info(model_id)
    cost = info.get("input_cost_per_token")
    assert cost and cost > 0, (
        f"{model_id} has no input_cost_per_token — AI-on-us charges would "
        f"silently drop. Either add the model to a local cost override or "
        f"remove it from COPILOT_MODEL_OPTIONS."
    )


@pytest.mark.parametrize("model_id", _PICKER_MODEL_IDS)
def test_picker_model_supports_function_calling(model_id: str) -> None:
    """The copilot uses tools (modify_workflow, get_block_info, etc.).
    A model that LiteLLM doesn't flag as function-calling-capable would
    fail every run as soon as a tool fires."""
    assert litellm.supports_function_calling(model=model_id), (
        f"{model_id} does not support function calling per litellm — copilot "
        f"would fail on first tool call. Remove from COPILOT_MODEL_OPTIONS."
    )
