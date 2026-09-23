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

These run as ordinary unit tests (no network, no DB), but only because
``tests/conftest.py`` sets ``LITELLM_LOCAL_MODEL_COST_MAP=True`` before
anything imports litellm: the registry these assertions read is then the
``model_prices_and_context_window_backup.json`` inside the pinned litellm
wheel, not a file fetched from GitHub at import time. Without that, the
assertions track whatever upstream publishes, and an upstream edit fails this
file on an unchanged main — which is what happened when three ids were dropped
from the live map. ``test_registry_is_the_pinned_local_cost_map`` below guards
the arrangement.

What that costs: these tests can no longer notice that a provider retired a
model, because the pinned snapshot still describes it. They catch our own
typos and slug drift against the litellm the service actually deploys, which
is what a gate on every PR should do; noticing a retirement is a job for
whoever bumps the litellm pin (the boot-time guard in main.py reads the
deployment's own map at startup and logs any picker entry it cannot price).
"""

from __future__ import annotations

import os

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info

from agentic_project_service.services.settings_registry import _LLM_MODEL_CHOICES


def test_registry_is_the_pinned_local_cost_map() -> None:
    """Every assertion below is only reproducible if litellm loaded its cost
    map from the pinned wheel. Assert both halves: the variable is set, and
    litellm actually honored it (it reads the variable once, at import time, so
    setting it after the first ``import litellm`` in the process is a silent
    no-op that would put this file back on the network)."""
    assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true", (
        "LITELLM_LOCAL_MODEL_COST_MAP is not set — tests/conftest.py sets it so "
        "this file asserts against the pinned litellm's own cost map instead of "
        "a JSON fetched from GitHub at import time."
    )
    source = get_model_cost_map_source_info()
    assert source["is_env_forced"] and source["source"] == "local", (
        "litellm loaded its cost map from "
        f"{source['source']} (env_forced={source['is_env_forced']}, "
        f"url={source['url']}) — something imported litellm before "
        "tests/conftest.py set LITELLM_LOCAL_MODEL_COST_MAP, so these "
        "assertions are running against the live upstream map."
    )


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
