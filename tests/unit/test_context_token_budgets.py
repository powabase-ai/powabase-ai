"""The two context-token budgets are peers, and their labels don't say so.

``KB_DEFAULT_MAX_CONTEXT_TOKENS`` ("Max Context Tokens", under Knowledge
Bases) bounds one ``knowledge_search`` result. ``DEFAULT_MAX_CONTEXT_TOKENS``
("Agent Max Context Tokens", under Agents) bounds the context an agent
preloads before its first LLM call. They add up within a run rather than one
containing the other — an earlier draft of this file asserted containment,
which is why the relation is spelled out here and pinned nowhere.

What is worth pinning is the pair of numbers, because they are a deliberate
product decision rather than a derived value: expansion measured at ~33k
tokens against a 32k limit, so the budget had to move above ordinary
operation, and the agent-side moved with it to keep the two comparable. A
future edit that halves one of them should have to say so.
"""

from __future__ import annotations

import pytest

from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY

BUDGET_KEYS = ("KB_DEFAULT_MAX_CONTEXT_TOKENS", "DEFAULT_MAX_CONTEXT_TOKENS")


@pytest.mark.parametrize("key", BUDGET_KEYS)
def test_the_budget_defaults_are_the_ones_that_were_chosen(key):
    assert SETTINGS_REGISTRY[key].default == 64000


@pytest.mark.parametrize("key", BUDGET_KEYS)
def test_both_budgets_stay_within_their_own_bounds(key):
    """A default outside its own min/max would be silently unreachable through
    the settings API."""
    setting = SETTINGS_REGISTRY[key]

    assert setting.min <= setting.default <= setting.max


@pytest.mark.parametrize("key", BUDGET_KEYS)
def test_each_budget_says_which_one_it_is_not(key):
    """Near-identical labels in different categories: the descriptions are the
    only thing telling an operator which is which, so each has to name the
    other rather than describing itself alone."""
    description = SETTINGS_REGISTRY[key].description.lower()

    assert "knowledge_search" in description
    assert "not one inside the other" in description or "backstop" in description
