"""The two context-token budgets are related, and the relationship is easy to
break by editing one of them.

``KB_DEFAULT_MAX_CONTEXT_TOKENS`` bounds a single ``knowledge_search`` result.
``DEFAULT_MAX_CONTEXT_TOKENS`` bounds the retrieval context an agent run
assembles. A search result has to fit inside the context a run builds, so the
KB budget must not exceed the agent one — and both settings' descriptions now
tell operators exactly that.
"""

from __future__ import annotations

from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY


def _default(key: str) -> int:
    return SETTINGS_REGISTRY[key].default


def test_a_search_result_fits_inside_an_agent_run_context():
    assert _default("KB_DEFAULT_MAX_CONTEXT_TOKENS") <= _default("DEFAULT_MAX_CONTEXT_TOKENS")


def test_both_budgets_stay_within_their_own_bounds():
    """A default outside its own min/max would be silently unreachable through
    the settings API."""
    for key in ("KB_DEFAULT_MAX_CONTEXT_TOKENS", "DEFAULT_MAX_CONTEXT_TOKENS"):
        setting = SETTINGS_REGISTRY[key]
        assert setting.min <= setting.default <= setting.max, key


def test_the_budgets_explain_how_they_differ():
    """They have near-identical labels and sit in different categories, so the
    descriptions are the only thing telling an operator which is which."""
    for key in ("KB_DEFAULT_MAX_CONTEXT_TOKENS", "DEFAULT_MAX_CONTEXT_TOKENS"):
        description = SETTINGS_REGISTRY[key].description.lower()
        assert "knowledge_search" in description, key
