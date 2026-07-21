"""Tests for current_call_recoupable contextvar + recoupable_llm_call() ctx manager.

Verifies the marker that BillingLogger reads to decide whether to charge llm_call
for a given LiteLLM completion. Set by agent/orch/workflow paths around their
litellm calls; defaults False everywhere else (platform-internal paths).
"""

import asyncio

import pytest

from agentic_project_service.services.billing_cloud.identity import (
    current_call_recoupable,
    recoupable_llm_call,
)


def test_default_is_false():
    """Marker defaults False — platform-internal paths must not opt in by accident."""
    assert current_call_recoupable.get() is False


def test_ctx_manager_sets_true_during_body():
    assert current_call_recoupable.get() is False
    with recoupable_llm_call():
        assert current_call_recoupable.get() is True
    assert current_call_recoupable.get() is False


def test_ctx_manager_resets_on_exception():
    with pytest.raises(RuntimeError):
        with recoupable_llm_call():
            assert current_call_recoupable.get() is True
            raise RuntimeError("boom")
    assert current_call_recoupable.get() is False


def test_ctx_manager_nested():
    """Nested entry is a no-op (already True); exit restores prior value."""
    with recoupable_llm_call():
        with recoupable_llm_call():
            assert current_call_recoupable.get() is True
        assert current_call_recoupable.get() is True
    assert current_call_recoupable.get() is False


def test_asyncio_task_inherits():
    """asyncio.Task should inherit the marker from its parent context."""

    async def _inner():
        return current_call_recoupable.get()

    async def _outer():
        with recoupable_llm_call():
            return await asyncio.create_task(_inner())

    assert asyncio.run(_outer()) is True
