"""Verify contextvars set in before_request propagate into asyncio.create_task'd
sub-coroutines (the agent loop spawns these for sub-agent runs).

Also verifies the spec's documented snapshot-at-creation semantics: changes
to the parent contextvar AFTER create_task() do NOT affect the already-spawned task."""

import asyncio
import threading
import pytest

from agentic_project_service.services.billing_cloud.identity import (
    current_byok_providers,
    current_call_recoupable,
    recoupable_llm_call,
)
from agentic_project_service.services.run_context import run_id_var


@pytest.fixture(autouse=True)
def reset_byok_providers():
    yield
    current_byok_providers.set(frozenset())


@pytest.mark.asyncio
async def test_contextvars_propagate_to_create_task():
    current_byok_providers.set(frozenset({"openai"}))
    token = run_id_var.set("run-1")
    captured = {}

    async def sub_coroutine():
        captured["byok"] = current_byok_providers.get()
        captured["run_id"] = run_id_var.get()

    try:
        await asyncio.create_task(sub_coroutine())
        assert captured["byok"] == frozenset({"openai"})
        assert captured["run_id"] == "run-1"
    finally:
        run_id_var.reset(token)


@pytest.mark.asyncio
async def test_contextvars_snapshot_at_create_task_time():
    """Mutating the parent's contextvar AFTER create_task does NOT affect the child."""
    current_byok_providers.set(frozenset({"parent-set"}))

    async def child():
        await asyncio.sleep(0.01)  # let parent mutate first
        return current_byok_providers.get()

    task = asyncio.create_task(child())
    current_byok_providers.set(frozenset({"parent-set-changed"}))
    result = await task

    # Snapshot at create_task() time = "parent-set"
    assert result == frozenset({"parent-set"})


def test_threading_thread_does_NOT_inherit_contextvars():
    """Per spec: threading.Thread starts a fresh context — contextvars don't propagate.
    Code paths that spawn worker threads need to manually carry context."""
    current_byok_providers.set(frozenset({"main-thread"}))
    captured = []

    def worker():
        captured.append(current_byok_providers.get())

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    # Worker thread sees the default (empty frozenset), NOT the main thread's value
    assert captured == [frozenset()]


def test_current_call_recoupable_does_NOT_propagate_to_thread():
    """current_call_recoupable obeys the same threading isolation as run_id_var.

    Important 7 from PR #440 review: if a worker thread were to inherit
    recoupable=True from a parent that's inside recoupable_llm_call(), every
    LLM call in that thread would incorrectly skip llm_call charges.

    threading.Thread starts with a fresh context (Python 3.7+ default), so
    the worker sees the default False regardless of what the main thread set.
    """
    captured = []

    with recoupable_llm_call():
        # Inside the wrap: main thread sees True
        assert current_call_recoupable.get() is True

        def worker():
            # Spawned INSIDE the with-block — if threading were to copy the
            # parent context, the worker would see True. It must see False.
            captured.append(current_call_recoupable.get())

        t = threading.Thread(target=worker)
        t.start()
        t.join()

    # Main thread restored to default after context manager exit
    assert current_call_recoupable.get() is False
    # Worker saw the default (False), NOT the main thread's in-block True
    assert captured == [False], f"Worker thread should NOT inherit recoupable=True; got {captured}"
