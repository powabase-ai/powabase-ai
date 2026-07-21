"""Verify kwargs["litellm_call_id"] is populated for streaming, non-streaming, tool calls.
If LiteLLM ever changes this, the BillingLogger's no-id skip would silently swallow charges."""

import asyncio
import litellm
import pytest


@pytest.fixture(autouse=True)
def reset_litellm_callbacks():
    # LiteLLM lazily mirrors `litellm.callbacks` into the internal
    # `_async_success_callback` list at first invocation, deduplicating by
    # class name. Across tests we define a local `class CaptureLogger` in each
    # function — they all share the class name "CaptureLogger" so the dedup
    # blocks new instances from being registered, and stale instances from
    # prior tests keep firing into dead capture dicts. Reset every list LiteLLM
    # consults so each test starts from a clean slate.
    saved_callbacks = list(litellm.callbacks)
    saved_success = list(litellm.success_callback)
    saved_failure = list(litellm.failure_callback)
    saved_input = list(litellm.input_callback)
    saved_async_success = list(litellm._async_success_callback)
    saved_async_failure = list(litellm._async_failure_callback)
    litellm.callbacks = []
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.input_callback = []
    litellm._async_success_callback = []
    litellm._async_failure_callback = []
    yield
    litellm.callbacks = saved_callbacks
    litellm.success_callback = saved_success
    litellm.failure_callback = saved_failure
    litellm.input_callback = saved_input
    litellm._async_success_callback = saved_async_success
    litellm._async_failure_callback = saved_async_failure


# LiteLLM 1.83.14 dispatches success callbacks with kwargs= and response_obj=
# as keyword arguments (plus start_time= / end_time=). The CaptureLogger
# methods accept **kw to remain robust to LiteLLM signature drift — matches the
# pattern used by BillingLogger's production callback.


@pytest.mark.asyncio
async def test_non_streaming_completion_has_litellm_call_id():
    captured_kwargs = {}

    class CaptureLogger(litellm.integrations.custom_logger.CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, **_):
            captured_kwargs.update(kwargs)

    litellm.callbacks = [CaptureLogger()]
    await litellm.acompletion(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="mock response",
    )
    await asyncio.sleep(0.1)
    assert "litellm_call_id" in captured_kwargs
    assert captured_kwargs["litellm_call_id"]


@pytest.mark.asyncio
async def test_streaming_completion_has_litellm_call_id():
    captured_kwargs = {}

    class CaptureLogger(litellm.integrations.custom_logger.CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, **_):
            captured_kwargs.update(kwargs)

    litellm.callbacks = [CaptureLogger()]
    response = await litellm.acompletion(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        mock_response="streamed response",
    )
    # Consume the stream
    async for _ in response:
        pass
    await asyncio.sleep(0.1)
    assert (
        "litellm_call_id" in captured_kwargs
    ), f"Streaming did not populate litellm_call_id; got keys: {list(captured_kwargs.keys())}"
    assert captured_kwargs["litellm_call_id"]


@pytest.mark.asyncio
async def test_multiple_completions_get_unique_litellm_call_ids():
    """3 sequential completion calls — each must get a unique litellm_call_id."""
    captured_ids = []

    class CaptureLogger(litellm.integrations.custom_logger.CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, **_):
            captured_ids.append(kwargs.get("litellm_call_id"))

    litellm.callbacks = [CaptureLogger()]
    for _ in range(3):
        await litellm.acompletion(
            model="anthropic/claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="ok",
        )
        await asyncio.sleep(0.05)

    assert len(captured_ids) == 3, f"Expected 3 callback fires, got {len(captured_ids)}"
    assert len(set(captured_ids)) == 3, f"litellm_call_ids not unique: {captured_ids}"
