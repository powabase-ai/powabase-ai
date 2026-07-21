"""OOM prevention tests for metadata enricher.

Each test isolates one defense: retry cap, circuit breaker, as_completed streaming,
error classification.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _stub_byok_resolver():
    """See test_query_enrichment_billing.py for the rationale. The
    BYOK-unification refactor (2026-06-03) wraps LLM calls in
    ``with_byok_and_recoup`` which needs a Flask app context for the
    resolver's DB query — short-circuited here so unit tests stay
    DB-free.
    """
    with patch(
        "agentic_project_service.services.llm_call.get_all_user_provider_keys",
        return_value={},
    ):
        yield


@pytest.mark.asyncio
async def test_enrich_single_item_passes_zero_retries():
    """LiteLLM internal retries must be capped at the call site to bound memory
    under rate-limit storms. The outer 3-attempt JSON-parse loop handles
    legitimate parse failures separately.
    """
    from agentic_project_service.services.metadata_enricher import MetadataEnricher

    fake_response = type(
        "R",
        (),
        {
            "choices": [
                type(
                    "C",
                    (),
                    {
                        "message": type("M", (), {"content": '{"name": "test"}'})(),
                        "finish_reason": "stop",
                    },
                )()
            ],
        },
    )()

    captured_kwargs = {}

    async def _capture(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_response

    enricher = MetadataEnricher(db_session=None, knowledge_base_id="kb-test")
    with patch("litellm.acompletion", side_effect=_capture):
        await enricher.enrich_single_item(
            item_text="hello",
            fields=[{"name": "name", "type": "text", "description": "name"}],
            model="gpt-4o-mini",
        )

    assert (
        captured_kwargs.get("num_retries") == 0
    ), f"num_retries should be 0 to bound memory; got {captured_kwargs.get('num_retries')!r}"
    assert (
        captured_kwargs.get("max_retries") == 0
    ), f"max_retries (OpenAI SDK) should be 0; got {captured_kwargs.get('max_retries')!r}"
    assert (
        captured_kwargs.get("timeout") is not None
    ), "timeout must be set to bound in-flight memory per call"


@pytest.mark.asyncio
async def test_circuit_breaker_trips_after_threshold():
    """After _CIRCUIT_BREAKER_THRESHOLD consecutive RateLimitErrors from the same
    provider, subsequent items in the same run short-circuit (raise without
    dispatching to LiteLLM).

    Bound on dispatches BEFORE circuit trips:
      THRESHOLD <= call_count <= THRESHOLD + METADATA_ENRICHMENT_MAX_CONCURRENT
    Why a range (F2): Semaphore(MAX_CONCURRENT=10) allows 10 concurrent
    dispatches before any error propagates back to trip the breaker.
    """
    from agentic_project_service.services.metadata_enricher import (
        MetadataEnricher,
        _CIRCUIT_BREAKER_THRESHOLD,
    )
    from agentic.knowledge.model_config import METADATA_ENRICHMENT_MAX_CONCURRENT
    from .conftest import _FakeDbSession
    import litellm.exceptions as llme

    assert _CIRCUIT_BREAKER_THRESHOLD == 5  # contract

    call_count = 0

    async def _always_rate_limit(**kwargs):
        nonlocal call_count
        call_count += 1
        raise llme.RateLimitError("quota exceeded", model=kwargs["model"], llm_provider="openai")

    import unittest.mock as mock

    enricher = MetadataEnricher(db_session=_FakeDbSession(), knowledge_base_id="kb-test")

    # Stub the enricher's get_enrichable_items + _ensure_error_column + store_result methods
    enricher.get_enrichable_items = lambda strategy, include_source_info=False: [
        {"id": f"item-{i:04d}", "text": f"text-{i}", "item_type": "chunk"} for i in range(50)
    ]
    enricher._ensure_error_column = lambda table_name: None
    enricher.store_result = lambda **kwargs: None

    with mock.patch("litellm.acompletion", side_effect=_always_rate_limit):
        await enricher.run_enrichment(
            fields=[{"name": "name", "type": "text", "description": "name"}],
            model="gpt-4o-mini",
            strategy="chunk_embed",
            table_name="test_table",
            incremental=False,
        )

    assert (
        _CIRCUIT_BREAKER_THRESHOLD
        <= call_count
        <= _CIRCUIT_BREAKER_THRESHOLD + METADATA_ENRICHMENT_MAX_CONCURRENT
    ), (
        f"Expected dispatches in [{_CIRCUIT_BREAKER_THRESHOLD}, "
        f"{_CIRCUIT_BREAKER_THRESHOLD + METADATA_ENRICHMENT_MAX_CONCURRENT}]; "
        f"got {call_count}. If well above the upper bound, the circuit-breaker "
        f"check inside _enrich_one isn't firing."
    )


@pytest.mark.asyncio
async def test_run_enrichment_uses_as_completed():
    """Verify run_enrichment uses asyncio.as_completed (not gather) so each
    item's result is processed and freed before the batch completes.

    Detection: patch asyncio.as_completed to mark it was called.
    """
    import asyncio
    from agentic_project_service.services.metadata_enricher import MetadataEnricher
    from .conftest import _FakeDbSession
    from unittest import mock

    as_completed_called = [False]

    original_as_completed = asyncio.as_completed

    def _track(*args, **kwargs):
        as_completed_called[0] = True
        return original_as_completed(*args, **kwargs)

    enricher = MetadataEnricher(db_session=_FakeDbSession(), knowledge_base_id="kb-test")
    enricher.get_enrichable_items = lambda strategy, include_source_info=False: [
        {"id": f"item-{i:04d}", "text": f"text-{i}", "item_type": "chunk"} for i in range(5)
    ]
    enricher._ensure_error_column = lambda table_name: None
    enricher.store_result = lambda **kwargs: None

    fake_response = type(
        "R",
        (),
        {
            "choices": [
                type(
                    "C",
                    (),
                    {
                        "message": type("M", (), {"content": '{"name": "test"}'})(),
                        "finish_reason": "stop",
                    },
                )()
            ],
        },
    )()

    async def _ok(**kwargs):
        return fake_response

    with (
        mock.patch("litellm.acompletion", side_effect=_ok),
        mock.patch.object(asyncio, "as_completed", side_effect=_track),
    ):
        await enricher.run_enrichment(
            fields=[{"name": "name", "type": "text", "description": "name"}],
            model="gpt-4o-mini",
            strategy="chunk_embed",
            table_name="test_table",
            incremental=False,
        )

    assert as_completed_called[0], (
        "run_enrichment should use asyncio.as_completed to stream batch results "
        "instead of asyncio.gather (which retains O(batch_size) results in memory)"
    )


def test_classify_enrichment_failure_rate_limit():
    """RateLimitError → actionable user-facing message."""
    from agentic_project_service.tasks.enrichment import _classify_enrichment_failure
    import litellm.exceptions as llme

    exc = llme.RateLimitError("quota exceeded", model="gpt-4o-mini", llm_provider="openai")
    msg = _classify_enrichment_failure(exc)
    assert "rate-limited" in msg.lower() or "rate limit" in msg.lower()
    assert "platform team" in msg.lower() or "platform has been notified" in msg.lower()


def test_classify_enrichment_failure_auth():
    from agentic_project_service.tasks.enrichment import _classify_enrichment_failure
    import litellm.exceptions as llme

    exc = llme.AuthenticationError("bad key", model="gpt-4o-mini", llm_provider="openai")
    msg = _classify_enrichment_failure(exc)
    assert "key" in msg.lower() or "rejected" in msg.lower()


def test_classify_enrichment_failure_other_returns_sanitized_message():
    """Non-provider errors must return a generic user-facing message, NOT a traceback.

    Important 4 from PR #440 review: raw tracebacks leak file paths and internal
    class names to Studio. The full traceback is captured via logger.error(exc_info=True)
    at the caller site.
    """
    from agentic_project_service.tasks.enrichment import _classify_enrichment_failure

    try:
        raise RuntimeError("boom-specific-message-12345")
    except RuntimeError as e:
        msg = _classify_enrichment_failure(e)

    # Must NOT contain the raw traceback markers or internal paths.
    assert (
        "boom-specific-message-12345" not in msg
    ), "Traceback content leaked to user-facing message"
    assert "Traceback" not in msg, "Raw traceback must not appear in user-facing message"
    # Must be a generic actionable message.
    assert (
        "platform team" in msg.lower() or "unexpected error" in msg.lower()
    ), f"Expected generic user-facing message; got: {msg!r}"


@pytest.mark.asyncio
async def test_circuit_breaker_trips_on_service_unavailable():
    """ServiceUnavailableError (5xx class) must also trip the circuit breaker.

    Important 1 from PR #440 review: pre-fix only RateLimitError and
    AuthenticationError tripped the breaker; a multi-hour provider 5xx outage
    could exhaust all 1000+ items without triggering the short-circuit.
    """
    from agentic_project_service.services.metadata_enricher import (
        MetadataEnricher,
        _CIRCUIT_BREAKER_THRESHOLD,
    )
    from agentic.knowledge.model_config import METADATA_ENRICHMENT_MAX_CONCURRENT
    from .conftest import _FakeDbSession
    import litellm.exceptions as llme
    import unittest.mock as mock

    call_count = 0

    async def _always_service_unavailable(**kwargs):
        nonlocal call_count
        call_count += 1
        raise llme.ServiceUnavailableError(
            "provider down", model=kwargs["model"], llm_provider="openai"
        )

    enricher = MetadataEnricher(db_session=_FakeDbSession(), knowledge_base_id="kb-test")
    enricher.get_enrichable_items = lambda strategy, include_source_info=False: [
        {"id": f"item-{i:04d}", "text": f"text-{i}", "item_type": "chunk"} for i in range(50)
    ]
    enricher._ensure_error_column = lambda table_name: None
    enricher.store_result = lambda **kwargs: None

    with mock.patch("litellm.acompletion", side_effect=_always_service_unavailable):
        await enricher.run_enrichment(
            fields=[{"name": "name", "type": "text", "description": "name"}],
            model="gpt-4o-mini",
            strategy="chunk_embed",
            table_name="test_table",
            incremental=False,
        )

    # Circuit breaker should have tripped; dispatches bounded by threshold + concurrency.
    assert (
        _CIRCUIT_BREAKER_THRESHOLD
        <= call_count
        <= _CIRCUIT_BREAKER_THRESHOLD + METADATA_ENRICHMENT_MAX_CONCURRENT
    ), (
        f"ServiceUnavailableError should trip circuit breaker; got {call_count} dispatches "
        f"(expected in [{_CIRCUIT_BREAKER_THRESHOLD}, "
        f"{_CIRCUIT_BREAKER_THRESHOLD + METADATA_ENRICHMENT_MAX_CONCURRENT}])"
    )
