"""OOM-prevention tests for graph_enricher — mirrors test_enrichment_oom_prevention.py.

Each test isolates one defense ported from PR #440's metadata hardening:
retry cap, circuit breaker, as_completed batching.
"""

import logging
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


def _fake_response(content='{"referenced_nodes": []}'):
    return type(
        "R",
        (),
        {
            "choices": [
                type(
                    "C",
                    (),
                    {
                        "message": type("M", (), {"content": content})(),
                        "finish_reason": "stop",
                    },
                )()
            ],
            "usage": type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})(),
        },
    )()


@pytest.mark.asyncio
async def test_enrich_single_node_passes_zero_retries():
    """LiteLLM internal retries must be capped at the call site to bound memory
    under rate-limit storms (the #437 mechanism, unfixed in graph until #445)."""
    from agentic_project_service.services.graph_enricher import _enrich_single_node

    captured = {}

    async def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_response()

    with patch("litellm.acompletion", side_effect=_capture):
        await _enrich_single_node(
            node={"node_id": "0001", "title": "t", "text": "body"},
            toc_context="[0001] t",
            valid_node_ids={"0001"},
            children_map={},
            id_to_title={"0001": "t"},
            model="gpt-4o-mini",
        )

    assert captured.get("num_retries") == 0, f"got {captured.get('num_retries')!r}"
    assert captured.get("max_retries") == 0, f"got {captured.get('max_retries')!r}"
    assert captured.get("timeout") == 60, f"got {captured.get('timeout')!r}"


@pytest.mark.asyncio
async def test_on_batch_complete_invoked_per_batch_and_can_abort(monkeypatch):
    """enrich_referenced_nodes must call on_batch_complete after each batch and
    stop dispatching when the callback returns 'abort'."""
    import agentic_project_service.services.graph_enricher as ge
    from agentic.knowledge import model_config

    monkeypatch.setattr(model_config, "GRAPHINDEX_ENRICHMENT_BATCH_SIZE", 2, raising=False)
    monkeypatch.setattr(ge, "GRAPHINDEX_ENRICHMENT_BATCH_SIZE", 2, raising=False)

    nodes = [{"node_id": f"{i:04d}", "title": "t", "text": "b", "meta": {}} for i in range(6)]

    class FakeStore:
        def __init__(self, *a, **k):
            pass

        def get_all_nodes_for_toc(self, toc_id):
            return nodes

        def update_node_meta(self, *a, **k):
            pass

        def update_node_enrichment_error(self, *a, **k):
            pass

    monkeypatch.setattr(ge, "GraphIndexStore", FakeStore)

    async def _ok(**kwargs):
        return _fake_response('{"referenced_nodes": []}')

    calls = []

    def _cb(batch_ok, batch_item_ids):
        calls.append((batch_ok, list(batch_item_ids)))
        return "abort" if len(calls) == 2 else "continue"

    class FakeSession:
        def flush(self):
            pass

        def commit(self):
            pass

    with patch("litellm.acompletion", side_effect=_ok):
        await ge.enrich_referenced_nodes(
            db_session=FakeSession(),
            knowledge_base_id="kb-1",
            toc_id="toc-1",
            model="gpt-4o-mini",
            on_batch_complete=_cb,
        )

    # 6 nodes / batch_size 2 = 3 batches, but abort fires after batch 2 -> only 2 callbacks
    assert len(calls) == 2, f"expected 2 batches before abort, got {len(calls)}"
    assert calls[0][0] == 2  # batch_ok count for a fully-successful batch of 2


@pytest.mark.asyncio
async def test_circuit_breaker_trips_and_short_circuits(monkeypatch, caplog):
    """After 5 consecutive RateLimitErrors the breaker trips and remaining nodes
    short-circuit; a PLATFORM_LLM_QUOTA_EXHAUSTED alert is logged."""
    import logging
    import litellm.exceptions as llme
    import agentic_project_service.services.graph_enricher as ge

    nodes = [{"node_id": f"{i:04d}", "title": "t", "text": "b", "meta": {}} for i in range(30)]

    class FakeStore:
        def __init__(self, *a, **k):
            pass

        def get_all_nodes_for_toc(self, toc_id):
            return nodes

        def update_node_meta(self, *a, **k):
            pass

        def update_node_enrichment_error(self, *a, **k):
            pass

    monkeypatch.setattr(ge, "GraphIndexStore", FakeStore)

    call_count = {"n": 0}

    async def _always_rate_limited(**kwargs):
        call_count["n"] += 1
        raise llme.RateLimitError("quota", model=kwargs["model"], llm_provider="openai")

    class FakeSession:
        def flush(self):
            pass

        def commit(self):
            pass

    with (
        caplog.at_level(logging.ERROR),
        patch("litellm.acompletion", side_effect=_always_rate_limited),
    ):
        _, errors = await ge.enrich_referenced_nodes(
            db_session=FakeSession(),
            knowledge_base_id="kb-1",
            toc_id="toc-1",
            model="gpt-4o-mini",
        )

    from agentic_project_service.services.graph_enricher import _CIRCUIT_BREAKER_THRESHOLD
    from agentic.knowledge.model_config import GRAPHINDEX_ENRICHMENT_MAX_CONCURRENT

    assert (
        _CIRCUIT_BREAKER_THRESHOLD
        <= call_count["n"]
        <= _CIRCUIT_BREAKER_THRESHOLD + GRAPHINDEX_ENRICHMENT_MAX_CONCURRENT
    )
    assert any("PLATFORM_LLM_QUOTA_EXHAUSTED" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_circuit_breaker_degraded_alert_for_infra_errors(monkeypatch, caplog):
    """5xx/timeout/connection trips must emit PLATFORM_LLM_PROVIDER_DEGRADED, not
    the QUOTA alert — so on-call follows the provider-outage runbook, not billing."""
    import logging
    import litellm.exceptions as llme
    import agentic_project_service.services.graph_enricher as ge

    nodes = [{"node_id": f"{i:04d}", "title": "t", "text": "b", "meta": {}} for i in range(30)]

    class FakeStore:
        def __init__(self, *a, **k):
            pass

        def get_all_nodes_for_toc(self, toc_id):
            return nodes

        def update_node_meta(self, *a, **k):
            pass

        def update_node_enrichment_error(self, *a, **k):
            pass

    monkeypatch.setattr(ge, "GraphIndexStore", FakeStore)

    async def _always_unavailable(**kwargs):
        raise llme.ServiceUnavailableError("503", model=kwargs["model"], llm_provider="openai")

    class FakeSession:
        def flush(self):
            pass

        def commit(self):
            pass

    with (
        caplog.at_level(logging.ERROR),
        patch("litellm.acompletion", side_effect=_always_unavailable),
    ):
        await ge.enrich_referenced_nodes(
            db_session=FakeSession(),
            knowledge_base_id="kb-1",
            toc_id="toc-1",
            model="gpt-4o-mini",
        )

    assert any(
        "PLATFORM_LLM_PROVIDER_DEGRADED" in r.message for r in caplog.records
    ), "degraded alert missing"
    assert not any(
        "PLATFORM_LLM_QUOTA_EXHAUSTED" in r.message for r in caplog.records
    ), "must not misroute to quota alert"


@pytest.mark.asyncio
async def test_cancelled_error_propagates_not_swallowed(monkeypatch):
    """asyncio.CancelledError (Celery revoke/timeout) must propagate out of the
    batch loop, not be recorded as a per-node error string."""
    import asyncio
    import agentic_project_service.services.graph_enricher as ge

    nodes = [{"node_id": "0001", "title": "t", "text": "b", "meta": {}}]

    class FakeStore:
        def __init__(self, *a, **k):
            pass

        def get_all_nodes_for_toc(self, toc_id):
            return nodes

        def update_node_meta(self, *a, **k):
            pass

        def update_node_enrichment_error(self, *a, **k):
            pass

    monkeypatch.setattr(ge, "GraphIndexStore", FakeStore)

    async def _cancelled(**kwargs):
        raise asyncio.CancelledError()

    class FakeSession:
        def flush(self):
            pass

        def commit(self):
            pass

    with patch("litellm.acompletion", side_effect=_cancelled):
        with pytest.raises(asyncio.CancelledError):
            await ge.enrich_referenced_nodes(
                db_session=FakeSession(),
                knowledge_base_id="kb-1",
                toc_id="toc-1",
                model="gpt-4o-mini",
            )


@pytest.mark.asyncio
async def test_failed_nodes_log_warning_and_persist_real_error(monkeypatch, caplog):
    """Swallowed-exception regression: failed/short-circuited nodes must log a
    WARNING and persist the REAL error, not a silent generic placeholder."""
    import litellm.exceptions as llme
    import agentic_project_service.services.graph_enricher as ge

    nodes = [{"node_id": f"{i:04d}", "title": "t", "text": "b", "meta": {}} for i in range(30)]
    persisted_errors = []

    class FakeStore:
        def __init__(self, *a, **k):
            pass

        def get_all_nodes_for_toc(self, toc_id):
            return nodes

        def update_node_meta(self, *a, **k):
            pass

        def update_node_enrichment_error(self, toc_id, node_id, error):
            persisted_errors.append(error)

    monkeypatch.setattr(ge, "GraphIndexStore", FakeStore)

    async def _rate_limited(**kwargs):
        raise llme.RateLimitError("quota", model=kwargs["model"], llm_provider="openai")

    class FakeSession:
        def flush(self):
            pass

        def commit(self):
            pass

    with caplog.at_level(logging.WARNING), patch("litellm.acompletion", side_effect=_rate_limited):
        _, errors = await ge.enrich_referenced_nodes(
            db_session=FakeSession(),
            knowledge_base_id="kb-1",
            toc_id="toc-1",
            model="gpt-4o-mini",
        )

    assert any(
        "Circuit breaker tripped" in e for e in errors
    ), "real breaker message missing from errors"
    assert any(
        e and "Circuit breaker tripped" in e for e in persisted_errors
    ), "real error not persisted (got generic placeholder?)"
    assert not any(
        e == "Enrichment task raised an unhandled exception" for e in persisted_errors
    ), "generic placeholder persisted instead of real error"
    assert any(
        "enrichment task failed" in r.message.lower() for r in caplog.records
    ), "no WARNING logged for failed node"
