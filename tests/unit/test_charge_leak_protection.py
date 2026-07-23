"""Charge-leak protection tests for Part 4 of #437 fix.

Verifies the pre-checks scale with actual work size and that per-batch
post_charge correctly aborts a run when billing returns 402.
"""

import logging
from unittest import mock
from unittest.mock import patch

import litellm.exceptions as llme
import pytest

from agentic_project_service.services.metadata_enricher import MetadataEnricher
from tests.unit.conftest import _FakeDbSession


@pytest.fixture(autouse=True)
def _stub_byok_resolver():
    """The BYOK-unification refactor (2026-06-03) wraps every LLM call
    in ``with_byok_and_recoup`` which calls ``get_all_user_provider_keys``
    → SQLAlchemy session → needs a Flask app context. Unit tests in
    this file mock litellm directly and don't spin up the app —
    short-circuit the resolver to return ``{}`` so the wrapper yields
    ``None`` (no BYOK injection) and the test path is otherwise
    unchanged.
    """
    with patch(
        "agentic_project_service.services.llm_call.get_all_user_provider_keys",
        return_value={},
    ):
        yield


def test_enrich_pre_check_scales_with_kb_size(recording_billing):
    """Pre-check must estimate cost from count_total_items, not a constant."""
    from agentic_project_service.routes import enrichment as enrichment_route

    with patch.object(enrichment_route, "_count_enrichable_items_for_kb", return_value=1000):
        enrichment_route._enrich_check_balance(kb_id="kb-test")

    # For 1000 items @ _ENRICHMENT_PER_ITEM_MAX_CREDITS=60, expect 60_000 credits
    assert recording_billing.balance_checks == [60_000], (
        f"Pre-check should scale with KB size; got {recording_billing.balance_checks} "
        f"for 1000-item KB"
    )


def test_enrich_pre_check_falls_back_when_kb_empty(recording_billing):
    """Fallback constant should fire when count returns 0 (new KB, first enrichment)."""
    from agentic_project_service.routes import enrichment as enrichment_route

    with patch.object(enrichment_route, "_count_enrichable_items_for_kb", return_value=0):
        enrichment_route._enrich_check_balance(kb_id="kb-test")

    assert recording_billing.balance_checks == [enrichment_route._ENRICHMENT_FALLBACK_ESTIMATE]


@pytest.mark.asyncio
async def test_per_batch_charging_aborts_on_402():
    """When billing returns 402 mid-job, subsequent batches must not dispatch."""
    from agentic_project_service.services.metadata_enricher import MetadataEnricher
    from .conftest import _FakeDbSession
    from unittest import mock

    items_enriched: list[str] = []

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

    async def _track(**kwargs):
        items_enriched.append(kwargs.get("messages", [{}])[0].get("content", ""))
        return fake_response

    batch_count = [0]

    def _callback(batch_ok: int, batch_item_ids: list[str]) -> str:
        batch_count[0] += 1
        return "abort" if batch_count[0] == 2 else "continue"

    enricher = MetadataEnricher(db_session=_FakeDbSession(), knowledge_base_id="kb-test")
    enricher.get_enrichable_items = lambda strategy, include_source_info=False: [
        {"id": f"item-{i:04d}", "text": f"text-{i}", "item_type": "chunk"} for i in range(150)
    ]
    enricher._ensure_error_column = lambda table_name: None
    enricher.store_result = lambda **kwargs: None

    with mock.patch("litellm.acompletion", side_effect=_track):
        await enricher.run_enrichment(
            fields=[{"name": "name", "type": "text", "description": "name"}],
            model="gpt-4o-mini",
            strategy="chunk_embed",
            table_name="test_table",
            incremental=False,
            on_batch_complete=_callback,
        )

    # 150 items @ batch=50 → 3 batches. Callback aborts after batch 2.
    # Items 0-49 (batch 1) + items 50-99 (batch 2) should be enriched.
    # Items 100-149 (batch 3) must NOT be dispatched.
    assert (
        len(items_enriched) == 100
    ), f"expected 100 items dispatched (2 batches) before abort; got {len(items_enriched)}"
    assert (
        batch_count[0] == 2
    ), f"callback should fire exactly twice (after batches 1 and 2); got {batch_count[0]}"


@pytest.mark.asyncio
async def test_circuit_breaker_emits_ops_alert(caplog):
    """When circuit breaker trips inside run_enrichment (always platform key),
    a high-severity log record with the marker 'PLATFORM_LLM_QUOTA_EXHAUSTED'
    must be emitted exactly once per provider per run."""
    import logging
    from agentic_project_service.services.metadata_enricher import MetadataEnricher
    from .conftest import _FakeDbSession
    from unittest import mock
    import litellm.exceptions as llme

    async def _always_rate_limit(**kwargs):
        raise llme.RateLimitError("quota exceeded", model=kwargs["model"], llm_provider="openai")

    enricher = MetadataEnricher(db_session=_FakeDbSession(), knowledge_base_id="kb-test")
    enricher.get_enrichable_items = lambda strategy, include_source_info=False: [
        {"id": f"item-{i:04d}", "text": f"text-{i}", "item_type": "chunk"} for i in range(50)
    ]
    enricher._ensure_error_column = lambda table_name: None
    enricher.store_result = lambda **kwargs: None

    with (
        caplog.at_level(logging.ERROR),
        mock.patch("litellm.acompletion", side_effect=_always_rate_limit),
    ):
        await enricher.run_enrichment(
            fields=[{"name": "name", "type": "text", "description": "name"}],
            model="gpt-4o-mini",
            strategy="chunk_embed",
            table_name="test_table",
            incremental=False,
        )

    # Assert on message text, not extra-attr (F8: robust to LogFilter/LogAdapter;
    # also matches CloudWatch metric filter pattern)
    alert_records = [r for r in caplog.records if "PLATFORM_LLM_QUOTA_EXHAUSTED" in r.message]
    assert len(alert_records) == 1, (
        f"Expected exactly one alert log; got {len(alert_records)}. "
        f"Records: {[r.message for r in caplog.records]}"
    )
    assert "openai" in alert_records[0].message.lower()  # provider in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc_class,expected_marker",
    [
        # IMP-NEW-1 quota/billing causes → PLATFORM_LLM_QUOTA_EXHAUSTED
        ("RateLimitError", "PLATFORM_LLM_QUOTA_EXHAUSTED"),
        ("AuthenticationError", "PLATFORM_LLM_QUOTA_EXHAUSTED"),
        # IMP-NEW-5: infrastructure/outage causes → PLATFORM_LLM_PROVIDER_DEGRADED
        ("ServiceUnavailableError", "PLATFORM_LLM_PROVIDER_DEGRADED"),
        ("InternalServerError", "PLATFORM_LLM_PROVIDER_DEGRADED"),
        ("BadGatewayError", "PLATFORM_LLM_PROVIDER_DEGRADED"),
        ("APIConnectionError", "PLATFORM_LLM_PROVIDER_DEGRADED"),
        ("Timeout", "PLATFORM_LLM_PROVIDER_DEGRADED"),
    ],
)
async def test_circuit_breaker_alert_label_by_trip_cause(caplog, exc_class, expected_marker):
    """IMP-NEW-1/IMP-NEW-5: circuit breaker emits the correct alert marker based on
    the exception that caused the trip.

    RateLimitError/AuthenticationError → PLATFORM_LLM_QUOTA_EXHAUSTED
    ServiceUnavailableError/InternalServerError/BadGatewayError/APIConnectionError/Timeout
    → PLATFORM_LLM_PROVIDER_DEGRADED

    Wrong label → on-call follows wrong runbook (quota fix vs provider-status check).
    """
    exc_cls = getattr(llme, exc_class)

    async def _always_raise(**kwargs):
        # litellm exception constructors differ by class; use a safe approach
        try:
            raise exc_cls("simulated error", model=kwargs["model"], llm_provider="openai")
        except TypeError:
            raise exc_cls("simulated error")  # noqa: B904

    enricher = MetadataEnricher(db_session=_FakeDbSession(), knowledge_base_id="kb-test")
    enricher.get_enrichable_items = lambda strategy, include_source_info=False: [
        {"id": f"item-{i:04d}", "text": f"text-{i}", "item_type": "chunk"} for i in range(50)
    ]
    enricher._ensure_error_column = lambda table_name: None
    enricher.store_result = lambda **kwargs: None

    with (
        caplog.at_level(logging.ERROR),
        mock.patch("litellm.acompletion", side_effect=_always_raise),
    ):
        await enricher.run_enrichment(
            fields=[{"name": "name", "type": "text", "description": "name"}],
            model="gpt-4o-mini",
            strategy="chunk_embed",
            table_name="test_table",
            incremental=False,
        )

    # Exactly one alert with the expected marker, none with the wrong one
    expected_records = [r for r in caplog.records if expected_marker in r.message]
    wrong_marker = (
        "PLATFORM_LLM_PROVIDER_DEGRADED"
        if expected_marker == "PLATFORM_LLM_QUOTA_EXHAUSTED"
        else "PLATFORM_LLM_QUOTA_EXHAUSTED"
    )
    wrong_records = [r for r in caplog.records if wrong_marker in r.message]
    assert len(expected_records) == 1, (
        f"Expected 1 record with {expected_marker}, got {len(expected_records)}. "
        f"exc_class={exc_class}. Records: {[r.message for r in caplog.records]}"
    )
    assert len(wrong_records) == 0, (
        f"Expected 0 records with wrong marker {wrong_marker}, got {len(wrong_records)}. "
        f"exc_class={exc_class}"
    )
    # Verify the extra={"alert": ...} kwarg was passed: Python logger merges extra
    # dict into LogRecord attributes directly (record.alert, not record.extra).
    expected_alert_tag = (
        "platform_llm_quota_exhausted"
        if expected_marker == "PLATFORM_LLM_QUOTA_EXHAUSTED"
        else "platform_llm_provider_degraded"
    )
    assert getattr(expected_records[0], "alert", None) == expected_alert_tag, (
        f"expected_records[0].alert should be {expected_alert_tag!r}; "
        f"got {getattr(expected_records[0], 'alert', None)!r}"
    )


@pytest.mark.asyncio
async def test_circuit_breaker_short_circuit_reraises_true_cause():
    """IMP-NEW-3: items short-circuited after the breaker trips must raise the
    same exception class that caused the trip, not always RateLimitError.

    Counterfactual: pre-fix, every short-circuited item raised RateLimitError
    regardless of the trip cause. After the fix, tripping with ServiceUnavailableError
    causes subsequent items to also raise ServiceUnavailableError.
    """
    call_count = [0]

    async def _service_unavailable_then_ok(**kwargs):
        call_count[0] += 1
        raise llme.ServiceUnavailableError(
            "upstream 503", model=kwargs["model"], llm_provider="openai"
        )

    # Use 50 items so the breaker (K=5) trips and remaining items short-circuit
    errors_seen: list[str] = []

    enricher = MetadataEnricher(db_session=_FakeDbSession(), knowledge_base_id="kb-test")
    enricher.get_enrichable_items = lambda strategy, include_source_info=False: [
        {"id": f"item-{i:04d}", "text": f"text-{i}", "item_type": "chunk"} for i in range(50)
    ]
    enricher._ensure_error_column = lambda table_name: None

    def capture_error(**kwargs):
        if kwargs.get("error"):
            errors_seen.append(kwargs["error"])

    enricher.store_result = capture_error

    with mock.patch("litellm.acompletion", side_effect=_service_unavailable_then_ok):
        await enricher.run_enrichment(
            fields=[{"name": "name", "type": "text", "description": "name"}],
            model="gpt-4o-mini",
            strategy="chunk_embed",
            table_name="test_table",
            incremental=False,
        )

    # After the breaker trips, short-circuited items should show ServiceUnavailableError
    # in their error messages (not "RateLimitError" from the old hardcoded raise)
    assert len(errors_seen) > 0, "Expected some items to have errors"
    short_circuit_errors = [e for e in errors_seen if "Circuit breaker tripped" in e]
    assert len(short_circuit_errors) > 0, "Expected some short-circuit errors"
    # The short-circuit errors must NOT say 'RateLimitError' in the class name
    # (old behavior was to always raise llme.RateLimitError with the trip message)
    for err in short_circuit_errors:
        assert (
            "ServiceUnavailableError" in err or "ServiceUnavailable" in err
        ), f"Short-circuit error should be ServiceUnavailableError type; got: {err!r}"


def test_enrich_pre_check_fails_closed_on_lookup_error(recording_billing):
    """When the underlying item-count lookup raises, pre-check must raise
    ServiceUnavailable (503) — not fall back to the old constant.

    Important 3 from PR #440 review: a DB timeout during pre-check was
    silently falling back to the 200-credit constant, transiently restoring
    the very behavior the PR replaces.

    The fail-closed logic lives inside _count_enrichable_items_for_kb itself:
    its except-block converts any exception into ServiceUnavailable.  We patch
    the INNER dependency _get_kb_strategy (called first inside
    _count_enrichable_items_for_kb) so the real except-clause is exercised.
    This is load-bearing: reverting the `raise ServiceUnavailable` production
    fix to `return 0` causes this test to FAIL (the caller sees 0, not 503).

    Mirrors test_workflow_pre_check_fails_closed_on_load_blocks_error which
    patches load_blocks (the inner raiser) rather than _workflow_pre_check.
    """
    from agentic_project_service.routes import enrichment as enrichment_route
    from werkzeug.exceptions import ServiceUnavailable

    # Patch the inner dependency so _count_enrichable_items_for_kb's own
    # except-block runs and converts to ServiceUnavailable.
    # _get_kb_strategy is imported from tasks.enrichment inside the try block,
    # so we patch it at its source module.
    with patch(
        "agentic_project_service.tasks.enrichment._get_kb_strategy",
        side_effect=RuntimeError("DB timeout"),
    ):
        try:
            enrichment_route._enrich_check_balance(kb_id="kb-test")
            assert False, "Expected ServiceUnavailable to be raised"
        except ServiceUnavailable:
            pass  # correct — fail closed
    assert recording_billing.balance_checks == []  # 503 must fire before balance check


def test_workflow_pre_check_fails_closed_on_load_blocks_error():
    """When load_blocks raises, _workflow_pre_check must raise ServiceUnavailable.

    Important 3 from PR #440 review: pre-fix fell back to the 2000-credit
    constant on any load_blocks failure.
    """
    from agentic_project_service.routes import workflows as workflows_route
    from werkzeug.exceptions import ServiceUnavailable

    with patch.object(
        workflows_route,
        "load_blocks",
        side_effect=RuntimeError("DB timeout"),
    ):
        try:
            workflows_route._workflow_pre_check(workflow_id="wf-test")
            assert False, "Expected ServiceUnavailable to be raised"
        except ServiceUnavailable:
            pass  # correct — fail closed


def test_workflow_pre_check_scales_with_block_count(recording_billing):
    """Pre-check must estimate cost from len(load_blocks(workflow_id)), not a constant."""
    from agentic_project_service.routes import workflows as workflows_route

    fake_blocks = [{"id": f"b{i}", "type": "llm_completion"} for i in range(50)]

    with patch.object(workflows_route, "load_blocks", return_value=fake_blocks):
        workflows_route._workflow_pre_check(workflow_id="wf-test")

    # Expect exactly 50 blocks × _WORKFLOW_PER_BLOCK_MAX_CREDITS (>= 2000 fallback).
    # Pin to == so any change to the formula is caught explicitly.
    assert recording_billing.balance_checks == [
        50 * workflows_route._WORKFLOW_PER_BLOCK_MAX_CREDITS
    ]


def test_extraction_pre_check_uses_conservative_page_estimate():
    """Bump from 10 to 100 reduces under-estimate risk on large PDFs."""
    from agentic_project_service.routes import sources as sources_route

    assert sources_route._EXTRACTION_ESTIMATED_PAGES >= 100, (
        f"Pre-check estimate too low ({sources_route._EXTRACTION_ESTIMATED_PAGES}); "
        f"large PDFs (200+ pages) routinely exceed 10 pages and cause platform leak"
    )


# Note: the consecutive-empty-batch-abort tests that used to live here
# (K=2 circuit breaker on make_per_batch_billing_callback) exercised the
# private services.billing_cloud.credits_client adapter, which this OSS
# build excludes. Removed along with the rest of the billing_cloud test
# suite — see the billing_port facade tests for the OSS-shipped equivalent.
