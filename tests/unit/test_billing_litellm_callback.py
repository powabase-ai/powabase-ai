"""Tests for BillingLogger — the LiteLLM CustomLogger that posts llm_call charges.
Identity comes from get_billing_context() (env-based); run_id from run_id_var."""

import hashlib
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_project_service.services.billing_cloud.identity import (
    current_byok_providers,
    recoupable_llm_call,
)
from agentic_project_service.services.billing_cloud.billing_litellm import BillingLogger
from agentic_project_service.services.run_context import run_id_var


@pytest.fixture(autouse=True)
def reset_state():
    yield
    current_byok_providers.set(frozenset())


@pytest.fixture
def enable_billing(monkeypatch):
    monkeypatch.setenv("BILLING_ENABLED", "true")
    monkeypatch.setenv("BILLING_AI_ON_US_ENABLED", "true")
    monkeypatch.setenv("BILLING_LLM_MARKUP_MULTIPLIER", "1.25")
    monkeypatch.setenv("BILLING_ORG_ID", "org-1")
    monkeypatch.setenv("PROJECT_ID", "proj-1")


def _make_kwargs(
    model="anthropic/claude-sonnet-4-6",
    response_cost=0.00375,
    call_type="completion",
    litellm_call_id="abc-123",
):
    return {
        "model": model,
        "response_cost": response_cost,
        "call_type": call_type,
        "litellm_call_id": litellm_call_id,
    }


def _make_response_obj(prompt_tokens=1000, completion_tokens=500):
    obj = MagicMock()
    obj.usage.prompt_tokens = prompt_tokens
    obj.usage.completion_tokens = completion_tokens
    return obj


@pytest.mark.asyncio
async def test_callback_charges_ai_on_us_call(enable_billing):
    logger = BillingLogger()
    token = run_id_var.set("run-1")
    try:
        with patch(
            "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
            new=AsyncMock(),
        ) as mock_post:
            await logger.async_log_success_event(_make_kwargs(), _make_response_obj(), 0, 1)
    finally:
        run_id_var.reset(token)
    assert mock_post.call_count == 1
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["action"] == "llm_call"
    assert call_kwargs["unit_credits"] == round(0.00375 * 1.25 * 100_000)
    assert call_kwargs["org_id"] == "org-1"
    assert call_kwargs["project_id"] == "proj-1"
    assert call_kwargs["metadata"]["run_id"] == "run-1"
    assert call_kwargs["metadata"]["model"] == "anthropic/claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_callback_idempotency_key_uses_litellm_call_id(enable_billing):
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
        new=AsyncMock(),
    ) as mock_post:
        await logger.async_log_success_event(
            _make_kwargs(litellm_call_id="uuid-deadbeef"), _make_response_obj(), 0, 1
        )
    expected = hashlib.sha256(b"org-1|llm_call|uuid-deadbeef").hexdigest()
    assert mock_post.call_args.kwargs["idempotency_key"] == expected


@pytest.mark.asyncio
async def test_callback_skip_when_billing_context_none(monkeypatch):
    monkeypatch.delenv("BILLING_ORG_ID", raising=False)
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
        new=AsyncMock(),
    ) as mock_post:
        await logger.async_log_success_event(_make_kwargs(), _make_response_obj(), 0, 1)
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_callback_skip_when_ai_on_us_flag_off(enable_billing, monkeypatch):
    monkeypatch.setenv("BILLING_AI_ON_US_ENABLED", "false")
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
        new=AsyncMock(),
    ) as mock_post:
        await logger.async_log_success_event(_make_kwargs(), _make_response_obj(), 0, 1)
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_callback_skip_embedding_calls(enable_billing):
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
        new=AsyncMock(),
    ) as mock_post:
        await logger.async_log_success_event(
            _make_kwargs(call_type="embedding"), _make_response_obj(), 0, 1
        )
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_callback_skip_when_byok_provider(enable_billing):
    current_byok_providers.set(frozenset({"anthropic"}))
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
        new=AsyncMock(),
    ) as mock_post:
        with recoupable_llm_call():
            await logger.async_log_success_event(
                _make_kwargs(model="anthropic/claude-sonnet-4-6"),
                _make_response_obj(),
                0,
                1,
            )
    mock_post.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        # Studio-facing form (what the call site passes in)
        "gemini/gemini-2.5-pro",
        # Production callback form (litellm strips the gemini/ prefix
        # before invoking the success hook, and at that point
        # get_llm_provider returns "vertex_ai" rather than "gemini" —
        # both names must alias to "google" or the BYOK skip won't fire)
        "gemini-2.5-pro",
    ],
)
async def test_callback_skip_when_byok_gemini_alias(enable_billing, model):
    """Regression: FE BYOK rows are stored under "google" (the only
    allowed name on the upsert endpoint), but litellm reports the call's
    provider as "gemini" (prefixed form) or "vertex_ai" (stripped form
    that BillingLogger actually sees). Without aliasing both to "google",
    a project with a stored google BYOK key would (a) have its key
    injected at call time so the user pays Google directly AND (b) get
    charged the AI-on-us markup — double-billed for Gemini.
    """
    current_byok_providers.set(frozenset({"google"}))
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
        new=AsyncMock(),
    ) as mock_post:
        with recoupable_llm_call():
            await logger.async_log_success_event(
                _make_kwargs(model=model),
                _make_response_obj(),
                0,
                1,
            )
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_callback_skip_when_response_cost_zero(enable_billing):
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
        new=AsyncMock(),
    ) as mock_post:
        await logger.async_log_success_event(
            _make_kwargs(response_cost=0.0), _make_response_obj(), 0, 1
        )
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_callback_skip_when_no_litellm_call_id(enable_billing):
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
        new=AsyncMock(),
    ) as mock_post:
        await logger.async_log_success_event(
            _make_kwargs(litellm_call_id=None), _make_response_obj(), 0, 1
        )
    mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# I7 — post_charge exceptions must not propagate (best-effort billing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callback_swallows_post_charge_exception(enable_billing, caplog):
    """A billing-service hiccup must NOT propagate into the LiteLLM dispatch.
    Charging is best-effort: log a warning and return cleanly."""
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
        new=AsyncMock(side_effect=ConnectionError("billing-service down")),
    ):
        with caplog.at_level(
            logging.WARNING, logger="agentic_project_service.services.billing_cloud.billing_litellm"
        ):
            # Must return without raising
            await logger.async_log_success_event(_make_kwargs(), _make_response_obj(), 0, 1)
    assert any("llm_call_post_charge_failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# C2 — sync LiteLLM hooks must also charge (agent.run / agent.stream paths)
# ---------------------------------------------------------------------------


def test_callback_charges_via_sync_log_success_event(enable_billing):
    """Sync litellm.completion path: log_success_event must dispatch a charge."""
    logger = BillingLogger()
    token = run_id_var.set("run-sync-1")
    try:
        with patch(
            "agentic_project_service.services.billing_cloud.billing_litellm._sync_post_charge",
        ) as mock_sync_post:
            logger.log_success_event(_make_kwargs(), _make_response_obj(), 0, 1)
    finally:
        run_id_var.reset(token)
    assert mock_sync_post.call_count == 1
    call_kwargs = mock_sync_post.call_args.kwargs
    assert call_kwargs["action"] == "llm_call"
    assert call_kwargs["org_id"] == "org-1"
    assert call_kwargs["project_id"] == "proj-1"
    # unit_credits is folded into metadata via the wrapper convention
    assert call_kwargs["metadata"]["unit_credits"] == round(0.00375 * 1.25 * 100_000)
    assert call_kwargs["metadata"]["run_id"] == "run-sync-1"
    assert call_kwargs["metadata"]["model"] == "anthropic/claude-sonnet-4-6"


def test_sync_callback_skip_when_byok_provider(enable_billing):
    """Sync hook honors BYOK skip (mirrors async path)."""
    current_byok_providers.set(frozenset({"anthropic"}))
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm._sync_post_charge",
    ) as mock_sync_post:
        with recoupable_llm_call():
            logger.log_success_event(
                _make_kwargs(model="anthropic/claude-sonnet-4-6"),
                _make_response_obj(),
                0,
                1,
            )
    mock_sync_post.assert_not_called()


@pytest.mark.parametrize(
    "model",
    ["gemini/gemini-2.5-pro", "gemini-2.5-pro"],
)
def test_sync_callback_skip_when_byok_gemini_alias(enable_billing, model):
    """Sync hook applies the same gemini→google + vertex_ai→google BYOK
    alias as the async path. Regression for the double-charge described
    above. Both the prefixed (Studio-facing) and stripped (production
    callback) forms must skip the charge.
    """
    current_byok_providers.set(frozenset({"google"}))
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm._sync_post_charge",
    ) as mock_sync_post:
        with recoupable_llm_call():
            logger.log_success_event(
                _make_kwargs(model=model),
                _make_response_obj(),
                0,
                1,
            )
    mock_sync_post.assert_not_called()


def test_sync_log_stream_event_does_not_charge(enable_billing, caplog):
    """M1: log_stream_event is dispatched per intermediate chunk by LiteLLM
    (litellm_core_utils/litellm_logging.py:2308 — ``if self.stream and
    complete_streaming_response is None: callback.log_stream_event(...)``).
    Final-chunk charging flows through log_success_event with
    complete_streaming_response. The stream-event hook must therefore be a
    no-op: no charge dispatch AND no unknown-cost WARN per chunk."""
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm._sync_post_charge",
    ) as mock_sync_post:
        with caplog.at_level(
            logging.WARNING, logger="agentic_project_service.services.billing_cloud.billing_litellm"
        ):
            logger.log_stream_event(_make_kwargs(), _make_response_obj(), 0, 1)
    mock_sync_post.assert_not_called()
    assert not any("llm_call_unknown_cost" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_async_log_stream_event_does_not_charge(enable_billing, caplog):
    """M1 async variant: async_log_stream_event must mirror log_stream_event —
    no per-chunk charge and no per-chunk WARN. Final-chunk async charging is
    handled by async_log_success_event with async_complete_streaming_response."""
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
        new=AsyncMock(),
    ) as mock_post:
        with caplog.at_level(
            logging.WARNING, logger="agentic_project_service.services.billing_cloud.billing_litellm"
        ):
            await logger.async_log_stream_event(_make_kwargs(), _make_response_obj(), 0, 1)
    mock_post.assert_not_called()
    assert not any("llm_call_unknown_cost" in rec.message for rec in caplog.records)


def test_sync_callback_swallows_post_charge_exception(enable_billing, caplog):
    """Sync hook: billing-service hiccup must not propagate (mirrors async I7)."""
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm._sync_post_charge",
        side_effect=ConnectionError("billing-service down"),
    ):
        with caplog.at_level(
            logging.WARNING, logger="agentic_project_service.services.billing_cloud.billing_litellm"
        ):
            # Must return without raising
            logger.log_success_event(_make_kwargs(), _make_response_obj(), 0, 1)
    assert any("llm_call_post_charge_failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# N2 — _markup() must handle set-but-empty BILLING_LLM_MARKUP_MULTIPLIER
# ---------------------------------------------------------------------------


def test_markup_falls_back_to_default_when_env_is_empty_string(monkeypatch):
    """Docker-mode `.env` rendering produces ``BILLING_LLM_MARKUP_MULTIPLIER=``
    (set-but-empty) when the CP host env is unset, because the docker-compose
    template uses ``${VAR:-}`` and the provisioner template_content stuffs an
    empty string into the placeholder. ``os.environ.get(KEY, default)`` returns
    the default ONLY when KEY is ABSENT; set-but-empty returns ``""``, and
    ``float("")`` raises ValueError. The fix must coerce empty to default."""
    monkeypatch.setenv("BILLING_LLM_MARKUP_MULTIPLIER", "")
    from agentic_project_service.services.billing_cloud.billing_litellm import _markup

    assert _markup() == 1.25


def test_markup_falls_back_to_default_when_env_unset(monkeypatch):
    """Belt-and-suspenders: unset env still resolves to 1.25."""
    monkeypatch.delenv("BILLING_LLM_MARKUP_MULTIPLIER", raising=False)
    from agentic_project_service.services.billing_cloud.billing_litellm import _markup

    assert _markup() == 1.25


def test_markup_honors_explicit_value(monkeypatch):
    monkeypatch.setenv("BILLING_LLM_MARKUP_MULTIPLIER", "1.5")
    from agentic_project_service.services.billing_cloud.billing_litellm import _markup

    assert _markup() == 1.5


# ---------------------------------------------------------------------------
# N17 — _markup() must reject operator typos (0, negative, huge, malformed)
# ---------------------------------------------------------------------------
#
# Reviewer N17: ``BILLING_LLM_MARKUP_MULTIPLIER=0`` makes ``float("0") ==
# 0.0`` and ``round(response_cost * 0 * 100_000) == 0`` — every charge
# silently bills 0 millicents. Add bounds [0.5, 5.0] with WARN + fallback
# so an operator typo cannot silently zero out revenue. 1.0 (no markup)
# is allowed; obvious typos (0, negative, huge, malformed strings) fall
# back to the 1.25 default.


def test_markup_zero_falls_back_to_default(monkeypatch, caplog):
    """BILLING_LLM_MARKUP_MULTIPLIER=0 → users charged 0 millicents on
    every llm_call. Must reject + fall back to 1.25 with a WARN."""
    monkeypatch.setenv("BILLING_LLM_MARKUP_MULTIPLIER", "0")
    from agentic_project_service.services.billing_cloud.billing_litellm import _markup

    with caplog.at_level(
        logging.WARNING, logger="agentic_project_service.services.billing_cloud.billing_litellm"
    ):
        assert _markup() == 1.25
    assert any("billing_markup_out_of_range" in rec.message for rec in caplog.records)


def test_markup_negative_falls_back_to_default(monkeypatch, caplog):
    """A negative markup is nonsensical (would credit users, not charge)
    — must reject + fall back."""
    monkeypatch.setenv("BILLING_LLM_MARKUP_MULTIPLIER", "-1.0")
    from agentic_project_service.services.billing_cloud.billing_litellm import _markup

    with caplog.at_level(
        logging.WARNING, logger="agentic_project_service.services.billing_cloud.billing_litellm"
    ):
        assert _markup() == 1.25
    assert any("billing_markup_out_of_range" in rec.message for rec in caplog.records)


def test_markup_huge_falls_back_to_default(monkeypatch, caplog):
    """A wildly-large markup (e.g. 100x) is almost certainly a typo
    (misplaced decimal). Reject + fall back."""
    monkeypatch.setenv("BILLING_LLM_MARKUP_MULTIPLIER", "100")
    from agentic_project_service.services.billing_cloud.billing_litellm import _markup

    with caplog.at_level(
        logging.WARNING, logger="agentic_project_service.services.billing_cloud.billing_litellm"
    ):
        assert _markup() == 1.25
    assert any("billing_markup_out_of_range" in rec.message for rec in caplog.records)


def test_markup_invalid_string_falls_back_to_default(monkeypatch, caplog):
    """Malformed numeric (operator typo like ``1.5x``) → ValueError on
    float(). Catch defensively and fall back."""
    monkeypatch.setenv("BILLING_LLM_MARKUP_MULTIPLIER", "1.5x")
    from agentic_project_service.services.billing_cloud.billing_litellm import _markup

    with caplog.at_level(
        logging.WARNING, logger="agentic_project_service.services.billing_cloud.billing_litellm"
    ):
        assert _markup() == 1.25
    assert any("billing_markup_invalid" in rec.message for rec in caplog.records)


def test_markup_lower_bound_1_0_is_allowed(monkeypatch):
    """1.0 (no markup) is a legitimate operator choice — must NOT fall back."""
    monkeypatch.setenv("BILLING_LLM_MARKUP_MULTIPLIER", "1.0")
    from agentic_project_service.services.billing_cloud.billing_litellm import _markup

    assert _markup() == 1.0


def test_markup_at_lower_bound_0_5_is_allowed(monkeypatch):
    """0.5 is the inclusive lower bound — must be accepted."""
    monkeypatch.setenv("BILLING_LLM_MARKUP_MULTIPLIER", "0.5")
    from agentic_project_service.services.billing_cloud.billing_litellm import _markup

    assert _markup() == 0.5


def test_markup_at_upper_bound_5_0_is_allowed(monkeypatch):
    """5.0 is the inclusive upper bound — must be accepted."""
    monkeypatch.setenv("BILLING_LLM_MARKUP_MULTIPLIER", "5.0")
    from agentic_project_service.services.billing_cloud.billing_litellm import _markup

    assert _markup() == 5.0


def test_markup_3_0_in_range_is_allowed(monkeypatch):
    """Mid-range value (3x markup) — must be accepted."""
    monkeypatch.setenv("BILLING_LLM_MARKUP_MULTIPLIER", "3.0")
    from agentic_project_service.services.billing_cloud.billing_litellm import _markup

    assert _markup() == 3.0


# ---------------------------------------------------------------------------
# N3 — _build_charge_args failure must not escape the LiteLLM hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_callback_swallows_build_charge_args_exception(enable_billing, caplog):
    """When ``response_obj.usage`` is None (some Bedrock variants, image-only
    models), ``response_obj.usage.prompt_tokens`` raises AttributeError inside
    _build_charge_args. The hook must catch the exception so it doesn't
    propagate into LiteLLM's outer dispatch loop."""
    logger = BillingLogger()
    bad_response_obj = MagicMock()
    bad_response_obj.usage = None  # AttributeError on .prompt_tokens
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
        new=AsyncMock(),
    ) as mock_post:
        with caplog.at_level(
            logging.WARNING, logger="agentic_project_service.services.billing_cloud.billing_litellm"
        ):
            # Must return without raising
            await logger.async_log_success_event(_make_kwargs(), bad_response_obj, 0, 1)
    mock_post.assert_not_called()
    assert any("llm_call_post_charge_failed" in rec.message for rec in caplog.records)


def test_sync_callback_swallows_build_charge_args_exception(enable_billing, caplog):
    """Sync mirror of the async N3 test."""
    logger = BillingLogger()
    bad_response_obj = MagicMock()
    bad_response_obj.usage = None
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm._sync_post_charge",
    ) as mock_sync_post:
        with caplog.at_level(
            logging.WARNING, logger="agentic_project_service.services.billing_cloud.billing_litellm"
        ):
            logger.log_success_event(_make_kwargs(), bad_response_obj, 0, 1)
    mock_sync_post.assert_not_called()
    assert any("llm_call_post_charge_failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# N7 — byok_lookup_degraded flag flows into ledger metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_byok_degraded_flag_in_charge_metadata_when_set(enable_billing):
    """When the before_request BYOK lookup failed (I14 catch-all swallow),
    the byok_lookup_degraded contextvar is True for the request and the
    BillingLogger must surface the flag in the charge metadata so ops can
    reconcile any over-billed AI-on-us calls post-hoc."""
    from agentic_project_service.services.billing_cloud.identity import byok_lookup_degraded

    token = byok_lookup_degraded.set(True)
    try:
        logger = BillingLogger()
        with patch(
            "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
            new=AsyncMock(),
        ) as mock_post:
            await logger.async_log_success_event(_make_kwargs(), _make_response_obj(), 0, 1)
    finally:
        byok_lookup_degraded.reset(token)
    assert mock_post.call_count == 1
    assert mock_post.call_args.kwargs["metadata"]["byok_lookup_degraded"] is True


@pytest.mark.asyncio
async def test_byok_degraded_flag_defaults_false(enable_billing):
    """Healthy path: byok_lookup_degraded defaults False and is still emitted
    in metadata so the schema is stable for ops dashboards."""
    logger = BillingLogger()
    with patch(
        "agentic_project_service.services.billing_cloud.billing_litellm.post_charge",
        new=AsyncMock(),
    ) as mock_post:
        await logger.async_log_success_event(_make_kwargs(), _make_response_obj(), 0, 1)
    assert mock_post.call_args.kwargs["metadata"]["byok_lookup_degraded"] is False
