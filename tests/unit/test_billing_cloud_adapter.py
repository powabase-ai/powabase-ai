from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from agentic_project_service.services import billing_port
from agentic_project_service.services.billing_cloud.adapter import CloudBillingAdapter
from agentic_project_service.services.billing_cloud.identity import (
    byok_lookup_degraded,
    current_byok_providers,
)
from agentic_project_service.services.billing_port import ChargeOutcome, NoopBillingAdapter
from agentic_project_service.services.billing_cloud.credits_client import ChargeResult


@pytest.fixture(autouse=True)
def _restore_default_adapter():
    billing_port.set_billing_adapter(NoopBillingAdapter())
    yield
    billing_port.set_billing_adapter(NoopBillingAdapter())


class _Ctx:
    org_id = "org-1"
    project_id = "proj-1"
    plan_tier = "free"


def test_charge_maps_ctx_and_builds_idempotency_key():
    adapter = CloudBillingAdapter()
    with (
        patch(
            "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
            return_value=_Ctx(),
        ),
        patch(
            "agentic_project_service.services.billing_cloud.adapter.make_idempotency_key",
            return_value="idem-abc",
        ) as mk,
        patch(
            "agentic_project_service.services.billing_cloud.adapter.post_charge",
            return_value=ChargeResult(success=True, charge_id="c1", balance=99),
        ) as pc,
    ):
        out = adapter.charge(
            action="tool_call",
            quantity=1,
            ref_type="tool_call",
            ref_id="r1",
            idempotency_parts=("step-7",),
        )
    mk.assert_called_once_with("org-1", "tool_call", "step-7")
    pc.assert_called_once()
    assert pc.call_args.kwargs["org_id"] == "org-1"
    assert pc.call_args.kwargs["idempotency_key"] == "idem-abc"
    assert out == ChargeOutcome(charged=True, insufficient_credits=False, balance=99)


def test_charge_returns_noop_outcome_when_billing_unconfigured():
    adapter = CloudBillingAdapter()
    with patch(
        "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
        return_value=None,
    ):
        out = adapter.charge(action="agent_run")
    assert out == ChargeOutcome(charged=False)


def test_charge_reports_insufficient_credits():
    adapter = CloudBillingAdapter()
    with (
        patch(
            "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
            return_value=_Ctx(),
        ),
        patch(
            "agentic_project_service.services.billing_cloud.adapter.make_idempotency_key",
            return_value="k",
        ),
        patch(
            "agentic_project_service.services.billing_cloud.adapter.post_charge",
            return_value=ChargeResult(
                success=False, failure_mode="insufficient_credits", balance=0
            ),
        ),
    ):
        out = adapter.charge(action="agent_run", idempotency_parts=("run-1",))
    assert out.insufficient_credits is True
    assert out.charged is False


def test_charge_idempotency_action_overrides_key_action():
    # Key uses idempotency_action (requested), NOT the billed action.
    adapter = CloudBillingAdapter()
    with (
        patch(
            "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
            return_value=_Ctx(),
        ),
        patch(
            "agentic_project_service.services.billing_cloud.adapter.make_idempotency_key",
            return_value="k",
        ) as mk,
        patch(
            "agentic_project_service.services.billing_cloud.adapter.post_charge",
            return_value=ChargeResult(success=True, charge_id="c", balance=1),
        ) as pc,
    ):
        adapter.charge(
            action="ocr_pages", idempotency_action="advanced_ocr", idempotency_parts=("src-1",)
        )
    mk.assert_called_once_with("org-1", "advanced_ocr", "src-1")  # KEY uses requested
    assert pc.call_args.kwargs["action"] == "ocr_pages"  # CHARGE uses actual


def test_charge_idempotency_action_defaults_to_billed_action():
    # Backward-compat: no idempotency_action -> key uses the billed action (Tasks 6/7/8 unchanged).
    adapter = CloudBillingAdapter()
    with (
        patch(
            "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
            return_value=_Ctx(),
        ),
        patch(
            "agentic_project_service.services.billing_cloud.adapter.make_idempotency_key",
            return_value="k",
        ) as mk,
        patch(
            "agentic_project_service.services.billing_cloud.adapter.post_charge",
            return_value=ChargeResult(success=True, charge_id="c", balance=1),
        ),
    ):
        adapter.charge(action="agent_run", idempotency_parts=("run-1",))
    mk.assert_called_once_with("org-1", "agent_run", "run-1")


def test_check_balance_delegates_with_ctx():
    adapter = CloudBillingAdapter()
    with (
        patch(
            "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
            return_value=_Ctx(),
        ),
        patch("agentic_project_service.services.billing_cloud.adapter.check_balance_or_503") as cb,
    ):
        adapter.check_balance(estimated_cost=5)
    cb.assert_called_once_with(
        org_id="org-1", project_id="proj-1", estimated_cost=5, plan_tier="free"
    )


def test_check_balance_is_noop_when_unconfigured():
    adapter = CloudBillingAdapter()
    with (
        patch(
            "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
            return_value=None,
        ),
        patch("agentic_project_service.services.billing_cloud.adapter.check_balance_or_503") as cb,
    ):
        adapter.check_balance(estimated_cost=5)
    cb.assert_not_called()


def test_llm_call_scope_arms_recoupable():
    adapter = CloudBillingAdapter()
    entered = []

    @contextmanager
    def fake_recoup():
        entered.append(True)
        yield

    with patch(
        "agentic_project_service.services.billing_cloud.adapter.recoupable_llm_call", fake_recoup
    ):
        with adapter.llm_call_scope():
            pass
    assert entered == [True]


def test_install_billing_registers_adapter_and_logger():
    app = MagicMock()
    app.before_request = MagicMock(return_value=lambda f: f)
    from agentic_project_service.services import billing_cloud

    with patch(
        "agentic_project_service.services.billing_cloud.adapter.register_billing_logger"
    ) as rbl:
        billing_cloud.install_billing(app)
    rbl.assert_called_once()
    assert isinstance(billing_port.get_billing_adapter(), CloudBillingAdapter)


def test_task_scope_sets_byok_providers_on_success():
    adapter = CloudBillingAdapter()
    with (
        patch(
            "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
            return_value=_Ctx(),
        ),
        patch(
            "agentic_project_service.services.billing_cloud.adapter.list_byok_providers",
            return_value=frozenset({"openai"}),
        ),
    ):
        with adapter.task_scope():
            assert current_byok_providers.get() == frozenset({"openai"})


def test_task_scope_degrades_on_lookup_failure():
    adapter = CloudBillingAdapter()
    with (
        patch(
            "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
            return_value=_Ctx(),
        ),
        patch(
            "agentic_project_service.services.billing_cloud.adapter.list_byok_providers",
            side_effect=Exception("boom"),
        ),
    ):
        with adapter.task_scope():
            assert byok_lookup_degraded.get() is True
            assert current_byok_providers.get() == frozenset()


def test_per_batch_callback_none_when_disabled():
    adapter = CloudBillingAdapter()
    with patch(
        "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
        return_value=_Ctx(),
    ):
        assert (
            adapter.per_batch_callback(config_id="c1", action="metadata_enrichment", enabled=False)
            is None
        )


def test_per_batch_callback_none_when_unconfigured():
    adapter = CloudBillingAdapter()
    with patch(
        "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
        return_value=None,
    ):
        assert (
            adapter.per_batch_callback(config_id="c1", action="metadata_enrichment", enabled=True)
            is None
        )


def test_per_batch_callback_returns_callable_when_enabled():
    adapter = CloudBillingAdapter()
    sentinel = MagicMock()
    with (
        patch(
            "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
            return_value=_Ctx(),
        ),
        patch(
            "agentic_project_service.services.billing_cloud.adapter.make_per_batch_billing_callback",
            return_value=sentinel,
        ),
    ):
        result = adapter.per_batch_callback(
            config_id="c1", action="metadata_enrichment", enabled=True
        )
    assert result is sentinel
