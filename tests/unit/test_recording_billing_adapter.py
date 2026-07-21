# tests/unit/test_recording_billing_adapter.py
import pytest

from agentic_project_service.services import billing_port
from agentic_project_service.services.billing_port import ChargeOutcome
from tests.support.billing import RecordingBillingAdapter


def test_records_charges_and_balance_checks():
    rec = RecordingBillingAdapter()
    billing_port.set_billing_adapter(rec)
    billing_port.check_balance(estimated_cost=3)
    out = billing_port.charge(action="agent_run", quantity=2, idempotency_parts=("run-1",))
    assert rec.balance_checks == [3]
    assert rec.charges[0]["action"] == "agent_run"
    assert rec.charges[0]["quantity"] == 2
    assert out == ChargeOutcome(charged=True)


def test_insufficient_flag_makes_charge_report_402():
    rec = RecordingBillingAdapter(insufficient=True)
    billing_port.set_billing_adapter(rec)
    out = billing_port.charge(action="metadata_enrichment")
    assert out.insufficient_credits is True
    assert out.charged is False


def test_raise_402_makes_check_balance_raise():
    from werkzeug.exceptions import HTTPException

    rec = RecordingBillingAdapter(raise_402=True)
    billing_port.set_billing_adapter(rec)
    with pytest.raises(HTTPException) as ei:
        billing_port.check_balance(estimated_cost=1)
    assert ei.value.code == 402


def test_per_batch_callback_records_wiring_and_continues():
    rec = RecordingBillingAdapter()
    billing_port.set_billing_adapter(rec)
    cb = billing_port.per_batch_callback(config_id="cfg-1", action="metadata_enrichment")
    assert rec.per_batch_calls == [
        {"config_id": "cfg-1", "action": "metadata_enrichment", "enabled": True}
    ]
    assert cb is not None
    assert cb(3, ["a", "b"]) == "continue"
    assert rec.batch_invocations == [
        {"batch_ok": 3, "ids": ["a", "b"], "action": "metadata_enrichment"}
    ]


def test_per_batch_callback_disabled_returns_none():
    rec = RecordingBillingAdapter()
    billing_port.set_billing_adapter(rec)
    assert billing_port.per_batch_callback(config_id="c", action="a", enabled=False) is None
    assert rec.per_batch_calls == [{"config_id": "c", "action": "a", "enabled": False}]


def test_per_batch_callback_abort_after_signals_abort():
    rec = RecordingBillingAdapter(per_batch_abort_after=0)
    billing_port.set_billing_adapter(rec)
    cb = billing_port.per_batch_callback(config_id="c", action="indexing_graphindex")
    assert cb(1, ["x"]) == "abort"
