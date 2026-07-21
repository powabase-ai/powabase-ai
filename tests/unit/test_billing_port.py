# tests/unit/test_billing_port.py
import contextlib

import pytest

from agentic_project_service.services import billing_port as billing
from agentic_project_service.services.billing_port import (
    BillingAdapter,
    ChargeOutcome,
    NoopBillingAdapter,
    get_billing_adapter,
    set_billing_adapter,
)


@pytest.fixture(autouse=True)
def _restore_default_adapter():
    saved = get_billing_adapter()
    set_billing_adapter(NoopBillingAdapter())
    yield
    set_billing_adapter(saved)


def test_default_adapter_is_noop():
    # A freshly imported module defaults to the no-op adapter.
    assert isinstance(get_billing_adapter(), NoopBillingAdapter)


def test_noop_charge_charges_nothing():
    out = billing.charge(action="agent_run", quantity=1, idempotency_parts=("run-1",))
    assert out == ChargeOutcome(charged=False, insufficient_credits=False, balance=None)


def test_noop_check_balance_never_raises():
    # OSS must never 402/503 — no-op returns None regardless of cost.
    assert billing.check_balance(estimated_cost=10_000_000) is None


def test_noop_llm_call_scope_is_nullcontext():
    with billing.llm_call_scope():
        pass


def test_noop_task_context_is_identity_at_call_time():
    calls = []

    @billing.task_context
    def task(x):
        calls.append(x)
        return x * 2

    assert task(3) == 6
    assert calls == [3]


def test_task_context_sets_has_billing_context_marker_on_wrapper():
    """The lint script (test_celery_tasks_have_billing_context.py) reads this
    marker on task.__wrapped__ (= wrapper) to confirm a Celery task opted
    into billing. Carried over from the now-deleted billing_context.py
    decorator's equivalent test — task_context is the public replacement."""

    @billing.task_context
    def task():
        return None

    assert task.__has_billing_context__ is True


def test_no_billing_context_marker_only():
    """Carried over from the now-deleted billing_context.py decorator's
    equivalent test — billing_port.no_billing_context is the public twin."""

    @billing.no_billing_context
    def task():
        return 42

    assert task() == 42
    assert task.__no_billing_context__ is True


def test_noop_per_batch_callback_is_none():
    assert billing.per_batch_callback(config_id="c1", action="metadata_enrichment") is None


def test_facade_resolves_adapter_at_call_time_not_import_time():
    # task_context is applied at import time in real code, but must dispatch to
    # whichever adapter is registered when the task RUNS (cloud installs late).
    seen = []

    class Spy(NoopBillingAdapter):
        @contextlib.contextmanager
        def task_scope(self):
            seen.append("enter")
            yield
            seen.append("exit")

    @billing.task_context
    def task():
        seen.append("body")

    set_billing_adapter(Spy())  # registered AFTER decoration
    task()
    assert seen == ["enter", "body", "exit"]


def test_charge_delegates_to_registered_adapter():
    class Spy(NoopBillingAdapter):
        def charge(self, **kw):
            return ChargeOutcome(charged=True, balance=42)

    set_billing_adapter(Spy())
    assert billing.charge(action="x").charged is True
    assert billing.charge(action="x").balance == 42


def test_noop_satisfies_the_protocol():
    assert isinstance(NoopBillingAdapter(), BillingAdapter)
