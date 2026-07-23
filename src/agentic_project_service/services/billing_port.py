# src/agentic_project_service/services/billing_port.py
"""The billing port — the single seam between powabase-ai and charging.

The public core NEVER charges. It calls this port; a registered ``BillingAdapter``
decides what a charge/balance-check means. The default adapter is a NO-OP, so the
open-source build carries zero charging logic. The private cloud edition registers
``CloudBillingAdapter`` (services/billing_cloud) at app startup, restoring the full
credit metering.

All facade functions resolve the adapter at CALL time (not import time) so late
registration by the cloud app-factory takes effect for decorators applied during
module import (e.g. ``@billing.task_context`` on Celery tasks).
"""

from __future__ import annotations

import contextlib
import functools
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class ChargeOutcome:
    """Neutral result of a charge attempt.

    charged: the adapter recorded a charge (cloud). ``False`` for the no-op.
    insufficient_credits: the org hit its cap (cloud 402). Reported for
        completeness; core call-sites are fire-and-forget and do not branch on
        it. Per-batch backpressure is handled separately, inside
        ``per_batch_callback`` (which returns "abort" on 402), not via this type.
    balance: post-charge balance when the adapter reports one, else None.
    """

    charged: bool
    insufficient_credits: bool = False
    balance: int | None = None


@runtime_checkable
class BillingAdapter(Protocol):
    def charge(
        self,
        *,
        action: str,
        quantity: int = 1,
        ref_type: str | None = None,
        ref_id: str | None = None,
        idempotency_action: str | None = None,
        idempotency_parts: tuple[str, ...] = (),
        metadata: dict | None = None,
    ) -> ChargeOutcome: ...

    def check_balance(self, *, estimated_cost: int) -> None: ...

    def llm_call_scope(self) -> AbstractContextManager[None]: ...

    def task_scope(self) -> AbstractContextManager[None]: ...

    def per_batch_callback(
        self, *, config_id: str, action: str, enabled: bool = True
    ) -> Callable[[int, list[str]], str] | None: ...


class NoopBillingAdapter:
    """Ships in the OSS build. Every method is inert: no charge, never 402/503,
    no metering, no BYOK context. BYOK-only self-host needs none of it."""

    def charge(self, **_kw) -> ChargeOutcome:
        return ChargeOutcome(charged=False)

    def check_balance(self, **_kw) -> None:
        return None

    def llm_call_scope(self) -> AbstractContextManager[None]:
        return contextlib.nullcontext()

    def task_scope(self) -> AbstractContextManager[None]:
        return contextlib.nullcontext()

    def per_batch_callback(self, **_kw):
        return None


_adapter: BillingAdapter = NoopBillingAdapter()


def set_billing_adapter(adapter: BillingAdapter) -> None:
    global _adapter
    _adapter = adapter


def get_billing_adapter() -> BillingAdapter:
    return _adapter


# --- facade (resolve _adapter at call time) -------------------------------


def charge(
    *,
    action: str,
    quantity: int = 1,
    ref_type: str | None = None,
    ref_id: str | None = None,
    idempotency_action: str | None = None,
    idempotency_parts: tuple[str, ...] = (),
    metadata: dict | None = None,
) -> ChargeOutcome:
    """idempotency_action: the stable identity action used to build the idempotency
    KEY, when it differs from the billed ``action`` (e.g. a charge whose billed
    category is resolved after a fallback — the key must stay retry-stable on the
    originally-requested action). Defaults to ``action``."""
    return _adapter.charge(
        action=action,
        quantity=quantity,
        ref_type=ref_type,
        ref_id=ref_id,
        idempotency_action=idempotency_action,
        idempotency_parts=idempotency_parts,
        metadata=metadata,
    )


def check_balance(*, estimated_cost: int) -> None:
    return _adapter.check_balance(estimated_cost=estimated_cost)


def llm_call_scope() -> AbstractContextManager[None]:
    return _adapter.llm_call_scope()


def per_batch_callback(*, config_id: str, action: str, enabled: bool = True):
    return _adapter.per_batch_callback(config_id=config_id, action=action, enabled=enabled)


def task_context(fn):
    """Celery-task decorator. Wraps the task body in the adapter's task_scope so
    the cloud adapter can set the request's BYOK context; the no-op does nothing.
    Resolves the adapter at CALL time so cloud's late registration is honored."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _adapter.task_scope():
            return fn(*args, **kwargs)

    wrapper.__has_billing_context__ = True
    return wrapper


def no_billing_context(fn):
    """Marker for tasks that never invoke LLMs — opts out of the entry-point lint.
    No runtime behavior (kept as the public twin of the old marker)."""
    fn.__no_billing_context__ = True
    return fn
