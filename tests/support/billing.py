"""Test double for the billing port. Records charge/balance calls so call-site
tests can assert billing wiring behaviorally without a live billing service."""

from __future__ import annotations

import contextlib

from werkzeug.exceptions import HTTPException, ServiceUnavailable

from agentic_project_service.services.billing_port import ChargeOutcome


class _PaymentRequired(HTTPException):
    code = 402
    description = "Payment Required"


class RecordingBillingAdapter:
    def __init__(
        self,
        *,
        insufficient=False,
        raise_402=False,
        raise_503=False,
        per_batch_abort_after=None,
    ):
        self.charges: list[dict] = []
        self.balance_checks: list[int] = []
        self.llm_scopes = 0
        # Records every per_batch_callback(...) wiring call so call-site tests
        # can assert config_id/action/enabled without a live billing service.
        self.per_batch_calls: list[dict] = []
        # Records every INVOCATION of a returned per-batch callback (batch_ok +
        # ids), so backpressure tests can assert which batches were charged.
        self.batch_invocations: list[dict] = []
        self._insufficient = insufficient
        self._raise_402 = raise_402
        self._raise_503 = raise_503
        # When set, the returned per-batch callback signals 'abort' once its
        # invocation index reaches this value (0 = abort on the first batch),
        # reproducing the cloud callback's insufficient-credits backpressure.
        self._per_batch_abort_after = per_batch_abort_after

    def charge(
        self,
        *,
        action,
        quantity=1,
        ref_type=None,
        ref_id=None,
        idempotency_action=None,
        idempotency_parts=(),
        metadata=None,
    ) -> ChargeOutcome:
        self.charges.append(
            {
                "action": action,
                "quantity": quantity,
                "ref_type": ref_type,
                "ref_id": ref_id,
                "idempotency_action": idempotency_action,
                "idempotency_parts": idempotency_parts,
                "metadata": metadata,
            }
        )
        if self._insufficient:
            return ChargeOutcome(charged=False, insufficient_credits=True, balance=0)
        return ChargeOutcome(charged=True)

    def check_balance(self, *, estimated_cost: int) -> None:
        self.balance_checks.append(estimated_cost)
        if self._raise_503:
            raise ServiceUnavailable("billing unreachable")
        if self._raise_402:
            raise _PaymentRequired()

    def llm_call_scope(self):
        self.llm_scopes += 1
        return contextlib.nullcontext()

    def task_scope(self):
        return contextlib.nullcontext()

    def per_batch_callback(self, *, config_id, action, enabled=True):
        self.per_batch_calls.append({"config_id": config_id, "action": action, "enabled": enabled})
        if not enabled:
            return None

        def _cb(batch_ok, batch_item_ids):
            idx = len(self.batch_invocations)
            self.batch_invocations.append(
                {"batch_ok": batch_ok, "ids": list(batch_item_ids), "action": action}
            )
            if self._per_batch_abort_after is not None and idx >= self._per_batch_abort_after:
                return "abort"
            return "continue"

        return _cb
