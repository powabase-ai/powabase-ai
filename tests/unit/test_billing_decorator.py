"""Tests for CloudBillingAdapter.task_scope() — sets/resets current_byok_providers
and byok_lookup_degraded around a Celery task body.

Formerly tested services/billing_context.py's ``@billing_context`` decorator
directly. That decorator was deleted when the charging modules relocated into
billing_cloud/ (its logic now lives in CloudBillingAdapter.task_scope, entered
via the public ``@billing.task_context`` decorator in services/billing_port.py
— see test_billing_port.py for task_context's own dispatch-at-call-time
tests). task_scope is a contextmanager, not a decorator, so each test below
drives it via ``with adapter.task_scope(): ...`` instead of decorating a
function. All byok/degraded lifecycle assertions from the original decorator
tests are preserved.
"""

import logging
from unittest.mock import patch

import pytest

from agentic_project_service.services.billing_cloud.adapter import CloudBillingAdapter
from agentic_project_service.services.billing_cloud.identity import (
    byok_lookup_degraded,
    current_byok_providers,
)


class _Ctx:
    org_id = "org-1"
    project_id = "proj-1"
    plan_tier = "free"


@pytest.fixture(autouse=True)
def reset_degraded():
    """Tests in this module drive task_scope, which sets byok_lookup_degraded
    in the test's contextvar binding; reset after each test so state doesn't
    leak between tests."""
    yield
    byok_lookup_degraded.set(False)


def _patch_ctx():
    return patch(
        "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
        return_value=_Ctx(),
    )


def _patch_list(return_value=None, side_effect=None):
    return patch(
        "agentic_project_service.services.billing_cloud.adapter.list_byok_providers",
        return_value=return_value,
        side_effect=side_effect,
    )


def test_task_scope_sets_byok_inside_call():
    adapter = CloudBillingAdapter()
    captured = {}
    with _patch_ctx(), _patch_list(return_value=frozenset({"openai"})):
        with adapter.task_scope():
            captured["byok"] = current_byok_providers.get()
    assert captured["byok"] == frozenset({"openai"})


def test_task_scope_resets_byok_after_call():
    adapter = CloudBillingAdapter()
    with _patch_ctx(), _patch_list(return_value=frozenset({"anthropic"})):
        with adapter.task_scope():
            pass
    assert current_byok_providers.get() == frozenset()


def test_task_scope_resets_on_exception():
    adapter = CloudBillingAdapter()
    with _patch_ctx(), _patch_list(return_value=frozenset({"anthropic"})):
        with pytest.raises(RuntimeError):
            with adapter.task_scope():
                raise RuntimeError("boom")
    assert current_byok_providers.get() == frozenset()


def test_task_scope_does_not_strand_byok_on_list_raise(caplog):
    """N16: list_byok_providers failure inside task_scope must mirror the
    HTTP before_request discipline — swallow, leave empty providers, set
    degraded=True, run the wrapped body so the task completes."""
    adapter = CloudBillingAdapter()
    ran = []
    with _patch_ctx(), _patch_list(side_effect=ConnectionError("DB blip")):
        with caplog.at_level(
            logging.WARNING,
            logger="agentic_project_service.services.billing_cloud.adapter",
        ):
            with adapter.task_scope():
                ran.append(True)
    assert ran == [True]
    assert current_byok_providers.get() == frozenset()
    assert any("byok_lookup_failed" in rec.message for rec in caplog.records)


def test_task_scope_pass_through_when_billing_unconfigured():
    adapter = CloudBillingAdapter()
    ran = []
    with (
        patch(
            "agentic_project_service.services.billing_cloud.adapter.get_billing_context",
            return_value=None,
        ),
        _patch_list() as mock_list,
    ):
        with adapter.task_scope():
            ran.append(current_byok_providers.get())
    mock_list.assert_not_called()
    assert ran == [frozenset()]


# ---------------------------------------------------------------------------
# N16 — byok_lookup_degraded flag must be set on Celery-side lookup failure
# ---------------------------------------------------------------------------
#
# Reviewer N16: Flask before_request hooks do NOT fire on Celery worker
# paths, so byok_lookup_degraded was only set on HTTP paths. task_scope runs
# INSIDE the Celery worker after broker dispatch — it is the right place to
# set the flag for Celery tasks. Pattern mirrors main.py:_set_billing_byok.


def test_task_scope_sets_degraded_on_lookup_failure():
    """When list_byok_providers raises inside task_scope, the
    byok_lookup_degraded contextvar must be True for the wrapped body so the
    BillingLogger downstream can fold the flag into the ledger row's
    metadata."""
    adapter = CloudBillingAdapter()
    captured = {}
    with _patch_ctx(), _patch_list(side_effect=ConnectionError("DB blip")):
        with adapter.task_scope():
            captured["degraded"] = byok_lookup_degraded.get()
            captured["byok"] = current_byok_providers.get()

    assert captured["degraded"] is True
    assert captured["byok"] == frozenset()


def test_task_scope_keeps_degraded_false_on_success():
    """Healthy path: byok_lookup_degraded MUST be False inside the task body.

    Pre-set the contextvar to True in the outer (test) context to pin
    task_scope's unconditional ``set(False)`` at entry — without that
    pre-bind the task body would inherit the outer True and the flag would
    be spuriously emitted on healthy traffic. Counterfactual: remove the
    pre-bind from CloudBillingAdapter.task_scope() -> task body sees True ->
    this test fails. (Round-4 M-R4-1, carried over from the deleted
    billing_context decorator's equivalent test.)
    """
    adapter = CloudBillingAdapter()
    captured = {}
    outer_token = byok_lookup_degraded.set(True)
    try:
        with _patch_ctx(), _patch_list(return_value=frozenset({"anthropic"})):
            with adapter.task_scope():
                captured["degraded"] = byok_lookup_degraded.get()
    finally:
        byok_lookup_degraded.reset(outer_token)

    assert captured["degraded"] is False


def test_task_scope_resets_degraded_after_call():
    """task_scope must reset byok_lookup_degraded in finally so a failed
    lookup in one task body doesn't leak to a sibling task body on the same
    worker thread."""
    adapter = CloudBillingAdapter()
    with _patch_ctx(), _patch_list(side_effect=ConnectionError("DB blip")):
        with adapter.task_scope():
            pass

    # After the context manager exits, the contextvar is back to its default.
    assert byok_lookup_degraded.get() is False
