"""Tests for Flask before_request — clears+sets current_byok_providers.

The hook is registered by services.billing_cloud.install_billing(), called from
main.create_app(). Identity comes from get_billing_context()
(env-based, per-pod). The hook must:
 1. Clear current_byok_providers at every request entry (gthread thread-pool
    reuse — stale state could leak between requests served by the same thread).
 2. Set current_byok_providers = list_byok_providers(ctx.project_id) when
    get_billing_context() returns a context; otherwise leave it cleared.

Fixture adaptation: the task body sketched ``create_app(testing=True)``; this
codebase's create_app() takes a ``testing`` flag (added in this commit) that
skips DB/migration init so unit tests run without Postgres.
"""

import logging
from unittest.mock import patch

import pytest

from agentic_project_service.services.billing_cloud.identity import (
    byok_lookup_degraded,
    current_byok_providers,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Tests in this module trigger the before_request hook via Flask's
    test_client. The hook calls ``current_byok_providers.set(...)`` and
    ``byok_lookup_degraded.set(...)`` in the test's contextvar binding,
    which leaks into sibling tests (notably test_billing_decorator.py)
    when both files run in the same pytest session. Reset after each
    test to match the pattern in test_billing_litellm_callback.py."""
    yield
    current_byok_providers.set(frozenset())
    byok_lookup_degraded.set(False)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("BILLING_ENABLED", "true")
    monkeypatch.setenv("BILLING_ORG_ID", "org-default")
    monkeypatch.setenv("PROJECT_ID", "proj-default")
    from agentic_project_service.main import create_app

    return create_app(testing=True)


def test_before_request_sets_byok_for_each_request(app):
    """At every request entry, before_request refreshes current_byok_providers."""
    captured = {}

    @app.route("/_test_probe")
    def probe():
        captured["byok"] = current_byok_providers.get()
        return "ok"

    with patch(
        "agentic_project_service.services.billing_cloud.adapter.list_byok_providers",
        return_value=frozenset({"anthropic"}),
    ):
        with app.test_client() as client:
            client.get("/_test_probe")
    assert captured["byok"] == frozenset({"anthropic"})


def test_before_request_clears_byok_between_requests(app):
    """Stale state from a prior request must not leak into the next."""
    current_byok_providers.set(frozenset({"stale"}))
    captured = {}

    @app.route("/_test_probe2")
    def probe():
        captured["byok"] = current_byok_providers.get()
        return "ok"

    with patch(
        "agentic_project_service.services.billing_cloud.adapter.list_byok_providers",
        return_value=frozenset({"fresh"}),
    ):
        with app.test_client() as client:
            client.get("/_test_probe2")
    assert captured["byok"] == frozenset({"fresh"})


def test_before_request_skips_when_billing_unconfigured(app, monkeypatch):
    monkeypatch.delenv("BILLING_ORG_ID")
    captured = {}

    @app.route("/_test_probe3")
    def probe():
        captured["byok"] = current_byok_providers.get()
        return "ok"

    with patch(
        "agentic_project_service.services.billing_cloud.adapter.list_byok_providers"
    ) as mock_list:
        with app.test_client() as client:
            client.get("/_test_probe3")
    mock_list.assert_not_called()
    assert captured["byok"] == frozenset()


# ---------------------------------------------------------------------------
# I14 — list_byok_providers failure must not 500 every subsequent request
# ---------------------------------------------------------------------------


def test_before_request_swallows_list_byok_providers_exception(app, caplog):
    """If list_byok_providers raises (e.g. PendingRollbackError), the hook
    must log a warning and leave current_byok_providers empty so the handler
    still runs."""
    captured = {}

    @app.route("/_test_probe4")
    def probe():
        captured["byok"] = current_byok_providers.get()
        return "ok"

    with patch(
        "agentic_project_service.services.billing_cloud.adapter.list_byok_providers",
        side_effect=RuntimeError("PendingRollbackError simulated"),
    ):
        with caplog.at_level(logging.WARNING, logger="agentic_project_service.main"):
            with app.test_client() as client:
                resp = client.get("/_test_probe4")

    assert resp.status_code == 200
    assert captured["byok"] == frozenset()
    assert any("byok_lookup_failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# N7 — byok_lookup_degraded flag must be set on lookup failure
# ---------------------------------------------------------------------------


def test_before_request_sets_byok_lookup_degraded_on_failure(app):
    """When list_byok_providers raises, the I14 except branch must set the
    byok_lookup_degraded contextvar to True. BillingLogger reads this flag
    and writes it into the ledger row's metadata so ops can audit any
    incorrectly-charged AI-on-us calls during the failure window."""
    from agentic_project_service.services.billing_cloud.identity import byok_lookup_degraded

    captured = {}

    @app.route("/_test_degraded_set")
    def probe():
        captured["degraded"] = byok_lookup_degraded.get()
        return "ok"

    with patch(
        "agentic_project_service.services.billing_cloud.adapter.list_byok_providers",
        side_effect=RuntimeError("PendingRollbackError simulated"),
    ):
        with app.test_client() as client:
            resp = client.get("/_test_degraded_set")

    assert resp.status_code == 200
    assert captured["degraded"] is True


def test_before_request_keeps_byok_lookup_degraded_false_on_success(app):
    """Healthy path: degraded flag stays False so it can be safely emitted
    on every charge without misclassifying healthy traffic."""
    from agentic_project_service.services.billing_cloud.identity import byok_lookup_degraded

    captured = {}

    @app.route("/_test_degraded_clear")
    def probe():
        captured["degraded"] = byok_lookup_degraded.get()
        return "ok"

    with patch(
        "agentic_project_service.services.billing_cloud.adapter.list_byok_providers",
        return_value=frozenset({"anthropic"}),
    ):
        with app.test_client() as client:
            resp = client.get("/_test_degraded_clear")

    assert resp.status_code == 200
    assert captured["degraded"] is False
