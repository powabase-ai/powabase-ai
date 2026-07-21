"""Tests for billing_context — env-driven identity for the PS process."""

from agentic_project_service.services.billing_cloud import identity as bc


def test_get_billing_context_returns_none_without_env(monkeypatch):
    """No BILLING_ORG_ID + no PROJECT_ID → returns None (billing wiring skipped)."""
    monkeypatch.delenv("BILLING_ORG_ID", raising=False)
    monkeypatch.delenv("PROJECT_ID", raising=False)
    monkeypatch.delenv("BILLING_PROJECT_ID", raising=False)
    assert bc.get_billing_context() is None


def test_get_billing_context_returns_none_with_only_org(monkeypatch):
    """Org set but no project → None (both must be present)."""
    monkeypatch.setenv("BILLING_ORG_ID", "org-1")
    monkeypatch.delenv("PROJECT_ID", raising=False)
    monkeypatch.delenv("BILLING_PROJECT_ID", raising=False)
    assert bc.get_billing_context() is None


def test_get_billing_context_returns_context_when_set(monkeypatch):
    """Both env vars set → BillingContext with default plan_tier="free"."""
    monkeypatch.setenv("BILLING_ORG_ID", "org-1")
    monkeypatch.setenv("PROJECT_ID", "proj-1")
    monkeypatch.delenv("BILLING_PLAN_TIER", raising=False)
    ctx = bc.get_billing_context()
    assert ctx is not None
    assert ctx.org_id == "org-1"
    assert ctx.project_id == "proj-1"
    assert ctx.plan_tier == "free"


def test_get_billing_context_accepts_billing_project_id_alias(monkeypatch):
    """BILLING_PROJECT_ID env var is accepted when PROJECT_ID is unset."""
    monkeypatch.setenv("BILLING_ORG_ID", "org-1")
    monkeypatch.delenv("PROJECT_ID", raising=False)
    monkeypatch.setenv("BILLING_PROJECT_ID", "proj-2")
    ctx = bc.get_billing_context()
    assert ctx is not None
    assert ctx.project_id == "proj-2"


def test_get_billing_context_reads_plan_tier(monkeypatch):
    """BILLING_PLAN_TIER overrides the default."""
    monkeypatch.setenv("BILLING_ORG_ID", "org-1")
    monkeypatch.setenv("PROJECT_ID", "proj-1")
    monkeypatch.setenv("BILLING_PLAN_TIER", "pro")
    ctx = bc.get_billing_context()
    assert ctx is not None
    assert ctx.plan_tier == "pro"
