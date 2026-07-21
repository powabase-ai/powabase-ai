import pytest
from agentic_project_service.services.billing_cloud import balance_cache
from agentic_project_service.services.billing_cloud.balance_cache import (
    check_balance_or_503,
    PaymentRequired,
    clear_all,
)


def setup_function():
    clear_all()


def test_paid_small_overrun_within_grace_passes(monkeypatch):
    monkeypatch.setattr(balance_cache, "_paid_soft_cap_grace", lambda: 3000)
    monkeypatch.setattr(balance_cache, "get_balance_cached", lambda **k: 100)
    # est 1000 <= balance 100 + grace 3000 → allowed (post-paid mid-flight overrun)
    check_balance_or_503(org_id="o", project_id="p", estimated_cost=1000, plan_tier="pro")


def test_paid_large_job_over_grace_rejected(monkeypatch):
    """Mechanism test for a HYPOTHETICAL future paid org (paid tier not shipped
    today) with a non-zero grace: a +6-balance org starting a 30,000-credit job
    must be refused. Pins the projected-balance form against the round-1
    multiplier bug, which would have let this through."""
    monkeypatch.setattr(balance_cache, "_paid_soft_cap_grace", lambda: 3000)
    monkeypatch.setattr(balance_cache, "get_balance_cached", lambda **k: 6)
    # est 30000 > balance 6 + grace 3000 = 3006 → 402
    with pytest.raises(PaymentRequired):
        check_balance_or_503(org_id="o", project_id="p", estimated_cost=30000, plan_tier="pro")


def test_free_tier_unchanged(monkeypatch):
    monkeypatch.setattr(balance_cache, "get_balance_cached", lambda **k: 50)
    with pytest.raises(PaymentRequired):
        check_balance_or_503(org_id="o", project_id="p", estimated_cost=1000, plan_tier="free")


def test_paid_default_grace_is_zero_prepaid(monkeypatch):
    """Shipped default GRACE=0 makes the (currently-unreachable) paid branch
    fail-CLOSED/prepaid, not fail-open: est > balance → 402, same as free."""
    monkeypatch.delenv("BILLING_PAID_TIER_SOFT_CAP_GRACE_CREDITS", raising=False)
    monkeypatch.setattr(balance_cache, "get_balance_cached", lambda **k: 6)
    with pytest.raises(PaymentRequired):
        check_balance_or_503(org_id="o", project_id="p", estimated_cost=30000, plan_tier="pro")
