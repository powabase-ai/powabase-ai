"""Tests for the Redis sliding-window rate limiter."""

from unittest.mock import MagicMock

import fakeredis
import pytest

from agentic_project_service.services.rate_limit import RedisRateLimiter


@pytest.fixture
def limiter():
    return RedisRateLimiter(redis_client=fakeredis.FakeStrictRedis())


class TestTryAcquire:
    def test_allows_under_limit(self, limiter):
        for _ in range(3):
            assert limiter.try_acquire("firecrawl", 3) is None

    def test_denies_over_limit_with_wait_hint(self, limiter):
        for _ in range(3):
            assert limiter.try_acquire("firecrawl", 3) is None
        wait = limiter.try_acquire("firecrawl", 3)
        assert wait is not None
        assert 0.0 < wait <= 60.0

    def test_services_are_independent(self, limiter):
        assert limiter.try_acquire("firecrawl", 1) is None
        assert limiter.try_acquire("firecrawl", 1) is not None
        assert limiter.try_acquire("exa", 1) is None

    def test_projects_are_independent(self, monkeypatch):
        client = fakeredis.FakeStrictRedis()
        limiter = RedisRateLimiter(redis_client=client)
        monkeypatch.setenv("PROJECT_REF", "proj-a")
        assert limiter.try_acquire("firecrawl", 1) is None
        monkeypatch.setenv("PROJECT_REF", "proj-b")
        assert limiter.try_acquire("firecrawl", 1) is None

    def test_fail_open_on_redis_error(self):
        broken = MagicMock()
        broken.register_script.side_effect = ConnectionError("redis down")
        broken.eval.side_effect = ConnectionError("redis down")
        limiter = RedisRateLimiter(redis_client=broken)
        assert limiter.try_acquire("firecrawl", 1) is None
        assert limiter.try_acquire("firecrawl", 1) is None  # still open


class TestAcquireBlocking:
    def test_immediate_success(self, limiter):
        assert limiter.acquire_blocking("firecrawl", 5, timeout_s=0.1) is True

    def test_times_out_when_saturated(self, limiter, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(
            "agentic_project_service.services.rate_limit.time.sleep",
            lambda s: sleeps.append(s),
        )
        assert limiter.try_acquire("firecrawl", 1) is None
        # Window is 60s and fake clock doesn't advance, so this must give up.
        assert limiter.acquire_blocking("firecrawl", 1, timeout_s=0.05) is False
        assert sleeps, "should have slept between polls"


class TestWaitHintPrecision:
    def test_sub_second_wait_hint_is_positive(self, monkeypatch):
        """Redis truncates Lua float replies to integers — sub-second waits
        must not collapse to 0 (0 causes busy-spins and zero countdowns)."""
        limiter = RedisRateLimiter(redis_client=fakeredis.FakeStrictRedis())
        base = 1_000_000.0
        monkeypatch.setattr("agentic_project_service.services.rate_limit.time.time", lambda: base)
        assert limiter.try_acquire("firecrawl", 1) is None
        monkeypatch.setattr(
            "agentic_project_service.services.rate_limit.time.time",
            lambda: base + 59.5,
        )
        wait = limiter.try_acquire("firecrawl", 1)
        assert wait is not None
        assert 0.0 < wait <= 0.6

    def test_window_expiry_allows_new_acquire(self, monkeypatch):
        limiter = RedisRateLimiter(redis_client=fakeredis.FakeStrictRedis())
        base = 1_000_000.0
        monkeypatch.setattr("agentic_project_service.services.rate_limit.time.time", lambda: base)
        assert limiter.try_acquire("firecrawl", 1) is None
        assert limiter.try_acquire("firecrawl", 1) is not None  # saturated
        monkeypatch.setattr(
            "agentic_project_service.services.rate_limit.time.time",
            lambda: base + 61.0,
        )
        assert limiter.try_acquire("firecrawl", 1) is None  # window aged out

    def test_acquire_blocking_sleep_has_floor(self, monkeypatch):
        """A near-zero wait hint must not busy-spin acquire_blocking."""
        limiter = RedisRateLimiter(redis_client=fakeredis.FakeStrictRedis())
        base = 1_000_000.0
        monkeypatch.setattr("agentic_project_service.services.rate_limit.time.time", lambda: base)
        assert limiter.try_acquire("firecrawl", 1) is None
        monkeypatch.setattr(
            "agentic_project_service.services.rate_limit.time.time",
            lambda: base + 59.99,
        )
        sleeps: list[float] = []
        monkeypatch.setattr(
            "agentic_project_service.services.rate_limit.time.sleep",
            lambda s: sleeps.append(s),
        )
        limiter.acquire_blocking("firecrawl", 1, timeout_s=0.05)
        assert sleeps, "saturated blocking acquire must sleep between polls"
        assert all(s >= 0.05 for s in sleeps)
