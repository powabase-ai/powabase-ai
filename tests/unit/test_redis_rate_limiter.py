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
