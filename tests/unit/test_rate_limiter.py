"""Tests for the in-memory rate limiter."""

import time

from agentic_project_service.services.rate_limit import RateLimiter


class TestRateLimiter:
    """Unit tests for the sliding-window rate limiter."""

    def test_allows_requests_under_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("user1") is True

    def test_blocks_requests_over_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False

    def test_different_users_independent(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False
        # user2 should still be allowed
        assert limiter.is_allowed("user2") is True

    def test_window_expiry(self):
        limiter = RateLimiter(max_requests=1, window_seconds=0.1)
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False
        time.sleep(0.15)
        assert limiter.is_allowed("user1") is True

    def test_zero_max_requests_always_blocks(self):
        limiter = RateLimiter(max_requests=0, window_seconds=60)
        assert limiter.is_allowed("user1") is False
