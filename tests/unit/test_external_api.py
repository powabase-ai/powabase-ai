"""Tests for Retry-After parsing and HTTP status classification."""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

from agentic_project_service.services.external_api import (
    PERMANENT_STATUSES,
    parse_retry_after,
)


class TestParseRetryAfter:
    def test_integer_seconds(self):
        assert parse_retry_after("30") == 30.0

    def test_none_returns_default(self):
        assert parse_retry_after(None) == 60.0

    def test_custom_default(self):
        assert parse_retry_after(None, default=10.0) == 10.0

    def test_garbage_returns_default(self):
        assert parse_retry_after("soon-ish") == 60.0

    def test_http_date(self):
        future = datetime.now(timezone.utc) + timedelta(seconds=45)
        result = parse_retry_after(format_datetime(future, usegmt=True))
        assert 40.0 <= result <= 46.0

    def test_past_http_date_clamps_to_zero(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=45)
        assert parse_retry_after(format_datetime(past, usegmt=True)) == 0.0

    def test_negative_seconds_returns_default(self):
        assert parse_retry_after("-5") == 60.0

    def test_cap(self):
        assert parse_retry_after("9999") == 300.0
        assert parse_retry_after("9999", cap=120.0) == 120.0


class TestPermanentStatuses:
    def test_membership(self):
        assert PERMANENT_STATUSES == frozenset({400, 401, 402, 403, 404, 410})

    def test_429_and_5xx_are_not_permanent(self):
        assert 429 not in PERMANENT_STATUSES
        assert 500 not in PERMANENT_STATUSES
        assert 503 not in PERMANENT_STATUSES
