"""Helpers for calling rate-limited external APIs (Firecrawl, Exa)."""

import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

# 4xx statuses that will not succeed on retry. 408/429 are deliberately
# absent (retryable); 5xx are transient by definition.
PERMANENT_STATUSES: frozenset[int] = frozenset({400, 401, 402, 403, 404, 410})


def parse_retry_after(value: str | None, default: float = 60.0, cap: float = 300.0) -> float:
    """Parse an HTTP Retry-After header value into a wait in seconds.

    Accepts integer seconds or an HTTP-date. Returns ``default`` when the
    value is missing or unparseable, clamps the result to ``[0, cap]``.
    """
    if value is None:
        return min(default, cap)
    value = value.strip()
    try:
        seconds = float(value)
        # Reject nan/inf explicitly: nan passes a `< 0` check and survives
        # min(), and a nan countdown later blows up inside Celery's retry.
        if not math.isfinite(seconds) or seconds < 0:
            return min(default, cap)
        return min(seconds, cap)
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return min(default, cap)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = (dt - datetime.now(timezone.utc)).total_seconds()
    return min(max(delta, 0.0), cap)
