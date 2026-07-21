"""Simple in-memory sliding-window rate limiter."""

import threading
import time
from functools import wraps

from flask import g, jsonify


class RateLimiter:
    """Per-key sliding-window rate limiter.

    This is per-process — not distributed across workers — but provides
    basic protection against single-user credit exhaustion.
    """

    def __init__(self, max_requests: int, window_seconds: int | float):
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            timestamps = self._requests.get(key, [])
            # Prune expired entries
            timestamps = [t for t in timestamps if now - t < self.window]
            if not timestamps and key in self._requests:
                del self._requests[key]
            if len(timestamps) >= self.max_requests:
                self._requests[key] = timestamps
                return False
            timestamps.append(now)
            self._requests[key] = timestamps
            return True


# Configured limit for workflow execution endpoints
execution_limiter = RateLimiter(max_requests=20, window_seconds=60)


def rate_limit_executions(f):
    """Decorator that rate-limits workflow execution endpoints per user."""

    @wraps(f)
    def decorated(*args, **kwargs):
        user_id = getattr(g, "user_id", None) or "anonymous"
        if not execution_limiter.is_allowed(user_id):
            return (
                jsonify({"error": "Rate limit exceeded. Max 20 executions per minute."}),
                429,
            )
        return f(*args, **kwargs)

    return decorated
