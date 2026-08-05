"""Simple in-memory sliding-window rate limiter."""

import logging
import os
import threading
import time
from functools import wraps

import redis
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


logger = logging.getLogger(__name__)

# Sliding window over a sorted set. Returns -1 when a slot was taken, else
# the number of seconds until the oldest entry ages out of the window.
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count < max_requests then
  local seq = redis.call('INCR', key .. ':seq')
  redis.call('ZADD', key, now, now .. '-' .. seq)
  redis.call('EXPIRE', key, math.ceil(window * 2))
  redis.call('EXPIRE', key .. ':seq', math.ceil(window * 2))
  return -1
end
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
return math.max((tonumber(oldest[2]) + window) - now, 0.001)
"""

_WINDOW_SECONDS = 60.0


class RedisRateLimiter:
    """Per-project sliding-window limiter shared across processes via Redis.

    Advisory and fail-open: a Redis outage must never block external calls,
    so any redis error logs a warning and reports "acquired".
    """

    def __init__(self, redis_client=None):
        self._client = redis_client
        self._script = None

    def _get_script(self):
        if self._script is None:
            client = self._client
            if client is None:
                broker_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
                client = redis.from_url(broker_url)
                self._client = client
            self._script = client.register_script(_SLIDING_WINDOW_LUA)
        return self._script

    def try_acquire(self, service: str, max_per_minute: int) -> float | None:
        """Take a slot for ``service``. None = acquired; float = wait seconds."""
        project_ref = os.getenv("PROJECT_REF", "default")
        key = f"ratelimit:{service}:{project_ref}"
        try:
            result = self._get_script()(
                keys=[key],
                args=[time.time(), _WINDOW_SECONDS, max_per_minute],
            )
        except Exception:
            logger.warning("Rate limiter unavailable for %s — failing open", service, exc_info=True)
            return None
        wait = float(result)
        return None if wait < 0 else wait

    def acquire_blocking(self, service: str, max_per_minute: int, timeout_s: float = 10.0) -> bool:
        """Poll try_acquire until acquired or ``timeout_s`` elapses."""
        deadline = time.monotonic() + timeout_s
        while True:
            wait = self.try_acquire(service, max_per_minute)
            if wait is None:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(wait, 1.0))


external_limiter = RedisRateLimiter()
