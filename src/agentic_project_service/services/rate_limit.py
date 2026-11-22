"""Rate limiters: a per-process window (RateLimiter) and the Redis window shared
by every process of a project (RedisRateLimiter)."""

import logging
import math
import os
import threading
import time
from functools import wraps

from flask import g, jsonify

from ..auth import is_service_role_request
from .broker_redis import broker_client
from .settings_registry import get_setting


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


# Workflow executions per minute, across every process of the project: each end
# user (a JWT with a user id) has their own window, and the service role key one
# window per project, sized by the WORKFLOW_EXECUTIONS_PER_MINUTE setting.
EXECUTIONS_PER_MINUTE = 20
DEFAULT_SERVICE_EXECUTIONS_PER_MINUTE = 120


def _execution_bucket() -> tuple[str, int]:
    if is_service_role_request():
        return "workflow-exec:service", get_setting("WORKFLOW_EXECUTIONS_PER_MINUTE")
    user_id = getattr(g, "user_id", None) or "anonymous"
    return f"workflow-exec:{user_id}", EXECUTIONS_PER_MINUTE


def rate_limit_executions(f):
    """Rate-limit workflow execution endpoints across every gunicorn worker and
    replica of the project (Redis window; fail-open)."""

    @wraps(f)
    def decorated(*args, **kwargs):
        bucket, limit = _execution_bucket()
        # external_limiter is defined below; it is resolved at call time.
        wait = external_limiter.try_acquire(bucket, limit)
        if wait is not None:
            message = f"Rate limit exceeded. Max {limit} executions per minute."
            resp = jsonify({"error": message})
            resp.status_code = 429
            resp.headers["Retry-After"] = str(max(1, math.ceil(wait)))
            return resp
        return f(*args, **kwargs)

    return decorated


logger = logging.getLogger(__name__)

# Sliding window over a sorted set. Returns -1 when a slot was taken, else
# the number of MILLISECONDS until the oldest entry ages out of the window.
# Milliseconds because Redis converts Lua number replies to integers by
# truncation — a fractional seconds return would collapse sub-second waits
# to 0 (busy-spins in acquire_blocking, zero countdowns in callers).
# The seq counter is KEYS[2] so both keys are declared (Redis Cluster
# routes by declared keys; an undeclared second key breaks there).
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local seq_key = KEYS[2]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count < max_requests then
  local seq = redis.call('INCR', seq_key)
  redis.call('ZADD', key, now, now .. '-' .. seq)
  redis.call('EXPIRE', key, math.ceil(window * 2))
  redis.call('EXPIRE', seq_key, math.ceil(window * 2))
  return -1
end
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
return math.ceil(math.max(((tonumber(oldest[2]) + window) - now) * 1000, 1))
"""

_WINDOW_SECONDS = 60.0


class RedisRateLimiter:
    """Per-project sliding-window limiter shared across processes via Redis.

    Advisory and fail-open: a Redis outage must never block external calls,
    so any redis error logs a warning and reports "acquired".
    """

    def __init__(self, redis_client=None):
        self._client = redis_client
        self._client_unavailable = False
        self._script = None

    def _redis(self):
        """The Redis client, or None when none can be built (warned once)."""
        if self._client is None:
            if self._client_unavailable:
                return None  # an unusable URL never becomes usable
            try:
                self._client = broker_client()
            except Exception:
                self._client_unavailable = True
                logger.warning(
                    "Rate limiter: Redis client unavailable; failing open", exc_info=True
                )
                return None
        return self._client

    def client_available(self) -> bool:
        """False when no Redis client can be built: the limiter then fails open."""
        return self._redis() is not None

    def _get_script(self):
        if self._script is None:
            client = self._redis()
            if client is None:
                return None
            self._script = client.register_script(_SLIDING_WINDOW_LUA)
        return self._script

    def try_acquire(self, service: str, max_per_minute: int) -> float | None:
        """Take a slot for ``service``. None = acquired; float = wait seconds."""
        project_ref = os.getenv("PROJECT_REF", "default")
        key = f"ratelimit:{service}:{project_ref}"
        try:
            script = self._get_script()
            if script is None:
                return None
            result = script(
                keys=[key, f"{key}:seq"],
                args=[time.time(), _WINDOW_SECONDS, max_per_minute],
            )
        except Exception:
            logger.warning("Rate limiter unavailable for %s — failing open", service, exc_info=True)
            return None
        wait_ms = float(result)
        return None if wait_ms < 0 else wait_ms / 1000.0

    def acquire_blocking(self, service: str, max_per_minute: int, timeout_s: float = 10.0) -> bool:
        """Poll try_acquire until acquired or ``timeout_s`` elapses."""
        deadline = time.monotonic() + timeout_s
        while True:
            wait = self.try_acquire(service, max_per_minute)
            if wait is None:
                return True
            if time.monotonic() >= deadline:
                return False
            # Floor of 50ms: a near-zero wait hint must not busy-spin
            # against Redis (this can run on a request thread).
            time.sleep(min(max(wait, 0.05), 1.0))


external_limiter = RedisRateLimiter()
