"""Admission control for runs (MAX_CONCURRENT_RUNS).

At most MAX_CONCURRENT_RUNS requests that run an agent, an orchestration or a
workflow are in flight at once in a project, across every gunicorn worker and
pod. A request over the limit is answered 429 with Retry-After before it takes a
gunicorn thread for the length of a run, so a caller can queue instead of
saturating the project.

The semaphore is a leased sorted set in the shared Redis,
``runs:active:{PROJECT_REF}``: one member per admitted request, score = lease
expiry by the Redis clock. Each process renews the leases it holds every
RENEW_SECONDS, so a process that dies frees its slots within LEASE_SECONDS.
Fail-open: when Redis cannot answer, the request runs ungated. The default
MAX_CONCURRENT_RUNS = 0 turns the gate off and makes no Redis call.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass

from .broker_redis import broker_client
from .extraction_gate import _REDIS_NOW

logger = logging.getLogger(__name__)

# A slot's lease and how often its holder renews it: a process that dies
# blocks its slots for at most LEASE_SECONDS; renewing three times per lease
# rides out a couple of failed renewals in a row.
LEASE_SECONDS = 60
RENEW_SECONDS = 20
RETRY_AFTER_SECONDS = 2
# /health?runs=1 and the metrics scrape read the count with a tighter bound
# than the request path, so a stuck Redis cannot stall a probe or a scrape.
HEALTH_SOCKET_TIMEOUT_SECONDS = 0.25

# -1 when a slot was taken, else the number of live holders.
_ACQUIRE_LUA = (
    _REDIS_NOW
    + """
local key = KEYS[1]
local lease = tonumber(ARGV[1])
local max_holders = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
local count = redis.call('ZCARD', key)
if count < max_holders then
  redis.call('ZADD', key, now + lease, ARGV[3])
  redis.call('EXPIRE', key, math.ceil(lease * 2))
  return -1
end
return count
"""
)

# 1 when the lease was still live and is extended, 0 when it is gone.
_RENEW_LUA = (
    _REDIS_NOW
    + """
local key = KEYS[1]
local lease = tonumber(ARGV[1])
local expires = redis.call('ZSCORE', key, ARGV[2])
if expires and tonumber(expires) > now then
  redis.call('ZADD', key, now + lease, ARGV[2])
  redis.call('EXPIRE', key, math.ceil(lease * 2))
  return 1
end
return 0
"""
)

_COUNT_LUA = (
    _REDIS_NOW
    + """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
return redis.call('ZCARD', KEYS[1])
"""
)


def _project_ref() -> str:
    return os.getenv("PROJECT_REF", "default")


@dataclass(frozen=True)
class Admission:
    admitted: bool
    token: str | None = None  # None when admitted without a slot (gate off or fail-open)
    active_runs: int | None = None  # set when refused


class RunGate:
    def __init__(self, redis_client=None, health_client=None, renew_in_background: bool = True):
        self._client = redis_client
        # An injected client (tests) also answers the health reads.
        self._health_client = health_client if health_client is not None else redis_client
        self._client_unavailable = False
        self._renew_in_background = renew_in_background
        self._scripts: dict[tuple[str, int], object] = {}
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._renewer: threading.Thread | None = None

    def key(self) -> str:
        return f"runs:active:{_project_ref()}"

    def _build(self, **timeouts):
        """A broker client, or None when none can be built (warned once)."""
        if self._client_unavailable:
            return None  # an unusable URL never becomes usable
        try:
            return broker_client(**timeouts)
        except Exception:
            self._client_unavailable = True
            logger.warning(
                "Run gate: Redis client unavailable; admitting runs ungated", exc_info=True
            )
            return None

    def _redis(self):
        if self._client is None:
            self._client = self._build()
        return self._client

    def _health_redis(self):
        if self._health_client is None:
            self._health_client = self._build(
                socket_timeout=HEALTH_SOCKET_TIMEOUT_SECONDS,
                connect_timeout=HEALTH_SOCKET_TIMEOUT_SECONDS,
            )
        return self._health_client

    def _script(self, name: str, source: str, client):
        cache_key = (name, id(client))
        script = self._scripts.get(cache_key)
        if script is None:
            script = self._scripts[cache_key] = client.register_script(source)
        return script

    def try_acquire(self, max_runs: int) -> Admission:
        if max_runs <= 0:
            return Admission(admitted=True)
        token = uuid.uuid4().hex
        try:
            client = self._redis()
            if client is None:
                return Admission(admitted=True)
            result = int(
                self._script("acquire", _ACQUIRE_LUA, client)(
                    keys=[self.key()], args=[LEASE_SECONDS, max_runs, token]
                )
            )
        except Exception:
            logger.warning("Run gate unavailable; admitting the run ungated", exc_info=True)
            return Admission(admitted=True)
        if result >= 0:
            return Admission(admitted=False, active_runs=result)
        with self._lock:
            self._held.add(token)
        self._ensure_renewer()
        return Admission(admitted=True, token=token)

    def release(self, token: str | None) -> None:
        if token is None:
            return
        with self._lock:
            self._held.discard(token)
        try:
            client = self._redis()
            if client is not None:
                client.zrem(self.key(), token)
        except Exception:
            logger.warning(
                "Could not release run slot %s; it expires within %ss",
                token,
                LEASE_SECONDS,
                exc_info=True,
            )

    def renew_all(self) -> int:
        """Extend every lease this process holds. Returns how many were renewed."""
        with self._lock:
            tokens = list(self._held)
        if not tokens:
            return 0
        client = self._redis()
        if client is None:
            return 0
        renewed = 0
        for token in tokens:
            try:
                ok = int(
                    self._script("renew", _RENEW_LUA, client)(
                        keys=[self.key()], args=[LEASE_SECONDS, token]
                    )
                )
            except Exception:
                logger.warning("Could not renew run slot %s", token, exc_info=True)
                continue
            if ok:
                renewed += 1
                continue
            with self._lock:
                # Released while this pass ran: not a lost lease.
                was_held = token in self._held
                self._held.discard(token)
            if was_held:
                logger.warning(
                    "Run slot %s expired before renewal; the project may briefly "
                    "exceed MAX_CONCURRENT_RUNS",
                    token,
                )
        return renewed

    def active_runs(self) -> int | None:
        """Live slots, project-wide. None when Redis cannot say (never raises)."""
        try:
            client = self._health_redis()
            if client is None:
                return None
            return int(self._script("count", _COUNT_LUA, client)(keys=[self.key()]))
        except Exception:
            logger.warning("Run gate unreadable; active_runs unknown", exc_info=True)
            return None

    def _ensure_renewer(self) -> None:
        if not self._renew_in_background:
            return
        with self._lock:
            # Also restarts after a fork: the child inherits a dead thread object.
            if self._renewer is not None and self._renewer.is_alive():
                return
            self._renewer = threading.Thread(
                target=self._renew_forever, name="run-gate-renew", daemon=True
            )
            self._renewer.start()

    def _renew_forever(self) -> None:
        while True:
            time.sleep(RENEW_SECONDS)
            try:
                self.renew_all()
            except Exception:
                logger.warning("Run gate renewal pass failed", exc_info=True)


default_gate = RunGate()
