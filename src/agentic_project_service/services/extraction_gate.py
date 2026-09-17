"""Limits on extracting large source files, shared by every worker of a project.

Two things live here, both kept in the project's Redis (the Celery broker) so
they hold across worker processes and pods:

* ``LargeExtractionGate`` — a counting semaphore. At most
  ``EXTRACTION_LARGE_MAX_CONCURRENT`` extractions of files larger than
  ``EXTRACTION_LARGE_FILE_BYTES`` run at once. Each slot is a lease that the
  holder renews from a heartbeat thread, so a holder that is killed stops
  blocking the others when its lease runs out rather than never.

* ``ExtractionAttempts`` — which extractions were in flight in each worker
  process, and how large their files were. When a worker is killed, every task
  it was running is redelivered; this record lets a redelivered task tell
  whether it was the plausible cause of the kill (the largest file in flight)
  or a bystander.

Both fail open: Redis is also the broker, so when it is unreachable no task is
being delivered anyway, and an extraction must not fail over bookkeeping.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass

import redis

logger = logging.getLogger(__name__)

DEFAULT_LARGE_FILE_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_CONCURRENT_LARGE = 1

# A slot's lease, and how often its holder renews it. A killed holder blocks
# the slot for at most LEASE_SECONDS; renewing several times per lease rides
# out a few failed renewals in a row.
LEASE_SECONDS = 300
RENEW_SECONDS = 60

# Bounds every Redis call made here, so a half-open connection cannot hang a
# heartbeat, or the release in a task's `finally`.
REDIS_SOCKET_TIMEOUT_SECONDS = 10

try:
    from prometheus_client import Counter

    _fail_open_total = Counter(
        "extraction_gate_fail_open_total",
        "Large-extraction gate operations that could not reach Redis and let "
        "extraction proceed ungated or unrecorded",
        ["operation"],
    )
except ImportError:  # pragma: no cover - prometheus is an optional dependency
    _fail_open_total = None


def _count_fail_open(operation: str) -> None:
    if _fail_open_total is not None:
        _fail_open_total.labels(operation=operation).inc()


def fail_open_count(operation: str) -> float:
    """Current value of the fail-open counter for *operation* (0 without prometheus)."""
    if _fail_open_total is None:
        return 0.0
    return _fail_open_total.labels(operation=operation)._value.get()


class Renewal(enum.Enum):
    RENEWED = "renewed"
    LOST = "lost"  # the lease had expired or was removed; the slot is not ours
    ERROR = "error"  # Redis could not be asked; the lease may well still be live


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        logger.warning("%s=%r is not a positive integer; using %d", name, raw, default)
        return default
    return value


def large_file_threshold() -> int:
    """Files larger than this many bytes are extracted under the gate."""
    return _positive_int_env("EXTRACTION_LARGE_FILE_BYTES", DEFAULT_LARGE_FILE_BYTES)


def max_concurrent_large_extractions() -> int:
    return _positive_int_env("EXTRACTION_LARGE_MAX_CONCURRENT", DEFAULT_MAX_CONCURRENT_LARGE)


def is_large_file(size: int | None) -> bool:
    """A file of unknown size is treated as large: the gate costs it a wait,
    while guessing small could put several huge files in one worker."""
    return size is None or size > large_file_threshold()


def _project_ref() -> str:
    return os.getenv("PROJECT_REF", "default")


def _broker_client():
    return redis.from_url(
        os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
        socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
        socket_connect_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
    )


def task_time_limit_seconds() -> int:
    """The task time limit, read the same way the Celery configuration reads
    it. It is also the broker's visibility timeout: an unacknowledged task is
    handed out again once this much time has passed."""
    return int(os.getenv("CELERY_TASK_TIME_LIMIT") or 21600)


def max_hold_seconds() -> int:
    """Longest a slot is kept renewed.

    The threads pool does not enforce the task time limit, so without a cap a
    hung extraction would hold its slot forever. The cap sits a lease and a
    minute below that limit, so by the time the broker redelivers an
    unfinished task its lease has lapsed, and the redelivery is judged as an
    interruption rather than first waiting on a lease that is about to vanish
    and then running beside the original.
    """
    return task_time_limit_seconds() - LEASE_SECONDS - 60


# Lease times come from the Redis server clock, never a worker's: pods whose
# clocks disagree would otherwise prune each other's live leases.
_REDIS_NOW = """
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
"""


# Prunes expired leases, then takes a slot if one is free. Scores are lease
# expiry times.
_ACQUIRE_LUA = (
    _REDIS_NOW
    + """
local key = KEYS[1]
local lease = tonumber(ARGV[1])
local max_holders = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
if redis.call('ZCARD', key) < max_holders then
  redis.call('ZADD', key, now + lease, ARGV[3])
  redis.call('EXPIRE', key, math.ceil(lease * 2))
  return 1
end
return 0
"""
)

# Extends a lease that has not yet expired. An expired one is not revived:
# its slot may already have gone to someone else.
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

_LIVE_LUA = (
    _REDIS_NOW
    + """
local expires = redis.call('ZSCORE', KEYS[1], ARGV[1])
if expires and tonumber(expires) > now then
  return 1
end
return 0
"""
)


class LargeExtractionSlot:
    """A held slot. ``token`` is None when the gate was unavailable and the
    extraction runs ungated."""

    def __init__(self, gate: LargeExtractionGate, token: str | None, heartbeat: bool):
        self._gate = gate
        self.token = token
        self._stop = threading.Event()
        self._heartbeat: threading.Thread | None = None
        if token is not None and heartbeat:
            self._heartbeat = threading.Thread(
                target=self._renew_until_released, name="large-extraction-lease", daemon=True
            )
            self._heartbeat.start()

    def _renew_until_released(self) -> None:
        give_up_at = time.monotonic() + max_hold_seconds()
        while not self._stop.wait(RENEW_SECONDS):
            if time.monotonic() >= give_up_at:
                logger.error(
                    "Large-extraction slot %s has been held longer than the task time "
                    "limit (%ss); no longer renewing it, so it frees within %ss",
                    self.token,
                    max_hold_seconds(),
                    LEASE_SECONDS,
                )
                return
            outcome = self._gate.renew(self.token)
            if outcome is Renewal.LOST:
                logger.warning(
                    "Large-extraction lease %s expired before it could be renewed; "
                    "another large extraction may start alongside this one",
                    self.token,
                )
                return
            # Renewal.ERROR: already logged; the lease is probably still live,
            # so keep trying on schedule.

    def release(self) -> None:
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=5)
        if self.token is not None:
            self._gate._release(self.token)


class LargeExtractionGate:
    def __init__(self, redis_client=None, heartbeat: bool = True):
        self._client = redis_client
        self._heartbeat = heartbeat
        self._scripts: dict[str, object] = {}

    def _key(self) -> str:
        return f"extraction:large-slots:{_project_ref()}"

    def _redis(self):
        if self._client is None:
            self._client = _broker_client()
        return self._client

    def _script(self, name: str, source: str):
        if name not in self._scripts:
            self._scripts[name] = self._redis().register_script(source)
        return self._scripts[name]

    def try_acquire(self, task_id: str) -> LargeExtractionSlot | None:
        """Take a slot. None means every slot is held."""
        token = f"{task_id}:{uuid.uuid4().hex}"
        try:
            taken = self._script("acquire", _ACQUIRE_LUA)(
                keys=[self._key()],
                args=[LEASE_SECONDS, max_concurrent_large_extractions(), token],
            )
        except Exception:
            logger.warning(
                "Large-extraction gate unavailable; extracting task %s ungated",
                task_id,
                exc_info=True,
            )
            _count_fail_open("acquire")
            return LargeExtractionSlot(self, None, heartbeat=False)
        if not int(taken):
            return None
        return LargeExtractionSlot(self, token, heartbeat=self._heartbeat)

    def renew(self, token: str) -> Renewal:
        try:
            renewed = self._script("renew", _RENEW_LUA)(
                keys=[self._key()], args=[LEASE_SECONDS, token]
            )
        except Exception:
            logger.warning("Could not renew large-extraction lease %s", token, exc_info=True)
            return Renewal.ERROR
        return Renewal.RENEWED if int(renewed) else Renewal.LOST

    def is_live(self, token: str) -> bool:
        """Whether *token* still holds a slot. False when it cannot be told."""
        try:
            live = self._script("live", _LIVE_LUA)(keys=[self._key()], args=[token])
        except Exception:
            logger.warning("Could not read large-extraction lease %s", token, exc_info=True)
            return False
        return bool(int(live))

    def _release(self, token: str) -> None:
        try:
            self._redis().zrem(self._key(), token)
        except Exception:
            logger.warning(
                "Could not release large-extraction lease %s; it expires in %ds",
                token,
                LEASE_SECONDS,
                exc_info=True,
            )


_incarnation: tuple[int, str] | None = None


def process_incarnation() -> str:
    """Identifies this worker process for the life of the process. A restarted
    worker, even with the same hostname and pid, is a new incarnation."""
    global _incarnation
    pid = os.getpid()
    if _incarnation is None or _incarnation[0] != pid:
        _incarnation = (pid, f"{socket.gethostname()}:{pid}:{uuid.uuid4().hex[:12]}")
    return _incarnation[1]


def _record_ttl_seconds() -> int:
    # The broker redelivers a killed task after its visibility timeout, which
    # is the task time limit; the record has to outlive that wait.
    return 2 * task_time_limit_seconds() + 3600


@dataclass(frozen=True)
class PreviousAttempt:
    """What was recorded about a task's last attempt, which never finished."""

    slot_token: str | None
    size: int | None
    largest_in_flight: int | None

    @property
    def plausible_cause(self) -> bool:
        """True unless a strictly larger file was in flight in the same worker.

        Every kill leaves at least one task for which this is True (the
        largest), so a file that keeps killing workers is always charged.
        """
        if self.size is None or self.largest_in_flight is None:
            return True
        return self.size >= self.largest_in_flight


class ExtractionAttempts:
    def __init__(self, redis_client=None, incarnation: str | None = None):
        self._client = redis_client
        self._incarnation = incarnation

    @property
    def incarnation(self) -> str:
        return self._incarnation or process_incarnation()

    def _redis(self):
        if self._client is None:
            self._client = _broker_client()
        return self._client

    @staticmethod
    def _in_flight_key(incarnation: str) -> str:
        return f"extraction:in-flight:{_project_ref()}:{incarnation}"

    @staticmethod
    def _attempt_key(task_id: str) -> str:
        return f"extraction:attempt:{_project_ref()}:{task_id}"

    def begin(self, task_id: str, size: int | None, slot_token: str | None) -> None:
        incarnation = self.incarnation
        ttl = _record_ttl_seconds()
        in_flight = self._in_flight_key(incarnation)
        record = json.dumps({"incarnation": incarnation, "size": size, "slot_token": slot_token})
        try:
            pipe = self._redis().pipeline()
            pipe.zadd(in_flight, {task_id: float("inf") if size is None else size})
            pipe.expire(in_flight, ttl)
            pipe.set(self._attempt_key(task_id), record, ex=ttl)
            pipe.execute()
        except Exception:
            logger.warning(
                "Could not record extraction attempt %s; if this worker dies the task "
                "is charged for it",
                task_id,
                exc_info=True,
            )
            _count_fail_open("record_attempt")

    def end(self, task_id: str) -> None:
        try:
            pipe = self._redis().pipeline()
            pipe.zrem(self._in_flight_key(self.incarnation), task_id)
            pipe.delete(self._attempt_key(task_id))
            pipe.execute()
        except Exception:
            logger.warning("Could not clear extraction attempt %s", task_id, exc_info=True)

    def previous(self, task_id: str) -> PreviousAttempt | None:
        try:
            client = self._redis()
            raw = client.get(self._attempt_key(task_id))
            if raw is None:
                return None
            record = json.loads(raw)
            peers = client.zrevrange(
                self._in_flight_key(record["incarnation"]), 0, 0, withscores=True
            )
        except Exception:
            logger.warning("Could not read extraction attempt %s", task_id, exc_info=True)
            return None
        largest = None
        if peers:
            # A file of unknown size in flight counts as larger than any known one.
            score = peers[0][1]
            largest = sys.maxsize if score == float("inf") else int(score)
        return PreviousAttempt(record.get("slot_token"), record.get("size"), largest)


# A waiting delivery refreshes its record every time it wakes (about once a
# minute). One nobody has refreshed for this long belongs to a message that
# was lost with the worker holding it.
WAITING_STALE_SECONDS = 600


class WaitingExtractions:
    """Deliveries sitting in some worker as countdown messages.

    A task waiting for a large-extraction slot, or for a retry, is held by
    the worker that fetched it until it is due. If that worker is killed, the
    broker hands the message back only after its visibility timeout (the task
    time limit, six hours by default). Recording each wait lets the project
    notice a wait nobody has refreshed and dispatch the task again.
    """

    def __init__(self, redis_client=None):
        self._client = redis_client

    def _redis(self):
        if self._client is None:
            self._client = _broker_client()
        return self._client

    @staticmethod
    def _index_key() -> str:
        return f"extraction:waiting:{_project_ref()}"

    @staticmethod
    def _record_key(task_id: str) -> str:
        return f"extraction:waiting:{_project_ref()}:{task_id}"

    def _now(self) -> float:
        seconds, micros = self._redis().time()
        return seconds + micros / 1_000_000

    def touch(self, task_id: str, source_id: str, dispatch: dict) -> None:
        """Record (or refresh) that *task_id* is waiting to run again.

        *dispatch* holds what is needed to send it again: ``args``,
        ``kwargs`` and ``retries``.
        """
        ttl = _record_ttl_seconds()
        try:
            now = self._now()
            pipe = self._redis().pipeline()
            pipe.set(
                self._record_key(task_id),
                json.dumps({"source_id": source_id, "dispatch": dispatch}),
                ex=ttl,
            )
            pipe.zadd(self._index_key(), {task_id: now})
            pipe.expire(self._index_key(), ttl)
            pipe.execute()
        except Exception:
            logger.warning(
                "Could not record that extraction task %s is waiting; if its worker dies "
                "it is not recovered before the broker redelivers it",
                task_id,
                exc_info=True,
            )
            _count_fail_open("record_waiting")

    def clear(self, task_id: str) -> None:
        try:
            pipe = self._redis().pipeline()
            pipe.delete(self._record_key(task_id))
            pipe.zrem(self._index_key(), task_id)
            pipe.execute()
        except Exception:
            logger.warning("Could not clear waiting record of task %s", task_id, exc_info=True)

    def get(self, task_id: str) -> dict | None:
        try:
            raw = self._redis().get(self._record_key(task_id))
        except Exception:
            logger.warning("Could not read waiting record of task %s", task_id, exc_info=True)
            return None
        return None if raw is None else json.loads(raw)

    def stale(self, older_than: float) -> list[tuple[str, dict]]:
        """Waits not refreshed for *older_than* seconds, oldest first."""
        try:
            client = self._redis()
            cutoff = self._now() - older_than
            task_ids = client.zrangebyscore(self._index_key(), "-inf", cutoff)
            stale = []
            for raw_id in task_ids:
                task_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
                record = self.get(task_id)
                if record is None:
                    client.zrem(self._index_key(), task_id)
                    continue
                stale.append((task_id, record))
            return stale
        except Exception:
            logger.warning("Could not list waiting extraction tasks", exc_info=True)
            return []

    def any_waiting(self) -> bool:
        try:
            return bool(self._redis().zcard(self._index_key()))
        except Exception:
            return False

    def claim_sweep(self, interval: float) -> bool:
        """True for the one caller per *interval* that should schedule a sweep."""
        try:
            return bool(
                self._redis().set(
                    f"extraction:waiting-sweep:{_project_ref()}", "1", nx=True, ex=int(interval)
                )
            )
        except Exception:
            logger.warning("Could not coordinate the waiting-extraction sweep", exc_info=True)
            return False
