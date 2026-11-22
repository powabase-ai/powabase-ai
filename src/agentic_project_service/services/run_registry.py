"""Registry of runs that may wait for a human approval.

A run's ExecutionContext lives in the process that runs it: its ReAct thread
blocks on a threading.Event inside that context, so the context can never move.
What moves is the decision. Redis records which process owns each run (hash
``runs:owner:{PROJECT_REF}:{run_id}``), and every process that registers a run
listens on its own channel ``runs:approvals:{PROJECT_REF}:{instance}``, so an
approval posted to any gunicorn worker on any replica is forwarded to the owner
(``deliver_remote``) and resumes the run there.

``register`` and ``unregister`` run on the request path and never touch Redis:
they update the local maps and leave the Redis side to one listener thread per
process. That thread subscribes (and waits for the confirmation) before it
writes any owner hash, so another process never sees an owner that nobody
listens for.

The owner hash lives only as long as a listener that answers: its TTL is
short and only the heartbeat renews it (a PING on the subscription every
HEARTBEAT_SECONDS; a PONG missing for PONG_TIMEOUT_SECONDS drops the
subscription, which is then rebuilt and every local run's hash written again).
And a forwarded decision counts as delivered only when the owner acknowledges
that it resumed the run, so the approve endpoint never answers "resumed" for a
run that cannot resume, even while Redis still counts a dead subscriber.

Separately, every run this process is running, approvable or not, has a
liveness record ``runs:alive:{PROJECT_REF}:{run_id}`` (value: the instance),
which the stale-run cleanup reads (tasks/cleanup.py). It says only that the
process is alive, so it does not wait for the subscription: the same thread
writes it with plain commands every HEARTBEAT_SECONDS, with a TTL of
OWNER_TTL_SECONDS, whether or not pub/sub works.

Redis is advisory and fail-open: when it is unreachable a run is still
registered locally, and an approval that reaches the owning process still
resumes it, exactly as before this registry used Redis. A process restart still
loses a waiting run; the stale-run cleanup marks it failed.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass

from agentic.execution.context import ExecutionContext

from .broker_redis import REDIS_SOCKET_TIMEOUT_SECONDS, broker_client

logger = logging.getLogger(__name__)

# The owner hash outlives a few missed heartbeats, not a dead listener: it is
# renewed on every PONG and deleted on unregister. The liveness record has the
# same TTL and is rewritten every HEARTBEAT_SECONDS.
OWNER_TTL_SECONDS = 60
HEARTBEAT_SECONDS = 15
PONG_TIMEOUT_SECONDS = 3
# How long the approving process waits for the owner to acknowledge a decision.
ACK_TIMEOUT_SECONDS = 3.0
ACK_POLL_SECONDS = 0.02
ACK_TTL_SECONDS = 30
# The approval hook reads only ``approved`` and ``reason``.
REASON_MAX_CHARS = 2000
# How long one listener iteration waits for a message; also the most a new
# registration waits before its owner hash is written.
LOOP_TIMEOUT_SECONDS = 0.1
RESUBSCRIBE_BACKOFF_SECONDS = 1.0
MAX_RESUBSCRIBE_BACKOFF_SECONDS = 30.0
# A failed owner-hash or liveness write (e.g. OOM under noeviction) is retried
# after this delay and warned about at most once per
# WRITE_WARNING_INTERVAL_SECONDS.
WRITE_RETRY_SECONDS = 1.0
WRITE_WARNING_INTERVAL_SECONDS = 60.0


def _project_ref() -> str:
    return os.getenv("PROJECT_REF", "default")


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


def wire_decision(decision: dict) -> dict:
    """The part of an approval body the approval hook reads, bounded in size.

    Only this goes to Redis: the request body is otherwise unbounded, and a
    large publish on the shared Redis could make it drop the subscriber.
    """
    wire = {"approved": bool(decision.get("approved"))}
    reason = decision.get("reason")
    if reason is not None:
        wire["reason"] = str(reason)[:REASON_MAX_CHARS]
    return wire


class Delivery(enum.Enum):
    """What became of a decision forwarded to another process."""

    RESUMED = "resumed"  # the owner handed the decision to the run
    GONE = "gone"  # nothing to resume: no listener, or the owner no longer has the run
    UNCONFIRMED = "unconfirmed"  # sent, but the owner never confirmed it
    UNAVAILABLE = "unavailable"  # not sent: Redis did not take the publish


class RegistryUnavailable(Exception):
    """Redis could not say who owns a run: not the same as "no such run"."""


@dataclass(frozen=True)
class RemoteRun:
    """Where a run that is not in this process lives, and who may approve it."""

    instance: str
    owner_user_id: str | None


class RunRegistry:
    def __init__(self, redis_client=None, instance: str | None = None, start_thread: bool = True):
        self._client = redis_client
        self._client_unavailable = False
        self._instance = instance
        self._instance_pid: int | None = None
        self._start_thread = start_thread
        # _lock guards the maps and the pending sets below, and is never held
        # over network I/O.
        self._lock = threading.Lock()
        self._runs: dict[str, ExecutionContext] = {}
        self._owners: dict[str, str | None] = {}
        # Runs with no approval gate: a liveness record says they are alive
        # (for the stale-run cleanup) but they are never approval targets.
        self._live: set[str] = set()
        self._dirty: set[str] = set()  # owner hashes to (re)write
        self._removed: set[str] = set()  # owner hashes to delete
        self._alive_dirty: set[str] = set()  # liveness records to write now
        self._alive_removed: set[str] = set()  # liveness records to delete
        # Owned by the listener thread (or by a test calling step()).
        self._pubsub = None
        self._subscribed = threading.Event()
        self._backoff = RESUBSCRIBE_BACKOFF_SECONDS
        self._retry_at = 0.0
        self._last_pong = 0.0
        self._ping_sent_at: float | None = None
        self._write_retry_at = 0.0
        self._alive_due = 0.0  # when every liveness record is next rewritten
        self._alive_retry_at = 0.0
        self._write_warned_at: float | None = None
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        self._stopped = threading.Event()

    def client_available(self) -> bool:
        """False when no Redis client can be built (an unusable broker URL):
        approvals are then process-local for the life of the process."""
        return self._redis() is not None

    def instance_id(self) -> str:
        """``<HOSTNAME>:<pid>:<8 hex>``, regenerated in a forked child."""
        pid = os.getpid()
        if self._instance is None or (self._instance_pid is not None and self._instance_pid != pid):
            self._instance = f"{os.getenv('HOSTNAME', 'local')}:{pid}:{uuid.uuid4().hex[:8]}"
            self._instance_pid = pid
        return self._instance

    def _redis(self):
        """The Redis client, or None when none can be built. Builds no connection."""
        if self._client is None:
            if self._client_unavailable:
                return None  # an unusable URL never becomes usable; warn once
            try:
                self._client = broker_client()
            except Exception:
                self._client_unavailable = True
                logger.warning(
                    "Run registry: Redis client unavailable; approvals stay process-local",
                    exc_info=True,
                )
                return None
        return self._client

    @staticmethod
    def owner_key(run_id: str) -> str:
        return f"runs:owner:{_project_ref()}:{run_id}"

    @staticmethod
    def alive_key(run_id: str) -> str:
        return f"runs:alive:{_project_ref()}:{run_id}"

    @staticmethod
    def channel(instance: str) -> str:
        return f"runs:approvals:{_project_ref()}:{instance}"

    @staticmethod
    def ack_key(token: str) -> str:
        return f"runs:ack:{_project_ref()}:{token}"

    # -- request path: local maps only, no Redis I/O ---------------------------

    def register(
        self, run_id: str, context: ExecutionContext, owner_user_id: str | None = None
    ) -> None:
        with self._lock:
            self._runs[run_id] = context
            self._owners[run_id] = owner_user_id
            self._dirty.add(run_id)
            self._removed.discard(run_id)
            self._alive_dirty.add(run_id)
            self._alive_removed.discard(run_id)
        self._ensure_listener()

    def register_live(self, run_id: str) -> None:
        """Publish that ``run_id`` is running in this process, without making it
        approvable: it gets a liveness record and no owner hash."""
        with self._lock:
            self._live.add(run_id)
            self._alive_dirty.add(run_id)
            self._alive_removed.discard(run_id)
        self._ensure_listener()

    def unregister(self, run_id: str) -> None:
        with self._lock:
            approvable = self._runs.pop(run_id, None) is not None
            known = approvable or run_id in self._live
            self._live.discard(run_id)
            self._owners.pop(run_id, None)
            self._dirty.discard(run_id)
            self._alive_dirty.discard(run_id)
            # Without a subscription the hash was either never written or is
            # rewritten from scratch after the next subscribe; it expires anyway.
            if approvable and self._subscribed.is_set():
                self._removed.add(run_id)
            if known:
                self._alive_removed.add(run_id)

    def context(self, run_id: str) -> ExecutionContext | None:
        return self._runs.get(run_id)

    def owner(self, run_id: str) -> str | None:
        return self._owners.get(run_id)

    def _ensure_listener(self) -> None:
        if not self._start_thread or self._stopped.is_set():
            return
        if self._thread is not None and self._thread.is_alive():
            return
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._redis() is None:
                return  # no Redis configured: no listener thread
            self._thread = threading.Thread(
                target=self._listen_forever, name="run-approvals", daemon=True
            )
            self._thread.start()

    # -- the approving process -------------------------------------------------

    def lookup_remote(self, run_id: str) -> RemoteRun | None:
        """The process that owns ``run_id`` according to Redis, or None when
        no process does (or no Redis is configured: approvals are then
        process-local).

        Raises:
            RegistryUnavailable: Redis did not answer.
        """
        client = self._redis()
        if client is None:
            return None
        try:
            data = client.hgetall(self.owner_key(run_id))
        except Exception as exc:
            logger.warning(
                "Run registry: could not read the owner of run %s", run_id, exc_info=True
            )
            raise RegistryUnavailable(f"could not read the owner of run {run_id}") from exc
        fields = {_text(k): _text(v) for k, v in (data or {}).items()}
        if not fields.get("instance"):
            return None
        return RemoteRun(instance=fields["instance"], owner_user_id=fields.get("owner") or None)

    def deliver_remote(self, run_id: str, remote: RemoteRun, decision: dict) -> Delivery:
        """Forward ``decision`` to the owning process.

        RESUMED when the owner acknowledged that it handed the decision to the
        run (it holds the run's context; whether the run was at an approval
        gate is the hook's business, as on the local path). GONE when nothing
        was delivered: no process listens on the owner's channel, or the owner
        no longer has the run. UNAVAILABLE when the publish itself failed.
        UNCONFIRMED when the decision was published but no acknowledgement
        arrived within ACK_TIMEOUT_SECONDS (a listener whose connection died
        without Redis noticing, or one that resumed the run but could not say
        so, or an acknowledgement that could not be read).
        """
        client = self._redis()
        if client is None:
            return Delivery.UNAVAILABLE
        token = uuid.uuid4().hex
        payload = json.dumps({"run_id": run_id, "decision": wire_decision(decision), "ack": token})
        try:
            if not client.publish(self.channel(remote.instance), payload):
                return Delivery.GONE
        except Exception:
            logger.warning(
                "Run registry: could not forward the approval of run %s", run_id, exc_info=True
            )
            return Delivery.UNAVAILABLE
        deadline = time.monotonic() + ACK_TIMEOUT_SECONDS
        while True:
            try:
                answer = client.get(self.ack_key(token))
            except Exception:
                logger.warning(
                    "Run registry: could not read the acknowledgement for run %s",
                    run_id,
                    exc_info=True,
                )
                return Delivery.UNCONFIRMED
            if answer is not None:
                return Delivery.RESUMED if _text(answer) == "1" else Delivery.GONE
            if time.monotonic() >= deadline:
                logger.warning(
                    "Run registry: process %s did not acknowledge the approval of run %s",
                    remote.instance,
                    run_id,
                )
                return Delivery.UNCONFIRMED
            time.sleep(ACK_POLL_SECONDS)

    # -- the owning process: the listener --------------------------------------

    def step(self, timeout: float | None = None) -> bool:
        """One listener iteration: write due liveness records, (re)subscribe if
        needed, write pending owner hashes, then wait up to ``timeout`` for one
        forwarded decision.

        Returns True iff a forwarded decision resumed a local run. Never raises:
        a failure of the subscription (connect, subscribe, PING, read) drops it and
        schedules a resubscribe; a failed owner-hash or liveness write is only
        retried.
        """
        timeout = LOOP_TIMEOUT_SECONDS if timeout is None else timeout
        self._write_liveness()
        if self._pubsub is None:
            if time.monotonic() < self._retry_at:
                return False
            try:
                self._connect()
            except Exception:
                logger.warning(
                    "Run registry: could not subscribe to approvals; retrying in %.0fs",
                    self._backoff,
                    exc_info=True,
                )
                self._drop()
                return False
        try:
            self._flush()
            self._heartbeat()
            return self._read(timeout)
        except Exception:
            logger.warning(
                "Run registry: approval channel failed; resubscribing in %.0fs",
                self._backoff,
                exc_info=True,
            )
            self._drop()
            return False

    def _connect(self) -> None:
        client = self._redis()
        if client is None:
            raise RuntimeError("no Redis client")
        pubsub = client.pubsub()
        try:
            pubsub.subscribe(self.channel(self.instance_id()))
            # SUBSCRIBE is only in effect once the server confirms it; until then a
            # PUBLISH from another process finds no receiver and reports the run gone.
            deadline = time.monotonic() + REDIS_SOCKET_TIMEOUT_SECONDS
            while (remaining := deadline - time.monotonic()) > 0:
                message = pubsub.get_message(timeout=remaining)
                if message is not None and message.get("type") == "subscribe":
                    break
            else:
                raise TimeoutError("Redis did not confirm the approvals subscription")
        except Exception:
            self._close_quietly(pubsub)
            raise
        self._pubsub = pubsub
        self._backoff = RESUBSCRIBE_BACKOFF_SECONDS
        self._last_pong = time.monotonic()
        self._ping_sent_at = None
        with self._lock:
            # Re-assert every approvable run: Redis may have lost or expired
            # their hashes while this process was not listening.
            self._dirty.update(self._runs)
        self._subscribed.set()

    def _flush(self) -> None:
        """Write pending owner hashes and deletes.

        A failure here is a command-side problem (Redis out of memory, a
        command connection reset), not a dead subscription: the changes stay
        pending and are retried, and the subscription keeps carrying decisions.
        """
        if time.monotonic() < self._write_retry_at:
            return
        with self._lock:
            dirty = {
                run_id: self._owners.get(run_id) for run_id in self._dirty if run_id in self._runs
            }
            removed = self._removed
            self._dirty, self._removed = set(), set()
        if not dirty and not removed:
            return
        try:
            pipe = self._redis().pipeline()
            instance = self.instance_id()
            for run_id, owner in dirty.items():
                key = self.owner_key(run_id)
                pipe.hset(key, mapping={"instance": instance, "owner": owner or ""})
                pipe.expire(key, OWNER_TTL_SECONDS)
            for run_id in removed:
                pipe.delete(self.owner_key(run_id))
            pipe.execute()
        except Exception as exc:  # noqa: BLE001 - logged by _warn_write_failure
            with self._lock:
                self._dirty.update(run_id for run_id in dirty if run_id in self._runs)
                self._removed.update(removed)
            self._write_retry_at = time.monotonic() + WRITE_RETRY_SECONDS
            self._warn_write_failure("run owner hashes", exc)

    def _write_liveness(self) -> None:
        """Write the liveness records of new runs, delete those of ended ones,
        and rewrite them all every HEARTBEAT_SECONDS. Plain commands only, so
        it works without the subscription. Never raises; a failure is retried."""
        now = time.monotonic()
        if now < self._alive_retry_at:
            return
        refresh = now >= self._alive_due
        with self._lock:
            local = set(self._runs) | self._live
            alive = local if refresh else self._alive_dirty & local
            removed = self._alive_removed - local
            self._alive_dirty, self._alive_removed = set(), set()
        if alive or removed:
            client = self._redis()
            if client is None:
                return
            try:
                pipe = client.pipeline(transaction=False)
                instance = self.instance_id()
                for run_id in alive:
                    pipe.set(self.alive_key(run_id), instance, ex=OWNER_TTL_SECONDS)
                for run_id in removed:
                    pipe.delete(self.alive_key(run_id))
                pipe.execute()
            except Exception as exc:  # noqa: BLE001 - logged by _warn_write_failure
                with self._lock:
                    self._alive_dirty.update(alive)
                    self._alive_removed.update(removed)
                self._alive_retry_at = time.monotonic() + WRITE_RETRY_SECONDS
                self._warn_write_failure("run liveness records", exc)
                return
        if refresh:
            self._alive_due = now + HEARTBEAT_SECONDS

    def _warn_write_failure(self, what: str, exc: BaseException) -> None:
        now = time.monotonic()
        if self._write_warned_at is None or now - self._write_warned_at >= (
            WRITE_WARNING_INTERVAL_SECONDS
        ):
            self._write_warned_at = now
            logger.warning(
                "Run registry: could not write %s; retrying (warned at most every %.0fs)",
                what,
                WRITE_WARNING_INTERVAL_SECONDS,
                exc_info=exc,
            )
        else:
            logger.debug("Run registry: %s write failed again", what, exc_info=exc)

    def _heartbeat(self) -> None:
        """PING the subscription; a PONG that does not come means it is dead.

        An idle subscriber never writes, so a connection lost without a FIN or
        RST would otherwise go unnoticed for good.
        """
        now = time.monotonic()
        if self._ping_sent_at is not None:
            if now - self._ping_sent_at > PONG_TIMEOUT_SECONDS:
                raise TimeoutError(f"no PONG on the approvals channel in {PONG_TIMEOUT_SECONDS}s")
        elif now - self._last_pong >= HEARTBEAT_SECONDS:
            self._pubsub.ping()
            self._ping_sent_at = now

    def _read(self, timeout: float) -> bool:
        message = self._pubsub.get_message(timeout=timeout)
        if message is None:
            return False
        if message.get("type") == "pong":
            self._last_pong = time.monotonic()
            self._ping_sent_at = None
            with self._lock:
                self._dirty.update(self._runs)  # renew every owner hash's TTL
            return False
        if message.get("type") != "message":
            return False
        return self._apply(message["data"])

    def _apply(self, data) -> bool:
        try:
            payload = json.loads(data)
            run_id = payload["run_id"]
            decision = payload["decision"]
            if not isinstance(decision, dict):
                raise TypeError("decision is not an object")
        except (ValueError, KeyError, TypeError):
            logger.warning("Run registry: ignoring a malformed approval message")
            return False
        context = self.context(run_id)
        resumed = False
        if context is None:
            logger.info("Run registry: approval for run %s arrived after it ended", run_id)
        else:
            # A failure here belongs to this decision, not to the channel: the
            # subscription stays, and the approver hears "not resumed".
            try:
                context.set_approval_decision(decision)
                resumed = True
            except Exception:
                logger.exception("Run registry: could not resume run %s", run_id)
        self._acknowledge(payload.get("ack"), resumed=resumed)
        return resumed

    def _acknowledge(self, token, resumed: bool) -> None:
        if not isinstance(token, str) or len(token) != 32 or not token.isalnum():
            return
        try:
            self._redis().set(self.ack_key(token), "1" if resumed else "0", ex=ACK_TTL_SECONDS)
        except Exception:
            logger.warning("Run registry: could not acknowledge an approval", exc_info=True)

    def _drop(self) -> None:
        self._subscribed.clear()
        pubsub, self._pubsub = self._pubsub, None
        self._close_quietly(pubsub)
        with self._lock:
            # Deletes are moot once nobody listens: rewrites follow the next
            # subscribe, and a stale hash only ever leads to a 404.
            self._removed = set()
        self._retry_at = time.monotonic() + self._backoff
        self._backoff = min(self._backoff * 2, MAX_RESUBSCRIBE_BACKOFF_SECONDS)

    @staticmethod
    def _close_quietly(pubsub) -> None:
        if pubsub is None:
            return
        try:
            pubsub.close()
        except Exception:
            logger.debug("Run registry: closing the approvals pubsub", exc_info=True)

    def _listen_forever(self) -> None:
        while not self._stopped.is_set():
            self.step()
            if self._pubsub is None:
                # Not subscribed: wake for the next resubscribe or liveness
                # write, whichever is first.
                now = time.monotonic()
                wake = min(self._retry_at, max(self._alive_due, self._alive_retry_at))
                with self._lock:
                    if self._alive_dirty or self._alive_removed:
                        wake = min(wake, max(now, self._alive_retry_at))
                self._stopped.wait(max(wake - now, LOOP_TIMEOUT_SECONDS))
        # close() may have run while this thread was still connecting.
        pubsub, self._pubsub = self._pubsub, None
        self._close_quietly(pubsub)

    def close(self) -> None:
        """Stop listening for forwarded approvals (tests; process shutdown)."""
        self._stopped.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)
        self._subscribed.clear()
        pubsub, self._pubsub = self._pubsub, None
        self._close_quietly(pubsub)


_default = RunRegistry()


def register_run(run_id: str, context: ExecutionContext, owner_user_id: str | None = None):
    _default.register(run_id, context, owner_user_id)


def register_live_run(run_id: str):
    """Publish a run that has no approval gate as alive (see RunRegistry.register_live)."""
    _default.register_live(run_id)


def get_active_run_context(run_id: str) -> ExecutionContext | None:
    return _default.context(run_id)


def get_active_run_owner(run_id: str) -> str | None:
    return _default.owner(run_id)


def unregister_run(run_id: str):
    _default.unregister(run_id)


def lookup_remote_run(run_id: str) -> RemoteRun | None:
    return _default.lookup_remote(run_id)


def deliver_approval(run_id: str, remote: RemoteRun, decision: dict) -> Delivery:
    return _default.deliver_remote(run_id, remote, decision)


def client_available() -> bool:
    return _default.client_available()
