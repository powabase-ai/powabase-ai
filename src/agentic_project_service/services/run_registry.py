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

Redis is advisory and fail-open: when it is unreachable a run is still
registered locally, and an approval that reaches the owning process still
resumes it, exactly as before this registry used Redis. A process restart still
loses a waiting run; the stale-run cleanup marks it failed.
"""

from __future__ import annotations

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

# Longer than any approval wait (the approval hook's default timeout is 300 s);
# the hash is deleted on unregister, the TTL only covers a process that died.
OWNER_TTL_SECONDS = 7200
# How long one listener iteration waits for a message; also the most a new
# registration waits before its owner hash is written.
LOOP_TIMEOUT_SECONDS = 0.1
RESUBSCRIBE_BACKOFF_SECONDS = 1.0
MAX_RESUBSCRIBE_BACKOFF_SECONDS = 30.0


def _project_ref() -> str:
    return os.getenv("PROJECT_REF", "default")


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


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
        self._dirty: set[str] = set()  # owner hashes to (re)write
        self._removed: set[str] = set()  # owner hashes to delete
        # Owned by the listener thread (or by a test calling step()).
        self._pubsub = None
        self._subscribed = threading.Event()
        self._backoff = RESUBSCRIBE_BACKOFF_SECONDS
        self._retry_at = 0.0
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        self._stopped = threading.Event()

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
    def channel(instance: str) -> str:
        return f"runs:approvals:{_project_ref()}:{instance}"

    # -- request path: local maps only, no Redis I/O ---------------------------

    def register(
        self, run_id: str, context: ExecutionContext, owner_user_id: str | None = None
    ) -> None:
        with self._lock:
            self._runs[run_id] = context
            self._owners[run_id] = owner_user_id
            self._dirty.add(run_id)
            self._removed.discard(run_id)
        self._ensure_listener()

    def unregister(self, run_id: str) -> None:
        with self._lock:
            known = self._runs.pop(run_id, None) is not None
            self._owners.pop(run_id, None)
            self._dirty.discard(run_id)
            # Without a subscription the hash was either never written or is
            # rewritten from scratch after the next subscribe; it expires anyway.
            if known and self._subscribed.is_set():
                self._removed.add(run_id)

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
        """The process that owns ``run_id`` according to Redis, or None."""
        client = self._redis()
        if client is None:
            return None
        try:
            data = client.hgetall(self.owner_key(run_id))
        except Exception:
            logger.warning(
                "Run registry: could not read the owner of run %s", run_id, exc_info=True
            )
            return None
        fields = {_text(k): _text(v) for k, v in (data or {}).items()}
        if not fields.get("instance"):
            return None
        return RemoteRun(instance=fields["instance"], owner_user_id=fields.get("owner") or None)

    def deliver_remote(self, run_id: str, remote: RemoteRun, decision: dict) -> bool:
        """Publish ``decision`` to the owner's channel; True iff it was received."""
        client = self._redis()
        if client is None:
            return False
        payload = json.dumps({"run_id": run_id, "decision": decision})
        try:
            receivers = client.publish(self.channel(remote.instance), payload)
        except Exception:
            logger.warning(
                "Run registry: could not forward the approval of run %s", run_id, exc_info=True
            )
            return False
        return int(receivers) > 0

    # -- the owning process: the listener --------------------------------------

    def step(self, timeout: float | None = None) -> bool:
        """One listener iteration: (re)subscribe if needed, write pending owner
        hashes, then wait up to ``timeout`` for one forwarded decision.

        Returns True iff a forwarded decision resumed a local run. Never raises:
        a Redis failure drops the subscription and schedules a resubscribe.
        """
        timeout = LOOP_TIMEOUT_SECONDS if timeout is None else timeout
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
        with self._lock:
            # Re-assert every local run: Redis may have lost or expired them
            # while this process was not listening.
            self._dirty.update(self._runs)
        self._subscribed.set()

    def _flush(self) -> None:
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
        except Exception:
            with self._lock:
                self._dirty.update(run_id for run_id in dirty if run_id in self._runs)
            raise

    def _read(self, timeout: float) -> bool:
        message = self._pubsub.get_message(timeout=timeout)
        if message is None or message.get("type") != "message":
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
        if context is None:
            logger.info("Run registry: approval for run %s arrived after it ended", run_id)
            return False
        context.set_approval_decision(decision)
        return True

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
                self._stopped.wait(max(self._retry_at - time.monotonic(), LOOP_TIMEOUT_SECONDS))
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


def get_active_run_context(run_id: str) -> ExecutionContext | None:
    return _default.context(run_id)


def get_active_run_owner(run_id: str) -> str | None:
    return _default.owner(run_id)


def unregister_run(run_id: str):
    _default.unregister(run_id)


def lookup_remote_run(run_id: str) -> RemoteRun | None:
    return _default.lookup_remote(run_id)


def deliver_approval(run_id: str, remote: RemoteRun, decision: dict) -> bool:
    return _default.deliver_remote(run_id, remote, decision)
