"""Registry of runs that may wait for a human approval.

A run's ExecutionContext lives in the process that runs it: its ReAct thread
blocks on a threading.Event inside that context, so the context can never move.
Redis records which process owns each run (hash
``runs:owner:{PROJECT_REF}:{run_id}``), so the approve endpoint on any gunicorn
worker or replica can find the owner of a run it does not hold itself.

Redis is advisory and fail-open: when it is unreachable a run is still
registered locally, and an approval that reaches the owning process still
resumes it, exactly as before this registry used Redis. A process restart still
loses a waiting run; the stale-run cleanup marks it failed.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass

import redis
from agentic.execution.context import ExecutionContext

logger = logging.getLogger(__name__)

# Longer than any approval wait (the approval hook's default timeout is 300 s);
# the hash is deleted on unregister, the TTL only covers a process that died.
OWNER_TTL_SECONDS = 7200
REDIS_SOCKET_TIMEOUT_SECONDS = 2


def _project_ref() -> str:
    return os.getenv("PROJECT_REF", "default")


def _broker_client():
    return redis.from_url(
        os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
        socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
        socket_connect_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
    )


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
        self._instance = instance
        self._instance_pid: int | None = None
        self._start_thread = start_thread
        self._runs: dict[str, ExecutionContext] = {}
        self._owners: dict[str, str | None] = {}
        self._lock = threading.Lock()
        self._client_unavailable = False

    def instance_id(self) -> str:
        """``<HOSTNAME>:<pid>:<8 hex>``, regenerated in a forked child."""
        pid = os.getpid()
        if self._instance is None or (self._instance_pid is not None and self._instance_pid != pid):
            self._instance = f"{os.getenv('HOSTNAME', 'local')}:{pid}:{uuid.uuid4().hex[:8]}"
            self._instance_pid = pid
        return self._instance

    def _redis(self):
        if self._client is None:
            if self._client_unavailable:
                return None  # an unusable URL never becomes usable; warn once
            try:
                self._client = _broker_client()
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

    def register(
        self, run_id: str, context: ExecutionContext, owner_user_id: str | None = None
    ) -> None:
        with self._lock:
            self._runs[run_id] = context
            self._owners[run_id] = owner_user_id
        client = self._redis()
        if client is None:
            return
        key = self.owner_key(run_id)
        try:
            # One MULTI, so the hash never exists without its TTL.
            pipe = client.pipeline()
            pipe.hset(key, mapping={"instance": self.instance_id(), "owner": owner_user_id or ""})
            pipe.expire(key, OWNER_TTL_SECONDS)
            pipe.execute()
        except Exception:
            logger.warning(
                "Run registry: could not record the owner of run %s; approvals for it "
                "stay process-local",
                run_id,
                exc_info=True,
            )

    def unregister(self, run_id: str) -> None:
        with self._lock:
            self._runs.pop(run_id, None)
            self._owners.pop(run_id, None)
        client = self._redis()
        if client is None:
            return
        try:
            client.delete(self.owner_key(run_id))
        except Exception:
            logger.warning(
                "Run registry: could not delete the owner of run %s; it expires in %ds",
                run_id,
                OWNER_TTL_SECONDS,
                exc_info=True,
            )

    def context(self, run_id: str) -> ExecutionContext | None:
        return self._runs.get(run_id)

    def owner(self, run_id: str) -> str | None:
        return self._owners.get(run_id)

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
