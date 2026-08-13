"""Self-rescheduling docs-RAG refresh task.

Seeded on worker startup by `seed_docs_refresh` (celery.py) ONLY on the singleton
system docs project — gated by DOCS_KB_REFRESH_ENABLED. There is no Celery beat
process in this deployment, so the task re-arms itself (same pattern as
scheduler_tick / the watchdog). A Redis SET-NX lock makes the work safe when more
than one worker replica is running (each replica's chain re-arms, but only one
runs the refresh per tick) — mirrors tasks/watchdog.py.
"""

import logging
import os
import time
import uuid

import redis

from ..celery import celery_app
from ..services import billing_port as billing
from ..services.docs_refresh import refresh_docs_kb

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SECONDS = 21600  # 6h
_MIN_INTERVAL_SECONDS = 60  # floor: never tight-loop even if the env is mis-set
_LOCK_KEY = "docs_refresh_lock"
# Worst-case runtime is two 300s git-clone timeouts + llms-full fetch + ingest, so
# the TTL must exceed that or a slow run's lock expires and a second worker starts
# a concurrent refresh. 30 min covers it; the lock still auto-expires if a worker
# dies mid-run.
_LOCK_TTL = 1800  # seconds
# Compare-and-delete: only release the lock if WE still hold it (value matches our
# per-run token). An unconditional DELETE could drop a *different* worker's lock
# after ours expired mid-run.
_RELEASE_LOCK_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] "
    "then return redis.call('del', KEYS[1]) else return 0 end"
)


# Cached client — reused across ticks instead of building a fresh
# ConnectionPool every time the task fires.
_redis_client: redis.Redis | None = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        broker_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
        _redis_client = redis.from_url(broker_url)
    project_ref = os.getenv("PROJECT_REF", "default")
    return _redis_client, f"{project_ref}:{_LOCK_KEY}"


def _next_interval() -> float:
    try:
        interval = float(os.getenv("DOCS_KB_REFRESH_INTERVAL_SECONDS") or _DEFAULT_INTERVAL_SECONDS)
    except ValueError:
        interval = _DEFAULT_INTERVAL_SECONDS
    return max(_MIN_INTERVAL_SECONDS, interval)


@celery_app.task(
    name="agentic_project_service.tasks.docs_refresh.refresh_docs_kb_task",
    ignore_result=True,
    max_retries=0,
)
@billing.task_context
def refresh_docs_kb_task() -> None:
    if os.getenv("DOCS_KB_REFRESH_ENABLED", "false").lower() != "true":
        return  # not the docs project — don't run AND don't self-perpetuate
    try:
        r, lock_key = _get_redis()
        token = uuid.uuid4().hex
        if r.set(lock_key, token, nx=True, ex=_LOCK_TTL):
            try:
                result = refresh_docs_kb()
                logger.info("docs_refresh task complete: %s", result)
            except Exception:
                logger.error("docs_refresh task failed", exc_info=True)
            finally:
                try:
                    # Only delete the lock if it's still ours (compare-and-delete).
                    r.eval(_RELEASE_LOCK_LUA, 1, lock_key, token)
                except Exception:
                    logger.warning("Failed to release docs_refresh lock", exc_info=True)
        else:
            logger.debug("docs_refresh lock held by another worker, skipping")
    finally:
        # Re-arm the next run (no beat process exists to do it). Retry so a brief
        # broker blip doesn't kill the chain permanently — mirrors watchdog/
        # scheduler_tick; a total failure needs a worker restart to recover.
        for attempt in range(3):
            try:
                refresh_docs_kb_task.apply_async(countdown=_next_interval())
                break
            except Exception:
                if attempt == 2:
                    logger.error(
                        "Failed to re-arm docs_refresh chain after 3 attempts; "
                        "needs a worker restart",
                        exc_info=True,
                    )
                else:
                    time.sleep(1)
