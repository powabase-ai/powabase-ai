"""Periodic cleanup tasks."""

import logging
import os
import time

from sqlalchemy import text

from ..celery import celery_app
from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
from ..services.broker_redis import broker_client
from ..services.settings_registry import get_setting

logger = logging.getLogger(__name__)

STALE_TICK_INTERVAL = 300  # seconds
STALE_LOCK_KEY = "stale_runs_lock"
STALE_LOCK_TTL = 60  # seconds


def run_stale_cleanup() -> dict:
    """Mark runs 'running' for longer than RUN_STALE_AFTER_SECONDS as failed.

    This handles a pod restart mid-run, which loses the in-process agent loop.
    Covers both agent runs and orchestration runs.
    """
    params = {"stale_after": int(get_setting("RUN_STALE_AFTER_SECONDS"))}
    agent_result = db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".agent_runs
            SET status = 'failed',
                error = 'Run interrupted (pod restart or timeout)',
                completed_at = NOW()
            WHERE status = 'running'
              AND started_at < NOW() - (:stale_after * INTERVAL '1 second')
        """),
        params,
    )

    orch_result = db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".orchestration_runs
            SET status = 'failed',
                error = 'Run interrupted (pod restart or timeout)',
                completed_at = NOW()
            WHERE status = 'running'
              AND started_at < NOW() - (:stale_after * INTERVAL '1 second')
        """),
        params,
    )

    db.session.commit()
    agent_count = agent_result.rowcount
    orch_count = orch_result.rowcount
    if agent_count > 0 or orch_count > 0:
        logger.info(
            "Cleaned up %d stale agent runs and %d stale orchestration runs",
            agent_count,
            orch_count,
        )
    return {"agent_runs": agent_count, "orchestration_runs": orch_count}


@celery_app.task(name="cleanup_stale_runs")
@billing.no_billing_context
def cleanup_stale_runs():
    return run_stale_cleanup()


def _get_redis():
    """Return (redis client, project-scoped lock key), like tasks/watchdog.py."""
    project_ref = os.getenv("PROJECT_REF", "default")
    return broker_client(), f"{project_ref}:{STALE_LOCK_KEY}"


@celery_app.task(name="stale_runs_tick", ignore_result=True, max_retries=0)
@billing.no_billing_context
def stale_runs_tick():
    """Run the stale-run cleanup, then re-enqueue self.

    The project stack runs no Celery beat, so the beat entry for
    cleanup_stale_runs never fires there; this self-rescheduling chain (seeded
    on worker_ready, like the indexed-sources watchdog) is what runs it. Every
    worker seeds a chain; the lock keeps the work to one at a time.
    """
    r, lock_key, got_lock = None, None, False
    try:
        r, lock_key = _get_redis()
        got_lock = r.set(lock_key, "1", nx=True, ex=STALE_LOCK_TTL)
    except Exception:
        logger.warning("Stale-run tick: Redis unavailable, skipping this tick", exc_info=True)
    if got_lock:
        try:
            run_stale_cleanup()
        except Exception:
            logger.exception("Stale-run cleanup failed")
        finally:
            try:
                r.delete(lock_key)
            except Exception:
                logger.warning("Failed to delete the stale-run lock", exc_info=True)

    for attempt in range(3):
        try:
            stale_runs_tick.apply_async(countdown=STALE_TICK_INTERVAL)
            break
        except Exception:
            if attempt == 2:
                logger.exception("Failed to re-enqueue the stale-run tick after 3 attempts")
            else:
                time.sleep(1)
