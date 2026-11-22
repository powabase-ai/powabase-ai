"""Periodic cleanup tasks."""

import logging
import os
import time

from sqlalchemy import text

from ..celery import celery_app
from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
from ..services.broker_redis import broker_client
from ..services.run_registry import RunRegistry
from ..services.settings_registry import get_setting

logger = logging.getLogger(__name__)

STALE_TICK_INTERVAL = 300  # seconds
STALE_LOCK_KEY = "stale_runs_lock"
STALE_LOCK_TTL = 60  # seconds


INTERRUPTED_ERROR = "Run interrupted (pod restart or timeout)"
# At most this many runs of each kind are failed per sweep; the next sweep
# takes the rest.
STALE_SWEEP_LIMIT = 1000

_TABLES = ("agent_runs", "orchestration_runs")


def _live_run_ids(client, run_ids: list[str]) -> set[str]:
    """The run ids that have an owner record: a process is running them and
    its registry listener renews the record (services/run_registry.py)."""
    pipe = client.pipeline(transaction=False)
    for run_id in run_ids:
        pipe.exists(RunRegistry.owner_key(run_id))
    return {run_id for run_id, live in zip(run_ids, pipe.execute(), strict=True) if live}


def run_stale_cleanup() -> dict:
    """Fail runs still 'running' after RUN_STALE_AFTER_SECONDS whose process is gone.

    This handles a pod restart mid-run, which loses the in-process run. A run
    is only failed when no process holds an owner record for it: a run that
    is merely long (a slow model, a long approval wait) keeps streaming. When
    Redis cannot say which runs are alive, nothing is failed this sweep.
    Covers both agent runs and orchestration runs.
    """
    params = {
        "stale_after": int(get_setting("RUN_STALE_AFTER_SECONDS")),
        "limit": STALE_SWEEP_LIMIT,
    }
    candidates = {}
    for table in _TABLES:
        candidates[table] = db.session.execute(
            text(f"""
                SELECT id, run_id FROM "{AI_SCHEMA}".{table}
                WHERE status = 'running'
                  AND started_at < NOW() - (:stale_after * INTERVAL '1 second')
                ORDER BY started_at
                LIMIT :limit
            """),
            params,
        ).fetchall()
    counts = {"agent_runs": 0, "orchestration_runs": 0}
    if not any(candidates.values()):
        return counts

    run_ids = [row[1] for rows in candidates.values() for row in rows if row[1]]
    try:
        live = _live_run_ids(broker_client(), run_ids) if run_ids else set()
    except Exception:
        logger.warning(
            "Stale-run cleanup: cannot tell which runs are alive (Redis unavailable); "
            "skipping this sweep",
            exc_info=True,
        )
        return counts

    for table in _TABLES:
        dead = [str(row[0]) for row in candidates[table] if row[1] not in live]
        if not dead:
            continue
        result = db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".{table}
                SET status = 'failed',
                    error = :error,
                    completed_at = NOW()
                WHERE id = ANY(CAST(:ids AS uuid[]))
                  AND status = 'running'
            """),
            {"ids": dead, "error": INTERRUPTED_ERROR},
        )
        counts[table] = result.rowcount
    db.session.commit()
    if counts["agent_runs"] or counts["orchestration_runs"]:
        logger.info(
            "Cleaned up %d stale agent runs and %d stale orchestration runs",
            counts["agent_runs"],
            counts["orchestration_runs"],
        )
    return counts


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
