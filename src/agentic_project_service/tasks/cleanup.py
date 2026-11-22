"""Periodic cleanup tasks."""

import logging

from sqlalchemy import text

from ..celery import celery_app
from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
from ..services.settings_registry import get_setting

logger = logging.getLogger(__name__)


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
