"""Periodic cleanup tasks."""

import logging

from sqlalchemy import text

from ..celery import celery_app
from ..db import db, AI_SCHEMA
from ..services import billing_port as billing

logger = logging.getLogger(__name__)


@celery_app.task(name="cleanup_stale_runs")
@billing.no_billing_context
def cleanup_stale_runs():
    """Mark runs stuck in 'running' for >10 minutes as failed.

    This handles the case where a pod restarts mid-run and the
    in-process agent loop is lost. Covers both agent runs and
    orchestration runs.
    """
    agent_result = db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".agent_runs
            SET status = 'failed',
                error = 'Run interrupted (pod restart or timeout)',
                completed_at = NOW()
            WHERE status = 'running'
              AND started_at < NOW() - INTERVAL '10 minutes'
        """)
    )

    orch_result = db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".orchestration_runs
            SET status = 'failed',
                error = 'Run interrupted (pod restart or timeout)',
                completed_at = NOW()
            WHERE status = 'running'
              AND started_at < NOW() - INTERVAL '10 minutes'
        """)
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
