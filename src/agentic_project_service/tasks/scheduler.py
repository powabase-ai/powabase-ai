"""Self-rescheduling Celery task for workflow scheduled execution.

Runs every 30 seconds, checks for deployed workflows with active schedules,
and executes any that are due. Uses a Redis lock to prevent duplicate tick
chains on worker restart.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime

from croniter import croniter
from sqlalchemy import text
from werkzeug.exceptions import BadRequest

from ..celery import celery_app
from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
from ..services.llm_availability import check_model_available

logger = logging.getLogger(__name__)

TICK_INTERVAL = 30  # seconds between scheduler ticks
LOCK_KEY = "scheduler_tick_lock"
LOCK_TTL = 60  # seconds
EXECUTION_TIMEOUT = 600  # 10 min for scheduled runs


def _get_redis():
    """Get the Redis client from the Celery broker connection."""
    import redis

    broker_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
    project_ref = os.getenv("PROJECT_REF", "default")
    r = redis.from_url(broker_url)
    # Use project-scoped lock key
    return r, f"{project_ref}:{LOCK_KEY}"


@celery_app.task(bind=True, ignore_result=True, max_retries=0)
@billing.task_context
def scheduler_tick(self):
    """Check for due scheduled workflows and execute them, then re-enqueue."""
    r, lock_key = _get_redis()

    # Acquire lock to prevent duplicate tick chains
    if not r.set(lock_key, "1", nx=True, ex=LOCK_TTL):
        logger.debug("Scheduler tick lock held by another worker, skipping")
        return

    try:
        _process_scheduled_workflows()
    except Exception:
        logger.error("Scheduler tick failed", exc_info=True)
    finally:
        # Release lock so next tick can proceed
        try:
            r.delete(lock_key)
        except Exception:
            logger.warning("Failed to delete scheduler lock", exc_info=True)
        # Re-enqueue the next tick with retry
        for attempt in range(3):
            try:
                scheduler_tick.apply_async(countdown=TICK_INTERVAL)
                break
            except Exception:
                if attempt == 2:
                    logger.error(
                        "Failed to re-enqueue scheduler tick after 3 attempts",
                        exc_info=True,
                    )
                else:
                    time.sleep(1)


def _process_scheduled_workflows():
    """Query and execute all due scheduled workflows."""
    rows = db.session.execute(
        text(f"""
            SELECT id, schedule_config, schedule_run_count, last_scheduled_at
            FROM "{AI_SCHEMA}".workflows
            WHERE state = 'deployed'
              AND schedule_config IS NOT NULL
              AND (schedule_config->>'enabled')::boolean = true
        """)
    ).fetchall()

    for row in rows:
        wf_id = str(row[0])
        sched = row[1]
        run_count = row[2] or 0
        last_scheduled_at = row[3]

        try:
            _maybe_execute_scheduled(wf_id, sched, run_count, last_scheduled_at)
        except Exception:
            logger.error("Failed to process schedule for workflow %s", wf_id, exc_info=True)


def _maybe_execute_scheduled(
    wf_id: str,
    sched: dict,
    run_count: int,
    last_scheduled_at: datetime | None,
):
    """Check if a scheduled workflow is due and execute it if so."""
    now = datetime.now(UTC)

    # Check max_runs
    max_runs = sched.get("max_runs")
    if max_runs is not None and run_count >= max_runs:
        return

    # Check start_at window
    start_at = sched.get("start_at")
    if start_at:
        try:
            start_dt = datetime.fromisoformat(start_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            logger.warning(
                "Workflow %s has invalid start_at '%s'; disabling schedule", wf_id, start_at
            )
            _disable_schedule(wf_id)
            return
        if now < start_dt:
            return

    # Check end_at window
    end_at = sched.get("end_at")
    if end_at:
        try:
            end_dt = datetime.fromisoformat(end_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            logger.warning("Workflow %s has invalid end_at '%s'; disabling schedule", wf_id, end_at)
            _disable_schedule(wf_id)
            return
        if now > end_dt:
            return

    # Determine if due
    sched_type = sched.get("type", "interval")
    is_due = False

    if sched_type == "interval":
        interval_seconds = sched.get("interval_seconds", 300)
        if last_scheduled_at is None:
            is_due = True
        else:
            elapsed = (now - last_scheduled_at).total_seconds()
            is_due = elapsed >= interval_seconds

    elif sched_type == "cron":
        cron_expr = sched.get("cron", "0 * * * *")
        if last_scheduled_at is None:
            is_due = True  # First run fires immediately after deploy
        else:
            try:
                cron = croniter(cron_expr, last_scheduled_at)
                next_fire = cron.get_next(datetime)
                if next_fire.tzinfo is None:
                    next_fire = next_fire.replace(tzinfo=UTC)
                is_due = next_fire <= now
            except (ValueError, KeyError):
                logger.error("Invalid cron expression '%s' for workflow %s", cron_expr, wf_id)
                return

    if not is_due:
        return

    # Execute the workflow
    logger.info("Executing scheduled workflow %s (run #%d)", wf_id, run_count + 1)
    try:
        _execute_scheduled_workflow(wf_id)
    except BadRequest as exc:
        # check_model_available aborted (no BYOK key, no platform key for the
        # block's provider). The workflow_executions INSERT inside
        # _execute_scheduled_workflow runs AFTER the gate, so no run row exists
        # yet — write one with status='failed' so the user sees the failure
        # in the UI. THEN fall through to the UPDATE below so the next 30s
        # tick does not immediately re-classify this workflow as due
        # (tight-loop fix).
        error_msg = exc.description or str(exc)
        logger.warning(
            "scheduled_workflow_aborted wf_id=%s error=%s",
            wf_id,
            error_msg,
        )
        exec_id = str(uuid.uuid4())
        try:
            db.session.execute(
                text(f"""
                    INSERT INTO "{AI_SCHEMA}".workflow_executions
                        (id, workflow_id, status, input, error, started_at, completed_at)
                    VALUES (:id, :wid, :status, CAST(:input AS jsonb), :error, :started, :completed)
                """),
                {
                    "id": exec_id,
                    "wid": wf_id,
                    "status": "failed",
                    "input": json.dumps({"_trigger": "schedule"}),
                    "error": error_msg,
                    "started": now,
                    "completed": now,
                },
            )
            db.session.commit()
        except Exception:
            # If the failed-row INSERT itself raises (DB blip, FK violation,
            # schema drift), we must NOT skip the UPDATE below — that's the
            # load-bearing barrier against the tight loop. Roll back the
            # session so the next UPDATE doesn't fail on a poisoned txn.
            db.session.rollback()
            logger.warning(
                "scheduled_failed_row_insert_failed wf_id=%s exec_id=%s",
                wf_id,
                exec_id,
                exc_info=True,
            )

    # Update schedule tracking
    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".workflows
            SET last_scheduled_at = :now,
                schedule_run_count = schedule_run_count + 1
            WHERE id = :id
        """),
        {"id": wf_id, "now": now},
    )
    db.session.commit()


def _disable_schedule(wf_id: str) -> None:
    """Disable a workflow's schedule due to invalid config."""
    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".workflows
            SET schedule_config = jsonb_set(
                COALESCE(schedule_config, '{{}}'::jsonb),
                '{{enabled}}', 'false'
            )
            WHERE id = :id
        """),
        {"id": wf_id},
    )
    db.session.commit()


def _execute_scheduled_workflow(wf_id: str):
    """Execute a workflow using service role credentials."""
    from ..routes._workflow_helpers import (
        build_block_logs,
        build_workflow_from_db,
        get_final_output,
        load_blocks,
        load_edges,
        make_agent_run_recorder,
        make_services,
        persist_block_logs,
        serialize_outputs,
    )

    wf = build_workflow_from_db(wf_id)
    if wf is None:
        logger.error("Scheduled workflow %s not found in DB", wf_id)
        return

    blocks_data = load_blocks(wf_id)
    edges_data = load_edges(wf_id)

    # Fail-fast on any agent block whose resolved model has neither a project
    # BYOK key nor a platform env key. Mirrors routes/workflows.py:503-515 and
    # routes/webhooks.py: without this, scheduled ticks of a BYOK-only-block
    # workflow leave the workflow_executions row in 'running' forever after
    # LiteLLM's generic "Missing API Key" surfaces mid-run.
    for block in blocks_data:
        if block.get("type") != "agent":
            continue
        cfg = block.get("config") or {}
        block_model = cfg.get("model")
        if not block_model and cfg.get("agent_id"):
            agent_row = db.session.execute(
                text(f'SELECT model FROM "{AI_SCHEMA}".agents WHERE id = :id'),
                {"id": cfg["agent_id"]},
            ).fetchone()
            block_model = agent_row[0] if agent_row else None
        if block_model:
            check_model_available(block_model)

    input_variables = {"_trigger": "schedule"}

    # Create execution record
    exec_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    db.session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".workflow_executions
                (id, workflow_id, status, input, started_at)
            VALUES (:id, :wid, 'running', CAST(:input AS jsonb), :started)
        """),
        {"id": exec_id, "wid": wf_id, "input": json.dumps(input_variables), "started": now},
    )
    db.session.commit()

    try:
        recorder, agent_run_ids_by_block = make_agent_run_recorder(exec_id, log_context="scheduler")
        # Use SERVICE_ROLE_KEY for auth
        service_role_key = os.getenv("SERVICE_ROLE_KEY", "")
        services = make_services(auth_token=service_role_key, agent_run_recorder=recorder)

        async def _run_with_timeout():
            return await asyncio.wait_for(
                wf.arun_detailed(variables=input_variables, services=services),
                timeout=EXECUTION_TIMEOUT,
            )

        with billing.llm_call_scope():
            output, events = asyncio.run(_run_with_timeout())

        duration_map = {
            e.block_id: e.duration_ms for e in events if e.type in ("block_complete", "block_error")
        }

        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".workflow_executions
                SET status = 'completed',
                    output = CAST(:output AS jsonb),
                    block_outputs = CAST(:block_outputs AS jsonb),
                    completed_at = :completed
                WHERE id = :id
            """),
            {
                "id": exec_id,
                "output": json.dumps(get_final_output(output, blocks_data)),
                "block_outputs": json.dumps(serialize_outputs(output)),
                "completed": datetime.now(UTC),
            },
        )
        db.session.commit()

        block_logs = build_block_logs(
            output,
            blocks_data,
            edges_data,
            duration_map,
            agent_run_ids_by_block=agent_run_ids_by_block,
        )
        if not persist_block_logs(exec_id, block_logs):
            logger.warning("Block logs failed to persist for execution %s", exec_id)

        logger.info("Scheduled workflow %s completed (exec %s)", wf_id, exec_id)

    except asyncio.TimeoutError:
        error_msg = f"Scheduled execution timed out after {EXECUTION_TIMEOUT}s"
        logger.error("Scheduled workflow %s timed out", wf_id)
        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".workflow_executions
                SET status = 'failed', error = :error, completed_at = :completed
                WHERE id = :id
            """),
            {"id": exec_id, "error": error_msg, "completed": datetime.now(UTC)},
        )
        db.session.commit()

    except Exception as e:
        logger.error("Scheduled workflow %s failed: %s", wf_id, e, exc_info=True)
        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".workflow_executions
                SET status = 'failed', error = :error, completed_at = :completed
                WHERE id = :id
            """),
            {"id": exec_id, "error": str(e), "completed": datetime.now(UTC)},
        )
        db.session.commit()
