"""Periodic cleanup tasks."""

import logging
import os
import threading
import time
import uuid

import redis
from sqlalchemy import text

from ..celery import celery_app
from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
from ..services.broker_redis import broker_client
from ..services.run_registry import HEARTBEAT_SECONDS, OWNER_TTL_SECONDS, RunRegistry
from ..services.settings_registry import get_setting_strict

logger = logging.getLogger(__name__)

STALE_TICK_INTERVAL = 300  # seconds
STALE_LOCK_KEY = "stale_runs_lock"
STALE_LOCK_TTL = 60  # seconds
STALE_CHAIN_KEY = "stale_runs_chain"
STALE_CHAIN_TTL = 3 * STALE_TICK_INTERVAL  # outlives a missed tick or two
# A new chain's first tick waits this long. A worker starts with a deploy or a
# restart, when API processes may still be rewriting liveness records they lost
# (that takes up to OWNER_TTL_SECONDS + HEARTBEAT_SECONDS), and when, in a
# rolling deploy, old API pods that write no records at all are still being
# replaced. With the two-sweep rule a run is failed no sooner than
# FIRST_TICK_COUNTDOWN + STALE_TICK_INTERVAL (10 minutes) after a worker starts.
FIRST_TICK_COUNTDOWN = STALE_TICK_INTERVAL


INTERRUPTED_ERROR = "Run interrupted (pod restart or timeout)"
# At most this many runs of each kind are failed per sweep; the next sweep
# takes the rest.
STALE_SWEEP_LIMIT = 1000

_TABLES = ("agent_runs", "orchestration_runs")

# A run whose liveness record is missing is failed only when the record is
# still missing at a later sweep, at least this long after the first miss:
# longer than a live process takes to write back a record that lapsed (its
# TTL, plus one rewrite). Two sweeps moments apart (the beat entry and the
# chain) count as one miss.
ABSENCE_CONFIRM_SECONDS = OWNER_TTL_SECONDS + HEARTBEAT_SECONDS
# The mark of a first miss lasts a few ticks; a mark that lapsed (sweeps
# skipped while Redis was down) only means the count starts again.
SUSPECT_TTL = 3 * STALE_TICK_INTERVAL


def _live_run_ids(client, run_ids: list[str]) -> set[str]:
    """The run ids that have a liveness record: a process is running them and
    rewrites the record every few seconds (services/run_registry.py)."""
    pipe = client.pipeline(transaction=False)
    for run_id in run_ids:
        pipe.exists(RunRegistry.alive_key(run_id))
    return {run_id for run_id, live in zip(run_ids, pipe.execute(), strict=True) if live}


def _suspect_key(table: str, row_id: str) -> str:
    return f"{os.getenv('PROJECT_REF', 'default')}:stale_run_suspect:{table}:{row_id}"


def _redis_now(client) -> int:
    """The Redis clock, so that sweeps on different workers agree."""
    seconds, _ = client.time()
    return int(seconds)


def _confirmed_dead(client, candidates: dict) -> dict[str, list[str]]:
    """The candidate row ids, per table, whose liveness record was missing at
    an earlier sweep and is still missing now (ABSENCE_CONFIRM_SECONDS later).

    A first miss is only marked. A run seen alive loses its mark, so a record
    that lapsed and came back starts the count again. Raises when Redis
    cannot read or write the marks: then nothing is confirmed.
    """
    rows = [(table, str(row[0]), row[1]) for table in _TABLES for row in candidates[table]]
    run_ids = [run_id for _, _, run_id in rows if run_id]
    live = _live_run_ids(client, run_ids) if run_ids else set()
    now = _redis_now(client)
    absent = [(table, row_id) for table, row_id, run_id in rows if run_id not in live]
    pipe = client.pipeline(transaction=False)
    for table, row_id in absent:
        pipe.get(_suspect_key(table, row_id))
    first_misses = pipe.execute() if absent else []

    dead: dict[str, list[str]] = {table: [] for table in _TABLES}
    pipe = client.pipeline(transaction=False)
    for (table, row_id), first_miss in zip(absent, first_misses, strict=True):
        key = _suspect_key(table, row_id)
        if first_miss is None:
            pipe.set(key, now, ex=SUSPECT_TTL)
        elif now - int(first_miss) >= ABSENCE_CONFIRM_SECONDS:
            dead[table].append(row_id)
            pipe.delete(key)
    for table, row_id, run_id in rows:
        if run_id in live:
            pipe.delete(_suspect_key(table, row_id))
    pipe.execute()
    return dead


def run_stale_cleanup() -> dict:
    """Fail runs still 'running' after RUN_STALE_AFTER_SECONDS whose process is gone.

    This handles a pod restart mid-run, which loses the in-process run. A run
    is only failed when no process has written a liveness record for it at
    two sweeps in a row, ABSENCE_CONFIRM_SECONDS or more apart: a run that is
    merely long (a slow model, a long approval wait) keeps streaming, and so
    does one whose process briefly could not write to Redis. When Redis
    cannot say which runs are alive, or cannot record a miss, or the
    settings cannot be read, nothing is failed this sweep. Covers agent runs
    and orchestration runs.
    """
    counts = {"agent_runs": 0, "orchestration_runs": 0}
    try:
        # Not get_setting: its 1800 s default is no safe guess when an
        # operator's override cannot be read.
        stale_after = int(get_setting_strict("RUN_STALE_AFTER_SECONDS"))
    except Exception:
        logger.warning(
            "Stale-run cleanup: RUN_STALE_AFTER_SECONDS unreadable; skipping this sweep",
            exc_info=True,
        )
        return counts
    params = {"stale_after": stale_after, "limit": STALE_SWEEP_LIMIT}
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
    if not any(candidates.values()):
        return counts

    try:
        confirmed = _confirmed_dead(broker_client(), candidates)
    except Exception:
        logger.warning(
            "Stale-run cleanup: cannot tell which runs are alive (Redis unavailable); "
            "skipping this sweep",
            exc_info=True,
        )
        return counts

    for table in _TABLES:
        dead = confirmed[table]
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


def _keys() -> tuple[str, str]:
    """(work lock, chain generation) keys, project-scoped."""
    project_ref = os.getenv("PROJECT_REF", "default")
    return f"{project_ref}:{STALE_LOCK_KEY}", f"{project_ref}:{STALE_CHAIN_KEY}"


# Deletes the lock only while it is still ours: a cleanup that outlived its
# STALE_LOCK_TTL must not delete the lock the next holder took.
_RELEASE_LOCK_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


# Puts back the generation a failed seed replaced, unless another seed has
# replaced it since. ARGV: the failed generation, the previous one ('' = none),
# the TTL.
_RESTORE_GENERATION_LUA = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return 0
end
if ARGV[2] == '' then
  return redis.call('DEL', KEYS[1])
end
redis.call('SET', KEYS[1], ARGV[2], 'EX', tonumber(ARGV[3]))
return 1
"""


def seed_stale_runs_chain() -> None:
    """Start a stale-run chain that replaces every earlier one.

    Called on worker_ready, and by a run start that finds no live chain
    (ensure_stale_runs_chain). Queued countdown ticks survive worker restarts,
    so a seed that only enqueued would add a chain per worker start. Instead
    each seed records a new generation, and a tick whose generation is no
    longer the recorded one stops instead of re-enqueueing: however many
    workers start, one chain is left.

    The generation is recorded before the first tick is enqueued (a tick that
    ran first would find the old generation and stop). If the enqueue fails,
    the previous generation is put back, so its chain keeps running. The
    first tick runs FIRST_TICK_COUNTDOWN later, not at once.
    """
    generation = uuid.uuid4().hex
    _, chain_key = _keys()
    client, recorded, previous = None, False, None
    try:
        client = broker_client()
        previous = client.set(chain_key, generation, ex=STALE_CHAIN_TTL, get=True)
        recorded = True
    except Exception:
        logger.warning(
            "Stale-run chain: could not record its generation; its first tick adopts it",
            exc_info=True,
        )
    try:
        _enqueue_first_tick(generation)
    except Exception:
        logger.exception("Stale-run chain: could not enqueue its first tick")
        if recorded:
            try:
                client.eval(
                    _RESTORE_GENERATION_LUA,
                    1,
                    chain_key,
                    generation,
                    _text(previous) if previous is not None else "",
                    STALE_CHAIN_TTL,
                )
            except Exception:
                logger.warning(
                    "Stale-run chain: could not restore the previous generation", exc_info=True
                )


def _enqueue_first_tick(generation: str) -> None:
    stale_runs_tick.apply_async(args=[generation], countdown=FIRST_TICK_COUNTDOWN)


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


CHAIN_CHECK_INTERVAL = 60  # seconds, per process
_last_chain_check = float("-inf")
_chain_check_lock = threading.Lock()


def _claim_chain_check() -> bool:
    global _last_chain_check
    now = time.monotonic()
    with _chain_check_lock:
        if now - _last_chain_check < CHAIN_CHECK_INTERVAL:
            return False
        _last_chain_check = now
        return True


def _check_chain() -> None:
    try:
        _, chain_key = _keys()
        if broker_client().exists(chain_key):
            return
    except redis.RedisError:
        logger.debug("Stale-run chain check: Redis unavailable", exc_info=True)
        return
    except Exception:
        # Not debug: on a project without beat this check is the only thing
        # that restarts a dead chain.
        logger.warning("Stale-run chain check failed", exc_info=True)
        return
    logger.warning("Stale-run chain: no live generation found; starting a new chain")
    seed_stale_runs_chain()


def ensure_stale_runs_chain() -> None:
    """Re-seed the chain when no live generation is recorded (a chain whose
    re-enqueue kept failing gives its generation up). At most once per
    CHAIN_CHECK_INTERVAL per process; never raises."""
    if _claim_chain_check():
        _check_chain()


def ensure_stale_runs_chain_soon() -> None:
    """ensure_stale_runs_chain on a daemon thread, for the request path."""
    if _claim_chain_check():
        threading.Thread(target=_check_chain, name="stale-runs-chain-check", daemon=True).start()


def _chain_is_current(client, chain_key: str, generation: str | None) -> bool:
    current = client.get(chain_key)
    if current is None and generation is not None:
        # The record was lost (a Redis restart or flush): the first chain to
        # tick takes it over, and the others stop when they see it.
        if client.set(chain_key, generation, nx=True, ex=STALE_CHAIN_TTL):
            return True
        current = client.get(chain_key)
    if current is None or generation is None:
        return False
    if _text(current) != generation:
        return False
    client.expire(chain_key, STALE_CHAIN_TTL)
    return True


@celery_app.task(name="stale_runs_tick", ignore_result=True, max_retries=0)
@billing.no_billing_context
def stale_runs_tick(generation: str | None = None):
    """Run the stale-run cleanup, then re-enqueue self.

    The project stack runs no Celery beat, so the beat entry for
    cleanup_stale_runs never fires there; this self-rescheduling chain (seeded
    on worker_ready by seed_stale_runs_chain) is what runs it. A tick of a
    superseded generation stops. The work lock keeps the cleanup to one at a
    time. When Redis is down the chain keeps going but does no work.
    """
    lock_key, chain_key = _keys()
    token = uuid.uuid4().hex
    client, got_lock = None, False
    try:
        client = broker_client()
        if not _chain_is_current(client, chain_key, generation):
            logger.info("Stale-run tick: a newer chain has taken over; this one stops")
            return
        got_lock = client.set(lock_key, token, nx=True, ex=STALE_LOCK_TTL)
    except Exception:
        logger.warning("Stale-run tick: Redis unavailable, skipping this tick", exc_info=True)
    if got_lock:
        try:
            run_stale_cleanup()
        except Exception:
            logger.exception("Stale-run cleanup failed")
        finally:
            try:
                client.eval(_RELEASE_LOCK_LUA, 1, lock_key, token)
            except Exception:
                logger.warning("Failed to release the stale-run lock", exc_info=True)

    for attempt in range(3):
        try:
            stale_runs_tick.apply_async(args=[generation], countdown=STALE_TICK_INTERVAL)
            return
        except Exception:
            if attempt == 2:
                logger.exception("Failed to re-enqueue the stale-run tick after 3 attempts")
            else:
                time.sleep(1)
    # This chain ends here. Give up its generation, so the next run start
    # (ensure_stale_runs_chain) finds none and seeds a new chain.
    if client is not None and generation is not None:
        try:
            client.eval(_RESTORE_GENERATION_LUA, 1, chain_key, generation, "", STALE_CHAIN_TTL)
        except Exception:
            logger.warning("Stale-run tick: could not give up its generation", exc_info=True)
