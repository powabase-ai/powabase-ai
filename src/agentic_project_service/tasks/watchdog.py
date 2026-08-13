"""Self-rescheduling watchdog for orphaned ai.indexed_sources rows.

Recovers one failure mode:
- Indexing-orphans: ``index_status='indexing'``, ``celery_task_id`` set but
  not present in any alive Celery task set (active/reserved/queued). This
  is the worker-died-mid-task scenario (OOM, container restart, etc.).

**Pending-orphans are out of scope** for this watchdog: the obvious
detection — "row in 'pending' but its task isn't alive" — can't be done
reliably because ``indexed_sources.celery_task_id`` is written only when
a worker starts executing the task (see ``indexing.py`` --
``update_indexed_source_status(..., "indexing", celery_task_id=task_id)``).
For a row that's *waiting in the Celery queue*, ``celery_task_id IS NULL``
regardless of whether the task is alive or lost; the current SQL therefore
cannot tell the two cases apart and the previous "pending-orphan" branch
re-dispatched every queued row >2 min old, producing duplicate tasks that
raced on the same indexed_source. Pending-rows lost to a broker outage
are recovered manually via ``POST /api/knowledge-bases/<kb_id>/reindex``
for now; reintroducing automatic pending-orphan recovery requires either
writing ``celery_task_id`` at dispatch time or matching by
``indexed_source_id`` instead of task ID.
"""

import json
import logging
import os
import time
from typing import Iterable

import redis
from sqlalchemy import text

from ..celery import celery_app
from ..db import AI_SCHEMA, commit_scoped_write, db
from ..services.ai_provider_keys_resolver import get_all_user_provider_keys
from ..services import billing_port as billing

logger = logging.getLogger(__name__)

TICK_INTERVAL = 300  # 5 minutes
LOCK_KEY = "indexed_sources_watchdog_lock"
LOCK_TTL = 60  # seconds


def _get_redis():
    """Return (redis client, project-scoped lock key)."""
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
    project_ref = os.getenv("PROJECT_REF", "default")
    r = redis.from_url(broker_url)
    return r, f"{project_ref}:{LOCK_KEY}"


@celery_app.task(bind=True, ignore_result=True, max_retries=0)
@billing.no_billing_context
def indexed_sources_watchdog_tick(self):
    """Detect + recover orphaned indexed_sources rows, then re-enqueue self."""
    r, lock_key = _get_redis()

    if not r.set(lock_key, "1", nx=True, ex=LOCK_TTL):
        logger.debug("Watchdog lock held by another worker, skipping")
    else:
        try:
            _run_one_tick()
        except Exception:
            logger.error("Watchdog tick failed", exc_info=True)
        finally:
            try:
                r.delete(lock_key)
            except Exception:
                logger.warning("Failed to delete watchdog lock", exc_info=True)

    # Re-enqueue with retry (matches scheduler_tick pattern)
    for attempt in range(3):
        try:
            indexed_sources_watchdog_tick.apply_async(countdown=TICK_INTERVAL)
            break
        except Exception:
            if attempt == 2:
                logger.error(
                    "Failed to re-enqueue watchdog tick after 3 attempts",
                    exc_info=True,
                )
            else:
                time.sleep(1)


INDEX_SOURCE_TASK_NAME = "agentic_project_service.tasks.indexing.index_source"


def _extract_ids_from_inspect_result(result: dict | None, name_filter: str) -> set[str]:
    """From {worker_name: [task_dict, ...]} extract task IDs whose name matches."""
    ids: set[str] = set()
    for tasks in (result or {}).values():
        for t in tasks or []:
            if t.get("name") == name_filter and t.get("id"):
                ids.add(t["id"])
    return ids


def _extract_ids_from_queue(messages: Iterable[bytes], name_filter: str) -> set[str]:
    """Parse Celery broker LRANGE results and extract task IDs whose name matches."""
    ids: set[str] = set()
    for raw in messages:
        try:
            env = json.loads(raw)
        except (ValueError, TypeError):
            continue
        headers = env.get("headers")
        if not isinstance(headers, dict):
            continue
        task_id = headers.get("id")
        task_name = headers.get("task")
        if task_id and task_name == name_filter:
            ids.add(task_id)
    return ids


def _collect_alive_task_ids(
    r, inspect, queue_key: str
) -> tuple[set[str], list[bytes], dict | None, dict | None]:
    """Union of active + reserved + queued index_source task IDs.

    Returns (alive_ids, raw_queue_messages, active_raw, reserved_raw).
    The raw inspect responses are returned so the caller can use them in the
    workers-unreachable safety gate without re-issuing the inspect RPCs.
    """
    queue_msgs = r.lrange(queue_key, 0, -1) or []
    queued = _extract_ids_from_queue(queue_msgs, INDEX_SOURCE_TASK_NAME)
    active_raw = inspect.active()
    reserved_raw = inspect.reserved()
    active = _extract_ids_from_inspect_result(active_raw, INDEX_SOURCE_TASK_NAME)
    reserved = _extract_ids_from_inspect_result(reserved_raw, INDEX_SOURCE_TASK_NAME)
    return queued | active | reserved, queue_msgs, active_raw, reserved_raw


def _run_one_tick() -> None:
    """One pass: collect alive task IDs, gate on safety, then recover orphans."""
    project_ref = os.getenv("PROJECT_REF", "default")
    queue_key = f"{project_ref}:celery"

    r, _ = _get_redis()
    inspect = celery_app.control.inspect(timeout=5)

    alive_ids, queue_msgs, active_raw, reserved_raw = _collect_alive_task_ids(r, inspect, queue_key)

    # "Workers unreachable" gate: no live tasks anywhere AND no responding workers.
    workers_responded = bool(active_raw or reserved_raw)
    if not alive_ids and not queue_msgs and not workers_responded:
        logger.warning(
            "Watchdog: no live tasks visible and no workers responded; "
            "skipping recovery to avoid amplifying outage"
        )
        return

    _find_and_recover_orphans(alive_ids)


ORPHAN_QUERY = f"""
    SELECT id, source_id, knowledge_base_id, celery_task_id, attempts
    FROM "{AI_SCHEMA}".indexed_sources
    WHERE index_status = 'indexing'
      AND last_dispatched_at < NOW() - INTERVAL '2 minutes'
      AND celery_task_id IS NOT NULL
      AND celery_task_id <> ALL(:alive_ids)
"""
# Scope: indexing-orphans only. A row in 'indexing' has its celery_task_id
# populated by the worker (indexing.py); if that task ID isn't visible in any
# alive Celery set, the worker died and we re-dispatch.
#
# Pending-rows are deliberately excluded: see module docstring for the
# rationale. The 2-minute last_dispatched_at guard remains as a dispatch-race
# buffer, even though for the 'indexing' case the celery_task_id-vs-alive_ids
# check alone is technically sufficient; the guard costs us nothing and
# preserves symmetry with /reindex's dispatch path.


def _find_and_recover_orphans(alive_ids: set[str]) -> int:
    """Find orphans; fail those at the attempts bound, re-dispatch the rest.

    Returns the number re-dispatched. Rows at/over ``MAX_ATTEMPTS`` are written
    terminal instead and counted separately -- this is what stops a source whose
    worker dies every time from being re-dispatched forever.
    """
    # Import locally to avoid circular import (indexing imports from celery, which
    # imports tasks at startup).
    from .indexing import MAX_ATTEMPTS, index_source

    rows = db.session.execute(
        text(ORPHAN_QUERY),
        {"alive_ids": list(alive_ids)},
    ).fetchall()

    if not rows:
        logger.debug("Watchdog: no orphans detected")
        return 0

    exhausted = [row for row in rows if row.attempts >= MAX_ATTEMPTS]
    recoverable = [row for row in rows if row.attempts < MAX_ATTEMPTS]

    # ORDER IS LOAD-BEARING: this terminal write MUST be issued and committed
    # BEFORE the bulk reset below, while these rows are still 'indexing'. The
    # reset flips its targets to 'pending'; run afterwards, this UPDATE's
    # predicate would match zero rows and the row would sit at 'pending' with
    # attempts already spent -- invisible to ORPHAN_QUERY (which only selects
    # 'indexing') and never re-dispatched, i.e. stuck non-terminal forever.
    #
    # Fenced on index_status, NOT on celery_task_id. Everywhere else a terminal
    # write is made by the task that claimed the row and is fenced on owning it;
    # here the owner is a dead task and the reconciler holds no task id of its
    # own, so an ownership predicate could never match. The status predicate is
    # the equivalent guard: it yields to whatever moved the row on.
    #
    # The message names no cause. At this layer a worker that hit a resource
    # ceiling, one reclaimed by the scheduler and one rolled by a redeploy are
    # the same observation -- the task stopped existing -- so naming any of them
    # would be a guess presented to the user as a diagnosis.
    for row in exhausted:
        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".indexed_sources
                SET index_status = 'failed',
                    error_message = :msg
                WHERE id = :id
                  AND index_status = 'indexing'
            """),
            {
                "id": row.id,
                "msg": (
                    f"Indexing did not complete after {row.attempts} attempts "
                    "(worker terminated before finishing)."
                ),
            },
        )
    if exhausted:
        db.session.commit()
        logger.warning(
            "Watchdog: %d orphaned indexed_sources rows reached the attempts "
            "bound of %d; wrote terminal 'failed' (no re-dispatch)",
            len(exhausted),
            MAX_ATTEMPTS,
        )

    if not recoverable:
        return 0

    orphan_ids = [row.id for row in recoverable]
    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".indexed_sources
            SET index_status = 'pending',
                error_message = NULL,
                last_dispatched_at = NOW()
            WHERE id = ANY(:ids)
              AND index_status = 'indexing'
        """),
        {"ids": orphan_ids},
    )
    db.session.commit()

    provider_keys = get_all_user_provider_keys()
    # Thread the billing key inputs deterministically per row so the recovered
    # task bills exactly once at the billing service. Identity (org/project) is
    # added by the billing adapter, which also no-ops the charge when billing is
    # unconfigured — so we thread unconditionally (no billing-context read here).
    #
    # Key shape matches the /reindex + batch paths in routes/knowledge_bases.py
    # (literal "indexing" action namespace + indexed_source_id), so a subsequent
    # reindex of the same row and a watchdog recovery converge on the same key
    # (idempotent at UNIQUE(org_id, idempotency_key)). The dead task never
    # reached its charge (which only fires AFTER index_status is flipped to
    # 'indexed'), so dedupe here is the correct outcome.
    recovered = 0
    for row in recoverable:
        try:
            indexed_source_id_str = str(row.id)
            # str() coercion matches the four other dispatch sites
            # (/add_source_to_kb, /reindex selective, /reindex failed_only,
            # reindex_knowledge_base). kombu's JSON serializer preserves
            # uuid.UUID type across the wire via a typed marker, so without
            # this coercion the worker receives UUID and the chunker raises
            # a Pydantic ValidationError on TextChunk(source_id=UUID(...)).
            index_source.delay(
                str(row.knowledge_base_id),
                str(row.source_id),
                indexed_source_id=indexed_source_id_str,
                provider_keys=provider_keys,
                idempotency_action="indexing",
                idempotency_parts=[indexed_source_id_str],
            )
            recovered += 1
        except Exception:
            # The reset above is already committed, so a publish that never
            # reached the broker leaves a 'pending' row with nobody coming for
            # it -- and ORPHAN_QUERY only ever selects 'indexing', so nothing
            # would find it again. Logging alone strands the row, and strands it
            # in the one component whose whole job is to un-strand rows: a
            # broker outage would do this to every orphan in a single tick.
            #
            # Make it terminal instead, the way the storage-error retry does.
            # Fenced on index_status, not ownership: the reset left this row
            # 'pending' still carrying the DEAD task's id, and the reconciler
            # holds no id of its own, so status is the only predicate that means
            # anything here. It yields correctly too -- a task that has since
            # claimed the row flipped it to 'indexing', and this matches nothing.
            logger.error(
                "Watchdog: failed to .delay() recovery for indexed_source_id=%s",
                row.id,
                exc_info=True,
            )
            # Through commit_scoped_write, NOT a bare execute+commit. This runs
            # inside an except handler inside the per-row loop, and the case
            # that reaches it is a broker outage -- i.e. every row fails, one
            # after another. A write that raises here would escape the handler,
            # escape the loop, and strand every orphan after the first, in the
            # component whose job is to un-strand them.
            #
            # Fenced on index_status AND attempts, and the second half is not
            # decoration. 'pending' does not identify THIS reconciler's row:
            # ten writers produce it, and the three /reindex routes in
            # particular set it without clearing celery_task_id. The window is
            # real and widest exactly when this path runs -- the bulk reset
            # commits for every row up front, then each .delay() is attempted in
            # turn, and a sick broker makes those slow. A user who hits Reindex
            # inside it had their freshly-reset row flipped to 'failed' by a
            # stale write from this loop, and the index_source they just queued
            # cannot claim a non-'pending' row, so it exited skipped and their
            # reindex silently did nothing.
            #
            # attempts separates the two: the reset above leaves it untouched,
            # so this row still carries the value ORPHAN_QUERY read (>= 1,
            # because the claim increments on the way into 'indexing'), while
            # every reindex route resets it to 0. A re-requested row therefore
            # no longer matches, and the rowcount says so.
            failed_rows = commit_scoped_write(
                f"""
                    UPDATE "{AI_SCHEMA}".indexed_sources
                    SET index_status = 'failed',
                        error_message = :msg
                    WHERE id = :id
                      AND index_status = 'pending'
                      AND attempts = :attempts
                """,
                {
                    "id": row.id,
                    "attempts": row.attempts,
                    # Names only what is known here: the retry could not be
                    # queued. Why the broker refused is not observable at this
                    # layer, so it is not guessed at.
                    "msg": "Indexing could not be re-queued after the worker stopped.",
                },
                what="watchdog unqueueable-recovery terminal write",
            )
            if not failed_rows:
                logger.warning(
                    "Watchdog: could not mark indexed_source_id=%s failed after a "
                    "dispatch failure; it was moved on by someone else, or the "
                    "write did not land",
                    row.id,
                )

    logger.info(
        "Watchdog: recovered %d/%d orphaned indexed_sources rows",
        recovered,
        len(recoverable),
    )
    return recovered


def seed_watchdog() -> None:
    """Bootstrap the self-rescheduling chain on worker startup.

    Called from a worker_ready signal handler in celery.py. Each call enqueues
    one tick which begins its own self-rescheduling chain — duplicate calls
    create N parallel chains. The Redis lock prevents duplicate WORK each tick
    (only one chain does the DB scan), but does not collapse the chain count.
    In practice this is fine: at 5-minute cadence the per-tick cost is trivial,
    and chains reset on the next worker restart.
    """
    indexed_sources_watchdog_tick.delay()
