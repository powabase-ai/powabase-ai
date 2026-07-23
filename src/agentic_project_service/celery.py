"""Celery application configuration for the project service."""

import os
from celery import Celery
from celery.signals import worker_ready

# Flask app instance for task context (lazy loaded)
_flask_app = None


def get_flask_app():
    """Get or create the Flask app for task context."""
    global _flask_app
    if _flask_app is None:
        from .main import create_app

        _flask_app = create_app()
    return _flask_app


def _create_celery():
    """Create and configure the Celery application (called once)."""
    celery = Celery(
        "agentic_project_service",
        broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
        backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
        include=[
            "agentic_project_service.tasks.extraction",
            "agentic_project_service.tasks.url_extraction",
            "agentic_project_service.tasks.indexing",
            "agentic_project_service.tasks.enrichment",
            "agentic_project_service.tasks.scheduler",
            "agentic_project_service.tasks.cleanup",
            "agentic_project_service.tasks.watchdog",
        ],
    )

    # Per-project isolation on shared Redis (K8s uses one shared Redis)
    project_ref = os.getenv("PROJECT_REF", "default")
    task_time_limit = int(os.getenv("CELERY_TASK_TIME_LIMIT") or 21600)  # default 6h

    celery.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=task_time_limit,
        task_soft_time_limit=max(task_time_limit - 300, task_time_limit // 2),
        worker_prefetch_multiplier=1,
        worker_pool="threads",
        worker_concurrency=int(os.getenv("CELERY_WORKER_CONCURRENCY", 4)),
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        beat_schedule={
            "cleanup-stale-runs": {
                "task": "cleanup_stale_runs",
                "schedule": 300.0,  # Every 5 minutes
            },
            "scheduler-heartbeat": {
                "task": "agentic_project_service.tasks.scheduler.scheduler_tick",
                "schedule": 300.0,  # Every 5 minutes — safety net if self-rescheduling chain breaks
            },
        },
        broker_transport_options={
            "global_keyprefix": f"{project_ref}:",
            "visibility_timeout": task_time_limit,
        },
        result_backend_transport_options={
            "global_keyprefix": f"{project_ref}:",
        },
    )

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            flask_app = get_flask_app()
            with flask_app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask

    return celery


def init_celery(app):
    """Bind Flask app context to the existing celery_app. Called from create_app()."""

    class ContextTask(celery_app.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app.Task = ContextTask


# Single Celery instance — never recreated
celery_app = _create_celery()


@worker_ready.connect
def seed_scheduler(**kwargs):
    """Bootstrap the self-rescheduling scheduler tick on worker startup."""
    from .tasks.scheduler import scheduler_tick

    scheduler_tick.delay()


@worker_ready.connect
def seed_indexed_sources_watchdog(**kwargs):
    """Bootstrap the self-rescheduling indexed_sources watchdog on worker startup."""
    from .tasks.watchdog import seed_watchdog

    seed_watchdog()


# ---------------------------------------------------------------------------
# Prometheus instrumentation for Celery tasks
#
# Registers signal handlers that maintain two metrics:
#   celery_tasks_total{task, status}     — Counter  # Status values: received | success | failure | retry.
#   celery_task_duration_seconds{task}   — Histogram
#
# Exposed by the worker via a tiny HTTP server on CELERY_METRICS_PORT (default
# 9100). Consumed by Prometheus and the observability dashboards.
#
# Pool model: this service is expected to run Celery with ``--pool=threads``.
# All worker threads share one Python interpreter and one
# Counter/Histogram instance, so the in-process registry is correct — no
# multiproc machinery needed. If a future deployment switches to prefork,
# this module needs prometheus_client.multiprocess + PROMETHEUS_MULTIPROC_DIR
# (otherwise each fork holds isolated counters and only one is scraped).
#
# If the prometheus_client library is not installed the handlers no-op, so
# adding observability deps is optional per-deployment.
# ---------------------------------------------------------------------------

try:
    import errno
    import time

    from celery.signals import (
        task_postrun,
        task_prerun,
        task_received,
        task_retry,
        task_revoked,
    )
    from prometheus_client import Counter, Histogram, start_http_server

    _celery_task_count = Counter(
        "celery_tasks_total",
        "Total Celery tasks processed, by status",
        ["task", "status"],
    )
    _celery_task_duration = Histogram(
        "celery_task_duration_seconds",
        "Celery task runtime in seconds",
        ["task"],
        buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
    )
    _task_start_times: dict[str, float] = {}

    @task_prerun.connect
    def _on_task_prerun(task_id=None, task=None, **_kw):
        if task_id:
            _task_start_times[task_id] = time.monotonic()

    @task_postrun.connect
    def _on_task_postrun(task_id=None, task=None, state=None, **_kw):
        # postrun fires for tasks that ran through the worker trace pipeline
        # with `state` set to the terminal value: SUCCESS, FAILURE, or
        # REJECTED (the rare case where a task body `raise Reject` explicitly
        # declines). We previously also subscribed to task_failure, which
        # double-counted failures (postrun saw state=FAILURE AND task_failure
        # fired), inflating {status="failure"} and masking partial-wedge in
        # WorkerThroughputZero. postrun alone is sufficient.
        #
        # REVOKED state does NOT reach postrun for the common revoke-before-pickup
        # path — `Request.revoked()` short-circuits before the trace pipeline.
        # `_on_task_revoked` below subscribes to `task_revoked` directly so the
        # operator-cancel scenario produces a `status="revoked"` increment.
        #
        # Worker-loss scenarios (SIGKILL on task_reject_on_worker_lost=True,
        # threads-pool hard kill from task_time_limit) intentionally SKIP
        # postrun. Those are exactly the cases WorkerThroughputZero exists to
        # detect: `received` increments but no terminal-state counter ever
        # increments → AND clause is true.
        started = _task_start_times.pop(task_id, None) if task_id else None
        name = getattr(task, "name", "unknown")
        _celery_task_count.labels(task=name, status=(state or "unknown").lower()).inc()
        if started is not None:
            _celery_task_duration.labels(task=name).observe(time.monotonic() - started)

    @task_revoked.connect
    def _on_task_revoked(sender=None, request=None, **_kw):
        # task_revoked fires from `worker/request.py` BEFORE the trace pipeline,
        # which is why `_on_task_postrun` doesn't observe state=REVOKED for the
        # common operator-cancel path. Without this dedicated handler the
        # `status=~"…|revoked"` clause in WorkerThroughputZero would be defending
        # a scenario that never produces a counter increment.
        name = (
            (getattr(request, "task", None) if request is not None else None)
            or getattr(sender, "name", None)
            or "unknown"
        )
        _celery_task_count.labels(task=name, status="revoked").inc()
        # Reclaim any prerun timestamp this task left behind. task_revoked
        # fires for revoke-during-execution AFTER prerun but WITHOUT a
        # matching postrun, so without explicit cleanup the dict entry
        # leaks indefinitely. For revoke-before-pickup, prerun never fired
        # and the pop default is None. Observing the duration for tasks
        # that did run keeps the histogram complete; tasks revoked before
        # pickup contribute neither a duration nor a leaked entry.
        task_id = getattr(request, "id", None) if request is not None else None
        if task_id:
            started = _task_start_times.pop(task_id, None)
            if started is not None:
                _celery_task_duration.labels(task=name).observe(time.monotonic() - started)

    @task_retry.connect
    def _on_task_retry(sender=None, **_kw):
        name = getattr(sender, "name", "unknown")
        _celery_task_count.labels(task=name, status="retry").inc()

    @task_received.connect
    def _on_task_received(sender=None, request=None, **_kw):
        """Increment celery_tasks_total{status="received"} when a worker receives a task.

        Combined with status="success" / "failure" counters, this lets the
        WorkerThroughputZero alert detect the wedge pattern (PR #135): tasks
        arriving but not completing. The counter-math approach is immune to
        worker-process-hang because counters are monotonic in-memory increments
        served by the metrics endpoint thread.
        """
        name = getattr(request, "task", None) or "unknown"
        _celery_task_count.labels(task=name, status="received").inc()

    @worker_ready.connect
    def _start_metrics_server(**_kw):
        """Expose Celery metrics on a local HTTP port for Prometheus scraping.

        With ``--pool=threads`` only the main process runs, so binding the
        port once in worker_ready is sufficient. If a future deployment
        switches to ``--pool=prefork``, this needs to migrate to
        PROMETHEUS_MULTIPROC_DIR + MultiProcessCollector — see the file
        docstring for the playbook.
        """
        port = int(os.getenv("CELERY_METRICS_PORT") or 9100)
        try:
            start_http_server(port)
        except OSError as e:
            # Port already bound (another worker on the same pod in dev);
            # that's fine — one worker's server serves the pod. Any other
            # bind failure (permission denied, no free fds, etc.) must NOT
            # be swallowed — silent failure here means no metrics endpoint,
            # which means WorkerThroughputZero can't ever fire.
            if e.errno != errno.EADDRINUSE:
                raise

except ImportError:  # pragma: no cover — observability deps are optional
    pass
