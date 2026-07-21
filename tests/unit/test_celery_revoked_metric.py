"""Pin: task_revoked signal increments celery_tasks_total{status="revoked"}.

Rationale (R3-S1): Celery's `Request.revoked()` short-circuits before the
worker trace pipeline, so `task_postrun` never fires for revoke-before-pickup.
The `status=~"...|revoked"` clause in WorkerThroughputZero would be defending
a scenario that produces no counter increment unless we subscribe to
`task_revoked` directly.
"""

from prometheus_client import REGISTRY

import agentic_project_service.celery  # noqa: F401 — registers signal handlers


def _read_counter(task: str, status: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "celery_tasks_total",
            labels={"task": task, "status": status},
        )
        or 0
    )


def test_task_revoked_increments_revoked_counter():
    """Fire task_revoked → counter goes up by exactly 1 with status=revoked."""
    from celery.signals import task_revoked

    before = _read_counter("test_task_revoke", "revoked")

    class FakeRequest:
        task = "test_task_revoke"

    task_revoked.send(sender=None, request=FakeRequest())

    after = _read_counter("test_task_revoke", "revoked")
    assert (
        after == before + 1
    ), f"Expected one increment per task_revoked event, got {after - before}."


def test_task_revoked_handler_is_subscribed():
    """Defense-in-depth: the handler must remain subscribed to task_revoked.

    Symmetric to test_no_task_failure_handler_subscribed in
    test_celery_failure_counted_once.py. The WorkerThroughputZero alert's
    `revoked` filter clause assumes this handler is present; removing the
    handler silently disables the operator-cancel defense.
    """
    from celery.signals import task_revoked

    receivers = [
        r
        for r in task_revoked.receivers
        if r[1]() is not None  # weakref dereference
    ]
    receiver_modules = {getattr(r[1](), "__module__", None) for r in receivers}
    assert "agentic_project_service.celery" in receiver_modules, (
        "agentic_project_service.celery is no longer subscribed to task_revoked — "
        "WorkerThroughputZero's `revoked` filter clause is now dead code."
    )


def test_task_revoked_during_execution_pops_prerun_entry():
    """Pin R4-S1: revoke-during-execution must reclaim the prerun timestamp.

    Without explicit cleanup in _on_task_revoked, the entry written by
    _on_task_prerun leaks for every revoked-while-running task. On busy
    workers that handle the most cancellations, the dict grows unbounded.
    """
    from celery.signals import task_prerun, task_revoked

    from agentic_project_service.celery import _task_start_times

    class FakeTask:
        name = "test_task_revoke_running"

    class FakeRequest:
        id = "running-id-r4s1"
        task = "test_task_revoke_running"

    # Establish: task starts (prerun fires -> entry written).
    task_prerun.send(sender=None, task_id="running-id-r4s1", task=FakeTask())
    assert (
        "running-id-r4s1" in _task_start_times
    ), "test prerequisite: prerun did not record the start time"

    # Operator revokes the running task (terminate=True path).
    task_revoked.send(sender=None, request=FakeRequest(), terminated=True, signum=15, expired=False)

    assert "running-id-r4s1" not in _task_start_times, (
        "_on_task_revoked did NOT pop the prerun entry — _task_start_times "
        "is leaking one dict entry per revoke-during-execution event."
    )


def test_task_revoked_before_pickup_does_not_error():
    """Pin R4-S1 counterpart: revoke-before-pickup has no prerun entry to pop.

    For revoke-before-pickup, _on_task_prerun never fired and there's
    nothing in _task_start_times for this task_id. dict.pop's default
    must absorb the miss without raising.
    """
    from celery.signals import task_revoked

    from agentic_project_service.celery import _task_start_times

    class FakeRequest:
        id = "never-picked-up-r4s1"
        task = "test_task_revoke_unstarted"

    assert "never-picked-up-r4s1" not in _task_start_times

    # Should not raise.
    task_revoked.send(
        sender=None, request=FakeRequest(), terminated=False, signum=None, expired=False
    )

    assert "never-picked-up-r4s1" not in _task_start_times
