"""Pin: a task failure increments celery_tasks_total{status="failure"} ONCE, not twice.

Regression test for S1: previously both task_postrun (with state=FAILURE) AND
task_failure were subscribed; failures double-counted, inflating the
(success+failure) denominator and masking partial-wedge in WorkerThroughputZero.
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


def test_failure_increments_counter_exactly_once():
    """One task_postrun(state=FAILURE) → counter goes up by exactly 1.

    This pins the per-event accounting of the postrun handler: a single
    terminal-state event must produce a single counter increment. The
    docstring on test_no_task_failure_handler_subscribed covers the
    regression-detection contract for the dropped _on_task_failure handler —
    this test alone can't detect that regression because it never sends
    task_failure (the dropped handler subscribed to task_failure, not
    task_postrun).
    """
    from celery.signals import task_postrun

    before = _read_counter("test_task_fail", "failure")

    class FakeTask:
        name = "test_task_fail"

    task_postrun.send(sender=None, task_id="abc", task=FakeTask(), state="FAILURE")

    after = _read_counter("test_task_fail", "failure")
    assert after == before + 1, f"Expected one increment per postrun event, got {after - before}."


def test_no_task_failure_handler_subscribed():
    """task_failure must have NO receivers from this module.

    Defense in depth: the double-count bug returns the moment a future PR
    re-introduces a @task_failure.connect handler.
    """
    from celery.signals import task_failure

    receivers = [
        r
        for r in task_failure.receivers
        if r[1]() is not None  # weakref dereference
    ]
    receiver_modules = {getattr(r[1](), "__module__", None) for r in receivers}
    assert "agentic_project_service.celery" not in receiver_modules, (
        "agentic_project_service.celery has re-subscribed to task_failure — "
        "this re-introduces the double-count bug fixed in S1."
    )
