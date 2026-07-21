"""Test that task_received signal increments the celery_tasks_total counter with status=received."""

from prometheus_client import REGISTRY

# Side-effect import: registers the `_on_task_received` handler via @task_received.connect.
# Without this, the conftest's `patch("agentic_project_service.celery.get_flask_app")` is
# the only thing importing celery — fragile if conftest changes.
import agentic_project_service.celery  # noqa: F401


def test_task_received_emits_counter():
    """Send a synthetic Celery task_received signal; assert counter increments."""
    from celery.signals import task_received

    # Snapshot counter value before
    before = (
        REGISTRY.get_sample_value(
            "celery_tasks_total",
            labels={"task": "test_task", "status": "received"},
        )
        or 0
    )

    # Fire the signal — Celery's signal.send dispatches to all connected handlers
    class FakeRequest:
        task = "test_task"

    task_received.send(sender=None, request=FakeRequest())

    after = REGISTRY.get_sample_value(
        "celery_tasks_total",
        labels={"task": "test_task", "status": "received"},
    )
    assert (
        after == before + 1
    ), f"Expected counter to increment from {before} to {before + 1}, got {after}"
