"""Lint-shaped test: every chargeable Celery task in tasks/ has @billing_context
OR @no_billing_context. Reads tasks via celery.app inspection."""

import importlib
import pkgutil


def _iter_task_functions():
    """Walk agentic_project_service.tasks.* and yield each Celery task object."""
    import agentic_project_service.tasks as tasks_pkg

    for _, name, _ in pkgutil.walk_packages(tasks_pkg.__path__, "agentic_project_service.tasks."):
        mod = importlib.import_module(name)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            # celery task objects have a .delay attribute and __wrapped__ (the user fn)
            if hasattr(obj, "delay") and hasattr(obj, "__wrapped__"):
                yield obj


def test_all_chargeable_tasks_wrapped():
    """Either @billing_context OR @no_billing_context must mark each task."""
    unwrapped = []
    for task in _iter_task_functions():
        fn = task.__wrapped__  # Celery wraps the user-supplied function; for @shared_task @billing_context def fn, __wrapped__ is the billing_context wrapper
        if getattr(fn, "__no_billing_context__", False):
            continue
        if getattr(fn, "__has_billing_context__", False):
            continue
        unwrapped.append(task.name)
    assert not unwrapped, (
        f"Celery tasks missing @billing_context or @no_billing_context: {unwrapped}.\n"
        "Add @billing_context for tasks that may invoke LLMs (chargeable). "
        "Add @no_billing_context for tasks that do not."
    )
