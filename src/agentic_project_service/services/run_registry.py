"""In-memory registry of active run contexts for approval endpoint.

WARNING: This registry is purely in-memory. If the project-service restarts
while a run is waiting for approval, the approval state is lost and the run
cannot be resumed. The user will see a stale "waiting for approval" UI.

Future improvement: persist approval state to the database so runs can be
resumed after restart, or auto-timeout pending approvals.
"""

from agentic.execution.context import ExecutionContext

_active_runs: dict[str, ExecutionContext] = {}


def register_run(run_id: str, context: ExecutionContext):
    _active_runs[run_id] = context


def get_active_run_context(run_id: str) -> ExecutionContext | None:
    return _active_runs.get(run_id)


def unregister_run(run_id: str):
    _active_runs.pop(run_id, None)
