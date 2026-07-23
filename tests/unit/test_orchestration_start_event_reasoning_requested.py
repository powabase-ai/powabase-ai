"""Test that the SSE start-event in routes/orchestrations.py emits
``reasoning_requested`` so the FE pill renders for orchestration runs.

Bug: orchestrations.py emitted only {event, run_id, session_id} on stream
start, so the FE's ReasoningPill (gated on msg.reasoning_requested per
runs/index.tsx:116) never rendered for orchestration runs — even when the
supervisor agent emitted reasoning_delta events through the same SSE pipe.

This is a source/AST inspection test rather than a route-level integration
test. The dev-DB alembic mismatch on this worktree blocks Flask-based route
tests; AST verification is strictly stronger because it asserts wiring at
every emit site, not just whichever single path a route test exercises.
Mirrors test_start_event_reasoning_requested.py for the agents route."""

from __future__ import annotations

import ast
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
ROUTES_ORCHESTRATIONS = (
    WORKTREE / "src" / "agentic_project_service" / "routes" / "orchestrations.py"
)


def _start_event_dicts(path: Path) -> list[ast.Dict]:
    """Find every dict literal whose 'event' key is the constant 'start'."""
    tree = ast.parse(path.read_text())
    matches: list[ast.Dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if (
                isinstance(k, ast.Constant)
                and k.value == "event"
                and isinstance(v, ast.Constant)
                and v.value == "start"
            ):
                matches.append(node)
                break
    return matches


def _has_reasoning_requested_key(d: ast.Dict) -> bool:
    """The dict must include a key ``reasoning_requested`` with any value
    (we don't pin the exact expression here — the agents route uses
    ``agent._resolved_effort_for(...)`` because the agent already exists at
    that point; the orchestrations route has to compute it from
    ``orchestrator_config`` + ``litellm.supports_reasoning`` because the
    orchestrator Agent isn't constructed until inside the worker thread)."""
    for k, _v in zip(d.keys, d.values):
        if isinstance(k, ast.Constant) and k.value == "reasoning_requested":
            return True
    return False


def test_orchestration_start_event_emits_reasoning_requested():
    """The SSE start event for orchestration runs must include
    ``reasoning_requested`` so the FE pill knows when to render."""
    dicts = _start_event_dicts(ROUTES_ORCHESTRATIONS)
    assert len(dicts) >= 1, (
        f"Expected at least 1 start-event dict in routes/orchestrations.py; found {len(dicts)}"
    )
    missing = [d.lineno for d in dicts if not _has_reasoning_requested_key(d)]
    assert missing == [], (
        f"start-event dict(s) in routes/orchestrations.py missing the "
        f"reasoning_requested key: lines {missing}"
    )
