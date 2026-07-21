"""Test that reasoning_effort wires through to orchestrator + worker Agents.

AST-based source inspection — same approach as test_agents_reasoning_effort.py
(Task 6) because route-level tests are blocked by a dev-DB alembic issue on
this worktree."""

from __future__ import annotations

import ast
from pathlib import Path


# Project root resolved from this test's location
WORKTREE = Path(__file__).resolve().parents[5]
PROJECT_SERVICE = (
    WORKTREE
    / "agentic-platform/packages/agentic-project-service/src/agentic_project_service/services/orchestration.py"
)
STRATEGIES = WORKTREE / "agentic/src/agentic/orchestration/strategies.py"


def _agent_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Agent"
    ]


def _has_reasoning_effort_kwarg(
    call: ast.Call, source_var: str, key: str = "reasoning_effort"
) -> bool:
    """Strict: kwarg value must be exactly `<source_var_chain>.get("reasoning_effort")`.

    source_var_chain can be a Name ('agent_row.settings or {}' is parsed as a
    BoolOp) or an Attribute (orchestration.orchestrator_config). The check
    walks the Call's structure.
    """
    for kw in call.keywords:
        if kw.arg != "reasoning_effort":
            continue
        v = kw.value
        if not (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Attribute)
            and v.func.attr == "get"
            and len(v.args) >= 1
            and isinstance(v.args[0], ast.Constant)
            and v.args[0].value == key
        ):
            return False
        # Verify the LHS of `.get(...)` mentions the expected source var
        target_str = ast.unparse(v.func.value)
        return source_var in target_str
    return False


def test_worker_agent_passes_reasoning_effort_from_agent_row_settings():
    """services/orchestration.py worker Agent uses agent_row.settings."""
    calls = _agent_calls(PROJECT_SERVICE)
    assert len(calls) >= 1, f"No Agent(...) calls found in {PROJECT_SERVICE}"
    matching = [c for c in calls if _has_reasoning_effort_kwarg(c, "agent_row.settings")]
    assert len(matching) >= 1, (
        "No Agent(...) call in services/orchestration.py passes "
        "reasoning_effort from agent_row.settings"
    )


def test_orchestrator_agent_passes_reasoning_effort_from_orchestrator_config():
    """strategies.py orchestrator Agent uses orchestration.orchestrator_config."""
    calls = _agent_calls(STRATEGIES)
    assert len(calls) >= 1, f"No Agent(...) calls found in {STRATEGIES}"
    matching = [c for c in calls if _has_reasoning_effort_kwarg(c, "orchestrator_config")]
    assert len(matching) >= 1, (
        "No Agent(...) call in strategies.py passes "
        "reasoning_effort from orchestration.orchestrator_config"
    )
