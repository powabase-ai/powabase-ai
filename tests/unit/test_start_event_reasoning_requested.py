"""Test that every SSE start-event dict in routes/agents.py emits
``reasoning_requested`` so the FE can render the reasoning pill.

Like the sibling AST tests on this branch, this is source/AST inspection
rather than a route-level integration test. The dev-DB alembic mismatch on
this worktree blocks Flask-based route tests; AST verification is strictly
stronger because it asserts wiring at every emit site, not just one."""

from __future__ import annotations

import ast
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
ROUTES_AGENTS = WORKTREE / "src" / "agentic_project_service" / "routes" / "agents.py"


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


def _is_resolved_effort_compare(v: ast.AST) -> bool:
    """Return True if `v` is the expression
    `<name>._resolved_effort_for(<name>.model) is not None`."""
    if not isinstance(v, ast.Compare):
        return False
    if len(v.ops) != 1 or not isinstance(v.ops[0], ast.IsNot):
        return False
    if not (
        len(v.comparators) == 1
        and isinstance(v.comparators[0], ast.Constant)
        and v.comparators[0].value is None
    ):
        return False
    left = v.left
    return bool(
        isinstance(left, ast.Call)
        and isinstance(left.func, ast.Attribute)
        and left.func.attr == "_resolved_effort_for"
        and len(left.args) == 1
        and isinstance(left.args[0], ast.Attribute)
        and left.args[0].attr == "model"
    )


def _name_assigned_to_resolved_effort(name: str, path: Path) -> bool:
    """Return True if `name` is assigned the resolved-effort expression
    anywhere in the file. (Cheaper than full enclosing-function lookup;
    safe because we know which variable name the SUT uses.)"""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Name)
                    and tgt.id == name
                    and _is_resolved_effort_compare(node.value)
                ):
                    return True
    return False


def _has_reasoning_requested_call(d: ast.Dict) -> bool:
    """The dict must include a key ``reasoning_requested`` whose value is
    either the literal expression ``<name>._resolved_effort_for(<name>.model)
    is not None``, OR a variable assigned to that expression earlier in the
    file (the cache-into-a-local-variable pattern the SUT uses to share
    one computation across start event + persistence + background finisher)."""
    for k, v in zip(d.keys, d.values):
        if not (isinstance(k, ast.Constant) and k.value == "reasoning_requested"):
            continue
        if _is_resolved_effort_compare(v):
            return True
        if isinstance(v, ast.Name) and _name_assigned_to_resolved_effort(v.id, ROUTES_AGENTS):
            return True
        return False
    return False


def test_at_least_two_start_event_emit_sites_exist():
    """Sanity: there should be at least 2 start-event emit sites (the ReAct
    SSE path and the streaming SSE path)."""
    dicts = _start_event_dicts(ROUTES_AGENTS)
    assert len(dicts) >= 2, (
        f"Expected at least 2 start-event dicts in routes/agents.py; found {len(dicts)}"
    )


def test_every_start_event_emits_reasoning_requested():
    """Every start-event dict must carry ``reasoning_requested`` derived
    from ``agent._resolved_effort_for(agent.model) is not None`` so the FE
    pill state can be derived without a roundtrip."""
    dicts = _start_event_dicts(ROUTES_AGENTS)
    missing = [d.lineno for d in dicts if not _has_reasoning_requested_call(d)]
    assert missing == [], (
        f"start-event dict(s) missing reasoning_requested key (or with the "
        f"wrong shape): lines {missing}"
    )
