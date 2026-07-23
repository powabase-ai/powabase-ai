"""Unit tests for the orchestration replay helpers + AST checks for the
messages endpoint.

Issue #106 Bug B: ``get_orchestration_session_messages`` must return per-run
reasoning replay metadata so the FE pill renders after page refresh. This
file tests:

  1. The pure ``_summary_from_events`` helper that reconstructs summary_text
     from persisted terminal ``reasoning`` events (reasoning_delta is filtered
     out at persistence time per orchestrations.py:657).
  2. AST-level wiring: the route must read ``reasoning_requested`` and the
     timestamp columns from the run row, and must include the four reasoning
     replay keys on assistant messages.
"""

from __future__ import annotations

import ast
from pathlib import Path

from agentic_project_service.routes.orchestrations import _summary_from_events

WORKTREE = Path(__file__).resolve().parents[2]
ROUTES_ORCHESTRATIONS = (
    WORKTREE / "src" / "agentic_project_service" / "routes" / "orchestrations.py"
)


# ===== _summary_from_events =====


def test_summary_concatenates_reasoning_event_contents():
    events = [
        {"type": "step_started", "step": 1},
        {"type": "reasoning", "step": 1, "content": "First thought.", "source": "thinking"},
        {"type": "tool_call", "step": 1, "name": "search"},
        {"type": "step_started", "step": 2},
        {"type": "reasoning", "step": 2, "content": "Second thought.", "source": "thinking"},
    ]
    assert _summary_from_events(events) == "First thought.\n\nSecond thought."


def test_summary_returns_none_when_no_reasoning_events():
    """No reasoning events → done-empty pill state on FE."""
    events = [
        {"type": "step_started", "step": 1},
        {"type": "tool_call", "step": 1, "name": "search"},
        {"type": "tool_result", "step": 1, "tool_name": "search"},
    ]
    assert _summary_from_events(events) is None


def test_summary_returns_none_for_empty_input():
    assert _summary_from_events([]) is None
    assert _summary_from_events(None) is None  # type: ignore[arg-type]


def test_summary_skips_empty_content_strings():
    events = [
        {"type": "reasoning", "content": ""},
        {"type": "reasoning", "content": "real text"},
    ]
    assert _summary_from_events(events) == "real text"


def test_summary_skips_non_string_content():
    events = [
        {"type": "reasoning", "content": None},
        {"type": "reasoning", "content": ["list", "not", "str"]},
        {"type": "reasoning", "content": "valid"},
    ]
    assert _summary_from_events(events) == "valid"


def test_summary_ignores_non_reasoning_event_types():
    events = [
        {"type": "step_started", "content": "should not appear"},
        {"type": "tool_call", "content": "also no"},
        {"type": "reasoning", "content": "yes"},
    ]
    assert _summary_from_events(events) == "yes"


def test_summary_skips_non_dict_entries():
    events = [
        "not a dict",
        None,
        {"type": "reasoning", "content": "ok"},
    ]
    assert _summary_from_events(events) == "ok"  # type: ignore[arg-type]


# ===== AST-level checks on the messages endpoint =====


def _function_def(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found in {path}")


def _attribute_accesses(node: ast.AST) -> set[str]:
    """All `<obj>.attr` reads under this node, returned as the attr names."""
    seen: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute):
            seen.add(n.attr)
    return seen


def _string_constants(node: ast.AST) -> set[str]:
    seen: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            seen.add(n.value)
    return seen


def test_messages_endpoint_reads_reasoning_requested_and_timestamps():
    fn = _function_def(ROUTES_ORCHESTRATIONS, "get_orchestration_session_messages")
    attrs = _attribute_accesses(fn)
    # Must access these columns from the run row to assemble the reply
    assert "reasoning_requested" in attrs, (
        "messages endpoint must read run.reasoning_requested to gate the pill"
    )
    assert "started_at" in attrs and "completed_at" in attrs, (
        "messages endpoint must read run.started_at and run.completed_at to "
        "compute reasoning_duration_ms"
    )


def test_messages_endpoint_emits_replay_keys():
    """Assistant message dicts must include the four FE-side replay keys.
    ``events`` is plural and may also appear in other contexts; we just
    assert all four key names appear as string constants in the function."""
    fn = _function_def(ROUTES_ORCHESTRATIONS, "get_orchestration_session_messages")
    constants = _string_constants(fn)
    for key in (
        "reasoning_requested",
        "reasoning_duration_ms",
        "reasoning",
        "events",
    ):
        assert key in constants, (
            f"messages endpoint must emit '{key}' key on assistant messages "
            f"(missing from string constants — got {sorted(constants)})"
        )
