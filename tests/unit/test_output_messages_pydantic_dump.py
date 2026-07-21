"""Test that the 4 output_messages writes in routes/agents.py and any in
routes/orchestrations.py construct Message(...).model_dump(exclude_none=True)
rather than dict literals.

AST-based — same approach as test_agents_reasoning_effort.py."""

from __future__ import annotations

import ast
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[5]
ROUTES_AGENTS = (
    WORKTREE
    / "agentic-platform/packages/agentic-project-service/src/agentic_project_service/routes/agents.py"
)
ROUTES_ORCH = (
    WORKTREE
    / "agentic-platform/packages/agentic-project-service/src/agentic_project_service/routes/orchestrations.py"
)


def _output_messages_assignments(path: Path) -> list[ast.AST]:
    """Find all `output_messages = [...]` assignments and `output_messages=[...]` kwargs."""
    tree = ast.parse(path.read_text())
    matches: list[ast.AST] = []
    for node in ast.walk(tree):
        # Form 1: output_messages = [...]
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "output_messages":
                    matches.append(node)
        # Form 2: output_messages=[...] in a call kwargs
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "output_messages":
                    matches.append(kw)
    return matches


def _value_is_pydantic_dump(value: ast.AST) -> bool:
    """Return True if the value is a list containing one element of the shape
    Message(...).model_dump(...) — i.e., a Call to .model_dump on a Message
    constructor."""
    if not isinstance(value, ast.List):
        return False
    if len(value.elts) != 1:
        return False
    elt = value.elts[0]
    # Expect: Call(func=Attribute(value=Call(func=Name("Message")), attr="model_dump"), ...)
    if not (isinstance(elt, ast.Call) and isinstance(elt.func, ast.Attribute)):
        return False
    if elt.func.attr != "model_dump":
        return False
    inner = elt.func.value
    if not (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == "Message"
    ):
        return False
    return True


def test_routes_agents_output_messages_use_pydantic_dump():
    nodes = _output_messages_assignments(ROUTES_AGENTS)
    # We expect at least 4 assignments + however many kwargs
    assert len(nodes) >= 4, f"Expected >=4 output_messages writes; found {len(nodes)}"
    failing = []
    for node in nodes:
        value = node.value if isinstance(node, ast.Assign) else node.value
        if not _value_is_pydantic_dump(value):
            failing.append(node.lineno if hasattr(node, "lineno") else "?")
    assert not failing, f"Sites NOT using Message.model_dump: {failing}"


def test_routes_orchestrations_output_messages_use_pydantic_dump():
    if not ROUTES_ORCH.exists():
        return  # ok if no orchestration write site
    nodes = _output_messages_assignments(ROUTES_ORCH)
    failing = []
    for node in nodes:
        value = node.value if isinstance(node, ast.Assign) else node.value
        if not _value_is_pydantic_dump(value):
            failing.append(node.lineno if hasattr(node, "lineno") else "?")
    assert not failing, f"Sites NOT using Message.model_dump: {failing}"
