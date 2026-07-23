"""Tests that ``reasoning_effort`` flows from agent_settings into Agent constructors.

This is Task 6 of the reasoning-streaming-redesign plan: each ``Agent(...)`` call
site in ``routes/agents.py`` must pass ``reasoning_effort=agent_settings.get("reasoning_effort")``.

Approach: source/AST inspection of ``routes/agents.py`` rather than route-level
integration tests. Rationale:

1. The route-level test infrastructure currently fails to bring up the Flask app
   on this branch due to a pre-existing dev-DB alembic version mismatch
   (the local DB is stamped at ``0017_ai_provider_keys`` from another branch
   while this branch only has migrations through 0016). That's an environment
   issue unrelated to this task; touching the DB to fix it is destructive and
   forbidden by project conventions.

2. AST-based verification *directly* asserts the wire-up at every site,
   by structure rather than by line number. It is strictly stronger than a
   single route-level test that exercises only one path — and it survives
   future refactors that move code around.

3. The fall-back AST/regex test was explicitly listed as an acceptable path
   in the task description.

If/when the dev-DB issue is resolved, route-level integration coverage can be
added in ``tests/route/test_agent_streaming.py`` to confirm end-to-end behavior.
"""

from __future__ import annotations

import ast
from pathlib import Path


AGENTS_PY = (
    Path(__file__).resolve().parents[2] / "src" / "agentic_project_service" / "routes" / "agents.py"
)


def _load_agent_calls() -> list[ast.Call]:
    """Return every ``Agent(...)`` Call node in routes/agents.py.

    Picks calls where ``func`` is a ``Name`` node with ``id == "Agent"``.
    """
    source = AGENTS_PY.read_text()
    tree = ast.parse(source, filename=str(AGENTS_PY))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Agent"
        ):
            calls.append(node)
    return calls


def _has_reasoning_effort_kwarg(call: ast.Call) -> bool:
    """Return True iff the call has a kwarg
    ``reasoning_effort=agent_settings.get("reasoning_effort")``.

    The exact form we require:
        keyword(arg="reasoning_effort",
                value=Call(func=Attribute(value=Name("agent_settings"), attr="get"),
                           args=[Constant("reasoning_effort")]))
    """
    for kw in call.keywords:
        if kw.arg != "reasoning_effort":
            continue
        v = kw.value
        if not isinstance(v, ast.Call):
            return False
        if not isinstance(v.func, ast.Attribute):
            return False
        if v.func.attr != "get":
            return False
        if not (isinstance(v.func.value, ast.Name) and v.func.value.id == "agent_settings"):
            return False
        if len(v.args) < 1:
            return False
        first_arg = v.args[0]
        if not (isinstance(first_arg, ast.Constant) and first_arg.value == "reasoning_effort"):
            return False
        return True
    return False


class TestAgentConstructorWiring:
    """Each Agent(...) call site in routes/agents.py must pass
    reasoning_effort=agent_settings.get("reasoning_effort")."""

    def test_three_agent_call_sites_exist(self) -> None:
        """Sanity: there should be exactly 3 Agent(...) construction sites."""
        calls = _load_agent_calls()
        assert len(calls) == 3, (
            f"Expected exactly 3 Agent(...) call sites in {AGENTS_PY}, "
            f"found {len(calls)} (lines: {[c.lineno for c in calls]})"
        )

    def test_all_agent_call_sites_pass_reasoning_effort(self) -> None:
        """Every Agent(...) call site must include
        reasoning_effort=agent_settings.get("reasoning_effort")."""
        calls = _load_agent_calls()
        missing = [c.lineno for c in calls if not _has_reasoning_effort_kwarg(c)]
        assert missing == [], (
            "These Agent(...) call sites are missing "
            'reasoning_effort=agent_settings.get("reasoning_effort"): '
            f"lines {missing}"
        )
