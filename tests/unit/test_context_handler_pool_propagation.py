"""Tests for run_id contextvar propagation through context_handler's
multi-KB parallel ThreadPoolExecutor (reviewer C7).

Two complementary checks:

1. Structural (AST) — assert the production `executor.submit(...)` at
   services/context_handler.py:637 wraps its callable in
   ``contextvars.copy_context().run``. Catches a regression where a
   future contributor removes the wrap and silently drops back to
   uuid4 idempotency keys on multi-KB retries.

2. Behavioral — drive a real ThreadPoolExecutor through the documented
   pattern with a spy on ``run_context.get_run_id()`` from inside
   workers. Confirms the wrapping pattern actually propagates the
   bound run_id (independent of the production file's specific shape).
"""

from __future__ import annotations

import ast
import contextvars
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentic_project_service.services import run_context


CONTEXT_HANDLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "agentic_project_service"
    / "services"
    / "context_handler.py"
)

# agentic library is installed as a path dependency in agentic-platform's
# pyproject. Resolve its on-disk location so the AST guards below can
# verify the in-tree files even when the package is editable-installed.
_AGENTIC_SRC = Path(__file__).resolve().parents[5] / "agentic" / "src" / "agentic"
AGENT_PATH = _AGENTIC_SRC / "agent" / "agent.py"
STRATEGIES_PATH = _AGENTIC_SRC / "orchestration" / "strategies.py"


def _executor_submit_calls(tree: ast.AST) -> list[ast.Call]:
    """Return every ``<something>.submit(...)`` call in the tree."""
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "submit"
        ):
            out.append(node)
    return out


def _is_copy_context_run(arg: ast.AST) -> bool:
    """True iff ``arg`` is the AST for ``copy_context().run`` in either
    qualified (``contextvars.copy_context().run``) or imported
    (``from contextvars import copy_context; copy_context().run``) form.

    Both are correct usage; tests should accept either to avoid forcing
    a particular import style.
    """
    if not (isinstance(arg, ast.Attribute) and arg.attr == "run"):
        return False
    inner = arg.value
    if not isinstance(inner, ast.Call):
        return False
    # Form 1: ``contextvars.copy_context().run``
    if (
        isinstance(inner.func, ast.Attribute)
        and inner.func.attr == "copy_context"
        and isinstance(inner.func.value, ast.Name)
        and inner.func.value.id == "contextvars"
    ):
        return True
    # Form 2: ``copy_context().run`` (via ``from contextvars import``)
    if isinstance(inner.func, ast.Name) and inner.func.id == "copy_context":
        return True
    return False


def test_context_handler_parallel_pool_submit_wraps_with_copy_context():
    """Structural guard: services/context_handler.py:637 must wrap its
    executor.submit callable with ``contextvars.copy_context().run`` so
    each worker enters a fresh copy of the parent context.

    Without this, get_run_id() returns None inside the worker and
    _derive_tool_idempotency_inputs (via search_knowledge_base) falls
    back to uuid4 — multi-KB retries then double-charge for
    vector_search / metadata_enrichment / reranker_call.

    The check looks at every submit call in the file (currently one) and
    requires its first positional arg to be the copy_context().run
    pattern, NOT bare ``_search_single_kb`` or similar.
    """
    src = CONTEXT_HANDLER_PATH.read_text()
    tree = ast.parse(src)
    submits = _executor_submit_calls(tree)
    assert submits, "context_handler.py must contain at least one executor.submit call"

    for call in submits:
        assert call.args, (
            f"executor.submit at line {call.lineno} has no positional args; "
            f"first arg must be contextvars.copy_context().run"
        )
        first = call.args[0]
        assert _is_copy_context_run(first), (
            f"executor.submit at line {call.lineno}: first positional arg is "
            f"{ast.unparse(first)!r}, expected contextvars.copy_context().run "
            f"(see agent.py/strategies.py for the same pattern + rationale)"
        )


def test_agentic_agent_concurrent_tool_pool_wraps_with_copy_context():
    """Companion structural guard for agentic.agent.agent's concurrent-tool
    ThreadPoolExecutor (the larger of the two original C4 sites). If a
    future contributor removes ``contextvars.copy_context().run`` from
    the per-submission wrap there, every parallel tool call mints uuid4
    keys again and retries double-charge.

    Skipped (rather than failed) when the agentic source isn't on disk
    — wheel-only installs don't ship .py files in a single greppable tree.
    """
    if not AGENT_PATH.exists():
        import pytest

        pytest.skip(f"agentic source not on disk at {AGENT_PATH}")
    tree = ast.parse(AGENT_PATH.read_text())
    submits = _executor_submit_calls(tree)
    # agent.py has several submits if the file grows; check that AT LEAST
    # one wraps with copy_context().run. Tighter: every submit inside the
    # concurrent-tool block. The block-local check is enforced by the
    # surrounding production logic; a per-file "at least one wrapped"
    # guard catches accidental wholesale removal.
    wrapped = [c for c in submits if c.args and _is_copy_context_run(c.args[0])]
    assert wrapped, (
        f"{AGENT_PATH}: no executor.submit call wraps with "
        f"contextvars.copy_context().run — the concurrent-tool pool fix "
        f"may have been reverted (see services/run_context.py for "
        f"the propagation contract)."
    )


def test_agentic_orchestration_parallel_engine_pool_wraps_with_copy_context():
    """Same guard as above for ParallelEngine in agentic.orchestration."""
    if not STRATEGIES_PATH.exists():
        import pytest

        pytest.skip(f"agentic source not on disk at {STRATEGIES_PATH}")
    tree = ast.parse(STRATEGIES_PATH.read_text())
    submits = _executor_submit_calls(tree)
    wrapped = [c for c in submits if c.args and _is_copy_context_run(c.args[0])]
    assert wrapped, (
        f"{STRATEGIES_PATH}: ParallelEngine's executor.submit must wrap "
        f"with contextvars.copy_context().run."
    )


def test_pool_submit_via_copy_context_propagates_run_id_to_workers():
    """Behavioral confirmation that the documented wrapping pattern
    actually does what context_handler depends on it doing: each
    submitted callable runs inside a fresh copy of the parent context,
    so get_run_id() inside the worker thread observes the binding.

    Mirrors how context_handler.py:637 submits _search_single_kb, but
    uses an in-test spy as the target to avoid pulling DB dependencies
    into this unit test. The structural test above pins the production
    wrap; this one pins the wrap's behavior."""
    captured: list[str | None] = []

    def spy(_arg=None):
        captured.append(run_context.get_run_id())

    # Pre-condition: the test thread has no run_id bound.
    assert run_context.get_run_id() is None

    token = run_context.set_run_id("run_ctx_handler_test")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(contextvars.copy_context().run, spy, i) for i in range(2)]
            for f in futures:
                f.result()
    finally:
        run_context.reset_run_id(token)

    assert captured == [
        "run_ctx_handler_test",
        "run_ctx_handler_test",
    ], f"expected workers to see the bound run_id; got {captured!r}"
