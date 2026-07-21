"""Tests that tasks/indexing.py wires the LiteLLM cost accumulator so
each indexed_source row gets its per-stage LLM cost rolled up into stats.

Uses textual/AST inspection because the Celery task has heavy DB/storage
dependencies that make full integration tests prohibitive in the unit
layer.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

INDEXING_PY = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "agentic_project_service"
    / "tasks"
    / "indexing.py"
)


def _source() -> str:
    return INDEXING_PY.read_text()


def test_indexing_imports_cost_accumulator():
    """The module must import install and init_accumulator from the agentic
    cost_accumulator helper."""
    src = _source()
    pattern = re.compile(
        r"from\s+agentic\.llm\.cost_accumulator\s+import\s+[^\n]*"
        r"(install|init_accumulator)"
    )
    assert pattern.search(src), (
        "tasks/indexing.py must import install and init_accumulator from "
        "agentic.llm.cost_accumulator."
    )


def test_indexing_calls_install_at_module_scope():
    """install() must be called at module scope so the LiteLLM callback is
    registered exactly once when the worker boots, not per-task."""
    src = _source()
    tree = ast.parse(src, filename=str(INDEXING_PY))
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "install"
        ):
            return
    raise AssertionError(
        "tasks/indexing.py must call install() at module scope to register the "
        "LiteLLM cost-accumulator callback once on worker boot."
    )


def test_indexing_initializes_accumulator_before_strategy_dispatch():
    """init_accumulator() must be called before the strategy dispatch so the
    ContextVar is set in the calling thread's context — every asyncio.run()
    that follows snapshots the same accumulator object."""
    src = _source()
    init_pos = src.find("init_accumulator(")
    dispatch_pos = src.find('if strategy == "page_index"')
    assert init_pos != -1, "init_accumulator() must be called somewhere"
    assert dispatch_pos != -1, "strategy dispatch chain not found"
    assert init_pos < dispatch_pos, (
        "init_accumulator() must be called before the strategy dispatch."
    )


def test_indexing_merges_costs_into_stats_before_persist():
    """The accumulator's to_dict() output must be merged into stats before
    update_indexed_source_result(...) writes to the DB, so the cost data
    lands on the same row."""
    src = _source()
    # Pattern: stats["llm_costs"] = <something with to_dict>
    pattern = re.compile(r'stats\[["\']llm_costs["\']\]\s*=\s*[^\n]*\.to_dict\(\)')
    assignment = pattern.search(src)
    assert assignment, (
        'tasks/indexing.py must assign stats["llm_costs"] = <acc>.to_dict() '
        "before calling update_indexed_source_result(...)."
    )
    persist_pos = src.find("update_indexed_source_result(indexed_source_id, stats)")
    assert persist_pos != -1, "update_indexed_source_result call not found"
    assert assignment.start() < persist_pos, (
        'stats["llm_costs"] = ... must come BEFORE update_indexed_source_result.'
    )
