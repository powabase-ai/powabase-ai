"""Tests that tasks.indexing copies reasoning_effort from indexing_config
into the 'extra' dict that feeds PageIndexAlgorithm.

Uses textual/AST inspection of run_graph_index_indexing because the function
has DB and Celery dependencies that make full integration tests prohibitive
in the unit-test layer. AST/regex tests directly verify the wire-up and
survive refactors that move surrounding code.
"""

from __future__ import annotations

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


def test_run_graph_index_propagates_reasoning_effort_to_extra():
    """The 'extra' dict assembled before invoking PageIndexAlgorithm must
    conditionally copy indexing_config['reasoning_effort'] when present."""
    src = _source()
    pattern = re.compile(
        r'if\s+["\']reasoning_effort["\']\s+in\s+indexing_config\s*:\s*\n'
        r'\s+extra\[["\']reasoning_effort["\']\]\s*=\s*indexing_config\[["\']reasoning_effort["\']\]'
    )
    assert pattern.search(src), (
        "tasks/indexing.py must conditionally copy "
        "indexing_config['reasoning_effort'] into the extra dict "
        "before calling PageIndexAlgorithm.aindex(...)."
    )


import ast


def test_run_graph_index_passes_enrichment_reasoning_effort():
    """The call to enrich_referenced_nodes(...) must pass
    reasoning_effort=indexing_config.get('enrichment_reasoning_effort')."""
    src = INDEXING_PY.read_text()
    tree = ast.parse(src, filename=str(INDEXING_PY))

    matched = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_target = (isinstance(func, ast.Name) and func.id == "enrich_referenced_nodes") or (
            isinstance(func, ast.Attribute) and func.attr == "enrich_referenced_nodes"
        )
        if not is_target:
            continue
        for kw in node.keywords:
            if kw.arg != "reasoning_effort":
                continue
            v = kw.value
            ok = (
                isinstance(v, ast.Call)
                and isinstance(v.func, ast.Attribute)
                and v.func.attr == "get"
                and isinstance(v.func.value, ast.Name)
                and v.func.value.id == "indexing_config"
                and len(v.args) == 1
                and isinstance(v.args[0], ast.Constant)
                and v.args[0].value == "enrichment_reasoning_effort"
            )
            if ok:
                matched = True
                break
        if matched:
            break

    assert matched, (
        "tasks/indexing.py must call enrich_referenced_nodes(...) with "
        'kwarg reasoning_effort=indexing_config.get("enrichment_reasoning_effort").'
    )
