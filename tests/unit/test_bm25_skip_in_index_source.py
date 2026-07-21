"""Tests that all sparse_store.add_and_save / remove_and_save callsites in
`tasks/indexing.py` are properly gated by `_should_build_bm25_now`.

Test approach: read the source code and assert every matching callsite is
preceded by `if _should_build_bm25_now(kb_id)`. This is intentionally
source-level — verifying the wrapping pattern is in place at every
location, which guards against future regressions when someone adds a
fourth strategy or rearranges code.
"""

from __future__ import annotations

import inspect
import re

from agentic_project_service.tasks import indexing as indexing_module


SOURCE = inspect.getsource(indexing_module)


def test_all_add_and_save_callsites_are_guarded():
    """Every `sparse_store.add_and_save(` callsite must be preceded by
    an `if _should_build_bm25_now(` check on a recent prior line."""
    lines = SOURCE.split("\n")
    add_callsite_lines = [
        i
        for i, line in enumerate(lines)
        if "sparse_store.add_and_save(" in line
        # Exclude docstring / comment mentions (lines that start with # or ''')
        and not line.lstrip().startswith(("#", '"""', "'''"))
    ]
    assert len(add_callsite_lines) >= 3, (
        "expected at least 3 add_and_save callsites (chunk_embed, page_index, "
        f"graph_index); found {len(add_callsite_lines)}"
    )
    for line_idx in add_callsite_lines:
        # Walk back up to 10 lines; expect to find the guard
        window = "\n".join(lines[max(0, line_idx - 10) : line_idx])
        assert re.search(r"if\s+_should_build_bm25_now\s*\(", window), (
            f"add_and_save callsite at line {line_idx + 1} not preceded by "
            f"_should_build_bm25_now guard. Window: {window!r}"
        )


def test_all_remove_and_save_callsites_are_guarded():
    """Every `sparse_store.remove_and_save(` callsite must be similarly guarded."""
    lines = SOURCE.split("\n")
    remove_callsite_lines = [
        i
        for i, line in enumerate(lines)
        if "sparse_store.remove_and_save(" in line
        # Exclude docstring / comment mentions (lines that start with # or ''')
        and not line.lstrip().startswith(("#", '"""', "'''"))
    ]
    assert len(remove_callsite_lines) >= 3, (
        f"expected >=3 remove_and_save callsites; found {len(remove_callsite_lines)}"
    )
    for line_idx in remove_callsite_lines:
        window = "\n".join(lines[max(0, line_idx - 10) : line_idx])
        assert re.search(r"if\s+_should_build_bm25_now\s*\(", window), (
            f"remove_and_save callsite at line {line_idx + 1} not preceded by "
            f"_should_build_bm25_now guard. Window: {window!r}"
        )
