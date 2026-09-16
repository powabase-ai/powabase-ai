"""The per-source bm25s file-index maintenance runs until pg_search serves the KB.

With pg_search serving a knowledge base, nothing reads its bm25s file index, so
re-tokenising every indexed source into it is pure cost (and for a large KB, a
large one). But "serves" is per knowledge base: its own partition, with its own
index ready. Until then search reads the file index, so it has to stay current.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentic_project_service.services import pg_bm25_index as pgb
from agentic_project_service.tasks import indexing

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


@pytest.fixture(autouse=True)
def _clear_caches():
    pgb.reset_pg_bm25_caches()
    yield
    pgb.reset_pg_bm25_caches()


def _session(installed: bool, relkind: str | None):
    session = MagicMock()
    session.get_execution_options.return_value = {}

    def execute(statement, params=None):
        sql = getattr(statement, "text", str(statement))
        result = MagicMock()
        if "pg_extension" in sql:
            result.first.return_value = (1,) if installed else None
        elif "relkind" in sql:
            result.first.return_value = (relkind,) if relkind else None
        else:
            result.first.return_value = None
        return result

    session.execute.side_effect = execute
    return session


@pytest.mark.parametrize(
    ("installed", "relkind", "strategy", "expected"),
    [
        (True, "p", "chunk_embed", "pg_search"),
        (True, "p", None, "pg_search"),
        (True, "r", "chunk_embed", "bm25s"),
        (False, "p", "chunk_embed", "bm25s"),
        (True, "p", "full_document", "pg_search"),
        (True, "p", "graph_index", "pg_search"),
        (True, "p", "page_index", None),
        (True, "p", "doc2json", None),
    ],
)
def test_keyword_index_backend(installed, relkind, strategy, expected):
    session = _session(installed, relkind)
    assert pgb.keyword_index_backend(session, strategy) == expected


def test_keyword_index_backend_probes_in_a_savepoint_and_never_raises():
    session = _session(True, "p")
    session.execute.side_effect = [MagicMock(first=MagicMock(return_value=(1,))), RuntimeError("x")]
    assert pgb.keyword_index_backend(session, "chunk_embed") == "bm25s"
    assert session.begin_nested.call_count == 2


def _gate(method, strategy, backend, auto=True, index_state="ready"):
    with (
        patch.object(indexing, "db", MagicMock()),
        patch.object(indexing, "_get_kb_retrieval_method", return_value=method),
        patch.object(indexing, "_get_kb_indexing_strategy", return_value=strategy),
        patch.object(indexing, "get_setting", return_value=auto),
        patch.object(indexing.pg_bm25_index, "keyword_index_backend", return_value=backend),
        patch.object(indexing.pg_bm25_index, "_read_index_state", return_value=index_state),
    ):
        return indexing._should_build_bm25_now(KB)


def test_the_file_index_append_is_skipped_when_pg_search_serves_the_kb():
    assert _gate("hybrid", "chunk_embed", "pg_search") is False


@pytest.mark.parametrize("index_state", ["absent", "building"])
def test_a_kb_without_its_own_ready_index_keeps_its_file_index_maintained(index_state):
    """pg_search installed and the table partitioned says nothing about THIS KB.

    A knowledge base that existed before the extension keeps its rows in the
    DEFAULT partition, and search keeps reading its bm25s file index, until an
    operator builds its own index. Skipping the per-source add/remove before
    then would freeze that file index: new sources invisible, removed ones
    still scored.
    """
    assert _gate("hybrid", "chunk_embed", "pg_search", index_state=index_state) is True
    assert _gate("full_text", "graph_index", "pg_search", index_state=index_state) is True


def test_maintenance_does_not_stop_on_a_stale_cached_ready():
    """The search path caches readiness for its TTL; an index dropped since
    must not stop the file index being maintained for that long."""
    pgb._ready_cache[(KB, "chunks")] = (float("inf"), True)
    assert _gate("hybrid", "chunk_embed", "pg_search", index_state="absent") is True


def test_pg_search_serves_kb_reads_the_catalog_for_the_kbs_own_index():
    session = MagicMock()
    with (
        patch.object(pgb, "keyword_index_backend", return_value="pg_search"),
        patch.object(pgb, "_read_index_state", return_value="ready") as state,
    ):
        assert pgb.pg_search_serves_kb(session, KB, "full_document") is True
    assert state.call_args.args[1:] == (KB, "full_documents")
    with (
        patch.object(pgb, "keyword_index_backend", return_value="bm25s"),
        patch.object(pgb, "_read_index_state") as state,
    ):
        assert pgb.pg_search_serves_kb(session, KB, "chunk_embed") is False
    state.assert_not_called()
    with (
        patch.object(pgb, "keyword_index_backend", return_value="pg_search"),
        patch.object(pgb, "_read_index_state", side_effect=RuntimeError("gone")),
    ):
        assert pgb.pg_search_serves_kb(session, KB, "chunk_embed") is False


def test_the_file_index_append_still_runs_without_pg_search():
    assert _gate("hybrid", "chunk_embed", "bm25s") is True
    assert _gate("full_text", "graph_index", "bm25s") is True


def test_the_existing_gates_still_apply():
    assert _gate("vector_search", "chunk_embed", "bm25s") is False
    assert _gate("hybrid", "chunk_embed", "bm25s", auto=False) is False


def test_a_failed_backend_probe_keeps_the_file_index_append():
    """Never lose the only keyword index a KB might have because a probe failed."""
    with (
        patch.object(indexing, "_get_kb_retrieval_method", return_value="hybrid"),
        patch.object(indexing, "get_setting", return_value=True),
        patch.object(indexing, "_get_kb_indexing_strategy", side_effect=RuntimeError("db gone")),
    ):
        assert indexing._should_build_bm25_now(KB) is True
