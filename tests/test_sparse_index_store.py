"""Tests for SparseIndexStore — the on-disk sparse index location helper.

Pure-Python test: no DB required. Exercises path construction with both
str and UUID kb identifiers.
"""

import os
import uuid

import pytest

from agentic_project_service.services.sparse_retrieval.sparse_index_store import (
    SparseIndexStore,
)


@pytest.fixture(autouse=True)
def db_cleanup():
    """Override conftest.py's autouse db_cleanup for this file only.

    Tests here are pure-Python (no Flask app, no Postgres). Overriding by
    the same fixture name short-circuits the DB-dependent setup that
    conftest's autouse otherwise pulls in via the ``app`` fixture.
    """
    yield


@pytest.mark.parametrize(
    "kb_id_factory",
    [
        pytest.param(lambda: str(uuid.uuid4()), id="str"),
        pytest.param(lambda: uuid.uuid4(), id="UUID"),
    ],
)
def test_get_index_path_accepts_str_or_uuid(kb_id_factory, tmp_path):
    """Regression: SparseIndexStore must accept either ``str`` or ``uuid.UUID``.

    The API layer hands us strings; SQLAlchemy hands us ``uuid.UUID``
    instances. Before the fix, passing a UUID raised:

        TypeError: join() argument must be str, bytes, or os.PathLike
        object, not 'UUID'

    inside ``get_index_path()`` (``os.path.join``). This surfaced during the
    cleanup branch in ``tasks/indexing.py`` where SQLAlchemy-typed
    ``knowledge_base_id`` flowed directly through to the store.
    """
    kb_id = kb_id_factory()
    store = SparseIndexStore(knowledge_base_id=kb_id, base_path=str(tmp_path))

    # Stored as a str regardless of input type
    assert store.kb_id == str(kb_id)
    assert isinstance(store.kb_id, str)

    # All three known item tables must yield a clean string path
    for item_table in ("chunks", "full_documents", "graph_index_nodes"):
        path = store.get_index_path(item_table=item_table)
        assert isinstance(path, str)
        assert str(kb_id) in path
        assert item_table in path
        assert path.endswith(os.path.join(str(kb_id), item_table, "bm25"))


def test_index_exists_returns_false_for_missing_directory(tmp_path):
    """``index_exists`` must not raise on a UUID-constructed store when the
    index hasn't been written yet — this is the *exact* call site that
    blew up in production (sparse_index_store.py:75 -> 64)."""
    store = SparseIndexStore(knowledge_base_id=uuid.uuid4(), base_path=str(tmp_path))
    assert store.index_exists("graph_index_nodes") is False
    assert store.index_exists("chunks") is False
    assert store.index_exists("full_documents") is False
