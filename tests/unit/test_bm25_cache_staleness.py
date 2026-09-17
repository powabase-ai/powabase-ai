"""The in-process BM25 cache must notice a rebuild done by another process.

`project-api` and `project-worker` are separate containers sharing the index
volume. Indexing runs in the worker, searching in the API — so every rebuild
happens in the process that is *not* holding the cached index.

`get_or_load_manager` returned its cached manager without ever looking at the
files again, so the API kept serving whatever it had loaded first. The
failure that surfaced this was not a stale-but-usable index: after a document
took a knowledge base from 43 nodes to 626, the API's search raised

    IndexError: index 59 is out of bounds for axis 0 with size 43

from inside bm25s — a retrieved corpus position indexed into an id array from
the older, smaller build. Hybrid search did not degrade to vector-only; it
failed outright, with `bm25_status` reporting nothing because
BM25_AUTO_INDEXING is on and the platform is supposed to be managing it.

`metadata.json` is written last and atomically by every build path, which
makes it the barrier to stamp the cache against.
"""

from __future__ import annotations

import os
import shutil
import time

import pytest

from agentic_project_service.services.sparse_retrieval.bm25_index import (
    BM25IndexManager,
)
from agentic_project_service.services.sparse_retrieval.sparse_index_store import (
    SparseIndexStore,
)

DOCS_V1 = ["alpha beta gamma", "delta epsilon zeta"]
IDS_V1 = ["a", "b"]
DOCS_V2 = DOCS_V1 + ["eta theta iota", "kappa lambda mu", "nu xi omicron"]
IDS_V2 = IDS_V1 + ["c", "d", "e"]


@pytest.fixture(autouse=True)
def _clear_cache():
    SparseIndexStore.clear_cache()
    yield
    SparseIndexStore.clear_cache()


def _rebuild_as_other_process(base_path, kb_id, documents, item_ids):
    """Write the index the way the worker does — on disk only.

    Deliberately *not* `rebuild_from_scratch`: that drops the cache entry, and
    `_managers` is class-level, so calling it here would clear the very cache
    under test. In the real deployment the worker's drop applies to the
    worker's own process and the API never hears about it, so the honest
    simulation is files changing with this process's cache left alone.
    """
    store = SparseIndexStore(knowledge_base_id=kb_id, base_path=str(base_path))
    manager = BM25IndexManager()
    manager.build_index(documents=documents, item_ids=item_ids)
    manager.save(store.get_index_path("chunks"))
    store.write_metadata(item_table="chunks", item_count=len(item_ids))


def test_a_rebuild_by_another_process_is_picked_up(tmp_path):
    kb_id = "kb-stale-1"
    _rebuild_as_other_process(tmp_path, kb_id, DOCS_V1, IDS_V1)

    reader = SparseIndexStore(knowledge_base_id=kb_id, base_path=str(tmp_path))
    first = reader.get_or_load_manager("chunks")
    assert len(first._item_ids) == len(IDS_V1)

    # The worker rebuilds. Nothing tells the reader.
    time.sleep(0.01)  # keep the mtime distinguishable on coarse filesystems
    _rebuild_as_other_process(tmp_path, kb_id, DOCS_V2, IDS_V2)

    second = reader.get_or_load_manager("chunks")

    assert len(second._item_ids) == len(IDS_V2), (
        "the reader kept serving the index it loaded first"
    )


def test_an_unchanged_index_is_not_reloaded(tmp_path):
    """The staleness check must not turn every search into a disk load."""
    kb_id = "kb-stale-2"
    _rebuild_as_other_process(tmp_path, kb_id, DOCS_V1, IDS_V1)

    reader = SparseIndexStore(knowledge_base_id=kb_id, base_path=str(tmp_path))
    first = reader.get_or_load_manager("chunks")
    second = reader.get_or_load_manager("chunks")

    assert second is first, "the manager was reloaded despite the index not changing"


def test_a_missing_sidecar_still_detects_a_change(tmp_path):
    """Indexes built before the sidecar existed have no metadata.json. They
    must not be treated as permanently fresh — the id mapping is enough of a
    signal, and it is what the IndexError came from."""
    kb_id = "kb-stale-3"
    _rebuild_as_other_process(tmp_path, kb_id, DOCS_V1, IDS_V1)
    index_path = reader_path = SparseIndexStore(
        knowledge_base_id=kb_id, base_path=str(tmp_path)
    ).get_index_path("chunks")
    os.remove(os.path.join(index_path, "metadata.json"))

    reader = SparseIndexStore(knowledge_base_id=kb_id, base_path=str(tmp_path))
    assert len(reader.get_or_load_manager("chunks")._item_ids) == len(IDS_V1)

    time.sleep(0.01)
    _rebuild_as_other_process(tmp_path, kb_id, DOCS_V2, IDS_V2)
    os.remove(os.path.join(reader_path, "metadata.json"))

    assert len(reader.get_or_load_manager("chunks")._item_ids) == len(IDS_V2)


def test_an_index_deleted_underneath_is_not_kept(tmp_path):
    """`delete_index` clears the cache in its own process; another one that
    had it loaded would otherwise search a corpus that no longer exists."""
    kb_id = "kb-stale-4"
    _rebuild_as_other_process(tmp_path, kb_id, DOCS_V1, IDS_V1)

    reader = SparseIndexStore(knowledge_base_id=kb_id, base_path=str(tmp_path))
    assert len(reader.get_or_load_manager("chunks")._item_ids) == len(IDS_V1)

    shutil.rmtree(reader.get_index_path("chunks"))

    assert reader.get_or_load_manager("chunks").is_empty(), (
        "a deleted index was still being served from cache"
    )
