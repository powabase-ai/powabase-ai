"""Correctness tests for the page-filtering logic introduced in #107.

The fix narrows ``_resolve_page_images`` from "fetch every image for any
matched source" to "fetch only the pages the matched items actually
reference, with a conservative fallback if any item lacks page metadata."

These tests verify behaviour, not memory. Memory savings are a downstream
consequence of fewer fetches; correctness is verified by checking that the
right pages get selected for each item-shape (chunk-embed vs graph-index)
and that the fallback fires when item metadata is missing.
"""

from __future__ import annotations


import pytest
from agentic.knowledge.models import RetrievedItem


# ---------------------------------------------------------------------------
# _pages_for_item — direct unit tests
# ---------------------------------------------------------------------------


def _make_item(meta=None, source_id="src-A", knowledge_base_id=None):
    return RetrievedItem(
        item_id="i",
        text="x",
        score=0.0,
        source_id=source_id,
        knowledge_base_id=knowledge_base_id,
        meta=meta,
    )


class TestPagesForItem:
    """``_pages_for_item`` should support both chunk-embed and graph-index
    metadata shapes. Reading only ``meta.pages`` historically silently
    excluded graph-index nodes, which fell through to the "all images"
    branch — the bug this PR fixes."""

    def test_chunk_embed_meta_pages_list(self):
        from agentic_project_service.services.knowledge_search import _pages_for_item

        item = _make_item(meta={"pages": [3, 4, 5]})
        assert _pages_for_item(item) == {3, 4, 5}

    def test_chunk_embed_single_page(self):
        from agentic_project_service.services.knowledge_search import _pages_for_item

        item = _make_item(meta={"pages": [7]})
        assert _pages_for_item(item) == {7}

    def test_graph_index_start_end_inclusive_range(self):
        from agentic_project_service.services.knowledge_search import _pages_for_item

        # graph_index_nodes use start_page/end_page, NOT meta.pages.
        # Range is inclusive on both ends.
        item = _make_item(meta={"start_page": 5, "end_page": 10})
        assert _pages_for_item(item) == {5, 6, 7, 8, 9, 10}

    def test_graph_index_only_start_page(self):
        from agentic_project_service.services.knowledge_search import _pages_for_item

        # When only start_page is set, treat as a single-page item.
        item = _make_item(meta={"start_page": 12})
        assert _pages_for_item(item) == {12}

    def test_chunk_embed_takes_precedence_over_graph_index(self):
        from agentic_project_service.services.knowledge_search import _pages_for_item

        # If both shapes are present (shouldn't happen in practice but defensive),
        # the explicit `pages` list wins. start_page/end_page only fires as fallback.
        item = _make_item(meta={"pages": [1, 2], "start_page": 99, "end_page": 100})
        assert _pages_for_item(item) == {1, 2}

    def test_no_page_metadata_returns_empty(self):
        from agentic_project_service.services.knowledge_search import _pages_for_item

        # Empty signals callers to fall back to "all images" for safety.
        item = _make_item(meta={"some_other_key": "value"})
        assert _pages_for_item(item) == set()

    def test_none_meta_returns_empty(self):
        from agentic_project_service.services.knowledge_search import _pages_for_item

        item = _make_item(meta=None)
        assert _pages_for_item(item) == set()

    def test_empty_pages_list_returns_empty(self):
        from agentic_project_service.services.knowledge_search import _pages_for_item

        item = _make_item(meta={"pages": []})
        assert _pages_for_item(item) == set()

    def test_string_page_numbers_coerced_to_int(self):
        from agentic_project_service.services.knowledge_search import _pages_for_item

        # Defensive: jsonb may round-trip integers as strings depending on
        # the writer.
        item = _make_item(meta={"pages": ["3", 4, "5"]})
        assert _pages_for_item(item) == {3, 4, 5}

    def test_unparseable_pages_skipped(self):
        from agentic_project_service.services.knowledge_search import _pages_for_item

        item = _make_item(meta={"pages": [1, "garbage", 3]})
        assert _pages_for_item(item) == {1, 3}


# ---------------------------------------------------------------------------
# _resolve_page_images filtering — integration with the page set logic
# ---------------------------------------------------------------------------


class _FakeRow:
    """Stub row returned by ``db_session.execute(...).fetchall()``."""

    def __init__(self, sid, derivatives):
        self._fields = (sid, derivatives)

    def __getitem__(self, idx):
        return self._fields[idx]


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeDbSession:
    def __init__(self, sources):
        # sources: list of (source_id, derivatives_dict)
        self._sources = sources

    def execute(self, *args, **kwargs):
        return _FakeResult([_FakeRow(sid, derivs) for sid, derivs in self._sources])


def _image_record(source_id, page):
    return {
        "page": page,
        "format": "png",
        "storage_path": f"sources/{source_id}/derivatives/image/page_{page}.png",
    }


def _build_source(source_id, page_count):
    """Construct a source-derivatives dict with image records 1..page_count."""
    return {"image": [_image_record(source_id, p) for p in range(1, page_count + 1)]}


@pytest.fixture
def fake_storage(monkeypatch):
    """Replace storage with a stub that records which paths were fetched."""

    class _FakeStorage:
        def __init__(self):
            self.fetched: list[str] = []

        def download_from_path(self, path):
            self.fetched.append(path)
            # Return small payload so base64 encoding succeeds quickly.
            return b"fake-image-bytes-for-" + path.encode()

        def has_public_url(self):
            return False

        def create_signed_url(self, *args, **kwargs):  # pragma: no cover
            return "http://signed/" + str(args)

    storage = _FakeStorage()

    def _get_storage():
        return storage

    monkeypatch.setattr(
        "agentic_project_service.services.context_handler.get_storage",
        _get_storage,
    )
    return storage


class TestResolvePageImagesFiltering:
    """The fix's core property: only fetch pages the items actually reference,
    falling back to all pages when any item lacks page metadata."""

    def test_chunk_embed_items_filter_to_referenced_pages(self, fake_storage):
        from agentic_project_service.services.context_handler import _resolve_page_images

        SOURCE = "src-A"
        # Source has 100 pages but items only reference pages 3, 4, 7
        db = _FakeDbSession([(SOURCE, _build_source(SOURCE, 100))])
        items = [
            _make_item(meta={"pages": [3, 4]}, source_id=SOURCE),
            _make_item(meta={"pages": [7]}, source_id=SOURCE),
        ]

        result = _resolve_page_images(db, items)

        fetched_pages = sorted(int(p.split("_")[-1].rstrip(".png")) for p in fake_storage.fetched)
        assert fetched_pages == [3, 4, 7], f"expected only pages 3,4,7 fetched; got {fetched_pages}"
        assert SOURCE in result
        assert {r["page"] for r in result[SOURCE]} == {3, 4, 7}

    def test_graph_index_items_filter_to_inclusive_range(self, fake_storage):
        """Bug 2 regression test: graph-index items use start_page/end_page,
        not meta.pages. Pre-fix, these silently fell through to "fetch all"."""
        from agentic_project_service.services.context_handler import _resolve_page_images

        SOURCE = "src-G"
        db = _FakeDbSession([(SOURCE, _build_source(SOURCE, 100))])
        items = [
            _make_item(meta={"start_page": 5, "end_page": 8}, source_id=SOURCE),
        ]

        result = _resolve_page_images(db, items)

        fetched_pages = sorted(int(p.split("_")[-1].rstrip(".png")) for p in fake_storage.fetched)
        assert fetched_pages == [5, 6, 7, 8]
        assert {r["page"] for r in result[SOURCE]} == {5, 6, 7, 8}

    def test_mixed_chunk_and_graph_items_take_union(self, fake_storage):
        from agentic_project_service.services.context_handler import _resolve_page_images

        SOURCE = "src-M"
        db = _FakeDbSession([(SOURCE, _build_source(SOURCE, 50))])
        items = [
            _make_item(meta={"pages": [1, 2]}, source_id=SOURCE),
            _make_item(meta={"start_page": 5, "end_page": 7}, source_id=SOURCE),
        ]

        _resolve_page_images(db, items)

        fetched_pages = sorted(int(p.split("_")[-1].rstrip(".png")) for p in fake_storage.fetched)
        assert fetched_pages == [1, 2, 5, 6, 7]

    def test_item_with_no_page_metadata_falls_back_to_all_for_that_source(self, fake_storage):
        """If we don't know what pages an item covers, the safe choice is to
        fetch every image for that source — preserves prior behavior on
        unforeseen item shapes."""
        from agentic_project_service.services.context_handler import _resolve_page_images

        SOURCE = "src-F"
        db = _FakeDbSession([(SOURCE, _build_source(SOURCE, 5))])
        items = [
            _make_item(meta={"pages": [1]}, source_id=SOURCE),
            # This item has no page info → triggers fallback for src-F
            _make_item(meta={"some_other_key": "x"}, source_id=SOURCE),
        ]

        _resolve_page_images(db, items)

        fetched_pages = sorted(int(p.split("_")[-1].rstrip(".png")) for p in fake_storage.fetched)
        assert fetched_pages == [
            1,
            2,
            3,
            4,
            5,
        ], "fallback should fetch all source pages when any item lacks metadata"

    def test_filtering_is_per_source(self, fake_storage):
        """Per-source independence: an item lacking metadata for source A
        shouldn't expand the page set fetched for source B."""
        from agentic_project_service.services.context_handler import _resolve_page_images

        db = _FakeDbSession(
            [
                ("src-A", _build_source("src-A", 10)),
                ("src-B", _build_source("src-B", 10)),
            ]
        )
        items = [
            # src-A has one item with no page info → fallback for A only
            _make_item(meta={}, source_id="src-A"),
            # src-B has explicit pages → stays filtered
            _make_item(meta={"pages": [2, 4]}, source_id="src-B"),
        ]

        _resolve_page_images(db, items)

        fetched_a = sorted(
            int(p.split("_")[-1].rstrip(".png")) for p in fake_storage.fetched if "src-A" in p
        )
        fetched_b = sorted(
            int(p.split("_")[-1].rstrip(".png")) for p in fake_storage.fetched if "src-B" in p
        )
        assert fetched_a == list(range(1, 11)), "src-A should fall back to all"
        assert fetched_b == [2, 4], "src-B should stay filtered"

    def test_empty_items_returns_empty_dict_no_fetches(self, fake_storage):
        from agentic_project_service.services.context_handler import _resolve_page_images

        db = _FakeDbSession([])
        result = _resolve_page_images(db, [])
        assert result == {}
        assert fake_storage.fetched == []

    def test_pages_outside_source_range_simply_yield_nothing(self, fake_storage):
        """If items reference pages that don't exist in the source's
        derivatives (e.g. a stale meta or off-by-one), the filter just
        yields nothing — no error, no over-fetch."""
        from agentic_project_service.services.context_handler import _resolve_page_images

        SOURCE = "src-X"
        # Source has only pages 1..3
        db = _FakeDbSession([(SOURCE, _build_source(SOURCE, 3))])
        items = [_make_item(meta={"pages": [99, 100]}, source_id=SOURCE)]

        result = _resolve_page_images(db, items)

        assert result == {}
        assert fake_storage.fetched == []
