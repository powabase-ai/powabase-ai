"""The document outline must never widen an image fetch.

An item with no page metadata makes both image paths fall back to "attach
every image of this source" — deliberate for unforeseen item shapes, and
wrong for the outline, which has no pages *by construction* and always names
a document whose other items already carry precise page coverage. One
outline in the pool therefore discards the page filter for every other item
from the same document.

Three call sites guard against that. Two were deletable with the whole suite
green — the image-attach branches in ``format_items_as_context``'s grouped
and ungrouped paths — and are covered here. The third, the page scan in
``_resolve_page_images``, was already pinned by
``test_page_resolution.py::TestResolvePageImagesFiltering::
test_structural_outline_does_not_widen_coverage_for_its_document``, which
fails when it is removed; a claim that all three deleted free was wrong.
"""

from __future__ import annotations

from agentic.knowledge.models import RetrievedItem

from agentic_project_service.services import knowledge_search

KB_ID = "kb-1"
SOURCE_ID = "src-A"


def _node_item(pages: list[int]) -> RetrievedItem:
    return RetrievedItem(
        item_id="node-1",
        text="body of 0002",
        score=0.5,
        source_id=SOURCE_ID,
        knowledge_base_id=KB_ID,
        meta={
            "node_id": "0002",
            "toc_id": "toc-a",
            "retrieval_method": "graph_expansion",
            "pages": pages,
            "start_page": pages[0],
            "end_page": pages[-1],
        },
    )


def _outline_item() -> RetrievedItem:
    return RetrievedItem(
        item_id="toc-a",
        text="[0001] Definitions",
        score=0.4,
        source_id=SOURCE_ID,
        knowledge_base_id=KB_ID,
        meta={
            "toc_id": "toc-a",
            "retrieval_method": "graph_toc",
            "score_type": "graph_doc_outline",
            "pages": [],
        },
    )


def _image_blocks(items, grouped: bool):
    source_image_map = {
        SOURCE_ID: [{"page": p, "content": f"b64-{p}", "format": "png"} for p in (1, 2, 3)]
    }
    content, _meta = knowledge_search.format_items_as_context(
        items,
        source_image_map=source_image_map,
        per_kb_context_mode={KB_ID: "image"},
        group_by_document=grouped,
    )
    assert isinstance(content, list), "image mode returns multimodal blocks"
    return [b for b in content if b.get("type") == "image_url"]


def test_the_grouped_image_path_attaches_no_images_for_an_outline():
    images = _image_blocks([_node_item([1]), _outline_item()], grouped=True)

    assert len(images) == 1, "the outline must not add pages 2 and 3"


def test_the_ungrouped_image_path_attaches_no_images_for_an_outline():
    images = _image_blocks([_node_item([1]), _outline_item()], grouped=False)

    assert len(images) == 1, "the outline must not add pages 2 and 3"
