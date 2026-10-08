"""format_items_as_context labels items with caller-assigned keys when given."""

import pytest
from agentic.knowledge.models import RetrievedItem

from agentic_project_service.services.knowledge_search import format_items_as_context

KB = "kb-1"


def _item(item_id: str, text: str, source_id: str | None, pages=None) -> RetrievedItem:
    meta = {"pages": pages} if pages else {}
    return RetrievedItem(
        item_id=item_id, text=text, score=0.9, source_id=source_id, knowledge_base_id=KB, meta=meta
    )


def test_default_labels_are_unchanged():
    content, _ = format_items_as_context(
        [_item("a", "alpha", "s1"), _item("b", "beta", "s2")], group_by_document=False
    )
    assert content == "[1] (Source: s1)\nalpha\n\n[2] (Source: s2)\nbeta"


def test_flat_text_uses_labels():
    content, _ = format_items_as_context(
        [_item("a", "alpha", "s1"), _item("b", "beta", "s2")],
        group_by_document=False,
        labels=[7, 9],
    )
    assert content == "[7] (Source: s1)\nalpha\n\n[9] (Source: s2)\nbeta"


def test_grouped_text_uses_labels():
    content, _ = format_items_as_context(
        [_item("a", "alpha", "s1"), _item("b", "beta", "s1")], labels=[7, 9]
    )
    assert "  [7]\n  alpha" in content
    assert "  [9]\n  beta" in content
    assert "[1]" not in content and "[2]" not in content


def test_already_shown_annotation_uses_labels():
    content, _ = format_items_as_context(
        [_item("a", "alpha", "s1", pages=[1]), _item("b", "alpha again", "s1", pages=[1])],
        labels=[7, 9],
    )
    assert "  [9] [Page: 1] (content already shown above)" in content


@pytest.mark.parametrize("grouped", [True, False])
def test_image_mode_uses_labels(grouped):
    source_image_map = {"s1": [{"page": 1, "content": "b64-1", "format": "png"}]}
    blocks, _ = format_items_as_context(
        [_item("a", "alpha", "s1", pages=[1])],
        per_kb_context_mode={KB: "image"},
        source_image_map=source_image_map,
        group_by_document=grouped,
        labels=[7],
    )
    texts = [b["text"] for b in blocks if b.get("type") == "text"]
    assert any(t.startswith("  [7]") or t.startswith("[7]") for t in texts), texts
    assert not any("[1]" in t for t in texts), texts


def test_labels_must_cover_every_item():
    with pytest.raises(ValueError):
        format_items_as_context([_item("a", "alpha", "s1")], labels=[1, 2])
