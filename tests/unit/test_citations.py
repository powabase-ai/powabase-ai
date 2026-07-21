"""Unit tests for citation labeling, parsing, and persistence."""

import pytest


class TestBuildCitationMap:
    """Test build_citation_map: takes retrieved_context items, returns citation_map."""

    def test_basic_labeling(self):
        from agentic_project_service.services.citations import build_citation_map

        items = [
            {
                "id": "chunk-1",
                "source_id": "src-1",
                "source_name": "doc.pdf",
                "text": "First chunk text",
                "meta": {"pages": [1]},
            },
            {
                "id": "chunk-2",
                "source_id": "src-1",
                "source_name": "doc.pdf",
                "text": "Second chunk text",
                "meta": {"pages": [2]},
            },
        ]
        citation_map = build_citation_map(items)

        assert citation_map == {
            "1": {
                "key": "1",
                "item_id": "chunk-1",
                "source_id": "src-1",
                "source_name": "doc.pdf",
                "text_excerpt": "First chunk text",
                "meta": {"pages": [1]},
            },
            "2": {
                "key": "2",
                "item_id": "chunk-2",
                "source_id": "src-1",
                "source_name": "doc.pdf",
                "text_excerpt": "Second chunk text",
                "meta": {"pages": [2]},
            },
        }

    def test_skips_diagnostics_items(self):
        from agentic_project_service.services.citations import build_citation_map

        items = [
            {"_type": "retrieval_diagnostics", "total_items": 1},
            {
                "id": "chunk-1",
                "source_id": "src-1",
                "source_name": "doc.pdf",
                "text": "Chunk text",
                "meta": {},
            },
        ]
        citation_map = build_citation_map(items)
        assert len(citation_map) == 1
        assert "1" in citation_map

    def test_by_value_items_have_null_item_id(self):
        from agentic_project_service.services.citations import build_citation_map

        items = [
            {"text": "External content", "meta": {"title": "Custom"}},
        ]
        citation_map = build_citation_map(items)
        assert citation_map["1"]["item_id"] is None
        assert citation_map["1"]["source_id"] is None
        assert citation_map["1"]["text_excerpt"] == "External content"
        assert citation_map["1"]["meta"] == {"title": "Custom"}

    def test_truncates_text_excerpt(self):
        from agentic_project_service.services.citations import build_citation_map

        items = [
            {
                "id": "chunk-1",
                "source_id": "src-1",
                "source_name": "doc.pdf",
                "text": "x" * 500,
                "meta": {},
            },
        ]
        citation_map = build_citation_map(items)
        assert len(citation_map["1"]["text_excerpt"]) <= 300

    def test_empty_items(self):
        from agentic_project_service.services.citations import build_citation_map

        assert build_citation_map([]) == {}

    def test_preserves_order(self):
        from agentic_project_service.services.citations import build_citation_map

        items = [
            {
                "id": f"chunk-{i}",
                "source_id": "src-1",
                "source_name": "doc.pdf",
                "text": f"Chunk {i}",
                "meta": {},
            }
            for i in range(5)
        ]
        citation_map = build_citation_map(items)
        for i in range(5):
            assert citation_map[str(i + 1)]["item_id"] == f"chunk-{i}"


class TestParseCitationsFromResponse:
    """Test parse_citations_from_response: extract used keys, strip invalid, return citations."""

    def test_basic_parsing(self):
        from agentic_project_service.services.citations import parse_citations_from_response

        citation_map = {
            "1": {
                "key": "1",
                "item_id": "a",
                "source_id": "s",
                "source_name": "d.pdf",
                "text_excerpt": "...",
                "meta": {},
            },
            "2": {
                "key": "2",
                "item_id": "b",
                "source_id": "s",
                "source_name": "d.pdf",
                "text_excerpt": "...",
                "meta": {},
            },
            "3": {
                "key": "3",
                "item_id": "c",
                "source_id": "s",
                "source_name": "d.pdf",
                "text_excerpt": "...",
                "meta": {},
            },
        }
        content = "According to the law [1], the court held [3] that..."
        cleaned, citations = parse_citations_from_response(content, citation_map)

        assert cleaned == content  # no invalid markers to strip
        assert len(citations) == 2
        assert citations[0]["key"] == "1"
        assert citations[1]["key"] == "3"

    def test_strips_invalid_citations(self):
        from agentic_project_service.services.citations import parse_citations_from_response

        citation_map = {
            "1": {
                "key": "1",
                "item_id": "a",
                "source_id": "s",
                "source_name": "d.pdf",
                "text_excerpt": "...",
                "meta": {},
            },
        }
        content = "Valid [1] and hallucinated [5] and [99]."
        cleaned, citations = parse_citations_from_response(content, citation_map)

        assert "[5]" not in cleaned
        assert "[99]" not in cleaned
        assert "[1]" in cleaned
        assert len(citations) == 1

    def test_no_citations_in_response(self):
        from agentic_project_service.services.citations import parse_citations_from_response

        citation_map = {
            "1": {
                "key": "1",
                "item_id": "a",
                "source_id": "s",
                "source_name": "d.pdf",
                "text_excerpt": "...",
                "meta": {},
            },
        }
        content = "No citations here."
        cleaned, citations = parse_citations_from_response(content, citation_map)

        assert cleaned == "No citations here."
        assert citations == []

    def test_empty_citation_map(self):
        from agentic_project_service.services.citations import parse_citations_from_response

        content = "Some text with [1] that should be stripped."
        cleaned, citations = parse_citations_from_response(content, {})

        assert "[1]" not in cleaned
        assert citations == []

    def test_duplicate_citations_deduplicated(self):
        from agentic_project_service.services.citations import parse_citations_from_response

        citation_map = {
            "1": {
                "key": "1",
                "item_id": "a",
                "source_id": "s",
                "source_name": "d.pdf",
                "text_excerpt": "...",
                "meta": {},
            },
        }
        content = "First [1] and again [1]."
        cleaned, citations = parse_citations_from_response(content, citation_map)

        assert len(citations) == 1
        # Both markers remain in text
        assert cleaned.count("[1]") == 2


class TestBuildCitationInstruction:
    """Test the citation instruction appended to system prompts."""

    def test_returns_instruction_string(self):
        from agentic_project_service.services.citations import build_citation_instruction

        instruction = build_citation_instruction()
        assert "[1]" in instruction
        assert "[1][2]" in instruction
        assert "[1, 2]" in instruction

    def test_instruction_is_nonempty(self):
        from agentic_project_service.services.citations import build_citation_instruction

        assert len(build_citation_instruction()) > 0
