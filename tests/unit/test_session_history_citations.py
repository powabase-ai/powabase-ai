"""Unit tests for citation marker stripping in session history."""


class TestStripCitationMarkers:
    def test_strips_basic_markers(self):
        from agentic_project_service.services.session import strip_citation_markers

        text = "According to the law [1], the court held [3] that..."
        assert strip_citation_markers(text) == "According to the law, the court held that..."

    def test_strips_adjacent_markers(self):
        from agentic_project_service.services.session import strip_citation_markers

        text = "Multiple sources [1][2][3] confirm this."
        assert strip_citation_markers(text) == "Multiple sources confirm this."

    def test_no_markers(self):
        from agentic_project_service.services.session import strip_citation_markers

        text = "No citations here."
        assert strip_citation_markers(text) == "No citations here."

    def test_preserves_non_citation_brackets(self):
        from agentic_project_service.services.session import strip_citation_markers

        text = "Array [index] and citation [1] mixed."
        result = strip_citation_markers(text)
        assert "[index]" in result
        assert "[1]" not in result

    def test_empty_string(self):
        from agentic_project_service.services.session import strip_citation_markers

        assert strip_citation_markers("") == ""
