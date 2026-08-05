"""update_source_status persists error_code; serializers expose it."""

from unittest.mock import MagicMock, patch

from agentic_project_service.tasks.extraction import update_source_status


def test_update_source_status_writes_error_code():
    with patch("agentic_project_service.tasks.extraction.db") as mock_db:
        mock_db.session = MagicMock()
        update_source_status(
            "src-1",
            "failed",
            "429 Too Many Requests",
            "task-1",
            error_code="rate_limited",
        )
        _sql, params = mock_db.session.execute.call_args[0]
        assert params["error_code"] == "rate_limited"
        assert params["status"] == "failed"
        assert "error_code" in str(_sql)


def test_update_source_status_error_code_defaults_to_none():
    with patch("agentic_project_service.tasks.extraction.db") as mock_db:
        mock_db.session = MagicMock()
        update_source_status("src-1", "extracted")
        _sql, params = mock_db.session.execute.call_args[0]
        assert params["error_code"] is None


def test_source_serializers_include_error_code():
    import inspect

    from agentic_project_service.routes import sources as sources_mod

    src = inspect.getsource(sources_mod)
    # Both raw-SQL serializers must select and emit the column.
    assert src.count('"error_code"') >= 2, "list_sources and get_source must both emit error_code"


def test_reextract_route_clears_error_code():
    """A re-extracted source must not keep serving a stale failure class
    (e.g. error_code='rate_limited') while its status is reset to pending."""
    import inspect

    from agentic_project_service.routes.sources import reextract_source

    src = inspect.getsource(reextract_source)
    assert "error_message = NULL" in src, "sanity check: reset UPDATE not found as expected"
    assert "error_code = NULL" in src, (
        "reextract's UPDATE resets error_message but not error_code — "
        "a re-extracted source keeps serving its stale failure class while pending"
    )
