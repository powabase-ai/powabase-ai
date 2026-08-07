"""Retrieval errors from knowledge_search must not be silently discarded.

``_make_search_handler`` wraps ``create_and_execute()``, whose result carries a
``errors`` list (populated by ``context_handler._search_single_kb`` on a failed
per-KB search: ``{"type": "kb_retrieval_error", "knowledge_base_id", "message",
"timestamp"}``). Before this fix, only ``formatted_context`` was read and
``errors`` was discarded — a failing search silently returned an empty (or
partial) context, and the LLM answered from priors with nothing signaling the
failure. ``context_truncation`` entries are a separate, benign error type
(a budget event, not a failure) and must NOT trigger the same note.
"""

from unittest.mock import MagicMock, patch

from agentic_project_service.services import tool_registry


def _handler(result):
    shared_session = MagicMock(name="shared_db_session")
    with (
        patch.object(tool_registry, "_get_flask_app", return_value=None),
        patch.object(tool_registry, "Session"),
        patch.object(
            tool_registry,
            "create_and_execute",
            return_value=("handler-1", result),
        ),
    ):
        handler = tool_registry._make_search_handler(shared_session)
        return handler(query="q", kb_configs=[{"id": "kb"}], max_tokens=100, session_history=None)


def test_errors_with_empty_context_surface_kb_id_and_message():
    content, _metadata = _handler(
        {
            "formatted_context": "",
            "errors": [
                {
                    "type": "kb_retrieval_error",
                    "knowledge_base_id": "kb-1",
                    "message": "connection timed out",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            ],
        }
    )
    assert isinstance(content, str)
    assert "kb-1" in content
    assert "connection timed out" in content


def test_errors_with_non_empty_context_keep_context_and_append_note():
    content, _metadata = _handler(
        {
            "formatted_context": "some retrieved context",
            "errors": [
                {
                    "type": "kb_retrieval_error",
                    "knowledge_base_id": "kb-2",
                    "message": "pgvector query failed",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            ],
        }
    )
    assert "some retrieved context" in content
    assert "kb-2" in content
    assert "pgvector query failed" in content


def test_context_truncation_only_errors_do_not_trigger_note():
    content, _metadata = _handler(
        {
            "formatted_context": "ctx text",
            "errors": [
                {
                    "type": "context_truncation",
                    "items_dropped": 3,
                    "dropped_item_ids": ["a", "b", "c"],
                    "dropped_items_by_kb": {},
                    "reason": "exceeded max_context_tokens (1000)",
                    "token_limit": 1000,
                    "estimated_tokens_at_drop": 1200,
                }
            ],
        }
    )
    assert content == "ctx text"


def test_raw_exception_text_is_not_pasted_unbounded_into_llm_context():
    """N1: the raw ``message`` (str(exception)) can carry a full SQL
    statement + bound params for a SQLAlchemy/psycopg failure. Only the
    first line, truncated to 160 chars, plus the error_type, may reach the
    LLM — everything after the first newline must be absent from the note,
    however sensitive it is."""
    first_line = "connection failed: " + ("a" * 140) + "UNIQUE_TAIL_MARKER_BEYOND_160"
    sql_leak = "SELECT * FROM secret_table WHERE token = 'abc123'"
    message = f"{first_line}\n{sql_leak}\nMORE INTERNAL DETAIL"
    content, _metadata = _handler(
        {
            "formatted_context": "",
            "errors": [
                {
                    "type": "kb_retrieval_error",
                    "knowledge_base_id": "kb-1",
                    "error_type": "OperationalError",
                    "message": message,
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            ],
        }
    )
    assert sql_leak not in content
    assert "MORE INTERNAL DETAIL" not in content
    assert "OperationalError" in content
    assert "kb-1" in content
    # Only the first 160 chars of the first line may appear.
    assert first_line[:160] in content
    assert "UNIQUE_TAIL_MARKER_BEYOND_160" not in content


def test_error_record_missing_error_type_still_produces_sane_note():
    """A record without ``error_type`` (e.g. one persisted before this fix
    landed) must still produce a bounded note, falling back to just the
    truncated first line."""
    content, _metadata = _handler(
        {
            "formatted_context": "",
            "errors": [
                {
                    "type": "kb_retrieval_error",
                    "knowledge_base_id": "kb-2",
                    "message": "boom",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            ],
        }
    )
    assert "kb-2" in content
    assert "boom" in content


def test_multimodal_errors_append_trailing_text_block():
    content, _metadata = _handler(
        {
            "formatted_context": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
            ],
            "errors": [
                {
                    "type": "kb_retrieval_error",
                    "knowledge_base_id": "kb-3",
                    "message": "embedding service unavailable",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            ],
        }
    )
    assert isinstance(content, list)
    last = content[-1]
    assert last["type"] == "text"
    assert "kb-3" in last["text"]
    assert "embedding service unavailable" in last["text"]
