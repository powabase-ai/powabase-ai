"""Test that ``get_chat_messages`` forwards ``reasoning_requested`` and
``reasoning`` from the per-run ``output_messages`` JSONB to the API response.

Without this projection, the FE never sees ``reasoning_requested`` on
historical (refetched) messages and ``derivePillState`` returns null —
the pill never renders for completed runs."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from agentic_project_service.services.session import get_chat_messages


def _make_db_session(rows: list[tuple], session_uuid: str = "uuid-123") -> MagicMock:
    """Build a mock db_session that returns the configured run rows.

    The function performs two queries:
      1. SELECT id FROM agent_sessions WHERE session_id = :session_id
      2. SELECT id, run_id, input_messages, output_messages, content,
                created_at, started_at, completed_at
         FROM agent_runs WHERE session_id = :session_id AND status = 'completed'
    Plus a citation fetch per run; we make that return [].
    """
    session_lookup = MagicMock()
    session_lookup.fetchone.return_value = (session_uuid,)

    runs_result = MagicMock()
    runs_result.__iter__ = lambda self: iter(rows)

    citations_result = MagicMock()
    citations_result.__iter__ = lambda self: iter([])

    db = MagicMock()
    # First call returns session lookup, second call returns runs, subsequent
    # calls (citation fetches) return empty.
    db.execute.side_effect = [session_lookup, runs_result, citations_result]
    return db


def _user_msg(content: str = "hi") -> dict:
    return {"role": "user", "content": content}


def _make_row(
    run_uuid: str,
    run_id: str,
    output_msgs: list[dict],
    content: str | None,
    created_at: datetime | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> tuple:
    """A row matching the SELECT columns: id, run_id, input_messages,
    output_messages, content, created_at, started_at, completed_at."""
    return (
        run_uuid,
        run_id,
        [_user_msg("question")],
        output_msgs,
        content,
        created_at or datetime(2026, 1, 1, tzinfo=UTC),
        started_at,
        completed_at,
    )


def test_forwards_reasoning_requested_when_content_field_populated() -> None:
    """The 'fast path' (top-level run.content non-empty) must also forward
    reasoning_requested + reasoning from output_messages."""
    output_msgs = [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_requested": True,
            "reasoning": {
                "thinking_blocks": [{"type": "thinking", "thinking": "reasoning"}],
                "summary_text": "summary",
            },
        }
    ]
    rows = [_make_row("uuid-r1", "run-1", output_msgs, content="answer")]
    db = _make_db_session(rows)

    result = get_chat_messages(db, "session-abc")

    # user + assistant
    assistant = next(m for m in result if m["role"] == "assistant")
    assert assistant["reasoning_requested"] is True
    assert assistant["reasoning"]["summary_text"] == "summary"
    assert assistant["reasoning"]["thinking_blocks"][0]["thinking"] == "reasoning"


def test_forwards_reasoning_requested_when_content_field_empty() -> None:
    """The fallback path (top-level run.content empty, fall through to
    output_messages) must also forward reasoning replay fields."""
    output_msgs = [
        {
            "role": "assistant",
            "content": "fallback",
            "reasoning_requested": True,
            "reasoning": {"thinking_blocks": [], "summary_text": "fallback summary"},
        }
    ]
    rows = [_make_row("uuid-r1", "run-1", output_msgs, content=None)]
    db = _make_db_session(rows)

    result = get_chat_messages(db, "session-abc")

    assistant = next(m for m in result if m["role"] == "assistant")
    assert assistant["reasoning_requested"] is True
    assert assistant["reasoning"]["summary_text"] == "fallback summary"


def test_omits_reasoning_keys_when_not_present_in_output_msgs() -> None:
    """When output_messages has no reasoning_requested/reasoning, the API
    response must not contain those keys (they're optional)."""
    output_msgs = [{"role": "assistant", "content": "no reasoning here"}]
    rows = [_make_row("uuid-r1", "run-1", output_msgs, content="no reasoning here")]
    db = _make_db_session(rows)

    result = get_chat_messages(db, "session-abc")

    assistant = next(m for m in result if m["role"] == "assistant")
    assert "reasoning_requested" not in assistant
    assert "reasoning" not in assistant


def test_reasoning_duration_ms_computed_from_started_completed() -> None:
    """When reasoning was requested AND both timestamps exist, the projection
    must include reasoning_duration_ms = (completed_at - started_at) ms.
    Without it, the FE pill displays 'Thought for ·' with no number after a
    page refresh because the FE's streamStartedAtRef is gone."""
    started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    completed = datetime(2026, 1, 1, 12, 0, 7, 500000, tzinfo=UTC)  # +7.5s
    output_msgs = [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_requested": True,
        }
    ]
    rows = [
        _make_row(
            "uuid-r1",
            "run-1",
            output_msgs,
            content="answer",
            started_at=started,
            completed_at=completed,
        )
    ]
    db = _make_db_session(rows)

    result = get_chat_messages(db, "session-abc")

    assistant = next(m for m in result if m["role"] == "assistant")
    assert assistant.get("reasoning_duration_ms") == 7500


def test_reasoning_duration_ms_omitted_when_no_reasoning_requested() -> None:
    """If the run didn't request reasoning, don't bother projecting duration —
    the pill is hidden anyway and we keep payloads small."""
    started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    completed = datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC)
    output_msgs = [{"role": "assistant", "content": "answer"}]
    rows = [
        _make_row(
            "uuid-r1",
            "run-1",
            output_msgs,
            content="answer",
            started_at=started,
            completed_at=completed,
        )
    ]
    db = _make_db_session(rows)

    result = get_chat_messages(db, "session-abc")

    assistant = next(m for m in result if m["role"] == "assistant")
    assert "reasoning_duration_ms" not in assistant


def test_reasoning_duration_ms_omitted_when_timestamps_null() -> None:
    """Pre-A0 rows may have NULL started_at/completed_at — must not crash, must
    not emit a duration."""
    output_msgs = [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_requested": True,
        }
    ]
    rows = [
        _make_row(
            "uuid-r1",
            "run-1",
            output_msgs,
            content="answer",
            started_at=None,
            completed_at=None,
        )
    ]
    db = _make_db_session(rows)

    result = get_chat_messages(db, "session-abc")

    assistant = next(m for m in result if m["role"] == "assistant")
    assert "reasoning_duration_ms" not in assistant


def test_falsey_reasoning_requested_omitted() -> None:
    """If reasoning_requested is explicitly False (no reasoning was requested),
    the projection should not surface a False key — derivePillState already
    treats falsy as 'hidden', and omitting matches the "not present" semantics
    used elsewhere on this branch."""
    output_msgs = [
        {
            "role": "assistant",
            "content": "ok",
            "reasoning_requested": False,
        }
    ]
    rows = [_make_row("uuid-r1", "run-1", output_msgs, content="ok")]
    db = _make_db_session(rows)

    result = get_chat_messages(db, "session-abc")

    assistant = next(m for m in result if m["role"] == "assistant")
    # `if assistant_output_msg.get("reasoning_requested"):` truthy-only forward
    assert "reasoning_requested" not in assistant
