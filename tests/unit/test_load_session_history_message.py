"""Test that load_session_history returns list[Message] preserving tool_calls
and reasoning across the session boundary (closes pre-existing tool_calls drop bug)."""

from __future__ import annotations

from unittest.mock import MagicMock

from agentic.agent.message import AnthropicReasoning, Message
from agentic_project_service.services.session import load_session_history


def _mock_session_with_run_messages(input_messages_lists, output_messages_lists):
    """Build a MagicMock db_session whose execute() returns rows with the given
    input_messages/output_messages JSONB arrays."""
    rows = list(zip(input_messages_lists, output_messages_lists))
    cursor = MagicMock()
    cursor.__iter__ = lambda self: iter(rows)
    db_session = MagicMock()
    db_session.execute.return_value = cursor
    return db_session


def test_load_session_history_returns_message_objects():
    db = _mock_session_with_run_messages(
        input_messages_lists=[[{"role": "user", "content": "hi"}]],
        output_messages_lists=[[{"role": "assistant", "content": "hello"}]],
    )
    history = load_session_history(db, "test-session-uuid")
    assert all(isinstance(m, Message) for m in history)
    assert len(history) == 2  # user + assistant


def test_load_session_history_preserves_tool_calls():
    """Closes the pre-existing tool_calls drop bug."""
    db = _mock_session_with_run_messages(
        input_messages_lists=[[{"role": "user", "content": "do thing"}]],
        output_messages_lists=[
            [
                {
                    "role": "assistant",
                    "content": "calling tool",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "x", "arguments": "{}"},
                        }
                    ],
                }
            ]
        ],
    )
    history = load_session_history(db, "test-session-uuid")
    assistant = next(m for m in history if m.role == "assistant")
    assert assistant.tool_calls is not None
    assert len(assistant.tool_calls) == 1
    assert assistant.tool_calls[0]["function"]["name"] == "x"


def test_load_session_history_preserves_anthropic_reasoning():
    db = _mock_session_with_run_messages(
        input_messages_lists=[[{"role": "user", "content": "Q"}]],
        output_messages_lists=[
            [
                {
                    "role": "assistant",
                    "content": "A",
                    "reasoning": {
                        "provider": "anthropic",
                        "thinking_blocks": [{"type": "thinking", "thinking": "x"}],
                        "summary_text": "I considered X.",
                    },
                    "reasoning_requested": True,
                }
            ]
        ],
    )
    history = load_session_history(db, "test-session-uuid")
    assistant = next(m for m in history if m.role == "assistant")
    assert isinstance(assistant.reasoning, AnthropicReasoning)
    assert assistant.reasoning.thinking_blocks == [{"type": "thinking", "thinking": "x"}]
    assert assistant.reasoning_requested is True


def test_load_session_history_strips_citations():
    """strip_citation_markers strips [N]-style markers from assistant content."""
    db = _mock_session_with_run_messages(
        input_messages_lists=[[{"role": "user", "content": "Q"}]],
        output_messages_lists=[
            [
                {
                    "role": "assistant",
                    "content": "Final answer [1] more text [2]",
                }
            ]
        ],
    )
    history = load_session_history(db, "test-session-uuid")
    assistant = next(m for m in history if m.role == "assistant")
    assert "[1]" not in (assistant.content or "")
    assert "[2]" not in (assistant.content or "")
