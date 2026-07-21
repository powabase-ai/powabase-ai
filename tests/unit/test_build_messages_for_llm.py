"""Tests for build_messages_for_llm conversion with cross-provider drop."""

from __future__ import annotations

from unittest.mock import patch

from agentic.agent.message import (
    AnthropicReasoning,
    Message,
    OpenAIReasoning,
)
from agentic_project_service.services.session import build_messages_for_llm


class _FakeContext:
    def __init__(self):
        self.events = []

    def emit_event(self, e):
        self.events.append(e)


def test_passes_through_when_no_reasoning():
    history = [Message(role="user", content="hello")]
    ctx = _FakeContext()
    with patch(
        "agentic_project_service.services.session.litellm.get_llm_provider",
        return_value=("claude-opus-4-7", "anthropic", None, None),
    ):
        out = build_messages_for_llm(
            session_history=history,
            target_model="anthropic/claude-opus-4-7",
            context=ctx,
            user_input="hi again",
        )
    assert out == [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "hi again"},
    ]
    assert ctx.events == []


def test_preserves_thinking_blocks_when_provider_matches():
    history = [
        Message(role="user", content="Q"),
        Message(
            role="assistant",
            content="A",
            reasoning=AnthropicReasoning(
                thinking_blocks=[{"type": "thinking", "thinking": "x", "signature": "s"}]
            ),
        ),
    ]
    ctx = _FakeContext()
    with patch(
        "agentic_project_service.services.session.litellm.get_llm_provider",
        return_value=("claude-opus-4-7", "anthropic", None, None),
    ):
        out = build_messages_for_llm(
            session_history=history,
            target_model="anthropic/claude-opus-4-7",
            context=ctx,
            user_input="Q2",
        )
    assistant = out[1]
    assert assistant["thinking_blocks"] == [{"type": "thinking", "thinking": "x", "signature": "s"}]
    assert ctx.events == []


def test_preserves_openai_psf_when_target_is_openai():
    """OpenAI encrypted_content_items survive same-provider replay."""
    history = [
        Message(role="user", content="Q"),
        Message(
            role="assistant",
            content="A",
            reasoning=OpenAIReasoning(
                encrypted_content_items=[{"type": "reasoning", "encrypted_content": "e"}],
                response_id="r1",
            ),
        ),
    ]
    ctx = _FakeContext()
    with patch(
        "agentic_project_service.services.session.litellm.get_llm_provider",
        return_value=("gpt-5.4", "openai", None, None),
    ):
        out = build_messages_for_llm(
            session_history=history,
            target_model="openai/responses/gpt-5.4",
            context=ctx,
            user_input="Q2",
        )
    assistant = out[1]
    assert assistant["provider_specific_fields"] == {
        "encrypted_content_items": [{"type": "reasoning", "encrypted_content": "e"}]
    }
    assert ctx.events == []


def test_drops_thinking_blocks_when_provider_changes():
    history = [
        Message(role="user", content="Q"),
        Message(
            role="assistant",
            content="A",
            reasoning=AnthropicReasoning(thinking_blocks=[{"type": "thinking", "thinking": "x"}]),
        ),
    ]
    ctx = _FakeContext()
    with patch(
        "agentic_project_service.services.session.litellm.get_llm_provider",
        return_value=("gpt-5.4", "openai", None, None),
    ):
        out = build_messages_for_llm(
            session_history=history,
            target_model="openai/gpt-5.4",
            context=ctx,
            user_input="Q2",
        )
    assistant = out[1]
    assert "thinking_blocks" not in assistant
    assert ctx.events == [
        {
            "type": "reasoning_dropped_at_provider_switch",
            "from_provider": "anthropic",
            "to_provider": "openai",
        }
    ]


def test_drops_openai_psf_when_target_is_anthropic():
    history = [
        Message(role="user", content="Q"),
        Message(
            role="assistant",
            content="A",
            reasoning=OpenAIReasoning(
                encrypted_content_items=[{"type": "reasoning", "encrypted_content": "e"}],
            ),
        ),
    ]
    ctx = _FakeContext()
    with patch(
        "agentic_project_service.services.session.litellm.get_llm_provider",
        return_value=("claude-opus-4-7", "anthropic", None, None),
    ):
        out = build_messages_for_llm(
            session_history=history,
            target_model="anthropic/claude-opus-4-7",
            context=ctx,
            user_input="Q2",
        )
    assistant = out[1]
    assert "provider_specific_fields" not in assistant
    assert len(ctx.events) == 1


def test_unknown_provider_does_not_drop():
    """If get_llm_provider raises (custom endpoint), preserve artifacts."""
    history = [
        Message(role="user", content="Q"),
        Message(
            role="assistant",
            content="A",
            reasoning=AnthropicReasoning(thinking_blocks=[{"type": "thinking", "thinking": "x"}]),
        ),
    ]
    ctx = _FakeContext()
    with patch(
        "agentic_project_service.services.session.litellm.get_llm_provider",
        side_effect=Exception("unknown"),
    ):
        out = build_messages_for_llm(
            session_history=history,
            target_model="custom/exotic",
            context=ctx,
            user_input="Q2",
        )
    assistant = out[1]
    assert "thinking_blocks" in assistant
    assert ctx.events == []


def test_appends_user_input_string():
    ctx = _FakeContext()
    with patch(
        "agentic_project_service.services.session.litellm.get_llm_provider",
        return_value=("x", "anthropic", None, None),
    ):
        out = build_messages_for_llm(
            session_history=[],
            target_model="anthropic/x",
            context=ctx,
            user_input="hello",
        )
    assert out == [{"role": "user", "content": "hello"}]


def test_appends_user_input_list():
    ctx = _FakeContext()
    with patch(
        "agentic_project_service.services.session.litellm.get_llm_provider",
        return_value=("x", "anthropic", None, None),
    ):
        out = build_messages_for_llm(
            session_history=[],
            target_model="anthropic/x",
            context=ctx,
            user_input=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        )
    assert out == [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]


def test_wraps_multimodal_content_array_as_single_user_message():
    """Regression for prod bug: callers in routes/agents.py build user_content
    as a list of multimodal content blocks (each with `type` but no `role`)
    when the KB is in image mode. Previously extend()-spread, causing
    `KeyError: 'role'` in agentic's normalize_messages, and (in some prod
    paths) `Invalid value for 'content': expected a string, got null` from
    OpenAI. The contract: a list whose items lack `role` is a content
    array — wrap it as the content of one user message.
    """
    ctx = _FakeContext()
    multimodal_content = [
        {"type": "text", "text": "Context from relevant documents:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
        {"type": "text", "text": "\n\nhvad er sundhedsprofilen?"},
    ]
    with patch(
        "agentic_project_service.services.session.litellm.get_llm_provider",
        return_value=("gpt-5-mini", "openai", None, None),
    ):
        out = build_messages_for_llm(
            session_history=[],
            target_model="openai/gpt-5-mini",
            context=ctx,
            user_input=multimodal_content,
        )
    assert all("role" in m for m in out), (
        f"role-less items leaked through: {[m for m in out if 'role' not in m]}"
    )
    assert out == [{"role": "user", "content": multimodal_content}]
