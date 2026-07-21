"""Route-level tests for agent SSE streaming endpoint (issue #106 / Task 14).

Covers β buffer + persistence filter + complete-event content + failure-path
events persistence in ``agents.py:run_agent_stream`` — specifically the
ReAct streaming branch (``if tools:``). The chat-style branch (no tools,
calls ``Agent.stream()``) is exercised only by the regression test that
verifies it remains unchanged (no content_delta emission).

Tests patch ``agentic.agent.agent.Agent.run`` to drive the route's worker
thread without touching a real LLM. The patched function emits events via
``context.emit_event`` (which puts them on the route's ``event_queue``),
then returns an ``AgentOutput``-shaped ``MagicMock``.
"""

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from agentic_project_service.db import db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_agent_with_tool(app, name="ReActAgent"):
    """Insert an agents row + assign one builtin tool so the route's
    ``if tools:`` branch is taken.

    Returns the agent_id as a string.
    """
    agent_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".agents (id, name, model, system_prompt)
                VALUES (:id, :name, 'gpt-4o-mini', 'You are helpful.')
                """
            ),
            {"id": agent_id, "name": name},
        )
        # Assign web_search (non-DB builtin) so load_all_tools_for_agent
        # returns a non-empty dict without DB introspection.
        db.session.execute(
            text(
                """
                INSERT INTO "ai".agent_tools
                    (id, agent_id, tool_type, tool_name, config_override)
                VALUES (:id, :agent_id, 'builtin', 'web_search', '{}')
                """
            ),
            {"id": str(uuid.uuid4()), "agent_id": agent_id},
        )
        db.session.commit()
    return agent_id


def _create_agent_no_tools(app, name="ChatAgent"):
    """Insert an agents row WITHOUT any tools so the route takes the
    chat-style ``Agent.stream()`` branch."""
    agent_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".agents (id, name, model, system_prompt)
                VALUES (:id, :name, 'gpt-4o-mini', 'You are helpful.')
                """
            ),
            {"id": agent_id, "name": name},
        )
        db.session.commit()
    return agent_id


def _parse_sse_events(body: bytes) -> list[dict]:
    """Parse SSE response body into a list of decoded JSON event dicts.

    Skips comment-only frames (``: keepalive``).
    """
    events: list[dict] = []
    text_body = body.decode()
    for line in text_body.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line[len("data: ") :]))
    return events


def _has_keepalive_comment(body: bytes) -> bool:
    return b": keepalive" in body


def _make_fake_agent_output(content="result", status_success=True):
    """Build a MagicMock that quacks like an AgentOutput."""
    from agentic.execution.status import ExecutionStatus

    fake_output = MagicMock()
    fake_output.content = content
    fake_output.status = ExecutionStatus.COMPLETED if status_success else ExecutionStatus.FAILED
    fake_output.error = None
    fake_output.usage = {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}
    fake_output.steps = 1
    fake_output.events = []
    fake_output.tool_calls = []
    fake_output.reasoning_steps = []
    fake_output.messages = []
    fake_output.started_at = None
    fake_output.completed_at = None
    # Default reasoning fields to None/False so tests that don't care about
    # reasoning don't pass an auto-vivified MagicMock to Pydantic's Message
    # validator (which checks reasoning.provider against an enum). Tests that
    # DO exercise reasoning override these via fake_output.reasoning_artifact = ...
    fake_output.reasoning_artifact = None
    fake_output.reasoning_requested = False
    return fake_output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def react_agent_id(app):
    """Agent id with one tool assigned (triggers ReAct branch)."""
    return _create_agent_with_tool(app, "ReActAlpha")


@pytest.fixture
def chat_agent_id(app):
    """Agent id without tools (triggers chat-style ``Agent.stream()`` branch)."""
    return _create_agent_no_tools(app, "ChatAlpha")


# ---------------------------------------------------------------------------
# MUST-HAVE tests — ReAct branch
# ---------------------------------------------------------------------------


class TestPersistenceFilter:
    """content_delta and reasoning_delta MUST NOT land in events_for_db."""

    def test_persistence_filter_excludes_deltas(
        self, client, app, mock_auth, auth_headers, react_agent_id, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        def fake_run(messages, *, context=None, tools=None, **kwargs):
            context.emit_event({"type": "content_delta", "delta": "Hello "})
            context.emit_event({"type": "content_delta", "delta": "world"})
            context.emit_event({"type": "reasoning_delta", "delta": "thinking..."})
            context.emit_event({"type": "tool_call", "name": "search"})
            return _make_fake_agent_output("Hello world")

        with patch("agentic.agent.agent.Agent.run", side_effect=fake_run):
            resp = client.post(
                f"/api/agents/{react_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        # The events JSONB column must NOT contain content_delta or reasoning_delta
        with app.app_context():
            row = db.session.execute(
                text('SELECT events FROM "ai".agent_runs WHERE input_messages::text LIKE :q'),
                {"q": '%"go"%'},
            ).fetchone()
            assert row is not None
            persisted_events = row[0] or []
            persisted_types = {e.get("type") for e in persisted_events}
            assert "content_delta" not in persisted_types
            assert "reasoning_delta" not in persisted_types
            # tool_call (terminal) is preserved
            assert "tool_call" in persisted_types


class TestTerminalChunkSource:
    """Terminal `chunk` event content depends on streaming flag (β)."""

    def test_terminal_chunk_uses_buffer_in_streaming_mode(
        self, client, app, mock_auth, auth_headers, react_agent_id, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        deltas = ["Step1: ", "checking ", "data. ", "Step2: ", "answer."]

        def fake_run(messages, *, context=None, tools=None, **kwargs):
            for d in deltas:
                context.emit_event({"type": "content_delta", "delta": d})
            # Last-step content differs from the buffer (intermediary prose was
            # streamed but output.content only carries the last step). The
            # terminal `chunk` MUST come from the buffer.
            return _make_fake_agent_output("Step2: answer.")

        with patch("agentic.agent.agent.Agent.run", side_effect=fake_run):
            resp = client.post(
                f"/api/agents/{react_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        events = _parse_sse_events(resp.get_data())
        chunk_events = [e for e in events if e.get("event") == "chunk"]
        assert len(chunk_events) == 1
        assert chunk_events[0]["content"] == "".join(deltas)

    def test_terminal_chunk_uses_output_content_when_streaming_disabled(
        self, client, app, mock_auth, auth_headers, react_agent_id, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "false")

        def fake_run(messages, *, context=None, tools=None, **kwargs):
            # Even if deltas were emitted, kill-switch off → buffer ignored,
            # chunk falls back to output.content.
            context.emit_event({"type": "content_delta", "delta": "ignored "})
            return _make_fake_agent_output("final-output")

        with patch("agentic.agent.agent.Agent.run", side_effect=fake_run):
            resp = client.post(
                f"/api/agents/{react_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        events = _parse_sse_events(resp.get_data())
        chunk_events = [e for e in events if e.get("event") == "chunk"]
        assert len(chunk_events) == 1
        assert chunk_events[0]["content"] == "final-output"


class TestFailurePathPersistsEvents:
    """M5 v3: failure-path persistence must include events_for_db."""

    def test_failure_path_persists_events_for_db(
        self, client, app, mock_auth, auth_headers, react_agent_id, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        def fake_run(messages, *, context=None, tools=None, **kwargs):
            # Emit some terminal events before raising — these must reach the
            # DB via the failure-path update_agent_run call.
            context.emit_event({"type": "tool_call", "name": "search"})
            context.emit_event({"type": "chunk", "content": "partial-before-fail"})
            raise RuntimeError("synthetic mid-stream failure")

        with patch("agentic.agent.agent.Agent.run", side_effect=fake_run):
            resp = client.post(
                f"/api/agents/{react_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        # Run row should be marked failed AND have events persisted.
        with app.app_context():
            row = db.session.execute(
                text(
                    'SELECT status, events, error FROM "ai".agent_runs '
                    "WHERE input_messages::text LIKE :q"
                ),
                {"q": '%"go"%'},
            ).fetchone()
            assert row is not None
            assert row[0] == "failed"
            persisted_events = row[1] or []
            persisted_types = {e.get("type") for e in persisted_events}
            assert "tool_call" in persisted_types
            assert "chunk" in persisted_types
            assert "synthetic mid-stream failure" in (row[2] or "")


# ---------------------------------------------------------------------------
# SHOULD-HAVE tests — ReAct branch
# ---------------------------------------------------------------------------


class TestCompleteEventContent:
    """B1: complete.content MUST equal final_content (= buffer in streaming mode)."""

    def test_complete_event_content_matches_buffer_in_streaming_mode(
        self, client, app, mock_auth, auth_headers, react_agent_id, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        deltas = ["A", "B", "C", "D"]

        def fake_run(messages, *, context=None, tools=None, **kwargs):
            for d in deltas:
                context.emit_event({"type": "content_delta", "delta": d})
            return _make_fake_agent_output("D-only")

        with patch("agentic.agent.agent.Agent.run", side_effect=fake_run):
            resp = client.post(
                f"/api/agents/{react_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        events = _parse_sse_events(resp.get_data())
        complete_events = [e for e in events if e.get("event") == "complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["content"] == "ABCD"


class TestRunContentPersisted:
    """B1: persisted run.content MUST equal final_content (= buffer in streaming)."""

    def test_run_content_persisted_matches_buffer_in_streaming_mode(
        self, client, app, mock_auth, auth_headers, react_agent_id, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        deltas = ["multi-", "step-", "buffered"]

        def fake_run(messages, *, context=None, tools=None, **kwargs):
            for d in deltas:
                context.emit_event({"type": "content_delta", "delta": d})
            return _make_fake_agent_output("buffered")

        with patch("agentic.agent.agent.Agent.run", side_effect=fake_run):
            resp = client.post(
                f"/api/agents/{react_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        with app.app_context():
            row = db.session.execute(
                text('SELECT content FROM "ai".agent_runs WHERE input_messages::text LIKE :q'),
                {"q": '%"go"%'},
            ).fetchone()
            assert row is not None
            assert row[0] == "multi-step-buffered"


class TestKeepalivePreserved:
    """30s queue-empty timeout must still emit ``: keepalive``."""

    def test_keepalive_preserved(
        self, client, app, mock_auth, auth_headers, react_agent_id, monkeypatch
    ):
        # Patch the route's queue.Queue.get to immediately raise queue.Empty
        # for the first few calls so the route's keepalive branch is exercised
        # without an actual 30s wait. After a few empties, return the sentinel.
        import queue as queue_mod

        original_get = queue_mod.Queue.get
        empty_calls = {"n": 0}

        def fake_get(self, *args, **kwargs):
            # First two calls: simulate timeout (no events available)
            if empty_calls["n"] < 2:
                empty_calls["n"] += 1
                raise queue_mod.Empty()
            return original_get(self, *args, **kwargs)

        def fake_run(messages, *, context=None, tools=None, **kwargs):
            return _make_fake_agent_output("ok")

        with (
            patch.object(queue_mod.Queue, "get", fake_get),
            patch("agentic.agent.agent.Agent.run", side_effect=fake_run),
        ):
            resp = client.post(
                f"/api/agents/{react_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        # The body should contain at least one ``: keepalive`` comment frame.
        assert _has_keepalive_comment(resp.get_data())


# ---------------------------------------------------------------------------
# Chat-style branch tests — issue #274 (delta + reasoning parity with ReAct β)
# ---------------------------------------------------------------------------


def _chat_stream_factory(
    deltas: list[str],
    reasoning_deltas: list[str] | None = None,
    final_content: str | None = None,
    reasoning_artifact=None,
    reasoning_requested: bool = False,
):
    """Build a `fake_stream` that mimics the new Agent.stream() callback API.

    Fires on_content_delta / on_reasoning_delta callbacks, yields the same
    content fragments, and returns an AgentOutput-shaped MagicMock with the
    given reasoning_artifact + reasoning_requested.
    """
    final = final_content if final_content is not None else "".join(deltas)
    reasoning_fragments = reasoning_deltas or []

    def fake_stream(
        self,
        messages,
        *,
        on_content_delta=None,
        on_reasoning_delta=None,
        **kwargs,
    ):
        from agentic.execution.status import ExecutionStatus

        for r in reasoning_fragments:
            if on_reasoning_delta is not None:
                on_reasoning_delta(r)
        for d in deltas:
            if on_content_delta is not None:
                on_content_delta(d)
            yield d

        fake_output = _make_fake_agent_output(final)
        fake_output.status = ExecutionStatus.COMPLETED
        fake_output.reasoning_artifact = reasoning_artifact
        fake_output.reasoning_requested = reasoning_requested
        return fake_output

    return fake_stream


class TestChatStyleStreamingWithDeltas:
    """Issue #274 contract — content_delta per fragment + a single terminal
    chunk with the full content."""

    def test_emits_content_delta_per_chunk(
        self, client, app, mock_auth, auth_headers, chat_agent_id, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
        fake_stream = _chat_stream_factory(["first ", "second ", "third"])

        with patch("agentic.agent.agent.Agent.stream", fake_stream):
            resp = client.post(
                f"/api/agents/{chat_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        events = _parse_sse_events(resp.get_data())
        delta_events = [e for e in events if e.get("event") == "content_delta"]
        assert [e["delta"] for e in delta_events] == ["first ", "second ", "third"]
        # N-7: payload must carry `type` alongside `event` for ReAct symmetry.
        for e in delta_events:
            assert e.get("type") == "content_delta"

    def test_emits_terminal_chunk_with_full_content(
        self, client, app, mock_auth, auth_headers, chat_agent_id, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
        fake_stream = _chat_stream_factory(["A", "B", "C"])

        with patch("agentic.agent.agent.Agent.stream", fake_stream):
            resp = client.post(
                f"/api/agents/{chat_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        events = _parse_sse_events(resp.get_data())
        chunk_events = [e for e in events if e.get("event") == "chunk"]
        assert len(chunk_events) == 1
        assert chunk_events[0]["content"] == "ABC"
        # Negative: no artifact in this run → no synthetic terminal
        # `reasoning` event should slip through.
        terminal_reasoning = [
            e for e in events if e.get("event") == "reasoning" and "delta" not in e
        ]
        assert terminal_reasoning == []

    def test_no_terminal_chunk_when_only_reasoning(
        self, client, app, mock_auth, auth_headers, chat_agent_id, monkeypatch
    ):
        """C-4: when the model emits reasoning only (no content), the
        terminal `chunk` guard skips emission — otherwise the FE would render
        an empty bubble."""
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
        fake_stream = _chat_stream_factory(
            deltas=[], reasoning_deltas=["thinking"], final_content=""
        )

        with patch("agentic.agent.agent.Agent.stream", fake_stream):
            resp = client.post(
                f"/api/agents/{chat_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        events = _parse_sse_events(resp.get_data())
        assert [e for e in events if e.get("event") == "chunk"] == []

    def test_no_step_or_tool_events_leak(
        self, client, app, mock_auth, auth_headers, chat_agent_id, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
        fake_stream = _chat_stream_factory(["x"])

        with patch("agentic.agent.agent.Agent.stream", fake_stream):
            resp = client.post(
                f"/api/agents/{chat_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        events = _parse_sse_events(resp.get_data())
        types = {e.get("event") for e in events}
        # Loop-specific event types must not appear on the naive branch.
        loop_only = {
            "step_started",
            "step_completed",
            "step_reset",
            "tool_call",
            "tool_result",
            "delegation_started",
            "delegation_completed",
            "approval_requested",
            "reactive_compact",
            "output_recovery",
        }
        assert types.isdisjoint(loop_only), f"leaked loop events: {types & loop_only}"


class TestChatStyleReasoningPersisted:
    """reasoning_delta reaches the FE + the artifact lands on the DB row."""

    def test_emits_reasoning_delta(
        self, client, app, mock_auth, auth_headers, chat_agent_id, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        # Construct a real artifact so the persisted reasoning is non-empty.
        from agentic.agent.message import AnthropicReasoning

        artifact = AnthropicReasoning(
            thinking_blocks=[{"type": "thinking", "thinking": "step 1"}],
            summary_text="I considered the options.",
            requested_effort="high",
        )
        fake_stream = _chat_stream_factory(
            deltas=["the ", "answer"],
            reasoning_deltas=["Considering...", " more"],
            reasoning_artifact=artifact,
            reasoning_requested=True,
        )

        with patch("agentic.agent.agent.Agent.stream", fake_stream):
            resp = client.post(
                f"/api/agents/{chat_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        events = _parse_sse_events(resp.get_data())
        reasoning_events = [e for e in events if e.get("event") == "reasoning_delta"]
        assert [e["delta"] for e in reasoning_events] == ["Considering...", " more"]
        # FE's buildReasoningSteps gates accumulation on ev.step + ev.type —
        # step-less or type-less events would never reach the pill.
        for e in reasoning_events:
            assert e.get("step") == 1, "naive branch uses a constant step=1"
            assert e.get("source") == "thinking"
            assert e.get("type") == "reasoning_delta"

    def test_persists_reasoning_artifact(
        self, client, app, mock_auth, auth_headers, chat_agent_id, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        from agentic.agent.message import AnthropicReasoning

        artifact = AnthropicReasoning(
            thinking_blocks=[{"type": "thinking", "thinking": "step 1"}],
            summary_text="I considered the options.",
            requested_effort="high",
        )
        fake_stream = _chat_stream_factory(
            deltas=["x"],
            reasoning_deltas=["..."],
            reasoning_artifact=artifact,
            reasoning_requested=True,
        )

        # Unique probe message so the SELECT below targets THIS test's row.
        probe = f"persist-artifact-probe-{uuid.uuid4().hex[:8]}"
        with patch("agentic.agent.agent.Agent.stream", fake_stream):
            resp = client.post(
                f"/api/agents/{chat_agent_id}/run/stream",
                json={"message": probe},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        with app.app_context():
            row = db.session.execute(
                text(
                    'SELECT output_messages FROM "ai".agent_runs WHERE input_messages::text LIKE :q'
                ),
                {"q": f"%{probe}%"},
            ).fetchone()
            assert row is not None
            messages = row[0] or []
            assert messages, "expected a persisted assistant message"
            reasoning = messages[0].get("reasoning") or {}
            assert reasoning.get("summary_text") == "I considered the options."
            assert reasoning.get("thinking_blocks")

    def test_emits_and_persists_terminal_reasoning_event(
        self, client, app, mock_auth, auth_headers, chat_agent_id, monkeypatch
    ):
        """After streaming, a terminal `reasoning` event with the full
        thinking text must reach BOTH the SSE stream AND agent_runs.events.

        Without this, the FE's pill expanded-panel renders empty after page
        reload — derivePillState sees msg.reasoning.summary_text and returns
        done-full, but buildReasoningSteps replays only from agent_runs.events
        (which would be empty), so the step list is []. ReAct emits this via
        the `reasoning_text` branch in `Agent._react_loop`; naive must mirror.
        """
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        from agentic.agent.message import AnthropicReasoning

        artifact = AnthropicReasoning(
            thinking_blocks=[{"type": "thinking", "thinking": "weighed both"}],
            summary_text="Both have mass 1 kg, so equal weight.",
            requested_effort="high",
        )
        fake_stream = _chat_stream_factory(
            deltas=["answer"],
            reasoning_deltas=["weighed both"],
            reasoning_artifact=artifact,
            reasoning_requested=True,
        )

        probe = f"terminal-reasoning-probe-{uuid.uuid4().hex[:8]}"
        with patch("agentic.agent.agent.Agent.stream", fake_stream):
            resp = client.post(
                f"/api/agents/{chat_agent_id}/run/stream",
                json={"message": probe},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        events = _parse_sse_events(resp.get_data())
        terminal_reasoning = [
            e for e in events if e.get("event") == "reasoning" and "delta" not in e
        ]
        assert len(terminal_reasoning) == 1
        assert terminal_reasoning[0]["content"] == "Both have mass 1 kg, so equal weight."
        assert terminal_reasoning[0]["step"] == 1
        assert terminal_reasoning[0]["source"] == "thinking"

        with app.app_context():
            row = db.session.execute(
                text('SELECT events FROM "ai".agent_runs WHERE input_messages::text LIKE :q'),
                {"q": f"%{probe}%"},
            ).fetchone()
            assert row is not None
            persisted_events = row[0] or []
            # Persisted shape: {"type": "reasoning", "step": 1, "source": ...,
            # "content": ...}. Delta events MUST NOT be persisted.
            reasoning_in_db = [e for e in persisted_events if e.get("type") == "reasoning"]
            assert len(reasoning_in_db) == 1
            assert reasoning_in_db[0]["content"] == "Both have mass 1 kg, so equal weight."
            delta_in_db = [
                e for e in persisted_events if e.get("type") in ("content_delta", "reasoning_delta")
            ]
            assert delta_in_db == [], "deltas must not land in events column"

    def test_redacted_thinking_does_not_emit_terminal_reasoning(
        self, client, app, mock_auth, auth_headers, chat_agent_id, monkeypatch
    ):
        """Anthropic redacted_thinking carries thinking_blocks but null
        summary_text. The terminal emission gates on summary_text, so this
        case must NOT produce a `reasoning` SSE event (its content field
        would be empty, which breaks the FE's step-text rendering)."""
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        from agentic.agent.message import AnthropicReasoning

        # summary_text=None mimics Anthropic redacted_thinking: the model
        # reasoned but the API returned only opaque blocks, no plaintext.
        artifact = AnthropicReasoning(
            thinking_blocks=[{"type": "redacted_thinking", "data": "[opaque]"}],
            summary_text=None,
            requested_effort="high",
        )
        fake_stream = _chat_stream_factory(
            deltas=["answer"],
            reasoning_artifact=artifact,
            reasoning_requested=True,
        )

        with patch("agentic.agent.agent.Agent.stream", fake_stream):
            resp = client.post(
                f"/api/agents/{chat_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        events = _parse_sse_events(resp.get_data())
        terminal_reasoning = [
            e for e in events if e.get("event") == "reasoning" and "delta" not in e
        ]
        assert terminal_reasoning == [], (
            "Empty summary_text must not produce an empty-content reasoning event"
        )


class TestChatStyleKillSwitchFallback:
    """``AGENT_LLM_STREAMING_ENABLED=false`` keeps the prior per-token
    ``chunk`` emission (rollback path)."""

    def test_kill_switch_off_emits_chunk_no_content_delta(
        self, client, app, mock_auth, auth_headers, chat_agent_id, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "false")
        fake_stream = _chat_stream_factory(["A", "B"])

        with patch("agentic.agent.agent.Agent.stream", fake_stream):
            resp = client.post(
                f"/api/agents/{chat_agent_id}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        events = _parse_sse_events(resp.get_data())
        delta_events = [e for e in events if e.get("event") == "content_delta"]
        chunk_events = [e for e in events if e.get("event") == "chunk"]
        assert delta_events == []
        # Kill-switch off: per-token chunk events (one per fragment) preserved.
        assert [e["content"] for e in chunk_events] == ["A", "B"]


class TestChatStyleBackgroundFinish:
    """When a client disconnects mid-stream, _finish_run_in_background must
    still persist the terminal `reasoning` event so the pill's expanded
    panel survives a page reload (mirror of the foreground synthesis)."""

    def test_background_finish_persists_terminal_reasoning_event(self, app, chat_agent_id):
        """Drive _finish_run_in_background directly with a stub generator
        that returns an AgentOutput carrying a reasoning artifact, and assert
        the events column ends up populated.
        """
        from agentic.agent.message import AnthropicReasoning
        from agentic.execution.status import ExecutionStatus
        from agentic_project_service.routes.agents import _finish_run_in_background
        from agentic_project_service.services.session import persist_agent_run
        from agentic_project_service.models.tenant import AgentRunStatus

        artifact = AnthropicReasoning(
            thinking_blocks=[{"type": "thinking", "thinking": "bg thought"}],
            summary_text="Considered after the client left.",
            requested_effort="high",
        )

        # Insert a session row + a RUNNING agent_runs row so the background
        # path has something to UPDATE. probe is unique per test.
        probe = f"bg-finish-probe-{uuid.uuid4().hex[:8]}"
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        session_id = f"sess_{uuid.uuid4().hex[:12]}"

        with app.app_context():
            session_uuid = str(uuid.uuid4())
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agent_sessions (id, session_id, agent_id, user_id)
                    VALUES (:id, :sid, :aid, :uid)
                    """
                ),
                {
                    "id": session_uuid,
                    "sid": session_id,
                    "aid": chat_agent_id,
                    "uid": str(uuid.uuid4()),
                },
            )
            persist_agent_run(
                db_session=db.session,
                db_session_uuid=session_uuid,
                run_id=run_id,
                status=AgentRunStatus.RUNNING,
                input_messages=[{"role": "user", "content": probe}],
                content="",
                started_at=datetime.now(UTC),
            )
            db.session.commit()

        # Stub LLM generator: yields one remaining chunk, then StopIteration
        # carrying the AgentOutput (matches what Agent.stream's worker would
        # return).
        def stub_gen():
            yield " trailing"
            fake_output = _make_fake_agent_output("final final")
            fake_output.status = ExecutionStatus.COMPLETED
            fake_output.reasoning_artifact = artifact
            fake_output.reasoning_requested = True
            return fake_output

        _finish_run_in_background(
            flask_app=app,
            run_id=run_id,
            llm_gen=stub_gen(),
            content_chunks=["already "],
            message=probe,
            query_enrichment=None,
            retrieved_context_for_db=None,
            context_handler_id=None,
            started_at=datetime.now(UTC),
            reasoning_requested=True,
        )

        with app.app_context():
            row = db.session.execute(
                text('SELECT events, output_messages FROM "ai".agent_runs WHERE run_id = :rid'),
                {"rid": run_id},
            ).fetchone()
            assert row is not None
            persisted_events = row[0] or []
            reasoning_in_db = [e for e in persisted_events if e.get("type") == "reasoning"]
            assert len(reasoning_in_db) == 1, (
                f"background-finish must persist the terminal reasoning event "
                f"(got events: {persisted_events})"
            )
            assert reasoning_in_db[0]["content"] == "Considered after the client left."
            # And the artifact still rides on output_messages — confirms the
            # background path didn't drop it on the floor either.
            messages = row[1] or []
            assert messages[0]["reasoning"]["summary_text"] == ("Considered after the client left.")


# ---------------------------------------------------------------------------
# Dedup tests — Phase 2 Task 3
# ---------------------------------------------------------------------------


def test_run_agent_stream_dedups_overlapping_tool_chunks(
    client, app, db_cleanup, mock_auth, auth_headers, mocker
):
    """Multi-knowledge_search runs must dedup chunks by UUID before merging
    retrieved_context, so [N] citation keys are unambiguous downstream.

    Two knowledge_search tool calls return overlapping chunk_b_id. After the
    run completes, retrieved_context in agent_runs must contain chunk_b_id
    exactly once.
    """
    chunk_a_id = "11111111-1111-1111-1111-111111111111"
    chunk_b_id = "22222222-2222-2222-2222-222222222222"  # the duplicate
    chunk_c_id = "33333333-3333-3333-3333-333333333333"
    handler_a_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    handler_b_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    # Create an agent with a tool (triggers ReAct / tool branch)
    agent_id = _create_agent_with_tool(app, "DedupeTestAgent")

    def fake_get_context_handler(db_session, hid, resolve_text=False):
        if hid == handler_a_id:
            return {
                "retrieved_context": [
                    {"_type": "retrieval_diagnostics", "total_items": 2},
                    {
                        "id": chunk_a_id,
                        "text": "chunk a",
                        "source_id": "s1",
                        "kb_name": "kb",
                    },
                    {
                        "id": chunk_b_id,
                        "text": "chunk b (call 1)",
                        "source_id": "s1",
                        "kb_name": "kb",
                    },
                ]
            }
        if hid == handler_b_id:
            return {
                "retrieved_context": [
                    {"_type": "retrieval_diagnostics", "total_items": 2},
                    {
                        "id": chunk_b_id,
                        "text": "chunk b (call 2, duplicate)",
                        "source_id": "s1",
                        "kb_name": "kb",
                    },
                    {
                        "id": chunk_c_id,
                        "text": "chunk c",
                        "source_id": "s1",
                        "kb_name": "kb",
                    },
                ]
            }
        return None

    mocker.patch(
        "agentic_project_service.routes.agents.get_context_handler",
        side_effect=fake_get_context_handler,
    )

    # Insert stub context_handlers rows so the agent_runs FK doesn't reject
    # the handler IDs that get_context_handler is mocked to return.
    with app.app_context():
        for hid in (handler_a_id, handler_b_id):
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".context_handlers (id, query, knowledge_base_configs)
                    VALUES (:id, 'test query', '[]')
                    """
                ),
                {"id": hid},
            )
        db.session.commit()

    probe = f"dedup-probe-{uuid.uuid4().hex[:8]}"

    def fake_run(messages, *, context=None, tools=None, **kwargs):
        # Emit two context_handler_created events to simulate two
        # knowledge_search tool calls returning overlapping chunks.
        context.emit_event({"type": "context_handler_created", "context_handler_id": handler_a_id})
        context.emit_event({"type": "context_handler_created", "context_handler_id": handler_b_id})
        out = _make_fake_agent_output(content="Answer using chunks.")
        # Explicitly set reasoning_artifact to None so Message() validation
        # doesn't hit a MagicMock when building output_messages.
        out.reasoning_artifact = None
        out.reasoning_requested = False
        return out

    mocker.patch("agentic.agent.agent.Agent.run", side_effect=fake_run)

    resp = client.post(
        f"/api/agents/{agent_id}/run/stream",
        json={"message": probe, "citations_enabled": False},
        headers=auth_headers,
        buffered=True,
    )
    assert resp.status_code == 200

    # Inspect persisted retrieved_context
    with app.app_context():
        row = db.session.execute(
            text(
                'SELECT retrieved_context FROM "ai".agent_runs WHERE input_messages::text LIKE :q'
            ),
            {"q": f"%{probe}%"},
        ).fetchone()
    assert row is not None, "agent_runs row not found"
    rc = row[0] or []

    non_diag = [item for item in rc if item.get("_type") != "retrieval_diagnostics"]
    ids = [item["id"] for item in non_diag]

    assert chunk_b_id in ids, "duplicate chunk should be present (at least once)"
    assert ids.count(chunk_b_id) == 1, (
        f"chunk_b expected exactly once, got {ids.count(chunk_b_id)} times; full ids: {ids}"
    )
    assert sorted(ids) == sorted([chunk_a_id, chunk_b_id, chunk_c_id]), (
        f"expected exactly 3 unique chunks, got: {ids}"
    )


def test_run_agent_stream_dedup_preserves_items_without_id(
    client, app, db_cleanup, mock_auth, auth_headers, mocker
):
    """Items in retrieved_context that somehow lack an 'id' field must
    be preserved through dedup, not silently dropped. Multiple such items
    can coexist (they're never considered duplicates of each other since
    there's no id to compare)."""
    handler_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    chunk_with_id = "dddddddd-dddd-dddd-dddd-dddddddddddd"

    agent_id = _create_agent_with_tool(app, "NoIdDedupeTestAgent")

    def fake_get_context_handler(db_session, hid, resolve_text=False):
        return {
            "retrieved_context": [
                {"_type": "retrieval_diagnostics", "total_items": 3},
                # Two items with no 'id' field — should both be kept
                {"text": "weird item without id, copy 1", "source_id": "s1", "kb_name": "kb"},
                {"text": "weird item without id, copy 2", "source_id": "s1", "kb_name": "kb"},
                # One normal item with id, for sanity check
                {"id": chunk_with_id, "text": "normal item", "source_id": "s1", "kb_name": "kb"},
            ]
        }

    mocker.patch(
        "agentic_project_service.routes.agents.get_context_handler",
        side_effect=fake_get_context_handler,
    )

    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".context_handlers (id, query, knowledge_base_configs)
                VALUES (:id, 'test query', '[]')
                """
            ),
            {"id": handler_id},
        )
        db.session.commit()

    probe = f"noid-probe-{uuid.uuid4().hex[:8]}"

    def fake_run(messages, *, context=None, tools=None, **kwargs):
        context.emit_event({"type": "context_handler_created", "context_handler_id": handler_id})
        out = _make_fake_agent_output(content="Answer using chunks.")
        out.reasoning_artifact = None
        out.reasoning_requested = False
        return out

    mocker.patch("agentic.agent.agent.Agent.run", side_effect=fake_run)

    resp = client.post(
        f"/api/agents/{agent_id}/run/stream",
        json={"message": probe, "citations_enabled": False},
        headers=auth_headers,
        buffered=True,
    )
    assert resp.status_code == 200

    with app.app_context():
        row = db.session.execute(
            text(
                'SELECT retrieved_context FROM "ai".agent_runs WHERE input_messages::text LIKE :q'
            ),
            {"q": f"%{probe}%"},
        ).fetchone()
    assert row is not None, "agent_runs row not found"
    rc = row[0] or []

    non_diag = [item for item in rc if item.get("_type") != "retrieval_diagnostics"]

    assert len([i for i in non_diag if "id" not in i]) == 2, "both no-id items should be preserved"
    assert chunk_with_id in [i.get("id") for i in non_diag], (
        "the normal id'd item should also be present"
    )
    assert len(non_diag) == 3, f"expected 3 total non-diagnostics items, got {len(non_diag)}"


# ---------------------------------------------------------------------------
# Citation tests — Phase 2 Task 4
# ---------------------------------------------------------------------------


def test_run_agent_stream_persists_citations_for_tool_based_run(
    client, app, db_cleanup, mock_auth, auth_headers, mocker, monkeypatch
):
    """Tool-based agentic run with [N] markers in the response must populate
    ai.message_citations and emit citations on the complete SSE event,
    matching the existing pre-fetched-context behavior.
    """
    # Use output.content as final_content (not content_buffer) so citation
    # markers are present. streaming_enabled=true uses content_buffer (empty
    # when fake_run doesn't emit deltas), losing the [N] markers entirely.
    monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "false")

    chunk_1_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    chunk_2_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    handler_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"

    agent_id = _create_agent_with_tool(app, "CitationTestAgent")

    # Insert stub context_handlers row to satisfy FK
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".context_handlers (id, query, knowledge_base_configs)
                VALUES (:id, 'test query', '[]')
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"id": handler_id},
        )
        db.session.commit()

    def fake_get_context_handler(db_session, hid, resolve_text=False):
        # source_id omitted so persist_citations inserts NULL (no FK violation)
        return {
            "retrieved_context": [
                {"_type": "retrieval_diagnostics", "total_items": 2},
                {"id": chunk_1_id, "text": "Provision 1 text", "kb_name": "kb"},
                {"id": chunk_2_id, "text": "Provision 2 text", "kb_name": "kb"},
            ]
        }

    mocker.patch(
        "agentic_project_service.routes.agents.get_context_handler",
        side_effect=fake_get_context_handler,
    )

    probe = f"citation-probe-{uuid.uuid4().hex[:8]}"

    def fake_run(messages, *, context=None, tools=None, **kwargs):
        context.emit_event({"type": "context_handler_created", "context_handler_id": handler_id})
        out = _make_fake_agent_output(
            content=f"Provision 1 says X [1]. Provision 2 says Y [2]. {probe}"
        )
        out.reasoning_artifact = None
        out.reasoning_requested = False
        return out

    mocker.patch("agentic.agent.agent.Agent.run", side_effect=fake_run)

    resp = client.post(
        f"/api/agents/{agent_id}/run/stream",
        json={"message": probe, "citations_enabled": True},
        headers=auth_headers,
        buffered=True,
    )
    assert resp.status_code == 200

    events = _parse_sse_events(resp.get_data())

    # Assertion 1: structured citations on complete event
    # Each citation dict from parse_citations_from_response has "key" (string)
    # and "item_id" fields. Keys are sequential strings "1", "2", ...
    complete_event = next(e for e in events if e.get("event") == "complete")
    citations = complete_event.get("citations") or []
    assert len(citations) == 2, f"expected 2 citations, got: {citations}"
    citation_by_key = {int(c["key"]): c for c in citations}
    assert citation_by_key[1]["item_id"] == chunk_1_id
    assert citation_by_key[2]["item_id"] == chunk_2_id

    # Assertion 2: persisted in ai.message_citations
    start_event = next(e for e in events if e.get("event") == "start")
    run_id = start_event["run_id"]
    with app.app_context():
        rows = db.session.execute(
            text(
                'SELECT citation_key, item_id FROM "ai".message_citations '
                'WHERE run_id = (SELECT id FROM "ai".agent_runs WHERE run_id = :rid)'
            ),
            {"rid": run_id},
        ).fetchall()
    assert len(rows) == 2, f"expected 2 message_citations rows, got: {rows}"
    persisted_keys = sorted(r[0] for r in rows)
    assert persisted_keys == [1, 2]

    # Assertion 3: content is the cleaned text + non-empty
    assert "[1]" in complete_event["content"]  # valid markers kept
    assert "[2]" in complete_event["content"]
    assert complete_event["content"]  # non-empty

    # Cleanup: db_cleanup doesn't truncate message_citations
    with app.app_context():
        db.session.execute(
            text(
                'DELETE FROM "ai".message_citations WHERE run_id IN '
                '(SELECT id FROM "ai".agent_runs WHERE run_id = :rid)'
            ),
            {"rid": run_id},
        )
        db.session.commit()


def test_run_agent_stream_no_markers_means_no_citations(
    client, app, db_cleanup, mock_auth, auth_headers, mocker, monkeypatch
):
    """Tool-based agentic run where the LLM response contains no [N]
    markers must produce 0 message_citations rows and no citations field
    on the complete SSE event. Guards against future regressions that
    might falsely create citation entries when no markers are present.
    """
    monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "false")

    chunk_1_id = "11111111-aaaa-bbbb-cccc-111111111111"
    chunk_2_id = "22222222-aaaa-bbbb-cccc-222222222222"
    handler_id = "33333333-aaaa-bbbb-cccc-333333333333"

    agent_id = _create_agent_with_tool(app, "NoCitationTestAgent")

    # Insert stub context_handlers row to satisfy FK
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".context_handlers (id, query, knowledge_base_configs)
                VALUES (:id, 'test query', '[]')
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"id": handler_id},
        )
        db.session.commit()

    def fake_get_context_handler(db_session, hid, resolve_text=False):
        return {
            "retrieved_context": [
                {"_type": "retrieval_diagnostics", "total_items": 2},
                {"id": chunk_1_id, "text": "Provision 1 text", "kb_name": "kb"},
                {"id": chunk_2_id, "text": "Provision 2 text", "kb_name": "kb"},
            ]
        }

    mocker.patch(
        "agentic_project_service.routes.agents.get_context_handler",
        side_effect=fake_get_context_handler,
    )

    probe = f"no-citation-probe-{uuid.uuid4().hex[:8]}"

    def fake_run(messages, *, context=None, tools=None, **kwargs):
        context.emit_event({"type": "context_handler_created", "context_handler_id": handler_id})
        out = _make_fake_agent_output(
            content=f"Provision 1 says X. Provision 2 says Y. No markers anywhere. {probe}"
        )
        out.reasoning_artifact = None
        out.reasoning_requested = False
        return out

    mocker.patch("agentic.agent.agent.Agent.run", side_effect=fake_run)

    resp = client.post(
        f"/api/agents/{agent_id}/run/stream",
        json={"message": probe, "citations_enabled": True},
        headers=auth_headers,
        buffered=True,
    )
    assert resp.status_code == 200

    events = _parse_sse_events(resp.get_data())

    complete_event = next(e for e in events if e.get("event") == "complete")

    # Citations field should be absent or empty — no [N] markers means no citations
    assert complete_event.get("citations", []) == [], (
        f"expected empty/absent citations, got: {complete_event.get('citations')}"
    )

    # ai.message_citations should have 0 rows for this run
    start_event = next(e for e in events if e.get("event") == "start")
    run_id = start_event["run_id"]
    with app.app_context():
        rows = db.session.execute(
            text(
                'SELECT citation_key FROM "ai".message_citations '
                'WHERE run_id = (SELECT id FROM "ai".agent_runs WHERE run_id = :rid)'
            ),
            {"rid": run_id},
        ).fetchall()
    assert rows == [], f"expected no message_citations rows, got: {rows}"

    # The original content without markers should be intact
    assert "[1]" not in complete_event["content"]
    assert "[2]" not in complete_event["content"]
    assert "Provision 1" in complete_event["content"]
    assert complete_event["content"]  # non-empty


def test_run_agent_stream_strips_invalid_marker(
    client, app, db_cleanup, mock_auth, auth_headers, mocker, monkeypatch
):
    """When the LLM emits [N] markers exceeding the citation_map size, the
    platform must strip them from content and not persist phantom citation
    rows. Only valid keys (those mapped to real chunks) survive."""
    monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "false")

    chunk_1_id = "44444444-aaaa-bbbb-cccc-444444444444"
    chunk_2_id = "55555555-aaaa-bbbb-cccc-555555555555"
    handler_id = "66666666-aaaa-bbbb-cccc-666666666666"

    agent_id = _create_agent_with_tool(app, "InvalidMarkerTestAgent")

    # Insert stub context_handlers row to satisfy FK
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".context_handlers (id, query, knowledge_base_configs)
                VALUES (:id, 'test query', '[]')
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"id": handler_id},
        )
        db.session.commit()

    def fake_get_context_handler(db_session, hid, resolve_text=False):
        # citation_map will have valid keys 1 and 2 only (no key 999)
        return {
            "retrieved_context": [
                {"_type": "retrieval_diagnostics", "total_items": 2},
                {"id": chunk_1_id, "text": "Provision 1 text", "kb_name": "kb"},
                {"id": chunk_2_id, "text": "Provision 2 text", "kb_name": "kb"},
            ]
        }

    mocker.patch(
        "agentic_project_service.routes.agents.get_context_handler",
        side_effect=fake_get_context_handler,
    )

    probe = f"invalid-marker-probe-{uuid.uuid4().hex[:8]}"

    def fake_run(messages, *, context=None, tools=None, **kwargs):
        context.emit_event({"type": "context_handler_created", "context_handler_id": handler_id})
        out = _make_fake_agent_output(
            content=f"Provision 1 says X [1]. Y [999] is invalid. Provision 2 says Z [2]. {probe}"
        )
        out.reasoning_artifact = None
        out.reasoning_requested = False
        return out

    mocker.patch("agentic.agent.agent.Agent.run", side_effect=fake_run)

    resp = client.post(
        f"/api/agents/{agent_id}/run/stream",
        json={"message": probe, "citations_enabled": True},
        headers=auth_headers,
        buffered=True,
    )
    assert resp.status_code == 200

    events = _parse_sse_events(resp.get_data())

    start_event = next(e for e in events if e.get("event") == "start")
    run_id = start_event["run_id"]
    complete_event = next(e for e in events if e.get("event") == "complete")

    content = complete_event["content"]
    # Valid markers preserved
    assert "[1]" in content, f"valid [1] missing from content: {content[:200]}"
    assert "[2]" in content, f"valid [2] missing from content: {content[:200]}"
    # Invalid marker stripped
    assert "[999]" not in content, f"invalid [999] should have been stripped: {content[:200]}"

    # Only 2 citations in complete event (keys 1 and 2, no 999)
    citations = complete_event.get("citations") or []
    assert len(citations) == 2, f"expected 2 citations, got: {citations}"
    keys_in_event = sorted(int(c["key"]) for c in citations)
    assert keys_in_event == [1, 2], f"expected keys [1, 2], got: {keys_in_event}"

    # Only 2 message_citations rows persisted — no phantom row for 999
    with app.app_context():
        rows = db.session.execute(
            text(
                'SELECT citation_key FROM "ai".message_citations '
                'WHERE run_id = (SELECT id FROM "ai".agent_runs WHERE run_id = :rid) '
                "ORDER BY citation_key"
            ),
            {"rid": run_id},
        ).fetchall()
    assert [r[0] for r in rows] == [1, 2], f"expected [1, 2], got: {[r[0] for r in rows]}"

    # Cleanup: db_cleanup doesn't truncate message_citations
    with app.app_context():
        db.session.execute(
            text(
                'DELETE FROM "ai".message_citations WHERE run_id IN '
                '(SELECT id FROM "ai".agent_runs WHERE run_id = :rid)'
            ),
            {"rid": run_id},
        )
        db.session.commit()


def test_run_agent_stream_tool_based_injects_citation_instruction_when_enabled(
    client, app, db_cleanup, mock_auth, auth_headers, mocker, monkeypatch
):
    """When citations_enabled=true AND the agent has tools, the platform must
    append the citation instruction to the system prompt so the LLM emits
    [N] markers that Patch 2 can parse."""
    monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "false")

    handler_id = "dddddddd-aaaa-bbbb-cccc-dddddddddddd"
    agent_id = _create_agent_with_tool(app, "CitationInstructionAgent")
    with app.app_context():
        db.session.execute(
            text(
                'INSERT INTO "ai".context_handlers (id, query, knowledge_base_configs) '
                "VALUES (:id, 'q', '[]') ON CONFLICT (id) DO NOTHING"
            ),
            {"id": handler_id},
        )
        db.session.commit()

    mocker.patch(
        "agentic_project_service.routes.agents.get_context_handler",
        return_value={
            "retrieved_context": [
                {"_type": "retrieval_diagnostics", "total_items": 1},
                {"id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee", "text": "x", "kb_name": "kb"},
            ]
        },
    )

    captured = {}

    def fake_run(messages, *, context=None, tools=None, **kwargs):
        context.emit_event({"type": "context_handler_created", "context_handler_id": handler_id})
        out = _make_fake_agent_output(content="answer [1]")
        out.reasoning_artifact = None
        out.reasoning_requested = False
        return out

    fake_agent = mocker.MagicMock()
    fake_agent.run.side_effect = fake_run

    def capture_agent(**kwargs):
        captured["system_prompt"] = kwargs.get("system_prompt", "")
        return fake_agent

    mocker.patch("agentic_project_service.routes.agents.Agent", side_effect=capture_agent)

    resp = client.post(
        f"/api/agents/{agent_id}/run/stream",
        json={"message": "test", "citations_enabled": True},
        headers=auth_headers,
        buffered=True,
    )
    assert resp.status_code == 200

    assert "include citations in brackets like [1], [2]" in captured["system_prompt"], (
        f"citation instruction missing from system prompt:\n{captured['system_prompt']}"
    )


def test_run_agent_stream_tool_based_no_citation_instruction_when_disabled(
    client, app, db_cleanup, mock_auth, auth_headers, mocker, monkeypatch
):
    """When citations_enabled is false/omitted, no citation instruction is appended
    even if the agent has tools — preserves the pre-Patch-3 behavior for callers
    that don't want structured citations."""
    monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "false")

    agent_id = _create_agent_with_tool(app, "NoCitationInstructionAgent")

    captured = {}

    def fake_run(messages, *, context=None, tools=None, **kwargs):
        out = _make_fake_agent_output(content="plain answer")
        out.reasoning_artifact = None
        out.reasoning_requested = False
        return out

    fake_agent = mocker.MagicMock()
    fake_agent.run.side_effect = fake_run

    def capture_agent(**kwargs):
        captured["system_prompt"] = kwargs.get("system_prompt", "")
        return fake_agent

    mocker.patch("agentic_project_service.routes.agents.Agent", side_effect=capture_agent)

    resp = client.post(
        f"/api/agents/{agent_id}/run/stream",
        json={"message": "test"},  # no citations_enabled
        headers=auth_headers,
        buffered=True,
    )
    assert resp.status_code == 200

    assert "include citations in brackets" not in captured["system_prompt"], (
        f"citation instruction unexpectedly present:\n{captured['system_prompt']}"
    )
