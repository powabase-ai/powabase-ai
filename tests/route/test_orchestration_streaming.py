"""Route-level tests for orchestration SSE streaming endpoint (issue #106 / Task 13).

Covers β buffer + persistence filter + complete-event content + failure-path
events persistence in ``orchestrations.py:run_orchestration_stream``.

Tests patch ``agentic.orchestration.orchestration.Orchestration.run`` to drive
the route's worker thread without touching a real LLM. The patched function
emits events via ``context.emit_event`` (which puts them on the route's
``event_queue``), then returns an ``OrchestrationOutput``.
"""

import json
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import text

from agentic_project_service.db import db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_agent(app, name="Specialist"):
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


def _create_orchestration(app, strategy="supervisor", agent_ids=None):
    orch_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "ai".orchestrations
                    (id, name, description, strategy, orchestrator_config, settings)
                VALUES (:id, 'test-orch', 'desc', :strategy, '{}', '{}')
                """
            ),
            {"id": orch_id, "strategy": strategy},
        )
        for i, aid in enumerate(agent_ids or []):
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".orchestration_entities
                        (id, orchestration_id, entity_type, entity_ref_id,
                         role_description, position, config)
                    VALUES (:eid, :orch, 'agent', :aid, 'role', :pos, '{}')
                    """
                ),
                {
                    "eid": str(uuid.uuid4()),
                    "orch": orch_id,
                    "aid": aid,
                    "pos": i,
                },
            )
        db.session.commit()
    return orch_id


def _parse_sse_events(body: bytes) -> list[dict]:
    """Parse SSE response body into a list of decoded JSON event dicts.

    Skips comment-only frames (``: keepalive``).
    """
    events: list[dict] = []
    text = body.decode()
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line[len("data: ") :]))
    return events


def _has_keepalive_comment(body: bytes) -> bool:
    return b": keepalive" in body


def _make_completed_output(execution_id: str, content: str = "last-step content"):
    """Build a minimal OrchestrationOutput marked COMPLETED."""
    from agentic.execution.status import ExecutionStatus
    from agentic.orchestration.output import OrchestrationOutput

    out = OrchestrationOutput(execution_id=execution_id)
    out.status = ExecutionStatus.COMPLETED
    out.content = content
    out.steps = 1
    out.usage = {"total_tokens": 5}
    out.events = []
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orch_with_agent(app):
    agent_id = _create_agent(app, "Alpha")
    orch_id = _create_orchestration(app, "supervisor", agent_ids=[agent_id])
    return orch_id


# ---------------------------------------------------------------------------
# MUST-HAVE tests
# ---------------------------------------------------------------------------


class TestPersistenceFilter:
    """content_delta and reasoning_delta MUST NOT land in events_for_db."""

    def test_persistence_filter_excludes_deltas(
        self, client, app, mock_auth, auth_headers, orch_with_agent, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None):
            # Emit a mix of deltas and a terminal event
            context.emit_event({"type": "content_delta", "delta": "Hello "})
            context.emit_event({"type": "content_delta", "delta": "world"})
            context.emit_event({"type": "reasoning_delta", "delta": "thinking..."})
            context.emit_event({"type": "tool_call", "name": "search"})
            return _make_completed_output(context.execution_id, "Hello world")

        with patch(
            "agentic.orchestration.orchestration.Orchestration.run",
            side_effect=fake_run,
        ):
            resp = client.post(
                f"/api/orchestrations/{orch_with_agent}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        # The events JSONB column must NOT contain content_delta or reasoning_delta
        with app.app_context():
            row = db.session.execute(
                text('SELECT events FROM "ai".orchestration_runs WHERE orchestration_id = :oid'),
                {"oid": orch_with_agent},
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
        self, client, app, mock_auth, auth_headers, orch_with_agent, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        deltas = ["Step1: ", "checking ", "data. ", "Step2: ", "answer."]

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None):
            for d in deltas:
                context.emit_event({"type": "content_delta", "delta": d})
            # Last-step content differs from the buffer (intermediary prose was
            # streamed but output.content only carries the last step). The
            # terminal `chunk` MUST come from the buffer.
            return _make_completed_output(context.execution_id, "Step2: answer.")

        with patch(
            "agentic.orchestration.orchestration.Orchestration.run",
            side_effect=fake_run,
        ):
            resp = client.post(
                f"/api/orchestrations/{orch_with_agent}/run/stream",
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
        self, client, app, mock_auth, auth_headers, orch_with_agent, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "false")

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None):
            # Even if deltas were emitted, kill-switch off → buffer ignored,
            # chunk falls back to output.content.
            context.emit_event({"type": "content_delta", "delta": "ignored "})
            return _make_completed_output(context.execution_id, "final-output")

        with patch(
            "agentic.orchestration.orchestration.Orchestration.run",
            side_effect=fake_run,
        ):
            resp = client.post(
                f"/api/orchestrations/{orch_with_agent}/run/stream",
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
        self, client, app, mock_auth, auth_headers, orch_with_agent, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None):
            # Emit some terminal events before raising — these must reach the
            # DB via the failure-path update_orchestration_run call.
            context.emit_event({"type": "tool_call", "name": "search"})
            context.emit_event({"type": "chunk", "content": "partial-before-fail"})
            raise RuntimeError("synthetic mid-stream failure")

        with patch(
            "agentic.orchestration.orchestration.Orchestration.run",
            side_effect=fake_run,
        ):
            resp = client.post(
                f"/api/orchestrations/{orch_with_agent}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        # Run row should be marked failed AND have events persisted.
        with app.app_context():
            row = db.session.execute(
                text(
                    'SELECT status, events, error FROM "ai".orchestration_runs '
                    "WHERE orchestration_id = :oid"
                ),
                {"oid": orch_with_agent},
            ).fetchone()
            assert row is not None
            assert row[0] == "failed"
            persisted_events = row[1] or []
            persisted_types = {e.get("type") for e in persisted_events}
            assert "tool_call" in persisted_types
            assert "chunk" in persisted_types
            assert "synthetic mid-stream failure" in (row[2] or "")


# ---------------------------------------------------------------------------
# SHOULD-HAVE tests
# ---------------------------------------------------------------------------


class TestCompleteEventContent:
    """B1: complete.content MUST equal final_content (= buffer in streaming mode)."""

    def test_complete_event_content_matches_buffer_in_streaming_mode(
        self, client, app, mock_auth, auth_headers, orch_with_agent, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        deltas = ["A", "B", "C", "D"]

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None):
            for d in deltas:
                context.emit_event({"type": "content_delta", "delta": d})
            return _make_completed_output(context.execution_id, "D-only")

        with patch(
            "agentic.orchestration.orchestration.Orchestration.run",
            side_effect=fake_run,
        ):
            resp = client.post(
                f"/api/orchestrations/{orch_with_agent}/run/stream",
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
        self, client, app, mock_auth, auth_headers, orch_with_agent, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        deltas = ["multi-", "step-", "buffered"]

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None):
            for d in deltas:
                context.emit_event({"type": "content_delta", "delta": d})
            return _make_completed_output(context.execution_id, "buffered")

        with patch(
            "agentic.orchestration.orchestration.Orchestration.run",
            side_effect=fake_run,
        ):
            resp = client.post(
                f"/api/orchestrations/{orch_with_agent}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        with app.app_context():
            row = db.session.execute(
                text('SELECT content FROM "ai".orchestration_runs WHERE orchestration_id = :oid'),
                {"oid": orch_with_agent},
            ).fetchone()
            assert row is not None
            assert row[0] == "multi-step-buffered"


class TestKeepalivePreserved:
    """30s queue-empty timeout must still emit ``: keepalive``."""

    def test_keepalive_preserved(
        self, client, app, mock_auth, auth_headers, orch_with_agent, monkeypatch
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

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None):
            return _make_completed_output(context.execution_id, "ok")

        with (
            patch.object(queue_mod.Queue, "get", fake_get),
            patch(
                "agentic.orchestration.orchestration.Orchestration.run",
                side_effect=fake_run,
            ),
        ):
            resp = client.post(
                f"/api/orchestrations/{orch_with_agent}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
            assert resp.status_code == 200

        # The body should contain at least one ``: keepalive`` comment frame.
        assert _has_keepalive_comment(resp.get_data())
