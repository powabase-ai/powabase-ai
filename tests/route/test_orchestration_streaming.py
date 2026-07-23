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

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None, hooks=None):
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

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None, hooks=None):
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

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None, hooks=None):
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

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None, hooks=None):
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

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None, hooks=None):
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

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None, hooks=None):
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

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None, hooks=None):
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


class TestPreResponseReconciliation:
    """A PreResponse hook modification must persist AND reach the complete event
    under streaming (stream-then-correct), even though the raw answer streamed."""

    def test_preresponse_modification_persists_under_streaming(
        self, client, app, mock_auth, auth_headers, orch_with_agent, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None, hooks=None):
            # Supervisor streams the raw, un-redacted answer...
            context.emit_event({"type": "content_delta", "delta": "RAW ANSWER"})
            # ...then a PreResponse hook modifies it (as agent.py:_emit_hook_executions would).
            context.emit_event(
                {
                    "type": "hook_result",
                    "hook_id": "h1",
                    "hook_event": "PreResponse",
                    "status": "succeeded",
                    "modified": True,
                    "blocked": False,
                    "latency_ms": 5,
                    "message": None,
                }
            )
            return _make_completed_output(context.execution_id, "REDACTED")

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
        complete = [e for e in events if e.get("event") == "complete"]
        assert len(complete) == 1
        assert complete[0]["content"] == "REDACTED"

        # C1: the hook_result frame must be labeled `hook_result` on the wire,
        # not clobbered to the hook's lifecycle event name.
        hook_frames = [e for e in events if e.get("type") == "hook_result"]
        assert len(hook_frames) == 1
        assert hook_frames[0]["event"] == "hook_result"
        assert hook_frames[0]["hook_event"] == "PreResponse"

        with app.app_context():
            row = db.session.execute(
                text('SELECT content FROM "ai".orchestration_runs WHERE orchestration_id = :oid'),
                {"oid": orch_with_agent},
            ).fetchone()
            assert row is not None
            assert row[0] == "REDACTED"  # redacted answer persisted
            assert row[0] != "RAW ANSWER"  # pre-edit answer NOT persisted


class TestEmptyRedactionReachesTheWire:
    """R6-C1 layer 2: a full redaction (`modified_output: ""`) must be emitted.

    Round 5 fixed `hooks.py` so an empty-string redaction propagates into
    `output.content`. The route then dropped it again: `if final_content:` is
    falsy for `""`, so the terminal correction chunk was never sent. A consumer
    reading the SSE stream (the downstream backend this feature exists for) sees
    the raw streamed answer and no correction — while the DB row and the audit
    record both say the answer was redacted.
    """

    def test_empty_redaction_emits_terminal_chunk(
        self, client, app, mock_auth, auth_headers, orch_with_agent, monkeypatch
    ):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None, hooks=None):
            context.emit_event({"type": "content_delta", "delta": "The SSN is 123-45-6789."})
            context.emit_event(
                {
                    "type": "hook_result",
                    "hook_id": "h1",
                    "hook_event": "PreResponse",
                    "status": "succeeded",
                    "modified": True,
                    "blocked": False,
                    "latency_ms": 5,
                    "message": None,
                }
            )
            # Hook withheld the answer entirely.
            return _make_completed_output(context.execution_id, "")

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
        chunks = [e for e in events if e.get("event") == "chunk"]
        assert len(chunks) == 1, (
            "The terminal correction chunk was not emitted for an empty "
            "redaction, so a streaming consumer never learns the answer was "
            f"withheld. Events seen: {[e.get('event') or e.get('type') for e in events]}"
        )
        assert chunks[0]["content"] == ""

        complete = [e for e in events if e.get("event") == "complete"]
        assert complete[0]["content"] == ""

        with app.app_context():
            row = db.session.execute(
                text('SELECT content FROM "ai".orchestration_runs WHERE orchestration_id = :oid'),
                {"oid": orch_with_agent},
            ).fetchone()
            assert row[0] == ""
            assert "123-45-6789" not in (row[0] or "")


class TestHooksReachTheEngine:
    """R6-C2: the route must actually hand the DB's hooks to the engine.

    `test_supervisor_hooks.py` proves `Orchestration.run(hooks=...)` forwards
    correctly, but constructs the Orchestration directly. The streaming route
    tests patch `Orchestration.run` wholesale. Both sides are verified against a
    mock; nothing observed the seam — so deleting the route's `hooks=` argument
    left the entire feature dead with the suite green.
    """

    def test_configured_hooks_are_passed_to_orchestration_run(
        self, client, app, mock_auth, auth_headers, orch_with_agent
    ):
        hook_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".hooks
                        (id, orchestration_id, event, type, config, enabled, position)
                    VALUES (:id, :oid, 'PreResponse', 'http',
                            '{"url": "https://vet.example/hook"}', true, 0)
                    """
                ),
                {"id": hook_id, "oid": orch_with_agent},
            )
            db.session.commit()

        captured = {}

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None, hooks=None):
            captured["hooks"] = hooks
            return _make_completed_output(context.execution_id, "done")

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

        assert captured.get("hooks"), (
            "The orchestration's configured hooks never reached the engine — "
            "every supervisor hook would silently never fire while CRUD, the "
            "list endpoint and the UI all still show them as active."
        )
        assert len(captured["hooks"]) == 1
        passed = captured["hooks"][0]
        assert str(passed.id) == hook_id
        assert passed.event == "PreResponse"
        assert passed.type == "http"
        assert passed.config["url"] == "https://vet.example/hook"

    def test_orchestration_without_hooks_passes_none(
        self, client, app, mock_auth, auth_headers, orch_with_agent
    ):
        """Control: no rows configured must not fabricate an empty hook list."""
        captured = {}

        def fake_run(input, context=None, *, history=None, on_delegate_complete=None, hooks=None):
            captured["hooks"] = hooks
            return _make_completed_output(context.execution_id, "done")

        with patch(
            "agentic.orchestration.orchestration.Orchestration.run",
            side_effect=fake_run,
        ):
            client.post(
                f"/api/orchestrations/{orch_with_agent}/run/stream",
                json={"message": "go"},
                headers=auth_headers,
                buffered=True,
            )
        assert captured["hooks"] is None
