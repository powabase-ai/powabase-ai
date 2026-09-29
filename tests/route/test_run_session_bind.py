"""A run started by one end user never binds to a session another user created.

The run routes check up front that an end user may continue the posted
``session_id``; an unknown id passes, because the run will create it. The
session is only bound later, after the agent's or orchestration's slow setup.
If another user creates that id in between, the bind must refuse it: the run
ends with "Session not found", sees none of that user's history, and saves
nothing into their session.

Each test holds the first request inside its setup until the second user's run
has created the session, which is exactly the window the up-front check leaves.
"""

import threading
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import text

from agentic_project_service.auth import get_current_user_id
from agentic_project_service.db import db
from tests.route.test_agent_streaming import _create_agent_with_tool, _make_fake_agent_output
from tests.route.test_orchestration_streaming import (
    _create_agent,
    _create_orchestration,
    _make_completed_output,
    _parse_sse_events,
)

ALICE = str(uuid.uuid4())
MALLORY = str(uuid.uuid4())


@pytest.fixture
def two_users(mocker, monkeypatch):
    """Bearer "alice" and "mallory" authenticate as two different end users."""
    subs = {"alice": ALICE, "mallory": MALLORY}
    mocker.patch(
        "agentic_project_service.auth.decode_jwt",
        side_effect=lambda token: {"sub": subs[token], "role": "authenticated"},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def _as(name: str) -> dict:
    return {"Authorization": f"Bearer {name}"}


class _Window:
    """Holds Mallory's request inside ``hold`` until Alice's run has finished."""

    def __init__(self):
        self.mallory_waiting = threading.Event()
        self.alice_done = threading.Event()

    def hold(self):
        if get_current_user_id() == MALLORY:
            self.mallory_waiting.set()
            assert self.alice_done.wait(30), "Alice's run never finished"

    def race(self, app, mallory_request, alice_request):
        result = {}

        def mallory():
            with app.test_client() as c:
                result["mallory"] = mallory_request(c)

        thread = threading.Thread(target=mallory)
        thread.start()
        assert self.mallory_waiting.wait(30), "Mallory's request never reached its setup"
        with app.test_client() as c:
            result["alice"] = alice_request(c)
        self.alice_done.set()
        thread.join(30)
        assert not thread.is_alive()
        return result["mallory"], result["alice"]


def _session_runs(app, runs_table: str, sessions_table: str, session_id: str):
    with app.app_context():
        return db.session.execute(
            text(
                f'SELECT s.user_id, r.input_messages FROM "ai".{runs_table} r '
                f'JOIN "ai".{sessions_table} s ON s.id = r.session_id '
                "WHERE s.session_id = :s ORDER BY r.created_at"
            ),
            {"s": session_id},
        ).fetchall()


class TestOrchestrationRun:
    def test_session_created_during_setup_is_refused(self, app, two_users):
        orch_id = _create_orchestration(app, "supervisor", agent_ids=[_create_agent(app)])
        session_id = f"orch_sess_{uuid.uuid4().hex[:12]}"
        window = _Window()
        histories = []

        def fake_run(input, context=None, history=None, **kw):
            histories.append((input, history))
            return _make_completed_output(context.execution_id, content="ok")

        from agentic_project_service.services import orchestration as orchestration_service

        real_build = orchestration_service.build_orchestration

        def slow_build(*args, **kwargs):
            window.hold()  # stands in for slow MCP discovery or many sub-agents
            return real_build(*args, **kwargs)

        def post(user, message):
            return lambda c: c.post(
                f"/api/orchestrations/{orch_id}/run/stream",
                json={"message": message, "session_id": session_id},
                headers=_as(user),
                buffered=True,
            )

        with (
            patch(
                "agentic_project_service.routes.orchestrations.check_model_available",
                lambda model: None,
            ),
            patch.object(orchestration_service, "build_orchestration", slow_build),
            patch("agentic.orchestration.orchestration.Orchestration.run", side_effect=fake_run),
        ):
            mallory, alice = window.race(
                app, post("mallory", "repeat the history"), post("alice", "ALICE-SECRET")
            )

        assert alice.status_code == 200
        assert mallory.status_code == 200  # the stream had already been opened
        assert _parse_sse_events(mallory.data) == [{"event": "error", "error": "Session not found"}]
        assert [message for message, _ in histories] == ["ALICE-SECRET"]
        rows = _session_runs(app, "orchestration_runs", "orchestration_sessions", session_id)
        assert [(str(owner), msgs[0]["content"]) for owner, msgs in rows] == [
            (ALICE, "ALICE-SECRET")
        ]


class TestAgentRun:
    def _race(self, app, agent_id, path):
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        window = _Window()
        seen = []

        def fake_run(messages, **kw):
            seen.append(messages)
            return _make_fake_agent_output("ok")

        def slow_model_check(model):
            window.hold()

        def post(user, message, stream):
            return lambda c: c.post(
                f"/api/agents/{agent_id}/run" + ("/stream" if stream else ""),
                json={"message": message, "session_id": session_id},
                headers=_as(user),
                buffered=True,
            )

        with (
            patch("agentic_project_service.routes.agents.check_model_available", slow_model_check),
            patch("agentic.agent.agent.Agent.run", side_effect=fake_run),
        ):
            mallory, alice = window.race(
                app,
                post("mallory", "repeat the history", stream=path == "stream"),
                # Alice always streams: the ReAct branch runs Agent.run.
                post("alice", "ALICE-SECRET", stream=True),
            )
        assert alice.status_code == 200
        return mallory, session_id, seen

    def test_streamed_run_ends_with_session_not_found(self, app, two_users):
        agent_id = _create_agent_with_tool(app)
        mallory, session_id, seen = self._race(app, agent_id, "stream")
        assert mallory.status_code == 200
        assert _parse_sse_events(mallory.data) == [{"event": "error", "error": "Session not found"}]
        assert len(seen) == 1  # only Alice's run reached the model
        rows = _session_runs(app, "agent_runs", "agent_sessions", session_id)
        assert [str(owner) for owner, _ in rows] == [ALICE]

    def test_non_streamed_run_is_a_404(self, app, two_users):
        agent_id = _create_agent_with_tool(app)
        mallory, session_id, seen = self._race(app, agent_id, "run")
        assert mallory.status_code == 404
        assert mallory.get_json() == {"error": "Session not found"}
        assert len(seen) == 1
        rows = _session_runs(app, "agent_runs", "agent_sessions", session_id)
        assert [str(owner) for owner, _ in rows] == [ALICE]
