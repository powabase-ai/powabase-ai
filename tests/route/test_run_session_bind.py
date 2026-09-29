"""An end user can never name a session into existence.

A backend's session ids can be guessable (one per phone number, say). If an
end user could create a session under such an id, their planted history would
be fed to the backend's later runs in it, and they could read what those runs
said. So on every run route, an end user naming an unknown ``session_id`` is
refused with 404 before any work, and nothing is created: end users start a
session by omitting the id, or with ``POST /api/agents/<id>/sessions``.
Without an id the run creates one, owned by the caller.
"""

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import text

from agentic_project_service.db import db
from tests.route.test_agent_streaming import _create_agent_with_tool, _make_fake_agent_output
from tests.route.test_orchestration_streaming import (
    _create_agent,
    _create_orchestration,
    _make_completed_output,
)

MALLORY = str(uuid.uuid4())


@pytest.fixture
def mallory(mocker, monkeypatch):
    mocker.patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": MALLORY, "role": "authenticated", "exp": 4102444800},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "agentic_project_service.routes.agents.check_model_available", lambda model: None
    )
    monkeypatch.setattr(
        "agentic_project_service.routes.orchestrations.check_model_available", lambda model: None
    )


HEADERS = {"Authorization": "Bearer mallory"}


def _sessions(app, table: str, session_id: str):
    with app.app_context():
        rows = db.session.execute(
            text(f'SELECT user_id FROM "ai".{table} WHERE session_id = :s'), {"s": session_id}
        ).fetchall()
    return [str(r[0]) if r[0] else None for r in rows]


class TestNamingAnUnknownSession:
    @pytest.mark.parametrize("suffix", ["run", "run/stream"])
    def test_agent_run_is_refused_and_creates_nothing(self, app, client, mallory, suffix):
        agent_id = _create_agent_with_tool(app, f"Squat-{suffix}")
        session_id = f"wa_{uuid.uuid4().int % 10**11}"
        with patch("agentic.agent.agent.Agent.run") as run:
            resp = client.post(
                f"/api/agents/{agent_id}/{suffix}",
                json={"message": "hi", "session_id": session_id},
                headers=HEADERS,
                buffered=True,
            )
        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Session not found"}
        run.assert_not_called()
        assert _sessions(app, "agent_sessions", session_id) == []

    def test_orchestration_run_is_refused_and_creates_nothing(self, app, client, mallory):
        orch_id = _create_orchestration(app, "supervisor", agent_ids=[_create_agent(app, "S")])
        session_id = f"wa_{uuid.uuid4().int % 10**11}"
        with patch("agentic.orchestration.orchestration.Orchestration.run") as run:
            resp = client.post(
                f"/api/orchestrations/{orch_id}/run/stream",
                json={"message": "hi", "session_id": session_id},
                headers=HEADERS,
                buffered=True,
            )
        assert resp.status_code == 404
        run.assert_not_called()
        assert _sessions(app, "orchestration_sessions", session_id) == []


class TestOmittingTheId:
    def test_an_agent_run_creates_a_session_owned_by_the_caller(self, app, client, mallory):
        agent_id = _create_agent_with_tool(app, "NewSession")
        with patch(
            "agentic.agent.agent.Agent.run",
            side_effect=lambda messages, **kw: _make_fake_agent_output("ok"),
        ):
            resp = client.post(
                f"/api/agents/{agent_id}/run", json={"message": "hi"}, headers=HEADERS
            )
        assert resp.status_code == 200, resp.get_data(as_text=True)
        session_id = resp.get_json()["session_id"]
        assert _sessions(app, "agent_sessions", session_id) == [MALLORY]

    def test_an_orchestration_run_creates_a_session_owned_by_the_caller(self, app, client, mallory):
        orch_id = _create_orchestration(app, "supervisor", agent_ids=[_create_agent(app, "T")])
        with patch(
            "agentic.orchestration.orchestration.Orchestration.run",
            side_effect=lambda input, context=None, **kw: _make_completed_output(
                context.execution_id
            ),
        ):
            resp = client.post(
                f"/api/orchestrations/{orch_id}/run/stream",
                json={"message": "hi"},
                headers=HEADERS,
                buffered=True,
            )
        assert resp.status_code == 200
        with app.app_context():
            owners = (
                db.session.execute(
                    text(
                        'SELECT user_id FROM "ai".orchestration_sessions '
                        "WHERE orchestration_id = :o"
                    ),
                    {"o": orch_id},
                )
                .scalars()
                .all()
            )
        assert [str(u) for u in owners] == [MALLORY]
