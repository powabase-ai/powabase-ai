"""An agent without tools passes its max_tokens setting to the model, as an agent with tools already does.

Without it, a provider that reserves the whole possible output against the account balance (OpenRouter does) reserves
the model's maximum for every run of such an agent, even a short one.
"""

import json
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.routes import agents as agents_route
from tests.route.test_agent_streaming import _chat_stream_factory, _make_fake_agent_output


def _agent_without_tools(app, settings: dict) -> str:
    agent_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                'INSERT INTO "ai".agents (id, name, model, system_prompt, settings) '
                "VALUES (:id, :name, 'gpt-4o-mini', 'You are helpful.', CAST(:settings AS jsonb))"
            ),
            {"id": agent_id, "name": f"ChatAgent-{agent_id[:8]}", "settings": json.dumps(settings)},
        )
        db.session.commit()
    return agent_id


@pytest.fixture
def built(monkeypatch):
    """The keyword arguments of every Agent the route builds; the real Agent is still built."""
    seen: list[dict] = []
    real = agents_route.Agent

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(agents_route, "Agent", spy)
    return seen


@pytest.mark.parametrize(("settings", "expected"), [({"max_tokens": 4096}, 4096), ({}, None)])
def test_the_streaming_run_passes_max_tokens(
    client, app, mock_auth, auth_headers, built, monkeypatch, settings, expected
):
    monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
    agent_id = _agent_without_tools(app, settings)
    with patch("agentic.agent.agent.Agent.stream", _chat_stream_factory(["ok"])):
        resp = client.post(
            f"/api/agents/{agent_id}/run/stream",
            json={"message": "go"},
            headers=auth_headers,
            buffered=True,
        )
    assert resp.status_code == 200
    assert built and built[-1].get("max_tokens") == expected


@pytest.mark.parametrize(("settings", "expected"), [({"max_tokens": 4096}, 4096), ({}, None)])
def test_the_synchronous_run_passes_max_tokens(
    client, app, mock_auth, auth_headers, built, settings, expected
):
    agent_id = _agent_without_tools(app, settings)
    with patch("agentic.agent.agent.Agent.run", return_value=_make_fake_agent_output("ok")):
        resp = client.post(
            f"/api/agents/{agent_id}/run", json={"message": "go"}, headers=auth_headers
        )
    assert resp.status_code == 200
    assert built and built[-1].get("max_tokens") == expected
