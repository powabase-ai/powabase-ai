"""Session metadata: a caller's own ids (e.g. a conversation id) on a session."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from agentic_project_service.routes import agents as agents_route
from agentic_project_service.routes._session_metadata import (
    MAX_SESSION_METADATA_BYTES,
    parse_session_metadata,
)

AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"
USER_ID = "11111111-1111-4111-8111-111111111111"
HEADERS = {"Authorization": "Bearer fake"}


def _service():
    return patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "service", "role": "service_role", "is_service_role": True},
    )


def _user(user_id=USER_ID):
    return patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": user_id, "role": "authenticated", "aud": "authenticated"},
    )


def _app(bp):
    app = Flask(__name__)
    app.register_blueprint(bp)
    return app


def test_parse_session_metadata():
    assert parse_session_metadata({}) == (None, None)
    assert parse_session_metadata({"metadata": None}) == (None, None)
    assert parse_session_metadata({"metadata": {}}) == ({}, None)
    assert parse_session_metadata({"metadata": {"conversation_id": "c1"}}) == (
        {"conversation_id": "c1"},
        None,
    )
    assert parse_session_metadata({"metadata": ["x"]}) == (None, "metadata must be a JSON object")
    assert parse_session_metadata({"metadata": "x"}) == (None, "metadata must be a JSON object")


def test_metadata_size_limit_is_on_serialised_bytes():
    at_limit = {"k": "x" * (MAX_SESSION_METADATA_BYTES - len('{"k":""}'))}
    assert parse_session_metadata({"metadata": at_limit}) == (at_limit, None)
    over = {"k": "x" * MAX_SESSION_METADATA_BYTES}
    assert parse_session_metadata({"metadata": over}) == (
        None,
        "metadata is limited to 16384 bytes of JSON",
    )
    # Non-ASCII counts by its UTF-8 size, not by characters.
    wide = {"k": "é" * (MAX_SESSION_METADATA_BYTES // 2)}
    assert parse_session_metadata({"metadata": wide})[1] is not None


def _create(body, auth=None):
    fake_session = MagicMock()
    fake_session.execute.return_value.fetchone.return_value = ("gpt-4o-mini",)
    with (
        patch.object(
            agents_route, "get_or_create_session", return_value=("u", "sess_1", True)
        ) as create,
        patch.object(agents_route.db, "session", fake_session),
        auth or _service(),
        _app(agents_route.agents_bp).test_client() as c,
    ):
        resp = c.post(f"/api/agents/{AGENT_ID}/sessions", json=body, headers=HEADERS)
    return resp, create


def test_agent_session_create_passes_metadata():
    resp, create = _create({"metadata": {"conversation_id": "c1"}})
    assert resp.status_code == 201
    assert create.call_args.kwargs["metadata"] == {"conversation_id": "c1"}


def test_an_end_user_may_set_metadata_on_their_own_session():
    resp, create = _create({"metadata": {"conversation_id": "c1"}}, auth=_user())
    assert resp.status_code == 201
    assert create.call_args.kwargs["user_id"] == USER_ID
    assert create.call_args.kwargs["metadata"] == {"conversation_id": "c1"}


def test_agent_session_create_without_metadata_passes_none_of_it():
    resp, create = _create({})
    assert resp.status_code == 201
    assert "metadata" not in create.call_args.kwargs


@pytest.mark.parametrize("bad", ["x", ["x"], 3])
def test_agent_session_create_rejects_bad_metadata(bad):
    resp, create = _create({"metadata": bad})
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "metadata must be a JSON object"}
    create.assert_not_called()


def test_agent_session_create_rejects_oversized_metadata():
    resp, create = _create({"metadata": {"k": "x" * MAX_SESSION_METADATA_BYTES}})
    assert resp.status_code == 400
    create.assert_not_called()


def test_agent_session_list_returns_metadata():
    now = datetime(2026, 11, 19, tzinfo=UTC)
    rows = [
        ("sess_1", 2, now, now, "hello", {"conversation_id": "c1"}),
        ("sess_2", 0, None, now, None, None),
    ]
    count = MagicMock()
    count.scalar.return_value = 2
    fake_session = MagicMock()
    fake_session.execute.side_effect = [rows, count]
    with (
        patch.object(agents_route.db, "session", fake_session),
        _service(),
        _app(agents_route.agents_bp).test_client() as c,
    ):
        resp = c.get(f"/api/agents/{AGENT_ID}/sessions", headers=HEADERS)
    assert resp.status_code == 200
    sessions = resp.get_json()["sessions"]
    assert sessions[0]["metadata"] == {"conversation_id": "c1"}
    assert sessions[1]["metadata"] == {}
    assert "s.metadata" in str(fake_session.execute.call_args_list[0].args[0])
