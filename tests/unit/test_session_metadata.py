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


@pytest.mark.parametrize(
    "bad",
    [
        {"n": float("nan")},
        {"n": float("inf")},
        {"s": "a\u0000b"},
        {"a\u0000": 1},
        {"s": "\ud800"},
        {"\udfff": 1},
    ],
)
def test_metadata_postgres_jsonb_cannot_store_is_rejected(bad):
    """NaN/Infinity (Flask's parser accepts them), NUL characters and lone
    surrogates would make the jsonb INSERT fail with a 500."""
    assert parse_session_metadata({"metadata": bad}) == (
        None,
        "metadata must be JSON without NaN, Infinity, NUL characters or lone surrogates",
    )


def test_a_lone_surrogate_in_a_request_body_is_400():
    resp, create = _create(None, raw='{"metadata": {"s": "\\ud800"}}')
    assert resp.status_code == 400
    create.assert_not_called()


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


def _create(body, auth=None, raw=None):
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
        if raw is not None:
            resp = c.post(
                f"/api/agents/{AGENT_ID}/sessions",
                data=raw,
                content_type="application/json",
                headers=HEADERS,
            )
        else:
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


# --- orchestration sessions ---

ORCH_ID = "4a8b2d3f-6c8e-4f22-8b4d-9e3a7c5f2b81"


def test_orchestration_session_service_stores_metadata(monkeypatch):
    from agentic_project_service.services import orchestration as orch_service

    made = {}

    class _Model:
        query = MagicMock()

        def __init__(self, **kwargs):
            made.update(kwargs)
            self.id = "db-uuid"
            self.session_id = kwargs["session_id"]

    monkeypatch.setattr(orch_service, "OrchestrationSessionModel", _Model)
    monkeypatch.setattr(orch_service, "db", MagicMock())
    _, session_id, is_new = orch_service.get_or_create_orchestration_session(
        ORCH_ID, user_id=USER_ID, metadata={"conversation_id": "c1"}
    )
    assert is_new and session_id.startswith("orch_sess_")
    assert made["metadata_"] == {"conversation_id": "c1"}
    assert made["user_id"] == USER_ID

    made.clear()
    orch_service.get_or_create_orchestration_session(ORCH_ID)
    assert made["metadata_"] == {}


def _orch_create(
    body=None, raw=None, orch_exists=True, auth=None, commit_error=None, create_error=None
):
    from agentic_project_service.routes import orchestrations as orchestrations_route
    from agentic_project_service.services import orchestration as orch_service

    fake_session = MagicMock()
    fake_session.get.return_value = object() if orch_exists else None
    if commit_error is not None:
        fake_session.commit.side_effect = commit_error
    kwargs = {"headers": HEADERS}
    if raw is not None:
        kwargs.update(data=raw, content_type="application/json")
    elif body is not None:
        kwargs["json"] = body
    with (
        patch.object(
            orch_service,
            "get_or_create_orchestration_session",
            return_value=("u", "orch_sess_1", True),
            side_effect=create_error,
        ) as create,
        patch.object(orchestrations_route.db, "session", fake_session),
        auth or _service(),
        _app(orchestrations_route.orchestrations_bp).test_client() as c,
    ):
        resp = c.post(f"/api/orchestrations/{ORCH_ID}/sessions", **kwargs)
    return resp, create, fake_session


def test_orchestration_session_create_route():
    resp, create, fake_session = _orch_create(
        {"user_id": USER_ID.upper(), "metadata": {"conversation_id": "c1"}}
    )
    assert resp.status_code == 201
    assert resp.get_json() == {"session_id": "orch_sess_1"}
    create.assert_called_once_with(ORCH_ID, user_id=USER_ID, metadata={"conversation_id": "c1"})
    fake_session.commit.assert_called_once()


def test_orchestration_session_create_with_no_body_is_ownerless():
    resp, create, _ = _orch_create()
    assert resp.status_code == 201
    create.assert_called_once_with(ORCH_ID, user_id=None, metadata=None)


def test_orchestration_session_create_unknown_orchestration_is_404():
    resp, create, _ = _orch_create({}, orch_exists=False)
    assert resp.status_code == 404
    create.assert_not_called()


def test_orchestration_session_create_malformed_id_is_404():
    from agentic_project_service.routes import orchestrations as orchestrations_route

    with _service(), _app(orchestrations_route.orchestrations_bp).test_client() as c:
        resp = c.post("/api/orchestrations/not-a-uuid/sessions", json={}, headers=HEADERS)
    assert resp.status_code == 404


@pytest.mark.parametrize("where", ["flush", "commit"])
def test_orchestration_session_create_race_with_a_delete_is_404(where):
    """The FK is not deferred, so a concurrent delete fails the INSERT at the
    service's flush; a commit-time failure is handled the same way."""
    from sqlalchemy.exc import IntegrityError

    error = IntegrityError("x", {}, Exception())
    if where == "flush":
        resp, _, fake_session = _orch_create({}, create_error=error)
    else:
        resp, _, fake_session = _orch_create({}, commit_error=error)
    assert resp.status_code == 404
    assert resp.get_json() == {"error": "Orchestration not found"}
    fake_session.rollback.assert_called_once()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"body": {"metadata": "x"}}, "metadata must be a JSON object"),
        ({"body": {"user_id": "nope"}}, "user_id must be a uuid"),
        ({"body": ["x"]}, "Request body must be a JSON object"),
        ({"raw": "{not json"}, "Request body must be valid JSON"),
    ],
)
def test_orchestration_session_create_bad_body_is_400(kwargs, error):
    resp, create, _ = _orch_create(**kwargs)
    assert resp.status_code == 400
    assert resp.get_json() == {"error": error}
    create.assert_not_called()


def test_orchestration_session_create_is_service_role_only():
    resp, create, _ = _orch_create({}, auth=_user())
    assert resp.status_code == 403
    create.assert_not_called()


def test_orchestration_session_list_returns_metadata():
    from agentic_project_service.routes import orchestrations as orchestrations_route

    class _Q:
        def __init__(self, rows):
            self.rows = rows

        def filter_by(self, **_):
            return self

        def order_by(self, *_):
            return self

        def limit(self, _):
            return self

        def all(self):
            return self.rows

    sessions = [
        MagicMock(session_id="orch_sess_1", created_at=None, metadata_={"conversation_id": "c1"}),
        MagicMock(session_id="orch_sess_2", created_at=None, metadata_=None),
    ]
    sessions_model = MagicMock()
    sessions_model.query = _Q(sessions)
    runs_model = MagicMock()
    runs_model.query = _Q([])
    with (
        patch.object(orchestrations_route, "OrchestrationSessionModel", sessions_model),
        patch.object(orchestrations_route, "OrchestrationRunModel", runs_model),
        _service(),
        _app(orchestrations_route.orchestrations_bp).test_client() as c,
    ):
        resp = c.get(f"/api/orchestrations/{ORCH_ID}/sessions", headers=HEADERS)
    assert resp.status_code == 200
    listed = resp.get_json()["sessions"]
    assert listed[0]["metadata"] == {"conversation_id": "c1"}
    assert listed[1]["metadata"] == {}


def test_the_text_of_an_escape_is_fine():
    meta = {"s": "\\u0000 is just text here"}
    assert parse_session_metadata({"metadata": meta}) == (meta, None)


def test_agent_session_create_rejects_nan_metadata():
    resp, create = _create({"metadata": {"n": float("nan")}})
    assert resp.status_code == 400
    create.assert_not_called()
