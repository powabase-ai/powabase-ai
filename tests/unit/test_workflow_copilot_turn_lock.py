"""Turn-lock + trailing history window for the workflow copilot (routes/copilot.py).

Ports routes/project_copilot.py's protections to the workflow copilot:

1. A Redis SET-NX turn lock rejects a second concurrent turn (409) instead of
   letting two turns interleave 'user' rows and corrupt role alternation.
2. The history query is BOUNDED (trailing window) so an orphaned trailing
   'user' row — left by a pod eviction/OOM between the user-row commit and the
   daemon thread's assistant insert — eventually ages out instead of wedging
   the session forever.
3. The immediate turn tolerates such an orphan: the built input never contains
   a user,user sequence (which the upstream provider 400s on).

DB-free: mirrors TestSSEStreamingFlow in test_copilot_react.py (module-level
``db`` mock + decode_jwt patch), with a fake Redis standing in for the lock.
"""

import json
from unittest.mock import MagicMock, patch

_AUTH_HEADERS = {"Authorization": "Bearer fake-token"}


def _make_test_app():
    from flask import Flask

    from agentic_project_service.routes.copilot import copilot_bp

    app = Flask(__name__)
    app.register_blueprint(copilot_bp)
    return app


def _wire_mock_db(mock_db, history_rows):
    """Configure the module-level ``db`` mock: session lookup + history query."""
    session_row = MagicMock()
    session_row.__getitem__ = lambda self, idx: "wf-123"
    mock_exec = MagicMock()
    mock_exec.fetchone.return_value = session_row
    mock_exec.fetchall.return_value = history_rows
    mock_db.session.execute.return_value = mock_exec


class _FakeRedis:
    """SET-NX + compare-and-delete-eval shape of the real client."""

    def __init__(self, held: bool = False):
        self.held = held
        self.set_calls: list[tuple] = []
        self.eval_calls: list[tuple] = []

    def set(self, key, value, nx=False, ex=None):
        self.set_calls.append((key, value, nx, ex))
        if self.held:
            return None  # another turn already holds the lock
        self.held = True
        return True

    def eval(self, script, numkeys, key, token):
        self.eval_calls.append((key, token))
        self.held = False
        return 1


def _post_chat(client, message="hello"):
    return client.post(
        "/api/copilot/sessions/sess-1/chat",
        json={"message": message, "workflow_state": {"nodes": [], "edges": []}},
        headers=_AUTH_HEADERS,
    )


@patch("agentic_project_service.routes.copilot.run_copilot_chat")
@patch("agentic_project_service.routes.copilot.db")
@patch(
    "agentic_project_service.auth.decode_jwt",
    return_value={"sub": "user-1", "role": "authenticated"},
)
def test_second_concurrent_turn_rejected_409(_mock_jwt, mock_db, mock_chat):
    """While a turn is in flight for the session, a second POST is rejected 409
    outright (project_copilot semantics) — no user row persisted, no agent run."""
    app = _make_test_app()
    _wire_mock_db(mock_db, [("user", "hello")])
    fake_redis = _FakeRedis(held=True)

    with patch(
        "agentic_project_service.routes.copilot._get_redis", return_value=fake_redis
    ):
        with app.test_client() as client:
            resp = _post_chat(client)

    assert resp.status_code == 409
    assert "in progress" in resp.get_json()["error"]
    mock_chat.assert_not_called()
    # The rejected turn must leave no orphan user row.
    for call in mock_db.session.execute.call_args_list:
        assert "INSERT INTO" not in str(call.args[0])


@patch("agentic_project_service.routes.copilot.run_copilot_chat")
@patch("agentic_project_service.routes.copilot.db")
@patch(
    "agentic_project_service.auth.decode_jwt",
    return_value={"sub": "user-1", "role": "authenticated"},
)
def test_history_query_is_bounded_and_lock_released(_mock_jwt, mock_db, mock_chat):
    """The history query applies the trailing window (LIMIT :lim), and the lock
    is released once the turn finishes."""
    from agentic_project_service.routes import copilot as route_mod

    app = _make_test_app()
    _wire_mock_db(mock_db, [("user", "hello")])
    fake_redis = _FakeRedis()
    mock_chat.return_value = ("Here is your answer", None)

    with patch(
        "agentic_project_service.routes.copilot._get_redis", return_value=fake_redis
    ):
        with app.test_client() as client:
            resp = _post_chat(client)
            assert resp.status_code == 200
            resp.get_data()  # drain the SSE stream so the turn completes

    history_calls = [
        call
        for call in mock_db.session.execute.call_args_list
        if "SELECT role, content" in str(call.args[0])
    ]
    assert history_calls, "chat should load conversation history"
    sql = str(history_calls[0].args[0])
    params = history_calls[0].args[1]
    assert "LIMIT :lim" in sql
    assert "ORDER BY created_at DESC" in sql  # trailing (most recent) window
    assert params["lim"] == route_mod._MAX_HISTORY_MESSAGES == 40

    # The turn is over — the lock was released (worker finally / response close).
    assert len(fake_redis.eval_calls) >= 1
    acquired_token = fake_redis.set_calls[0][1]
    assert fake_redis.eval_calls[0][1] == acquired_token


@patch("agentic_project_service.routes.copilot.run_copilot_chat")
@patch("agentic_project_service.routes.copilot.db")
@patch(
    "agentic_project_service.auth.decode_jwt",
    return_value={"sub": "user-1", "role": "authenticated"},
)
def test_trailing_orphan_user_row_never_builds_user_user(_mock_jwt, mock_db, mock_chat):
    """A trailing orphan 'user' row (crashed prior turn) must not produce a
    user,user sequence in the input handed to run_copilot_chat."""
    app = _make_test_app()
    _wire_mock_db(
        mock_db,
        [
            ("user", "first question"),
            ("assistant", "first answer"),
            ("user", "orphaned by a crash"),
            ("user", "new message"),
        ],
    )
    captured = {}

    def fake_chat(messages, workflow_state, on_event=None):
        captured["messages"] = messages
        return ("ok", None)

    mock_chat.side_effect = fake_chat

    with patch(
        "agentic_project_service.routes.copilot._get_redis", return_value=_FakeRedis()
    ):
        with app.test_client() as client:
            resp = _post_chat(client, message="new message")
            assert resp.status_code == 200
            resp.get_data()

    messages = captured["messages"]
    roles = [m["role"] for m in messages]
    for a, b in zip(roles, roles[1:]):
        assert (a, b) != ("user", "user"), f"user,user pair in built input: {roles}"
    # Both user turns survive (repair inserts a placeholder, it doesn't drop rows).
    user_contents = [m["content"] for m in messages if m["role"] == "user"]
    assert "orphaned by a crash" in user_contents
    assert "new message" in user_contents


@patch("agentic_project_service.routes.copilot.run_copilot_chat")
@patch("agentic_project_service.routes.copilot.db")
@patch(
    "agentic_project_service.auth.decode_jwt",
    return_value={"sub": "user-1", "role": "authenticated"},
)
def test_window_leading_assistant_trimmed_to_user_boundary(_mock_jwt, mock_db, mock_chat):
    """Once the window slides, it can start with an assistant row; the input
    must be trimmed to a user boundary (Anthropic requires user-first)."""
    app = _make_test_app()
    _wire_mock_db(
        mock_db,
        [
            ("assistant", "window starts mid-conversation"),
            ("user", "a question"),
            ("assistant", "an answer"),
            ("user", "new message"),
        ],
    )
    captured = {}

    def fake_chat(messages, workflow_state, on_event=None):
        captured["messages"] = messages
        return ("ok", None)

    mock_chat.side_effect = fake_chat

    with patch(
        "agentic_project_service.routes.copilot._get_redis", return_value=_FakeRedis()
    ):
        with app.test_client() as client:
            resp = _post_chat(client, message="new message")
            assert resp.status_code == 200
            resp.get_data()

    assert captured["messages"][0]["role"] == "user"


@patch("agentic_project_service.routes.copilot.run_copilot_chat")
@patch("agentic_project_service.routes.copilot.db")
@patch(
    "agentic_project_service.auth.decode_jwt",
    return_value={"sub": "user-1", "role": "authenticated"},
)
def test_lock_released_when_pre_thread_setup_fails(_mock_jwt, mock_db, mock_chat):
    """A failure between lock acquisition and worker start (e.g. the user-row
    INSERT raising) must release the lock instead of wedging the session
    until the TTL backstop expires."""
    app = _make_test_app()
    _wire_mock_db(mock_db, [("user", "hello")])
    mock_db.session.commit.side_effect = RuntimeError("db down")
    fake_redis = _FakeRedis()

    with patch(
        "agentic_project_service.routes.copilot._get_redis", return_value=fake_redis
    ):
        with app.test_client() as client:
            resp = _post_chat(client)
            assert resp.status_code == 500  # the failure still surfaces

    assert len(fake_redis.eval_calls) == 1  # released exactly once
    assert fake_redis.held is False
    mock_chat.assert_not_called()
