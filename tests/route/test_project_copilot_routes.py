"""Route tests for the Project Copilot (sessions + SSE chat).

The agent run is mocked (``run_project_copilot_chat``) so these tests don't need
a real LLM — they verify session lifecycle, history, and that the SSE stream
emits ``trigger_guide``/``complete`` and persists the assistant message (with the
guide_event) from the background worker thread.
"""

import json


def _events(resp) -> list[dict]:
    """Parse an SSE response body into a list of event dicts."""
    out = []
    for line in resp.get_data(as_text=True).splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: "):]))
    return out


def test_create_session_is_idempotent(client, mock_auth, auth_headers):
    # First call creates (201) or returns a residual session (200); either way the
    # second call must return the SAME single session — that's the singleton guarantee.
    r1 = client.post("/api/project-copilot/sessions", headers=auth_headers)
    assert r1.status_code in (200, 201)
    sid = r1.get_json()["id"]

    r2 = client.post("/api/project-copilot/sessions", headers=auth_headers)
    assert r2.status_code == 200
    assert r2.get_json()["id"] == sid


def test_get_session_null_then_present(client, mock_auth, auth_headers):
    r0 = client.get("/api/project-copilot/sessions", headers=auth_headers)
    assert r0.status_code == 200
    assert r0.get_json()["session"] is None

    sid = client.post("/api/project-copilot/sessions", headers=auth_headers).get_json()["id"]
    r1 = client.get("/api/project-copilot/sessions", headers=auth_headers)
    assert r1.get_json()["session"]["id"] == sid


def test_chat_requires_message(client, mock_auth, auth_headers):
    sid = client.post("/api/project-copilot/sessions", headers=auth_headers).get_json()["id"]
    resp = client.post(
        f"/api/project-copilot/sessions/{sid}/chat", headers=auth_headers, json={"message": "  "}
    )
    assert resp.status_code == 400


def test_chat_unknown_session_404(client, mock_auth, auth_headers):
    resp = client.post(
        "/api/project-copilot/sessions/00000000-0000-0000-0000-000000000000/chat",
        headers=auth_headers,
        json={"message": "hi"},
    )
    assert resp.status_code == 404


def test_chat_streams_guide_and_persists(client, mock_auth, auth_headers, mocker):
    sid = client.post("/api/project-copilot/sessions", headers=auth_headers).get_json()["id"]

    mocker.patch(
        "agentic_project_service.routes.project_copilot.run_project_copilot_chat",
        return_value=("Here's how to connect your coding agent.", "connect", None),
    )

    resp = client.post(
        f"/api/project-copilot/sessions/{sid}/chat",
        headers=auth_headers,
        json={"message": "how do I connect?"},
    )
    assert resp.status_code == 200
    events = _events(resp)
    kinds = [e.get("event") for e in events]
    assert "trigger_guide" in kinds
    assert "complete" in kinds

    trigger = next(e for e in events if e["event"] == "trigger_guide")
    assert trigger["sequence_id"] == "connect"
    complete = next(e for e in events if e["event"] == "complete")
    assert complete["content"] == "Here's how to connect your coding agent."

    # history now has the user + assistant message, assistant carrying guide_event
    msgs = client.get(
        f"/api/project-copilot/sessions/{sid}/messages", headers=auth_headers
    ).get_json()["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant"]
    assert msgs[1]["guide_event"] == {"sequence_id": "connect"}


def test_chat_without_guide_has_no_trigger(client, mock_auth, auth_headers, mocker):
    sid = client.post("/api/project-copilot/sessions", headers=auth_headers).get_json()["id"]
    mocker.patch(
        "agentic_project_service.routes.project_copilot.run_project_copilot_chat",
        return_value=("Just some text, no walkthrough.", None, None),
    )
    resp = client.post(
        f"/api/project-copilot/sessions/{sid}/chat",
        headers=auth_headers,
        json={"message": "hello"},
    )
    events = _events(resp)
    assert "trigger_guide" not in [e.get("event") for e in events]
    assert "complete" in [e.get("event") for e in events]


def test_chat_history_window_trims_to_user_boundary(app, client, mock_auth, auth_headers, mocker):
    """Once a session exceeds the trailing-window cap, the window can begin with an
    assistant message (even-length window, newest row is the just-inserted user
    turn). Anthropic 400s on a leading assistant, so the route must trim to a
    'user' boundary — otherwise every turn fails once a session passes the cap."""
    import uuid

    from sqlalchemy import text

    from agentic_project_service.db import db

    sid = client.post("/api/project-copilot/sessions", headers=auth_headers).get_json()["id"]
    # Seed 42 alternating messages in the past (user at even i), so the newest-40
    # window + the about-to-be-inserted user message starts with an assistant.
    with app.app_context():
        for i in range(42):
            role = "user" if i % 2 == 0 else "assistant"
            db.session.execute(
                text(
                    "INSERT INTO ai.project_copilot_messages (id, session_id, role, content, created_at) "
                    "VALUES (:id, :sid, :role, :c, now() - make_interval(secs => :off))"
                ),
                {"id": str(uuid.uuid4()), "sid": sid, "role": role, "c": f"m{i}", "off": 100 - i},
            )
        db.session.commit()

    captured = {}

    def _fake(messages, on_event=None):
        captured["messages"] = messages
        return ("ok", None, None)

    mocker.patch(
        "agentic_project_service.routes.project_copilot.run_project_copilot_chat",
        side_effect=_fake,
    )
    resp = client.post(
        f"/api/project-copilot/sessions/{sid}/chat", headers=auth_headers, json={"message": "newest"}
    )
    assert resp.status_code == 200
    assert captured["messages"], "run_project_copilot_chat should receive a window"
    assert captured["messages"][0]["role"] == "user"  # trimmed to a user boundary


def test_chat_402_when_out_of_credits(client, mock_auth, auth_headers):
    """An out-of-credits AI-on-us turn is refused with 402 before any work, and
    persists no orphan user message.

    Installs a RecordingBillingAdapter configured to raise 402 (tests/support/
    billing.py) directly via billing_port.set_billing_adapter — same pattern as
    test_agents_route_billing.py. The autouse _billing_adapter_isolation fixture
    restores the previous adapter afterwards. The response-body assertion on the
    old services.balance_cache.PaymentRequired's {"error": "insufficient_credits"}
    shape is dropped: that shape was produced by the private billing_cloud error
    handler, which is excluded from this OSS build (same reasoning already applied
    to the removed webhooks-route 402 body test — see test_webhooks_route_billing.py).
    The behavioral contract this test guards — 402, no orphan row — is unchanged.
    """
    from agentic_project_service.services import billing_port
    from tests.support.billing import RecordingBillingAdapter

    sid = client.post("/api/project-copilot/sessions", headers=auth_headers).get_json()["id"]
    billing_port.set_billing_adapter(RecordingBillingAdapter(raise_402=True))
    resp = client.post(
        f"/api/project-copilot/sessions/{sid}/chat", headers=auth_headers, json={"message": "hi"}
    )
    assert resp.status_code == 402
    msgs = client.get(
        f"/api/project-copilot/sessions/{sid}/messages", headers=auth_headers
    ).get_json()["messages"]
    assert msgs == []  # rejected turn left no user message


def test_chat_guide_only_turn_persists_empty_content(client, mock_auth, auth_headers, mocker):
    """A guide-only turn (no prose, just a triggered guide) must persist the SAME
    "" that streamed in `complete` — not a literal "(empty)" placeholder — so a
    reload renders identically to what streamed."""
    sid = client.post("/api/project-copilot/sessions", headers=auth_headers).get_json()["id"]

    mocker.patch(
        "agentic_project_service.routes.project_copilot.run_project_copilot_chat",
        return_value=("", "connect", None),
    )

    resp = client.post(
        f"/api/project-copilot/sessions/{sid}/chat",
        headers=auth_headers,
        json={"message": "show me how to connect"},
    )
    assert resp.status_code == 200
    complete = next(e for e in _events(resp) if e["event"] == "complete")
    assert complete["content"] == ""

    msgs = client.get(
        f"/api/project-copilot/sessions/{sid}/messages", headers=auth_headers
    ).get_json()["messages"]
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == ""  # reloads identically to what streamed, no "(empty)"


def test_chat_assistant_persist_failure_preserves_alternation(
    client, mock_auth, auth_headers, mocker
):
    """If the assistant-message INSERT fails after a successful generation, the
    turn must not leave an orphan `user` row: a placeholder `assistant` row
    lands instead so the next turn's user/assistant alternation stays valid."""
    from agentic_project_service.db import db

    sid = client.post("/api/project-copilot/sessions", headers=auth_headers).get_json()["id"]

    mocker.patch(
        "agentic_project_service.routes.project_copilot.run_project_copilot_chat",
        return_value=("Here's how to connect.", None, None),
    )

    real_execute = db.session.execute
    state = {"failed_once": False}

    def _flaky_execute(clause, *args, **kwargs):
        # The successful assistant-message insert is the only call whose params
        # carry a "guide" key (see the INSERT in routes/project_copilot.py) —
        # fail it exactly once so the placeholder-insert retry (no "guide" key)
        # goes through normally.
        params = args[0] if args else kwargs.get("parameters")
        if not state["failed_once"] and isinstance(params, dict) and "guide" in params:
            state["failed_once"] = True
            raise RuntimeError("simulated persist failure")
        return real_execute(clause, *args, **kwargs)

    mocker.patch.object(db.session, "execute", side_effect=_flaky_execute)

    resp = client.post(
        f"/api/project-copilot/sessions/{sid}/chat",
        headers=auth_headers,
        json={"message": "how do I connect?"},
    )
    assert resp.status_code == 200
    events = _events(resp)
    assert "error" in [e.get("event") for e in events]

    msgs = client.get(
        f"/api/project-copilot/sessions/{sid}/messages", headers=auth_headers
    ).get_json()["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant"]  # alternation preserved, no orphan user row
    assert msgs[1]["content"]  # placeholder text, not empty


class _FakeRedisLock:
    """Minimal fake mirroring the SET NX EX + compare-and-delete EVAL contract the
    in-flight turn lock uses (see routes/project_copilot.py)."""

    def __init__(self):
        self.store: dict[str, str] = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return False
        self.store[key] = value
        return True

    def eval(self, script, numkeys, key, token):
        if self.store.get(key) == token:
            del self.store[key]
            return 1
        return 0


def test_chat_conflict_when_turn_already_in_flight(client, mock_auth, auth_headers, mocker):
    """A second turn is rejected 409 while one is in-flight for the session, and
    leaves no orphan user message. Once the in-flight marker clears, a new turn
    is accepted again."""
    from agentic_project_service.routes import project_copilot as route_mod

    fake = _FakeRedisLock()
    mocker.patch.object(route_mod, "_get_redis", return_value=fake)

    sid = client.post("/api/project-copilot/sessions", headers=auth_headers).get_json()["id"]

    # Simulate another turn already in-flight for this session.
    held_token = route_mod._try_acquire_turn_lock(sid)
    assert held_token is not None

    resp = client.post(
        f"/api/project-copilot/sessions/{sid}/chat", headers=auth_headers, json={"message": "hi"}
    )
    assert resp.status_code == 409
    assert resp.get_json()["error"] == "a copilot turn is already in progress"

    # The rejected turn left no orphan user message.
    msgs = client.get(
        f"/api/project-copilot/sessions/{sid}/messages", headers=auth_headers
    ).get_json()["messages"]
    assert msgs == []

    # Once the in-flight turn completes (lock released), a new turn is accepted.
    route_mod._release_turn_lock(sid, held_token)
    mocker.patch(
        "agentic_project_service.routes.project_copilot.run_project_copilot_chat",
        return_value=("ok", None, None),
    )
    resp2 = client.post(
        f"/api/project-copilot/sessions/{sid}/chat",
        headers=auth_headers,
        json={"message": "hi again"},
    )
    assert resp2.status_code == 200


def test_chat_emits_notice_when_docs_degraded(client, mock_auth, auth_headers, mocker):
    """When run_project_copilot_chat reports a docs-unavailable notice, the stream
    emits a `notice` event before `complete` so the UI can warn the answer wasn't
    grounded."""
    sid = client.post("/api/project-copilot/sessions", headers=auth_headers).get_json()["id"]
    mocker.patch(
        "agentic_project_service.routes.project_copilot.run_project_copilot_chat",
        return_value=("Answered from general knowledge.", None, "docs_unavailable"),
    )
    resp = client.post(
        f"/api/project-copilot/sessions/{sid}/chat",
        headers=auth_headers,
        json={"message": "how do I connect?"},
    )
    assert resp.status_code == 200
    events = _events(resp)
    notice = next(e for e in events if e.get("event") == "notice")
    assert notice["kind"] == "docs_unavailable"
    assert "complete" in [e.get("event") for e in events]
