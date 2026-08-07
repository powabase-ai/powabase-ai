"""Route behavior for runtime_knowledge_bases."""

from unittest.mock import MagicMock, patch

from agentic_project_service.routes import agents as agents_route
from agentic_project_service.routes import orchestrations as orch_route
from agentic_project_service.services import billing_port
from agentic_project_service.services import tool_registry as tool_registry_module
from tests.support.billing import RecordingBillingAdapter


def _make_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(agents_route.agents_bp)
    return app


def _make_orchestrations_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(orch_route.orchestrations_bp)
    return app


def _auth_headers():
    return {"Authorization": "Bearer fake"}


def test_non_streaming_run_rejects_runtime_kbs():
    """/run has no tool loop — the field must 400, not silently degrade."""
    app = _make_test_app()
    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run",
                json={"message": "hi", "runtime_knowledge_bases": [{"id": "kb-1"}]},
                headers=_auth_headers(),
            )
    assert resp.status_code == 400
    assert "run/stream" in resp.get_json()["error"]


def test_non_streaming_run_rejects_empty_runtime_kbs_list():
    """An explicit empty list must 400 the same way a non-empty one does —
    matching /run/stream's `is not None` gate. A bare truthiness check
    (`if data.get(...)`) treats `[]` as falsy and silently lets it through."""
    app = _make_test_app()
    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run",
                json={"message": "hi", "runtime_knowledge_bases": []},
                headers=_auth_headers(),
            )
    assert resp.status_code == 400
    assert "run/stream" in resp.get_json()["error"]


def test_stream_run_400s_on_invalid_runtime_kbs_before_streaming():
    app = _make_test_app()
    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with patch.object(
            agents_route,
            "validate_runtime_knowledge_bases",
            return_value=([], "unknown knowledge base id(s): kb-ghost"),
        ):
            with app.test_client() as client:
                resp = client.post(
                    "/api/agents/agent-1/run/stream",
                    json={"message": "hi", "runtime_knowledge_bases": [{"id": "kb-ghost"}]},
                    headers=_auth_headers(),
                )
    assert resp.status_code == 400
    assert "kb-ghost" in resp.get_json()["error"]


def test_runtime_kbs_do_not_trip_mutual_exclusivity():
    """Combining runtime_knowledge_bases with context_items must not 400 on
    the exclusivity rule (later failures from unmocked internals are fine —
    assert only that THIS error is absent)."""
    app = _make_test_app()
    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with patch.object(
            agents_route, "validate_runtime_knowledge_bases", return_value=([{"id": "kb-1"}], None)
        ):
            with app.test_client() as client:
                resp = client.post(
                    "/api/agents/agent-1/run/stream",
                    json={
                        "message": "hi",
                        "runtime_knowledge_bases": [{"id": "kb-1"}],
                        "context_items": [{"text": "note"}],
                    },
                    headers=_auth_headers(),
                )
    body = resp.get_data(as_text=True)
    assert "Only one of" not in body


def test_run_stream_wires_validated_runtime_kb_configs_to_tool_loader():
    """Fix F4: no test previously proved that the *validated* configs
    returned by ``validate_runtime_knowledge_bases`` actually reach
    ``load_all_tools_for_agent`` from ``/run/stream`` — deleting the
    ``runtime_kb_configs=`` kwarg at that call site passes the rest of the
    suite. Drives a real POST through the route with just enough patched
    (auth, billing, the agent SELECT, the model/key gates, and the tool
    loader itself — imported INSIDE the generator, so it must be patched at
    its source module, ``tool_registry.load_all_tools_for_agent``, not on
    ``agents_route``) to reach that call, and captures the kwargs it was
    invoked with. Anything the generator does AFTER that call (here: the
    real ``agentic.Agent`` LLM call, deliberately made to raise) is allowed
    to fail — this test only pins the wiring, not the rest of the stream.
    """
    app = _make_test_app()
    billing_port.set_billing_adapter(RecordingBillingAdapter())

    kb_id = "11111111-1111-1111-1111-111111111111"
    fake_agent_row = ("agent-1", "Test Agent", "anthropic/claude-sonnet-4-6", "", {})

    captured: dict = {}

    def fake_load_all_tools_for_agent(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {}

    mock_session = MagicMock()
    mock_session.execute.return_value.fetchone.return_value = fake_agent_row

    with (
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
        patch.object(
            agents_route,
            "validate_runtime_knowledge_bases",
            return_value=([{"id": kb_id}], None),
        ),
        patch.object(agents_route.db, "session", mock_session),
        patch.object(agents_route, "check_model_available", return_value=None),
        patch.object(
            agents_route, "resolve_api_key_or_raise_for_drop", return_value="fake-api-key"
        ),
        patch.object(agents_route, "Agent") as mock_agent_cls,
        patch.object(
            tool_registry_module,
            "load_all_tools_for_agent",
            side_effect=fake_load_all_tools_for_agent,
        ),
    ):
        # The naive (tool-less) streaming path is reached once our patched
        # loader returns {} tools. Fail fast and deterministically there
        # instead of attempting a real LLM call.
        mock_agent_cls.return_value.stream.side_effect = RuntimeError("no LLM in this unit test")

        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run/stream",
                json={"message": "hi", "runtime_knowledge_bases": [{"id": kb_id}]},
                headers=_auth_headers(),
            )
        resp.get_data()  # force the SSE generator to actually run

    assert "kwargs" in captured, "load_all_tools_for_agent was never invoked"
    assert captured["kwargs"].get("runtime_kb_configs") == [{"id": kb_id}]


def test_orchestration_stream_400s_on_invalid_runtime_kbs_before_streaming():
    """Mirrors the agents-route test: the orchestration stream route must
    validate runtime_knowledge_bases and 400 before opening the SSE stream."""
    app = _make_orchestrations_test_app()
    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with patch.object(
            orch_route,
            "validate_runtime_knowledge_bases",
            return_value=([], "unknown knowledge base id(s): kb-ghost"),
        ):
            with app.test_client() as client:
                resp = client.post(
                    "/api/orchestrations/orch-1/run/stream",
                    json={"message": "hi", "runtime_knowledge_bases": [{"id": "kb-ghost"}]},
                    headers=_auth_headers(),
                )
    assert resp.status_code == 400
    assert "kb-ghost" in resp.get_json()["error"]
