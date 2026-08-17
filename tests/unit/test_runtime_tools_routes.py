"""Route behavior for runtime_tools."""

from unittest.mock import MagicMock, patch

from agentic_project_service.routes import agents as agents_route
from agentic_project_service.services import billing_port
from agentic_project_service.services import tool_registry as tool_registry_module
from tests.support.billing import RecordingBillingAdapter


def _make_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(agents_route.agents_bp)
    return app


def _auth_headers():
    return {"Authorization": "Bearer fake"}


def test_non_streaming_run_rejects_runtime_tools():
    """/run has no tool loop — the field must 400, not silently degrade."""
    app = _make_test_app()
    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run",
                json={
                    "message": "hi",
                    "runtime_tools": [{"type": "builtin", "name": "web_search"}],
                },
                headers=_auth_headers(),
            )
    assert resp.status_code == 400
    assert "run/stream" in resp.get_json()["error"]


def test_non_streaming_run_rejects_empty_runtime_tools_list():
    """[] must 400 like a non-empty list — the guard is `is not None`, not
    truthiness."""
    app = _make_test_app()
    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run",
                json={"message": "hi", "runtime_tools": []},
                headers=_auth_headers(),
            )
    assert resp.status_code == 400
    assert "run/stream" in resp.get_json()["error"]


def test_stream_run_400s_on_invalid_runtime_tools_before_streaming():
    app = _make_test_app()
    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with patch.object(
            agents_route,
            "validate_runtime_tools",
            return_value=([], "unknown builtin tool: 'telepathy'"),
        ):
            with app.test_client() as client:
                resp = client.post(
                    "/api/agents/agent-1/run/stream",
                    json={
                        "message": "hi",
                        "runtime_tools": [{"type": "builtin", "name": "telepathy"}],
                    },
                    headers=_auth_headers(),
                )
    assert resp.status_code == 400
    assert "telepathy" in resp.get_json()["error"]


def test_runtime_tools_do_not_trip_mutual_exclusivity():
    """runtime_tools composes with context_items (and with
    runtime_knowledge_bases) — the exclusivity 400 must not fire."""
    app = _make_test_app()
    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with (
            patch.object(
                agents_route,
                "validate_runtime_tools",
                return_value=([{"type": "builtin", "name": "web_search"}], None),
            ),
            patch.object(
                agents_route,
                "validate_runtime_knowledge_bases",
                return_value=([{"id": "kb-1"}], None),
            ),
        ):
            with app.test_client() as client:
                resp = client.post(
                    "/api/agents/agent-1/run/stream",
                    json={
                        "message": "hi",
                        "runtime_tools": [{"type": "builtin", "name": "web_search"}],
                        "runtime_knowledge_bases": [{"id": "kb-1"}],
                        "context_items": [{"text": "note"}],
                    },
                    headers=_auth_headers(),
                )
    body = resp.get_data(as_text=True)
    assert "Only one of" not in body


def test_run_stream_wires_validated_runtime_tool_configs_to_tool_loader():
    """The validated configs returned by ``validate_runtime_tools`` must reach
    ``load_all_tools_for_agent`` as the ``runtime_tool_configs`` kwarg.
    Patched at ``tool_registry.load_all_tools_for_agent`` (imported INSIDE the
    SSE generator). Anything after that call may fail — only the wiring is
    pinned."""
    app = _make_test_app()
    billing_port.set_billing_adapter(RecordingBillingAdapter())

    fake_agent_row = ("agent-1", "Test Agent", "anthropic/claude-sonnet-4-6", "", {})
    validated = [{"type": "builtin", "name": "web_search"}]
    captured: dict = {}

    def fake_load_all_tools_for_agent(*args, **kwargs):
        captured["kwargs"] = kwargs
        return {}

    mock_session = MagicMock()
    mock_session.execute.return_value.fetchone.return_value = fake_agent_row

    with (
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
        patch.object(agents_route, "validate_runtime_tools", return_value=(validated, None)),
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
        mock_agent_cls.return_value.stream.side_effect = RuntimeError("no LLM in this unit test")

        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run/stream",
                json={
                    "message": "hi",
                    "runtime_tools": [{"type": "builtin", "name": "web_search"}],
                },
                headers=_auth_headers(),
            )
        resp.get_data()  # force the SSE generator to run

    assert "kwargs" in captured, "load_all_tools_for_agent was never invoked"
    assert captured["kwargs"].get("runtime_tool_configs") == validated


def test_stream_and_db_writes_never_echo_runtime_tool_headers():
    """Inline runtime tool headers are secrets. Drive a real POST through the
    route with REAL validation (an inline-definition entry needs no DB
    query), and assert the secret appears neither in the SSE response bytes
    nor in any statement/params the route hands to db.session.execute while
    persisting the run. Pins that runtime tool configs never leak into the
    stream or run records."""
    app = _make_test_app()
    billing_port.set_billing_adapter(RecordingBillingAdapter())

    secret = "sk-runtime-header-secret"
    fake_agent_row = ("agent-1", "Test Agent", "anthropic/claude-sonnet-4-6", "", {})

    mock_session = MagicMock()
    mock_session.execute.return_value.fetchone.return_value = fake_agent_row

    with (
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
        patch.object(agents_route.db, "session", mock_session),
        patch.object(agents_route, "check_model_available", return_value=None),
        patch.object(
            agents_route, "resolve_api_key_or_raise_for_drop", return_value="fake-api-key"
        ),
        patch.object(agents_route, "Agent") as mock_agent_cls,
        patch.object(tool_registry_module, "load_all_tools_for_agent", return_value={}),
    ):
        mock_agent_cls.return_value.stream.side_effect = RuntimeError("no LLM in this unit test")

        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run/stream",
                json={
                    "message": "hi",
                    "runtime_tools": [
                        {
                            "type": "custom",
                            "definition": {
                                "name": "case_lookup",
                                "description": "d",
                                "input_schema": {"type": "object", "properties": {}},
                                "config": {
                                    "endpoint": "https://api.example.com/cases",
                                    "headers": {"Authorization": secret},
                                },
                            },
                        }
                    ],
                },
                headers=_auth_headers(),
            )
        body = resp.get_data(as_text=True)

    assert secret not in body
    for call in mock_session.execute.call_args_list:
        assert secret not in repr(call)


def _make_orchestrations_test_app():
    from flask import Flask

    from agentic_project_service.routes import orchestrations as orch_route

    app = Flask(__name__)
    app.register_blueprint(orch_route.orchestrations_bp)
    return app


def test_orchestration_stream_400s_on_invalid_runtime_tools_before_streaming():
    from agentic_project_service.routes import orchestrations as orch_route

    app = _make_orchestrations_test_app()
    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with patch.object(
            orch_route,
            "validate_runtime_tools",
            return_value=([], "unknown builtin tool: 'telepathy'"),
        ):
            with app.test_client() as client:
                resp = client.post(
                    "/api/orchestrations/orch-1/run/stream",
                    json={
                        "message": "hi",
                        "runtime_tools": [{"type": "builtin", "name": "telepathy"}],
                    },
                    headers=_auth_headers(),
                )
    assert resp.status_code == 400
    assert "telepathy" in resp.get_json()["error"]
