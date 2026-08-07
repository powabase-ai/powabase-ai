"""Route behavior for runtime_knowledge_bases."""

from unittest.mock import patch

from agentic_project_service.routes import agents as agents_route
from agentic_project_service.routes import orchestrations as orch_route


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
