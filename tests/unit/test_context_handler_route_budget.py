"""Which budget `/api/context-handlers` defaults to when the caller omits one.

The route runs a knowledge-base retrieval, not an agent run, so the default
belongs to ``KB_DEFAULT_MAX_CONTEXT_TOKENS`` — the setting whose description
says it bounds one retrieval. It read the agent-side setting instead, and
because it always passed *some* value, ``execute_retrieval``'s own KB-budget
fallback was unreachable from here: the two could be given different values
by an operator and the route would follow the one that isn't about it.

Both ship at 64000, so this changes no default today. It changes which
setting an operator's override reaches.
"""

from __future__ import annotations

from unittest.mock import patch

from agentic_project_service.routes import context_handlers as ch_routes


def _client():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(ch_routes.context_handlers_bp)
    return app.test_client()


_DISTINCT_BUDGETS = {
    "KB_DEFAULT_MAX_CONTEXT_TOKENS": 11111,
    "DEFAULT_MAX_CONTEXT_TOKENS": 22222,
}


def _post(body, settings=None):
    """Both settings ship at 64000, so a test that only reads the value back
    cannot tell which one was consulted. These give them different values."""
    recorded = {}
    resolved = settings if settings is not None else _DISTINCT_BUDGETS

    def _fake_execute(db_session, query, knowledge_base_configs, max_context_tokens):
        recorded["max_context_tokens"] = max_context_tokens
        return "handler-1", {
            "status": "COMPLETED",
            "formatted_context": "",
            "retrieved_context": [],
            "metadata": {},
            "errors": [],
        }

    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with patch.object(ch_routes, "create_and_execute", _fake_execute):
            with patch.object(ch_routes, "get_setting", lambda key: resolved[key]):
                with patch.object(ch_routes.db, "session"):
                    response = _client().post(
                        "/api/context-handlers",
                        json=body,
                        headers={"Authorization": "Bearer fake"},
                    )
    return response, recorded


def test_an_omitted_budget_comes_from_the_kb_setting():
    response, recorded = _post({"query": "q", "knowledge_bases": [{"id": "kb-1"}]})

    assert response.status_code == 201
    assert recorded["max_context_tokens"] == _DISTINCT_BUDGETS["KB_DEFAULT_MAX_CONTEXT_TOKENS"]


def test_an_explicit_budget_still_wins():
    _, recorded = _post(
        {"query": "q", "knowledge_bases": [{"id": "kb-1"}], "max_context_tokens": 4321}
    )

    assert recorded["max_context_tokens"] == 4321
