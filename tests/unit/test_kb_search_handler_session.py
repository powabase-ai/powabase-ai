"""Tests for the knowledge_search handler's DB-session isolation.

The agent loop runs ``is_concurrency_safe`` tools (knowledge_search is one)
in a ThreadPoolExecutor, submitting each via ``contextvars.copy_context().run``.
Because Flask 2.2+ keeps the app context in a ContextVar and Flask-SQLAlchemy
scopes ``db.session`` to that app context, copying the context into worker
threads makes every parallel knowledge_search share ONE Session. SQLAlchemy
Sessions are not thread-safe, so two threads committing it at once raise
"Method 'commit()' can't be called here; method 'commit()' is already in
progress".

The fix: the search handler must run each invocation on its OWN short-lived
session bound to the request session's engine (mirroring the multi-KB path in
context_handler._search_single_kb), and must never commit the shared session.
"""

from unittest.mock import MagicMock, patch

from agentic_project_service.services import tool_registry


def test_search_handler_does_not_commit_shared_session():
    """The handler must never commit the caller's (shared, scoped) session.

    Committing the request-scoped session from inside a worker thread is what
    collides when multiple knowledge_search calls run in parallel.
    """
    shared_session = MagicMock(name="shared_db_session")

    with (
        patch.object(tool_registry, "_get_flask_app", return_value=None),
        patch.object(tool_registry, "Session") as session_cls,
        patch.object(
            tool_registry,
            "create_and_execute",
            return_value=("handler-1", {"formatted_context": "ctx"}),
        ),
    ):
        handler = tool_registry._make_search_handler(shared_session)
        handler(query="q", kb_configs=[{"id": "kb"}], max_tokens=100, session_history=None)

    shared_session.commit.assert_not_called()
    # A dedicated per-call session was created and committed instead.
    call_session = session_cls.return_value
    call_session.commit.assert_called_once()
    call_session.close.assert_called_once()


def test_search_handler_runs_on_own_session_bound_to_request_engine():
    """create_and_execute must receive a fresh session bound to the request
    session's engine — not the shared session itself."""
    shared_session = MagicMock(name="shared_db_session")
    engine = shared_session.get_bind.return_value

    with (
        patch.object(tool_registry, "_get_flask_app", return_value=None),
        patch.object(tool_registry, "Session") as session_cls,
        patch.object(
            tool_registry,
            "create_and_execute",
            return_value=("handler-1", {"formatted_context": "ctx"}),
        ) as fake_create,
    ):
        handler = tool_registry._make_search_handler(shared_session)
        handler(query="q", kb_configs=[{"id": "kb"}], max_tokens=100, session_history=None)

    session_cls.assert_called_once_with(bind=engine)
    passed_session = fake_create.call_args.kwargs["db_session"]
    assert passed_session is session_cls.return_value
    assert passed_session is not shared_session
