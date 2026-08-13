"""The copilot tables exist with the shape routes/project_copilot.py queries.

Pins the columns the route layer actually selects (id, created_at on sessions;
session_id, role, content, created_at on messages) plus the two indexes. A
migration that creates differently-shaped tables would otherwise fail only at
runtime, inside an SSE stream, where the error is hardest to read.
"""

from sqlalchemy import inspect, text

from agentic_project_service.db import AI_SCHEMA, db


def test_copilot_tables_exist_with_expected_columns(app):
    with app.app_context():
        insp = inspect(db.engine)
        sessions = {c["name"] for c in insp.get_columns("project_copilot_sessions", schema=AI_SCHEMA)}
        messages = {c["name"] for c in insp.get_columns("project_copilot_messages", schema=AI_SCHEMA)}
    assert {"id", "created_at"} <= sessions
    assert {"id", "session_id", "role", "content", "created_at"} <= messages


def test_messages_are_indexed_by_session_and_time(app):
    """The history query orders by (session_id, created_at); without the index it seq-scans."""
    with app.app_context():
        rows = db.session.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE schemaname = :s AND tablename = 'project_copilot_messages'"
            ),
            {"s": AI_SCHEMA},
        ).scalars().all()
    assert any("session_id" in d and "created_at" in d for d in rows)
