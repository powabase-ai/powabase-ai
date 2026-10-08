"""ai.message_citations and ai.agent_mcp_servers have the 0034 shape.

Runs on a real Postgres. The app fixture replays migrations 0009 and 0034
(tests/support/citation_schema.py), which is how a deployed database gets them.
"""

import uuid

from sqlalchemy import inspect, text

from agentic_project_service.db import AI_SCHEMA, db
from agentic_project_service.models.tenant import AgentRunStatus
from agentic_project_service.services.session import persist_agent_run


def _columns(table: str) -> dict:
    return {c["name"]: c for c in inspect(db.engine).get_columns(table, schema=AI_SCHEMA)}


def test_citation_unit_columns_and_nullability(app):
    with app.app_context():
        cols = _columns("message_citations")
    for name in ("tool_name", "call_id", "title", "url", "knowledge_base_id", "source_id"):
        assert cols[name]["nullable"] is True, name
    assert cols["kind"]["nullable"] is False
    assert cols["cited"]["nullable"] is False


def test_mcp_servers_have_a_nullable_citation_mapping(app):
    with app.app_context():
        cols = _columns("agent_mcp_servers")
    assert cols["citation_mapping"]["nullable"] is True


def test_a_row_written_without_the_new_columns_is_a_cited_kb_chunk(app):
    """The row a pod still on the previous release writes during a rolling deploy."""
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    with app.app_context():
        run_uuid = persist_agent_run(
            db_session=db.session,
            run_id=run_id,
            status=AgentRunStatus.COMPLETED,
            input_messages=[{"role": "user", "content": "hi"}],
        )
        db.session.execute(
            text(
                'INSERT INTO "ai".message_citations (run_id, citation_key, text_excerpt) '
                "VALUES (:run_id, 1, 'x')"
            ),
            {"run_id": run_uuid},
        )
        db.session.commit()
        kind, cited = db.session.execute(
            text('SELECT kind, cited FROM "ai".message_citations WHERE run_id = :run_id'),
            {"run_id": run_uuid},
        ).one()
    assert (kind, cited) == ("kb_chunk", True)
