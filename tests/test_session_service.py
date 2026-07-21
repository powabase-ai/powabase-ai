"""Tests for update_agent_run and persist_agent_run in session service."""

import uuid

import pytest
from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.models.tenant import AgentRunStatus
from agentic_project_service.services.session import update_agent_run


@pytest.fixture
def db_session(app):
    from agentic_project_service.db import db

    with app.app_context():
        yield db.session
        db.session.rollback()


class TestUpdateAgentRun:
    def _create_session_and_run(self, app):
        """Helper: insert an agent_session + agent_run, return (session_id, run_id)."""
        session_id = str(uuid.uuid4())
        sess_id = f"sess_{uuid.uuid4().hex[:12]}"
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agent_sessions (id, session_id, agent_id)
                    VALUES (:sid, :sess_id, :aid)
                    """
                ),
                {
                    "sid": session_id,
                    "sess_id": sess_id,
                    "aid": str(uuid.uuid4()),
                },
            )
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agent_runs
                        (id, session_id, run_id, status)
                    VALUES (:id, :sid, :rid, 'running')
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "sid": session_id,
                    "rid": run_id,
                },
            )
            db.session.commit()
        return session_id, run_id

    def test_update_status(self, app):
        _, run_id = self._create_session_and_run(app)
        with app.app_context():
            update_agent_run(db.session, run_id, status=AgentRunStatus.COMPLETED)
            db.session.commit()

            row = db.session.execute(
                text('SELECT status FROM "ai".agent_runs WHERE run_id = :rid'),
                {"rid": run_id},
            ).fetchone()
            assert row[0] == "completed"

    def test_update_content(self, app):
        _, run_id = self._create_session_and_run(app)
        with app.app_context():
            update_agent_run(db.session, run_id, content="response text")
            db.session.commit()

            row = db.session.execute(
                text('SELECT content FROM "ai".agent_runs WHERE run_id = :rid'),
                {"rid": run_id},
            ).fetchone()
            assert row[0] == "response text"

    def test_update_jsonb_fields(self, app):
        _, run_id = self._create_session_and_run(app)
        with app.app_context():
            msgs = [{"role": "assistant", "content": "hi"}]
            usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            update_agent_run(db.session, run_id, output_messages=msgs, usage=usage)
            db.session.commit()

            # `usage` JSONB was promoted to typed cols in migration 0018.
            row = db.session.execute(
                text(
                    "SELECT output_messages, prompt_tokens, completion_tokens, total_tokens "
                    'FROM "ai".agent_runs WHERE run_id = :rid'
                ),
                {"rid": run_id},
            ).fetchone()
            assert row[0] == msgs
            assert row[1] == 10
            assert row[2] == 5
            assert row[3] == 15

    def test_no_fields_noop(self, app):
        _, run_id = self._create_session_and_run(app)
        with app.app_context():
            # Should not raise
            update_agent_run(db.session, run_id)
            db.session.commit()

            row = db.session.execute(
                text('SELECT status FROM "ai".agent_runs WHERE run_id = :rid'),
                {"rid": run_id},
            ).fetchone()
            assert row[0] == "running"  # unchanged

    def test_updates_parent_session_timestamp(self, app):
        session_id, run_id = self._create_session_and_run(app)
        with app.app_context():
            before = db.session.execute(
                text('SELECT updated_at FROM "ai".agent_sessions WHERE id = :sid'),
                {"sid": session_id},
            ).fetchone()[0]

            update_agent_run(db.session, run_id, status=AgentRunStatus.COMPLETED)
            db.session.commit()

            after = db.session.execute(
                text('SELECT updated_at FROM "ai".agent_sessions WHERE id = :sid'),
                {"sid": session_id},
            ).fetchone()[0]
            assert after >= before

    def test_nonexistent_run_no_error(self, app):
        with app.app_context():
            # Should not raise for a missing run_id
            update_agent_run(
                db.session,
                "run_nonexistent",
                status=AgentRunStatus.COMPLETED,
            )
            db.session.commit()


def test_get_session_owner_returns_user_id_for_owned_session(app, db_session):
    """get_session_owner returns the user_id for a session with an owner."""
    from agentic_project_service.services.session import (
        get_or_create_session,
        get_session_owner,
    )

    agent_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    session_id = "sess_" + uuid.uuid4().hex[:16]

    # Insert a minimal agent row so the FK constraint passes (if any)
    db_session.execute(
        text("INSERT INTO ai.agents (id, name, model) VALUES (:id, 'test', 'gpt-4o')"),
        {"id": agent_id},
    )
    get_or_create_session(db_session, agent_id, session_id=session_id, user_id=user_id)
    db_session.commit()

    assert get_session_owner(db_session, session_id) == user_id


def test_get_session_owner_returns_none_for_missing_session(app, db_session):
    """get_session_owner returns None when the session doesn't exist."""
    from agentic_project_service.services.session import get_session_owner

    assert get_session_owner(db_session, "sess_does_not_exist") is None


def test_get_session_owner_returns_none_for_null_user_id(app, db_session):
    """Legacy sessions with NULL user_id return None (treated as unowned)."""
    from agentic_project_service.services.session import get_session_owner

    agent_id = str(uuid.uuid4())
    session_db_id = str(uuid.uuid4())
    session_id = "sess_legacy_" + uuid.uuid4().hex[:8]

    db_session.execute(
        text("INSERT INTO ai.agents (id, name, model) VALUES (:id, 'test', 'gpt-4o')"),
        {"id": agent_id},
    )
    db_session.execute(
        text(
            "INSERT INTO ai.agent_sessions (id, session_id, agent_id, user_id) "
            "VALUES (:id, :sid, :aid, NULL)"
        ),
        {"id": session_db_id, "sid": session_id, "aid": agent_id},
    )
    db_session.commit()

    assert get_session_owner(db_session, session_id) is None


class TestPersistAgentRunParentIds:
    def test_persist_with_parent_orchestration_run_id(self, app):
        """persist_agent_run stores parent_orchestration_run_id when provided."""
        from agentic_project_service.services.session import persist_agent_run

        orch_run_uuid = str(uuid.uuid4())
        with app.app_context():
            # Insert a minimal orchestration_runs row to satisfy the FK.
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".orchestration_runs (id, run_id, status)
                    VALUES (:id, :rid, 'running')
                    """
                ),
                {"id": orch_run_uuid, "rid": f"orch_{uuid.uuid4().hex[:12]}"},
            )
            db.session.commit()

            run_id = f"run_{uuid.uuid4().hex[:12]}"
            persist_agent_run(
                db_session=db.session,
                run_id=run_id,
                status=AgentRunStatus.COMPLETED,
                input_messages=[{"role": "user", "content": "hi"}],
                output_messages=[{"role": "assistant", "content": "hello"}],
                parent_orchestration_run_id=orch_run_uuid,
            )
            db.session.commit()

            row = db.session.execute(
                text(
                    "SELECT parent_orchestration_run_id, session_id "
                    'FROM "ai".agent_runs WHERE run_id = :rid'
                ),
                {"rid": run_id},
            ).fetchone()
            assert str(row[0]) == orch_run_uuid
            assert row[1] is None  # No session — delegate runs are session-less

    def test_persist_with_parent_workflow_execution_id(self, app):
        """persist_agent_run stores parent_workflow_execution_id when provided."""
        from agentic_project_service.services.session import persist_agent_run

        wf_id = str(uuid.uuid4())
        exec_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".workflows (id, name) VALUES (:wf, :name)
                    """
                ),
                {"wf": wf_id, "name": f"wf-{uuid.uuid4().hex[:6]}"},
            )
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".workflow_executions (id, workflow_id, status)
                    VALUES (:id, :wf, 'running')
                    """
                ),
                {"id": exec_id, "wf": wf_id},
            )
            db.session.commit()

            run_id = f"run_{uuid.uuid4().hex[:12]}"
            persist_agent_run(
                db_session=db.session,
                run_id=run_id,
                status=AgentRunStatus.COMPLETED,
                input_messages=[{"role": "user", "content": "hi"}],
                output_messages=[{"role": "assistant", "content": "hello"}],
                parent_workflow_execution_id=exec_id,
            )
            db.session.commit()

            row = db.session.execute(
                text(
                    "SELECT parent_workflow_execution_id, session_id "
                    'FROM "ai".agent_runs WHERE run_id = :rid'
                ),
                {"rid": run_id},
            ).fetchone()
            assert str(row[0]) == exec_id
            assert row[1] is None

    def test_persist_without_session_id_skips_session_timestamp_update(self, app):
        """When db_session_uuid is None, persist_agent_run must not try to update a session row."""
        from agentic_project_service.services.session import persist_agent_run

        with app.app_context():
            run_id = f"run_{uuid.uuid4().hex[:12]}"
            # Call must not raise even though no session exists.
            persist_agent_run(
                db_session=db.session,
                run_id=run_id,
                status=AgentRunStatus.COMPLETED,
                input_messages=[{"role": "user", "content": "hi"}],
                output_messages=[{"role": "assistant", "content": "hello"}],
            )
            db.session.commit()

            row = db.session.execute(
                text('SELECT session_id FROM "ai".agent_runs WHERE run_id = :rid'),
                {"rid": run_id},
            ).fetchone()
            assert row is not None
            assert row[0] is None
