"""Prompt-cache writes (`cache_creation_tokens`) round-trip through the run tables.

Runs against a real Postgres: the write paths and both raw-SQL read paths build
their column lists by hand, and `list_runs_for_session` reads its row by
position, so only a real row proves the columns line up.
"""

import uuid

from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.models.tenant import AgentRunStatus
from agentic_project_service.services.session import update_agent_run


class TestCacheCreationTokensRoundTrip:
    """Prompt-cache writes persist on agent and orchestration runs and read back
    through every path that rebuilds `usage` from the typed columns.
    """

    USAGE = {
        "prompt_tokens": 1000,
        "completion_tokens": 20,
        "cached_tokens": 400,
        "cache_creation_tokens": 500,
        "total_tokens": 1020,
    }

    def _column(self, table: str, run_id: str):
        return db.session.execute(
            text(f'SELECT cache_creation_tokens FROM "ai".{table} WHERE run_id = :rid'),
            {"rid": run_id},
        ).scalar_one()

    def _persist(self, usage, db_session_uuid=None, model=None) -> str:
        from agentic_project_service.services.session import persist_agent_run

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        persist_agent_run(
            db_session=db.session,
            run_id=run_id,
            status=AgentRunStatus.COMPLETED,
            input_messages=[{"role": "user", "content": "hi"}],
            output_messages=[{"role": "assistant", "content": "hello"}],
            usage=usage,
            db_session_uuid=db_session_uuid,
            model=model,
        )
        db.session.commit()
        return run_id

    def test_persist_then_get_run_by_id(self, app):
        from agentic_project_service.services.session import get_run_by_id

        with app.app_context():
            run_id = self._persist(self.USAGE)
            assert self._column("agent_runs", run_id) == 500
            assert get_run_by_id(db.session, run_id)["usage"] == self.USAGE

    def test_absent_key_persists_null_and_is_omitted(self, app):
        from agentic_project_service.services.session import get_run_by_id

        usage = {k: v for k, v in self.USAGE.items() if k != "cache_creation_tokens"}
        with app.app_context():
            run_id = self._persist(usage)
            assert self._column("agent_runs", run_id) is None
            assert get_run_by_id(db.session, run_id)["usage"] == usage

    def test_update_agent_run_sets_column(self, app):
        with app.app_context():
            run_id = self._persist(None)
            update_agent_run(db.session, run_id, usage=self.USAGE)
            db.session.commit()
            assert self._column("agent_runs", run_id) == 500

    def test_orm_usage_shim(self, app):
        from agentic_project_service.models.tenant import AgentRun

        with app.app_context():
            run_id = self._persist(self.USAGE)
            run = db.session.query(AgentRun).filter_by(run_id=run_id).one()
            assert run.usage == self.USAGE

    def test_list_runs_for_session(self, app, monkeypatch):
        from agentic_project_service.services import session as session_service

        # ai.message_citations comes from the base schema, not the ORM, so a
        # create_all-bootstrapped database has no such table. Citations are
        # not under test here.
        monkeypatch.setattr(session_service, "fetch_citations_for_runs", lambda *_: {})

        session_uuid = str(uuid.uuid4())
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        with app.app_context():
            db.session.execute(
                text(
                    'INSERT INTO "ai".agent_sessions (id, session_id, agent_id) '
                    "VALUES (:id, :sid, NULL)"
                ),
                {"id": session_uuid, "sid": session_id},
            )
            db.session.commit()
            self._persist(self.USAGE, db_session_uuid=session_uuid, model="test-model")

            [run] = session_service.list_runs_for_session(db.session, session_id)
            assert run["usage"] == self.USAGE
            # The columns after the usage block are read by position too.
            assert run["model"] == "test-model"
            assert run["agent_id"] is None

    def test_update_orchestration_run_sets_column_and_orm_usage(self, app):
        from agentic_project_service.models.tenant import OrchestrationRunModel
        from agentic_project_service.services.orchestration import update_orchestration_run

        run_id = f"orch_run_{uuid.uuid4().hex[:12]}"
        with app.app_context():
            db.session.add(OrchestrationRunModel(run_id=run_id, status="running"))
            db.session.commit()

            update_orchestration_run(run_id, status="completed", usage=self.USAGE)
            db.session.commit()

            assert self._column("orchestration_runs", run_id) == 500
            db.session.expire_all()
            run = db.session.query(OrchestrationRunModel).filter_by(run_id=run_id).one()
            assert run.usage == self.USAGE
