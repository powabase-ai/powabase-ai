"""An end user's session bind, against Postgres.

``get_or_create_session`` and ``get_or_create_orchestration_session`` with
``end_user_id`` create the session with ``INSERT ... ON CONFLICT DO NOTHING``
and refuse an existing one that is not the caller's own in this agent or
orchestration. These tests run that SQL for real, including two creators of
one session id racing: whoever commits first owns it, and the other is refused
rather than bound to it or failing on the unique constraint.
"""

import threading
import time
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from agentic_project_service.db import db
from agentic_project_service.services import orchestration as orchestration_service
from agentic_project_service.services.session import (
    SessionNotAccessible,
    get_or_create_session,
)


def _agent(app) -> str:
    agent_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                'INSERT INTO "ai".agents (id, name, model, system_prompt) '
                "VALUES (:id, 'bind', 'gpt-4o-mini', '')"
            ),
            {"id": agent_id},
        )
        db.session.commit()
    return agent_id


def _orchestration(app) -> str:
    orch_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                'INSERT INTO "ai".orchestrations '
                "(id, name, description, strategy, orchestrator_config, settings) "
                "VALUES (:id, 'bind', '', 'supervisor', '{}', '{}')"
            ),
            {"id": orch_id},
        )
        db.session.commit()
    return orch_id


def _owners(app, table: str, session_id: str) -> list[tuple[str | None, str | None]]:
    scope = "agent_id" if table == "agent_sessions" else "orchestration_id"
    with app.app_context():
        rows = db.session.execute(
            text(f'SELECT user_id, {scope} FROM "ai".{table} WHERE session_id = :s'),
            {"s": session_id},
        ).fetchall()
    return [(str(u) if u else None, str(o) if o else None) for u, o in rows]


def _bind_agent(app, agent_id, session_id, end_user_id):
    with app.app_context():
        result = get_or_create_session(
            db.session,
            agent_id,
            session_id=session_id,
            user_id=end_user_id,
            end_user_id=end_user_id,
        )
        db.session.commit()
        return result


def _bind_orchestration(app, orch_id, session_id, end_user_id):
    with app.app_context():
        result = orchestration_service.get_or_create_orchestration_session(
            orchestration_id=orch_id,
            session_id=session_id,
            user_id=end_user_id,
            end_user_id=end_user_id,
        )
        db.session.commit()
        return result


# ---------------------------------------------------------------------------
# Agent sessions
# ---------------------------------------------------------------------------


class TestAgentSessionBind:
    def test_new_id_is_created_owned_by_the_caller(self, app):
        agent_id, alice = _agent(app), str(uuid.uuid4())
        _, session_id, is_new = _bind_agent(app, agent_id, "sess_new", alice)
        assert (session_id, is_new) == ("sess_new", True)
        assert _owners(app, "agent_sessions", "sess_new") == [(alice, agent_id)]

    def test_own_session_is_continued(self, app):
        agent_id, alice = _agent(app), str(uuid.uuid4())
        first, _, _ = _bind_agent(app, agent_id, "sess_own", alice)
        again, _, is_new = _bind_agent(app, agent_id, "sess_own", alice)
        assert (again, is_new) == (first, False)

    def test_another_users_session_is_refused(self, app):
        agent_id, alice, bob = _agent(app), str(uuid.uuid4()), str(uuid.uuid4())
        _bind_agent(app, agent_id, "sess_alice", alice)
        with pytest.raises(SessionNotAccessible):
            _bind_agent(app, agent_id, "sess_alice", bob)
        assert _owners(app, "agent_sessions", "sess_alice") == [(alice, agent_id)]

    def test_ownerless_session_is_refused(self, app):
        agent_id, alice = _agent(app), str(uuid.uuid4())
        with app.app_context():
            get_or_create_session(db.session, agent_id, session_id="sess_backend")
            db.session.commit()
        with pytest.raises(SessionNotAccessible):
            _bind_agent(app, agent_id, "sess_backend", alice)
        assert _owners(app, "agent_sessions", "sess_backend") == [(None, agent_id)]

    def test_own_session_of_another_agent_is_refused(self, app):
        agent_a, agent_b, alice = _agent(app), _agent(app), str(uuid.uuid4())
        _bind_agent(app, agent_a, "sess_a", alice)
        with pytest.raises(SessionNotAccessible):
            _bind_agent(app, agent_b, "sess_a", alice)
        assert _owners(app, "agent_sessions", "sess_a") == [(alice, agent_a)]


# ---------------------------------------------------------------------------
# Orchestration sessions
# ---------------------------------------------------------------------------


class TestOrchestrationSessionBind:
    def test_new_id_is_created_owned_by_the_caller(self, app):
        orch_id, alice = _orchestration(app), str(uuid.uuid4())
        _, session_id, is_new = _bind_orchestration(app, orch_id, "orch_sess_new", alice)
        assert (session_id, is_new) == ("orch_sess_new", True)
        assert _owners(app, "orchestration_sessions", "orch_sess_new") == [(alice, orch_id)]

    def test_own_session_is_continued(self, app):
        orch_id, alice = _orchestration(app), str(uuid.uuid4())
        first, _, _ = _bind_orchestration(app, orch_id, "orch_sess_own", alice)
        again, _, is_new = _bind_orchestration(app, orch_id, "orch_sess_own", alice)
        assert (again, is_new) == (first, False)

    def test_another_users_session_is_refused(self, app):
        orch_id, alice, bob = _orchestration(app), str(uuid.uuid4()), str(uuid.uuid4())
        _bind_orchestration(app, orch_id, "orch_sess_alice", alice)
        with pytest.raises(SessionNotAccessible):
            _bind_orchestration(app, orch_id, "orch_sess_alice", bob)
        assert _owners(app, "orchestration_sessions", "orch_sess_alice") == [(alice, orch_id)]

    def test_ownerless_session_is_refused(self, app):
        orch_id, alice = _orchestration(app), str(uuid.uuid4())
        with app.app_context():
            orchestration_service.get_or_create_orchestration_session(
                orchestration_id=orch_id, session_id="orch_sess_backend"
            )
            db.session.commit()
        with pytest.raises(SessionNotAccessible):
            _bind_orchestration(app, orch_id, "orch_sess_backend", alice)

    def test_own_session_of_another_orchestration_is_refused(self, app):
        orch_a, orch_b, alice = _orchestration(app), _orchestration(app), str(uuid.uuid4())
        _bind_orchestration(app, orch_a, "orch_sess_a", alice)
        with pytest.raises(SessionNotAccessible):
            _bind_orchestration(app, orch_b, "orch_sess_a", alice)
        assert _owners(app, "orchestration_sessions", "orch_sess_a") == [(alice, orch_a)]


# ---------------------------------------------------------------------------
# Two creators of one id: exactly one owns it
# ---------------------------------------------------------------------------


def _wait_until_blocked(app, timeout=10.0) -> None:
    """Return once some backend is waiting on a lock (the second INSERT)."""
    deadline = time.monotonic() + timeout
    with app.app_context():
        while time.monotonic() < deadline:
            waiting = db.session.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE wait_event_type = 'Lock' AND datname = current_database()"
                )
            ).scalar()
            db.session.rollback()
            if waiting:
                return
            time.sleep(0.05)
    raise AssertionError("the second creator never waited on the first")


@pytest.mark.parametrize("kind", ["agent", "orchestration"])
def test_a_concurrent_creator_waits_and_is_refused(app, kind):
    """Alice's create is in flight when Bob creates the same id.

    Bob's INSERT waits for Alice's transaction, then finds her row: he is
    refused. A check-then-insert would have either bound Bob to her session or
    failed on the unique constraint.
    """
    alice, bob = str(uuid.uuid4()), str(uuid.uuid4())
    if kind == "agent":
        scope_id, table, session_id = _agent(app), "agent_sessions", "sess_race"
    else:
        scope_id, table, session_id = _orchestration(app), "orchestration_sessions", "orch_race"

    def bind(session: Session, user_id: str):
        if kind == "agent":
            return get_or_create_session(
                session, scope_id, session_id=session_id, user_id=user_id, end_user_id=user_id
            )
        # The orchestration bind uses the app's scoped session.
        with app.app_context():
            db.session.registry.set(session)
            try:
                return orchestration_service.get_or_create_orchestration_session(
                    orchestration_id=scope_id,
                    session_id=session_id,
                    user_id=user_id,
                    end_user_id=user_id,
                )
            finally:
                db.session.registry.clear()

    outcome: dict[str, object] = {}

    with app.app_context():
        engine = db.engine
    alice_session, bob_session = Session(engine), Session(engine)
    try:
        _, _, alice_new = bind(alice_session, alice)  # inserted, not yet committed

        def bob_binds():
            try:
                outcome["bob"] = bind(bob_session, bob)
                bob_session.commit()
            except Exception as exc:  # noqa: BLE001 - recorded for the assertion
                outcome["bob"] = exc
                bob_session.rollback()

        bob_thread = threading.Thread(target=bob_binds)
        bob_thread.start()
        _wait_until_blocked(app)
        alice_session.commit()
        bob_thread.join(10)
        assert not bob_thread.is_alive()
    finally:
        alice_session.close()
        bob_session.close()

    assert alice_new is True
    assert isinstance(outcome["bob"], SessionNotAccessible), outcome["bob"]
    assert _owners(app, table, session_id) == [(alice, scope_id)]
