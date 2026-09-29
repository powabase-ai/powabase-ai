"""An end user's session bind, against Postgres.

With ``end_user_id``, ``get_or_create_session`` and
``get_or_create_orchestration_session`` continue a named session only if it
already exists and is the caller's own in this agent or orchestration, and
create a session only when no id is named. An end user can never name a
session into existence: a backend's ids can be guessable, and a session
planted under one would feed the backend's later runs and expose them.
"""

import uuid

import pytest
from sqlalchemy import text

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


def _owned_agent_session(app, agent_id, session_id, user_id):
    """What POST /api/agents/<id>/sessions does: a session created for a user."""
    with app.app_context():
        get_or_create_session(db.session, agent_id, session_id=session_id, user_id=user_id)
        db.session.commit()


def _owned_orchestration_session(app, orch_id, session_id, user_id):
    with app.app_context():
        orchestration_service.get_or_create_orchestration_session(
            orchestration_id=orch_id, session_id=session_id, user_id=user_id
        )
        db.session.commit()


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
    def test_naming_an_unknown_id_is_refused_and_creates_nothing(self, app):
        agent_id, mallory = _agent(app), str(uuid.uuid4())
        with pytest.raises(SessionNotAccessible):
            _bind_agent(app, agent_id, "wa_15551234567", mallory)
        assert _owners(app, "agent_sessions", "wa_15551234567") == []

    def test_no_id_creates_one_owned_by_the_caller(self, app):
        agent_id, alice = _agent(app), str(uuid.uuid4())
        _, session_id, is_new = _bind_agent(app, agent_id, None, alice)
        assert is_new is True
        assert _owners(app, "agent_sessions", session_id) == [(alice, agent_id)]

    def test_own_session_is_continued(self, app):
        agent_id, alice = _agent(app), str(uuid.uuid4())
        _owned_agent_session(app, agent_id, "sess_own", alice)
        _, session_id, is_new = _bind_agent(app, agent_id, "sess_own", alice)
        assert (session_id, is_new) == ("sess_own", False)

    def test_another_users_session_is_refused(self, app):
        agent_id, alice, bob = _agent(app), str(uuid.uuid4()), str(uuid.uuid4())
        _owned_agent_session(app, agent_id, "sess_alice", alice)
        with pytest.raises(SessionNotAccessible):
            _bind_agent(app, agent_id, "sess_alice", bob)
        assert _owners(app, "agent_sessions", "sess_alice") == [(alice, agent_id)]

    def test_ownerless_session_is_refused(self, app):
        agent_id, alice = _agent(app), str(uuid.uuid4())
        _owned_agent_session(app, agent_id, "sess_backend", None)
        with pytest.raises(SessionNotAccessible):
            _bind_agent(app, agent_id, "sess_backend", alice)
        assert _owners(app, "agent_sessions", "sess_backend") == [(None, agent_id)]

    def test_own_session_of_another_agent_is_refused(self, app):
        agent_a, agent_b, alice = _agent(app), _agent(app), str(uuid.uuid4())
        _owned_agent_session(app, agent_a, "sess_a", alice)
        with pytest.raises(SessionNotAccessible):
            _bind_agent(app, agent_b, "sess_a", alice)
        assert _owners(app, "agent_sessions", "sess_a") == [(alice, agent_a)]


# ---------------------------------------------------------------------------
# Orchestration sessions
# ---------------------------------------------------------------------------


class TestOrchestrationSessionBind:
    def test_naming_an_unknown_id_is_refused_and_creates_nothing(self, app):
        orch_id, mallory = _orchestration(app), str(uuid.uuid4())
        with pytest.raises(SessionNotAccessible):
            _bind_orchestration(app, orch_id, "wa_15551234567", mallory)
        assert _owners(app, "orchestration_sessions", "wa_15551234567") == []

    def test_no_id_creates_one_owned_by_the_caller(self, app):
        orch_id, alice = _orchestration(app), str(uuid.uuid4())
        _, session_id, is_new = _bind_orchestration(app, orch_id, None, alice)
        assert is_new is True
        assert _owners(app, "orchestration_sessions", session_id) == [(alice, orch_id)]

    def test_own_session_is_continued(self, app):
        orch_id, alice = _orchestration(app), str(uuid.uuid4())
        _owned_orchestration_session(app, orch_id, "orch_sess_own", alice)
        _, session_id, is_new = _bind_orchestration(app, orch_id, "orch_sess_own", alice)
        assert (session_id, is_new) == ("orch_sess_own", False)

    def test_another_users_session_is_refused(self, app):
        orch_id, alice, bob = _orchestration(app), str(uuid.uuid4()), str(uuid.uuid4())
        _owned_orchestration_session(app, orch_id, "orch_sess_alice", alice)
        with pytest.raises(SessionNotAccessible):
            _bind_orchestration(app, orch_id, "orch_sess_alice", bob)
        assert _owners(app, "orchestration_sessions", "orch_sess_alice") == [(alice, orch_id)]

    def test_ownerless_session_is_refused(self, app):
        orch_id, alice = _orchestration(app), str(uuid.uuid4())
        _owned_orchestration_session(app, orch_id, "orch_sess_backend", None)
        with pytest.raises(SessionNotAccessible):
            _bind_orchestration(app, orch_id, "orch_sess_backend", alice)

    def test_own_session_of_another_orchestration_is_refused(self, app):
        orch_a, orch_b, alice = _orchestration(app), _orchestration(app), str(uuid.uuid4())
        _owned_orchestration_session(app, orch_a, "orch_sess_a", alice)
        with pytest.raises(SessionNotAccessible):
            _bind_orchestration(app, orch_b, "orch_sess_a", alice)
        assert _owners(app, "orchestration_sessions", "orch_sess_a") == [(alice, orch_a)]


# ---------------------------------------------------------------------------
# The squatting attack the rule exists for
# ---------------------------------------------------------------------------


def test_a_backends_predictable_session_cannot_be_planted(app):
    """Mallory cannot pre-create the id a backend will later use for Alice."""
    agent_id, alice, mallory = _agent(app), str(uuid.uuid4()), str(uuid.uuid4())
    with pytest.raises(SessionNotAccessible):
        _bind_agent(app, agent_id, "wa_15551234567", mallory)
    # The backend's own run creates the session for Alice, with nothing planted.
    _owned_agent_session(app, agent_id, "wa_15551234567", alice)
    assert _owners(app, "agent_sessions", "wa_15551234567") == [(alice, agent_id)]
    with app.app_context():
        runs = db.session.execute(
            text(
                'SELECT count(*) FROM "ai".agent_runs r JOIN "ai".agent_sessions s '
                "ON s.id = r.session_id WHERE s.session_id = :s"
            ),
            {"s": "wa_15551234567"},
        ).scalar()
    assert runs == 0
    # And Mallory still cannot bind to it.
    with pytest.raises(SessionNotAccessible):
        _bind_agent(app, agent_id, "wa_15551234567", mallory)
