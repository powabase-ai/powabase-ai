"""Reading ``ai.project_settings`` must not break the caller's transaction, and a
caller that needs to must be able to tell "no override" from "could not read".

The read now sits on every unrestricted vector search (``VECTOR_EXACT_SEARCH_MAX_ROWS``),
on the same session as the search. A SELECT that fails outside a savepoint aborts
that transaction, and the search that follows raises ``InFailedSqlTransaction``
instead of degrading to the registry default.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from agentic_project_service.services import settings_registry

KEY = "VECTOR_EXACT_SEARCH_MAX_ROWS"


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class _SettingsSession:
    def __init__(self, *, rows=(), fail=False):
        self.rows = list(rows)
        self.fail = fail
        self.depth = 0
        self.depth_at_the_read: int | None = None

    def connection(self):
        """The session's connection: here the same object, savepoints and all."""
        return self

    @contextmanager
    def begin_nested(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def execute(self, clause, params=None):
        self.depth_at_the_read = self.depth
        if self.fail:
            raise RuntimeError('relation "ai.project_settings" does not exist')
        return _Rows(self.rows)


@pytest.fixture
def session(monkeypatch):
    def install(**kwargs):
        fake = _SettingsSession(**kwargs)
        monkeypatch.setattr(settings_registry.db, "session", fake, raising=False)
        monkeypatch.setattr(settings_registry, "has_app_context", lambda: False)
        return fake

    return install


def test_the_read_runs_in_a_savepoint(session):
    fake = session(rows=[(KEY, "0")])
    assert settings_registry.get_setting(KEY) == 0
    assert fake.depth_at_the_read == 1


def test_a_failed_read_still_answers_the_default_through_get_setting(session):
    session(fail=True)
    assert settings_registry.get_setting(KEY) == settings_registry.SETTINGS_REGISTRY[KEY].default


def test_a_failed_read_is_distinguishable_through_get_setting_strict(session):
    session(fail=True)
    with pytest.raises(settings_registry.SettingsUnreadable):
        settings_registry.get_setting_strict(KEY)


def test_get_setting_strict_answers_like_get_setting_when_the_read_works(session):
    session(rows=[(KEY, "1234")])
    assert settings_registry.get_setting_strict(KEY) == 1234
    session(rows=[])
    assert (
        settings_registry.get_setting_strict(KEY)
        == settings_registry.SETTINGS_REGISTRY[KEY].default
    )


# ---------------------------------------------------------------------------
# The read must not flush the caller's pending ORM state
# ---------------------------------------------------------------------------


def _sqlite_session():
    """A real ORM session on SQLite, with ``ai`` attached and SAVEPOINT working.

    pysqlite manages transactions itself and breaks SAVEPOINT unless told not to;
    the two listeners are SQLAlchemy's documented recipe for handing that back.
    """
    from sqlalchemy import Column, Integer, String, create_engine, event, text
    from sqlalchemy.orm import Session, declarative_base
    from sqlalchemy.pool import StaticPool

    engine = create_engine("sqlite://", poolclass=StaticPool)

    @event.listens_for(engine, "connect")
    def _connect(dbapi_connection, _record):
        dbapi_connection.isolation_level = None
        dbapi_connection.execute("ATTACH DATABASE ':memory:' AS ai")

    @event.listens_for(engine, "begin")
    def _begin(connection):
        connection.exec_driver_sql("BEGIN")

    Base = declarative_base()

    class Widget(Base):
        __tablename__ = "widgets"
        id = Column(Integer, primary_key=True)
        name = Column(String, nullable=False)

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE ai.project_settings (key TEXT, value TEXT)"))
        conn.execute(text(f"INSERT INTO ai.project_settings VALUES ('{KEY}', '1234')"))
    return Session(engine), Widget


def test_the_read_does_not_flush_pending_orm_state(monkeypatch):
    """``Session.begin_nested()`` flushes first. A pending object whose flush
    would fail -- nothing to do with settings -- then surfaced as "Failed to load
    project_settings overrides": defaults returned, "unreadable" cached for the
    whole app context, and the caller's session left needing a rollback. The read
    takes a savepoint on the connection instead, which flushes nothing."""
    session, Widget = _sqlite_session()
    monkeypatch.setattr(settings_registry.db, "session", session, raising=False)
    monkeypatch.setattr(settings_registry, "has_app_context", lambda: False)
    broken = Widget(name=None)  # NOT NULL: flushing this raises IntegrityError
    session.add(broken)

    assert settings_registry.get_setting_strict(KEY) == 1234
    assert broken in session.new, "the read flushed the caller's pending object"

    # The session is still usable: the failure belongs to whoever flushes it.
    session.expunge(broken)
    from sqlalchemy import text

    assert session.execute(text("SELECT 1")).scalar() == 1
    session.close()
