"""Failure logging of the pg_search start-up hook and revision 0030.

Both promise never to raise, so the log line must survive an exception whose
message is empty, and the start-up hook must not claim "this server provides
it" about a failure that happened before it could check.
"""

from __future__ import annotations

import importlib.util
import logging
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_project_service import _pg_search_extension as hook


class _Row:
    def __init__(self, value):
        self._value = value

    def first(self):
        return self._value


class _Conn:
    def __init__(self, *, present=False, available=True, create_exc=None):
        self.present = present
        self.available = available
        self.create_exc = create_exc

    def execute(self, clause, params=None):
        sql = str(clause)
        if "pg_available_extensions" in sql:
            return _Row((1,) if self.available else None)
        return _Row((1,) if self.present else None)

    def exec_driver_sql(self, sql):
        if self.create_exc is not None:
            raise self.create_exc


class _Engine:
    def __init__(self, conn=None, begin_exc=None):
        self.conn = conn
        self.begin_exc = begin_exc

    @contextmanager
    def begin(self):
        if self.begin_exc is not None:
            raise self.begin_exc
        yield self.conn


def _errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]


def test_a_connection_failure_is_not_reported_as_a_failed_create(caplog):
    engine = _Engine(begin_exc=ConnectionError("could not connect to server"))
    with caplog.at_level(logging.INFO):
        assert hook.ensure_pg_search_extension(engine) == "failed"
    errors = _errors(caplog)
    assert len(errors) == 1
    assert "could not connect to server" in errors[0]
    assert "provides it" not in errors[0]


def test_a_failed_create_says_the_server_provides_it(caplog):
    engine = _Engine(_Conn(create_exc=RuntimeError("must be loaded via shared_preload_libraries")))
    with caplog.at_level(logging.INFO):
        assert hook.ensure_pg_search_extension(engine) == "failed"
    errors = _errors(caplog)
    assert len(errors) == 1
    assert "provides it" in errors[0]
    assert "shared_preload_libraries" in errors[0]


@pytest.mark.parametrize("message", ["", "   ", "\n"])
@pytest.mark.parametrize("where", ["probe", "create"])
def test_an_empty_error_message_is_logged_not_raised(caplog, message, where):
    exc = RuntimeError(message)
    engine = _Engine(begin_exc=exc) if where == "probe" else _Engine(_Conn(create_exc=exc))
    with caplog.at_level(logging.INFO):
        assert hook.ensure_pg_search_extension(engine) == "failed"
    assert any("RuntimeError" in m for m in _errors(caplog))


def _revision_0030():
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "0030_create_pg_search_extension.py"
    )
    spec = importlib.util.spec_from_file_location("mig_0030_unit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Bind(_Conn):
    @contextmanager
    def begin_nested(self):
        yield


@pytest.mark.parametrize("message", ["", "  \n"])
def test_revision_0030_logs_an_empty_error_message_instead_of_raising(caplog, monkeypatch, message):
    revision = _revision_0030()
    bind = _Bind(create_exc=RuntimeError(message))
    monkeypatch.setattr(revision, "op", SimpleNamespace(get_bind=lambda: bind))
    with caplog.at_level(logging.INFO, logger="alembic.runtime.migration"):
        revision.upgrade()
    assert any("RuntimeError" in m for m in _errors(caplog))
