"""What the data tools tell the model when something fails.

The model's context, and the SSE stream a client reads, must not carry the
database host, the login, the SQL text or its parameters, or a storage URL.
Only a rejection the model can act on, or the database's one-line primary
message for an error in the query itself, goes back; anything else is logged
with the agent id and replaced by a generic message.
"""

import json
import logging
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import psycopg
import psycopg.errors
import pytest
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError

from agentic_project_service.services.agent_sql import AgentSqlRejected
from agentic_project_service.services.storage import StorageError
from agentic_project_service.services.tool_caller import ToolCaller
from agentic_project_service.tools import builtin
from agentic_project_service.tools.builtin import (
    database_query_handler,
    database_write_handler,
    storage_read_handler,
    storage_write_handler,
)

AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"
USER = ToolCaller(claims={"sub": "user-1", "role": "authenticated"}, token="user-jwt")
SCHEMAS = {"public": ["customers"]}

SQL_TEXT = "SELECT secret_column FROM customers WHERE token = %(p)s"
PARAM = "PARAM-VALUE-7731"
HOST = "db-internal.example:6543"
LOGIN = "agent_login_x"


def _pg_error(cls, primary):
    """A psycopg error carrying a server diagnostic, as a real one would."""

    class _WithDiag(cls):
        @property
        def diag(self):
            return types.SimpleNamespace(message_primary=primary, sqlstate=cls.sqlstate)

    return _WithDiag(f"{primary}\nDETAIL: on {HOST} as {LOGIN}")


def _wrapped(sa_cls, orig):
    return sa_cls(SQL_TEXT, {"p": PARAM}, orig)


def _assert_nothing_internal(text):
    for leaked in (SQL_TEXT, "secret_column", PARAM, HOST, LOGIN, "[SQL", "sqlalche.me"):
        assert leaked not in text


def _query(error):
    with patch.object(builtin.agent_sql, "run_query", side_effect=error):
        raw = database_query_handler(
            {
                "query": "SELECT 1",
                "_caller": USER,
                "_agent_id": AGENT_ID,
                "_schemas_config": SCHEMAS,
            },
            None,
        )
    return raw, json.loads(raw)["error"]


def _write(error):
    @contextmanager
    def failing(*a, **k):
        raise error
        yield  # pragma: no cover

    with patch.object(builtin.agent_sql, "agent_transaction", failing):
        raw = database_write_handler(
            {
                "table": "customers",
                "operation": "insert",
                "data": {"name": "widget"},
                "_caller": USER,
                "_agent_id": AGENT_ID,
                "_schemas_config": SCHEMAS,
            },
            None,
        )
    result = json.loads(raw)
    assert result["success"] is False
    return raw, result["message"]


@pytest.fixture(params=["database_query", "database_write"])
def run_tool(request):
    return _query if request.param == "database_query" else _write


class TestDatabaseToolErrors:
    def test_a_rejection_is_shown_as_is(self, run_tool, caplog):
        raw, message = run_tool(AgentSqlRejected("Function ts_stat is not allowed"))
        assert message == "Function ts_stat is not allowed"

    def test_a_query_error_shows_only_the_primary_message(self, run_tool, caplog):
        orig = _pg_error(
            psycopg.errors.InsufficientPrivilege, "permission denied for table customers"
        )
        raw, message = run_tool(_wrapped(ProgrammingError, orig))
        assert message == "permission denied for table customers"
        _assert_nothing_internal(raw)

    def test_a_row_level_security_violation_is_reported(self, run_tool):
        orig = _pg_error(
            psycopg.errors.InsufficientPrivilege,
            'new row violates row-level security policy for table "customers"',
        )
        raw, message = run_tool(_wrapped(ProgrammingError, orig))
        assert "row-level security" in message
        _assert_nothing_internal(raw)

    def test_a_statement_timeout_is_reported(self, run_tool):
        orig = _pg_error(
            psycopg.errors.QueryCanceled, "canceling statement due to statement timeout"
        )
        raw, message = run_tool(_wrapped(OperationalError, orig))
        assert message == "canceling statement due to statement timeout"

    @pytest.mark.parametrize(
        "orig",
        [
            # A client-side connection failure: no server diagnostic at all.
            psycopg.OperationalError(f'connection to server at "{HOST}" failed'),
            _pg_error(
                psycopg.errors.InvalidPassword,
                f'password authentication failed for user "{LOGIN}"',
            ),
            _pg_error(psycopg.errors.TooManyConnections, f"too many connections for {LOGIN}"),
            _pg_error(psycopg.errors.AdminShutdown, f"terminating connection on {HOST}"),
            _pg_error(psycopg.errors.InternalError_, f"could not open file on {HOST}"),
            _pg_error(psycopg.errors.IoError, f"could not read block in {HOST}"),
        ],
        ids=["no-diagnostic", "auth", "resources", "shutdown", "internal", "io"],
    )
    def test_a_server_or_connection_error_is_generic_and_logged(self, run_tool, caplog, orig):
        with caplog.at_level(logging.ERROR, logger=builtin.logger.name):
            raw, message = run_tool(_wrapped(OperationalError, orig))
        assert message == builtin._DB_TOOL_FAILED
        _assert_nothing_internal(raw)
        assert any(AGENT_ID in r.getMessage() and r.exc_info for r in caplog.records)

    def test_a_non_psycopg_dbapi_error_is_generic(self, run_tool, caplog):
        with caplog.at_level(logging.ERROR, logger=builtin.logger.name):
            raw, message = run_tool(DBAPIError(SQL_TEXT, {"p": PARAM}, RuntimeError(HOST)))
        assert message == builtin._DB_TOOL_FAILED
        _assert_nothing_internal(raw)

    def test_anything_else_is_generic_and_logged_with_the_agent(self, run_tool, caplog):
        with caplog.at_level(logging.ERROR, logger=builtin.logger.name):
            raw, message = run_tool(RuntimeError(f"pool exhausted on {HOST} for {LOGIN}"))
        assert message == builtin._DB_TOOL_FAILED
        _assert_nothing_internal(raw)
        (record,) = [r for r in caplog.records if r.exc_info]
        assert AGENT_ID in record.getMessage()
        assert HOST in str(record.exc_info[1])  # the operator still sees the cause


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

STORAGE_HOST = "storage-internal.example:5000"


def _service_storage(**behaviour):
    storage = MagicMock()
    for name, value in behaviour.items():
        setattr(getattr(storage, name), "side_effect", value)
    return patch.object(builtin, "get_storage", return_value=storage)


def _read(operation, path="a.txt"):
    return storage_read_handler(
        {
            "operation": operation,
            "bucket": "docs",
            "path": path,
            "_caller": ToolCaller.service(),
        },
        None,
    )


class TestStorageToolErrors:
    def test_a_storage_status_is_reported_without_the_body(self, caplog):
        error = StorageError(
            f'Failed to download file: {{"url": "http://{STORAGE_HOST}/object/docs"}}',
            status_code=500,
        )
        with (
            _service_storage(download_from_path=error),
            caplog.at_level(logging.ERROR, logger=builtin.logger.name),
        ):
            raw = _read("download")
        message = json.loads(raw)["error"]
        assert "500" in message
        assert STORAGE_HOST not in raw
        assert any(r.exc_info for r in caplog.records)

    def test_a_missing_file_says_so(self):
        error = StorageError(f"File not found: http://{STORAGE_HOST}/x", status_code=404)
        with _service_storage(download_from_path=error):
            raw = _read("download")
        assert "404" in json.loads(raw)["error"]
        assert STORAGE_HOST not in raw

    def test_a_transport_error_is_generic_and_logged(self, caplog):
        error = StorageError(f"Storage request failed: connect to http://{STORAGE_HOST} refused")
        with (
            _service_storage(download_from_path=error),
            caplog.at_level(logging.ERROR, logger=builtin.logger.name),
        ):
            raw = _read("download")
        assert json.loads(raw)["error"] == builtin._STORAGE_TOOL_FAILED
        assert STORAGE_HOST not in raw
        assert any(r.exc_info for r in caplog.records)

    def test_a_list_failure_reports_the_status_not_the_body(self):
        response = MagicMock(status_code=502, text=f"bad gateway at http://{STORAGE_HOST}")
        storage = MagicMock()
        storage._request.return_value = response
        with patch.object(builtin, "get_storage", return_value=storage):
            raw = _read("list", path="")
        assert "502" in json.loads(raw)["error"]
        assert STORAGE_HOST not in raw

    def test_a_list_transport_error_is_generic(self):
        with _service_storage(_request=StorageError(f"failed: http://{STORAGE_HOST}")):
            raw = _read("list", path="")
        assert json.loads(raw)["error"] == builtin._STORAGE_TOOL_FAILED

    def test_a_signed_url_failure_hides_the_host(self):
        storage = MagicMock()
        storage.download_from_path.return_value = bytes([0xFF, 0xFE])
        storage.create_signed_url.side_effect = StorageError(
            f"Failed to create signed URL: http://{STORAGE_HOST}", status_code=400
        )
        with patch.object(builtin, "get_storage", return_value=storage):
            raw = _read("download")
        assert "400" in json.loads(raw)["error"]
        assert STORAGE_HOST not in raw

    @pytest.mark.parametrize("status", [None, 403])
    def test_a_write_failure_hides_the_host(self, status, caplog):
        error = StorageError(f"Failed to upload file: http://{STORAGE_HOST}", status_code=status)
        with (
            _service_storage(upload=error),
            caplog.at_level(logging.ERROR, logger=builtin.logger.name),
        ):
            raw = storage_write_handler(
                {
                    "bucket": "docs",
                    "path": "a.txt",
                    "content": "x",
                    "_caller": ToolCaller.service(),
                },
                None,
            )
        message = json.loads(raw)["error"]
        assert STORAGE_HOST not in raw
        if status:
            assert str(status) in message
        else:
            assert message == builtin._STORAGE_TOOL_FAILED
        assert any(r.exc_info for r in caplog.records)

    def test_an_unexpected_error_is_generic(self):
        with _service_storage(upload=RuntimeError(f"boom at {STORAGE_HOST}")):
            raw = storage_write_handler(
                {
                    "bucket": "docs",
                    "path": "a.txt",
                    "content": "x",
                    "_caller": ToolCaller.service(),
                },
                None,
            )
        assert json.loads(raw)["error"] == builtin._STORAGE_TOOL_FAILED


class TestStorageErrorCarriesTheStatus:
    """The storage client records the HTTP status so the tools can report it."""

    @pytest.fixture
    def storage(self, monkeypatch):
        import httpx

        from agentic_project_service.services import storage as storage_mod

        monkeypatch.delenv("STORAGE_URL", raising=False)

        def fake_request(method, url, **kwargs):
            return httpx.Response(503, text="unavailable", request=httpx.Request(method, url))

        monkeypatch.setattr(storage_mod.httpx, "request", fake_request)
        return storage_mod.SupabaseStorage(url="http://storage.test", service_key="k")

    @pytest.mark.parametrize(
        "call",
        [
            lambda s: s.download("docs", "a.txt"),
            lambda s: s.upload("docs", "a.txt", b"x"),
            lambda s: s.create_signed_url("docs", "a.txt"),
        ],
    )
    def test_status_code(self, storage, call):
        with pytest.raises(StorageError) as info:
            call(storage)
        assert info.value.status_code == 503

    def test_a_storage_error_without_a_status(self):
        assert StorageError("x").status_code is None
