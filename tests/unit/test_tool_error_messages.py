"""What the data tools tell the model when something fails.

The model's context, and the SSE stream a client reads, must not carry the
database host, the login, the SQL text or its parameters, or a storage URL.
Only a rejection the model can act on, or the database's one-line primary
message for an error in the query itself, goes back; anything else is logged
with the agent id and replaced by a generic message.
"""

import json
import logging
import time
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import httpx
import psycopg
import psycopg.errors
import pytest
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError

from agentic_project_service.services import storage as storage_mod
from agentic_project_service.services.agent_sql import AgentSqlRejected, AgentToolsUnavailable
from agentic_project_service.services.storage import StorageError, SupabaseStorage
from agentic_project_service.services.tool_caller import ToolCaller
from agentic_project_service.tools import builtin
from agentic_project_service.tools.builtin import (
    database_query_handler,
    database_write_handler,
    storage_read_handler,
    storage_write_handler,
)

AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"
USER_TOKEN = "user-jwt-4471"
USER = ToolCaller(
    claims={"sub": "user-1", "role": "authenticated", "exp": time.time() + 3600},
    token=USER_TOKEN,
)
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


class TestDatabaseToolErrorsAreLogged:
    """What the operator sees: every refusal and query failure is logged with
    the agent, at WARNING when it points at the deployment (a missing grant, a
    timeout). Nothing at WARNING or above carries the SQL or its parameters."""

    def _records(self, caplog, run_tool, error):
        with caplog.at_level(logging.DEBUG, logger=builtin.logger.name):
            run_tool(error)
        return [r for r in caplog.records if r.name == builtin.logger.name]

    @pytest.mark.parametrize(
        ("cls", "primary"),
        [
            (psycopg.errors.InsufficientPrivilege, "permission denied for table customers"),
            (psycopg.errors.QueryCanceled, "canceling statement due to statement timeout"),
            (psycopg.errors.LockNotAvailable, "canceling statement due to lock timeout"),
        ],
        ids=["42501", "57014", "55P03"],
    )
    def test_a_grant_or_timeout_failure_is_a_warning(self, run_tool, caplog, cls, primary):
        records = self._records(
            caplog, run_tool, _wrapped(ProgrammingError, _pg_error(cls, primary))
        )
        (warning,) = [r for r in records if r.levelno >= logging.WARNING]
        assert warning.levelno == logging.WARNING
        assert AGENT_ID in warning.getMessage()
        assert cls.sqlstate in warning.getMessage()
        _assert_nothing_internal(warning.getMessage())
        assert warning.exc_info is None

    @pytest.mark.parametrize(
        "error",
        [
            AgentSqlRejected("Function ts_stat is not allowed"),
            AgentToolsUnavailable("The user's session has expired; ask them to sign in again"),
        ],
        ids=["rejected", "unavailable"],
    )
    def test_a_refusal_is_logged_at_info(self, run_tool, caplog, error):
        records = self._records(caplog, run_tool, error)
        assert not [r for r in records if r.levelno >= logging.WARNING]
        (info,) = [r for r in records if r.levelno == logging.INFO]
        assert AGENT_ID in info.getMessage()
        assert str(error) in info.getMessage()

    def test_another_query_error_is_logged_without_its_message(self, run_tool, caplog):
        """A data error's message can quote a value from the row."""
        orig = _pg_error(
            psycopg.errors.InvalidTextRepresentation,
            f'invalid input syntax for type integer: "{PARAM}"',
        )
        records = self._records(caplog, run_tool, _wrapped(ProgrammingError, orig))
        (info,) = [r for r in records if r.levelno == logging.INFO]
        assert AGENT_ID in info.getMessage() and "22P02" in info.getMessage()
        assert PARAM not in info.getMessage()

    def test_the_query_text_is_logged_only_at_debug(self, caplog):
        query = "SELECT secret_column FROM customers WHERE id = 7731"
        orig = _pg_error(
            psycopg.errors.QueryCanceled, "canceling statement due to statement timeout"
        )
        with (
            caplog.at_level(logging.DEBUG, logger=builtin.logger.name),
            patch.object(
                builtin.agent_sql, "run_query", side_effect=_wrapped(OperationalError, orig)
            ),
        ):
            database_query_handler(
                {
                    "query": query,
                    "_caller": USER,
                    "_agent_id": AGENT_ID,
                    "_schemas_config": SCHEMAS,
                },
                None,
            )
        with_query = [r for r in caplog.records if query in r.getMessage()]
        assert with_query and all(r.levelno == logging.DEBUG for r in with_query)

    def test_the_token_is_never_logged(self, run_tool, caplog):
        with caplog.at_level(logging.DEBUG):
            run_tool(RuntimeError("boom"))
            run_tool(AgentSqlRejected("Function ts_stat is not allowed"))
        assert USER_TOKEN not in caplog.text


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

    @pytest.mark.parametrize("status", [None, 500])
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


# What storage-api v1.33 answers, as observed from a real server. It reports a
# missing object or bucket, and an object the caller's policies hide, with
# HTTP 400 and the real status as a string in the body's statusCode.
MISSING_OBJECT = (400, {"statusCode": "404", "error": "not_found", "message": "Object not found"})
MISSING_BUCKET = (
    400,
    {"statusCode": "404", "error": "Bucket not found", "message": "Bucket not found"},
)
POLICY_DENIED = (
    400,
    {
        "statusCode": "403",
        "error": "Unauthorized",
        "message": "new row violates row-level security policy",
    },
)
INVALID_KEY = (400, {"statusCode": "400", "error": "InvalidKey", "message": "Invalid key"})


@pytest.fixture
def storage_answers(monkeypatch):
    """Point both storage factories at a real client whose requests all get
    the answer ``set(status, body)`` chose; a download succeeds with bytes that
    are not UTF-8 when ``binary`` is set, so a signed URL is asked for."""
    monkeypatch.delenv("STORAGE_URL", raising=False)
    state = types.SimpleNamespace(status=200, body={}, binary=False, sent=[])

    def fake_request(method, url, **kwargs):
        request = httpx.Request(method, url)
        state.sent.append(f"{method} {request.url.path}")
        if state.binary and method == "GET":
            return httpx.Response(200, content=bytes([0xFF, 0xFE]), request=request)
        if isinstance(state.body, str):
            return httpx.Response(state.status, text=state.body, request=request)
        return httpx.Response(state.status, json=state.body, request=request)

    def client(bearer=None):
        return SupabaseStorage(url="http://storage.test", service_key="service-key", bearer=bearer)

    monkeypatch.setattr(storage_mod.httpx, "request", fake_request)
    monkeypatch.setattr(builtin, "get_storage", client)
    monkeypatch.setattr(builtin, "get_storage_for_user", client)

    def set_answer(status, body, *, binary=False):
        state.status, state.body, state.binary = status, body, binary
        return state

    return set_answer


STORAGE_CALLS = [
    ("list", lambda c: _call_read(c, "list", "reports/")),
    ("download", lambda c: _call_read(c, "download", "reports/q3.csv")),
    ("write", lambda c: _call_write(c, "reports/q3.csv", "x")),
]


def _call_read(caller, operation, path):
    return json.loads(
        storage_read_handler(
            {"operation": operation, "bucket": "docs", "path": path, "_caller": caller}, None
        )
    )


def _call_write(caller, path, content):
    return json.loads(
        storage_write_handler(
            {"bucket": "docs", "path": path, "content": content, "_caller": caller}, None
        )
    )


@pytest.mark.parametrize("caller", [ToolCaller.service(), USER], ids=["service", "end-user"])
@pytest.mark.parametrize(("op", "call"), STORAGE_CALLS, ids=[op for op, _ in STORAGE_CALLS])
class TestWhatStorageAnswered:
    """The model is told what went wrong in words it can act on, from the
    status storage-api reports rather than the HTTP status it sends it in."""

    @pytest.mark.parametrize(
        "answer",
        [MISSING_OBJECT, MISSING_BUCKET, (404, {"message": "not found"})],
        ids=["object-400", "bucket-400", "http-404"],
    )
    def test_not_found(self, storage_answers, caller, op, call, answer):
        storage_answers(*answer)
        assert call(caller) == {"error": f"Storage {op} failed: not found"}

    @pytest.mark.parametrize(
        "answer",
        [POLICY_DENIED, (403, {"message": "forbidden"})],
        ids=["policy-400", "http-403"],
    )
    def test_access_denied(self, storage_answers, caller, op, call, answer):
        storage_answers(*answer)
        assert call(caller) == {"error": f"Storage {op} failed: access denied"}

    @pytest.mark.parametrize(
        "answer",
        [INVALID_KEY, (400, "<html>bad request</html>"), (500, {"message": "boom"})],
        ids=["body-400", "not-json", "http-500"],
    )
    def test_anything_else_gives_the_http_status(self, storage_answers, caller, op, call, answer):
        storage_answers(*answer)
        assert call(caller) == {"error": f"Storage {op} failed (HTTP {answer[0]})"}


class TestSignedUrlAnswers:
    def test_a_signed_url_for_a_missing_object_is_not_found(self, storage_answers):
        state = storage_answers(*MISSING_OBJECT, binary=True)
        result = _call_read(USER, "download", "reports/q3.pdf")
        assert result == {"error": "Storage download failed: not found"}
        assert state.sent[-1] == "POST /storage/v1/object/sign/docs/reports/q3.pdf"


class TestStorageFailuresAreLogged:
    def test_a_refusal_is_logged_with_the_target(self, storage_answers, caplog):
        storage_answers(*POLICY_DENIED)
        with caplog.at_level(logging.DEBUG, logger=builtin.logger.name):
            _call_write(USER, "reports/q3.csv", "x")
        (record,) = [r for r in caplog.records if r.name == builtin.logger.name]
        message = record.getMessage()
        assert record.levelno == logging.INFO
        assert "write" in message and "'docs'" in message and "'reports/q3.csv'" in message
        assert "403" in message
        assert USER_TOKEN not in caplog.text

    def test_a_server_error_is_logged_with_the_target_and_cause(self, storage_answers, caplog):
        storage_answers(500, {"message": f"boom at {STORAGE_HOST}"})
        with caplog.at_level(logging.DEBUG, logger=builtin.logger.name):
            _call_read(USER, "download", "reports/q3.csv")
        (record,) = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert "download" in record.getMessage() and "'reports/q3.csv'" in record.getMessage()
        assert record.exc_info  # the operator still sees the cause
        assert USER_TOKEN not in caplog.text


class TestExpiredSession:
    """An end user's storage calls stop when their session does, as their
    database calls do. storage-api itself answers an expired token with
    "Bucket not found", which would send the model looking for the wrong fault."""

    EXPIRED = "The user's session has expired; ask them to sign in again"

    @pytest.mark.parametrize(
        "claims",
        [{"exp": int(time.time()) - 5}, {}, {"exp": "4102444800"}, {"exp": None}],
        ids=["past", "missing", "string", "null"],
    )
    @pytest.mark.parametrize(("op", "call"), STORAGE_CALLS, ids=[op for op, _ in STORAGE_CALLS])
    def test_is_refused_before_storage(self, op, call, claims, caplog):
        caller = ToolCaller(claims={"sub": "user-1", **claims}, token=USER_TOKEN)
        with (
            patch.object(builtin, "get_storage") as service_storage,
            patch.object(builtin, "get_storage_for_user") as user_storage,
            caplog.at_level(logging.INFO, logger=builtin.logger.name),
        ):
            result = call(caller)
        assert result == {"error": self.EXPIRED}
        service_storage.assert_not_called()
        user_storage.assert_not_called()
        assert any(r.levelno == logging.INFO for r in caplog.records)
        assert USER_TOKEN not in caplog.text

    @pytest.mark.parametrize(("op", "call"), STORAGE_CALLS, ids=[op for op, _ in STORAGE_CALLS])
    def test_a_live_session_and_the_service_role_go_ahead(self, storage_answers, op, call):
        storage_answers(200, [])
        for caller in (USER, ToolCaller.service()):
            assert "error" not in call(caller)


@pytest.mark.parametrize("content", [7, None, {"text": "x"}, ["x"], b"x"])
def test_non_string_content_says_so(content):
    with (
        patch.object(builtin, "get_storage") as service_storage,
        patch.object(builtin, "get_storage_for_user") as user_storage,
    ):
        result = _call_write(ToolCaller.service(), "a.txt", content)
    assert result == {"error": "content must be a string"}
    service_storage.assert_not_called()
    user_storage.assert_not_called()


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
