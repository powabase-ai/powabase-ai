"""Tests for built-in tool handlers — code_execute, storage_read, storage_write.

These are pure unit tests: no DB, no Flask app context needed.
The code_execute_handler only calls an external HTTP sandbox — no DB access.
The storage handlers mock get_storage() to avoid real storage connections.
"""

import json
from unittest.mock import MagicMock, patch

import agentic_project_service.tools.builtin as builtin_mod
from agentic_project_service.tools.builtin import (
    BUILTIN_HANDLERS,
    BUILTIN_TOOL_DEFINITIONS,
    code_execute_handler,
    database_write_handler,
    storage_read_handler,
    storage_write_handler,
    web_search_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_def(name):
    """Return the BUILTIN_TOOL_DEFINITIONS entry for the given tool name."""
    return next((t for t in BUILTIN_TOOL_DEFINITIONS if t["name"] == name), None)


# ---------------------------------------------------------------------------
# Tool definition presence
# ---------------------------------------------------------------------------


class TestCodeExecuteToolDefinition:
    def test_code_execute_in_definitions(self):
        assert _tool_def("code_execute") is not None

    def test_code_execute_in_handlers(self):
        assert "code_execute" in BUILTIN_HANDLERS

    def test_handler_callable(self):
        assert callable(BUILTIN_HANDLERS["code_execute"])

    def test_definition_has_required_fields(self):
        defn = _tool_def("code_execute")
        assert "name" in defn
        assert "description" in defn
        assert "input_schema" in defn

    def test_input_schema_requires_language_and_code(self):
        defn = _tool_def("code_execute")
        required = defn["input_schema"]["required"]
        assert "language" in required
        assert "code" in required

    def test_language_enum_contains_python_and_javascript(self):
        defn = _tool_def("code_execute")
        lang_prop = defn["input_schema"]["properties"]["language"]
        assert "python" in lang_prop["enum"]
        assert "javascript" in lang_prop["enum"]


# ---------------------------------------------------------------------------
# Unsupported language
# ---------------------------------------------------------------------------


class TestCodeExecuteUnsupportedLanguage:
    def test_rejects_ruby(self):
        result = code_execute_handler(
            {"language": "ruby", "code": "puts 'hello'"},
            context=None,
        )
        data = json.loads(result)
        assert "error" in data
        assert "unsupported" in data["error"].lower()

    def test_rejects_bash(self):
        result = code_execute_handler(
            {"language": "bash", "code": "echo hello"},
            context=None,
        )
        data = json.loads(result)
        assert "error" in data

    def test_rejects_empty_language(self):
        result = code_execute_handler(
            {"language": "", "code": "print(1)"},
            context=None,
        )
        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# Successful Python execution
# ---------------------------------------------------------------------------


class TestCodeExecutePythonSuccess:
    def test_returns_stdout_from_sandbox(self, monkeypatch):
        monkeypatch.setenv("CODE_SANDBOX_URL", "http://sandbox.local/execute")
        monkeypatch.setenv("CODE_SANDBOX_API_KEY", "test-key-123")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stdout": "hello world\n",
            "stderr": "",
            "exit_code": 0,
            "files": [],
        }

        with patch(
            "agentic_project_service.tools.builtin.http_requests.post",
            return_value=mock_response,
        ):
            result = code_execute_handler(
                {"language": "python", "code": "print('hello world')"},
                context=None,
            )

        data = json.loads(result)
        assert data["stdout"] == "hello world\n"
        assert data["stderr"] == ""
        assert data["exit_code"] == 0
        assert data["files"] == []

    def test_posts_to_sandbox_url(self, monkeypatch):
        monkeypatch.setenv("CODE_SANDBOX_URL", "http://sandbox.local/execute")
        monkeypatch.setenv("CODE_SANDBOX_API_KEY", "test-key-123")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "files": [],
        }

        with patch(
            "agentic_project_service.tools.builtin.http_requests.post",
            return_value=mock_response,
        ) as mock_post:
            code_execute_handler(
                {"language": "python", "code": "x = 1"},
                context=None,
            )

        mock_post.assert_called_once()
        # First positional arg is URL
        assert mock_post.call_args.args[0] == "http://sandbox.local/execute"

    def test_sends_api_key_header(self, monkeypatch):
        monkeypatch.setenv("CODE_SANDBOX_URL", "http://sandbox.local/execute")
        monkeypatch.setenv("CODE_SANDBOX_API_KEY", "test-key-123")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "files": [],
        }

        with patch(
            "agentic_project_service.tools.builtin.http_requests.post",
            return_value=mock_response,
        ) as mock_post:
            code_execute_handler(
                {"language": "python", "code": "x = 1"},
                context=None,
            )

        headers = mock_post.call_args.kwargs.get("headers", {})
        assert "test-key-123" in str(headers)

    def test_sends_language_code_timeout_in_payload(self, monkeypatch):
        monkeypatch.setenv("CODE_SANDBOX_URL", "http://sandbox.local/execute")
        monkeypatch.setenv("CODE_SANDBOX_API_KEY", "test-key-123")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "files": [],
        }

        with patch(
            "agentic_project_service.tools.builtin.http_requests.post",
            return_value=mock_response,
        ) as mock_post:
            code_execute_handler(
                {"language": "python", "code": "x = 1", "timeout": 60},
                context=None,
            )

        payload = mock_post.call_args.kwargs.get("json", {})
        assert payload["language"] == "python"
        assert payload["code"] == "x = 1"
        assert payload["timeout"] == 60

    def test_default_timeout_is_30(self, monkeypatch):
        monkeypatch.setenv("CODE_SANDBOX_URL", "http://sandbox.local/execute")
        monkeypatch.setenv("CODE_SANDBOX_API_KEY", "test-key-123")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "files": [],
        }

        with patch(
            "agentic_project_service.tools.builtin.http_requests.post",
            return_value=mock_response,
        ) as mock_post:
            code_execute_handler(
                {"language": "python", "code": "x = 1"},
                context=None,
            )

        payload = mock_post.call_args.kwargs.get("json", {})
        assert payload["timeout"] == 30

    def test_javascript_language_accepted(self, monkeypatch):
        monkeypatch.setenv("CODE_SANDBOX_URL", "http://sandbox.local/execute")
        monkeypatch.setenv("CODE_SANDBOX_API_KEY", "test-key-123")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stdout": "42\n",
            "stderr": "",
            "exit_code": 0,
            "files": [],
        }

        with patch(
            "agentic_project_service.tools.builtin.http_requests.post",
            return_value=mock_response,
        ):
            result = code_execute_handler(
                {"language": "javascript", "code": "console.log(42)"},
                context=None,
            )

        data = json.loads(result)
        assert data["stdout"] == "42\n"


# ---------------------------------------------------------------------------
# Sandbox unavailable / error cases
# ---------------------------------------------------------------------------


class TestCodeExecuteSandboxUnavailable:
    def test_sandbox_url_not_configured_returns_error(self, monkeypatch):
        monkeypatch.delenv("CODE_SANDBOX_URL", raising=False)
        monkeypatch.delenv("CODE_SANDBOX_API_KEY", raising=False)

        result = code_execute_handler(
            {"language": "python", "code": "print(1)"},
            context=None,
        )
        data = json.loads(result)
        assert "error" in data

    def test_sandbox_connection_error_returns_error_json(self, monkeypatch):
        monkeypatch.setenv("CODE_SANDBOX_URL", "http://sandbox.local/execute")
        monkeypatch.setenv("CODE_SANDBOX_API_KEY", "test-key-123")

        with patch(
            "agentic_project_service.tools.builtin.http_requests.post",
            side_effect=Exception("Connection refused"),
        ):
            result = code_execute_handler(
                {"language": "python", "code": "print(1)"},
                context=None,
            )

        data = json.loads(result)
        assert "error" in data

    def test_sandbox_non_200_returns_error_json(self, monkeypatch):
        monkeypatch.setenv("CODE_SANDBOX_URL", "http://sandbox.local/execute")
        monkeypatch.setenv("CODE_SANDBOX_API_KEY", "test-key-123")

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"

        with patch(
            "agentic_project_service.tools.builtin.http_requests.post",
            return_value=mock_response,
        ):
            result = code_execute_handler(
                {"language": "python", "code": "print(1)"},
                context=None,
            )

        data = json.loads(result)
        assert "error" in data

    def test_result_is_valid_json_on_all_error_paths(self, monkeypatch):
        """Even on failure, handler must return valid JSON (never raise)."""
        monkeypatch.setenv("CODE_SANDBOX_URL", "http://sandbox.local/execute")
        monkeypatch.setenv("CODE_SANDBOX_API_KEY", "test-key-123")

        with patch(
            "agentic_project_service.tools.builtin.http_requests.post",
            side_effect=RuntimeError("unexpected"),
        ):
            result = code_execute_handler(
                {"language": "python", "code": "print(1)"},
                context=None,
            )

        # Must be parseable JSON
        data = json.loads(result)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# web_search — platform vs tenant error billing classification (PR #619 I3)
# ---------------------------------------------------------------------------


def _exa_http_error_resp(status):
    """A mock Exa response whose raise_for_status() raises an HTTPError that
    carries the given status_code (mirrors requests' real behavior)."""
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status.side_effect = builtin_mod.http_requests.exceptions.HTTPError(
        response=resp
    )
    return resp


class TestWebSearchErrorBilling:
    """Platform-transient Exa failures (5xx, 429, timeout, connection) carry the
    `_platform_error` marker so the billing wrapper skips the charge; tenant-
    fault 4xx stays billed (no marker). Guards the new $0.10 deep-reasoning
    tier from billing a result the tenant never received."""

    def _run(self, monkeypatch, post):
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        monkeypatch.setattr("agentic_project_service.tools.builtin.http_requests.post", post)
        return json.loads(
            web_search_handler({"query": "q", "search_type": "deep-reasoning"}, None)
        )

    def test_5xx_marks_platform_error(self, monkeypatch):
        out = self._run(monkeypatch, lambda *a, **k: _exa_http_error_resp(503))
        assert out.get("_platform_error") is True

    def test_429_marks_platform_error(self, monkeypatch):
        out = self._run(monkeypatch, lambda *a, **k: _exa_http_error_resp(429))
        assert out.get("_platform_error") is True

    def test_4xx_is_tenant_fault_not_platform_error(self, monkeypatch):
        out = self._run(monkeypatch, lambda *a, **k: _exa_http_error_resp(400))
        assert "_platform_error" not in out

    def test_timeout_marks_platform_error(self, monkeypatch):
        def _raise(*a, **k):
            raise builtin_mod.http_requests.exceptions.Timeout("slow")

        out = self._run(monkeypatch, _raise)
        assert out.get("_platform_error") is True

    def test_connection_error_marks_platform_error(self, monkeypatch):
        def _raise(*a, **k):
            raise builtin_mod.http_requests.exceptions.ConnectionError("down")

        out = self._run(monkeypatch, _raise)
        assert out.get("_platform_error") is True


# ---------------------------------------------------------------------------
# storage_read — tool definition
# ---------------------------------------------------------------------------


class TestStorageReadToolDefinition:
    def test_storage_read_in_definitions(self):
        assert _tool_def("storage_read") is not None

    def test_storage_read_in_handlers(self):
        assert "storage_read" in BUILTIN_HANDLERS

    def test_handler_callable(self):
        assert callable(BUILTIN_HANDLERS["storage_read"])

    def test_definition_has_required_fields(self):
        defn = _tool_def("storage_read")
        assert "name" in defn
        assert "description" in defn
        assert "input_schema" in defn

    def test_input_schema_requires_operation_and_bucket(self):
        defn = _tool_def("storage_read")
        required = defn["input_schema"]["required"]
        assert "operation" in required
        assert "bucket" in required

    def test_operation_enum_contains_list_and_download(self):
        defn = _tool_def("storage_read")
        op_prop = defn["input_schema"]["properties"]["operation"]
        assert "list" in op_prop["enum"]
        assert "download" in op_prop["enum"]


# ---------------------------------------------------------------------------
# storage_read — list operation
# ---------------------------------------------------------------------------


class TestStorageReadList:
    def _make_storage_mock(self, objects):
        """Return a mock storage object whose _request returns a list response."""
        mock_storage = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = objects
        mock_storage._request.return_value = mock_response
        return mock_storage

    def test_list_returns_objects(self):
        objects = [{"name": "file1.txt"}, {"name": "file2.csv"}]
        mock_storage = self._make_storage_mock(objects)

        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=mock_storage,
        ):
            result = storage_read_handler(
                {"operation": "list", "bucket": "my-bucket", "path": ""},
                context=None,
            )

        data = json.loads(result)
        assert data["objects"] == objects

    def test_list_passes_bucket_and_prefix(self):
        mock_storage = self._make_storage_mock([])

        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=mock_storage,
        ):
            storage_read_handler(
                {"operation": "list", "bucket": "my-bucket", "path": "docs/"},
                context=None,
            )

        # _request should be called with POST to list path for the bucket
        call_args = mock_storage._request.call_args
        assert "my-bucket" in call_args.args[1]

    def test_list_empty_result(self):
        mock_storage = self._make_storage_mock([])

        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=mock_storage,
        ):
            result = storage_read_handler(
                {"operation": "list", "bucket": "empty-bucket", "path": ""},
                context=None,
            )

        data = json.loads(result)
        assert data["objects"] == []

    def test_list_missing_bucket_returns_error(self):
        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=MagicMock(),
        ):
            result = storage_read_handler(
                {"operation": "list", "bucket": "", "path": ""},
                context=None,
            )

        data = json.loads(result)
        assert "error" in data

    def test_list_result_is_valid_json(self):
        mock_storage = self._make_storage_mock([{"name": "f.txt"}])

        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=mock_storage,
        ):
            result = storage_read_handler(
                {"operation": "list", "bucket": "b", "path": ""},
                context=None,
            )

        # Must be parseable JSON and a dict
        assert isinstance(json.loads(result), dict)


# ---------------------------------------------------------------------------
# storage_read — download operation (text file)
# ---------------------------------------------------------------------------


class TestStorageReadDownloadText:
    def test_download_text_returns_content_string(self):
        mock_storage = MagicMock()
        mock_storage.download_from_path.return_value = b"hello world"

        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=mock_storage,
        ):
            result = storage_read_handler(
                {"operation": "download", "bucket": "my-bucket", "path": "notes.txt"},
                context=None,
            )

        data = json.loads(result)
        assert data["content"] == "hello world"
        assert data["encoding"] == "utf-8"

    def test_download_passes_correct_path(self):
        mock_storage = MagicMock()
        mock_storage.download_from_path.return_value = b"data"

        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=mock_storage,
        ):
            storage_read_handler(
                {"operation": "download", "bucket": "bucket-a", "path": "dir/file.txt"},
                context=None,
            )

        mock_storage.download_from_path.assert_called_once_with("bucket-a/dir/file.txt")

    def test_download_missing_bucket_returns_error(self):
        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=MagicMock(),
        ):
            result = storage_read_handler(
                {"operation": "download", "bucket": "", "path": "file.txt"},
                context=None,
            )

        data = json.loads(result)
        assert "error" in data

    def test_download_missing_path_returns_error(self):
        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=MagicMock(),
        ):
            result = storage_read_handler(
                {"operation": "download", "bucket": "b", "path": ""},
                context=None,
            )

        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# storage_read — download operation (binary fallback to signed URL)
# ---------------------------------------------------------------------------


class TestStorageReadDownloadBinary:
    def test_binary_file_returns_signed_url(self):
        mock_storage = MagicMock()
        # download_from_path returns bytes that cannot be decoded as UTF-8
        mock_storage.download_from_path.return_value = bytes([0xFF, 0xFE, 0x00, 0x01])
        mock_storage.create_signed_url.return_value = "https://example.com/signed/image.png"

        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=mock_storage,
        ):
            result = storage_read_handler(
                {"operation": "download", "bucket": "assets", "path": "image.png"},
                context=None,
            )

        data = json.loads(result)
        assert data["encoding"] == "binary"
        assert data["signed_url"] == "https://example.com/signed/image.png"

    def test_binary_file_calls_create_signed_url_with_correct_args(self):
        mock_storage = MagicMock()
        mock_storage.download_from_path.return_value = bytes([0xFF, 0xFE])
        mock_storage.create_signed_url.return_value = "https://example.com/signed/img"

        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=mock_storage,
        ):
            storage_read_handler(
                {"operation": "download", "bucket": "assets", "path": "img.png"},
                context=None,
            )

        mock_storage.create_signed_url.assert_called_once_with("assets", "img.png")


# ---------------------------------------------------------------------------
# storage_write — tool definition
# ---------------------------------------------------------------------------


class TestStorageWriteToolDefinition:
    def test_storage_write_in_definitions(self):
        assert _tool_def("storage_write") is not None

    def test_storage_write_in_handlers(self):
        assert "storage_write" in BUILTIN_HANDLERS

    def test_handler_callable(self):
        assert callable(BUILTIN_HANDLERS["storage_write"])

    def test_definition_has_required_fields(self):
        defn = _tool_def("storage_write")
        assert "name" in defn
        assert "description" in defn
        assert "input_schema" in defn

    def test_input_schema_requires_bucket_path_content(self):
        defn = _tool_def("storage_write")
        required = defn["input_schema"]["required"]
        assert "bucket" in required
        assert "path" in required
        assert "content" in required


# ---------------------------------------------------------------------------
# storage_write — upload
# ---------------------------------------------------------------------------


class TestStorageWriteUpload:
    def test_upload_returns_path_url_size(self):
        mock_storage = MagicMock()
        mock_storage.upload.return_value = "my-bucket/output/result.txt"

        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=mock_storage,
        ):
            result = storage_write_handler(
                {
                    "bucket": "my-bucket",
                    "path": "output/result.txt",
                    "content": "hello storage",
                    "content_type": "text/plain",
                },
                context=None,
            )

        data = json.loads(result)
        assert "path" in data
        assert "size" in data
        assert data["size"] == len("hello storage".encode("utf-8"))

    def test_upload_calls_storage_with_correct_args(self):
        mock_storage = MagicMock()
        mock_storage.upload.return_value = "bucket/path/file.txt"

        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=mock_storage,
        ):
            storage_write_handler(
                {
                    "bucket": "bucket",
                    "path": "path/file.txt",
                    "content": "some content",
                    "content_type": "text/plain",
                },
                context=None,
            )

        mock_storage.upload.assert_called_once_with(
            "bucket",
            "path/file.txt",
            "some content".encode("utf-8"),
            "text/plain",
        )

    def test_upload_missing_bucket_returns_error(self):
        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=MagicMock(),
        ):
            result = storage_write_handler(
                {"bucket": "", "path": "file.txt", "content": "data"},
                context=None,
            )

        data = json.loads(result)
        assert "error" in data

    def test_upload_missing_path_returns_error(self):
        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=MagicMock(),
        ):
            result = storage_write_handler(
                {"bucket": "b", "path": "", "content": "data"},
                context=None,
            )

        data = json.loads(result)
        assert "error" in data

    def test_upload_default_content_type_is_text_plain(self):
        mock_storage = MagicMock()
        mock_storage.upload.return_value = "bucket/file.txt"

        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=mock_storage,
        ):
            storage_write_handler(
                {"bucket": "bucket", "path": "file.txt", "content": "data"},
                context=None,
            )

        call_args = mock_storage.upload.call_args
        # 4th positional arg (index 3) is content_type
        assert call_args.args[3] == "text/plain"

    def test_upload_returns_valid_json_on_storage_error(self):
        mock_storage = MagicMock()
        mock_storage.upload.side_effect = Exception("bucket does not exist")

        with patch(
            "agentic_project_service.tools.builtin.get_storage",
            return_value=mock_storage,
        ):
            result = storage_write_handler(
                {"bucket": "bad", "path": "f.txt", "content": "x"},
                context=None,
            )

        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# database_write_handler
# ---------------------------------------------------------------------------


class TestDatabaseWriteHandler:
    def test_invalid_operation_returns_error(self):
        result = database_write_handler(
            {"table": "users", "operation": "drop"},
            context=None,
        )
        data = json.loads(result)
        assert data["success"] is False
        assert "operation" in data["message"].lower() or "invalid" in data["message"].lower()

    def test_insert_empty_data_returns_error(self):
        result = database_write_handler(
            {"table": "users", "operation": "insert", "data": {}},
            context=None,
        )
        data = json.loads(result)
        assert data["success"] is False
        assert "data" in data["message"].lower() or "insert" in data["message"].lower()

    def test_delete_empty_where_returns_error(self):
        result = database_write_handler(
            {"table": "users", "operation": "delete", "where": {}},
            context=None,
        )
        data = json.loads(result)
        assert data["success"] is False
        assert "where" in data["message"].lower() or "delete" in data["message"].lower()

    def test_update_empty_where_returns_error(self):
        result = database_write_handler(
            {"table": "users", "operation": "update", "data": {"name": "Alice"}, "where": {}},
            context=None,
        )
        data = json.loads(result)
        assert data["success"] is False
        assert "where" in data["message"].lower() or "update" in data["message"].lower()

    def test_sql_injection_via_table_name_is_rejected(self):
        result = database_write_handler(
            {
                "table": "users; DROP TABLE users--",
                "operation": "insert",
                "data": {"name": "Alice"},
            },
            context=None,
        )
        data = json.loads(result)
        assert data["success"] is False
        assert "invalid" in data["message"].lower() or "table" in data["message"].lower()

    def test_sql_injection_via_column_name_is_rejected(self):
        result = database_write_handler(
            {
                "table": "users",
                "operation": "insert",
                "data": {"name; DROP TABLE users--": "Alice"},
            },
            context=None,
        )
        data = json.loads(result)
        assert data["success"] is False

    def test_successful_insert_calls_db_session(self):
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_db = MagicMock()
        mock_db.session = mock_session

        with patch("agentic_project_service.tools.builtin.db", mock_db):
            result = database_write_handler(
                {"table": "items", "operation": "insert", "data": {"name": "widget"}},
                context=None,
            )

        data = json.loads(result)
        assert data["success"] is True
        assert data["rows_affected"] == 1
        mock_session.commit.assert_called_once()

    def test_insert_strips_auto_generated_serial_columns(self):
        """Auto-generated columns (SERIAL/IDENTITY) should be stripped from INSERT data."""
        # Track execute calls to inspect SQL
        call_log = []

        def mock_execute(stmt, params=None):
            sql_str = str(stmt) if not hasattr(stmt, "text") else stmt.text
            call_log.append({"sql": sql_str, "params": params})
            mock_result = MagicMock()
            mock_result.rowcount = 1
            # For the information_schema query, return a column named "id"
            if "information_schema" in sql_str:
                mock_result.__iter__ = lambda self: iter([("id",)])
            return mock_result

        mock_session = MagicMock()
        mock_session.execute.side_effect = mock_execute
        mock_db = MagicMock()
        mock_db.session = mock_session

        with patch("agentic_project_service.tools.builtin.db", mock_db):
            result = database_write_handler(
                {
                    "table": "items",
                    "operation": "insert",
                    "data": {"id": 1, "name": "widget"},
                },
                context=None,
            )

        data = json.loads(result)
        assert data["success"] is True
        # The INSERT SQL should NOT contain "id" column
        insert_calls = [c for c in call_log if "INSERT" in c["sql"]]
        assert len(insert_calls) == 1
        assert '"id"' not in insert_calls[0]["sql"]
        assert '"name"' in insert_calls[0]["sql"]
        # The params should not contain "id"
        assert "id" not in insert_calls[0]["params"]

    def test_insert_only_auto_gen_columns_returns_error(self):
        """If all columns are auto-generated, return an error."""

        def mock_execute(stmt, params=None):
            sql_str = str(stmt) if not hasattr(stmt, "text") else stmt.text
            mock_result = MagicMock()
            mock_result.rowcount = 0
            if "information_schema" in sql_str:
                mock_result.__iter__ = lambda self: iter([("id",)])
            return mock_result

        mock_session = MagicMock()
        mock_session.execute.side_effect = mock_execute
        mock_db = MagicMock()
        mock_db.session = mock_session

        with patch("agentic_project_service.tools.builtin.db", mock_db):
            result = database_write_handler(
                {
                    "table": "items",
                    "operation": "insert",
                    "data": {"id": 1},
                },
                context=None,
            )

        data = json.loads(result)
        assert data["success"] is False
        assert "auto-generated" in data["message"]

    def test_insert_without_auto_gen_columns_passes_through(self):
        """If no auto-gen columns exist, data passes through unchanged."""

        def mock_execute(stmt, params=None):
            sql_str = str(stmt) if not hasattr(stmt, "text") else stmt.text
            mock_result = MagicMock()
            mock_result.rowcount = 1
            if "information_schema" in sql_str:
                # No auto-gen columns
                mock_result.__iter__ = lambda self: iter([])
            return mock_result

        mock_session = MagicMock()
        mock_session.execute.side_effect = mock_execute
        mock_db = MagicMock()
        mock_db.session = mock_session

        with patch("agentic_project_service.tools.builtin.db", mock_db):
            result = database_write_handler(
                {
                    "table": "items",
                    "operation": "insert",
                    "data": {"id": 1, "name": "widget"},
                },
                context=None,
            )

        data = json.loads(result)
        assert data["success"] is True
        assert data["rows_affected"] == 1

    def test_insert_introspection_failure_still_inserts(self):
        """If introspection query fails, INSERT proceeds with all columns (graceful degradation)."""
        call_log = []

        def mock_execute(stmt, params=None):
            sql_str = str(stmt) if not hasattr(stmt, "text") else stmt.text
            call_log.append({"sql": sql_str, "params": params})
            if "information_schema" in sql_str:
                raise RuntimeError("permission denied for information_schema")
            mock_result = MagicMock()
            mock_result.rowcount = 1
            return mock_result

        mock_session = MagicMock()
        mock_session.execute.side_effect = mock_execute
        mock_db = MagicMock()
        mock_db.session = mock_session

        with patch("agentic_project_service.tools.builtin.db", mock_db):
            result = database_write_handler(
                {
                    "table": "items",
                    "operation": "insert",
                    "data": {"id": 1, "name": "widget"},
                },
                context=None,
            )

        data = json.loads(result)
        assert data["success"] is True
        assert data["rows_affected"] == 1
        # Both columns should be present since stripping was skipped
        insert_calls = [c for c in call_log if "INSERT" in c["sql"]]
        assert len(insert_calls) == 1
        assert '"id"' in insert_calls[0]["sql"]
        assert '"name"' in insert_calls[0]["sql"]


# ---------------------------------------------------------------------------
# web_search — deep-search tiers via search_type
# ---------------------------------------------------------------------------


def _exa_ok_response():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"results": []}
    return mock_response


class TestWebSearchSchema:
    def test_search_type_enum_includes_deep_tiers(self):
        """Deep tiers are values of search_type (Exa's single `type` field),
        alongside the standard matching algorithms."""
        props = _tool_def("web_search")["input_schema"]["properties"]
        enum = set(props["search_type"]["enum"])
        assert {"auto", "neural", "keyword"}.issubset(enum)
        assert "deep" in enum
        assert "deep-reasoning" in enum

    def test_no_separate_deep_boolean(self):
        """The deep tiers live on search_type, not a separate boolean param."""
        props = _tool_def("web_search")["input_schema"]["properties"]
        assert "deep" not in props


class TestWebSearchType:
    def test_deep_reasoning_passed_through_as_type(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        with patch(
            "agentic_project_service.tools.builtin.http_requests.post",
            return_value=_exa_ok_response(),
        ) as mock_post:
            web_search_handler(
                {"query": "anthropic", "search_type": "deep-reasoning"}, context=None
            )

        assert mock_post.call_args.kwargs.get("json", {})["type"] == "deep-reasoning"

    def test_deep_passed_through_as_type(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        with patch(
            "agentic_project_service.tools.builtin.http_requests.post",
            return_value=_exa_ok_response(),
        ) as mock_post:
            web_search_handler({"query": "anthropic", "search_type": "deep"}, context=None)

        assert mock_post.call_args.kwargs.get("json", {})["type"] == "deep"

    def test_default_search_type_is_auto(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        with patch(
            "agentic_project_service.tools.builtin.http_requests.post",
            return_value=_exa_ok_response(),
        ) as mock_post:
            web_search_handler({"query": "anthropic"}, context=None)

        assert mock_post.call_args.kwargs.get("json", {})["type"] == "auto"
