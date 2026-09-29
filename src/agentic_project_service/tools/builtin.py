"""Built-in tool implementations and definitions."""

import contextvars
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import litellm
import psycopg
import requests as http_requests
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from ..services import agent_sql
from ..services.agent_sql import AgentSqlRejected
from ..services.external_api import parse_retry_after
from ..services.llm_call import with_llm_key
from ..services.rate_limit import external_limiter
from ..services.settings_registry import get_setting
from ..services.storage import SOURCES_BUCKET, StorageError, get_storage, get_storage_for_user
from ..services.tool_caller import ToolCaller

logger = logging.getLogger(__name__)

# Matched against the whole name: "$" would also match before a trailing newline.
_IDENTIFIER_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{0,63}")


def _validate_identifier(name, label="identifier"):
    if not _IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"Invalid {label}: {name!r}")


_SUPPORTED_LANGUAGES = {"python", "javascript"}

BUILTIN_TOOL_DEFINITIONS = [
    {
        "name": "database_write",
        "description": (
            "Insert, update, or delete records in the project database (public schema only). "
            "Structured operations only — no raw SQL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Table name in the public schema",
                },
                "operation": {
                    "type": "string",
                    "enum": ["insert", "update", "delete"],
                    "description": "Operation to perform",
                },
                "data": {
                    "oneOf": [
                        {"type": "object", "description": "Column values for a single row"},
                        {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Array of objects for batch insert",
                        },
                    ],
                    "description": "Column values for insert/update. Single object or array of objects (batch insert).",
                },
                "where": {
                    "type": "object",
                    "description": "Filter conditions for update/delete",
                },
            },
            "required": ["table", "operation"],
        },
    },
    {
        "name": "database_query",
        "description": (
            "Run a read-only SQL SELECT query against the project database. Returns JSON rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL SELECT query to execute",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "http_request",
        "description": ("Make an HTTP request to an external API. Returns the response body."),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to request"},
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE"],
                    "default": "GET",
                },
                "headers": {"type": "object", "description": "HTTP headers"},
                "body": {
                    "type": "object",
                    "description": "Request body (for POST/PUT)",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "code_execute",
        "description": (
            "Execute Python or JavaScript code in a sandboxed environment. "
            "Code runs in isolation with no access to the project database or storage. "
            "Use print() to output results. Write files to /output/ to generate downloadable files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript"],
                    "description": "Programming language to execute",
                },
                "code": {
                    "type": "string",
                    "description": "Source code to execute",
                },
                "timeout": {
                    "type": "integer",
                    "default": 30,
                    "description": "Max execution time in seconds",
                },
            },
            "required": ["language", "code"],
        },
    },
    {
        "name": "storage_read",
        "description": (
            "Read files or list directory contents from project storage buckets. "
            "Use 'list' to browse objects in a bucket prefix, or 'download' to retrieve file contents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["list", "download"],
                    "description": "Operation to perform",
                },
                "bucket": {
                    "type": "string",
                    "description": "Storage bucket name",
                },
                "path": {
                    "type": "string",
                    "description": "Directory prefix (for list) or file path (for download)",
                    "default": "",
                },
            },
            "required": ["operation", "bucket"],
        },
    },
    {
        "name": "storage_write",
        "description": (
            "Upload file content to a project storage bucket. "
            "Returns the storage path, public URL (if available), and file size."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bucket": {
                    "type": "string",
                    "description": "Storage bucket name",
                },
                "path": {
                    "type": "string",
                    "description": "Destination file path within the bucket",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to upload",
                },
                "content_type": {
                    "type": "string",
                    "description": "MIME type of the content",
                    "default": "text/plain",
                },
            },
            "required": ["bucket", "path", "content"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for current information using Exa.ai. "
            "Returns relevant results with titles, URLs, and content. "
            "Supports domain filtering, date ranges, and different search modes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10)",
                    "default": 5,
                },
                "search_type": {
                    "type": "string",
                    "enum": ["auto", "neural", "keyword", "deep", "deep-reasoning"],
                    "description": "Search mode: 'neural' for semantic/meaning-based search, 'keyword' for exact term matching, 'auto' to let the engine decide. 'deep' and 'deep-reasoning' run Exa's agentic deep search (slower, higher quality, and more expensive — deep-reasoning most of all).",
                    "default": "auto",
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only return results from these domains (e.g. ['arxiv.org', 'github.com']).",
                },
                "exclude_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclude results from these domains (e.g. ['reddit.com', 'pinterest.com']).",
                },
                "start_date": {
                    "type": "string",
                    "description": "Only return results published after this date (ISO 8601 format, e.g. '2024-01-01T00:00:00.000Z').",
                },
                "end_date": {
                    "type": "string",
                    "description": "Only return results published before this date (ISO 8601 format, e.g. '2024-06-01T00:00:00.000Z').",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "company",
                        "news",
                        "research paper",
                        "tweet",
                        "github",
                        "wikipedia",
                        "personal site",
                    ],
                    "description": "Filter results to a specific content category.",
                },
                "content_mode": {
                    "type": "string",
                    "enum": ["highlights", "full_text", "compact_text"],
                    "description": "How much content to return per result: 'highlights' (key snippets, default), 'compact_text' (shorter full text), 'full_text' (complete page text).",
                    "default": "highlights",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_scrape",
        "description": (
            "Extract content from a web page URL. Returns the page content as clean markdown. "
            "Use this to read articles, documentation, or any web page. "
            "Set include_images=true to also analyze images on the page with AI vision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to scrape"},
                "formats": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["markdown", "html", "links"]},
                    "description": "Output format(s). Use 'links' to extract all hyperlinks from the page.",
                    "default": ["markdown"],
                },
                "include_images": {
                    "type": "boolean",
                    "description": "If true, analyze images found on the page using AI vision and include descriptions inline.",
                    "default": False,
                },
                "only_main_content": {
                    "type": "boolean",
                    "description": "If true, extract only the main content and filter out navigation, headers, footers, and sidebars.",
                    "default": True,
                },
                "exclude_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "CSS selectors to exclude from the output (e.g. ['.ads', '#cookie-banner', 'nav']).",
                },
                "wait_for": {
                    "type": "integer",
                    "description": "Milliseconds to wait before scraping, useful for pages that load content dynamically with JavaScript.",
                },
                "mobile": {
                    "type": "boolean",
                    "description": "If true, emulate a mobile device user agent. Useful for mobile-specific content or avoiding desktop paywalls.",
                    "default": False,
                },
            },
            "required": ["url"],
        },
    },
]


def _resolve_table(table, schemas_config):
    """Resolve a table name (bare or schema-qualified) against schemas_config.

    Returns (schema, bare_table, error_message). On success error_message is None.
    """
    available = sorted(f"{s}.{t}" for s, tables in schemas_config.items() for t in tables)

    if "." in table:
        schema, bare = table.split(".", 1)
        if schema not in schemas_config or bare not in schemas_config[schema]:
            return None, None, (f"Table '{table}' not found. Available: {available}")
        return schema, bare, None

    # Bare name — find which schema it belongs to
    matching = [s for s, tables in schemas_config.items() if table in tables]
    if len(matching) == 0:
        return None, None, (f"Table '{table}' not found. Available: {available}")
    if len(matching) > 1:
        return (
            None,
            None,
            (
                f"Ambiguous table '{table}' exists in multiple schemas: {matching}. "
                f"Use schema-qualified name, e.g. '{matching[0]}.{table}'"
            ),
        )
    return matching[0], table, None


_NO_CALLER_MESSAGE = "This tool is not available here: the run does not say who it acts for"
_NO_TABLES_MESSAGE = "No tables are configured on this agent's database tool"


def _pop_caller(arguments) -> ToolCaller | None:
    """The caller the tool loader injected. Anything else in the key is ignored:
    tool arguments come from the model, which must never choose who it acts as."""
    caller = arguments.pop("_caller", None)
    return caller if isinstance(caller, ToolCaller) else None


_DB_TOOL_FAILED = "The database tool failed; the error has been logged"

# SQLSTATE classes about the server or the connection rather than the query:
# their messages name hosts, logins and files, and the model cannot fix them.
# 57014 (query_canceled, e.g. a statement timeout) is about the query.
_SERVER_SQLSTATE_CLASSES = frozenset({"08", "28", "53", "57", "58", "F0", "XX"})

# Query errors that point at the deployment rather than the model's query:
# a missing grant (the login's grants have drifted from the tool's tables, or
# an end user's policies refuse the row), and the statement or lock timeouts.
# Their primary messages name only a relation, never a value from a row.
_OPERATOR_SQLSTATES = {
    "42501": "was refused a permission",
    "57014": "timed out",
    "55P03": "timed out waiting for a lock",
}


def _database_error_message(exc: Exception, agent_id) -> str:
    """What a database tool tells the model about ``exc``, which is logged.

    A rejection is shown as is. An error in the query itself is reduced to the
    database's primary message: never the SQL text, its parameters or
    SQLAlchemy's wrapper. Anything else is logged and replaced. Nothing logged
    here carries the SQL, its parameters or a message that could quote them.
    """
    if isinstance(exc, AgentSqlRejected):
        logger.info("Database tool for agent %s refused: %s", agent_id, exc)
        return str(exc)
    if isinstance(exc, DBAPIError) and isinstance(exc.orig, psycopg.Error):
        sqlstate = exc.orig.sqlstate or ""
        primary = exc.orig.diag.message_primary
        if primary and (sqlstate == "57014" or sqlstate[:2] not in _SERVER_SQLSTATE_CLASSES):
            if sqlstate in _OPERATOR_SQLSTATES:
                logger.warning(
                    "Database tool for agent %s %s (SQLSTATE %s): %s",
                    agent_id,
                    _OPERATOR_SQLSTATES[sqlstate],
                    sqlstate,
                    primary,
                )
            else:
                # Its message can quote a value from the row, so only the code.
                logger.info(
                    "Database tool for agent %s: query error (SQLSTATE %s)", agent_id, sqlstate
                )
            return primary
    logger.exception("Database tool failed for agent %s", agent_id)
    return _DB_TOOL_FAILED


def database_write_handler(arguments, context):
    """Perform a structured INSERT, UPDATE, or DELETE on the configured schema(s).

    Runs as the run's caller (see services/agent_sql.py), never on the
    service's own session.
    """
    caller = _pop_caller(arguments)
    agent_id = arguments.pop("_agent_id", None)
    original_table = arguments.get("table", "")
    operation = arguments.get("operation", "")
    data = arguments.get("data") or {}
    where = arguments.get("where") or {}
    arguments.pop("_allowed_schemas", None)
    arguments.pop("_allowed_tables", None)
    schemas_config = arguments.pop("_schemas_config", {})

    # Validate operation early
    if operation not in ("insert", "update", "delete"):
        return json.dumps(
            {
                "success": False,
                "message": f"Invalid operation: '{operation}'. Must be insert, update, or delete.",
            }
        )

    # Resolve table name (supports both "table" and "schema.table")
    if not schemas_config:
        return json.dumps({"success": False, "message": _NO_TABLES_MESSAGE})
    schema, table, err = _resolve_table(original_table, schemas_config)
    if err:
        return json.dumps({"success": False, "message": err})

    # Defense-in-depth: validate schema names
    effective_schemas = [schema]
    for s in effective_schemas:
        if not _IDENTIFIER_RE.fullmatch(s):
            return json.dumps({"success": False, "message": f"Invalid schema name: {s}"})

    # Normalize data: accept single object or array of objects for batch insert
    if isinstance(data, dict):
        rows = [data] if data else []
    elif isinstance(data, list) and all(isinstance(r, dict) for r in data):
        rows = data
    else:
        return json.dumps(
            {
                "success": False,
                "message": f"'data' must be an object (single row) or array of objects (batch insert). Got {type(data).__name__}.",
            }
        )

    # Validate identifiers to prevent SQL injection
    try:
        _validate_identifier(table, "table name")
        for row in rows:
            for key in row:
                _validate_identifier(key, "column name")
        for key in where:
            _validate_identifier(key, "column name")
    except ValueError as exc:
        return json.dumps({"success": False, "message": str(exc)})

    # Validate required fields per operation
    if operation == "insert":
        if not rows:
            return json.dumps({"success": False, "message": "INSERT requires non-empty data."})
        if not any(row for row in rows):
            return json.dumps(
                {"success": False, "message": "INSERT requires at least one row with columns."}
            )

    if operation == "update":
        if not rows:
            return json.dumps({"success": False, "message": "UPDATE requires non-empty data."})
        if not where:
            return json.dumps(
                {
                    "success": False,
                    "message": "UPDATE requires non-empty where (mass updates not allowed).",
                }
            )

    if operation == "delete" and not where:
        return json.dumps(
            {
                "success": False,
                "message": "DELETE requires non-empty where (mass deletes not allowed).",
            }
        )

    # For insert, validate column consistency across all rows before touching the DB
    if operation == "insert":
        columns = list(rows[0].keys())
        col_set = set(columns)
        for i, row in enumerate(rows[1:], start=2):
            if set(row.keys()) != col_set:
                return json.dumps(
                    {
                        "success": False,
                        "message": f"All rows must have the same columns. Row 1 has {sorted(col_set)}, row {i} has {sorted(row.keys())}.",
                    }
                )

    if caller is None or agent_id is None:
        return json.dumps({"success": False, "message": _NO_CALLER_MESSAGE})

    try:
        with agent_sql.agent_transaction(
            caller, agent_id, effective_schemas, read_only=False
        ) as conn:
            # What the name resolves to is only known here, on the caller's
            # search path: a view that runs as its owner would skip the
            # caller's row policies, so it is refused before any write.
            agent_sql.check_write_target(conn, schema, table)
            total_affected = _run_write(conn, operation, table, rows, where, effective_schemas)
        return json.dumps(
            {
                "success": True,
                "rows_affected": total_affected,
                "message": f"{operation} completed.",
            }
        )
    except _NothingToInsert:
        return json.dumps(
            {
                "success": False,
                "message": "INSERT data contains only auto-generated columns — nothing to insert.",
            }
        )
    except Exception as e:
        return json.dumps({"success": False, "message": _database_error_message(e, agent_id)})


class _NothingToInsert(Exception):
    """Every column the caller gave is generated by the database."""


def _run_write(conn, operation, table, rows, where, effective_schemas) -> int:
    """Execute a validated write on ``conn``; return the rows affected."""
    if operation == "insert":
        # Strip auto-generated columns (SERIAL, IDENTITY) to prevent sequence desync
        auto_gen_cols = set()
        try:
            schema_name = effective_schemas[0] if effective_schemas else "public"
            conn.execute(text("SAVEPOINT _autogen_check"))
            result = conn.execute(
                text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                      AND table_name = :table
                      AND (
                          column_default LIKE 'nextval(%'
                          OR is_identity = 'YES'
                      )
                """),
                {"schema": schema_name, "table": table},
            )
            auto_gen_cols = {r[0] for r in result}
            conn.execute(text("RELEASE SAVEPOINT _autogen_check"))
        except Exception:
            conn.execute(text("ROLLBACK TO SAVEPOINT _autogen_check"))
            # Introspection failed; proceed without stripping

        if auto_gen_cols:
            rows = [{k: v for k, v in row.items() if k not in auto_gen_cols} for row in rows]
            if not rows or not any(row for row in rows):
                raise _NothingToInsert()

        columns = list(rows[0].keys())
        col_sql = ", ".join(f'"{k}"' for k in columns)
        placeholders = ", ".join(f":{k}" for k in columns)
        sql = f'INSERT INTO "{table}" ({col_sql}) VALUES ({placeholders})'
        total_affected = 0
        for row in rows:
            result = conn.execute(text(sql), row)
            total_affected += result.rowcount
        return total_affected

    if operation == "update":
        update_data = rows[0]  # update uses single object
        set_clause = ", ".join(f'"{k}" = :set_{k}' for k in update_data.keys())
        where_clause = " AND ".join(f'"{k}" = :where_{k}' for k in where.keys())
        params = {f"set_{k}": v for k, v in update_data.items()}
        params.update({f"where_{k}": v for k, v in where.items()})
        sql = f'UPDATE "{table}" SET {set_clause} WHERE {where_clause}'
        return conn.execute(text(sql), params).rowcount

    # delete
    where_clause = " AND ".join(f'"{k}" = :where_{k}' for k in where.keys())
    params = {f"where_{k}": v for k, v in where.items()}
    sql = f'DELETE FROM "{table}" WHERE {where_clause}'
    return conn.execute(text(sql), params).rowcount


def database_query_handler(arguments, context):
    """Run read-only SQL against the project's Postgres, as the run's caller.

    The SQL is parsed and held to the agent's configured tables before it
    runs, on a non-superuser login (see services/agent_sql.py).
    """
    sql = arguments.get("query", "").strip().rstrip(";").strip()
    caller = _pop_caller(arguments)
    agent_id = arguments.pop("_agent_id", None)
    arguments.pop("_allowed_schemas", None)
    arguments.pop("_allowed_tables", None)
    schemas_config = arguments.pop("_schemas_config", {})

    # Defense-in-depth: validate schema names
    for s in schemas_config:
        if not _IDENTIFIER_RE.fullmatch(s):
            return json.dumps({"error": f"Invalid schema name: {s}"})

    if not schemas_config:
        return json.dumps({"error": _NO_TABLES_MESSAGE})
    if caller is None or agent_id is None:
        return json.dumps({"error": _NO_CALLER_MESSAGE})

    try:
        rows = agent_sql.run_query(caller, agent_id, sql, schemas_config)
    except Exception as e:
        logger.debug("Database tool query for agent %s failed: %s", agent_id, sql)
        return json.dumps({"error": _database_error_message(e, agent_id)})
    return json.dumps(rows, default=str)[:50000]


def http_request_handler(arguments, context):
    """Call an external HTTP API."""
    try:
        response = http_requests.request(
            method=arguments.get("method", "GET"),
            url=arguments["url"],
            headers=arguments.get("headers"),
            json=arguments.get("body"),
            timeout=30,
        )
        return response.text[:10000]
    except Exception as e:
        return json.dumps({"error": str(e)})


def code_execute_handler(arguments, context):
    """Execute Python or JavaScript code via an external sandbox API."""
    language = arguments.get("language", "")
    code = arguments.get("code", "")
    timeout = arguments.get("timeout", 30)

    if language not in _SUPPORTED_LANGUAGES:
        return json.dumps(
            {
                "error": f"Unsupported language: '{language}'. Must be one of: {sorted(_SUPPORTED_LANGUAGES)}"
            }
        )

    sandbox_url = os.environ.get("CODE_SANDBOX_URL", "")
    api_key = os.environ.get("CODE_SANDBOX_API_KEY", "")

    if not sandbox_url:
        logger.error("CODE_SANDBOX_URL missing from pod env — platform misconfiguration")
        return json.dumps(
            {
                "error": "Code execution is currently unavailable. Please try again later.",
                # Same marker as web_search/web_scrape — tool_registry's
                # billing wrapper skips post_charge so the tenant is not
                # debited for the platform's own misconfiguration.
                "_platform_error": True,
            }
        )

    try:
        response = http_requests.post(
            sandbox_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"language": language, "code": code, "timeout": timeout},
            timeout=timeout + 10,
        )
        if response.status_code != 200:
            return json.dumps(
                {"error": f"Sandbox returned status {response.status_code}: {response.text[:500]}"}
            )
        return json.dumps(response.json())
    except Exception as e:
        logger.warning("code_execute sandbox error: %s", e)
        return json.dumps({"error": f"Sandbox unavailable: {e}"})


# A bucket name the tools accept, matched against the whole name (``$`` would
# also match before a trailing newline). Anything else (a slash, a dot
# segment, an escape) could name a different bucket once the URL is resolved.
_BUCKET_RE = re.compile(r"[A-Za-z0-9_-]+")
_UNSAFE_PATH_CHARS = frozenset("%\\?#")

_STORAGE_TOOL_FAILED = "The storage tool failed; the error has been logged"


def _storage_target_error(bucket, path, *, prefix: bool = False) -> str | None:
    """Why ``bucket``/``path`` may not be used, or None.

    Every path segment must be a plain name: not empty, ``.`` or ``..``, and
    free of ``%``, ``\\``, ``?``, ``#`` and control characters. A list
    ``prefix`` may be empty, or end with ``/``; an object path may not.
    """
    if not isinstance(bucket, str) or not _BUCKET_RE.fullmatch(bucket):
        return "bucket must be a bucket name (letters, digits, '-' and '_')"
    if bucket.lower() == SOURCES_BUCKET:
        return f"The '{SOURCES_BUCKET}' bucket is internal and not available to agents"
    if not isinstance(path, str):
        return "path must be a string"
    if prefix and path == "":
        return None
    segments = path.split("/")
    if prefix and segments[-1] == "":
        segments.pop()  # "reports/2026/" lists a folder
    for segment in segments:
        if (
            segment in ("", ".", "..")
            or any(ch in _UNSAFE_PATH_CHARS for ch in segment)
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in segment)
        ):
            return (
                f"Invalid path {path!r}: each '/'-separated part must be a name, not "
                "empty, '.' or '..', without '%', '\\', '?', '#' or control characters"
            )
    return None


# Sub-delimiters and ":"/"@" are left as they are, as httpx and the Supabase
# clients send them: storage-api signs a key as the request path spells it, so
# a signed URL made from an encoded "+" or "&" would not verify. None of them
# is URL syntax inside a path segment, and "%", "?", "#", a backslash and dot
# segments never get this far (see _storage_target_error).
_PATH_SEGMENT_SAFE = "!$&'()*+,=:@"


def _encode_object_path(path: str) -> str:
    """``path`` with each segment percent-encoded where needed, for an object URL.

    Storage decodes it back to the same key; nothing in it can be read as URL
    syntax on the way. ``;`` is encoded, since some servers read it as a path
    parameter.
    """
    return "/".join(quote(segment, safe=_PATH_SEGMENT_SAFE) for segment in path.split("/"))


_SESSION_EXPIRED_MESSAGE = "The user's session has expired; ask them to sign in again"

# What a storage refusal means to the model, by the status Storage reports.
_STORAGE_REFUSALS = {404: "not found", 403: "access denied"}


def _storage_error_message(
    operation: str, exc: Exception, caller, bucket, path, agent_id=None
) -> str:
    """What a storage tool tells the model: what Storage reported, never the
    response body or an internal URL. The details are logged."""
    if isinstance(exc, StorageError) and exc.status_code:
        refusal = _STORAGE_REFUSALS.get(exc.reported_status)
        if refusal:
            # An everyday answer (a wrong name, the caller's storage policies),
            # not a fault in the deployment.
            logger.info(
                "storage_%s for %r (agent %s) refused (status %s): bucket=%r path=%r",
                operation,
                caller,
                agent_id,
                exc.reported_status,
                bucket,
                path,
            )
            return f"Storage {operation} failed: {refusal}"
        logger.exception(
            "storage_%s for %r (agent %s) failed (HTTP %s): bucket=%r path=%r",
            operation,
            caller,
            agent_id,
            exc.status_code,
            bucket,
            path,
        )
        return f"Storage {operation} failed (HTTP {exc.status_code})"
    logger.exception(
        "storage_%s for %r (agent %s) failed: bucket=%r path=%r",
        operation,
        caller,
        agent_id,
        bucket,
        path,
    )
    return _STORAGE_TOOL_FAILED


def _expired_session_error(caller: ToolCaller, operation: str) -> str | None:
    """The refusal for an end user whose session is over, or None. storage-api
    itself answers an expired token as if the bucket did not exist."""
    if not caller.session_expired():
        return None
    logger.info("storage_%s for %r refused: the session has expired", operation, caller)
    return _SESSION_EXPIRED_MESSAGE


def _storage_for(caller: ToolCaller):
    """Storage as the caller: the end user's own token, or the service role."""
    return get_storage_for_user(caller.token) if caller.is_end_user else get_storage()


def storage_read_handler(arguments, context):
    """List objects in a bucket prefix or download a file from project storage."""
    caller = _pop_caller(arguments)
    agent_id = arguments.pop("_agent_id", None)
    if caller is None:
        return json.dumps({"error": _NO_CALLER_MESSAGE})
    operation = arguments.get("operation", "")
    bucket = arguments.get("bucket", "")
    path = arguments.get("path", "") or ""

    expired = _expired_session_error(caller, "read")
    if expired:
        return json.dumps({"error": expired})
    if not bucket:
        return json.dumps({"error": "bucket is required"})

    if operation == "list":
        invalid = _storage_target_error(bucket, path, prefix=True)
        if invalid:
            return json.dumps({"error": invalid})
        try:
            storage = _storage_for(caller)
            # NOTE: Uses storage._request (private API) — should be replaced with
            # a public list_objects method if the storage service API changes.
            response = storage._request(
                "POST",
                f"/object/list/{bucket}",
                json={"prefix": path, "limit": 1000, "offset": 0},
            )
            if response.status_code != 200:
                raise StorageError.from_response("Failed to list objects", response)
            return json.dumps({"bucket": bucket, "prefix": path, "objects": response.json()})
        except Exception as e:
            return json.dumps(
                {"error": _storage_error_message("list", e, caller, bucket, path, agent_id)}
            )

    elif operation == "download":
        if not path:
            return json.dumps({"error": "path is required for download"})
        invalid = _storage_target_error(bucket, path)
        if invalid:
            return json.dumps({"error": invalid})
        encoded = _encode_object_path(path)
        try:
            storage = _storage_for(caller)
            data = storage.download_from_path(f"{bucket}/{encoded}")
            try:
                content = data.decode("utf-8")
                return json.dumps(
                    {"bucket": bucket, "path": path, "encoding": "utf-8", "content": content}
                )
            except UnicodeDecodeError:
                signed_url = storage.create_signed_url(bucket, encoded)
                return json.dumps(
                    {"bucket": bucket, "path": path, "encoding": "binary", "signed_url": signed_url}
                )
        except Exception as e:
            return json.dumps(
                {"error": _storage_error_message("download", e, caller, bucket, path, agent_id)}
            )

    else:
        return json.dumps({"error": f"Invalid operation: '{operation}'. Must be list or download."})


def storage_write_handler(arguments, context):
    """Upload text content to a project storage bucket."""
    caller = _pop_caller(arguments)
    agent_id = arguments.pop("_agent_id", None)
    if caller is None:
        return json.dumps({"error": _NO_CALLER_MESSAGE})
    bucket = arguments.get("bucket", "")
    path = arguments.get("path", "")
    content = arguments.get("content", "")
    content_type = arguments.get("content_type", "text/plain")

    expired = _expired_session_error(caller, "write")
    if expired:
        return json.dumps({"error": expired})
    if not bucket:
        return json.dumps({"error": "bucket is required"})
    if not path:
        return json.dumps({"error": "path is required"})
    invalid = _storage_target_error(bucket, path)
    if invalid:
        return json.dumps({"error": invalid})
    if not isinstance(content, str):
        return json.dumps({"error": "content must be a string"})

    try:
        encoded = content.encode("utf-8")
        storage = _storage_for(caller)
        storage.upload(bucket, _encode_object_path(path), encoded, content_type)
        return json.dumps({"path": f"{bucket}/{path}", "size": len(encoded)})
    except Exception as e:
        return json.dumps(
            {"error": _storage_error_message("write", e, caller, bucket, path, agent_id)}
        )


def web_search_handler(arguments, context):
    """Search the web using Exa.ai."""
    query = arguments.get("query", "")
    num_results = max(1, min(10, arguments.get("num_results", 5)))
    search_type = arguments.get("search_type", "auto")
    include_domains = arguments.get("include_domains")
    exclude_domains = arguments.get("exclude_domains")
    start_date = arguments.get("start_date")
    end_date = arguments.get("end_date")
    category = arguments.get("category")
    content_mode = arguments.get("content_mode", "highlights")

    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        logger.error("EXA_API_KEY missing from pod env — platform misconfiguration")
        return json.dumps(
            {
                "error": "Web search is currently unavailable. Please try again later.",
                # Marker read by tool_registry._check_and_strip_platform_error
                # — the billing wrapper skips post_charge so the tenant is
                # not debited for a platform-side misconfiguration. The
                # wrapper strips the marker before returning to the agent.
                "_platform_error": True,
            }
        )

    rate_limit = get_setting("EXA_RATE_LIMIT_PER_MINUTE")
    if not external_limiter.acquire_blocking("exa", rate_limit, timeout_s=10.0):
        return json.dumps(
            {
                "error": "Web search is temporarily rate limited. Please try again shortly.",
                "_platform_error": True,
            }
        )

    # Build contents config based on content_mode
    if content_mode == "full_text":
        contents = {"text": True, "highlights": True}
    elif content_mode == "compact_text":
        contents = {"text": {"maxCharacters": 3000}, "highlights": True}
    else:
        contents = {"highlights": True}

    # Build Exa request payload
    payload = {
        "query": query,
        "numResults": num_results,
        "type": search_type,
        "contents": contents,
    }
    if include_domains:
        payload["includeDomains"] = include_domains
    if exclude_domains:
        payload["excludeDomains"] = exclude_domains
    if start_date:
        payload["startPublishedDate"] = start_date
    if end_date:
        payload["endPublishedDate"] = end_date
    if category:
        payload["category"] = category

    try:
        resp = http_requests.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if resp.status_code == 429:
            delay = parse_retry_after(resp.headers.get("Retry-After"))
            if delay <= 15.0:
                time.sleep(delay)
                resp = http_requests.post(
                    "https://api.exa.ai/search",
                    headers={"x-api-key": api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=30,
                )
        if resp.status_code == 429:
            return json.dumps(
                {
                    "error": "Web search is temporarily rate limited. Please try again shortly.",
                    "_platform_error": True,
                }
            )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for r in data.get("results", []):
            result = {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "publishedDate": r.get("publishedDate", ""),
                "highlights": r.get("highlights", []),
            }
            if content_mode in ("full_text", "compact_text") and r.get("text"):
                result["text"] = r["text"]
            results.append(result)

        output = json.dumps(results, default=str)
        max_chars = 50000 if content_mode == "full_text" else 20000
        return output[:max_chars]
    except http_requests.exceptions.RequestException as e:
        # Tenant-fault 4xx (bad query/params) stays billed as anti-gaming;
        # platform-transient failures — 5xx, 429 rate-limits, timeouts, and
        # connection errors — carry the `_platform_error` marker so the billing
        # wrapper skips post_charge. The tenant must not pay (especially the
        # $0.10 deep-reasoning tier, which is the most timeout-prone) for a
        # result they never received.
        status = getattr(getattr(e, "response", None), "status_code", None)
        tenant_fault = status is not None and 400 <= status < 500 and status != 429
        logger.warning("web_search request error (status=%s): %s", status, e)
        err: dict = {"error": str(e)}
        if not tenant_fault:
            err["_platform_error"] = True
        return json.dumps(err)
    except Exception as e:
        # Unexpected (e.g. malformed response) — not a tenant fault, don't bill.
        logger.warning("web_search error: %s", e)
        return json.dumps({"error": str(e), "_platform_error": True})


_IMAGE_RE = re.compile(r"(!\[([^\]]*)\]\(([^)]+)\))")

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"})


def _is_direct_image_url(url: str) -> bool:
    """Check if URL points directly to an image file based on extension."""
    from urllib.parse import urlparse

    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _IMAGE_EXTENSIONS)


def _analyze_single_image(image_url, model, timeout):
    """Call vision LLM for a single image URL. Returns description or error placeholder."""
    try:
        with with_llm_key(model) as api_key:
            response = litellm.completion(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail."},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                timeout=timeout,
                api_key=api_key,
            )
        return response.choices[0].message.content or "[No description returned]"
    except Exception as e:
        logger.warning("Vision analysis failed for %s: %s", image_url, e)
        return "[Image could not be analyzed]"


def web_scrape_handler(arguments, context):
    """Scrape web page content using Firecrawl, optionally analyzing images inline."""
    url = arguments.get("url", "")
    formats = arguments.get("formats") or ["markdown"]
    include_images = arguments.get("include_images", False)
    only_main_content = arguments.get("only_main_content", True)
    exclude_tags = arguments.get("exclude_tags")
    wait_for = arguments.get("wait_for")
    mobile = arguments.get("mobile", False)

    # Direct image URL → skip Firecrawl, go straight to vision analysis
    if _is_direct_image_url(url):
        vision_model = get_setting("VISION_MODEL") or "gpt-4.1-mini"
        vision_timeout = get_setting("VISION_TIMEOUT") or 30
        max_chars = get_setting("WEB_SCRAPE_MAX_CHARS") or 200000
        description = _analyze_single_image(url, vision_model, vision_timeout)
        result = {
            "metadata": {"sourceURL": url, "type": "image"},
            "markdown": f"![image]({url})\n\n> **[Image description]:** {description}",
        }
        return json.dumps(result, default=str)[:max_chars]

    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not api_key:
        logger.error("FIRECRAWL_API_KEY missing from pod env — platform misconfiguration")
        return json.dumps(
            {
                "error": "Web scraping is currently unavailable. Please try again later.",
                "_platform_error": True,
            }
        )

    rate_limit = get_setting("FIRECRAWL_RATE_LIMIT_PER_MINUTE")
    if not external_limiter.acquire_blocking("firecrawl", rate_limit, timeout_s=10.0):
        return json.dumps(
            {
                "error": "Web scraping is temporarily rate limited. Please try again shortly.",
                "_platform_error": True,
            }
        )

    max_chars = get_setting("WEB_SCRAPE_MAX_CHARS") or 200000
    firecrawl_base = get_setting("FIRECRAWL_API_BASE").rstrip("/")

    # Build Firecrawl request payload
    payload = {"url": url, "formats": formats, "onlyMainContent": only_main_content}
    if exclude_tags:
        payload["excludeTags"] = exclude_tags
    if wait_for is not None:
        payload["waitFor"] = wait_for
    if mobile:
        payload["mobile"] = True

    try:
        resp = http_requests.post(
            f"{firecrawl_base}/scrape",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if resp.status_code == 429:
            delay = parse_retry_after(resp.headers.get("Retry-After"))
            if delay <= 15.0:
                time.sleep(delay)
                resp = http_requests.post(
                    f"{firecrawl_base}/scrape",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60,
                )
        if resp.status_code == 429:
            return json.dumps(
                {
                    "error": "Web scraping is temporarily rate limited. Please try again shortly.",
                    "_platform_error": True,
                }
            )
        resp.raise_for_status()
        data = resp.json().get("data", {})
    except Exception as e:
        logger.warning("web_scrape error: %s", e)
        return json.dumps({"error": str(e)})

    if not include_images:
        output = json.dumps(data, default=str)
        return output[:max_chars]

    # --- Image analysis mode ---
    markdown = data.get("markdown", "")
    if not markdown:
        output = json.dumps(data, default=str)
        return output[:max_chars]

    matches = _IMAGE_RE.findall(markdown)
    if not matches:
        output = json.dumps(data, default=str)
        return output[:max_chars]

    max_images = get_setting("WEB_SCRAPE_MAX_IMAGES") or 10
    vision_model = get_setting("VISION_MODEL") or "gpt-4.1-mini"
    vision_timeout = get_setting("VISION_TIMEOUT") or 30
    max_workers = get_setting("VISION_MAX_WORKERS") or 3

    # Deduplicate by URL while preserving order, cap at max_images
    seen_urls = set()
    unique_matches = []
    for full_match, alt, img_url in matches:
        if img_url not in seen_urls:
            seen_urls.add(img_url)
            unique_matches.append((full_match, alt, img_url))
        if len(unique_matches) >= max_images:
            break

    # Analyze images concurrently
    descriptions = {}
    # PR 421 R4 C8: vision OCR fires litellm.completion inside each worker.
    # ThreadPoolExecutor.submit() does NOT propagate caller contextvars
    # (current_byok_providers, byok_lookup_degraded, run_id_var),
    # so BillingLogger's BYOK skip reads frozenset() and charges the
    # OCR call against AI-on-us even when the project has a valid BYOK
    # key. Snapshot the parent context per-submission and wrap submission
    # in ctx.run — mirrors the pattern in agentic/agent/agent.py:824.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(
                contextvars.copy_context().run,
                _analyze_single_image,
                img_url,
                vision_model,
                vision_timeout,
            ): img_url
            for _, _, img_url in unique_matches
        }
        for future in as_completed(future_to_url):
            img_url = future_to_url[future]
            descriptions[img_url] = future.result()

    # Replace image references by walking matches in reverse so earlier
    # replacements don't shift the positions of later ones.  We use the
    # match *spans* from re.finditer to avoid the subtle bug where
    # str.replace would match inside already-enriched text.
    spans = list(_IMAGE_RE.finditer(markdown))
    for m in reversed(spans):
        img_url = m.group(3)
        if img_url in descriptions:
            desc = descriptions[img_url]
            # Wrap every line in a blockquote so multi-line descriptions
            # render correctly in markdown.
            quoted = "\n> ".join(desc.split("\n"))
            enriched = f"{m.group(0)}\n\n> **[Image description]:** {quoted}"
            markdown = markdown[: m.start()] + enriched + markdown[m.end() :]

    data["markdown"] = markdown
    output = json.dumps(data, default=str)
    return output[:max_chars]


BUILTIN_HANDLERS = {
    "database_write": database_write_handler,
    "database_query": database_query_handler,
    "http_request": http_request_handler,
    "code_execute": code_execute_handler,
    "storage_read": storage_read_handler,
    "storage_write": storage_write_handler,
    "web_search": web_search_handler,
    "web_scrape": web_scrape_handler,
}
