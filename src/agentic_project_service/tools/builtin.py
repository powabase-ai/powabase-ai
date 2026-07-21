"""Built-in tool implementations and definitions."""

import contextvars
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
import requests as http_requests
from sqlalchemy import text

from ..db import db
from ..services.llm_call import with_llm_key
from ..services.settings_registry import get_setting
from ..services.storage import get_storage

logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")


def _validate_identifier(name, label="identifier"):
    if not _IDENTIFIER_RE.match(name):
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


def database_write_handler(arguments, context):
    """Perform a structured INSERT, UPDATE, or DELETE on the configured schema(s)."""
    original_table = arguments.get("table", "")
    operation = arguments.get("operation", "")
    data = arguments.get("data") or {}
    where = arguments.get("where") or {}
    allowed_schemas = arguments.pop("_allowed_schemas", ["public"])
    allowed_tables = arguments.pop("_allowed_tables", None)
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
    if schemas_config:
        schema, table, err = _resolve_table(original_table, schemas_config)
        if err:
            return json.dumps({"success": False, "message": err})
    else:
        # Fallback to legacy flat set check
        table = original_table
        schema = None
        if "." in table:
            parts = table.split(".", 1)
            schema, table = parts[0], parts[1]
            # Validate schema is in the allowed list
            if schema not in allowed_schemas:
                return json.dumps(
                    {
                        "success": False,
                        "message": f"Schema '{schema}' is not allowed. Allowed schemas: {sorted(allowed_schemas)}",
                    }
                )
        if allowed_tables is not None and table not in allowed_tables:
            return json.dumps(
                {
                    "success": False,
                    "message": f"Table '{original_table}' is not in the configured access list. Available: {sorted(allowed_tables)}",
                }
            )

    # Defense-in-depth: validate schema names
    effective_schemas = [schema] if schema else allowed_schemas
    for s in effective_schemas:
        if not _IDENTIFIER_RE.match(s):
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

    try:
        search_path = ", ".join(f'"{s}"' for s in effective_schemas)
        db.session.execute(text(f"SET LOCAL search_path TO {search_path}"))

        if operation == "insert":
            # Strip auto-generated columns (SERIAL, IDENTITY) to prevent sequence desync
            auto_gen_cols = set()
            try:
                schema_name = effective_schemas[0] if effective_schemas else "public"
                db.session.execute(text("SAVEPOINT _autogen_check"))
                result = db.session.execute(
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
                db.session.execute(text("RELEASE SAVEPOINT _autogen_check"))
            except Exception:
                db.session.execute(text("ROLLBACK TO SAVEPOINT _autogen_check"))
                # Introspection failed; proceed without stripping

            if auto_gen_cols:
                rows = [{k: v for k, v in row.items() if k not in auto_gen_cols} for row in rows]
                if not rows or not any(row for row in rows):
                    return json.dumps(
                        {
                            "success": False,
                            "message": "INSERT data contains only auto-generated columns — nothing to insert.",
                        }
                    )
                # Recompute columns from the stripped rows
                columns = list(rows[0].keys())

            col_sql = ", ".join(f'"{k}"' for k in columns)
            placeholders = ", ".join(f":{k}" for k in columns)
            sql = f'INSERT INTO "{table}" ({col_sql}) VALUES ({placeholders})'
            total_affected = 0
            for row in rows:
                result = db.session.execute(text(sql), row)
                total_affected += result.rowcount

        elif operation == "update":
            update_data = rows[0]  # update uses single object
            set_clause = ", ".join(f'"{k}" = :set_{k}' for k in update_data.keys())
            where_clause = " AND ".join(f'"{k}" = :where_{k}' for k in where.keys())
            params = {f"set_{k}": v for k, v in update_data.items()}
            params.update({f"where_{k}": v for k, v in where.items()})
            sql = f'UPDATE "{table}" SET {set_clause} WHERE {where_clause}'
            result = db.session.execute(text(sql), params)
            total_affected = result.rowcount

        else:  # delete
            where_clause = " AND ".join(f'"{k}" = :where_{k}' for k in where.keys())
            params = {f"where_{k}": v for k, v in where.items()}
            sql = f'DELETE FROM "{table}" WHERE {where_clause}'
            result = db.session.execute(text(sql), params)
            total_affected = result.rowcount

        db.session.commit()
        return json.dumps(
            {
                "success": True,
                "rows_affected": total_affected,
                "message": f"{operation} completed.",
            }
        )

    except Exception as e:
        db.session.rollback()
        return json.dumps({"success": False, "message": str(e)})


def database_query_handler(arguments, context):
    """Run read-only SQL against the project's Postgres."""
    sql = arguments.get("query", "").strip().rstrip(";").strip()
    allowed_schemas = arguments.pop("_allowed_schemas", ["public"])
    schemas_config = arguments.pop("_schemas_config", {})
    arguments.pop("_allowed_tables", None)

    # Use schemas_config if available, otherwise fall back to allowed_schemas
    effective_schemas = list(schemas_config.keys()) if schemas_config else allowed_schemas

    # Defense-in-depth: validate schema names
    for s in effective_schemas:
        if not _IDENTIFIER_RE.match(s):
            return json.dumps({"error": f"Invalid schema name: {s}"})

    if not sql.upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries are allowed"})

    # Block obvious multi-statement attacks (after stripping trailing semicolons)
    if ";" in sql:
        return json.dumps({"error": "Multi-statement queries are not allowed"})

    try:
        search_path = ", ".join(f'"{s}"' for s in effective_schemas)
        db.session.execute(text(f"SET LOCAL search_path TO {search_path}"))
        result = db.session.execute(text(sql))
        rows = [dict(row._mapping) for row in result]
        output = json.dumps(rows, default=str)
        db.session.rollback()
        return output[:50000]
    except Exception as e:
        db.session.rollback()
        return json.dumps({"error": str(e)})


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


def storage_read_handler(arguments, context):
    """List objects in a bucket prefix or download a file from project storage."""
    operation = arguments.get("operation", "")
    bucket = arguments.get("bucket", "")
    path = arguments.get("path", "") or ""

    if not bucket:
        return json.dumps({"error": "bucket is required"})

    if operation == "list":
        try:
            storage = get_storage()
            # NOTE: Uses storage._request (private API) — should be replaced with
            # a public list_objects method if the storage service API changes.
            response = storage._request(
                "POST",
                f"/object/list/{bucket}",
                json={"prefix": path, "limit": 1000, "offset": 0},
            )
            if response.status_code != 200:
                return json.dumps({"error": f"Failed to list objects: {response.text}"})
            return json.dumps({"bucket": bucket, "prefix": path, "objects": response.json()})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif operation == "download":
        if not path:
            return json.dumps({"error": "path is required for download"})
        try:
            storage = get_storage()
            data = storage.download_from_path(f"{bucket}/{path}")
            try:
                content = data.decode("utf-8")
                return json.dumps(
                    {"bucket": bucket, "path": path, "encoding": "utf-8", "content": content}
                )
            except UnicodeDecodeError:
                signed_url = storage.create_signed_url(bucket, path)
                return json.dumps(
                    {"bucket": bucket, "path": path, "encoding": "binary", "signed_url": signed_url}
                )
        except Exception as e:
            return json.dumps({"error": str(e)})

    else:
        return json.dumps({"error": f"Invalid operation: '{operation}'. Must be list or download."})


def storage_write_handler(arguments, context):
    """Upload text content to a project storage bucket."""
    bucket = arguments.get("bucket", "")
    path = arguments.get("path", "")
    content = arguments.get("content", "")
    content_type = arguments.get("content_type", "text/plain")

    if not bucket:
        return json.dumps({"error": "bucket is required"})
    if not path:
        return json.dumps({"error": "path is required"})

    try:
        encoded = content.encode("utf-8")
        storage = get_storage()
        storage_path = storage.upload(bucket, path, encoded, content_type)
        return json.dumps({"path": storage_path, "size": len(encoded)})
    except Exception as e:
        return json.dumps({"error": str(e)})


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
