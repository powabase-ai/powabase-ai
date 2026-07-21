"""Database CRUD routes — proxy to PostgREST via Kong."""

import logging
import os
import re

import httpx
from flask import Blueprint, Response, jsonify, request

from ..auth import require_auth

logger = logging.getLogger(__name__)

database_bp = Blueprint("database", __name__, url_prefix="/api/database")

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Only these schemas are accessible via user-facing database routes.
# The `ai` schema is internal (managed by dedicated /api/agents, /api/workflows routes).
ALLOWED_SCHEMAS = frozenset({"public"})

SYSTEM_SCHEMAS = frozenset(
    {
        "ai",
        "auth",
        "storage",
        "extensions",
        "graphql",
        "graphql_public",
        "realtime",
        "vault",
        "pgsodium",
        "pgsodium_masks",
        "supabase_functions",
        "supabase_migrations",
        "pg_catalog",
        "information_schema",
        "pg_toast",
        "net",
        "cron",
    }
)


def _kong_url() -> str:
    return os.getenv("SUPABASE_URL", "http://kong:8000").rstrip("/")


def _service_key() -> str:
    return os.getenv("SERVICE_ROLE_KEY", "")


def _postgrest_headers(*, prefer: str | None = None, schema: str = "public") -> dict[str, str]:
    key = _service_key()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Profile": schema,
        "Content-Profile": schema,
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _validate_table(table: str) -> str | None:
    """Return an error message if the table name is invalid, else None."""
    if not table or not _TABLE_NAME_RE.match(table):
        return f"Invalid table name: {table!r}"
    return None


def _validate_schema(schema: str) -> str | None:
    """Return an error message if the schema name is invalid or not allowed."""
    if not schema or not _TABLE_NAME_RE.match(schema):
        return f"Invalid schema name: {schema!r}"
    if schema not in ALLOWED_SCHEMAS:
        return f"Schema '{schema}' is not accessible"
    return None


@database_bp.route("/tables", methods=["GET"])
@require_auth
def list_tables():
    """List tables in the given schema (read-only SQL on information_schema)."""
    from ..db import db
    from sqlalchemy import text

    schema = request.args.get("schema", "public")
    err = _validate_schema(schema)
    if err:
        return jsonify({"error": err}), 400

    try:
        result = db.session.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = :schema AND table_type = 'BASE TABLE' "
                "ORDER BY table_name"
            ),
            {"schema": schema},
        )
        tables = [row[0] for row in result.fetchall()]
        return jsonify({"tables": tables})
    except Exception as e:
        logger.error("Failed to list tables: %s", e)
        return jsonify({"error": str(e)}), 500


@database_bp.route("/tables/<table>", methods=["GET"])
@require_auth
def list_rows(table: str):
    """List rows from a table via PostgREST (supports ?limit=&offset=&schema=)."""
    err = _validate_table(table)
    if err:
        return jsonify({"error": err}), 400

    schema = request.args.get("schema", "public")
    err = _validate_schema(schema)
    if err:
        return jsonify({"error": err}), 400

    try:
        limit = int(request.args.get("limit", "50"))
    except (ValueError, TypeError):
        limit = 50
    try:
        offset = int(request.args.get("offset", "0"))
    except (ValueError, TypeError):
        offset = 0

    url = f"{_kong_url()}/rest/v1/{table}"
    try:
        resp = httpx.get(
            url,
            headers=_postgrest_headers(schema=schema),
            params={"limit": str(limit), "offset": str(offset)},
            timeout=15,
        )
        return Response(resp.content, status=resp.status_code, mimetype="application/json")
    except Exception as e:
        logger.error("list_rows failed: %s", e)
        return jsonify({"error": str(e)}), 502


@database_bp.route("/tables/<table>/<row_id>", methods=["GET"])
@require_auth
def get_row(table: str, row_id: str):
    """Get a single row by id."""
    err = _validate_table(table)
    if err:
        return jsonify({"error": err}), 400

    schema = request.args.get("schema", "public")
    err = _validate_schema(schema)
    if err:
        return jsonify({"error": err}), 400

    url = f"{_kong_url()}/rest/v1/{table}"
    headers = _postgrest_headers(schema=schema)
    headers["Accept"] = "application/vnd.pgrst.object+json"
    try:
        resp = httpx.get(url, headers=headers, params={"id": f"eq.{row_id}"}, timeout=15)
        return Response(resp.content, status=resp.status_code, mimetype="application/json")
    except Exception as e:
        logger.error("get_row failed: %s", e)
        return jsonify({"error": str(e)}), 502


@database_bp.route("/tables/<table>", methods=["POST"])
@require_auth
def create_row(table: str):
    """Create a row via PostgREST."""
    err = _validate_table(table)
    if err:
        return jsonify({"error": err}), 400

    schema = request.args.get("schema", "public")
    err = _validate_schema(schema)
    if err:
        return jsonify({"error": err}), 400

    body = request.get_data()
    url = f"{_kong_url()}/rest/v1/{table}"
    try:
        resp = httpx.post(
            url,
            headers=_postgrest_headers(prefer="return=representation", schema=schema),
            content=body,
            timeout=15,
        )
        return Response(resp.content, status=resp.status_code, mimetype="application/json")
    except Exception as e:
        logger.error("create_row failed: %s", e)
        return jsonify({"error": str(e)}), 502


@database_bp.route("/tables/<table>/<row_id>", methods=["PATCH"])
@require_auth
def update_row(table: str, row_id: str):
    """Update a row by id via PostgREST."""
    err = _validate_table(table)
    if err:
        return jsonify({"error": err}), 400

    schema = request.args.get("schema", "public")
    err = _validate_schema(schema)
    if err:
        return jsonify({"error": err}), 400

    body = request.get_data()
    url = f"{_kong_url()}/rest/v1/{table}"
    try:
        resp = httpx.patch(
            url,
            headers=_postgrest_headers(prefer="return=representation", schema=schema),
            params={"id": f"eq.{row_id}"},
            content=body,
            timeout=15,
        )
        return Response(resp.content, status=resp.status_code, mimetype="application/json")
    except Exception as e:
        logger.error("update_row failed: %s", e)
        return jsonify({"error": str(e)}), 502


@database_bp.route("/tables/<table>/<row_id>", methods=["DELETE"])
@require_auth
def delete_row(table: str, row_id: str):
    """Delete a row by id via PostgREST."""
    err = _validate_table(table)
    if err:
        return jsonify({"error": err}), 400

    schema = request.args.get("schema", "public")
    err = _validate_schema(schema)
    if err:
        return jsonify({"error": err}), 400

    url = f"{_kong_url()}/rest/v1/{table}"
    try:
        resp = httpx.delete(
            url,
            headers=_postgrest_headers(schema=schema),
            params={"id": f"eq.{row_id}"},
            timeout=15,
        )
        return Response(resp.content, status=resp.status_code, mimetype="application/json")
    except Exception as e:
        logger.error("delete_row failed: %s", e)
        return jsonify({"error": str(e)}), 502


@database_bp.route("/openapi", methods=["GET"])
@require_auth
def get_openapi_spec():
    """Return the project's PostgREST OpenAPI/Swagger spec.

    FE consumers (api-docs tab, settings/api) use this to render the
    PostgREST reference. PostgREST serves the OpenAPI spec at its
    /rest/v1/ root by default.
    """
    url = f"{_kong_url()}/rest/v1/"
    try:
        resp = httpx.get(
            url,
            headers=_postgrest_headers(),
            timeout=15,
        )
        return Response(resp.content, status=resp.status_code, mimetype="application/json")
    except Exception as e:
        logger.error("get_openapi_spec failed: %s", e)
        return jsonify({"error": str(e)}), 502
