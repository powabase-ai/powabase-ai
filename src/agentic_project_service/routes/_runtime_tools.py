"""Validation for the per-request ``runtime_tools`` field.

Deliberately strict (mirroring ``_runtime_kb``): unknown entry types, unknown
or runtime-blocked builtin names, malformed entries, and oversized lists all
400 before any stream opens.

Error messages must never echo header values — headers may carry secrets;
name the offending key instead.
"""

import re
import uuid
from urllib.parse import urlparse

from sqlalchemy import text

from ..db import AI_SCHEMA
from ..tools.builtin import BUILTIN_TOOL_DEFINITIONS

RUNTIME_TOOLS_MAX_ENTRIES = 10

# The superuser-connection and code-execution builtins stay agent-config-only:
# granting them is a deliberate, auditable attachment, not a request field.
RUNTIME_BLOCKED_BUILTINS = {"database_query", "database_write", "code_execute"}

_BUILTIN_DEFS = {d["name"]: d for d in BUILTIN_TOOL_DEFINITIONS}

_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}

# MCP server names land inside generated tool names (mcp__{name}__{tool});
# constrain them to a safe identifier.
_MCP_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_ALLOWED_KEYS_BY_TYPE = {
    "builtin": {"type", "name", "config_override"},
    "custom": {"type", "tool_id", "definition"},
    "mcp": {"type", "name", "url", "transport", "headers"},
}

_ALLOWED_DEFINITION_KEYS = {"name", "description", "input_schema", "config"}
_ALLOWED_DEFINITION_CONFIG_KEYS = {"endpoint", "method", "headers", "timeout_seconds"}


def _normalize_uuid(value):
    """Parse a UUID and return its canonical (lowercase, dashed) string form.

    Non-canonical forms parse but break set-comparison against the DB and the
    later builder lookup by string key — silently dropping the tool. Raises
    ValueError/TypeError on anything unparseable.
    """
    return str(uuid.UUID(str(value)))


def _validate_builtin_entry(entry):
    name = entry.get("name")
    if not name or not isinstance(name, str):
        return "each builtin runtime_tools entry must have a 'name'"
    if name in RUNTIME_BLOCKED_BUILTINS:
        return (
            f"builtin tool {name!r} cannot be attached at runtime "
            f"(blocked: {', '.join(sorted(RUNTIME_BLOCKED_BUILTINS))}); "
            "attach it to the agent instead"
        )
    defn = _BUILTIN_DEFS.get(name)
    if defn is None:
        return f"unknown builtin tool: {name!r}"
    override = entry.get("config_override")
    if override is not None:
        if not isinstance(override, dict):
            return f"'config_override' must be an object: {override!r}"
        schema_props = (defn.get("input_schema") or {}).get("properties", {})
        unknown = set(override) - set(schema_props)
        if unknown:
            return f"config_override key(s) not in {name!r} input schema: {sorted(unknown)}"
    return None


def _validate_http_url(value, label):
    if not value or not isinstance(value, str):
        return f"{label} is required and must be an http(s) URL"
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return f"{label} must be an http(s) URL: {value!r}"
    return None


def _validate_headers(value, label):
    """Headers may carry secrets: on rejection, name the key, never the value."""
    if value is None:
        return None
    if not isinstance(value, dict):
        return f"{label} 'headers' must be an object"
    for k, v in value.items():
        if not isinstance(k, str) or not isinstance(v, str):
            return f"{label} 'headers' entries must map string keys to string values (key: {k!r})"
    return None


def _validate_definition(definition):
    if not isinstance(definition, dict):
        return "'definition' must be an object"
    unknown_keys = set(definition) - _ALLOWED_DEFINITION_KEYS
    if unknown_keys:
        return f"unknown key(s) in runtime tool definition: {sorted(unknown_keys)}"
    name = definition.get("name")
    if not name or not isinstance(name, str):
        return "runtime tool definition requires a non-empty 'name'"
    if name in _BUILTIN_DEFS:
        return f"runtime tool definition name {name!r} shadows a builtin tool"
    description = definition.get("description")
    if not description or not isinstance(description, str):
        return f"runtime tool definition {name!r} requires a non-empty 'description'"
    input_schema = definition.get("input_schema")
    if not isinstance(input_schema, dict) or input_schema.get("type") != "object":
        return (
            f"runtime tool definition {name!r} requires an 'input_schema' object "
            'with "type": "object"'
        )
    config = definition.get("config")
    if not isinstance(config, dict):
        return f"runtime tool definition {name!r} requires a 'config' object"
    unknown_keys = set(config) - _ALLOWED_DEFINITION_CONFIG_KEYS
    if unknown_keys:
        return f"unknown key(s) in runtime tool definition config: {sorted(unknown_keys)}"
    err = _validate_http_url(config.get("endpoint"), f"definition {name!r} config 'endpoint'")
    if err:
        return err
    method = config.get("method")
    if method is not None and method not in _HTTP_METHODS:
        return f"invalid method in runtime tool definition {name!r}: {method!r}"
    err = _validate_headers(config.get("headers"), f"definition {name!r} config")
    if err:
        return err
    # The offending value is deliberately omitted from the message — uniform
    # no-echo habit for everything inside a definition config.
    timeout = config.get("timeout_seconds")
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not (1 <= timeout <= 600):
            return (
                f"'timeout_seconds' in runtime tool definition {name!r} must be "
                "an integer between 1 and 600"
            )
    return None


def _validate_custom_entry(entry):
    has_ref = entry.get("tool_id") is not None
    has_def = entry.get("definition") is not None
    if has_ref == has_def:
        return "each custom runtime_tools entry must have exactly one of 'tool_id' or 'definition'"
    if has_ref:
        try:
            entry["tool_id"] = _normalize_uuid(entry["tool_id"])
        except (ValueError, AttributeError, TypeError):
            return f"invalid tool id: {entry.get('tool_id')!r}"
        return None
    return _validate_definition(entry["definition"])


def validate_runtime_tools(data, db_session, ai_schema: str = AI_SCHEMA):
    raw = data.get("runtime_tools")
    if raw is None:
        return [], None
    if not isinstance(raw, list) or not raw:
        return [], "'runtime_tools' must be a non-empty list of objects"
    if len(raw) > RUNTIME_TOOLS_MAX_ENTRIES:
        return [], f"'runtime_tools' accepts at most {RUNTIME_TOOLS_MAX_ENTRIES} entries"

    normalized_entries = []
    for entry in raw:
        if not isinstance(entry, dict) or entry.get("type") not in _ALLOWED_KEYS_BY_TYPE:
            return [], (
                "each 'runtime_tools' entry must be an object with a 'type' of "
                "'builtin', 'custom', or 'mcp'"
            )
        entry_type = entry["type"]
        unknown_keys = set(entry) - _ALLOWED_KEYS_BY_TYPE[entry_type]
        if unknown_keys:
            return [], (
                f"unknown key(s) in runtime_tools {entry_type} entry: {sorted(unknown_keys)}"
            )
        entry = dict(entry)  # normalized copy; never mutate the caller's object
        if entry_type == "builtin":
            err = _validate_builtin_entry(entry)
        elif entry_type == "custom":
            err = _validate_custom_entry(entry)
        else:
            err = f"runtime_tools type {entry_type!r} not yet supported"
        if err:
            return [], err
        normalized_entries.append(entry)

    ref_entries = [e for e in normalized_entries if e["type"] == "custom" and "tool_id" in e]
    seen_tool_ids: set[str] = set()
    for e in ref_entries:
        if e["tool_id"] in seen_tool_ids:
            return [], f"duplicate tool id: {e['tool_id']}"
        seen_tool_ids.add(e["tool_id"])
    if ref_entries:
        ids = [e["tool_id"] for e in ref_entries]
        rows = db_session.execute(
            text(f'SELECT id, name FROM "{ai_schema}".tools WHERE id = ANY(:ids)'),
            {"ids": ids},
        )
        name_by_id = {str(row[0]): row[1] for row in rows}
        missing = [i for i in ids if i not in name_by_id]
        if missing:
            return [], f"unknown tool id(s): {', '.join(missing)}"
        for e in ref_entries:
            e["_resolved_name"] = name_by_id[e["tool_id"]]

    return normalized_entries, None
