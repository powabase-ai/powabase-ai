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
        else:
            err = f"runtime_tools type {entry_type!r} not yet supported"
        if err:
            return [], err
        normalized_entries.append(entry)

    return normalized_entries, None
