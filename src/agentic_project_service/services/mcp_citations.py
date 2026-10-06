"""Citation keys for MCP tool results.

An agent's MCP server opts in with a ``citation_mapping``:

    {"<tool name as the server advertises it>": {
        "items": "<JSONPath>" | "none",
        "title": "<dotted field>",   # optional, read from each matched item
        "url": "<dotted field>",     # optional
    }}

* A mapped tool whose result parses as JSON gets one unit per JSON object the
  path matches. Each matched object gains ``"cite": "[n]"``.
* ``"items": "none"`` leaves the tool's results unkeyed.
* A tool without an entry, a non-JSON body, or a path matching no object makes
  the whole result one unit, prefixed ``[n] ``. An empty mapping ``{}``
  therefore keys every tool by call.

Only a subset of JSONPath is supported: ``$``, then any of ``.name``,
``['name']``, ``[n]``, ``[*]`` and ``.*``. Anything else is rejected when the
mapping is written.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from .citation_registry import KIND_TOOL_CALL, KIND_TOOL_ITEM, CitationRegistry, CitationUnit
from .citations import TEXT_EXCERPT_MAX_LEN

CITE_FIELD = "cite"
ITEMS_NONE = "none"

_RULE_KEYS = frozenset({"items", "title", "url"})
_ARGUMENT_SUMMARY_MAX = 80
# What the engine's MCP client returns instead of raising (agentic.mcp.client).
_EMPTY_RESULT = "(empty response)"
_ENGINE_ERROR_RE = re.compile(r"^Error(?: \([^)]*\))?: |^Error calling MCP tool: ")
_SEGMENT_RE = re.compile(
    r"(?P<star>\.\*|\[\*\])"
    r"|\.(?P<name>[A-Za-z_][A-Za-z0-9_-]*)"
    r"|\[(?P<index>-?\d+)\]"
    r"|\['(?P<quoted>[^']+)'\]"
)

Segment = tuple[str, str | int | None]


def compile_items_path(path: str) -> tuple[Segment, ...]:
    if not isinstance(path, str) or not path.startswith("$"):
        raise ValueError(f"JSONPath must start with '$': {path!r}")
    segments: list[Segment] = []
    pos = 1
    while pos < len(path):
        match = _SEGMENT_RE.match(path, pos)
        if match is None:
            raise ValueError(
                f"unsupported JSONPath syntax at position {pos} in {path!r}; "
                "supported: $, .name, ['name'], [n], [*], .*"
            )
        if match.group("star") is not None:
            segments.append(("star", None))
        elif match.group("name") is not None:
            segments.append(("key", match.group("name")))
        elif match.group("quoted") is not None:
            segments.append(("key", match.group("quoted")))
        else:
            segments.append(("index", int(match.group("index"))))
        pos = match.end()
    return tuple(segments)


def select_objects(document: Any, segments: tuple[Segment, ...]) -> list[dict]:
    nodes = [document]
    for kind, arg in segments:
        following: list[Any] = []
        for node in nodes:
            if kind == "key":
                if isinstance(node, dict) and arg in node:
                    following.append(node[arg])
            elif kind == "index":
                if isinstance(node, list) and -len(node) <= arg < len(node):
                    following.append(node[arg])
            elif isinstance(node, list):
                following.extend(node)
            elif isinstance(node, dict):
                following.extend(node.values())
        nodes = following
    return [node for node in nodes if isinstance(node, dict)]


def _valid_dotted(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and all(value.split("."))


def citation_mapping_error(mapping: Any) -> str | None:
    """Why ``mapping`` is not a valid citation_mapping, or None if it is."""
    if mapping is None:
        return None
    if not isinstance(mapping, dict):
        return "citation_mapping must be an object or null"
    for tool, rule in mapping.items():
        if not tool:
            return "citation_mapping keys must be non-empty tool names"
        where = f"citation_mapping[{tool!r}]"
        if not isinstance(rule, dict):
            return f"{where} must be an object"
        unknown = sorted(set(rule) - _RULE_KEYS)
        if unknown:
            return f"{where} has unknown key(s) {unknown}; allowed: items, title, url"
        items = rule.get("items")
        if not isinstance(items, str):
            return f'{where}.items is required: a JSONPath or "none"'
        if items != ITEMS_NONE:
            try:
                compile_items_path(items)
            except ValueError as exc:
                return f"{where}.items: {exc}"
        for name in ("title", "url"):
            if name in rule and not _valid_dotted(rule[name]):
                return f'{where}.{name} must be a dotted field name like "meta.title"'
    return None


def _field(obj: dict, dotted: str | None) -> str | None:
    if not dotted:
        return None
    value: Any = obj
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value if isinstance(value, str) and value else None


def _key_items(
    result: str, *, tool_name: str, rule: dict, registry: CitationRegistry, call_id: str | None
) -> str | None:
    try:
        document = json.loads(result)
    except ValueError:
        return None
    matches = select_objects(document, compile_items_path(rule["items"]))
    if not matches:
        return None
    units = []
    for match in matches:
        raw = copy.deepcopy(match)
        units.append(
            CitationUnit(
                kind=KIND_TOOL_ITEM,
                tool_name=tool_name,
                call_id=call_id,
                title=_field(match, rule.get("title")),
                url=_field(match, rule.get("url")),
                text_excerpt=json.dumps(raw, ensure_ascii=False)[:TEXT_EXCERPT_MAX_LEN],
                meta=raw,
            )
        )
    for match, key in zip(matches, registry.register_many(units)):
        match[CITE_FIELD] = f"[{key}]"
    return json.dumps(document, ensure_ascii=False)


def _argument_summary(arguments: dict | None) -> str:
    summary = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True, default=str)
    if len(summary) > _ARGUMENT_SUMMARY_MAX:
        summary = summary[: _ARGUMENT_SUMMARY_MAX - 1] + "…"
    return summary


def key_mcp_result(
    result: Any,
    *,
    tool_name: str,
    arguments: dict | None,
    rule: dict | None,
    registry: CitationRegistry,
    call_id: str | None,
) -> Any:
    """Register citation units for one MCP tool result; return what the model sees."""
    if not isinstance(result, str) or result == _EMPTY_RESULT or _ENGINE_ERROR_RE.match(result):
        return result
    if rule is not None:
        if rule.get("items") == ITEMS_NONE:
            return result
        keyed = _key_items(
            result, tool_name=tool_name, rule=rule, registry=registry, call_id=call_id
        )
        if keyed is not None:
            return keyed
    key = registry.register(
        CitationUnit(
            kind=KIND_TOOL_CALL,
            tool_name=tool_name,
            call_id=call_id,
            title=f"{tool_name} {_argument_summary(arguments)}",
            text_excerpt=result[:TEXT_EXCERPT_MAX_LEN],
            meta={"raw": result, "arguments": arguments or {}},
        )
    )
    return f"[{key}] {result}"
