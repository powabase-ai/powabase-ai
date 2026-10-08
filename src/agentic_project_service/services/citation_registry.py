"""Run-wide citation registry for agent runs.

Every unit a run can cite gets the next integer key, starting at 1. A unit is
a knowledge-base chunk, an item inside a tool's JSON result, or a whole tool
result. Keys are never reused or deduplicated, so a key the model writes names
exactly one unit of this run.

The registry is bound to the run through ``citation_registry_var``, and tool
handlers read it when they are called. The agent loop runs concurrency-safe
tools on a thread pool, submitting each through
``contextvars.copy_context().run``. That copy is shallow, so every worker sees
the same registry object, and keys are assigned under the registry's lock.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from agentic.knowledge.models import RetrievedItem

from .citations import TEXT_EXCERPT_MAX_LEN

KIND_KB_CHUNK = "kb_chunk"
KIND_TOOL_ITEM = "tool_item"
KIND_TOOL_CALL = "tool_call"
_KINDS = frozenset({KIND_KB_CHUNK, KIND_TOOL_ITEM, KIND_TOOL_CALL})


@dataclass(frozen=True)
class CitationUnit:
    kind: str
    tool_name: str | None = None
    call_id: str | None = None
    title: str | None = None
    url: str | None = None
    item_id: str | None = None
    source_id: str | None = None
    source_name: str = ""
    knowledge_base_id: str | None = None
    text_excerpt: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown citation kind: {self.kind!r}")


class CitationRegistry:
    """Assigns run-wide citation keys. Thread-safe."""

    def __init__(self, on_register: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._units: list[CitationUnit] = []
        self._on_register = on_register

    def register(self, unit: CitationUnit) -> int:
        return self.register_many([unit])[0]

    def register_many(self, units: Iterable[CitationUnit]) -> list[int]:
        """Register units in order; their keys are contiguous.

        ``on_register`` runs after the lock is released, once per unit in this
        batch's key order. Batches from concurrent tool calls interleave, so
        ``citation_registered`` events are not guaranteed to arrive in key
        order across calls, and they carry no ``seq``/``ts``. Consumers must
        index units by ``key``.
        """
        batch = list(units)
        with self._lock:
            first = len(self._units) + 1
            self._units.extend(batch)
        keys = list(range(first, first + len(batch)))
        if self._on_register is not None:
            for key, unit in zip(keys, batch):
                self._on_register(
                    {
                        "type": "citation_registered",
                        "key": key,
                        "kind": unit.kind,
                        "tool_name": unit.tool_name,
                        "title": unit.title,
                        "url": unit.url,
                    }
                )
        return keys

    def __len__(self) -> int:
        with self._lock:
            return len(self._units)

    def citation_map(self) -> dict[str, dict[str, Any]]:
        """Every registered unit, keyed "1", "2", ... like ``build_citation_map``."""
        with self._lock:
            units = list(self._units)
        return {
            str(key): {
                "key": str(key),
                "kind": unit.kind,
                "tool_name": unit.tool_name,
                "call_id": unit.call_id,
                "title": unit.title,
                "url": unit.url,
                "item_id": unit.item_id,
                "source_id": unit.source_id,
                "source_name": unit.source_name,
                "knowledge_base_id": unit.knowledge_base_id,
                "text_excerpt": unit.text_excerpt,
                "meta": unit.meta,
            }
            for key, unit in enumerate(units, start=1)
        }


citation_registry_var: contextvars.ContextVar[CitationRegistry | None] = contextvars.ContextVar(
    "citation_registry", default=None
)
current_tool_call_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_tool_call_id", default=None
)


def get_citation_registry() -> CitationRegistry | None:
    return citation_registry_var.get()


def get_current_tool_call_id() -> str | None:
    return current_tool_call_id_var.get()


def track_tool_call_id(event: dict[str, Any]) -> None:
    """Record the id of the tool call the agent loop is about to execute.

    The loop emits a ``tool_call`` event and then runs the tool, synchronously,
    in the same context: the per-call context copy for a concurrent call, or
    the loop's own context for an exclusive one. Setting the variable from the
    event callback therefore makes the id visible to that tool's handler, and
    to no other call's.
    """
    if event.get("type") == "tool_call":
        current_tool_call_id_var.set(event.get("call_id"))


def kb_chunk_unit_from_item(
    item: RetrievedItem, *, tool_name: str | None, call_id: str | None
) -> CitationUnit:
    meta = dict(item.meta or {})
    source_name = meta.get("source_name") or ""
    return CitationUnit(
        kind=KIND_KB_CHUNK,
        tool_name=tool_name,
        call_id=call_id,
        title=source_name or meta.get("doc_name") or None,
        item_id=item.item_id or None,
        source_id=item.source_id,
        source_name=source_name,
        knowledge_base_id=item.knowledge_base_id,
        text_excerpt=(item.text or "")[:TEXT_EXCERPT_MAX_LEN],
        meta=meta,
    )


def kb_chunk_unit_from_context(entry: dict[str, Any]) -> CitationUnit:
    """A unit for one pre-fetched retrieved_context entry (not a tool call)."""
    meta = entry.get("meta") or {}
    source_name = entry.get("source_name") or meta.get("source_name") or ""
    return CitationUnit(
        kind=KIND_KB_CHUNK,
        title=source_name or None,
        item_id=entry.get("id"),
        source_id=entry.get("source_id"),
        source_name=source_name,
        knowledge_base_id=entry.get("knowledge_base_id"),
        text_excerpt=(entry.get("text") or "")[:TEXT_EXCERPT_MAX_LEN],
        meta=meta,
    )


def seed_registry_from_retrieved_context(
    registry: CitationRegistry, retrieved_context: list[dict[str, Any]] | None
) -> list[int]:
    """Register pre-fetched context first, in list order.

    Pre-fetched context is formatted with labels [1]..[k] in this order, so
    the labels become valid run-wide keys and tool results continue from k+1.
    """
    entries = [
        entry
        for entry in retrieved_context or []
        if isinstance(entry, dict) and entry.get("_type") != "retrieval_diagnostics"
    ]
    return registry.register_many(kb_chunk_unit_from_context(entry) for entry in entries)
