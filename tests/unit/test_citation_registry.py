"""The run-wide citation registry: keys, events, concurrency, and the engine
behaviour that lets a tool handler learn its own call id."""

import contextvars
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import pytest
from agentic.agent.agent import Agent
from agentic.agent.tools import BuiltinTool
from agentic.execution.context import ExecutionContext
from agentic.knowledge.models import RetrievedItem

from agentic_project_service.services.citation_registry import (
    CitationRegistry,
    CitationUnit,
    citation_registry_var,
    get_citation_registry,
    get_current_tool_call_id,
    kb_chunk_unit_from_item,
    seed_registry_from_retrieved_context,
    track_tool_call_id,
)
from agentic_project_service.services.citations import build_citation_map


def _unit(**overrides) -> CitationUnit:
    return CitationUnit(**{"kind": "tool_call", "tool_name": "lookup", **overrides})


class TestKeys:
    def test_keys_start_at_one_in_first_seen_order(self):
        registry = CitationRegistry()
        assert registry.register(_unit(title="a")) == 1
        assert registry.register(_unit(title="b")) == 2
        assert [u["title"] for u in registry.citation_map().values()] == ["a", "b"]

    def test_the_same_unit_twice_gets_two_keys(self):
        registry = CitationRegistry()
        unit = _unit(title="same")
        assert [registry.register(unit), registry.register(unit)] == [1, 2]
        assert len(registry) == 2

    def test_register_many_is_contiguous(self):
        registry = CitationRegistry()
        registry.register(_unit())
        assert registry.register_many([_unit(), _unit(), _unit()]) == [2, 3, 4]

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError):
            CitationUnit(kind="chunk")

    def test_citation_map_shape(self):
        registry = CitationRegistry()
        registry.register(
            CitationUnit(
                kind="tool_item",
                tool_name="mcp__docs__search_items",
                call_id="call_1",
                title="Item A",
                url="https://example.org/a",
                text_excerpt='{"name": "Item A"}',
                meta={"name": "Item A"},
            )
        )
        assert registry.citation_map() == {
            "1": {
                "key": "1",
                "kind": "tool_item",
                "tool_name": "mcp__docs__search_items",
                "call_id": "call_1",
                "title": "Item A",
                "url": "https://example.org/a",
                "item_id": None,
                "source_id": None,
                "source_name": "",
                "knowledge_base_id": None,
                "text_excerpt": '{"name": "Item A"}',
                "meta": {"name": "Item A"},
            }
        }


class TestEvents:
    def test_each_key_is_announced(self):
        events: list[dict] = []
        registry = CitationRegistry(on_register=events.append)
        registry.register_many(
            [
                _unit(kind="tool_item", tool_name="t", title="A", url="https://example.org/a"),
                _unit(kind="tool_call", tool_name="t", title="t {}", url=None),
            ]
        )
        assert events == [
            {
                "type": "citation_registered",
                "key": 1,
                "kind": "tool_item",
                "tool_name": "t",
                "title": "A",
                "url": "https://example.org/a",
            },
            {
                "type": "citation_registered",
                "key": 2,
                "kind": "tool_call",
                "tool_name": "t",
                "title": "t {}",
                "url": None,
            },
        ]


class TestConcurrency:
    def test_parallel_workers_get_distinct_contiguous_keys(self):
        """Mirrors the agent loop: each worker runs in its own copy of the context."""
        registry = CitationRegistry()
        token = citation_registry_var.set(registry)
        try:

            def worker(n: int) -> list[int]:
                bound = get_citation_registry()
                return [bound.register(_unit(tool_name=f"t{n}")) for _ in range(25)]

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(contextvars.copy_context().run, worker, n) for n in range(8)]
                keys = [k for f in futures for k in f.result()]
        finally:
            citation_registry_var.reset(token)
        assert sorted(keys) == list(range(1, 201))
        assert len(registry) == 200


class TestToolCallId:
    """Pins the engine behaviour the platform relies on to learn a call id
    without an engine change: the agent loop emits ``tool_call`` and then runs
    the tool, synchronously, in the same context."""

    def _tools(self):
        def handler(arguments, context):
            time.sleep(0.01)
            return get_current_tool_call_id()

        return {
            "probe": BuiltinTool(
                name="probe",
                description="Return the current call id.",
                input_schema={"type": "object", "properties": {}},
                handler=handler,
                is_concurrency_safe=True,
            )
        }

    def _call(self, call_id: str):
        return SimpleNamespace(id=call_id, function=SimpleNamespace(name="probe", arguments="{}"))

    def test_each_concurrent_call_sees_its_own_id(self):
        agent = Agent(model="gpt-4o-mini")
        tools = self._tools()
        context = ExecutionContext(on_event=track_tool_call_id)
        calls = [self._call(f"call_{n}") for n in range(6)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(
                    contextvars.copy_context().run,
                    agent._execute_single_tool,
                    tc,
                    tools,
                    context,
                    1,
                ): tc.id
                for tc in calls
            }
            seen = {futures[f]: f.result()[0] for f in as_completed(futures)}
        assert seen == {f"call_{n}": f"call_{n}" for n in range(6)}

    def test_sequential_calls_each_see_their_own_id(self):
        agent = Agent(model="gpt-4o-mini")
        tools = self._tools()
        context = ExecutionContext(on_event=track_tool_call_id)

        def loop():
            return [
                agent._execute_single_tool(self._call(cid), tools, context, 1)[0]
                for cid in ("call_a", "call_b")
            ]

        assert contextvars.copy_context().run(loop) == ["call_a", "call_b"]

    def test_other_events_leave_the_id_alone(self):
        def run():
            track_tool_call_id({"type": "tool_call", "call_id": "call_x"})
            track_tool_call_id({"type": "tool_result", "call_id": "call_y"})
            return get_current_tool_call_id()

        assert contextvars.copy_context().run(run) == "call_x"


class TestKbChunkUnits:
    def test_unit_from_retrieved_item(self):
        item = RetrievedItem(
            item_id="11111111-1111-1111-1111-111111111111",
            text="x" * 400,
            score=0.9,
            source_id="src-1",
            knowledge_base_id="kb-1",
            meta={"source_name": "a.pdf", "pages": [2]},
        )
        unit = kb_chunk_unit_from_item(item, tool_name="knowledge_search", call_id="call_1")
        assert unit == CitationUnit(
            kind="kb_chunk",
            tool_name="knowledge_search",
            call_id="call_1",
            title="a.pdf",
            item_id="11111111-1111-1111-1111-111111111111",
            source_id="src-1",
            source_name="a.pdf",
            knowledge_base_id="kb-1",
            text_excerpt="x" * 300,
            meta={"source_name": "a.pdf", "pages": [2]},
        )

    def test_seed_keeps_the_prefetched_labels(self):
        """Pre-fetched context is labelled [1]..[k] in list order (the same order
        build_citation_map numbers it). Seeding must give each item that key."""
        context = [
            {"_type": "retrieval_diagnostics", "total_items": 2},
            {
                "id": "a",
                "text": "A text",
                "source_id": "s",
                "source_name": "a.pdf",
                "knowledge_base_id": "kb",
                "meta": {"pages": [1]},
            },
            {"text": "by value", "meta": {}},
        ]
        registry = CitationRegistry()
        assert seed_registry_from_retrieved_context(registry, context) == [1, 2]
        legacy = build_citation_map(context)
        units = registry.citation_map()
        assert units.keys() == legacy.keys()
        for key, expected in legacy.items():
            for field in ("item_id", "source_id", "source_name", "text_excerpt", "meta"):
                assert units[key][field] == expected[field], (key, field)
        assert units["1"]["knowledge_base_id"] == "kb"
        assert units["1"]["title"] == "a.pdf"
        assert units["2"]["kind"] == "kb_chunk"
        assert units["2"]["tool_name"] is None

    def test_seed_with_no_context(self):
        registry = CitationRegistry()
        assert seed_registry_from_retrieved_context(registry, None) == []
        assert len(registry) == 0
