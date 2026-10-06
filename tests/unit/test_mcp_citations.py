"""MCP results become citation units: per mapped JSON item, or per call."""

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from agentic_project_service.services.citation_registry import CitationRegistry
from agentic_project_service.services.mcp_citations import (
    citation_mapping_error,
    compile_items_path,
    key_mcp_result,
    select_objects,
)

TOOL = "mcp__docs__search_items"
SEARCH = {
    "total": 2,
    "results": [
        {"name": "Item A", "link": "https://example.org/a", "meta": {"lang": "de"}},
        {"name": "Ärger B", "link": "https://example.org/b"},
    ],
}
RULE = {"items": "$.results[*]", "title": "name", "url": "link"}


def _key(result, rule, registry=None, arguments=None):
    registry = registry or CitationRegistry()
    out = key_mcp_result(
        result,
        tool_name=TOOL,
        arguments=arguments if arguments is not None else {"q": "a"},
        rule=rule,
        registry=registry,
        call_id="call_1",
    )
    return out, registry


class TestPaths:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("$", ()),
            ("$.results[*]", (("key", "results"), ("star", None))),
            ("$['hits'][0]", (("key", "hits"), ("index", 0))),
            ("$.a.*.b", (("key", "a"), ("star", None), ("key", "b"))),
            ("$.items[-1]", (("key", "items"), ("index", -1))),
        ],
    )
    def test_supported_paths(self, path, expected):
        assert compile_items_path(path) == expected

    @pytest.mark.parametrize(
        "path", ["results[*]", "$..name", "$.results[?(@.x)]", "$.a[1:2]", "$.", ""]
    )
    def test_unsupported_paths_raise(self, path):
        with pytest.raises(ValueError):
            compile_items_path(path)

    def test_select_returns_objects_in_document_order(self):
        doc = {"a": [{"n": 1}, "x", {"n": 2}], "b": {"n": 3}}
        assert select_objects(doc, compile_items_path("$.a[*]")) == [{"n": 1}, {"n": 2}]
        assert select_objects(doc, compile_items_path("$")) == [doc]
        assert select_objects(doc, compile_items_path("$.missing[*]")) == []


class TestMappedItems:
    def test_each_matched_item_gets_a_key_and_the_json_stays_valid(self):
        out, registry = _key(json.dumps(SEARCH), RULE)
        doc = json.loads(out)
        assert [r["cite"] for r in doc["results"]] == ["[1]", "[2]"]
        assert doc["total"] == 2
        units = registry.citation_map()
        assert [(u["kind"], u["title"], u["url"]) for u in units.values()] == [
            ("tool_item", "Item A", "https://example.org/a"),
            ("tool_item", "Ärger B", "https://example.org/b"),
        ]
        assert units["1"]["meta"] == SEARCH["results"][0]
        assert "cite" not in units["1"]["meta"]
        assert units["1"]["call_id"] == "call_1"
        assert units["1"]["tool_name"] == TOOL

    def test_non_ascii_is_preserved_unescaped(self):
        out, _ = _key(json.dumps(SEARCH), RULE)
        assert "Ärger B" in out

    def test_dotted_title_and_non_string_fields(self):
        doc = {"results": [{"meta": {"title": "Deep"}, "link": 7}]}
        out, registry = _key(
            json.dumps(doc), {"items": "$.results[*]", "title": "meta.title", "url": "link"}
        )
        unit = registry.citation_map()["1"]
        assert (unit["title"], unit["url"]) == ("Deep", None)

    def test_an_existing_cite_field_is_overwritten_and_preserved_in_meta(self):
        """A provider-supplied cite field is overwritten with the key and kept in meta."""
        doc = {"results": [{"name": "A", "cite": "provider-cite-1"}]}
        out, registry = _key(json.dumps(doc), RULE)
        assert json.loads(out)["results"][0]["cite"] == "[1]"
        assert registry.citation_map()["1"]["meta"]["cite"] == "provider-cite-1"

    def test_items_none_is_left_alone(self):
        out, registry = _key(json.dumps(SEARCH), {"items": "none"})
        assert out == json.dumps(SEARCH)
        assert len(registry) == 0


class TestWholeCall:
    def test_unmapped_tool_is_one_prefixed_unit(self):
        out, registry = _key("Plain answer text.", None, arguments={"q": "alpha", "n": 2})
        assert out == "[1] Plain answer text."
        unit = registry.citation_map()["1"]
        assert unit["kind"] == "tool_call"
        assert unit["title"] == f'{TOOL} {{"n": 2, "q": "alpha"}}'
        assert unit["meta"] == {"raw": "Plain answer text.", "arguments": {"q": "alpha", "n": 2}}
        assert unit["text_excerpt"] == "Plain answer text."

    def test_long_arguments_are_summarised(self):
        _, registry = _key("x", None, arguments={"q": "y" * 200})
        title = registry.citation_map()["1"]["title"]
        assert title.startswith(f"{TOOL} ")
        assert len(title) == len(TOOL) + 1 + 80
        assert title.endswith("…")

    def test_non_json_body_with_a_mapping_is_one_unit(self):
        out, registry = _key("# Heading\n\nmarkdown body", RULE)
        assert out == "[1] # Heading\n\nmarkdown body"
        assert registry.citation_map()["1"]["kind"] == "tool_call"

    @pytest.mark.parametrize(
        "body",
        [
            json.dumps({"results": ["a", "b"]}),
            json.dumps({"results": []}),
            json.dumps({"hits": [{"name": "moved"}]}),
            json.dumps([1, 2, 3]),
            "null",
        ],
    )
    def test_path_matching_no_objects_falls_back_to_one_whole_call_unit(self, body):
        """A path matching no objects keeps the whole result as one keyed unit."""
        out, registry = _key(body, RULE)
        assert out == f"[1] {body}"
        assert [u["kind"] for u in registry.citation_map().values()] == ["tool_call"]


class TestNoUnit:
    @pytest.mark.parametrize(
        "result",
        [
            "Error: tool exploded",
            "Error (code -32602): invalid params",
            "Error (code none): odd",
            "Error calling MCP tool: timed out",
            "(empty response)",
        ],
    )
    def test_engine_error_and_empty_results_get_no_key(self, result):
        """Engine error and empty results get no citation key."""
        for rule in (None, RULE):
            out, registry = _key(result, rule)
            assert out == result
            assert len(registry) == 0

    def test_non_string_result_is_left_alone(self):
        blocks = [{"type": "text", "text": "x"}]
        out, registry = _key(blocks, None)
        assert out is blocks
        assert len(registry) == 0


class TestConcurrency:
    def test_parallel_calls_get_distinct_keys(self):
        registry = CitationRegistry()

        def call(_):
            return key_mcp_result(
                json.dumps(SEARCH),
                tool_name=TOOL,
                arguments={},
                rule=RULE,
                registry=registry,
                call_id=None,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            outs = list(pool.map(lambda n: contextvars.copy_context().run(call, n), range(8)))
        cites = [r["cite"] for out in outs for r in json.loads(out)["results"]]
        assert sorted(cites, key=lambda c: int(c[1:-1])) == [f"[{n}]" for n in range(1, 17)]


class TestMappingValidation:
    @pytest.mark.parametrize(
        "mapping",
        [
            None,
            {},
            {"search_items": {"items": "$.results[*]"}},
            {"stats": {"items": "none"}},
            {"get_item": {"items": "$", "title": "meta.title", "url": "link"}},
            {"hits": {"items": "$['hits'][0]"}},
        ],
    )
    def test_valid(self, mapping):
        assert citation_mapping_error(mapping) is None

    @pytest.mark.parametrize(
        "mapping,fragment",
        [
            ([], "must be an object or null"),
            ({"t": "x"}, "must be an object"),
            ({"t": {}}, "items is required"),
            ({"t": {"items": 3}}, "items is required"),
            ({"t": {"items": "results"}}, "must start with '$'"),
            ({"t": {"items": "$..x"}}, "unsupported JSONPath"),
            ({"t": {"items": "$[?(@.a)]"}}, "unsupported JSONPath"),
            ({"t": {"items": "none", "tittle": "x"}}, "unknown key"),
            ({"t": {"items": "$", "title": ""}}, "dotted field"),
            ({"t": {"items": "$", "url": "a..b"}}, "dotted field"),
            ({"": {"items": "$"}}, "non-empty tool names"),
        ],
    )
    def test_invalid(self, mapping, fragment):
        error = citation_mapping_error(mapping)
        assert error is not None and fragment in error, error
