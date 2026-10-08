"""knowledge_search labels its chunks with run-wide citation keys.

Two searches in one run used to label their chunks [1]..[k] each, so a marker
on the second search's chunk resolved to the first search's chunk. The keys now
come from the run's citation registry.
"""

from unittest.mock import MagicMock, patch

from agentic.knowledge.models import RetrievedItem

from agentic_project_service.services import context_handler as ch
from agentic_project_service.services import tool_registry
from agentic_project_service.services.citation_registry import (
    CitationRegistry,
    citation_registry_var,
    current_tool_call_id_var,
)

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


def _item(item_id: str, text: str, score: float) -> RetrievedItem:
    return RetrievedItem(
        item_id=item_id, text=text, score=score, source_id=None, knowledge_base_id=KB, meta={}
    )


def _session() -> MagicMock:
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = [(KB, "a kb", {}, {})]
    session.execute.return_value.fetchone.return_value = None
    return session


def _search_returns(monkeypatch, items):
    defaults = {"KB_DEFAULT_TOP_K": 5, "KB_DEFAULT_MAX_CONTEXT_TOKENS": 4000}
    monkeypatch.setattr(ch, "get_setting", lambda key: defaults.get(key, 5))
    monkeypatch.setattr(ch, "search_knowledge_base", lambda **kwargs: list(items))


def _non_diag(out):
    return [i for i in out["retrieved_context"] if i.get("_type") != "retrieval_diagnostics"]


class TestExecuteRetrieval:
    def test_registered_keys_label_the_context_and_the_items(self, monkeypatch):
        _search_returns(monkeypatch, [_item("c1", "first hit", 0.9), _item("c2", "second", 0.8)])
        seen: list[list[str]] = []

        def register(items):
            seen.append([i.item_id for i in items])
            return [5, 6]

        out = ch.execute_retrieval(
            db_session=_session(),
            query="q",
            knowledge_base_configs=[{"id": KB}],
            register_items=register,
        )
        assert seen == [["c1", "c2"]]
        assert "[5]" in out["formatted_context"] and "[6]" in out["formatted_context"]
        assert "[1]" not in out["formatted_context"]
        assert [i["citation_key"] for i in _non_diag(out)] == [5, 6]

    def test_without_a_registry_nothing_changes(self, monkeypatch):
        _search_returns(monkeypatch, [_item("c1", "first hit", 0.9), _item("c2", "second", 0.8)])
        out = ch.execute_retrieval(
            db_session=_session(), query="q", knowledge_base_configs=[{"id": KB}]
        )
        assert "[1]" in out["formatted_context"] and "[2]" in out["formatted_context"]
        assert all("citation_key" not in i for i in _non_diag(out))

    def test_chunks_dropped_by_the_budget_are_registered_but_not_shown(self, monkeypatch):
        _search_returns(monkeypatch, [_item("c1", "short", 0.9), _item("c2", "x" * 4000, 0.8)])
        seen: list[list[str]] = []

        def register(items):
            seen.append([i.item_id for i in items])
            return [5, 6]

        out = ch.execute_retrieval(
            db_session=_session(),
            query="q",
            knowledge_base_configs=[{"id": KB}],
            max_context_tokens=50,
            register_items=register,
        )
        assert seen == [["c1", "c2"]]
        assert "[5]" in out["formatted_context"]
        assert "[6]" not in out["formatted_context"]
        dropped = _non_diag(out)[1]
        assert dropped["included_in_context"] is False
        assert dropped["citation_key"] == 6


def _items(query: str) -> list[RetrievedItem]:
    return [
        RetrievedItem(
            item_id=f"{query}-{n}",
            text=f"{query} text {n}",
            score=0.9,
            source_id=None,
            knowledge_base_id="kb-1",
            meta={"source_name": f"{query}.txt"},
        )
        for n in (1, 2)
    ]


class TestSearchHandler:
    def _run(self, fake_create, queries):
        with (
            patch.object(tool_registry, "_get_flask_app", return_value=None),
            patch.object(tool_registry, "Session"),
            patch.object(tool_registry, "create_and_execute", side_effect=fake_create),
        ):
            handler = tool_registry._make_search_handler(MagicMock(name="shared_db_session"))
            for query in queries:
                handler(query=query, kb_configs=[{"id": "kb-1"}], max_tokens=100, session_history=None)

    def test_two_searches_in_one_run_get_consecutive_keys(self):
        registry = CitationRegistry()
        labels: list[list[int]] = []

        def fake_create(**kwargs):
            labels.append(kwargs["register_items"](_items(kwargs["query"])))
            return "handler-1", {"formatted_context": "ctx", "errors": []}

        reg_token = citation_registry_var.set(registry)
        call_token = current_tool_call_id_var.set("call_b")
        try:
            self._run(fake_create, ["alpha", "beta"])
        finally:
            current_tool_call_id_var.reset(call_token)
            citation_registry_var.reset(reg_token)

        assert labels == [[1, 2], [3, 4]]
        unit = registry.citation_map()["3"]
        assert unit["kind"] == "kb_chunk"
        assert unit["tool_name"] == "knowledge_search"
        assert unit["call_id"] == "call_b"
        assert unit["knowledge_base_id"] == "kb-1"
        assert unit["item_id"] == "beta-1"
        assert unit["title"] == "beta.txt"

    def test_without_a_registry_no_callback_is_passed(self):
        passed: list = []

        def fake_create(**kwargs):
            passed.append(kwargs["register_items"])
            return "handler-1", {"formatted_context": "ctx", "errors": []}

        self._run(fake_create, ["alpha"])
        assert passed == [None]
