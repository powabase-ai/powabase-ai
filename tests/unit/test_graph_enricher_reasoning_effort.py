"""Tests that _enrich_single_node passes reasoning_effort through the
agentic.llm.routing helpers to litellm.acompletion."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agentic_project_service.services.graph_enricher import (
    _enrich_single_node,
)


@pytest.fixture(autouse=True)
def _stub_byok_resolver():
    """See test_query_enrichment_billing.py for the rationale. The
    BYOK-unification refactor (2026-06-03) wraps LLM calls in
    ``with_byok_and_recoup`` which needs a Flask app context for the
    resolver's DB query — short-circuited here so unit tests stay
    DB-free.
    """
    with patch(
        "agentic_project_service.services.llm_call.get_all_user_provider_keys",
        return_value={},
    ):
        yield


def _fake_llm_response(content: str = '{"referenced_nodes": []}') -> object:
    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5

    class _Msg:
        message = type("M", (), {"content": content})
        finish_reason = "stop"

    class _R:
        choices = [_Msg()]
        usage = _Usage()

    return _R()


@pytest.fixture
def base_node():
    return {
        "node_id": "0001",
        "title": "Intro",
        "text": "hello",
        "depth": 0,
        "parent_node_id": None,
    }


def test_no_reasoning_effort_omits_kwargs(base_node):
    with patch(
        "agentic_project_service.services.graph_enricher.litellm.acompletion",
        new=AsyncMock(return_value=_fake_llm_response()),
    ) as mock_acompletion:
        asyncio.run(
            _enrich_single_node(
                node=base_node,
                toc_context="[0001] Intro",
                valid_node_ids={"0001"},
                children_map={},
                id_to_title={"0001": "Intro"},
                model="gpt-4o",
            )
        )
    kwargs = mock_acompletion.await_args.kwargs
    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs


def test_reasoning_effort_for_anthropic_requests_summarized(base_node):
    """opus 4.7/4.8 etc. (adaptive display defaults to "omitted") get an
    explicit adaptive+summarized thinking config + output_config effort so
    reasoning is surfaced; no top-level reasoning_effort; no extra_body."""
    with patch(
        "agentic_project_service.services.graph_enricher.litellm.acompletion",
        new=AsyncMock(return_value=_fake_llm_response()),
    ) as mock_acompletion:
        asyncio.run(
            _enrich_single_node(
                node=base_node,
                toc_context="[0001] Intro",
                valid_node_ids={"0001"},
                children_map={},
                id_to_title={"0001": "Intro"},
                model="anthropic/claude-opus-4-7",
                reasoning_effort="low",
            )
        )
    kwargs = mock_acompletion.await_args.kwargs
    assert kwargs.get("thinking") == {"type": "adaptive", "display": "summarized"}
    assert kwargs.get("output_config") == {"effort": "low"}
    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs


def test_reasoning_effort_for_openai_routes_through_responses(base_node):
    with (
        patch(
            "agentic_project_service.services.graph_enricher.litellm.acompletion",
            new=AsyncMock(return_value=_fake_llm_response()),
        ) as mock_acompletion,
        patch(
            "agentic.llm.routing.litellm.supports_reasoning",
            return_value=True,
        ),
        # Hermetic: force summary-off regardless of ambient env. Requesting a
        # reasoning summary 400s on unverified OpenAI orgs, so it is omitted by
        # default (see agentic.llm.routing._reasoning_summary_enabled).
        patch.dict("os.environ", {"OPENAI_REASONING_SUMMARY": ""}),
    ):
        asyncio.run(
            _enrich_single_node(
                node=base_node,
                toc_context="[0001] Intro",
                valid_node_ids={"0001"},
                children_map={},
                id_to_title={"0001": "Intro"},
                model="openai/gpt-5-mini",
                reasoning_effort="low",
            )
        )
    kwargs = mock_acompletion.await_args.kwargs
    assert "reasoning_effort" not in kwargs
    assert kwargs["model"] == "openai/responses/gpt-5-mini"
    assert kwargs["extra_body"] == {"reasoning": {"effort": "low"}}


def test_reasoning_effort_for_openai_includes_summary_when_opted_in(base_node):
    """With OPENAI_REASONING_SUMMARY=1 (verified org), the summary is requested
    again so the reasoning-display UI gets summary text."""
    with (
        patch(
            "agentic_project_service.services.graph_enricher.litellm.acompletion",
            new=AsyncMock(return_value=_fake_llm_response()),
        ) as mock_acompletion,
        patch(
            "agentic.llm.routing.litellm.supports_reasoning",
            return_value=True,
        ),
        patch.dict("os.environ", {"OPENAI_REASONING_SUMMARY": "1"}),
    ):
        asyncio.run(
            _enrich_single_node(
                node=base_node,
                toc_context="[0001] Intro",
                valid_node_ids={"0001"},
                children_map={},
                id_to_title={"0001": "Intro"},
                model="openai/gpt-5-mini",
                reasoning_effort="low",
            )
        )
    kwargs = mock_acompletion.await_args.kwargs
    assert kwargs["model"] == "openai/responses/gpt-5-mini"
    assert kwargs["extra_body"] == {
        "reasoning": {"effort": "low", "summary": "detailed"}
    }


def test_enrich_referenced_nodes_threads_effort_to_single_node(monkeypatch):
    """enrich_referenced_nodes must pass reasoning_effort through to
    _enrich_single_node for every node it processes."""
    import asyncio

    from agentic_project_service.services import graph_enricher

    captured_efforts: list[str | None] = []

    async def fake_single_node(*, reasoning_effort=None, **_kwargs):
        captured_efforts.append(reasoning_effort)
        return [], None

    # Stub the DB layer
    class _StoreStub:
        def __init__(self, **_):
            pass

        def get_all_nodes_for_toc(self, _toc_id):
            return [
                {
                    "node_id": "0001",
                    "title": "A",
                    "text": "x",
                    "depth": 0,
                    "parent_node_id": None,
                    "meta": {},
                },
                {
                    "node_id": "0002",
                    "title": "B",
                    "text": "y",
                    "depth": 0,
                    "parent_node_id": None,
                    "meta": {},
                },
            ]

        def update_node_meta(self, *_a, **_kw):
            pass

        def update_node_enrichment_error(self, *_a, **_kw):
            pass

    monkeypatch.setattr(graph_enricher, "_enrich_single_node", fake_single_node)
    monkeypatch.setattr(graph_enricher, "GraphIndexStore", _StoreStub)

    class _DB:
        def flush(self):
            pass

    asyncio.run(
        graph_enricher.enrich_referenced_nodes(
            db_session=_DB(),
            knowledge_base_id="kb1",
            toc_id="toc1",
            model="gpt-5-mini",
            reasoning_effort="low",
        )
    )

    assert captured_efforts == ["low", "low"]
