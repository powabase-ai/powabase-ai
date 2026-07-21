"""Tests that _enrich_single_node tags every litellm call with
metadata={"stage": "enrichment"}, so the LiteLLM cost-accumulator callback
can route enrichment costs into their own per-stage bucket on the
indexed_source row.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agentic_project_service.services.graph_enricher import _enrich_single_node


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


def _base_node() -> dict:
    return {
        "node_id": "0001",
        "title": "Intro",
        "text": "hello",
        "depth": 0,
        "parent_node_id": None,
    }


def test_enrichment_call_tags_metadata_stage_enrichment():
    """Every enrichment call must carry metadata={"stage": "enrichment"}."""
    with patch(
        "agentic_project_service.services.graph_enricher.litellm.acompletion",
        new=AsyncMock(return_value=_fake_llm_response()),
    ) as mock_acompletion:
        asyncio.run(
            _enrich_single_node(
                node=_base_node(),
                toc_context="[0001] Intro",
                valid_node_ids={"0001"},
                children_map={},
                id_to_title={"0001": "Intro"},
                model="gpt-4o",
            )
        )
    kwargs = mock_acompletion.await_args.kwargs
    assert kwargs.get("metadata") == {"stage": "enrichment"}


def test_metadata_present_with_reasoning_effort_set():
    """Adding metadata must not break the existing reasoning_effort plumbing."""
    with patch(
        "agentic_project_service.services.graph_enricher.litellm.acompletion",
        new=AsyncMock(return_value=_fake_llm_response()),
    ) as mock_acompletion:
        asyncio.run(
            _enrich_single_node(
                node=_base_node(),
                toc_context="[0001] Intro",
                valid_node_ids={"0001"},
                children_map={},
                id_to_title={"0001": "Intro"},
                model="anthropic/claude-opus-4-7",
                reasoning_effort="low",
            )
        )
    kwargs = mock_acompletion.await_args.kwargs
    assert kwargs.get("metadata") == {"stage": "enrichment"}
    assert kwargs.get("thinking") == {"type": "adaptive", "display": "summarized"}
    assert kwargs.get("output_config") == {"effort": "low"}
