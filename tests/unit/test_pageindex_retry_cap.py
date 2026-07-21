"""Capture-kwargs test: litellm.acompletion in pageindex utils._llm_completion
must pass the configured retry/timeout knobs — num_retries=PAGEINDEX_LLM_NUM_RETRIES,
max_retries=0, timeout=PAGEINDEX_LLM_TIMEOUT.

History: #445 hard-capped these at num_retries=0 / timeout=60 for bounded memory.
They were later made env-driven (PAGEINDEX_LLM_NUM_RETRIES default 1,
PAGEINDEX_LLM_TIMEOUT default 300) because slow reasoning models (gpt-5*) doing
large ToC/summary extraction legitimately exceed 60s, and a single timeout with
zero retries failed the whole indexing job. This test pins the wiring to those
constants (not brittle literals) so it tracks the env-configurable contract.

Test approach: behavioral capture via _llm_completion, a top-level async
function.  Mocks needed:
  - agentic.knowledge.indexing._pageindex_lib.utils._reasoning_effort_var
    (via monkeypatch on the contextvar, or default None is fine)
  - agentic.llm.routing.maybe_route_through_responses  -> returns model unchanged
  - agentic.llm.routing.reasoning_call_kwargs           -> returns {}
  - litellm.acompletion                                 -> capture + return fake resp
"""

from unittest.mock import patch

import pytest


def _fake_response(content: str = "hello"):
    return type(
        "R",
        (),
        {
            "choices": [
                type(
                    "C",
                    (),
                    {
                        "message": type("M", (), {"content": content})(),
                        "finish_reason": "stop",
                    },
                )()
            ],
        },
    )()


@pytest.mark.asyncio
async def test_llm_completion_passes_retry_cap():
    """_llm_completion must pass the configured num_retries/timeout knobs and
    max_retries=0 to litellm.acompletion. Pinned to PAGEINDEX_LLM_NUM_RETRIES /
    PAGEINDEX_LLM_TIMEOUT (env-driven) rather than literals."""
    from agentic.knowledge.indexing._pageindex_lib.utils import _llm_completion
    from agentic.knowledge.model_config import (
        PAGEINDEX_LLM_NUM_RETRIES,
        PAGEINDEX_LLM_TIMEOUT,
    )

    captured = {}

    async def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_response("answer text")

    with (
        patch(
            "agentic.llm.routing.maybe_route_through_responses",
            side_effect=lambda model, effort: model,
        ),
        patch(
            "agentic.llm.routing.reasoning_call_kwargs",
            return_value={},
        ),
        patch("litellm.acompletion", side_effect=_capture),
    ):
        content, finish_reason = await _llm_completion(
            model="gpt-4o-mini",
            prompt="What is 2+2?",
        )

    assert content == "answer text"
    assert finish_reason == "finished"
    assert captured.get("num_retries") == PAGEINDEX_LLM_NUM_RETRIES, (
        f"got {captured.get('num_retries')!r}"
    )
    assert captured.get("max_retries") == 0, f"got {captured.get('max_retries')!r}"
    assert captured.get("timeout") == PAGEINDEX_LLM_TIMEOUT, f"got {captured.get('timeout')!r}"


def test_get_start_page_helpers_do_not_nameerror():
    """re was unimported (would be masked by noqa) — these helpers NameError'd at runtime.

    Regression test for issue #445: get_first_start_page_from_text() and
    get_last_start_page_from_text() use re.search and re.finditer but re was
    never imported.
    """
    from agentic.knowledge.indexing._pageindex_lib.utils import (
        get_first_start_page_from_text,
        get_last_start_page_from_text,
    )

    # Text with two <start_index_N> markers
    text = "foo <start_index_3> bar <start_index_7> baz"

    # Should not raise NameError; should find the markers
    first = get_first_start_page_from_text(text)
    last = get_last_start_page_from_text(text)

    assert first == 3
    assert last == 7


def test_get_start_page_helpers_no_match():
    """When no markers are found, helpers return -1."""
    from agentic.knowledge.indexing._pageindex_lib.utils import (
        get_first_start_page_from_text,
        get_last_start_page_from_text,
    )

    text = "no markers here"

    first = get_first_start_page_from_text(text)
    last = get_last_start_page_from_text(text)

    assert first == -1
    assert last == -1
