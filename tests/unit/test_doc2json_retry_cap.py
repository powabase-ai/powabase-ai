"""Capture-kwargs test: every litellm.acompletion call in doc2json.py must
pass num_retries=0, max_retries=0, timeout=60 to bound LiteLLM-internal
retries under a rate-limit storm (issue #445).

Test approach: behavioral capture via _process_window, which is the most
isolatable path to a call site — it takes only plain Python types (strings,
ints, dicts) and requires only two light mocks:
  - litellm.supports_response_schema  -> returns False (skip structured output branch)
  - litellm.acompletion               -> capture kwargs, return a minimal fake response

The third call site (_generate_combined_summary) is reached the same way,
tested in a second test.  The first call site (_extract_window_images) takes
base64 image data making a unit test impractical; the source-level count
assertion in test_all_call_sites_have_caps covers it instead.
"""

import json
from unittest.mock import patch

import pytest


def _fake_response(content: str):
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
            "usage": type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})(),
        },
    )()


@pytest.mark.asyncio
async def test_process_window_passes_retry_cap():
    """_process_window (call site ~line 647) must pass num_retries=0,
    max_retries=0, timeout=60 to litellm.acompletion."""
    from agentic.knowledge.indexing.doc2json import Doc2JSONAlgorithm

    algo = Doc2JSONAlgorithm(model="gpt-4o-mini")
    json_schema = {"fields": [{"name": "title", "type": "string", "description": "Title"}]}

    captured = {}

    async def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_response(json.dumps({"summary": "a summary", "extraction": {"title": "Test"}}))

    with (
        patch("litellm.supports_response_schema", return_value=False),
        patch("litellm.acompletion", side_effect=_capture),
    ):
        await algo._process_window(
            window_text="Some document text.",
            window_index=0,
            total_windows=1,
            json_schema=json_schema,
            current_json={"title": None},
            model="gpt-4o-mini",
        )

    assert captured.get("num_retries") == 0, f"got {captured.get('num_retries')!r}"
    assert captured.get("max_retries") == 0, f"got {captured.get('max_retries')!r}"
    assert captured.get("timeout") == 60, f"got {captured.get('timeout')!r}"


@pytest.mark.asyncio
async def test_generate_combined_summary_passes_retry_cap():
    """_generate_combined_summary (call site ~line 952) must pass
    num_retries=0, max_retries=0, timeout=60 to litellm.acompletion."""
    from agentic.knowledge.indexing.doc2json import Doc2JSONAlgorithm

    algo = Doc2JSONAlgorithm(model="gpt-4o-mini")
    json_schema = {"fields": [{"name": "title", "type": "string"}]}

    captured = {}

    async def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_response("A combined summary of the document.")

    with patch("litellm.acompletion", side_effect=_capture):
        result = await algo._generate_combined_summary(
            window_summaries=[
                {"summary": "First section summary."},
                {"summary": "Second section summary."},
            ],
            extracted_json={"title": "Test"},
            json_schema=json_schema,
            model="gpt-4o-mini",
        )

    assert result == "A combined summary of the document."
    assert captured.get("num_retries") == 0, f"got {captured.get('num_retries')!r}"
    assert captured.get("max_retries") == 0, f"got {captured.get('max_retries')!r}"
    assert captured.get("timeout") == 60, f"got {captured.get('timeout')!r}"


def test_all_call_sites_have_caps():
    """Source-level safety net: every litellm.acompletion( in doc2json.py
    must be immediately followed (within the same call block) by num_retries=0.
    Guards the _extract_window_images call site that is impractical to invoke
    in a unit test due to requiring base64 image data.
    """
    import importlib.util
    import pathlib

    spec = importlib.util.find_spec("agentic.knowledge.indexing.doc2json")
    assert spec is not None, "agentic package not installed"
    source = pathlib.Path(spec.origin)
    text = source.read_text()

    acompletion_count = text.count("litellm.acompletion(")
    num_retries_zero_count = text.count("num_retries=0,")

    assert acompletion_count == num_retries_zero_count, (
        f"Found {acompletion_count} litellm.acompletion( calls but only "
        f"{num_retries_zero_count} num_retries=0 kwargs in doc2json.py. "
        "Each call site must pass num_retries=0 (#445)."
    )
