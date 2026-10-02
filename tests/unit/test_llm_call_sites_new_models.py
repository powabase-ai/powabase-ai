"""The service's own litellm call sites must reach the newest models without a
parameter they reject.

Four sites call litellm directly rather than through the engine's Agent.
Three of them send ``temperature=0`` -- query enrichment, graph enrichment
and metadata enrichment. Claude Opus 5.5 / Fable 5.1 reject any temperature,
and GPT-6 / GPT-5.6 reject a non-default one while reasoning; litellm refuses
to send such a request unless the call passes ``drop_params=True``. The
fourth, the full-document summary, sends no temperature; its test pins that
the request still reaches the wire. These tests run the real functions with
HTTP intercepted at ``httpx.Client.send`` / ``httpx.AsyncClient.send`` and
assert what would have gone out. No network.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

# The running service imports powabase-agentic, which registers models newer
# than the pinned litellm's bundled cost map. Import it here too.
import agentic  # noqa: F401

from agentic_project_service.services import graph_enricher as ge
from agentic_project_service.services import metadata_enricher as me
from agentic_project_service.services import query_enrichment as qe
from agentic_project_service.tasks import indexing as ix

# (model, reasoning effort) pairs whose provider rejects temperature=0.
_REJECTS_TEMPERATURE = [
    ("claude-opus-5-5", None),
    ("claude-fable-5-1", "high"),
    ("gpt-6-astra", "high"),
    ("gpt-5.6", "medium"),
]


class _Captured(Exception):
    pass


@contextlib.contextmanager
def _fake_key(model):
    yield "sk-test"


@pytest.fixture
def sent():
    """Record each outgoing model call's body and fail it before it leaves."""
    bodies: list[dict] = []

    def record(request: httpx.Request):
        if request.method != "POST":
            raise httpx.ConnectError("no network in tests", request=request)
        bodies.append(json.loads(request.content or b"{}"))
        raise _Captured(str(request.url))

    def send(self, request, *args, **kwargs):
        record(request)

    async def asend(self, request, *args, **kwargs):
        record(request)

    with (
        patch.object(httpx.Client, "send", send),
        patch.object(httpx.AsyncClient, "send", asend),
        patch.object(qe, "with_llm_key", _fake_key),
        patch.object(qe.billing, "check_balance", lambda **kwargs: None),
        patch.object(ge, "with_llm_key", _fake_key),
        patch.object(me, "with_llm_key", _fake_key),
        patch.object(ix, "with_llm_key", _fake_key),
        patch.object(ix, "run_scope", lambda *a, **k: contextlib.nullcontext()),
    ):
        yield bodies


def _run(call):
    try:
        result = call()
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    except Exception:
        pass


def _the_one_request(bodies: list[dict]) -> dict:
    assert bodies, (
        "the call never reached the HTTP layer -- litellm refused the request "
        "client-side, typically over a temperature this model rejects"
    )
    return bodies[0]


@pytest.mark.parametrize("model,effort", _REJECTS_TEMPERATURE)
def test_query_enrichment(sent, model, effort):
    _run(lambda: qe.enrich_query("what is x?", "hybrid", model=model, reasoning_effort=effort))
    assert "temperature" not in _the_one_request(sent)


@pytest.mark.parametrize("model,effort", _REJECTS_TEMPERATURE)
def test_graph_enrichment(sent, model, effort):
    node = {"node_id": "0001", "title": "t", "text": "see section 2"}
    _run(
        lambda: ge._enrich_single_node(
            node,
            "toc",
            {"0001", "0002"},
            {},
            {"0001": "t"},
            model,
            api_key="sk-test",
            reasoning_effort=effort,
        )
    )
    assert "temperature" not in _the_one_request(sent)


@pytest.mark.parametrize("model", sorted({m for m, _ in _REJECTS_TEMPERATURE}))
def test_metadata_enrichment(sent, model):
    enricher = me.MetadataEnricher(MagicMock(), "kb")
    fields = [{"name": "a", "type": "text", "description": "d"}]
    _run(lambda: enricher.enrich_single_item("text", fields, model))
    assert "temperature" not in _the_one_request(sent)


@pytest.mark.parametrize("model,effort", _REJECTS_TEMPERATURE)
def test_full_document_summary(sent, model, effort):
    config = {"summary_model": model, "reasoning_effort": effort}
    _run(lambda: ix.run_full_document_indexing("kb", "is", "src", "hello world", config))
    assert "temperature" not in _the_one_request(sent)


def test_gemini_keeps_its_temperature(sent):
    """drop_params removes only what a model rejects."""
    _run(lambda: qe.enrich_query("what is x?", "hybrid", model="gemini/gemini-3.8-flash"))
    body = _the_one_request(sent)
    assert body["generationConfig"]["temperature"] == 0
