"""Wiring tests for the KB model-dropdown reasoning-effort feature.

The KB config UI lets every LLM model field reveal a reasoning-effort
dropdown for reasoning-capable models. The selected effort is persisted in
the KB's indexing/retrieval config and must reach the underlying litellm
call. These tests pin that plumbing for the paths added alongside
graph_index (which already had test_indexing_config_reasoning_effort.py /
test_graph_enricher_reasoning_effort.py):

  * query enrichment            — behavioural (mock litellm.completion)
  * page_index / full_document  — source wiring (indexing.py)
  * doc2json extraction         — source wiring (indexing.py + doc2json.py)
  * tree_search retrieval       — source wiring (tree_search.py)

The behavioural test reuses the per-provider shape contract verified in
agentic.llm.routing: Anthropic gets a top-level ``reasoning_effort``; OpenAI
reasoning models route through the Responses bridge with the effort packed
into ``extra_body['reasoning']``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import patch

import agentic
import pytest

# ---------------------------------------------------------------------------
# Behavioural: query enrichment threads reasoning_effort to litellm
# ---------------------------------------------------------------------------

from agentic_project_service.services.query_enrichment import enrich_query


@pytest.fixture(autouse=True)
def _stub_byok_resolver():
    """BYOK resolution hits the DB via a Flask app context; short-circuit it
    so these stay DB-free (mirrors test_graph_enricher_reasoning_effort.py)."""
    with patch(
        "agentic_project_service.services.llm_call.get_all_user_provider_keys",
        return_value={},
    ):
        yield


def _fake_completion(content: str = '{"enriched_query": "x", "keywords": "y"}'):
    class _Msg:
        message = type("M", (), {"content": content})

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5

    class _R:
        choices = [_Msg()]
        usage = _Usage()

    return _R()


def test_query_enrichment_no_effort_omits_reasoning_kwargs():
    with patch("litellm.completion", return_value=_fake_completion()) as mock:
        enrich_query(query="what about pricing", retrieval_method="vector_search")
    kwargs = mock.call_args.kwargs
    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs


def test_query_enrichment_anthropic_passes_top_level_effort():
    with patch("litellm.completion", return_value=_fake_completion()) as mock:
        enrich_query(
            query="what about pricing",
            retrieval_method="vector_search",
            model="anthropic/claude-opus-4-6",
            reasoning_effort="low",
        )
    kwargs = mock.call_args.kwargs
    assert kwargs.get("reasoning_effort") == "low"
    assert "extra_body" not in kwargs


def test_query_enrichment_openai_routes_through_responses():
    with (
        patch("litellm.completion", return_value=_fake_completion()) as mock,
        patch("agentic.llm.routing.litellm.supports_reasoning", return_value=True),
        patch.dict("os.environ", {"OPENAI_REASONING_SUMMARY": ""}),
    ):
        enrich_query(
            query="what about pricing",
            retrieval_method="vector_search",
            model="openai/gpt-5-mini",
            reasoning_effort="medium",
        )
    kwargs = mock.call_args.kwargs
    assert kwargs["model"] == "openai/responses/gpt-5-mini"
    assert kwargs["extra_body"] == {"reasoning": {"effort": "medium"}}
    assert "reasoning_effort" not in kwargs


# ---------------------------------------------------------------------------
# Source-wiring guards (robust to refactors that move surrounding code)
# ---------------------------------------------------------------------------

_PS_SRC = Path(__file__).resolve().parents[2] / "src" / "agentic_project_service"
_INDEXING_PY = _PS_SRC / "tasks" / "indexing.py"

# agentic ships as the powabase-agentic PyPI package; resolve its installed
# on-disk location so the AST guards below can verify the real source.
_AGENTIC_SRC = Path(agentic.__file__).resolve().parent / "knowledge"
_DOC2JSON_PY = _AGENTIC_SRC / "indexing" / "doc2json.py"
_TREE_SEARCH_PY = _AGENTIC_SRC / "retrieval" / "tree_search.py"


def _funcs(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
    return out


def _copies_effort_to_extra(src: str) -> bool:
    """`if "reasoning_effort" in indexing_config: extra[...] = indexing_config[...]`"""
    pat = re.compile(
        r'if\s+["\']reasoning_effort["\']\s+in\s+indexing_config\s*:\s*\n'
        r'\s+extra\[["\']reasoning_effort["\']\]\s*=\s*indexing_config\[["\']reasoning_effort["\']\]'
    )
    return bool(pat.search(src))


def test_page_index_copies_reasoning_effort_to_extra():
    fn = _funcs(_INDEXING_PY)["_do_run_page_index_indexing"]
    assert _copies_effort_to_extra(ast.get_source_segment(_INDEXING_PY.read_text(), fn))


def test_doc2json_indexing_copies_reasoning_effort_to_extra():
    fn = _funcs(_INDEXING_PY)["run_doc2json_indexing"]
    assert _copies_effort_to_extra(ast.get_source_segment(_INDEXING_PY.read_text(), fn))


def test_full_document_passes_reasoning_kwargs_to_acompletion():
    """run_full_document_indexing must route the summary model and spread
    reasoning_call_kwargs(...) into the litellm.acompletion call."""
    src = _INDEXING_PY.read_text()
    body = ast.get_source_segment(src, _funcs(_INDEXING_PY)["run_full_document_indexing"])
    assert 'indexing_config.get("reasoning_effort")' in body
    assert "maybe_route_through_responses(summary_model, reasoning_effort)" in body
    assert "reasoning_call_kwargs(reasoning_effort, routed_summary_model)" in body
    assert "**reasoning_kwargs" in body


def test_doc2json_threads_reasoning_effort_to_all_llm_calls():
    """doc2json.py reads reasoning_effort from extra and spreads
    reasoning_call_kwargs(...) into every extraction/summary acompletion."""
    src = _DOC2JSON_PY.read_text()
    assert 'reasoning_effort = extra.get("reasoning_effort")' in src
    # All three litellm.acompletion calls must carry the reasoning kwargs.
    assert src.count("**reasoning_call_kwargs(reasoning_effort, model)") == 3
    # And each helper must accept the parameter.
    funcs = _funcs(_DOC2JSON_PY)
    for name in ("_process_window", "_process_window_with_images", "_generate_combined_summary"):
        params = {a.arg for a in funcs[name].args.args}
        assert "reasoning_effort" in params, f"{name} missing reasoning_effort param"


def test_tree_search_reads_effort_and_routes_both_calls():
    """Both tree-search retrieval stages must read retrieval_reasoning_effort
    from config and route the model + spread reasoning kwargs."""
    src = _TREE_SEARCH_PY.read_text()
    assert src.count('config.get("retrieval_reasoning_effort")') == 2
    assert src.count("maybe_route_through_responses(retrieval_model, reasoning_effort)") == 2
    assert src.count("reasoning_call_kwargs(reasoning_effort, routed_model)") == 2
    assert src.count("**reasoning_kwargs") == 2


def test_knowledge_search_passes_retrieval_effort_into_algo_config():
    """The tree_search algorithm only reads retrieval_reasoning_effort from the
    config dict it is handed, so knowledge_search.py must copy it from
    retrieval_config into BOTH algo_config dicts (sync + async paths).
    Without this the effort is silently dropped before reaching the LLM."""
    src = (_PS_SRC / "services" / "knowledge_search.py").read_text()
    assert (
        src.count(
            '"retrieval_reasoning_effort": retrieval_config.get("retrieval_reasoning_effort")'
        )
        == 2
    ), "both tree_search algo_config dicts must forward retrieval_reasoning_effort"
