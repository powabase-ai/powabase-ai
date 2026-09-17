"""The keyword-timeout 503 tells the truth about a build in progress.

While a knowledge base's BM25 index is being built -- right after its move,
typically -- keyword search falls back to the bounded scan and can time out.
The 503 said the knowledge base "has no BM25 index" and that retrying would
time out again, while a build was about to fix exactly that. And for a
knowledge base on pg_search it promised that setting the stored method builds
the index automatically, which a PATCH no longer does for rows in the shared
DEFAULT partition.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from tests.unit.test_search_route_keyword_timeout import _app, _kb

from agentic_project_service.services.base_vector_store import KeywordSearchTimeout

R = "agentic_project_service.routes.knowledge_bases"


def _error(kb, *, status=None, backend="pg_search", auto_indexing=True, request_method="full_text"):
    kb_id = str(uuid.uuid4())
    with (
        patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"}),
        patch(f"{R}.db"),
        patch(
            f"{R}.get_setting",
            side_effect=lambda key: auto_indexing if key == "BM25_AUTO_INDEXING" else None,
        ),
        patch(f"{R}._fetch_kb_or_404", return_value=kb),
        patch(f"{R}._keyword_index_backend", return_value=backend),
        patch(f"{R}._bm25_status_detail", return_value=(status, None)),
        patch(
            "agentic_project_service.services.knowledge_search.search_knowledge_base",
            side_effect=KeywordSearchTimeout(kb_id, 10000),
        ),
        _app().test_client() as c,
    ):
        resp = c.post(
            f"/api/knowledge-bases/{kb_id}/search",
            headers={"Authorization": "Bearer fake.jwt.token"},
            json={"query": "weather", "retrieval_method": request_method},
        )
    assert resp.status_code == 503
    assert resp.get_json()["code"] == "keyword_search_timeout"
    return resp.get_json()["error"]


@pytest.mark.parametrize("status", ["queued", "moving", "building", "retrying"])
def test_a_build_in_progress_is_named_and_retrying_later_is_the_advice(status):
    error = _error(_kb("chunk_embed", "hybrid"), status=status)
    assert "being built" in error
    assert f"bm25_status {status}" in error
    assert "has no BM25 index" not in error
    assert "will time out again" not in error
    assert "vector_search" in error


@pytest.mark.parametrize("status", [None, "absent", "failed", "needs_build", "unavailable"])
def test_without_a_build_in_progress_the_remedy_stands(status):
    error = _error(_kb("chunk_embed", "hybrid"), status=status)
    assert "being built" not in error
    assert "Retrying the same search does not start a build" in error


def test_on_pg_search_setting_the_stored_method_is_not_promised_to_build_the_index():
    error = _error(_kb("chunk_embed", "vector_search"), backend="pg_search", auto_indexing=True)
    assert "automatically" not in error
    assert "build-bm25" in error
    assert "hybrid or full_text" in error


def test_on_the_file_index_the_automatic_build_is_still_named():
    error = _error(_kb("chunk_embed", "vector_search"), backend="bm25s", auto_indexing=True)
    assert "automatically" in error
