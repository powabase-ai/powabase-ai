"""Tests for POST /api/knowledge-bases/<kb_id>/build-bm25.

These cover the request guards and a KB whose keyword leg reads the bm25s
file index. Which task runs on the pg_search path is covered in
test_routes_keyword_index_dispatch.py.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from agentic_project_service.routes import knowledge_bases as kb_route


def _make_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(kb_route.knowledge_bases_bp)
    return app


@pytest.fixture(autouse=True)
def _file_index_backend():
    with patch.object(kb_route, "_keyword_index_backend", return_value="bm25s"):
        yield


def _auth_headers():
    return {"Authorization": "Bearer fake-service-role-key"}


# Patch decode_jwt so require_auth passes without a real JWT_SECRET / token.
_FAKE_JWT = patch(
    "agentic_project_service.auth.decode_jwt",
    return_value={"sub": "user-1", "role": "authenticated"},
)


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
def test_build_bm25_dispatches_task_for_hybrid_kb(mock_task, mock_fetch, _jwt):
    kb_id = str(uuid.uuid4())
    mock_fetch.return_value = {
        "id": kb_id,
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    mock_task.delay.return_value = MagicMock(id="task-abc")

    with _make_test_app().test_client() as client:
        resp = client.post(f"/api/knowledge-bases/{kb_id}/build-bm25", headers=_auth_headers())
    assert resp.status_code == 202
    body = resp.get_json()
    assert (body["task_id"], body["knowledge_base_id"]) == ("task-abc", kb_id)
    mock_task.delay.assert_called_once_with(kb_id)


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
def test_build_bm25_dispatches_task_for_full_text_kb(mock_task, mock_fetch, _jwt):
    kb_id = str(uuid.uuid4())
    mock_fetch.return_value = {
        "id": kb_id,
        "retrieval_config": {"method": "full_text"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    mock_task.delay.return_value = MagicMock(id="task-xyz")

    with _make_test_app().test_client() as client:
        resp = client.post(f"/api/knowledge-bases/{kb_id}/build-bm25", headers=_auth_headers())
    assert resp.status_code == 202


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
def test_build_bm25_rejects_vector_search_kb(mock_task, mock_fetch, _jwt):
    kb_id = str(uuid.uuid4())
    mock_fetch.return_value = {
        "id": kb_id,
        "retrieval_config": {"method": "vector_search"},
        "indexing_config": {"strategy": "chunk_embed"},
    }

    with _make_test_app().test_client() as client:
        resp = client.post(f"/api/knowledge-bases/{kb_id}/build-bm25", headers=_auth_headers())
    assert resp.status_code == 400
    assert "vector_search" in resp.get_json()["error"]
    mock_task.delay.assert_not_called()


@_FAKE_JWT
def test_build_bm25_rejects_invalid_uuid(_jwt):
    with _make_test_app().test_client() as client:
        resp = client.post("/api/knowledge-bases/not-a-uuid/build-bm25", headers=_auth_headers())
    # _require_uuid returns 404 for invalid UUIDs (existing helper convention)
    assert resp.status_code == 404


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
def test_build_bm25_returns_503_on_broker_failure(mock_task, mock_fetch, _jwt):
    """If Celery broker is unreachable, .delay() raises; endpoint returns 503 JSON."""
    kb_id = str(uuid.uuid4())
    mock_fetch.return_value = {
        "id": kb_id,
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    mock_task.delay.side_effect = Exception("broker unreachable")

    with _make_test_app().test_client() as client:
        resp = client.post(f"/api/knowledge-bases/{kb_id}/build-bm25", headers=_auth_headers())
    assert resp.status_code == 503
    assert "Failed to start" in resp.get_json()["error"]


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
def test_build_bm25_rejects_a_strategy_with_no_item_table(mock_task, mock_fetch, _jwt):
    """Answering 202 here was a false promise: the task then failed once.

    doc2json has no BM25 item table, so refuse it up front, from the same map
    the task reads.
    """
    kb_id = str(uuid.uuid4())
    mock_fetch.return_value = {
        "id": kb_id,
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"strategy": "doc2json"},
    }

    with _make_test_app().test_client() as client:
        resp = client.post(f"/api/knowledge-bases/{kb_id}/build-bm25", headers=_auth_headers())
    assert resp.status_code == 400
    assert "doc2json" in resp.get_json()["error"]
    mock_task.delay.assert_not_called()


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
def test_build_bm25_treats_a_missing_strategy_as_chunk_embed(mock_task, mock_fetch, _jwt):
    kb_id = str(uuid.uuid4())
    mock_fetch.return_value = {
        "id": kb_id,
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"chunk_size": 800},
    }
    mock_task.delay.return_value = MagicMock(id="task-default")

    with _make_test_app().test_client() as client:
        resp = client.post(f"/api/knowledge-bases/{kb_id}/build-bm25", headers=_auth_headers())
    assert resp.status_code == 202
    mock_task.delay.assert_called_once_with(kb_id)


# ---------------------------------------------------------------------------
# Legacy rows can hold a config column as a JSON string. The keyword-timeout
# 503 sends such a KB here, so this endpoint must answer JSON, never an HTML 500.
# ---------------------------------------------------------------------------


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
def test_build_bm25_parses_a_string_retrieval_config(mock_task, mock_fetch, _jwt):
    kb_id = str(uuid.uuid4())
    mock_fetch.return_value = {
        "id": kb_id,
        "retrieval_config": '{"method": "hybrid"}',
        "indexing_config": {"strategy": "chunk_embed"},
    }
    mock_task.delay.return_value = MagicMock(id="task-str")

    with _make_test_app().test_client() as client:
        resp = client.post(f"/api/knowledge-bases/{kb_id}/build-bm25", headers=_auth_headers())
    assert resp.status_code == 202
    body = resp.get_json()
    assert (body["task_id"], body["knowledge_base_id"]) == ("task-str", kb_id)
    mock_task.delay.assert_called_once_with(kb_id)


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
def test_build_bm25_rejects_an_unparseable_retrieval_config_as_json(mock_task, mock_fetch, _jwt):
    kb_id = str(uuid.uuid4())
    mock_fetch.return_value = {
        "id": kb_id,
        "retrieval_config": "{not json",
        "indexing_config": {"strategy": "chunk_embed"},
    }

    with _make_test_app().test_client() as client:
        resp = client.post(f"/api/knowledge-bases/{kb_id}/build-bm25", headers=_auth_headers())
    assert resp.status_code == 400
    assert resp.is_json, resp.data[:200]
    assert "retrieval_config" in resp.get_json()["error"]
    mock_task.delay.assert_not_called()


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
def test_build_bm25_refuses_a_string_indexing_config_as_json(mock_task, mock_fetch, _jwt):
    # The build task reads indexing_config as an object and the search path
    # already rejects a string one, so it is refused here rather than parsed —
    # parsing would only move the crash into the task after a 202.
    kb_id = str(uuid.uuid4())
    mock_fetch.return_value = {
        "id": kb_id,
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": '{"strategy": "chunk_embed"}',
    }

    with _make_test_app().test_client() as client:
        resp = client.post(f"/api/knowledge-bases/{kb_id}/build-bm25", headers=_auth_headers())
    assert resp.status_code == 400
    assert resp.is_json, resp.data[:200]
    assert "indexing_config" in resp.get_json()["error"]
    mock_task.delay.assert_not_called()
