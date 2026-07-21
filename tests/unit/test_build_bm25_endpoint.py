"""Tests for POST /api/knowledge-bases/<kb_id>/build-bm25."""

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
    assert body == {"task_id": "task-abc", "knowledge_base_id": kb_id}
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
