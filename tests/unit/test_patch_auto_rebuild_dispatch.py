"""Tests that PATCH /knowledge-bases/<id> auto-dispatches build_bm25_for_kb
when retrieval method transitions from non-BM25 to BM25 (and the
BM25_AUTO_INDEXING setting is on)."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from agentic_project_service.routes import knowledge_bases as kb_route

_FAKE_JWT = "fake.jwt.token"


def _make_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(kb_route.knowledge_bases_bp)
    return app


def _auth_headers():
    return {"Authorization": f"Bearer {_FAKE_JWT}"}


# Patch decode_jwt so require_auth passes without a real JWT_SECRET / token.
_FAKE_DECODE = patch(
    "agentic_project_service.auth.decode_jwt",
    return_value={"role": "service_role"},
)


@_FAKE_DECODE
@patch("agentic_project_service.routes.knowledge_bases._read_existing_retrieval_config")
@patch("agentic_project_service.routes.knowledge_bases.get_setting")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
@patch("agentic_project_service.routes.knowledge_bases.get_knowledge_base")
@patch("agentic_project_service.routes.knowledge_bases.db")
def test_patch_dispatches_on_vec_to_hybrid_with_setting_on(
    mock_db, mock_get, mock_task, mock_setting, mock_old_config, _jwt
):
    mock_old_config.return_value = {"method": "vector_search"}
    mock_setting.return_value = True
    mock_get.return_value = ({"id": "kb"}, 200)

    kb_id = str(uuid.uuid4())
    with _make_test_app().test_client() as client:
        resp = client.patch(
            f"/api/knowledge-bases/{kb_id}",
            headers=_auth_headers(),
            json={"retrieval_config": {"method": "hybrid"}},
        )
    assert resp.status_code in (200, 204)
    mock_task.delay.assert_called_once_with(kb_id)


@_FAKE_DECODE
@patch("agentic_project_service.routes.knowledge_bases._read_existing_retrieval_config")
@patch("agentic_project_service.routes.knowledge_bases.get_setting")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
@patch("agentic_project_service.routes.knowledge_bases.get_knowledge_base")
@patch("agentic_project_service.routes.knowledge_bases.db")
def test_patch_does_not_dispatch_when_setting_off(
    mock_db, mock_get, mock_task, mock_setting, mock_old_config, _jwt
):
    mock_old_config.return_value = {"method": "vector_search"}
    mock_setting.return_value = False  # manual mode
    mock_get.return_value = ({"id": "kb"}, 200)

    kb_id = str(uuid.uuid4())
    with _make_test_app().test_client() as client:
        client.patch(
            f"/api/knowledge-bases/{kb_id}",
            headers=_auth_headers(),
            json={"retrieval_config": {"method": "hybrid"}},
        )
    mock_task.delay.assert_not_called()


@_FAKE_DECODE
@patch("agentic_project_service.routes.knowledge_bases._read_existing_retrieval_config")
@patch("agentic_project_service.routes.knowledge_bases.get_setting")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
@patch("agentic_project_service.routes.knowledge_bases.get_knowledge_base")
@patch("agentic_project_service.routes.knowledge_bases.db")
def test_patch_does_not_dispatch_on_hybrid_to_full_text(
    mock_db, mock_get, mock_task, mock_setting, mock_old_config, _jwt
):
    """Both methods use BM25; no rebuild needed."""
    mock_old_config.return_value = {"method": "hybrid"}
    mock_setting.return_value = True
    mock_get.return_value = ({"id": "kb"}, 200)

    kb_id = str(uuid.uuid4())
    with _make_test_app().test_client() as client:
        client.patch(
            f"/api/knowledge-bases/{kb_id}",
            headers=_auth_headers(),
            json={"retrieval_config": {"method": "full_text"}},
        )
    mock_task.delay.assert_not_called()


@_FAKE_DECODE
@patch("agentic_project_service.routes.knowledge_bases._read_existing_retrieval_config")
@patch("agentic_project_service.routes.knowledge_bases.get_setting")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
@patch("agentic_project_service.routes.knowledge_bases.get_knowledge_base")
@patch("agentic_project_service.routes.knowledge_bases.db")
def test_patch_does_not_dispatch_on_hybrid_to_vec(
    mock_db, mock_get, mock_task, mock_setting, mock_old_config, _jwt
):
    mock_old_config.return_value = {"method": "hybrid"}
    mock_setting.return_value = True
    mock_get.return_value = ({"id": "kb"}, 200)

    kb_id = str(uuid.uuid4())
    with _make_test_app().test_client() as client:
        client.patch(
            f"/api/knowledge-bases/{kb_id}",
            headers=_auth_headers(),
            json={"retrieval_config": {"method": "vector_search"}},
        )
    mock_task.delay.assert_not_called()


@_FAKE_DECODE
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
@patch("agentic_project_service.routes.knowledge_bases.get_knowledge_base")
@patch("agentic_project_service.routes.knowledge_bases.db")
def test_patch_does_not_dispatch_when_only_name_changes(mock_db, mock_get, mock_task, _jwt):
    """PATCH that doesn't touch retrieval_config doesn't dispatch."""
    mock_get.return_value = ({"id": "kb"}, 200)

    kb_id = str(uuid.uuid4())
    with _make_test_app().test_client() as client:
        client.patch(
            f"/api/knowledge-bases/{kb_id}",
            headers=_auth_headers(),
            json={"name": "new name"},
        )
    mock_task.delay.assert_not_called()


@_FAKE_DECODE
@patch("agentic_project_service.routes.knowledge_bases._read_existing_retrieval_config")
@patch("agentic_project_service.routes.knowledge_bases.get_setting")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
@patch("agentic_project_service.routes.knowledge_bases.get_knowledge_base")
@patch("agentic_project_service.routes.knowledge_bases.db")
def test_patch_succeeds_when_broker_dispatch_fails(
    mock_db, mock_get, mock_task, mock_setting, mock_old_config, _jwt
):
    """Broker failure during auto-dispatch must not turn the PATCH into a 500.

    The PATCH has already committed; the user should still get the KB
    response back. The bm25_status will reflect 'absent' on next GET.
    """
    mock_old_config.return_value = {"method": "vector_search"}
    mock_setting.return_value = True
    mock_get.return_value = ({"id": "kb"}, 200)
    mock_task.delay.side_effect = Exception("broker unreachable")

    kb_id = str(uuid.uuid4())
    with _make_test_app().test_client() as client:
        resp = client.patch(
            f"/api/knowledge-bases/{kb_id}",
            headers=_auth_headers(),
            json={"retrieval_config": {"method": "hybrid"}},
        )
    # PATCH still succeeds — the dispatch failure is logged, not raised
    assert resp.status_code in (200, 204)
