"""Tests for the bm25_status field added to the get_knowledge_base response."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agentic_project_service.routes import knowledge_bases as kb_route

_FAKE_JWT = "fake.jwt.token"

# Patch decode_jwt in the auth module so require_auth passes without a real JWT.
_AUTH_PATCH = patch(
    "agentic_project_service.auth.decode_jwt",
    return_value={"role": "service_role"},
)


def _make_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(kb_route.knowledge_bases_bp)
    return app


def _auth_headers():
    return {"Authorization": f"Bearer {_FAKE_JWT}"}


def _mock_db_counts_empty():
    """Return a db mock whose session.execute returns an empty counts result."""
    mock_db = MagicMock()
    mock_db.session.execute.return_value = iter([])  # empty GROUP BY result
    return mock_db


# ---- Endpoint-level tests: bm25_status appears or is omitted in response ----


@_AUTH_PATCH
@patch("agentic_project_service.routes.knowledge_bases._compute_bm25_status")
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
@patch("agentic_project_service.routes.knowledge_bases._compute_drift")
@patch("agentic_project_service.routes.knowledge_bases.db", new_callable=MagicMock)
def test_bm25_status_included_when_present(mock_db, mock_drift, mock_fetch, mock_status, _jwt):
    mock_db.session.execute.return_value = iter([])  # empty source_counts GROUP BY
    kb_id = "11111111-1111-1111-1111-111111111111"
    mock_fetch.return_value = {
        "id": kb_id,
        "name": "kb-1",
        "description": None,
        "indexing_config": {"strategy": "chunk_embed"},
        "retrieval_config": {"method": "hybrid"},
        "created_at": None,
        "updated_at": None,
    }
    mock_drift.return_value = "none"
    mock_status.return_value = "absent"

    with _make_test_app().test_client() as client:
        resp = client.get(f"/api/knowledge-bases/{kb_id}", headers=_auth_headers())
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["bm25_status"] == "absent"


@_AUTH_PATCH
@patch("agentic_project_service.routes.knowledge_bases._compute_bm25_status")
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
@patch("agentic_project_service.routes.knowledge_bases._compute_drift")
@patch("agentic_project_service.routes.knowledge_bases.db", new_callable=MagicMock)
def test_bm25_status_omitted_when_none(mock_db, mock_drift, mock_fetch, mock_status, _jwt):
    """If _compute_bm25_status returns None, the field is omitted from the response."""
    mock_db.session.execute.return_value = iter([])
    kb_id = "22222222-2222-2222-2222-222222222222"
    mock_fetch.return_value = {
        "id": kb_id,
        "name": "kb-2",
        "description": None,
        "indexing_config": {"strategy": "chunk_embed"},
        "retrieval_config": {"method": "vector_search"},
        "created_at": None,
        "updated_at": None,
    }
    mock_drift.return_value = "none"
    mock_status.return_value = None

    with _make_test_app().test_client() as client:
        resp = client.get(f"/api/knowledge-bases/{kb_id}", headers=_auth_headers())
    body = resp.get_json()
    assert "bm25_status" not in body


# ---- _compute_bm25_status logic tests (no Flask app context needed) ----


@patch("agentic_project_service.routes.knowledge_bases.SparseIndexStore")
@patch("agentic_project_service.routes.knowledge_bases.get_setting")
def test_compute_status_returns_none_for_vector_search(mock_setting, mock_store_cls):
    from agentic_project_service.routes.knowledge_bases import _compute_bm25_status

    kb = {
        "id": "kb",
        "retrieval_config": {"method": "vector_search"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    assert _compute_bm25_status(kb) is None
    mock_setting.assert_not_called()


@patch("agentic_project_service.routes.knowledge_bases.SparseIndexStore")
@patch("agentic_project_service.routes.knowledge_bases.get_setting")
def test_compute_status_returns_none_when_auto_indexing_on(mock_setting, mock_store_cls):
    from agentic_project_service.routes.knowledge_bases import _compute_bm25_status

    mock_setting.return_value = True  # auto on
    kb = {
        "id": "kb",
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    assert _compute_bm25_status(kb) is None


@patch("agentic_project_service.routes.knowledge_bases._count_items_for_kb_bm25")
@patch("agentic_project_service.routes.knowledge_bases.SparseIndexStore")
@patch("agentic_project_service.routes.knowledge_bases.get_setting")
def test_compute_status_absent_when_no_files(mock_setting, mock_store_cls, mock_count):
    from agentic_project_service.routes.knowledge_bases import _compute_bm25_status

    mock_setting.return_value = False  # manual mode
    store = mock_store_cls.return_value
    store.index_exists.return_value = False

    kb = {
        "id": "kb",
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    assert _compute_bm25_status(kb) == "absent"


@patch("agentic_project_service.routes.knowledge_bases._count_items_for_kb_bm25")
@patch("agentic_project_service.routes.knowledge_bases.SparseIndexStore")
@patch("agentic_project_service.routes.knowledge_bases.get_setting")
def test_compute_status_ready_when_counts_match(mock_setting, mock_store_cls, mock_count):
    from agentic_project_service.routes.knowledge_bases import _compute_bm25_status

    mock_setting.return_value = False
    store = mock_store_cls.return_value
    store.index_exists.return_value = True
    store.read_metadata.return_value = {"item_count": 500}
    mock_count.return_value = 500

    kb = {
        "id": "kb",
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    assert _compute_bm25_status(kb) == "ready"


@patch("agentic_project_service.routes.knowledge_bases._count_items_for_kb_bm25")
@patch("agentic_project_service.routes.knowledge_bases.SparseIndexStore")
@patch("agentic_project_service.routes.knowledge_bases.get_setting")
def test_compute_status_stale_when_count_grew(mock_setting, mock_store_cls, mock_count):
    from agentic_project_service.routes.knowledge_bases import _compute_bm25_status

    mock_setting.return_value = False
    store = mock_store_cls.return_value
    store.index_exists.return_value = True
    store.read_metadata.return_value = {"item_count": 500}
    mock_count.return_value = 750

    kb = {
        "id": "kb",
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    assert _compute_bm25_status(kb) == "stale"


@patch("agentic_project_service.routes.knowledge_bases._count_items_for_kb_bm25")
@patch("agentic_project_service.routes.knowledge_bases.SparseIndexStore")
@patch("agentic_project_service.routes.knowledge_bases.get_setting")
def test_compute_status_stale_when_metadata_missing(mock_setting, mock_store_cls, mock_count):
    """Legacy index pre-dating the sidecar — treat as stale to prompt rebuild."""
    from agentic_project_service.routes.knowledge_bases import _compute_bm25_status

    mock_setting.return_value = False
    store = mock_store_cls.return_value
    store.index_exists.return_value = True
    store.read_metadata.return_value = None
    mock_count.return_value = 100

    kb = {
        "id": "kb",
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    assert _compute_bm25_status(kb) == "stale"
