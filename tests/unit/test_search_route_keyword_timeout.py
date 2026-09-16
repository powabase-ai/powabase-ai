"""POST /knowledge-bases/<id>/search maps a keyword-search timeout to 503, and
reports a hybrid search that silently lost its keyword leg."""

import uuid
from unittest.mock import patch

from agentic.knowledge.models import RetrievedItem

from agentic_project_service.routes import knowledge_bases as kb_route
from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services.base_vector_store import KeywordSearchTimeout


def _app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(kb_route.knowledge_bases_bp)
    return app


@patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"})
@patch("agentic_project_service.routes.knowledge_bases.db")
@patch("agentic_project_service.services.knowledge_search.search_knowledge_base")
def test_keyword_timeout_is_503(mock_search, _db, _jwt):
    kb_id = str(uuid.uuid4())
    mock_search.side_effect = KeywordSearchTimeout(kb_id, 10000)
    with _app().test_client() as c:
        resp = c.post(
            f"/api/knowledge-bases/{kb_id}/search",
            headers={"Authorization": "Bearer fake.jwt.token"},
            json={"query": "weather", "retrieval_method": "full_text"},
        )
    assert resp.status_code == 503
    body = resp.get_json()
    assert body["code"] == "keyword_search_timeout"
    assert body["timeout_ms"] == 10000
    # Nothing self-heals here, so the body must name the two real remedies
    # rather than promise a build that no code path queues.
    assert "build-bm25" in body["error"]
    assert "vector_search" in body["error"]
    assert "queued" not in body["error"]


def _item() -> RetrievedItem:
    return RetrievedItem(
        item_id="v1",
        text="a vector hit",
        score=0.9,
        source_id=None,
        knowledge_base_id="kb",
        meta={},
    )


def _post(client, kb_id: str, method: str = "hybrid"):
    return client.post(
        f"/api/knowledge-bases/{kb_id}/search",
        headers={"Authorization": "Bearer fake.jwt.token"},
        json={"query": "weather", "retrieval_method": method},
    )


@patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"})
@patch("agentic_project_service.routes.knowledge_bases.db")
@patch("agentic_project_service.services.knowledge_search.search_knowledge_base")
def test_degraded_hybrid_search_reports_the_dropped_leg(mock_search, _db, _jwt):
    """A hybrid answer built from vector results alone must say so.

    Otherwise it is indistinguishable from a healthy hybrid answer: the items
    still carry retrieval_method="hybrid" and nothing else changes.
    """
    kb_id = str(uuid.uuid4())

    def degrade(**kwargs):
        bvs.record_retrieval_degradation(bvs.KEYWORD_SEARCH_TIMEOUT)
        return [_item()]

    mock_search.side_effect = degrade

    with _app().test_client() as c:
        resp = _post(c, kb_id)

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["degraded"] == ["keyword_search_timeout"]
    # The resolved method is unchanged: the search really did run as hybrid.
    assert body["retrieval_method"] == "hybrid"


@patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"})
@patch("agentic_project_service.routes.knowledge_bases.db")
@patch("agentic_project_service.services.knowledge_search.search_knowledge_base")
def test_healthy_search_omits_the_degraded_field(mock_search, _db, _jwt):
    kb_id = str(uuid.uuid4())
    mock_search.return_value = [_item()]

    with _app().test_client() as c:
        resp = _post(c, kb_id)

    assert resp.status_code == 200
    assert "degraded" not in resp.get_json()


@patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"})
@patch("agentic_project_service.routes.knowledge_bases.db")
@patch("agentic_project_service.services.knowledge_search.search_knowledge_base")
def test_degradation_does_not_leak_into_the_next_request(mock_search, _db, _jwt):
    """The record is per-request state, not per-process."""
    kb_id = str(uuid.uuid4())
    calls = {"n": 0}

    def maybe_degrade(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            bvs.record_retrieval_degradation(bvs.KEYWORD_SEARCH_TIMEOUT)
        return [_item()]

    mock_search.side_effect = maybe_degrade

    with _app().test_client() as c:
        first = _post(c, kb_id)
        second = _post(c, kb_id)

    assert first.get_json()["degraded"] == ["keyword_search_timeout"]
    assert "degraded" not in second.get_json()


def test_recording_a_degradation_outside_a_request_is_a_no_op():
    """Celery tasks and bare threads have no request context to write to."""
    bvs.record_retrieval_degradation(bvs.KEYWORD_SEARCH_TIMEOUT)
    assert bvs.get_retrieval_degradations() == []
