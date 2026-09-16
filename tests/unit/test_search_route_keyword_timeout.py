"""POST /knowledge-bases/<id>/search maps a keyword-search timeout to 503, and
reports a hybrid search that silently lost its keyword leg."""

import uuid
from unittest.mock import patch

import pytest
from agentic.knowledge.models import RetrievedItem

from agentic_project_service.routes import knowledge_bases as kb_route
from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services.base_vector_store import KeywordSearchTimeout


def _app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(kb_route.knowledge_bases_bp)
    return app


def _kb(strategy: str, stored_method: str) -> dict:
    return {
        "id": "kb",
        "indexing_config": {"strategy": strategy},
        "retrieval_config": {"method": stored_method},
    }


def _timeout_503(kb: dict | tuple, request_method: str = "full_text"):
    """Drive the 503 path with a given KB row and return the parsed body."""
    kb_id = str(uuid.uuid4())
    with (
        patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"}),
        patch("agentic_project_service.routes.knowledge_bases.db"),
        patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404", return_value=kb),
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
    body = resp.get_json()
    assert body["code"] == "keyword_search_timeout"
    assert body["timeout_ms"] == 10000
    # Every branch must keep saying that nothing self-heals.
    assert "No build starts on its own" in body["error"]
    assert "queued" not in body["error"]
    return body


def test_keyword_timeout_is_503():
    """The buildable case: mapped strategy, stored method already hybrid."""
    body = _timeout_503(_kb("chunk_embed", "hybrid"))
    assert "build-bm25" in body["error"]
    assert "vector_search" in body["error"]


@pytest.mark.parametrize("strategy", ["doc2json", "page_index"])
def test_unmapped_strategy_is_never_told_to_build(strategy):
    """These strategies have no BM25 item table, so a build cannot help them.

    POST /build-bm25 would answer 202 and the task would then die with
    ValueError and retry twice. They are also the knowledge bases permanently on
    the tsvector fallback, i.e. the likeliest 503 producers, so naming that
    endpoint here would send every one of them down a dead end.
    """
    body = _timeout_503(_kb(strategy, "hybrid"))
    assert "build-bm25" not in body["error"]
    assert "vector_search" in body["error"]
    assert strategy in body["error"]


def test_stored_method_must_allow_a_build_before_one_is_suggested():
    """A per-request retrieval_method override does not make /build-bm25 work.

    That endpoint 400s unless the KB's STORED method is hybrid or full_text, so
    the remedy has to name that step first.
    """
    body = _timeout_503(_kb("chunk_embed", "vector_search"), request_method="full_text")
    assert "stored retrieval method" in body["error"]
    assert "hybrid or full_text" in body["error"]
    assert "build-bm25" in body["error"]


def test_unresolvable_kb_falls_back_to_the_generic_remedy():
    """A 404 tuple or a failed lookup must not produce a nonsense strategy name."""
    body = _timeout_503((None, 404))
    assert "build-bm25" in body["error"]
    assert "vector_search" in body["error"]
    assert "None" not in body["error"]


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
