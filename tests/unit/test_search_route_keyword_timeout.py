"""POST /knowledge-bases/<id>/search maps a keyword-search timeout to 503."""

import uuid
from unittest.mock import patch

from agentic_project_service.routes import knowledge_bases as kb_route
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
            json={"query": "appeal", "retrieval_method": "full_text"},
        )
    assert resp.status_code == 503
    body = resp.get_json()
    assert body["code"] == "keyword_search_timeout"
    assert body["timeout_ms"] == 10000
    assert "BM25 index" in body["error"]
