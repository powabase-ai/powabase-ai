"""POST /api/internal/docs/search answers a keyword-search timeout with a 503.

The docs KB can hit the same bounded keyword fallback as any other KB. That is
a designed, temporary degradation, so it must not reach the generic handler,
which logs at ERROR with a traceback and answers 500. search_docs already treats
any non-200 as "temporarily unavailable", so the status change is safe for it.

Minimal Flask app, as in test_internal_docs_ratelimit.py, so no Postgres needed.
"""

import logging
from unittest.mock import MagicMock

import pytest

from agentic_project_service.routes import internal_docs as internal_docs_route
from agentic_project_service.services.base_vector_store import KeywordSearchTimeout

_TOKEN = "trail-map-token"
_KB_ID = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def client(mocker, monkeypatch):
    from flask import Flask

    monkeypatch.setenv("DOCS_SEARCH_TOKEN", _TOKEN)
    monkeypatch.setenv("DOCS_KB_ID", _KB_ID)
    mocker.patch("agentic_project_service.routes.internal_docs.db", new_callable=MagicMock)
    mocker.patch("agentic_project_service.routes.internal_docs._rate_limited", return_value=False)
    app = Flask(__name__)
    app.register_blueprint(internal_docs_route.internal_docs_bp)
    with app.test_client() as c:
        yield c


def test_keyword_timeout_is_a_503_with_one_warning_and_no_error(client, mocker, caplog):
    mocker.patch(
        "agentic_project_service.routes.internal_docs.search_knowledge_base",
        side_effect=KeywordSearchTimeout(_KB_ID, 10000),
    )
    with caplog.at_level(logging.DEBUG, logger=internal_docs_route.logger.name):
        resp = client.post(
            "/api/internal/docs/search",
            headers={"X-Docs-Search-Token": _TOKEN, "Content-Type": "application/json"},
            json={"query": "weather on the ridge"},
        )

    assert resp.status_code == 503
    body = resp.get_json()
    assert body["code"] == "keyword_search_timeout"
    assert body["error"]

    route_records = [r for r in caplog.records if r.name == internal_docs_route.logger.name]
    assert [r.levelname for r in route_records if r.levelname == "ERROR"] == []
    assert len([r for r in route_records if r.levelname == "WARNING"]) == 1
