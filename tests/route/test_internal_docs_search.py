"""Tests for POST /api/internal/docs/search (Project Copilot — central docs RAG).

This endpoint lives on the singleton "system docs" project. A per-project
Project Copilot's `search_docs` tool calls it over internal HTTP, authenticating
with a shared bearer token (``DOCS_SEARCH_TOKEN``) — NOT a per-project JWT, since
the caller is a *different* project's service that does not hold this project's
service_role_key.

Auth is fail-closed: if ``DOCS_SEARCH_TOKEN`` is unset/empty the endpoint rejects
every call. ``DOCS_KB_ID`` names the hidden full_document docs KB to search; when
unset the endpoint is "not a docs project" and returns 503.

``search_knowledge_base`` is patched so these tests don't need real embeddings.
"""

from dataclasses import dataclass

import pytest


@dataclass
class _FakeItem:
    """Mirror of agentic.knowledge.models.RetrievedItem (only the read fields)."""

    item_id: str
    text: str
    score: float
    source_id: str | None = None
    knowledge_base_id: str | None = None
    meta: dict | None = None


_TOKEN = "s3cr3t-docs-token"
_KB_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def _docs_env(monkeypatch):
    """Configure this app instance as a docs project with a known token."""
    monkeypatch.setenv("DOCS_SEARCH_TOKEN", _TOKEN)
    monkeypatch.setenv("DOCS_KB_ID", _KB_ID)


def _hdr(token: str) -> dict:
    return {"X-Docs-Search-Token": token, "Content-Type": "application/json"}


def test_search_rejects_when_token_env_unset(client, monkeypatch):
    """Fail-closed: no DOCS_SEARCH_TOKEN configured -> 401 even with a header."""
    monkeypatch.delenv("DOCS_SEARCH_TOKEN", raising=False)
    monkeypatch.setenv("DOCS_KB_ID", _KB_ID)
    resp = client.post("/api/internal/docs/search", headers=_hdr("anything"), json={"query": "x"})
    assert resp.status_code == 401


def test_search_rejects_wrong_token(client, _docs_env):
    resp = client.post(
        "/api/internal/docs/search", headers=_hdr("wrong"), json={"query": "x"}
    )
    assert resp.status_code == 401


def test_search_rejects_missing_token_header(client, _docs_env):
    resp = client.post(
        "/api/internal/docs/search",
        headers={"Content-Type": "application/json"},
        json={"query": "x"},
    )
    assert resp.status_code == 401


def test_search_503_when_kb_not_configured(client, monkeypatch):
    """Correct token but no DOCS_KB_ID -> this isn't a docs project -> 503."""
    monkeypatch.setenv("DOCS_SEARCH_TOKEN", _TOKEN)
    monkeypatch.delenv("DOCS_KB_ID", raising=False)
    resp = client.post(
        "/api/internal/docs/search", headers=_hdr(_TOKEN), json={"query": "x"}
    )
    assert resp.status_code == 503


def test_search_400_on_empty_query(client, _docs_env):
    resp = client.post("/api/internal/docs/search", headers=_hdr(_TOKEN), json={"query": "  "})
    assert resp.status_code == 400


def test_search_returns_formatted_results(client, _docs_env, mocker):
    """A valid call returns text/score/source plus title+url pulled from meta."""
    captured = {}

    def _fake_search(db_session, knowledge_base_id, query, top_k=5, **kwargs):
        captured["kb_id"] = knowledge_base_id
        captured["query"] = query
        captured["top_k"] = top_k
        return [
            _FakeItem(
                item_id="i1",
                text="To connect your coding agent, copy the connection string...",
                score=0.91,
                source_id="src-1",
                meta={"title": "Auth connection", "url": "https://docs.powabase.ai/guides/auth-connection"},
            ),
            _FakeItem(item_id="i2", text="Tables live in the editor.", score=0.42, source_id="src-2", meta={}),
        ]

    mocker.patch(
        "agentic_project_service.routes.internal_docs.search_knowledge_base",
        side_effect=_fake_search,
    )

    resp = client.post(
        "/api/internal/docs/search",
        headers=_hdr(_TOKEN),
        json={"query": "how do I connect my coding agent?", "top_k": 6},
    )
    assert resp.status_code == 200
    body = resp.get_json()

    # the configured KB + passed-through args were used
    assert captured["kb_id"] == _KB_ID
    assert captured["query"] == "how do I connect my coding agent?"
    assert captured["top_k"] == 6

    results = body["results"]
    assert len(results) == 2
    assert results[0]["text"].startswith("To connect your coding agent")
    assert results[0]["score"] == pytest.approx(0.91)
    assert results[0]["source_id"] == "src-1"
    assert results[0]["title"] == "Auth connection"
    assert results[0]["url"] == "https://docs.powabase.ai/guides/auth-connection"
    # missing meta -> title/url are None, not a crash
    assert results[1]["title"] is None
    assert results[1]["url"] is None


def test_search_empty_kb_returns_200_empty_with_not_ready_flag(client, _docs_env, mocker):
    """A genuinely empty/not-yet-indexed KB raises EmptyKnowledgeBaseError -> 200
    (not 500), but MUST be distinguishable from a real no-match: it carries
    ``kb_not_ready: true`` so the copilot can degrade to a grounding-unavailable
    notice instead of answering ungrounded with no signal."""
    from agentic_project_service.services.knowledge_search import EmptyKnowledgeBaseError

    mocker.patch(
        "agentic_project_service.routes.internal_docs.search_knowledge_base",
        side_effect=EmptyKnowledgeBaseError("No documents indexed in this knowledge base."),
    )
    resp = client.post("/api/internal/docs/search", headers=_hdr(_TOKEN), json={"query": "x"})
    assert resp.status_code == 200
    assert resp.get_json() == {"results": [], "kb_not_ready": True}


def test_search_real_no_match_does_not_set_kb_not_ready(client, _docs_env, mocker):
    """A working search that genuinely matched nothing (search_knowledge_base
    returns an empty list, no exception) must NOT carry the kb_not_ready flag —
    only the EmptyKnowledgeBaseError path is "not ready"."""
    mocker.patch(
        "agentic_project_service.routes.internal_docs.search_knowledge_base",
        return_value=[],
    )
    resp = client.post("/api/internal/docs/search", headers=_hdr(_TOKEN), json={"query": "x"})
    assert resp.status_code == 200
    assert resp.get_json() == {"results": []}


def test_search_broken_index_returns_500(client, _docs_env, mocker):
    """A NON-empty ValueError (bad retrieval_config / embedding-dim mismatch /
    pgvector parse) is a broken index, NOT an empty KB — it must 500 (visible in
    monitoring), not be masked as 200 []."""
    mocker.patch(
        "agentic_project_service.routes.internal_docs.search_knowledge_base",
        side_effect=ValueError("Retrieval method 'x' is not compatible with strategy 'y'."),
    )
    resp = client.post("/api/internal/docs/search", headers=_hdr(_TOKEN), json={"query": "x"})
    assert resp.status_code == 500


def test_search_prefers_source_auto_metadata_for_citation(client, app, _docs_env, mocker):
    """The production full_document path carries title/url on the ingested source's
    ``auto_metadata`` (that's where docs_refresh records them), NOT on the item —
    so src_meta must win over the item's own meta."""
    from sqlalchemy import text

    from agentic_project_service.db import db

    src_id = "22222222-2222-2222-2222-222222222222"
    with app.app_context():
        db.session.execute(
            text(
                "INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status, "
                "content_hash, auto_metadata) VALUES (:id, :n, 'md', 'sources/x', 'extracted', :h, "
                "CAST(:m AS jsonb))"
            ),
            {
                "id": src_id,
                "n": "docs:guides/auth.md",
                "h": "hash-" + src_id,
                "m": '{"title": "SRC TITLE", "url": "https://docs.powabase.ai/guides/auth"}',
            },
        )
        db.session.commit()

    mocker.patch(
        "agentic_project_service.routes.internal_docs.search_knowledge_base",
        return_value=[
            _FakeItem(
                item_id="i1",
                text="body",
                score=0.9,
                source_id=src_id,
                meta={"title": "ITEM TITLE", "url": "https://item.example/x"},
            )
        ],
    )
    resp = client.post("/api/internal/docs/search", headers=_hdr(_TOKEN), json={"query": "x"})
    assert resp.status_code == 200
    result = resp.get_json()["results"][0]
    assert result["title"] == "SRC TITLE"
    assert result["url"] == "https://docs.powabase.ai/guides/auth"


def test_search_rejects_overlong_query(client, _docs_env):
    resp = client.post(
        "/api/internal/docs/search", headers=_hdr(_TOKEN), json={"query": "a" * 5000}
    )
    assert resp.status_code == 400


def test_get_redis_client_is_cached(mocker):
    """The Redis client is built once and reused across calls — no fresh
    ConnectionPool per request (see routes/internal_docs.py's _get_redis)."""
    from agentic_project_service.routes import internal_docs

    mocker.patch.object(internal_docs, "_redis_client", None)
    fake_client = object()
    from_url_mock = mocker.patch(
        "agentic_project_service.routes.internal_docs.redis.from_url",
        return_value=fake_client,
    )

    first = internal_docs._get_redis()
    second = internal_docs._get_redis()

    assert first is second is fake_client
    from_url_mock.assert_called_once()


def test_search_defaults_top_k(client, _docs_env, mocker):
    """top_k omitted -> a sensible default is passed to the search."""
    captured = {}

    def _fake_search(db_session, knowledge_base_id, query, top_k=5, **kwargs):
        captured["top_k"] = top_k
        return []

    mocker.patch(
        "agentic_project_service.routes.internal_docs.search_knowledge_base",
        side_effect=_fake_search,
    )
    resp = client.post("/api/internal/docs/search", headers=_hdr(_TOKEN), json={"query": "x"})
    assert resp.status_code == 200
    assert captured["top_k"] == 8
