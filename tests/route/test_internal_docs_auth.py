"""The internal docs endpoint must fail CLOSED when its token is unset.

This endpoint is reachable at :5000 by every other project's copilot, bypassing
Kong. An unset-token default of "allow" would make it an open, unauthenticated
retrieval endpoint on the cluster network.
"""

import pytest


@pytest.fixture
def client_without_token(app, monkeypatch):
    monkeypatch.delenv("DOCS_SEARCH_TOKEN", raising=False)
    return app.test_client()


def test_unset_token_rejects_every_call(client_without_token):
    r = client_without_token.post("/api/internal/docs/search", json={"query": "x"})
    assert r.status_code in (401, 403), f"fail-open: got {r.status_code}"


def test_wrong_token_is_rejected(app, monkeypatch):
    monkeypatch.setenv("DOCS_SEARCH_TOKEN", "correct-token")
    monkeypatch.setenv("DOCS_KB_ID", "00000000-0000-0000-0000-0000d0c5d0c5")
    r = app.test_client().post(
        "/api/internal/docs/search",
        json={"query": "x"},
        headers={"X-Docs-Search-Token": "wrong-token"},
    )
    assert r.status_code in (401, 403)


def test_unset_kb_id_returns_503_not_a_crash(app, monkeypatch):
    """Not the docs project -> the endpoint is inert, not broken."""
    monkeypatch.setenv("DOCS_SEARCH_TOKEN", "correct-token")
    monkeypatch.delenv("DOCS_KB_ID", raising=False)
    r = app.test_client().post(
        "/api/internal/docs/search",
        json={"query": "x"},
        headers={"X-Docs-Search-Token": "correct-token"},
    )
    assert r.status_code == 503
