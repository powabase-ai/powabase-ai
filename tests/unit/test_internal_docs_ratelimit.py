"""Tests for the Redis fixed-window rate-limit on POST /api/internal/docs/search.

The docs-search endpoint is reachable two ways: (1) directly at :5000 by other
projects' Project Copilot `search_docs` tool inside K8s (bypassing Kong's
NetworkPolicy carve-out — see the handler module docstring/comment), and (2)
via Kong on self-host (which additionally gets a `rate-limiting` plugin at the
edge). Only an application-level limit in the handler covers BOTH paths, so
this suite exercises that limiter directly against the Flask test client.

Redis is mocked — these tests never touch a live Redis. The limiter must also
fail OPEN: a Redis error must never turn into a 429 or a 500.

Moved into tests/unit/ (from tests/route/) so CI actually runs it — these two
invariants (per-caller-key isolation + fail-open) were previously only
exercised locally via `make test` (which needs Postgres+pgvector), even
though the route itself needs no DB for these mocked scenarios. Uses the
minimal-Flask-app pattern (see test_bm25_status_field.py) instead of the
full `client`/`app` fixtures from the parent conftest, which require a real
Postgres to bootstrap.
"""

from unittest.mock import MagicMock

import pytest

from agentic_project_service.routes import internal_docs as internal_docs_route

_TOKEN = "s3cr3t-docs-token"
_KB_ID = "11111111-1111-1111-1111-111111111111"


def _make_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(internal_docs_route.internal_docs_bp)
    return app


@pytest.fixture
def client():
    with _make_test_app().test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def _mock_db(mocker):
    """The success path passes db.session to search_knowledge_base (mocked
    below) — patch db so evaluating db.session doesn't need this minimal
    Flask app to be bound via db.init_app()."""
    mocker.patch("agentic_project_service.routes.internal_docs.db", new_callable=MagicMock)


@pytest.fixture
def _docs_env(monkeypatch):
    monkeypatch.setenv("DOCS_SEARCH_TOKEN", _TOKEN)
    monkeypatch.setenv("DOCS_KB_ID", _KB_ID)


def _hdr(token: str, caller: str = "proj-a") -> dict:
    return {
        "X-Docs-Search-Token": token,
        "X-Caller-Project": caller,
        "Content-Type": "application/json",
    }


@pytest.fixture
def _mock_search(mocker):
    """Retrieval always returns an empty result set — only the rate-limit gate
    is under test here."""
    mocker.patch(
        "agentic_project_service.routes.internal_docs.search_knowledge_base",
        return_value=[],
    )


class _FakeRedis:
    """Hand-rolled fake mirroring the atomic INCR+EXPIRE Lua-EVAL pattern."""

    def __init__(self):
        self.counts: dict[str, int] = {}

    def eval(self, script, numkeys, *keys_and_args):
        key = keys_and_args[0]
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]


def test_rate_limit_returns_429_over_cap(client, _docs_env, _mock_search, mocker):
    """With the cap monkeypatched down to 2, the 3rd call in the same minute
    (same project + epoch-minute key) gets rejected before retrieval runs."""
    fake = _FakeRedis()
    mocker.patch(
        "agentic_project_service.routes.internal_docs._get_redis", return_value=fake
    )
    mocker.patch("agentic_project_service.routes.internal_docs._RATE_LIMIT_PER_MIN", 2)
    # Keep the global backstop out of the way — this test is exercising the
    # per-caller tier only.
    mocker.patch(
        "agentic_project_service.routes.internal_docs._RATE_LIMIT_GLOBAL_PER_MIN", 10_000
    )

    for _ in range(2):
        resp = client.post("/api/internal/docs/search", headers=_hdr(_TOKEN), json={"query": "x"})
        assert resp.status_code == 200

    resp = client.post("/api/internal/docs/search", headers=_hdr(_TOKEN), json={"query": "x"})
    assert resp.status_code == 429
    assert resp.get_json() == {"error": "rate limited"}


def test_under_cap_passes(client, _docs_env, _mock_search, mocker):
    """Calls under the cap succeed; retrieval is reached and returns 200."""
    fake = _FakeRedis()
    mocker.patch(
        "agentic_project_service.routes.internal_docs._get_redis", return_value=fake
    )
    mocker.patch("agentic_project_service.routes.internal_docs._RATE_LIMIT_PER_MIN", 120)
    mocker.patch(
        "agentic_project_service.routes.internal_docs._RATE_LIMIT_GLOBAL_PER_MIN", 10_000
    )

    for _ in range(5):
        resp = client.post("/api/internal/docs/search", headers=_hdr(_TOKEN), json={"query": "x"})
        assert resp.status_code == 200
        assert resp.get_json() == {"results": []}


def test_fail_open_when_redis_down(client, _docs_env, _mock_search, mocker):
    """Redis unreachable/erroring must ALLOW the request, not 429/500."""
    mocker.patch(
        "agentic_project_service.routes.internal_docs._get_redis",
        side_effect=ConnectionError("redis unreachable"),
    )
    mocker.patch("agentic_project_service.routes.internal_docs._RATE_LIMIT_PER_MIN", 1)

    for _ in range(5):
        resp = client.post("/api/internal/docs/search", headers=_hdr(_TOKEN), json={"query": "x"})
        assert resp.status_code == 200
        assert resp.get_json() == {"results": []}


def test_rate_limit_is_per_caller(client, _docs_env, _mock_search, mocker):
    """Two callers with different X-Caller-Project values must get their OWN
    counters: proj-a hitting the cap must not 429 proj-b in the same window.

    This pins the actual fix — before it, the limiter keyed on the docs
    project's OWN (constant) PROJECT_REF, so every caller shared one bucket
    fleet-wide and proj-a exhausting it would 429 proj-b too."""
    fake = _FakeRedis()
    mocker.patch(
        "agentic_project_service.routes.internal_docs._get_redis", return_value=fake
    )
    mocker.patch("agentic_project_service.routes.internal_docs._RATE_LIMIT_PER_MIN", 2)
    mocker.patch(
        "agentic_project_service.routes.internal_docs._RATE_LIMIT_GLOBAL_PER_MIN", 10_000
    )

    for _ in range(2):
        resp = client.post(
            "/api/internal/docs/search", headers=_hdr(_TOKEN, "proj-a"), json={"query": "x"}
        )
        assert resp.status_code == 200

    # proj-a is now at cap; its 3rd call in this window 429s.
    resp = client.post(
        "/api/internal/docs/search", headers=_hdr(_TOKEN, "proj-a"), json={"query": "x"}
    )
    assert resp.status_code == 429

    # proj-b has never called before — its own bucket is still under cap.
    resp = client.post(
        "/api/internal/docs/search", headers=_hdr(_TOKEN, "proj-b"), json={"query": "x"}
    )
    assert resp.status_code == 200
    assert resp.get_json() == {"results": []}


def test_rate_limit_window_resets(client, _docs_env, _mock_search, mocker):
    """Once a caller hits the cap, advancing into the next epoch-minute must
    give that same caller a fresh bucket (the window key includes the minute,
    so it isn't stuck 429ing forever)."""
    fake = _FakeRedis()
    mocker.patch(
        "agentic_project_service.routes.internal_docs._get_redis", return_value=fake
    )
    mocker.patch("agentic_project_service.routes.internal_docs._RATE_LIMIT_PER_MIN", 1)
    mocker.patch(
        "agentic_project_service.routes.internal_docs._RATE_LIMIT_GLOBAL_PER_MIN", 10_000
    )

    base_time = 1_800_000_000.0  # arbitrary, aligned to a minute boundary below
    mocker.patch(
        "agentic_project_service.routes.internal_docs.time.time",
        return_value=base_time - (base_time % 60),
    )

    resp = client.post(
        "/api/internal/docs/search", headers=_hdr(_TOKEN, "proj-a"), json={"query": "x"}
    )
    assert resp.status_code == 200

    resp = client.post(
        "/api/internal/docs/search", headers=_hdr(_TOKEN, "proj-a"), json={"query": "x"}
    )
    assert resp.status_code == 429

    # Advance one full minute into the next epoch-minute window.
    mocker.patch(
        "agentic_project_service.routes.internal_docs.time.time",
        return_value=base_time - (base_time % 60) + 60,
    )
    resp = client.post(
        "/api/internal/docs/search", headers=_hdr(_TOKEN, "proj-a"), json={"query": "x"}
    )
    assert resp.status_code == 200
    assert resp.get_json() == {"results": []}


def test_global_rate_limit_backstop_trips_across_different_callers(
    client, _docs_env, _mock_search, mocker
):
    """A header-spoofing caller that rotates X-Caller-Project on every request
    never trips the per-caller cap (each value's own bucket stays low), but the
    COARSE GLOBAL backstop must still bound total fleet spend on the endpoint.
    With the per-caller cap left high and the global cap monkeypatched down to
    3, the 4th request — using a brand-new caller value every time — 429s."""
    fake = _FakeRedis()
    mocker.patch(
        "agentic_project_service.routes.internal_docs._get_redis", return_value=fake
    )
    mocker.patch("agentic_project_service.routes.internal_docs._RATE_LIMIT_PER_MIN", 120)
    mocker.patch("agentic_project_service.routes.internal_docs._RATE_LIMIT_GLOBAL_PER_MIN", 3)

    for i in range(3):
        resp = client.post(
            "/api/internal/docs/search",
            headers=_hdr(_TOKEN, f"spoofed-caller-{i}"),
            json={"query": "x"},
        )
        assert resp.status_code == 200

    # A 4th distinct caller value — per-caller bucket is fresh (count=1, well
    # under the 120 per-caller cap) — but the shared global bucket is now at 4,
    # over the cap of 3, so the global tier rejects it.
    resp = client.post(
        "/api/internal/docs/search",
        headers=_hdr(_TOKEN, "spoofed-caller-3"),
        json={"query": "x"},
    )
    assert resp.status_code == 429
    assert resp.get_json() == {"error": "rate limited"}
