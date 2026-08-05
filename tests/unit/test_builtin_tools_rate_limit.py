"""web_scrape / web_search rate-limit behavior."""

import json
from unittest.mock import MagicMock

import pytest

import agentic_project_service.tools.builtin as builtin_mod


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")
    monkeypatch.setenv("EXA_API_KEY", "fake-key")
    monkeypatch.setattr(
        builtin_mod,
        "get_setting",
        lambda k: {
            "FIRECRAWL_API_BASE": "https://example.invalid/v1",
            "FIRECRAWL_RATE_LIMIT_PER_MINUTE": 30,
            "EXA_RATE_LIMIT_PER_MINUTE": 60,
            "WEB_SCRAPE_MAX_CHARS": 200000,
        }.get(k),
    )


def _resp(status, headers=None, payload=None):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = payload or {"data": {"markdown": "hi"}}
    if status >= 400:
        r.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        r.raise_for_status.return_value = None
    return r


def test_web_scrape_denied_by_limiter_returns_platform_error(monkeypatch):
    monkeypatch.setattr(builtin_mod.external_limiter, "acquire_blocking", lambda *a, **kw: False)
    result = json.loads(builtin_mod.web_scrape_handler({"url": "https://example.com"}, {}))
    assert result["_platform_error"] is True
    assert "rate limited" in result["error"].lower()


def test_web_scrape_retries_once_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(builtin_mod.external_limiter, "acquire_blocking", lambda *a, **kw: True)
    sleeps = []
    monkeypatch.setattr(builtin_mod.time, "sleep", lambda s: sleeps.append(s))
    post = MagicMock(side_effect=[_resp(429, headers={"Retry-After": "3"}), _resp(200)])
    monkeypatch.setattr(builtin_mod.http_requests, "post", post)
    result = json.loads(builtin_mod.web_scrape_handler({"url": "https://example.com"}, {}))
    assert "error" not in result
    assert post.call_count == 2
    assert sleeps == [3.0]


def test_web_scrape_second_429_returns_platform_error(monkeypatch):
    monkeypatch.setattr(builtin_mod.external_limiter, "acquire_blocking", lambda *a, **kw: True)
    monkeypatch.setattr(builtin_mod.time, "sleep", lambda s: None)
    post = MagicMock(
        side_effect=[
            _resp(429, headers={"Retry-After": "3"}),
            _resp(429, headers={"Retry-After": "3"}),
        ]
    )
    monkeypatch.setattr(builtin_mod.http_requests, "post", post)
    result = json.loads(builtin_mod.web_scrape_handler({"url": "https://example.com"}, {}))
    assert result["_platform_error"] is True
    assert post.call_count == 2


def test_web_scrape_long_retry_after_gives_up_immediately(monkeypatch):
    monkeypatch.setattr(builtin_mod.external_limiter, "acquire_blocking", lambda *a, **kw: True)
    post = MagicMock(return_value=_resp(429, headers={"Retry-After": "120"}))
    monkeypatch.setattr(builtin_mod.http_requests, "post", post)
    result = json.loads(builtin_mod.web_scrape_handler({"url": "https://example.com"}, {}))
    assert result["_platform_error"] is True
    assert post.call_count == 1


def _exa_resp(status, headers=None):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = {"results": []}
    if status >= 400:
        r.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        r.raise_for_status.return_value = None
    return r


def test_web_search_denied_by_limiter_returns_platform_error(monkeypatch):
    monkeypatch.setattr(
        builtin_mod.external_limiter, "acquire_blocking", lambda *a, **kw: False
    )
    result = json.loads(builtin_mod.web_search_handler({"query": "q"}, {}))
    assert result["_platform_error"] is True
    assert "rate limited" in result["error"].lower()


def test_web_search_retries_once_on_429(monkeypatch):
    monkeypatch.setattr(
        builtin_mod.external_limiter, "acquire_blocking", lambda *a, **kw: True
    )
    monkeypatch.setattr(builtin_mod.time, "sleep", lambda s: None)
    post = MagicMock(
        side_effect=[_exa_resp(429, headers={"Retry-After": "2"}), _exa_resp(200)]
    )
    monkeypatch.setattr(builtin_mod.http_requests, "post", post)
    result = json.loads(builtin_mod.web_search_handler({"query": "q"}, {}))
    # web_search_handler's success path returns a bare JSON array of results
    # (json.dumps(results, ...)), not an object — so the top-level shape here
    # is a list, unlike web_scrape_handler's dict.
    assert result == []
    assert post.call_count == 2


def test_web_search_second_429_returns_platform_error(monkeypatch):
    monkeypatch.setattr(
        builtin_mod.external_limiter, "acquire_blocking", lambda *a, **kw: True
    )
    monkeypatch.setattr(builtin_mod.time, "sleep", lambda s: None)
    post = MagicMock(return_value=_exa_resp(429, headers={"Retry-After": "2"}))
    monkeypatch.setattr(builtin_mod.http_requests, "post", post)
    result = json.loads(builtin_mod.web_search_handler({"query": "q"}, {}))
    assert result["_platform_error"] is True
    assert post.call_count == 2
