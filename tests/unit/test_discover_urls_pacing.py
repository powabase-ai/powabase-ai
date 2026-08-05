"""_discover_urls_crawl skips Firecrawl /map when the limiter denies."""

from unittest.mock import MagicMock

import agentic_project_service.routes.sources as sources_mod


def _settings(key):
    return {
        "FIRECRAWL_API_BASE": "https://example.invalid/v1",
        "FIRECRAWL_RATE_LIMIT_PER_MINUTE": 30,
    }[key]


def test_denied_limiter_skips_map_and_uses_fallback(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")
    monkeypatch.setattr(sources_mod, "get_setting", _settings)
    monkeypatch.setattr(sources_mod.external_limiter, "acquire_blocking", lambda *a, **kw: False)

    calls = []

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **kw):
            calls.append(("post", url))
            raise AssertionError("must not call Firecrawl /map when denied")

        def get(self, url, **kw):
            calls.append(("get", url))
            r = MagicMock()
            r.raise_for_status.return_value = None
            r.text = '<a href="https://example.com/page1">x</a>'
            return r

    monkeypatch.setattr(sources_mod.httpx, "Client", FakeClient)
    urls = sources_mod._discover_urls_crawl("https://example.com", 10)
    assert ("get", "https://example.com") in calls
    assert all(kind != "post" for kind, _ in calls)
    assert "https://example.com/page1" in urls


def test_allowed_limiter_calls_map(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")
    monkeypatch.setattr(sources_mod, "get_setting", _settings)
    monkeypatch.setattr(sources_mod.external_limiter, "acquire_blocking", lambda *a, **kw: True)

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **kw):
            r = MagicMock()
            r.raise_for_status.return_value = None
            r.json.return_value = {"links": ["https://example.com/a"]}
            return r

    monkeypatch.setattr(sources_mod.httpx, "Client", FakeClient)
    urls = sources_mod._discover_urls_crawl("https://example.com", 10)
    assert urls == ["https://example.com/a"]
