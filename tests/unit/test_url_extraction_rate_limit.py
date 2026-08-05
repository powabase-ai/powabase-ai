# tests/unit/test_url_extraction_rate_limit.py
"""Rate-limit behavior of extract_url_source: pacing, 429 re-queue,
permanent-vs-transient classification, and terminal error codes."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from celery.exceptions import MaxRetriesExceededError, Retry

import agentic_project_service.tasks.url_extraction as url_mod


def _make_source():
    return {
        "id": "src-1",
        "name": "x",
        "file_type": "text/html",
        "storage_path": "x",
        "extraction_status": "pending",
        "derivatives": {},
        "metadata": {},
        "auto_metadata": {},
    }


@pytest.fixture
def status_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(url_mod, "get_source", lambda _: _make_source())
    monkeypatch.setattr(
        url_mod,
        "update_source_status",
        lambda *a, **kw: calls.append((a, kw)),
    )
    monkeypatch.setattr(
        url_mod, "update_source_extraction_result", lambda *a, **kw: None
    )
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")
    monkeypatch.setattr(
        url_mod,
        "get_setting",
        lambda k: {
            "FIRECRAWL_API_BASE": "https://example.invalid/v1",
            "FIRECRAWL_RATE_LIMIT_PER_MINUTE": 30,
            "URL_IMPORT_MAX_IMAGES_PER_PAGE": 0,
            "URL_IMPORT_IMAGE_MAX_SIZE_MB": 1,
        }[k],
    )
    return calls


def _fake_httpx_client(monkeypatch, status_code, headers=None):
    """Firecrawl POST returns the given status; raise_for_status mimics httpx."""

    class R:
        def __init__(self):
            self.status_code = status_code
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"{self.status_code}",
                    request=httpx.Request("POST", "https://example.invalid/v1/scrape"),
                    response=SimpleNamespace(
                        status_code=self.status_code, headers=self.headers
                    ),
                )

        def json(self):
            return {"data": {"markdown": "hi", "html": "<p>hi</p>", "metadata": {}}}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            return R()

        def get(self, *a, **kw):
            return R()

    monkeypatch.setattr(url_mod.httpx, "Client", FakeClient)


@pytest.fixture
def retry_spy(monkeypatch):
    """Capture self.retry calls; raises Retry like the real bound method."""
    spy = MagicMock(side_effect=Retry("retry"))
    monkeypatch.setattr(url_mod.extract_url_source, "retry", spy)
    return spy


def test_pacing_denial_requeues(status_calls, retry_spy, monkeypatch):
    _fake_httpx_client(monkeypatch, 200)
    monkeypatch.setattr(
        url_mod.external_limiter, "try_acquire", lambda *a, **kw: 12.5
    )
    with pytest.raises(Retry):
        url_mod.extract_url_source.run("src-1", "bucket-1", "https://example.com")
    kwargs = retry_spy.call_args.kwargs
    assert 12.5 <= kwargs["countdown"] <= 17.5  # wait + jitter(0-5)
    assert kwargs["max_retries"] == url_mod.RATE_LIMIT_MAX_RETRIES
    # No terminal failure was written.
    assert not [c for c in status_calls if "failed" in c[0]]


def test_429_requeues_with_retry_after(status_calls, retry_spy, monkeypatch):
    _fake_httpx_client(monkeypatch, 429, headers={"Retry-After": "42"})
    monkeypatch.setattr(
        url_mod.external_limiter, "try_acquire", lambda *a, **kw: None
    )
    with pytest.raises(Retry):
        url_mod.extract_url_source.run("src-1", "bucket-1", "https://example.com")
    kwargs = retry_spy.call_args.kwargs
    assert 42.0 <= kwargs["countdown"] <= 47.0
    assert kwargs["max_retries"] == url_mod.RATE_LIMIT_MAX_RETRIES


def test_429_budget_exhausted_fails_with_rate_limited(
    status_calls, monkeypatch
):
    _fake_httpx_client(monkeypatch, 429, headers={"Retry-After": "42"})
    monkeypatch.setattr(
        url_mod.external_limiter, "try_acquire", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        url_mod.extract_url_source,
        "retry",
        MagicMock(side_effect=MaxRetriesExceededError()),
    )
    result = url_mod.extract_url_source.run(
        "src-1", "bucket-1", "https://example.com"
    )
    assert result["status"] == "error"
    terminal = [c for c in status_calls if "failed" in c[0]]
    assert terminal and terminal[-1][1].get("error_code") == "rate_limited"


def test_permanent_4xx_fails_immediately(status_calls, retry_spy, monkeypatch):
    _fake_httpx_client(monkeypatch, 404)
    monkeypatch.setattr(
        url_mod.external_limiter, "try_acquire", lambda *a, **kw: None
    )
    result = url_mod.extract_url_source.run(
        "src-1", "bucket-1", "https://example.com"
    )
    assert result["status"] == "error"
    retry_spy.assert_not_called()
    terminal = [c for c in status_calls if "failed" in c[0]]
    assert terminal and terminal[-1][1].get("error_code") == "permanent"


def test_5xx_retries_then_transient(status_calls, monkeypatch):
    _fake_httpx_client(monkeypatch, 503)
    monkeypatch.setattr(
        url_mod.external_limiter, "try_acquire", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        url_mod.extract_url_source,
        "retry",
        MagicMock(side_effect=MaxRetriesExceededError()),
    )
    result = url_mod.extract_url_source.run(
        "src-1", "bucket-1", "https://example.com"
    )
    assert result["status"] == "error"
    terminal = [c for c in status_calls if "failed" in c[0]]
    assert terminal and terminal[-1][1].get("error_code") == "transient"


def test_success_path_unaffected(status_calls, retry_spy, monkeypatch, mock_db_session):
    _fake_httpx_client(monkeypatch, 200)
    monkeypatch.setattr(
        url_mod.external_limiter, "try_acquire", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        url_mod,
        "get_storage",
        lambda: SimpleNamespace(
            ensure_bucket=lambda *a: None, upload=lambda **kw: "sources/x"
        ),
    )
    monkeypatch.setattr(url_mod, "get_derivative_storage_path", lambda *a, **kw: "x")
    monkeypatch.setattr(url_mod, "get_source_storage_path", lambda *a, **kw: "x")
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    result = url_mod.extract_url_source.run(
        "src-1", "bucket-1", "https://example.com"
    )
    assert result["status"] == "success"
