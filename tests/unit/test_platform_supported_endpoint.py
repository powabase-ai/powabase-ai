"""Tests for GET /api/ai-provider-keys/platform_supported.

Two-factor AI-on-us rule per credit-system v1.5 spec:
provider P is AI-on-us-available iff
  (1) P is in LiteLLM's pricing JSON (charging works), AND
  (2) PS pod env carries a platform key for P.

This endpoint surfaces factor (2). Factor (1) is implicit in the
``_PROVIDER_ENV`` allowlist (every entry is a LiteLLM-supported provider).

Uses the minimal-Flask-app + mocked-auth pattern from
``test_bm25_status_field.py`` so the test runs in the unit/ directory
(parent conftest's DB setup is bypassed by ``tests/unit/conftest.py``).
"""

from __future__ import annotations

from unittest.mock import patch

from agentic_project_service.routes import ai_provider_keys as akeys_route

_FAKE_JWT = "fake.jwt.token"

_AUTH_PATCH = patch(
    "agentic_project_service.auth.decode_jwt",
    return_value={"role": "service_role"},
)


def _make_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(akeys_route.ai_provider_keys_bp)
    return app


def _auth_headers():
    return {"Authorization": f"Bearer {_FAKE_JWT}"}


BASE = "/api/ai-provider-keys/platform_supported"


@_AUTH_PATCH
def test_endpoint_returns_providers_for_each_set_env(_jwt, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with _make_test_app().test_client() as client:
        resp = client.get(BASE, headers=_auth_headers())
    assert resp.status_code == 200
    providers = set(resp.get_json()["providers"])
    assert providers == {"openai", "anthropic"}


@_AUTH_PATCH
def test_endpoint_returns_empty_when_no_env_keys(_jwt, monkeypatch):
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    with _make_test_app().test_client() as client:
        resp = client.get(BASE, headers=_auth_headers())
    assert resp.status_code == 200
    assert resp.get_json()["providers"] == []
