"""EXA_API_KEY and FIRECRAWL_API_KEY are platform-paid, env-injected, NOT
per-project settings. These tests pin that contract end-to-end:

- The settings registry must NOT expose either key as a tenant-managed setting
  (it would render a Studio Settings > Tools field and create the illusion of
  tenant control).
- web_search_handler reads EXA_API_KEY from os.environ directly.
- web_scrape_handler reads FIRECRAWL_API_KEY from os.environ directly.
- When either env is empty, the error message must NOT direct the tenant at
  Studio Settings > Tools — the misconfiguration is operator-side.
"""

import json
from unittest.mock import MagicMock, patch


from agentic_project_service.tools.builtin import (
    code_execute_handler,
    web_scrape_handler,
    web_search_handler,
)


class TestSettingsRegistryDoesNotExposePlatformKeys:
    """The registry drives the Studio Settings UI. Platform-paid secrets must
    not appear there at all — they are wired through env, not per-project DB.
    """

    def test_exa_api_key_not_in_registry(self):
        from agentic_project_service.services.settings_registry import (
            SETTINGS_REGISTRY,
        )

        assert "EXA_API_KEY" not in SETTINGS_REGISTRY

    def test_firecrawl_api_key_not_in_registry(self):
        from agentic_project_service.services.settings_registry import (
            SETTINGS_REGISTRY,
        )

        assert "FIRECRAWL_API_KEY" not in SETTINGS_REGISTRY

    def test_get_all_settings_does_not_surface_exa(self):
        from agentic_project_service.services.settings_registry import (
            get_all_settings,
        )

        result = get_all_settings()
        all_keys = [s["key"] for cat in result["categories"].values() for s in cat["settings"]]
        assert "EXA_API_KEY" not in all_keys

    def test_get_all_settings_does_not_surface_firecrawl(self):
        from agentic_project_service.services.settings_registry import (
            get_all_settings,
        )

        result = get_all_settings()
        all_keys = [s["key"] for cat in result["categories"].values() for s in cat["settings"]]
        assert "FIRECRAWL_API_KEY" not in all_keys


class TestWebSearchHandlerReadsExaKeyFromEnv:
    """web_search_handler must read EXA_API_KEY from os.environ, not from
    the per-project settings DB. The key is provisioned by the platform via
    AWS SM -> ExternalSecret -> pod env, identical to MISTRAL_API_KEY etc.
    """

    def test_env_key_flows_into_exa_request_header(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "platform-exa-key-xyz")
        with patch("agentic_project_service.tools.builtin.http_requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"results": []}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            web_search_handler({"query": "anything"}, context=None)

            mock_post.assert_called_once()
            _args, kwargs = mock_post.call_args
            assert kwargs["headers"]["x-api-key"] == "platform-exa-key-xyz"

    def test_missing_env_returns_platform_misconfiguration_error(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        result = json.loads(web_search_handler({"query": "anything"}, context=None))
        assert "error" in result
        # The misconfiguration is operator-side; the tenant cannot fix it
        # via Studio Settings.
        assert "Settings > Tools" not in result["error"]
        assert "Settings &gt; Tools" not in result["error"]

    def test_missing_env_does_not_call_exa(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        with patch("agentic_project_service.tools.builtin.http_requests.post") as mock_post:
            web_search_handler({"query": "anything"}, context=None)
            mock_post.assert_not_called()


class TestWebScrapeHandlerReadsFirecrawlKeyFromEnv:
    """web_scrape_handler must read FIRECRAWL_API_KEY from os.environ. Same
    rationale as Exa: Firecrawl is a platform-paid service per the credit
    pricing catalog.
    """

    def test_env_key_flows_into_firecrawl_request_header(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "platform-firecrawl-key-abc")
        # FIRECRAWL_API_BASE remains a non-secret setting; mock its read.
        with (
            patch("agentic_project_service.tools.builtin.get_setting") as mock_get_setting,
            patch("agentic_project_service.tools.builtin.http_requests.post") as mock_post,
        ):
            mock_get_setting.side_effect = lambda k: {
                "WEB_SCRAPE_MAX_CHARS": 200000,
                "FIRECRAWL_API_BASE": "https://api.firecrawl.dev/v1",
            }.get(k)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"data": {"markdown": ""}}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            web_scrape_handler(
                {"url": "https://example.com", "formats": ["markdown"]},
                context=None,
            )

            mock_post.assert_called_once()
            _args, kwargs = mock_post.call_args
            assert kwargs["headers"]["Authorization"] == "Bearer platform-firecrawl-key-abc"

    def test_missing_env_returns_platform_misconfiguration_error(self, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        # Avoid the direct-image-URL short-circuit by using a non-image URL.
        result = json.loads(
            web_scrape_handler(
                {"url": "https://example.com/page", "formats": ["markdown"]},
                context=None,
            )
        )
        assert "error" in result
        assert "Settings > Tools" not in result["error"]
        assert "Settings &gt; Tools" not in result["error"]

    def test_missing_env_does_not_call_firecrawl(self, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        with patch("agentic_project_service.tools.builtin.http_requests.post") as mock_post:
            web_scrape_handler(
                {"url": "https://example.com/page", "formats": ["markdown"]},
                context=None,
            )
            mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Handler-level marker emission contract.
# ---------------------------------------------------------------------------
#
# The wrapper-side test in test_tool_registry_billing.py asserts the wrapper
# correctly RESPONDS to the `_platform_error: true` marker (skip billing,
# strip key). But that test uses its own mock handler — if a real handler
# silently dropped the marker from its return JSON, the wrapper test would
# still pass and the tenant would get debited on every platform misconfig.
#
# These tests pin the marker EMISSION from each handler that participates
# in the platform-error billing contract.


class TestPlatformErrorMarkerEmission:
    """Each handler that returns a platform-misconfig error MUST emit
    `_platform_error: true` in its JSON. Without it, the billing wrapper
    has no signal to skip post_charge and the tenant is debited for the
    platform's own misconfiguration.

    Counterfactual: drop the marker from any handler's error return →
    the corresponding test below fails.
    """

    def test_web_search_handler_emits_platform_error_marker(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        result = json.loads(web_search_handler({"query": "anything"}, context=None))
        assert result.get("_platform_error") is True

    def test_web_scrape_handler_emits_platform_error_marker(self, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        result = json.loads(
            web_scrape_handler(
                {"url": "https://example.com/page", "formats": ["markdown"]},
                context=None,
            )
        )
        assert result.get("_platform_error") is True

    def test_code_execute_handler_emits_platform_error_marker(self, monkeypatch):
        # CODE_SANDBOX_URL has zero infra wiring across the fleet today —
        # missing-env is the default state, not an edge case. The marker
        # prevents per-call debits while the operator wires the sandbox.
        monkeypatch.delenv("CODE_SANDBOX_URL", raising=False)
        result = json.loads(
            code_execute_handler({"language": "python", "code": "print(1)"}, context=None)
        )
        assert result.get("_platform_error") is True
