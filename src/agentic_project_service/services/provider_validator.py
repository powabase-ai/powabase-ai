"""Provider key validator with structured return so callers can
distinguish 401/403 (user error → block) from 5xx/network (tolerate).

Diverges from the control-plane copy: the Mistral branch in
``_endpoint_for`` has been removed. Project-service only handles the
four user-facing providers (openai, anthropic, google, openrouter).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

HARD_FAIL_STATUSES = {401, 403}


@dataclass
class ValidationResult:
    ok: bool
    provider_status: int | None
    error: str | None


def is_hard_fail(result: ValidationResult) -> bool:
    """True when the upstream provider explicitly rejected the key."""
    return not result.ok and result.provider_status in HARD_FAIL_STATUSES


def validate_provider_key(provider: str, api_key: str) -> ValidationResult:
    """Call the provider's lightweight auth/list endpoint to verify the key."""
    endpoint, headers = _endpoint_for(provider, api_key)
    if endpoint is None:
        return ValidationResult(ok=True, provider_status=None, error=None)

    try:
        resp = requests.get(endpoint, headers=headers, timeout=10)
    except requests.RequestException as exc:
        logger.warning("Provider validation network error (%s): %s", provider, exc)
        return ValidationResult(ok=False, provider_status=None, error=str(exc))

    if resp.status_code == 200:
        return ValidationResult(ok=True, provider_status=200, error=None)
    return ValidationResult(
        ok=False,
        provider_status=resp.status_code,
        error=f"Provider returned {resp.status_code}",
    )


def _endpoint_for(provider: str, api_key: str) -> tuple[str | None, dict[str, str]]:
    if provider == "openai":
        return "https://api.openai.com/v1/models", {"Authorization": f"Bearer {api_key}"}
    if provider == "anthropic":
        return "https://api.anthropic.com/v1/models", {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    if provider == "google":
        return f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}", {}
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1/auth/key", {"Authorization": f"Bearer {api_key}"}
    # Unknown providers — no validator, treat as OK
    return None, {}
