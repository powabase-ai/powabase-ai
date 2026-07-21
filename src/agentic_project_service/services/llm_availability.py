"""Model availability + BYOK provider enumeration. RESERVE — ships in OSS.

check_model_available answers "can this project run this model?" by looking at the
project's provider keys (DB) and the pod's platform env keys — independent of
billing. In OSS this is BYOK-only; in cloud the same check also admits AI-on-us
platform keys. It intentionally does NOT read any billing contextvar (which is CUT
behind the billing port and would be empty in the OSS build)."""

from __future__ import annotations

import logging
import os

from ..db import db
from ..models.tenant import AIProviderKey

logger = logging.getLogger(__name__)


def platform_supports(provider: str) -> bool:
    from .ai_provider_keys_resolver import _PROVIDER_ENV

    env_name = _PROVIDER_ENV.get(provider)
    if env_name is None:
        return False
    return bool(os.environ.get(env_name))


def list_byok_providers(project_id: str | None = None) -> frozenset[str]:
    """Providers with a valid key for this (single) project. Does NOT decrypt or
    self-heal. project_id is accepted for caller-readability; PS is single-project
    so it is not used in the filter."""
    rows = db.session.query(AIProviderKey.provider).filter_by(is_valid=True).all()
    return frozenset(row.provider for row in rows)


def check_model_available(model: str) -> None:
    """Raise Flask 400 when the model has neither a project provider-key nor a
    platform env key. Called at agent/orchestration/workflow run entry."""
    import litellm
    from flask import abort

    from .ai_provider_keys_resolver import _BYOK_PROVIDER_ALIAS

    try:
        _, provider, _, _ = litellm.get_llm_provider(model)
    except Exception:
        provider = model.split("/")[0]
    byok_provider = _BYOK_PROVIDER_ALIAS.get(provider, provider)
    # Mirror the before_request BYOK hook's swallow (services/billing_cloud/
    # adapter.py): a broken session (e.g. PendingRollbackError left by a prior
    # aborted handler) must degrade to "no BYOK key seen", NOT 500 the run entry.
    # Today check_model_available reads the before_request-populated contextvar and
    # never DB-queries; this rewrite DB-queries, so the try/except preserves the
    # degraded-path behavior. Log on the swallow (as the sibling hooks do) so a DB
    # blip here is visible — in the OSS build this is the only signal on this path.
    try:
        providers = list_byok_providers()
    except Exception as e:
        logger.error(
            "byok_lookup_failed in check_model_available: BYOK lookup errored; "
            "degrading to no-BYOK, so a model only a project key covers may be "
            "spuriously rejected (400) until lookup recovers. model=%s error=%r",
            model,
            e,
            exc_info=True,
        )
        providers = frozenset()
    if byok_provider in providers:
        return
    if platform_supports(byok_provider):
        return
    abort(
        400,
        description=(
            f"Model {model} requires BYOK. Add an API key for {provider} "
            f"in Settings → LLM Provider Keys."
        ),
    )
