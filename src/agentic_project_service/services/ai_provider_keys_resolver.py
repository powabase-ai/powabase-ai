"""User-supplied LLM provider keys — DB-only (steady state, PR2+)."""

from __future__ import annotations

import logging
import os

import litellm

from ..db import db
from ..models.tenant import AIProviderKey
from .encryption import decrypt_api_key

logger = logging.getLogger(__name__)

_PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    # litellm.get_llm_provider("gemini/...") returns provider="gemini", not
    # "google". Map it explicitly so both BYOK lookups (the "google" row
    # stored by the FE) and env fallback resolve correctly.
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Some providers' BYOK row uses a different name than litellm's provider
# string. Normalize before looking up in provider_keys.
#
# Two provider names for Google: ``litellm.get_llm_provider("gemini/X")``
# returns ``"gemini"`` (the AI Studio path), but at BillingLogger callback
# time litellm has stripped the prefix from kwargs["model"] and the same
# function returns ``"vertex_ai"`` for bare ``X``. Both must alias to
# ``"google"`` so the FE BYOK row name matches in either context.
_BYOK_PROVIDER_ALIAS = {
    "gemini": "google",
    "vertex_ai": "google",
}


def get_all_user_provider_keys() -> dict[str, str]:
    """Return user-supplied provider keys for the current project.

    .. warning::
        Has a DB write side effect. Self-heals by ``DELETE``-ing and
        ``commit``-ing rows whose ciphertext cannot be decrypted with the
        current ``API_KEY_ENCRYPTION_KEY``. Callers in the middle of an
        open transaction with pending writes (e.g., ``db.session.add(...)``
        without a commit) will see those pending writes flushed by this
        call's commit. See :func:`get_user_provider_keys_with_dropped` for
        the full rationale.
    """
    keys, _ = get_user_provider_keys_with_dropped()
    return keys


def get_user_provider_keys_with_dropped() -> tuple[dict[str, str], list[str]]:
    """Return ``(provider→plaintext keys, providers whose rows were dropped)``.

    Self-heal rationale: a Fernet decryption failure means the ciphertext can
    never be recovered with the current encryption key (the original key is
    gone). Leaving the row in place silently fails every subsequent agent run
    for that provider AND keeps the FE/Settings UI showing the slot as
    "configured" while the credential is dead. Deleting the row surfaces the
    truth: the slot is empty and the user can re-add the key.

    Catch is narrowed to ``ValueError`` deliberately. ``decrypt_api_key`` only
    raises ``ValueError`` on a per-row InvalidToken; anything else (e.g.,
    ``RuntimeError`` from a misconfigured ``API_KEY_ENCRYPTION_KEY``, or a
    SQLAlchemy error) is an environment problem, not a per-row problem — we
    must propagate it instead of treating every row as dead and wiping the
    table.

    ``dropped``'s truth source is **decryptability**, not deletion success
    (issue #246, review round 4). If the conditional DELETE itself fails
    (DB blip, lock conflict), the user still gets the actionable
    ``ProviderKeyDecryptDropped`` error — otherwise the resolver silently
    restores the original "Missing API Key" failure mode. We only REMOVE a
    provider from ``dropped`` when ``rowcount == 0`` (concurrent re-add
    detected — the row's ciphertext changed under us and is now valid).
    """
    out: dict[str, str] = {}
    decrypt_failed: list[str] = []
    bad_rows: list[AIProviderKey] = []

    for row in AIProviderKey.query.filter_by(is_valid=True).all():
        try:
            out[row.provider] = decrypt_api_key(row.api_key_encrypted)
        except ValueError as exc:
            logger.error(
                "Decrypt failed for stored %s key — will attempt to delete row "
                "so the user can re-add",
                row.provider,
                exc_info=exc,
            )
            decrypt_failed.append(row.provider)
            bad_rows.append(row)

    # Initialize dropped from decrypt_failed (truth source). Only remove
    # entries for the concurrent-re-add case below.
    dropped: list[str] = list(decrypt_failed)

    for row in bad_rows:
        provider = row.provider
        try:
            # SAVEPOINT isolates this delete from any pending writes in the
            # caller's outer transaction — a transient DB failure here only
            # undoes the SAVEPOINT, not the caller's work.
            #
            # Conditional delete keyed on (id, api_key_encrypted): if the row's
            # ciphertext changed between our read above and now (e.g., the user
            # re-added the key in another tab between agent retries), the
            # rowcount is 0 and we leave the freshly-good row alone — AND
            # remove the provider from ``dropped`` so we don't surface a
            # false-positive error to the user.
            with db.session.begin_nested():
                rowcount = (
                    db.session.query(AIProviderKey)
                    .filter_by(id=row.id, api_key_encrypted=row.api_key_encrypted)
                    .delete()
                )
            if rowcount == 0:
                logger.warning(
                    "Skipping delete of %s row %s — ciphertext changed since "
                    "read (likely concurrent re-add). Provider removed from "
                    "dropped — row is now valid.",
                    provider,
                    row.id,
                )
                try:
                    dropped.remove(provider)
                except ValueError:
                    pass
        except Exception as exc:
            # SAVEPOINT auto-rolled-back on context exit; outer txn intact.
            # Provider stays in ``dropped`` — user still sees the actionable
            # error on the next agent run, and ops will see this log line.
            logger.error(
                "Failed to delete undecryptable %s row %s: %s. User-facing "
                "error will still surface; operator must investigate the DB "
                "issue separately.",
                provider,
                row.id,
                exc,
            )

    return out, dropped


def resolve_api_key_for_model(model: str, provider_keys: dict[str, str]) -> str | None:
    """Resolve the API key for an LLM model. Used at LiteLLM call time.

    Looks up `provider_keys` first; falls back to pod env var (operator default).
    Returns None if neither has a key — LiteLLM will surface the error.
    """
    try:
        _, provider, _, _ = litellm.get_llm_provider(model)
    except Exception:
        return None
    byok_key = _BYOK_PROVIDER_ALIAS.get(provider, provider)
    return provider_keys.get(byok_key) or os.environ.get(
        _PROVIDER_ENV.get(provider, ""), ""
    ) or None


class ProviderKeyDecryptDropped(Exception):
    """A stored provider key was just dropped because it could not be decrypted.

    Routes that resolve a key for a single model (agent runs, orchestrations)
    catch this and surface a clear, actionable error to the user — instead of
    letting LiteLLM emit the generic "Missing API Key" message.
    """

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(
            f"Stored {provider} API key could not be decrypted "
            f"(encryption key changed since save). It has been removed; "
            f"please re-add it in Settings → LLM Provider Keys."
        )


def resolve_api_key_or_raise_for_drop_using(
    model: str,
    provider_keys: dict[str, str],
    dropped: list[str],
) -> str | None:
    """Like :func:`resolve_api_key_or_raise_for_drop` but accepts pre-computed
    ``(provider_keys, dropped)`` instead of querying the DB.

    Use when resolving multiple models in one operation (e.g., orchestration
    with N sub-agents): call :func:`get_user_provider_keys_with_dropped` once,
    then this helper per model — avoids re-querying and re-attempting the
    self-heal for every model in the loop.
    """
    api_key = resolve_api_key_for_model(model, provider_keys)
    if api_key is None and dropped:
        try:
            _, provider, _, _ = litellm.get_llm_provider(model)
        except Exception:
            provider = None
        if provider in dropped:
            raise ProviderKeyDecryptDropped(provider)
    return api_key


def resolve_api_key_or_raise_for_drop(model: str) -> str | None:
    """Resolve the provider API key for ``model``; raise on a recent decrypt drop.

    Convenience wrapper for the agent-run path: combines
    :func:`get_user_provider_keys_with_dropped` and
    :func:`resolve_api_key_for_model`, and raises
    :class:`ProviderKeyDecryptDropped` when the model's provider was just
    dropped because its row could not be decrypted. Returns ``None`` when no
    key was ever configured (LiteLLM will then surface the standard "Missing
    API Key" error downstream — same as before this change).
    """
    provider_keys, dropped = get_user_provider_keys_with_dropped()
    return resolve_api_key_or_raise_for_drop_using(model, provider_keys, dropped)
