"""AI Provider Keys — CRUD + batch PUT + validate.

Endpoints (all under /api/ai-provider-keys):
  GET    ""            — list masked keys
  POST   ""            — upsert single key
  PUT    ""            — batch upsert (null/empty = no-op)
  DELETE "/<provider>" — delete a key, 204
  POST   "/validate"   — validate without storing
"""

import logging
import os
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from ..auth import require_auth
from ..db import db
from ..models.tenant import AIProviderKey
from ..services.ai_provider_keys_resolver import (
    _PROVIDER_ENV,
    get_user_provider_keys_with_dropped,
)
from ..services.encryption import encrypt_api_key
from ..services.provider_validator import is_hard_fail, validate_provider_key

logger = logging.getLogger(__name__)

ai_provider_keys_bp = Blueprint("ai_provider_keys", __name__, url_prefix="/api/ai-provider-keys")

ALLOWED_PROVIDERS = {"openai", "anthropic", "google", "openrouter"}


def _upsert_key(provider: str, api_key: str) -> tuple[AIProviderKey, bool]:
    """Validate, encrypt, and upsert a single provider key.

    Returns (row, created) where created=True when a new row was inserted.
    Raises ValueError with a field-level message on hard-fail validation.
    Stores the row with is_valid=False / last_validated_at=NULL on soft failures.
    """
    result = validate_provider_key(provider, api_key)
    if is_hard_fail(result):
        raise ValueError(result.error or f"Provider rejected key (HTTP {result.provider_status})")

    encrypted = encrypt_api_key(api_key)
    is_valid = result.ok
    last_validated_at = datetime.now(timezone.utc) if result.ok else None

    existing = db.session.query(AIProviderKey).filter_by(provider=provider).first()
    if existing:
        existing.api_key_encrypted = encrypted
        existing.is_valid = is_valid
        existing.last_validated_at = last_validated_at
        return existing, False
    else:
        row = AIProviderKey(
            provider=provider,
            api_key_encrypted=encrypted,
            is_valid=is_valid,
            last_validated_at=last_validated_at,
        )
        db.session.add(row)
        return row, True


@ai_provider_keys_bp.route("", methods=["GET"])
@require_auth
def list_keys():
    """Return all stored provider keys (masked).

    Triggers the resolver's self-heal as a side effect first: any row whose
    ciphertext can't be decrypted with the current ``API_KEY_ENCRYPTION_KEY``
    is deleted before the response. Without this call, a user who only opens
    Settings (without trying an agent run) would see the dead slot as
    "configured" indefinitely. The return value is discarded; we re-query for
    the response so the masked display reflects post-heal state.
    """
    get_user_provider_keys_with_dropped()
    rows = db.session.query(AIProviderKey).order_by(AIProviderKey.provider).all()
    return jsonify([r.to_dict() for r in rows])


@ai_provider_keys_bp.route("/platform_supported", methods=["GET"])
@require_auth
def platform_supported():
    """Return providers AI-on-us is available for at this pod.

    Two-factor rule (credit-system v1.5): provider P is AI-on-us-available
    iff (1) P is in LiteLLM's pricing JSON AND (2) the pod env carries a
    platform key for P (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.).
    Every entry in ``_PROVIDER_ENV`` is LiteLLM-supported, so this endpoint
    only needs to check factor (2) — the env-var presence.

    The FE Settings → LLM Provider Keys page uses this to render the
    "AI-on-us active" vs "BYOK required" badges per provider row.
    """
    # Filter to providers the FE Settings page actually lets users save
    # under (``ALLOWED_PROVIDERS``). Aliases like ``gemini`` exist in
    # ``_PROVIDER_ENV`` so the resolver can read ``GEMINI_API_KEY``, but
    # the FE only stores BYOK rows under the canonical name ``google`` —
    # surfacing ``gemini`` as a separate row would mislead operators.
    # An alias env-var still counts as support for its canonical provider
    # (a pod with only GEMINI_API_KEY set should report ``google`` as
    # AI-on-us-available).
    from ..services.ai_provider_keys_resolver import _BYOK_PROVIDER_ALIAS
    canonical_supported: set[str] = set()
    for p, env in _PROVIDER_ENV.items():
        if not os.environ.get(env):
            continue
        canonical = _BYOK_PROVIDER_ALIAS.get(p, p)
        if canonical in ALLOWED_PROVIDERS:
            canonical_supported.add(canonical)
    return jsonify({"providers": sorted(canonical_supported)})


@ai_provider_keys_bp.route("", methods=["POST"])
@require_auth
def upsert_key():
    """Upsert a single provider key. Body: {provider, api_key}."""
    data = request.get_json(silent=True) or {}
    provider = data.get("provider", "").strip().lower()
    api_key = data.get("api_key", "").strip()

    if not provider or not api_key:
        return jsonify({"error": "provider and api_key are required"}), 400
    if provider not in ALLOWED_PROVIDERS:
        return jsonify(
            {"error": f"Unknown provider: {provider}. Allowed: {sorted(ALLOWED_PROVIDERS)}"}
        ), 400

    try:
        row, created = _upsert_key(provider, api_key)
        db.session.commit()
    except ValueError as exc:
        db.session.rollback()
        return jsonify({"error": "Validation failed", "fields": {provider: str(exc)}}), 400

    status = 201 if created else 200
    return jsonify(row.to_dict()), status


@ai_provider_keys_bp.route("", methods=["PUT"])
@require_auth
def batch_upsert_keys():
    """Batch upsert. Body: {openai?, anthropic?, google?, openrouter?}.

    Null/empty values are no-ops. On any hard-fail, the entire batch is
    rolled back and a 400 with field-level errors is returned.
    """
    data = request.get_json(silent=True) or {}

    updates = {
        p: v
        for p, v in data.items()
        if p in ALLOWED_PROVIDERS and v  # skip null/empty
    }

    unknown = [p for p in data if p not in ALLOWED_PROVIDERS]
    if unknown:
        return jsonify({"error": f"Unknown providers: {unknown}"}), 400

    if not updates:
        # Nothing to do — return current state
        rows = db.session.query(AIProviderKey).order_by(AIProviderKey.provider).all()
        return jsonify([r.to_dict() for r in rows])

    results = []
    try:
        for provider, api_key in updates.items():
            row, _ = _upsert_key(provider, api_key)
            results.append(row)
        db.session.commit()
    except ValueError as exc:
        db.session.rollback()
        # Re-identify which provider caused the failure via re-validation
        # (exc message already contains the reason; provider is in scope from loop)
        return jsonify({"error": "Validation failed", "fields": {provider: str(exc)}}), 400

    return jsonify([r.to_dict() for r in results])


@ai_provider_keys_bp.route("/<provider>", methods=["DELETE"])
@require_auth
def delete_key(provider: str):
    """Delete a provider key. Returns 204."""
    provider = provider.strip().lower()
    if provider not in ALLOWED_PROVIDERS:
        return jsonify({"error": f"Unknown provider: {provider}"}), 400

    db.session.query(AIProviderKey).filter_by(provider=provider).delete()
    db.session.commit()
    return "", 204


@ai_provider_keys_bp.route("/validate", methods=["POST"])
@require_auth
def validate_key():
    """Validate a key without storing it. Body: {provider, api_key}."""
    data = request.get_json(silent=True) or {}
    provider = data.get("provider", "").strip().lower()
    api_key = data.get("api_key", "").strip()

    if not provider or not api_key:
        return jsonify({"error": "provider and api_key are required"}), 400
    if provider not in ALLOWED_PROVIDERS:
        return jsonify({"error": f"Unknown provider: {provider}"}), 400

    result = validate_provider_key(provider, api_key)
    payload: dict = {"is_valid": result.ok}
    if result.error:
        payload["error"] = result.error
    return jsonify(payload)
