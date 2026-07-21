"""Settings API — generic CRUD for project-level configuration overrides."""

import logging

from flask import Blueprint, jsonify, request
from sqlalchemy import text

from ..auth import require_auth
from ..db import AI_SCHEMA, db
from ..services.settings_registry import (
    CATEGORY_META,
    SECRET_MASK,
    SETTINGS_REGISTRY,
    get_all_settings,
    validate_setting,
)

logger = logging.getLogger(__name__)

settings_bp = Blueprint("settings", __name__, url_prefix="/api/settings")


@settings_bp.route("", methods=["GET"])
@require_auth
def get_settings():
    """Return all settings with defaults and current overrides."""
    return jsonify(get_all_settings())


@settings_bp.route("", methods=["PUT"])
@require_auth
def update_settings():
    """Bulk-update settings. Body: { "settings": { key: value, ... } }"""
    data = request.get_json(silent=True) or {}
    updates = data.get("settings", {})

    if not updates:
        return jsonify({"error": "No settings provided"}), 400

    errors: dict[str, str] = {}
    valid: dict[str, str] = {}

    for key, value in updates.items():
        # Skip secret settings sent back with the mask placeholder unchanged
        defn = SETTINGS_REGISTRY.get(key)
        if defn and defn.secret and str(value) == SECRET_MASK:
            continue
        ok, msg = validate_setting(key, value)
        if not ok:
            errors[key] = msg
        else:
            valid[key] = str(value)

    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    for key, str_value in valid.items():
        db.session.execute(
            text(
                f'INSERT INTO "{AI_SCHEMA}".project_settings (key, value) '
                f"VALUES (:key, :value) "
                f"ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, "
                f"updated_at = now()"
            ),
            {"key": key, "value": str_value},
        )
    db.session.commit()

    # Invalidate per-request cache so subsequent reads see new values
    from flask import g

    g._settings_cache = None

    return jsonify({"ok": True, "updated": list(valid.keys())})


@settings_bp.route("/<key>", methods=["DELETE"])
@require_auth
def reset_setting(key: str):
    """Remove a single override, reverting to default."""
    if key not in SETTINGS_REGISTRY:
        return jsonify({"error": f"Unknown setting: {key}"}), 404

    db.session.execute(
        text(f'DELETE FROM "{AI_SCHEMA}".project_settings WHERE key = :key'),
        {"key": key},
    )
    db.session.commit()

    from flask import g

    g._settings_cache = None

    defn = SETTINGS_REGISTRY[key]
    return jsonify({"ok": True, "key": key, "default": "" if defn.secret else defn.default})


@settings_bp.route("/reset-category", methods=["POST"])
@require_auth
def reset_category():
    """Reset all overrides in a category. Body: { "category": "copilot" }"""
    data = request.get_json(silent=True) or {}
    category = data.get("category")

    if category not in CATEGORY_META:
        return jsonify({"error": f"Unknown category: {category}"}), 400

    keys = [k for k, d in SETTINGS_REGISTRY.items() if d.category == category]
    if keys:
        keys_param = "{" + ",".join(keys) + "}"
        db.session.execute(
            text(
                f'DELETE FROM "{AI_SCHEMA}".project_settings WHERE key = ANY(CAST(:keys AS text[]))'
            ),
            {"keys": keys_param},
        )
        db.session.commit()

    from flask import g

    g._settings_cache = None

    return jsonify({"ok": True, "category": category, "reset_keys": keys})
