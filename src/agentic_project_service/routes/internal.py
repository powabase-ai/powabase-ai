"""Internal (service-role-only) endpoints called by the control plane.

Currently exposes GET /api/internal/mau-count, which the CP hourly MAU drain
(control-plane services/drain/mau.py::fetch_ps_mau_count) polls per project to
read the trailing-30d MAU counts it bills against the compute-tier bundle.
"""

from flask import Blueprint, g, jsonify
from sqlalchemy import text

from ..auth import require_auth
from ..db import db

internal_bp = Blueprint("internal", __name__, url_prefix="/api/internal")

# Trailing-30d MAU queries, run against THIS project's local `auth` schema.
#
# ⚠ REVENUE-DETERMINING — these are billed against the compute-tier bundle, and
# the SSO / third-party definitions are a DEFENSIBLE DEFAULT that the billing
# owner must confirm. The spec is internally inconsistent on what "third-party"
# means: §1.2 says auth.identities + auth.providers; §3.7 says introspect
# auth.audit_log_entries actions. We implement the is_sso_user / OAuth-
# auth.identities interpretation because audit_log_entries are fragile (rows get
# pruned), so an audit-log count would silently undercount. `regular_mau` is
# unambiguous. The three query strings are isolated as constants so the owner
# can adjust the SSO / third-party definitions in one place after confirming.

# Unambiguous: any user who signed in within the trailing 30 days.
_REGULAR_MAU_SQL = text(
    "SELECT count(*) FROM auth.users WHERE last_sign_in_at >= now() - INTERVAL '30 days'"
)

# DEFAULT (owner-confirm): active users flagged is_sso_user (GoTrue's SAML/SSO
# boolean on auth.users).
_SSO_MAU_SQL = text(
    "SELECT count(*) FROM auth.users "
    "WHERE last_sign_in_at >= now() - INTERVAL '30 days' "
    "AND is_sso_user = true"
)

# DEFAULT (owner-confirm): active non-SSO users who have a third-party OAuth
# identity (google/github/etc.), excluding first-party email/phone identities
# and SSO users.
_THIRD_PARTY_MAU_SQL = text(
    "SELECT count(DISTINCT u.id) FROM auth.users u "
    "JOIN auth.identities i ON i.user_id = u.id "
    "WHERE u.last_sign_in_at >= now() - INTERVAL '30 days' "
    "AND u.is_sso_user = false "
    "AND i.provider NOT IN ('email', 'phone')"
)


@internal_bp.route("/mau-count", methods=["GET"])
@require_auth
def mau_count():
    """Return this project's trailing-30d MAU / SSO-MAU / third-party-MAU counts.

    Service-role only: the CP drain authenticates with the per-project
    service_role_key, which the PS auth decodes and marks is_service_role.
    """
    if (getattr(g, "jwt_payload", None) or {}).get("is_service_role") is not True:
        return jsonify({"error": "Service role required"}), 403

    regular = db.session.execute(_REGULAR_MAU_SQL).scalar() or 0
    sso = db.session.execute(_SSO_MAU_SQL).scalar() or 0
    third_party = db.session.execute(_THIRD_PARTY_MAU_SQL).scalar() or 0

    return jsonify(
        {
            "regular_mau": int(regular),
            "sso_mau": int(sso),
            "third_party_mau": int(third_party),
        }
    )
