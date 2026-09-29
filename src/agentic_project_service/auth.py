"""JWT Authentication for project service.

Validates tokens issued by the project's Supabase Auth.
"""

import functools
import hmac
import logging
import os
import uuid

import jwt
from flask import g, jsonify, request

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Authentication error with status code."""

    def __init__(self, message: str, status_code: int = 401):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def get_token_from_header() -> str | None:
    """Extract JWT token from Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        return None

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    return parts[1]


def decode_jwt(token: str) -> dict:
    """Decode and validate a Supabase JWT token.

    Accepts both:
    - User tokens with audience="authenticated"
    - Service role tokens (bypass audience check)

    Only the second carries ``is_service_role``, and only because the bearer
    is exactly ``SERVICE_ROLE_KEY``: a user token that claims it has the claim
    removed.
    """
    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        raise AuthError("JWT_SECRET not configured", 500)

    # Check if this is the service role key
    service_role_key = os.getenv("SERVICE_ROLE_KEY")
    if service_role_key and hmac.compare_digest(token.encode(), service_role_key.encode()):
        # Decode without audience validation for service role
        try:
            payload = jwt.decode(
                token,
                jwt_secret,
                algorithms=["HS256"],
                options={"verify_aud": False},
            )
            # Mark this as a service role request
            payload["is_service_role"] = True
            return payload
        except jwt.InvalidTokenError as e:
            raise AuthError(f"Invalid service token: {str(e)}") from None

    # For regular user tokens, validate audience. GoTrue always sets exp and
    # sub; a token without them never expires or names nobody.
    try:
        payload = jwt.decode(
            token,
            jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("Token has expired") from None
    except jwt.InvalidAudienceError:
        raise AuthError("Invalid token audience") from None
    except jwt.InvalidTokenError as e:
        raise AuthError(f"Invalid token: {str(e)}") from None
    # Stock GoTrue never issues this claim, but a custom access-token hook or a
    # project minting its own tokens can put anything in a JWT it signs.
    payload.pop("is_service_role", None)
    return payload


def get_current_user_id() -> str | None:
    """Get the current authenticated user's ID."""
    return getattr(g, "user_id", None)


def is_service_role_request() -> bool:
    """True when the current request authenticated with the service role key."""
    return getattr(g, "is_service_role", False) is True


def _is_uuid(value) -> bool:
    """True for a uuid in its canonical hyphenated spelling (any case)."""
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value.lower()
    except ValueError:
        return False


def _authenticate():
    """Validate the bearer token and populate ``g``.

    Returns an error response tuple, or None when the caller is authenticated.
    """
    token = get_token_from_header()
    if not token:
        return jsonify({"error": "Authorization header required"}), 401

    try:
        payload = decode_jwt(token)
    except AuthError as e:
        return jsonify({"error": e.message}), e.status_code

    # decode_jwt sets the flag only when the bearer is the service role key.
    is_service_role = payload.get("is_service_role") is True
    # Every end-user check downstream compares a row's owner with this id, so
    # an end-user token must carry one: without it, "owned by nobody" and "the
    # caller" would both be None.
    if not is_service_role:
        if not _is_uuid(payload.get("sub")):
            return jsonify({"error": "Invalid token: sub must be a user id"}), 401
        # Owners are compared as canonical strings, so one spelling of the id
        # throughout: the tools act with these same claims.
        payload["sub"] = str(uuid.UUID(payload["sub"]))

    g.is_service_role = is_service_role
    g.user_id = payload.get("sub")
    g.user_role = payload.get("role", "authenticated")
    g.jwt_payload = payload
    return None


# Every /api route takes exactly one of the two decorators below (or, for the
# few that verify a different credential, neither — see the route inventory
# test). The project's anon key is public and the gateway accepts it on every
# route, so any signed-in end user can present their own JWT anywhere; which
# decorator a route carries is the whole of its access control.


def require_service_role(f):
    """Route callable only with the project's service role key.

    The default for every route: managing knowledge bases, sources, agents,
    workflows, settings, keys and tables is done from a trusted backend or the
    dashboard. An end user's JWT gets 403.
    """

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        error = _authenticate()
        if error:
            return error
        if not is_service_role_request():
            # The token itself is never logged.
            logger.info(
                "Refused %s %s: service role key required (sub=%s)",
                request.method,
                request.path,
                get_current_user_id(),
            )
            return jsonify({"error": "This endpoint requires the project's service role key"}), 403
        return f(*args, **kwargs)

    decorated.auth_mode = "service_role"
    return decorated


def require_user_auth(f):
    """Route an end user may call with their own JWT (the service role may too).

    Only for routes that scope everything they read or change to the caller's
    own sessions and runs. Adding a route here is a security decision.
    """

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        error = _authenticate()
        if error:
            return error
        return f(*args, **kwargs)

    decorated.auth_mode = "user"
    return decorated


# Run-body fields that make an agent read project data it was not configured
# with. The service role may use them; an end user may not, or any user could
# read any knowledge base by naming it in a run.
_END_USER_FORBIDDEN_RUN_FIELDS = (
    "knowledge_bases",
    "runtime_knowledge_bases",
    "context_handler_id",
)


def end_user_run_body_error(data: dict) -> str | None:
    """Return why an end user may not send this run body, or None if they may.

    Context the caller supplies themselves (``context_override``, by-value
    ``context_items``) is allowed; anything that references stored data is not.
    """
    named = [field for field in _END_USER_FORBIDDEN_RUN_FIELDS if data.get(field)]
    items = data.get("context_items") or []
    if any(isinstance(item, dict) and item.get("item_id") for item in items):
        named.append("context_items[].item_id")
    if not named:
        return None
    return (
        f"{', '.join(named)} may only be set with the project's service role key; "
        "an end user's run uses the knowledge bases configured on the agent"
    )
