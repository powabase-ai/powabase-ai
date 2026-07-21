"""JWT Authentication for project service.

Validates tokens issued by the project's Supabase Auth.
"""

import functools
import os

import jwt
from flask import g, jsonify, request


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
    """
    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        raise AuthError("JWT_SECRET not configured", 500)

    # Check if this is the service role key
    service_role_key = os.getenv("SERVICE_ROLE_KEY")
    if service_role_key and token == service_role_key:
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

    # For regular user tokens, validate audience
    try:
        payload = jwt.decode(
            token,
            jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthError("Token has expired") from None
    except jwt.InvalidAudienceError:
        raise AuthError("Invalid token audience") from None
    except jwt.InvalidTokenError as e:
        raise AuthError(f"Invalid token: {str(e)}") from None


def get_current_user_id() -> str | None:
    """Get the current authenticated user's ID."""
    return getattr(g, "user_id", None)


def require_auth(f):
    """Decorator to require authentication for a route."""

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_header()
        if not token:
            return jsonify({"error": "Authorization header required"}), 401

        try:
            payload = decode_jwt(token)
            g.user_id = payload.get("sub")
            g.user_role = payload.get("role", "authenticated")
            g.jwt_payload = payload
        except AuthError as e:
            return jsonify({"error": e.message}), e.status_code

        return f(*args, **kwargs)

    return decorated


def optional_auth(f):
    """Decorator for routes that optionally use authentication."""

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_header()
        if token:
            try:
                payload = decode_jwt(token)
                g.user_id = payload.get("sub")
                g.user_role = payload.get("role", "authenticated")
                g.jwt_payload = payload
            except AuthError:
                g.user_id = None
                g.user_role = None
                g.jwt_payload = None
        else:
            g.user_id = None
            g.user_role = None
            g.jwt_payload = None

        return f(*args, **kwargs)

    return decorated
