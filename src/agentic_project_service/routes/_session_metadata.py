"""The optional ``metadata`` object on session-create bodies.

Stored in ``agent_sessions.metadata`` / ``orchestration_sessions.metadata``
(JSONB) and returned by the session lists, so a caller can keep its own ids on
a session, e.g. ``{"conversation_id": "..."}``.
"""

import json

MAX_SESSION_METADATA_BYTES = 16 * 1024
_UNSTORABLE = "metadata must be JSON without NaN, Infinity or NUL characters"


def _has_nul(value) -> bool:
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, dict):
        return any(_has_nul(k) or _has_nul(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_has_nul(v) for v in value)
    return False


def parse_session_metadata(data: dict) -> tuple[dict | None, str | None]:
    """Return (metadata, error). Absent or null is (None, None).

    The limit is on the compact UTF-8 serialisation.
    """
    metadata = data.get("metadata")
    if metadata is None:
        return None, None
    if not isinstance(metadata, dict):
        return None, "metadata must be a JSON object"
    # Postgres jsonb stores neither NaN/Infinity (which Flask's JSON parser
    # accepts) nor NUL characters; either would fail the INSERT with a 500.
    try:
        serialised = json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    except ValueError:
        return None, _UNSTORABLE
    if _has_nul(metadata):
        return None, _UNSTORABLE
    size = len(serialised.encode("utf-8"))
    if size > MAX_SESSION_METADATA_BYTES:
        return None, f"metadata is limited to {MAX_SESSION_METADATA_BYTES} bytes of JSON"
    return metadata, None
