"""The optional ``metadata`` object on session-create bodies.

Stored in ``agent_sessions.metadata`` / ``orchestration_sessions.metadata``
(JSONB) and returned by the session lists, so a caller can keep its own ids on
a session, e.g. ``{"conversation_id": "..."}``.

Untrusted: on agent sessions an end user's own JWT may set it too, on a
session that user owns. An integrator that needs metadata it can trust must
write it with the service role key (orchestration session create accepts only
that) and not rely on metadata of sessions end users created.
"""

import json

MAX_SESSION_METADATA_BYTES = 16 * 1024
_UNSTORABLE = "metadata must be JSON without NaN, Infinity, NUL characters or lone surrogates"


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
    # Postgres jsonb stores none of NaN/Infinity (which Flask's JSON parser
    # accepts), NUL characters or lone surrogates (a "\ud800" escape parses to
    # one); each would fail the INSERT with a 500.
    try:
        serialised = json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        size = len(serialised.encode("utf-8"))  # raises on a lone surrogate
    except (ValueError, UnicodeEncodeError):
        return None, _UNSTORABLE
    if _has_nul(metadata):
        return None, _UNSTORABLE
    if size > MAX_SESSION_METADATA_BYTES:
        return None, f"metadata is limited to {MAX_SESSION_METADATA_BYTES} bytes of JSON"
    return metadata, None
