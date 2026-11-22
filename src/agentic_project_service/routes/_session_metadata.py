"""The optional ``metadata`` object on session-create bodies.

Stored in ``agent_sessions.metadata`` / ``orchestration_sessions.metadata``
(JSONB) and returned by the session lists, so a caller can keep its own ids on
a session, e.g. ``{"conversation_id": "..."}``.
"""

import json

MAX_SESSION_METADATA_BYTES = 16 * 1024


def parse_session_metadata(data: dict) -> tuple[dict | None, str | None]:
    """Return (metadata, error). Absent or null is (None, None).

    The limit is on the compact UTF-8 serialisation.
    """
    metadata = data.get("metadata")
    if metadata is None:
        return None, None
    if not isinstance(metadata, dict):
        return None, "metadata must be a JSON object"
    size = len(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if size > MAX_SESSION_METADATA_BYTES:
        return None, f"metadata is limited to {MAX_SESSION_METADATA_BYTES} bytes of JSON"
    return metadata, None
