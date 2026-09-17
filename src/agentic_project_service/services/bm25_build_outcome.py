"""Persisted outcome of a knowledge base's BM25 partition move/build.

The move from the DEFAULT partition into a knowledge base's own partition
(and the index build that follows it) can retry several times before it
succeeds or gives up. This module upserts the latest status into
``ai.bm25_index_builds`` (migration 0032) so a KB's ``bm25_status`` can be
reported even after the worker restarts, and reads it back.

Both functions accept a SQLAlchemy Engine, Connection or Session as ``bind``
and never raise: an Engine gets its own short transaction, while a
Connection or Session runs inside a savepoint (``begin_nested``), so a
failure here can never abort a caller's own ambient transaction.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..db import AI_SCHEMA

logger = logging.getLogger(__name__)

STATUSES = frozenset({"queued", "moving", "building", "ready", "retrying", "failed", "unavailable"})

_TABLE = f"{AI_SCHEMA}.bm25_index_builds"

_UPSERT_SQL = text(f"""
    INSERT INTO {_TABLE}
        (knowledge_base_id, item_table, status, reason, attempts, updated_at)
    VALUES (CAST(:kb_id AS uuid), :item_table, :status, :reason, :attempts, now())
    ON CONFLICT (knowledge_base_id, item_table) DO UPDATE SET
        status = EXCLUDED.status,
        reason = EXCLUDED.reason,
        attempts = EXCLUDED.attempts,
        updated_at = EXCLUDED.updated_at
""")

_LATEST_SQL = text(f"""
    SELECT status, reason, item_table, attempts, updated_at
    FROM {_TABLE}
    WHERE knowledge_base_id = CAST(:kb_id AS uuid)
    ORDER BY updated_at DESC
    LIMIT 1
""")


def _run(bind, fn):
    """Run ``fn`` against a connection derived from ``bind``.

    An Engine gets its own short transaction (nothing to protect); a
    Connection or Session runs ``fn`` inside a savepoint, so a failure
    rolls back only the savepoint and leaves the caller's own transaction
    usable.
    """
    if isinstance(bind, Engine):
        with bind.begin() as conn:
            return fn(conn)
    with bind.begin_nested():
        return fn(bind)


def _log_warning(action: str, kb_id: str, exc: Exception) -> None:
    message = str(exc).splitlines()[0] if str(exc) else repr(exc)
    logger.warning(
        "Could not %s bm25 build outcome for knowledge_base_id=%s: %s",
        action,
        kb_id,
        message,
    )


def record_bm25_build_outcome(
    bind,
    kb_id: str,
    item_table: str,
    status: str,
    reason: str | None = None,
    attempts: int | None = None,
) -> None:
    """Upsert the latest build outcome for (kb_id, item_table).

    ``bind`` is a SQLAlchemy Engine, Connection or Session. Never raises:
    logs a WARNING and returns on any error (including an unknown status or
    a missing table on an unmigrated database).
    """
    try:
        if status not in STATUSES:
            raise ValueError(f"unknown bm25 build status: {status!r}")
        kb_uuid = str(uuid.UUID(str(kb_id)))
        params = {
            "kb_id": kb_uuid,
            "item_table": item_table,
            "status": status,
            "reason": reason,
            "attempts": attempts,
        }
        _run(bind, lambda conn: conn.execute(_UPSERT_SQL, params))
    except Exception as exc:  # noqa: BLE001 - never raise; logged instead
        _log_warning("record", kb_id, exc)


def read_bm25_build_outcome(bind, kb_id: str) -> dict[str, Any] | None:
    """Latest outcome row for the KB across item tables.

    Returns ``{"status", "reason", "item_table", "attempts", "updated_at"}``
    for the most recently updated row, or ``None`` when there is none or the
    table does not exist. Never raises.
    """
    try:
        kb_uuid = str(uuid.UUID(str(kb_id)))
        params = {"kb_id": kb_uuid}

        def _read(conn):
            row = conn.execute(_LATEST_SQL, params).mappings().first()
            if row is None:
                return None
            return {
                "status": row["status"],
                "reason": row["reason"],
                "item_table": row["item_table"],
                "attempts": row["attempts"],
                "updated_at": row["updated_at"],
            }

        return _run(bind, _read)
    except Exception as exc:  # noqa: BLE001 - never raise; logged instead
        _log_warning("read", kb_id, exc)
        return None
