"""Enable the optional pg_search extension at start-up when the server has it.

Revision 0030 creates the extension, but a revision runs once per database: a
project whose Postgres only gains pg_search later -- an image swap after 0030
was stamped -- would otherwise never get it enabled without a manual
``CREATE EXTENSION``. This hook runs the same guarded CREATE on every start.

It is idempotent and never raises. Outcomes, and how each is logged:

* ``present`` -- already created; nothing logged;
* ``unavailable`` -- the server has no pg_search control file, the normal state
  without the extension; INFO;
* ``created`` -- INFO;
* ``failed`` -- the server provides pg_search but it could not be created
  (not preloaded, ``vector`` unavailable, or the role lacks the privilege);
  ERROR with the server's message, because keyword search then silently stays
  on the slower paths.

``pg_search`` requires ``vector``; ``CASCADE`` creates it if it is missing.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

logger = logging.getLogger(__name__)

EXTENSION = "pg_search"


def ensure_pg_search_extension(engine) -> str:
    """Create pg_search if this server provides it; return what happened."""
    try:
        with engine.begin() as conn:
            if conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = :name"), {"name": EXTENSION}
            ).first():
                return "present"
            if not conn.execute(
                text("SELECT 1 FROM pg_available_extensions WHERE name = :name"),
                {"name": EXTENSION},
            ).first():
                logger.info(
                    "%s is not available on this server; keyword search uses the bm25s "
                    "file index or the tsvector fallback",
                    EXTENSION,
                )
                return "unavailable"
            conn.exec_driver_sql(f"CREATE EXTENSION IF NOT EXISTS {EXTENSION} CASCADE")
    except Exception as exc:  # noqa: BLE001 - start-up must not fail over an optional extension
        logger.error(
            "Could not create the %s extension although this server provides it: %s. "
            "Check that it is in shared_preload_libraries, that the vector extension is "
            "available, and that this role may create extensions. Keyword search keeps "
            "using the bm25s file index or the tsvector fallback.",
            EXTENSION,
            str(getattr(exc, "orig", exc)).strip().splitlines()[0],
        )
        return "failed"
    logger.info("Created the %s extension", EXTENSION)
    return "created"
