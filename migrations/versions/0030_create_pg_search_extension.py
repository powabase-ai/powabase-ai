"""Create the pg_search extension when this Postgres has it.

Enables Postgres-native BM25 keyword search. Deliberately conditional: the
extension is optional, most deployments will not have the shared library at
all, and a keyword search works without it (bm25s file index, else the
tsvector fallback). So this revision must never fail the boot over it.
Idempotent via IF NOT EXISTS.

``pg_search`` requires the ``vector`` extension. ``CASCADE`` creates it when a
bootstrap has not already done so (and does nothing when it has).

Three separate things can be missing, and each one is only observable at a
different point:

* the control file -- ``pg_available_extensions`` does not list ``pg_search``,
  checked before issuing anything, and logged at INFO: this is the normal
  state of a server without the extension;
* the preload -- on PG15 ``CREATE EXTENSION`` fails with "pg_search must be
  loaded via shared_preload_libraries" even though the control file is there;
* the privilege -- the connected role may not be allowed to create it (it is
  not a trusted extension).

The last two only show up as a failed CREATE. That runs under a savepoint, so
the migration transaction stays usable for the revisions after this one, and
the failure is logged at ERROR from Python with the server's message: the
extension *is* installed on the server, so not being able to enable it is
something an operator has to see. It does not raise.

This revision runs once per database. A server that only gains the extension
later (an image swap after this revision was stamped) is covered by the
start-up hook in ``agentic_project_service._pg_search_extension``, which runs
the same guarded CREATE on every start.

Revision ID: 0030
Revises: 0029
Create Date: 2026-09-15
"""

import logging

from alembic import op
from sqlalchemy import text

revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

EXTENSION = "pg_search"


def upgrade():
    bind = op.get_bind()
    available = bind.execute(
        text("SELECT 1 FROM pg_available_extensions WHERE name = :name"), {"name": EXTENSION}
    ).first()
    if not available:
        logger.info(
            "%s is not available in this PostgreSQL install; skipping CREATE EXTENSION. "
            "Keyword search keeps using the bm25s file index or the tsvector fallback.",
            EXTENSION,
        )
        return

    try:
        with bind.begin_nested():
            bind.exec_driver_sql(f"CREATE EXTENSION IF NOT EXISTS {EXTENSION} CASCADE")
    except Exception as exc:  # noqa: BLE001 - reported below, never fatal
        logger.error(
            "Could not create the %s extension although this server provides it: %s. "
            "Check that it is in shared_preload_libraries, that the vector extension "
            "is available, and that the migrating role may create extensions. Keyword "
            "search keeps using the bm25s file index or the tsvector fallback; the "
            "start-up hook retries on every start.",
            EXTENSION,
            str(getattr(exc, "orig", exc)).strip().splitlines()[0],
        )
        return

    installed = bind.execute(
        text("SELECT 1 FROM pg_extension WHERE extname = :name"), {"name": EXTENSION}
    ).first()
    if installed:
        logger.info("Created the %s extension", EXTENSION)
    else:  # pragma: no cover - CREATE succeeded without creating it
        logger.error("CREATE EXTENSION %s reported success but it is not installed", EXTENSION)


def downgrade():
    """Deliberately a no-op.

    DROP EXTENSION would take every ``USING bm25`` index with it, and the
    upgrade is already conditional -- so the reverse of "created it if this
    server had it" is "leave it alone". Drop it by hand if that is really
    wanted.
    """
