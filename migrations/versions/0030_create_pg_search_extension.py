"""Create the pg_search extension when this Postgres has it.

Enables Postgres-native BM25 keyword search. Deliberately conditional: the
extension is optional, most deployments will not have the shared library at
all, and a keyword search works without it (bm25s file index, else the
tsvector fallback). So this migration must be a no-op wherever pg_search is
absent rather than a boot failure.

Three separate things can be missing, and each one is only observable at a
different point:

* the control file -- ``pg_available_extensions`` does not list ``pg_search``,
  checked before issuing anything;
* the preload -- on PG15 ``CREATE EXTENSION`` fails with "pg_search must be
  loaded via shared_preload_libraries" even though the control file is there;
* the privilege -- the connected role may not be allowed to create it.

Only the first is checkable up front, so the CREATE is wrapped in an exception
handler and downgraded to a warning. Idempotent via IF NOT EXISTS.

Revision ID: 0030
Revises: 0029
Create Date: 2026-09-15
"""

import logging

from alembic import op

revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")


def upgrade():
    available = (
        op.get_bind()
        .exec_driver_sql("SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search'")
        .first()
    )
    if not available:
        logger.info(
            "pg_search is not available in this PostgreSQL install; skipping "
            "CREATE EXTENSION. Keyword search keeps using the bm25s file index "
            "or the tsvector fallback."
        )
        return

    op.execute("""
        DO $$
        BEGIN
            EXECUTE 'CREATE EXTENSION IF NOT EXISTS pg_search';
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'Could not create the pg_search extension (%): keyword '
                          'search will keep using the existing index paths', SQLERRM;
        END $$;
    """)


def downgrade():
    """Deliberately a no-op.

    DROP EXTENSION would take every ``USING bm25`` index with it, and the
    upgrade is already conditional -- so the reverse of "created it if this
    server had it" is "leave it alone". Drop it by hand if that is really
    wanted.
    """
