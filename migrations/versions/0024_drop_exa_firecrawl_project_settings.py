"""Drop orphan EXA_API_KEY / FIRECRAWL_API_KEY rows from ai.project_settings.

EXA_API_KEY and FIRECRAWL_API_KEY are now platform-paid env-injected secrets
(read from os.environ in the tool handlers), no longer tenant-managed
settings. Any rows for these keys that pre-existed in ai.project_settings
are unreachable — get_setting() no longer consults the DB for them and the
Studio Settings UI no longer renders fields for them. Drop the dead rows so
the table doesn't carry confusing artifacts for anyone querying it later.

Defensive — DELETE is a no-op when no rows match (most projects never had
a tenant override). The downgrade is intentionally a no-op: the secrets
were per-project user-supplied keys we cannot recover, and rolling back
this migration does not restore the now-removed registry entries either.

Revision ID: 0024
Revises: 0023
Create Date: 2026-06-18
"""

from alembic import op

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        DELETE FROM ai.project_settings
         WHERE key IN ('EXA_API_KEY', 'FIRECRAWL_API_KEY');
    """)


def downgrade():
    # Intentional no-op: the deleted rows held per-project user-supplied
    # secrets that cannot be reconstructed.
    pass
