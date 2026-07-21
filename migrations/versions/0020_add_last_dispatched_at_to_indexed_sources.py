"""Add last_dispatched_at to ai.indexed_sources for watchdog.

The indexed_sources watchdog needs a timestamp it can trust to be "after the
dispatcher finished" so it won't race with /reindex (which commits
status='pending' BEFORE calling .delay()). One column, additive, DEFAULT
NOW() so existing INSERTs need no code change.

Revision ID: 0020
Revises: 0019
Create Date: 2026-05-15
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE ai.indexed_sources
            ADD COLUMN IF NOT EXISTS last_dispatched_at TIMESTAMPTZ DEFAULT NOW()
    """)
    # Backfill existing rows from created_at (best-effort proxy for first dispatch time).
    # Skipped if created_at is NULL (very old rows from before created_at was added).
    op.execute("""
        UPDATE ai.indexed_sources
           SET last_dispatched_at = created_at
         WHERE last_dispatched_at IS NULL
           AND created_at IS NOT NULL
    """)


def downgrade():
    op.execute("""
        ALTER TABLE ai.indexed_sources
            DROP COLUMN IF EXISTS last_dispatched_at
    """)
