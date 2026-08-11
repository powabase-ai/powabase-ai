"""add attempts to ai.indexed_sources

Bounds indexing retries: the reconciler fails a row terminally once this
counter reaches the configured maximum, instead of re-dispatching forever.

Revision ID: 0027
Revises: 0026
"""
from alembic import op

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE ai.indexed_sources "
        "ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0"
    )


def downgrade():
    op.execute("ALTER TABLE ai.indexed_sources DROP COLUMN IF EXISTS attempts")
