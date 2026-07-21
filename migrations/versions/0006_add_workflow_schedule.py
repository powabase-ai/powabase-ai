"""Add schedule columns to workflows table.

Revision ID: 0006
Revises: 0005b
Create Date: 2026-03-23

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0006"
down_revision = "0005b"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE ai.workflows
        ADD COLUMN IF NOT EXISTS schedule_config JSONB DEFAULT NULL;
        ALTER TABLE ai.workflows
        ADD COLUMN IF NOT EXISTS schedule_run_count INTEGER DEFAULT 0;
        ALTER TABLE ai.workflows
        ADD COLUMN IF NOT EXISTS last_scheduled_at TIMESTAMPTZ;
    """)


def downgrade():
    op.execute("""
        ALTER TABLE ai.workflows DROP COLUMN IF EXISTS last_scheduled_at;
        ALTER TABLE ai.workflows DROP COLUMN IF EXISTS schedule_run_count;
        ALTER TABLE ai.workflows DROP COLUMN IF EXISTS schedule_config;
    """)
