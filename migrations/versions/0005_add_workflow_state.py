"""Add state and webhook_armed_until columns to workflows table.

Revision ID: 0005
Revises: 0004
Create Date: 2026-03-20

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE ai.workflows
        ADD COLUMN IF NOT EXISTS state VARCHAR(20) DEFAULT 'internal'
            CHECK (state IN ('internal', 'deployed'));
        ALTER TABLE ai.workflows
        ADD COLUMN IF NOT EXISTS webhook_armed_until TIMESTAMPTZ;
    """)


def downgrade():
    op.execute("""
        ALTER TABLE ai.workflows DROP COLUMN IF EXISTS webhook_armed_until;
        ALTER TABLE ai.workflows DROP COLUMN IF EXISTS state;
    """)
