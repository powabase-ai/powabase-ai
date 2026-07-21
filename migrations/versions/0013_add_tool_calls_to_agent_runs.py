"""Add tool_calls column to agent_runs.

Stores full ToolCallRecord data (arguments + results) for historical
observability in the message debug panel.

Revision ID: 0013
Revises: 0012
Create Date: 2026-04-08

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE ai.agent_runs
        ADD COLUMN IF NOT EXISTS tool_calls JSONB
    """)


def downgrade():
    op.execute("""
        ALTER TABLE ai.agent_runs
        DROP COLUMN IF EXISTS tool_calls
    """)
