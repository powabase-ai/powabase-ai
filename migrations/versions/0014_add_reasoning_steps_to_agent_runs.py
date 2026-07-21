"""Add reasoning_steps column to agent_runs.

Stores per-step intermediary assistant reasoning from the ReAct loop,
enabling debugging, auditability, and execution trace replay.

Revision ID: 0014
Revises: 0013
Create Date: 2026-04-09

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE ai.agent_runs
        ADD COLUMN IF NOT EXISTS reasoning_steps JSONB
    """)


def downgrade():
    op.execute("""
        ALTER TABLE ai.agent_runs
        DROP COLUMN IF EXISTS reasoning_steps
    """)
