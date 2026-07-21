"""Add hooks table.

Revision ID: 0008b
Revises: 0008
Create Date: 2026-03-27

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0008b"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.hooks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID REFERENCES ai.agents(id) ON DELETE CASCADE,
            orchestration_id UUID REFERENCES ai.orchestrations(id) ON DELETE CASCADE,
            event VARCHAR(50) NOT NULL,
            matcher VARCHAR(255),
            type VARCHAR(50) NOT NULL,
            config JSONB NOT NULL,
            enabled BOOLEAN DEFAULT true,
            position INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            CHECK (agent_id IS NOT NULL OR orchestration_id IS NOT NULL)
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_hooks_agent_id ON ai.hooks(agent_id);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_hooks_orchestration_id ON ai.hooks(orchestration_id);
    """)


def downgrade():
    op.execute("DROP TABLE IF EXISTS ai.hooks;")
