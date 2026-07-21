"""Add agent_mcp_servers table.

Revision ID: 0007b
Revises: 0007
Create Date: 2026-03-27

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0007b"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.agent_mcp_servers (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID NOT NULL REFERENCES ai.agents(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            transport VARCHAR(50) NOT NULL DEFAULT 'http',
            url TEXT NOT NULL,
            headers JSONB DEFAULT '{}',
            config JSONB DEFAULT '{}',
            enabled BOOLEAN DEFAULT true,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(agent_id, name)
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_mcp_servers_agent_id
            ON ai.agent_mcp_servers(agent_id);
    """)


def downgrade():
    op.execute("DROP TABLE IF EXISTS ai.agent_mcp_servers;")
