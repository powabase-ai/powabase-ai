"""Add tools tables and ReAct loop columns.

Revision ID: 0005b
Revises: 0005
Create Date: 2026-03-27

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0005b"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade():
    # Tools table
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.tools (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            type VARCHAR(50) NOT NULL,
            input_schema JSONB NOT NULL,
            config JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Agent-tool assignment
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.agent_tools (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID NOT NULL REFERENCES ai.agents(id) ON DELETE CASCADE,
            tool_id UUID REFERENCES ai.tools(id) ON DELETE CASCADE,
            tool_type VARCHAR(50) NOT NULL,
            tool_name VARCHAR(255) NOT NULL,
            config_override JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_tools_agent_id ON ai.agent_tools(agent_id)
    """)

    # Agent-KB assignment
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.agent_knowledge_bases (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID NOT NULL REFERENCES ai.agents(id) ON DELETE CASCADE,
            knowledge_base_id UUID NOT NULL REFERENCES ai.knowledge_bases(id) ON DELETE CASCADE,
            config JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(agent_id, knowledge_base_id)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_kb_agent_id ON ai.agent_knowledge_bases(agent_id)
    """)

    # New columns on agent_runs
    op.execute("""
        ALTER TABLE ai.agent_runs
            ADD COLUMN IF NOT EXISTS parent_orchestration_run_id UUID,
            ADD COLUMN IF NOT EXISTS steps INTEGER,
            ADD COLUMN IF NOT EXISTS events JSONB DEFAULT '[]'
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_runs_parent_orch_run
            ON ai.agent_runs(parent_orchestration_run_id)
    """)


def downgrade():
    op.execute("DROP INDEX IF EXISTS ai.idx_agent_runs_parent_orch_run")
    op.execute("ALTER TABLE ai.agent_runs DROP COLUMN IF EXISTS events")
    op.execute("ALTER TABLE ai.agent_runs DROP COLUMN IF EXISTS steps")
    op.execute("ALTER TABLE ai.agent_runs DROP COLUMN IF EXISTS parent_orchestration_run_id")
    op.execute("DROP TABLE IF EXISTS ai.agent_knowledge_bases")
    op.execute("DROP TABLE IF EXISTS ai.agent_tools")
    op.execute("DROP TABLE IF EXISTS ai.tools")
