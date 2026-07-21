"""Add workflow_block_logs table for persistent per-block execution data.

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-18

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.workflow_block_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            execution_id UUID NOT NULL REFERENCES ai.workflow_executions(id) ON DELETE CASCADE,
            block_id VARCHAR(255) NOT NULL,
            block_type VARCHAR(50) NOT NULL,
            block_name VARCHAR(255),
            status VARCHAR(50) NOT NULL CHECK (status IN ('success', 'error', 'skipped')),
            execution_order INTEGER NOT NULL,
            duration_ms REAL,
            input JSONB DEFAULT '{}',
            output JSONB DEFAULT '{}',
            error TEXT,
            config_snapshot JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_wf_block_logs_exec
            ON ai.workflow_block_logs (execution_id);
        CREATE INDEX IF NOT EXISTS idx_wf_block_logs_exec_order
            ON ai.workflow_block_logs (execution_id, execution_order);

        -- RLS
        ALTER TABLE ai.workflow_block_logs ENABLE ROW LEVEL SECURITY;

        -- Service role full access
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY service_role_all_workflow_block_logs ON ai.workflow_block_logs FOR ALL TO service_role USING (true) WITH CHECK (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;

        -- Authenticated read
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_read_workflow_block_logs ON ai.workflow_block_logs FOR SELECT TO authenticated USING (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;

        -- Authenticated write
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_write_workflow_block_logs ON ai.workflow_block_logs FOR INSERT TO authenticated WITH CHECK (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;

        -- Authenticated update
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_update_workflow_block_logs ON ai.workflow_block_logs FOR UPDATE TO authenticated USING (true) WITH CHECK (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;

        -- Grant permissions
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.workflow_block_logs TO authenticated, service_role;
        GRANT SELECT ON ai.workflow_block_logs TO anon;
    """)


def downgrade():
    op.execute("""
        DROP TABLE IF EXISTS ai.workflow_block_logs CASCADE;
    """)
