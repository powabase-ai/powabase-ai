"""Add workflow tables for DAG-based pipeline execution.

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-17

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.workflows (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            variables JSONB DEFAULT '{}',
            version INTEGER DEFAULT 1,
            color VARCHAR(50),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS ai.workflow_blocks (
            id VARCHAR(255) NOT NULL,
            workflow_id UUID NOT NULL REFERENCES ai.workflows(id) ON DELETE CASCADE,
            type VARCHAR(50) NOT NULL,
            name VARCHAR(255),
            position_x REAL DEFAULT 0,
            position_y REAL DEFAULT 0,
            config JSONB DEFAULT '{}',
            enabled BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (workflow_id, id)
        );

        CREATE TABLE IF NOT EXISTS ai.workflow_edges (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workflow_id UUID NOT NULL REFERENCES ai.workflows(id) ON DELETE CASCADE,
            source_block_id VARCHAR(255) NOT NULL,
            target_block_id VARCHAR(255) NOT NULL,
            source_handle VARCHAR(50) DEFAULT 'output',
            target_handle VARCHAR(50) DEFAULT 'input',
            condition VARCHAR(255),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS ai.workflow_executions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workflow_id UUID NOT NULL REFERENCES ai.workflows(id) ON DELETE CASCADE,
            status VARCHAR(50) DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'completed', 'failed')),
            input JSONB DEFAULT '{}',
            output JSONB DEFAULT '{}',
            block_outputs JSONB DEFAULT '{}',
            error TEXT,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_ai_workflow_blocks_wf ON ai.workflow_blocks (workflow_id);
        CREATE INDEX IF NOT EXISTS idx_ai_workflow_edges_wf ON ai.workflow_edges (workflow_id);
        CREATE INDEX IF NOT EXISTS idx_ai_workflow_executions_wf ON ai.workflow_executions (workflow_id);
        CREATE INDEX IF NOT EXISTS idx_ai_workflow_executions_status ON ai.workflow_executions (status);

        -- RLS
        ALTER TABLE ai.workflows ENABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.workflow_blocks ENABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.workflow_edges ENABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.workflow_executions ENABLE ROW LEVEL SECURITY;

        -- Service role full access
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY service_role_all_workflows ON ai.workflows FOR ALL TO service_role USING (true) WITH CHECK (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY service_role_all_workflow_blocks ON ai.workflow_blocks FOR ALL TO service_role USING (true) WITH CHECK (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY service_role_all_workflow_edges ON ai.workflow_edges FOR ALL TO service_role USING (true) WITH CHECK (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY service_role_all_workflow_executions ON ai.workflow_executions FOR ALL TO service_role USING (true) WITH CHECK (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;

        -- Authenticated read/write
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_read_workflows ON ai.workflows FOR SELECT TO authenticated USING (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_write_workflows ON ai.workflows FOR INSERT TO authenticated WITH CHECK (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_update_workflows ON ai.workflows FOR UPDATE TO authenticated USING (true) WITH CHECK (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_delete_workflows ON ai.workflows FOR DELETE TO authenticated USING (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;

        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_read_workflow_blocks ON ai.workflow_blocks FOR SELECT TO authenticated USING (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_write_workflow_blocks ON ai.workflow_blocks FOR INSERT TO authenticated WITH CHECK (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;

        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_read_workflow_edges ON ai.workflow_edges FOR SELECT TO authenticated USING (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_write_workflow_edges ON ai.workflow_edges FOR INSERT TO authenticated WITH CHECK (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;

        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_read_workflow_executions ON ai.workflow_executions FOR SELECT TO authenticated USING (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_write_workflow_executions ON ai.workflow_executions FOR INSERT TO authenticated WITH CHECK (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
        DO $$ BEGIN
            EXECUTE 'CREATE POLICY auth_update_workflow_executions ON ai.workflow_executions FOR UPDATE TO authenticated USING (true) WITH CHECK (true)';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;

        -- Updated at trigger
        DO $$ BEGIN
            CREATE TRIGGER set_updated_at_workflows
                BEFORE UPDATE ON ai.workflows
                FOR EACH ROW EXECUTE FUNCTION ai.trigger_set_updated_at();
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;

        -- Grant permissions
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.workflows TO authenticated, service_role;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.workflow_blocks TO authenticated, service_role;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.workflow_edges TO authenticated, service_role;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.workflow_executions TO authenticated, service_role;
        GRANT SELECT ON ai.workflows TO anon;
        GRANT SELECT ON ai.workflow_blocks TO anon;
        GRANT SELECT ON ai.workflow_edges TO anon;
        GRANT SELECT ON ai.workflow_executions TO anon;
    """)


def downgrade():
    op.execute("""
        DROP TABLE IF EXISTS ai.workflow_executions CASCADE;
        DROP TABLE IF EXISTS ai.workflow_edges CASCADE;
        DROP TABLE IF EXISTS ai.workflow_blocks CASCADE;
        DROP TABLE IF EXISTS ai.workflows CASCADE;
    """)
