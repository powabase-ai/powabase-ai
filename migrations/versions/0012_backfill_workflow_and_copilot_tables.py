"""Backfill workflow and copilot tables skipped by migration chain renumbering.

Root cause: PR #54 (merged 2026-04-02) inserted workflow migrations into the
middle of the existing chain, renumbering non-workflow migrations with "b"
suffixes (e.g. old 0005=tools became 0005b, old 0008=hooks became 0008b).
Crucially, the NEW 0008 (copilot_tables) reused the same revision ID as the
OLD 0008 (hooks).  All existing projects were already at alembic_version=0008
(the old hooks migration).  After PR #54 deployed, Alembic matched '0008' to
the new chain's 0008 (copilot_tables) and only ran 0008b→0011 — completely
skipping 0003-0007 (workflow tables) and 0008 (copilot tables).

This migration idempotently creates all 8 missing tables using the final
schema from ai_schema.sql.

Revision ID: 0012
Revises: 0011
Create Date: 2026-04-09

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade():
    # ── Workflow tables (from skipped 0003-0007) ────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.workflows (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            variables JSONB DEFAULT '{}',
            version INTEGER DEFAULT 1,
            color VARCHAR(50),
            state VARCHAR(20) DEFAULT 'internal'
                CHECK (state IN ('internal', 'deployed')),
            webhook_armed_until TIMESTAMPTZ,
            schedule_config JSONB DEFAULT NULL,
            schedule_run_count INTEGER DEFAULT 0,
            last_scheduled_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS ai.workflow_blocks (
            id UUID NOT NULL DEFAULT gen_random_uuid(),
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
            source_block_id UUID NOT NULL,
            target_block_id UUID NOT NULL,
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

        CREATE TABLE IF NOT EXISTS ai.workflow_block_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            execution_id UUID NOT NULL REFERENCES ai.workflow_executions(id) ON DELETE CASCADE,
            block_id UUID NOT NULL,
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
    """)

    # ── Copilot tables (from skipped 0008) ──────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.project_settings (
            key VARCHAR(255) PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMPTZ DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS ai.copilot_sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workflow_id UUID NOT NULL REFERENCES ai.workflows(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS ai.copilot_messages (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID NOT NULL REFERENCES ai.copilot_sessions(id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            workflow_diff JSONB,
            pre_snapshot JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    # ── Indexes ─────────────────────────────────────────────────────────
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_ai_workflow_blocks_wf ON ai.workflow_blocks (workflow_id);
        CREATE INDEX IF NOT EXISTS idx_ai_workflow_edges_wf ON ai.workflow_edges (workflow_id);
        CREATE INDEX IF NOT EXISTS idx_ai_workflow_executions_wf ON ai.workflow_executions (workflow_id);
        CREATE INDEX IF NOT EXISTS idx_ai_workflow_executions_status ON ai.workflow_executions (status);
        CREATE INDEX IF NOT EXISTS idx_wf_block_logs_exec ON ai.workflow_block_logs (execution_id);
        CREATE INDEX IF NOT EXISTS idx_wf_block_logs_exec_order ON ai.workflow_block_logs (execution_id, execution_order);
        CREATE INDEX IF NOT EXISTS idx_copilot_sessions_workflow ON ai.copilot_sessions (workflow_id);
        CREATE INDEX IF NOT EXISTS idx_copilot_messages_session ON ai.copilot_messages (session_id, created_at);
    """)

    # ── RLS ──────────────────────────────────────────────────────────────
    op.execute("""
        ALTER TABLE ai.workflows ENABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.workflow_blocks ENABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.workflow_edges ENABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.workflow_executions ENABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.workflow_block_logs ENABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.project_settings ENABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.copilot_sessions ENABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.copilot_messages ENABLE ROW LEVEL SECURITY;
    """)

    # ── Policies (idempotent via DROP IF EXISTS + CREATE) ────────────────
    op.execute("""
        -- workflows
        DO $$ BEGIN EXECUTE 'CREATE POLICY service_role_all_workflows ON ai.workflows FOR ALL TO service_role USING (true) WITH CHECK (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_read_workflows ON ai.workflows FOR SELECT TO authenticated USING (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_write_workflows ON ai.workflows FOR INSERT TO authenticated WITH CHECK (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_update_workflows ON ai.workflows FOR UPDATE TO authenticated USING (true) WITH CHECK (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_delete_workflows ON ai.workflows FOR DELETE TO authenticated USING (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;

        -- workflow_blocks
        DO $$ BEGIN EXECUTE 'CREATE POLICY service_role_all_workflow_blocks ON ai.workflow_blocks FOR ALL TO service_role USING (true) WITH CHECK (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_read_workflow_blocks ON ai.workflow_blocks FOR SELECT TO authenticated USING (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_write_workflow_blocks ON ai.workflow_blocks FOR INSERT TO authenticated WITH CHECK (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;

        -- workflow_edges
        DO $$ BEGIN EXECUTE 'CREATE POLICY service_role_all_workflow_edges ON ai.workflow_edges FOR ALL TO service_role USING (true) WITH CHECK (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_read_workflow_edges ON ai.workflow_edges FOR SELECT TO authenticated USING (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_write_workflow_edges ON ai.workflow_edges FOR INSERT TO authenticated WITH CHECK (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;

        -- workflow_executions
        DO $$ BEGIN EXECUTE 'CREATE POLICY service_role_all_workflow_executions ON ai.workflow_executions FOR ALL TO service_role USING (true) WITH CHECK (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_read_workflow_executions ON ai.workflow_executions FOR SELECT TO authenticated USING (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_write_workflow_executions ON ai.workflow_executions FOR INSERT TO authenticated WITH CHECK (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_update_workflow_executions ON ai.workflow_executions FOR UPDATE TO authenticated USING (true) WITH CHECK (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;

        -- workflow_block_logs
        DO $$ BEGIN EXECUTE 'CREATE POLICY service_role_all_workflow_block_logs ON ai.workflow_block_logs FOR ALL TO service_role USING (true) WITH CHECK (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_read_workflow_block_logs ON ai.workflow_block_logs FOR SELECT TO authenticated USING (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_write_workflow_block_logs ON ai.workflow_block_logs FOR INSERT TO authenticated WITH CHECK (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        DO $$ BEGIN EXECUTE 'CREATE POLICY auth_update_workflow_block_logs ON ai.workflow_block_logs FOR UPDATE TO authenticated USING (true) WITH CHECK (true)'; EXCEPTION WHEN duplicate_object THEN NULL; END $$;

        -- project_settings
        DROP POLICY IF EXISTS service_role_all_project_settings ON ai.project_settings;
        CREATE POLICY service_role_all_project_settings ON ai.project_settings FOR ALL TO service_role USING (true) WITH CHECK (true);
        DROP POLICY IF EXISTS auth_read_project_settings ON ai.project_settings;
        CREATE POLICY auth_read_project_settings ON ai.project_settings FOR SELECT TO authenticated USING (true);

        -- copilot_sessions
        DROP POLICY IF EXISTS service_role_all_copilot_sessions ON ai.copilot_sessions;
        CREATE POLICY service_role_all_copilot_sessions ON ai.copilot_sessions FOR ALL TO service_role USING (true) WITH CHECK (true);
        DROP POLICY IF EXISTS auth_read_copilot_sessions ON ai.copilot_sessions;
        CREATE POLICY auth_read_copilot_sessions ON ai.copilot_sessions FOR SELECT TO authenticated USING (true);

        -- copilot_messages
        DROP POLICY IF EXISTS service_role_all_copilot_messages ON ai.copilot_messages;
        CREATE POLICY service_role_all_copilot_messages ON ai.copilot_messages FOR ALL TO service_role USING (true) WITH CHECK (true);
        DROP POLICY IF EXISTS auth_read_copilot_messages ON ai.copilot_messages;
        CREATE POLICY auth_read_copilot_messages ON ai.copilot_messages FOR SELECT TO authenticated USING (true);
    """)

    # ── Triggers ─────────────────────────────────────────────────────────
    op.execute("""
        DO $$ BEGIN
            CREATE TRIGGER set_updated_at_workflows
                BEFORE UPDATE ON ai.workflows
                FOR EACH ROW EXECUTE FUNCTION ai.trigger_set_updated_at();
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;

        DROP TRIGGER IF EXISTS set_updated_at_copilot_sessions ON ai.copilot_sessions;
        CREATE TRIGGER set_updated_at_copilot_sessions
            BEFORE UPDATE ON ai.copilot_sessions
            FOR EACH ROW EXECUTE FUNCTION ai.trigger_set_updated_at();
    """)

    # ── Grants ───────────────────────────────────────────────────────────
    op.execute("""
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.workflows TO authenticated, service_role;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.workflow_blocks TO authenticated, service_role;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.workflow_edges TO authenticated, service_role;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.workflow_executions TO authenticated, service_role;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.workflow_block_logs TO authenticated, service_role;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.project_settings TO authenticated, service_role;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.copilot_sessions TO authenticated, service_role;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.copilot_messages TO authenticated, service_role;
        GRANT SELECT ON ai.workflows TO anon;
        GRANT SELECT ON ai.workflow_blocks TO anon;
        GRANT SELECT ON ai.workflow_edges TO anon;
        GRANT SELECT ON ai.workflow_executions TO anon;
        GRANT SELECT ON ai.workflow_block_logs TO anon;
        GRANT SELECT ON ai.project_settings TO anon;
        GRANT SELECT ON ai.copilot_sessions TO anon;
        GRANT SELECT ON ai.copilot_messages TO anon;
    """)


def downgrade():
    # Don't drop — this is a backfill for tables that should have existed.
    pass
