"""Add project_settings, copilot_sessions, and copilot_messages tables.

Revision ID: 0008
Revises: 0007b
Create Date: 2026-03-25

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0008"
down_revision = "0007b"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.project_settings (
            key VARCHAR(255) PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMPTZ DEFAULT now()
        );

        ALTER TABLE ai.project_settings ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS service_role_all_project_settings ON ai.project_settings;
        CREATE POLICY service_role_all_project_settings ON ai.project_settings
            FOR ALL TO service_role USING (true) WITH CHECK (true);
        DROP POLICY IF EXISTS auth_read_project_settings ON ai.project_settings;
        CREATE POLICY auth_read_project_settings ON ai.project_settings
            FOR SELECT TO authenticated USING (true);

        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.project_settings TO authenticated;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.project_settings TO service_role;
        GRANT SELECT ON ai.project_settings TO anon;

        CREATE TABLE IF NOT EXISTS ai.copilot_sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workflow_id UUID NOT NULL REFERENCES ai.workflows(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_copilot_sessions_workflow
            ON ai.copilot_sessions (workflow_id);

        CREATE TABLE IF NOT EXISTS ai.copilot_messages (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID NOT NULL REFERENCES ai.copilot_sessions(id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            workflow_diff JSONB,
            pre_snapshot JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_copilot_messages_session
            ON ai.copilot_messages (session_id, created_at);

        -- RLS
        ALTER TABLE ai.copilot_sessions ENABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.copilot_messages ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS service_role_all_copilot_sessions ON ai.copilot_sessions;
        CREATE POLICY service_role_all_copilot_sessions ON ai.copilot_sessions
            FOR ALL TO service_role USING (true) WITH CHECK (true);
        DROP POLICY IF EXISTS service_role_all_copilot_messages ON ai.copilot_messages;
        CREATE POLICY service_role_all_copilot_messages ON ai.copilot_messages
            FOR ALL TO service_role USING (true) WITH CHECK (true);
        DROP POLICY IF EXISTS auth_read_copilot_sessions ON ai.copilot_sessions;
        CREATE POLICY auth_read_copilot_sessions ON ai.copilot_sessions
            FOR SELECT TO authenticated USING (true);
        DROP POLICY IF EXISTS auth_read_copilot_messages ON ai.copilot_messages;
        CREATE POLICY auth_read_copilot_messages ON ai.copilot_messages
            FOR SELECT TO authenticated USING (true);

        -- Grants
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.copilot_sessions TO authenticated;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.copilot_sessions TO service_role;
        GRANT SELECT ON ai.copilot_sessions TO anon;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.copilot_messages TO authenticated;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ai.copilot_messages TO service_role;
        GRANT SELECT ON ai.copilot_messages TO anon;

        -- Updated-at trigger
        DROP TRIGGER IF EXISTS set_updated_at_copilot_sessions ON ai.copilot_sessions;
        CREATE TRIGGER set_updated_at_copilot_sessions
            BEFORE UPDATE ON ai.copilot_sessions
            FOR EACH ROW EXECUTE FUNCTION ai.trigger_set_updated_at();
    """)


def downgrade():
    op.execute("""
        DROP TABLE IF EXISTS ai.copilot_messages CASCADE;
        DROP TABLE IF EXISTS ai.copilot_sessions CASCADE;
        DROP TABLE IF EXISTS ai.project_settings CASCADE;
    """)
