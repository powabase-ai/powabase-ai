"""Add project_copilot_sessions and project_copilot_messages tables.

The Project Copilot is a project-scoped onboarding/guidance assistant (distinct
from the workflow-scoped ai.copilot_sessions). One resumable session per project;
messages may carry a ``guide_event`` recording a triggered guide-bubble sequence.

Revision ID: 0028
Revises: 0027
Create Date: 2026-06-26

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.project_copilot_sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS ai.project_copilot_messages (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID NOT NULL
                REFERENCES ai.project_copilot_sessions(id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            guide_event JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_project_copilot_messages_session
            ON ai.project_copilot_messages (session_id, created_at);

        -- Enforce the per-project singleton: at most one session row. Makes
        -- concurrent get-or-create converge instead of splitting chat history.
        CREATE UNIQUE INDEX IF NOT EXISTS project_copilot_sessions_singleton
            ON ai.project_copilot_sessions ((true));

        -- RLS
        ALTER TABLE ai.project_copilot_sessions ENABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.project_copilot_messages ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS service_role_all_project_copilot_sessions
            ON ai.project_copilot_sessions;
        CREATE POLICY service_role_all_project_copilot_sessions
            ON ai.project_copilot_sessions
            FOR ALL TO service_role USING (true) WITH CHECK (true);
        DROP POLICY IF EXISTS service_role_all_project_copilot_messages
            ON ai.project_copilot_messages;
        CREATE POLICY service_role_all_project_copilot_messages
            ON ai.project_copilot_messages
            FOR ALL TO service_role USING (true) WITH CHECK (true);
        DROP POLICY IF EXISTS auth_read_project_copilot_sessions
            ON ai.project_copilot_sessions;
        CREATE POLICY auth_read_project_copilot_sessions
            ON ai.project_copilot_sessions
            FOR SELECT TO authenticated USING (true);
        DROP POLICY IF EXISTS auth_read_project_copilot_messages
            ON ai.project_copilot_messages;
        CREATE POLICY auth_read_project_copilot_messages
            ON ai.project_copilot_messages
            FOR SELECT TO authenticated USING (true);

        -- Write policies (INSERT/UPDATE/DELETE) — keep in lockstep with
        -- ai_schema.sql so a retrofitted project matches a freshly-provisioned
        -- one (the GRANTs below already allow writes; without these policies a
        -- retrofitted project would silently differ under RLS).
        DROP POLICY IF EXISTS auth_write_project_copilot_sessions
            ON ai.project_copilot_sessions;
        CREATE POLICY auth_write_project_copilot_sessions
            ON ai.project_copilot_sessions
            FOR INSERT TO authenticated WITH CHECK (true);
        DROP POLICY IF EXISTS auth_update_project_copilot_sessions
            ON ai.project_copilot_sessions;
        CREATE POLICY auth_update_project_copilot_sessions
            ON ai.project_copilot_sessions
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
        DROP POLICY IF EXISTS auth_delete_project_copilot_sessions
            ON ai.project_copilot_sessions;
        CREATE POLICY auth_delete_project_copilot_sessions
            ON ai.project_copilot_sessions
            FOR DELETE TO authenticated USING (true);
        DROP POLICY IF EXISTS auth_write_project_copilot_messages
            ON ai.project_copilot_messages;
        CREATE POLICY auth_write_project_copilot_messages
            ON ai.project_copilot_messages
            FOR INSERT TO authenticated WITH CHECK (true);
        DROP POLICY IF EXISTS auth_update_project_copilot_messages
            ON ai.project_copilot_messages;
        CREATE POLICY auth_update_project_copilot_messages
            ON ai.project_copilot_messages
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
        DROP POLICY IF EXISTS auth_delete_project_copilot_messages
            ON ai.project_copilot_messages;
        CREATE POLICY auth_delete_project_copilot_messages
            ON ai.project_copilot_messages
            FOR DELETE TO authenticated USING (true);

        -- Grants
        GRANT SELECT, INSERT, UPDATE, DELETE
            ON ai.project_copilot_sessions TO authenticated;
        GRANT SELECT, INSERT, UPDATE, DELETE
            ON ai.project_copilot_sessions TO service_role;
        GRANT SELECT ON ai.project_copilot_sessions TO anon;
        GRANT SELECT, INSERT, UPDATE, DELETE
            ON ai.project_copilot_messages TO authenticated;
        GRANT SELECT, INSERT, UPDATE, DELETE
            ON ai.project_copilot_messages TO service_role;
        GRANT SELECT ON ai.project_copilot_messages TO anon;

        -- Updated-at trigger (ai.trigger_set_updated_at is defined in ai_schema.sql)
        DROP TRIGGER IF EXISTS set_updated_at_project_copilot_sessions
            ON ai.project_copilot_sessions;
        CREATE TRIGGER set_updated_at_project_copilot_sessions
            BEFORE UPDATE ON ai.project_copilot_sessions
            FOR EACH ROW EXECUTE FUNCTION ai.trigger_set_updated_at();
    """)


def downgrade():
    op.execute("""
        DROP TABLE IF EXISTS ai.project_copilot_messages CASCADE;
        DROP TABLE IF EXISTS ai.project_copilot_sessions CASCADE;
    """)
