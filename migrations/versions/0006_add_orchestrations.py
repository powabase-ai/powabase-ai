"""Add orchestration tables and FK on agent_runs.

Revision ID: 0006b
Revises: 0006
Create Date: 2026-03-27

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0006b"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade():
    # Orchestrations table
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.orchestrations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            strategy VARCHAR(50) NOT NULL DEFAULT 'supervisor',
            orchestrator_config JSONB DEFAULT '{}',
            settings JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Orchestration entities table
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.orchestration_entities (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            orchestration_id UUID NOT NULL REFERENCES ai.orchestrations(id) ON DELETE CASCADE,
            entity_type VARCHAR(50) NOT NULL,
            entity_ref_id UUID NOT NULL,
            role_description TEXT,
            config JSONB DEFAULT '{}',
            position INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_orch_entities_orch_id
            ON ai.orchestration_entities(orchestration_id);
    """)

    # Orchestration sessions table
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.orchestration_sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id VARCHAR(255) UNIQUE NOT NULL,
            orchestration_id UUID REFERENCES ai.orchestrations(id) ON DELETE SET NULL,
            user_id UUID,
            session_data JSONB DEFAULT '{}',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Orchestration runs table
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.orchestration_runs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES ai.orchestration_sessions(id) ON DELETE CASCADE,
            run_id VARCHAR(255) UNIQUE NOT NULL,
            orchestration_id UUID REFERENCES ai.orchestrations(id) ON DELETE SET NULL,
            status VARCHAR(50) DEFAULT 'running',
            input_messages JSONB,
            content TEXT,
            events JSONB DEFAULT '[]',
            usage JSONB,
            error TEXT,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # FK constraint on agent_runs.parent_orchestration_run_id
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_name = 'fk_agent_runs_parent_orch_run'
            ) THEN
                ALTER TABLE ai.agent_runs
                    ADD CONSTRAINT fk_agent_runs_parent_orch_run
                    FOREIGN KEY (parent_orchestration_run_id)
                    REFERENCES ai.orchestration_runs(id) ON DELETE SET NULL;
            END IF;
        END $$;
    """)


def downgrade():
    # Drop FK constraint on agent_runs first
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_name = 'fk_agent_runs_parent_orch_run'
            ) THEN
                ALTER TABLE ai.agent_runs
                    DROP CONSTRAINT fk_agent_runs_parent_orch_run;
            END IF;
        END $$;
    """)

    # Drop tables in reverse dependency order
    op.execute("DROP TABLE IF EXISTS ai.orchestration_runs CASCADE;")
    op.execute("DROP TABLE IF EXISTS ai.orchestration_sessions CASCADE;")
    op.execute("DROP TABLE IF EXISTS ai.orchestration_entities CASCADE;")
    op.execute("DROP TABLE IF EXISTS ai.orchestrations CASCADE;")
