"""add parent_workflow_execution_id FK to agent_runs + agent_run_id on workflow_block_logs

Revision ID: 0016
Revises: 0015b
Create Date: 2026-04-21
"""

from alembic import op

revision = "0016"
down_revision = "0015b"
branch_labels = None
depends_on = None


def upgrade():
    # Add parent_workflow_execution_id to agent_runs (FK guarded — workflow_executions is defined
    # in ai_schema.sql, which may or may not be present in every environment).
    op.execute("""
        ALTER TABLE ai.agent_runs
            ADD COLUMN IF NOT EXISTS parent_workflow_execution_id UUID
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_runs_parent_wf_exec
            ON ai.agent_runs(parent_workflow_execution_id)
    """)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'ai' AND table_name = 'workflow_executions'
            ) AND NOT EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_name = 'fk_agent_runs_parent_wf_exec'
            ) THEN
                ALTER TABLE ai.agent_runs
                    ADD CONSTRAINT fk_agent_runs_parent_wf_exec
                    FOREIGN KEY (parent_workflow_execution_id)
                    REFERENCES ai.workflow_executions(id) ON DELETE SET NULL;
            END IF;
        END $$
    """)

    # Add agent_run_id to workflow_block_logs (guarded — table may not exist in every env).
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'ai' AND table_name = 'workflow_block_logs'
            ) THEN
                ALTER TABLE ai.workflow_block_logs
                    ADD COLUMN IF NOT EXISTS agent_run_id UUID;
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.table_constraints
                    WHERE constraint_name = 'fk_wf_block_logs_agent_run'
                ) THEN
                    ALTER TABLE ai.workflow_block_logs
                        ADD CONSTRAINT fk_wf_block_logs_agent_run
                        FOREIGN KEY (agent_run_id)
                        REFERENCES ai.agent_runs(id) ON DELETE SET NULL;
                END IF;
            END IF;
        END $$
    """)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'ai' AND table_name = 'workflow_block_logs'
                  AND column_name = 'agent_run_id'
            ) THEN
                CREATE INDEX IF NOT EXISTS idx_wf_block_logs_agent_run
                    ON ai.workflow_block_logs(agent_run_id);
            END IF;
        END $$
    """)


def downgrade():
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'ai' AND table_name = 'workflow_block_logs'
                  AND column_name = 'agent_run_id'
            ) THEN
                ALTER TABLE ai.workflow_block_logs
                    DROP CONSTRAINT IF EXISTS fk_wf_block_logs_agent_run;
                DROP INDEX IF EXISTS ai.idx_wf_block_logs_agent_run;
                ALTER TABLE ai.workflow_block_logs DROP COLUMN IF EXISTS agent_run_id;
            END IF;
        END $$
    """)
    op.execute("DROP INDEX IF EXISTS ai.idx_agent_runs_parent_wf_exec")
    op.execute("ALTER TABLE ai.agent_runs DROP CONSTRAINT IF EXISTS fk_agent_runs_parent_wf_exec")
    op.execute("ALTER TABLE ai.agent_runs DROP COLUMN IF EXISTS parent_workflow_execution_id")
