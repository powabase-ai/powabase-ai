"""Add missing indexes for orchestration tables.

Revision ID: 0011
Revises: 0010
Create Date: 2026-04-02

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_orch_sessions_orch_id"
        " ON ai.orchestration_sessions(orchestration_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_orch_runs_session_id ON ai.orchestration_runs(session_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_orch_runs_orch_id"
        " ON ai.orchestration_runs(orchestration_id);"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_orch_runs_status ON ai.orchestration_runs(status);")


def downgrade():
    op.execute("DROP INDEX IF EXISTS ai.idx_orch_sessions_orch_id;")
    op.execute("DROP INDEX IF EXISTS ai.idx_orch_runs_session_id;")
    op.execute("DROP INDEX IF EXISTS ai.idx_orch_runs_orch_id;")
    op.execute("DROP INDEX IF EXISTS ai.idx_orch_runs_status;")
