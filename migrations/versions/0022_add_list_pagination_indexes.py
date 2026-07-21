"""Add composite indexes for paginated list endpoints

Revision ID: 0022
Revises: 0021
Create Date: 2026-05-21

Adds two composite indexes backing the new aggregate subselects in:
  - GET /api/knowledge-bases (source_counts rollup):
    indexed_sources(knowledge_base_id, index_status)
  - GET /api/orchestrations (last_run_at subselect):
    orchestration_runs(session_id, created_at DESC)

The KB / Agents / Orchestrations list aggregates also reference
agent_runs(agent_id, created_at DESC), orchestration_entities(orchestration_id),
and orchestration_sessions(orchestration_id). Those are already created by
ai_schema.sql under different names (idx_agent_runs_agent_created,
idx_orch_entities_orch_id, idx_orch_sessions_orch_id) — no need to duplicate.

Uses CREATE INDEX CONCURRENTLY to avoid blocking writes on busy tables.
CONCURRENTLY cannot run inside a transaction block; each statement runs
inside its own autocommit_block() context, consistent with the pattern
established by migration 0019. Failures of a single CONCURRENTLY call
leave the index in an INVALID state. To recover, run
`flask db downgrade -1` first (the down migration drops INVALID indexes
via DROP INDEX IF EXISTS), then `flask db upgrade` to recreate cleanly.
Simply re-running upgrade() will NOT work: the IF NOT EXISTS clause sees
the index name is taken (INVALID or not) and silently skips creation.
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def upgrade():
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "idx_indexed_sources_kb_status "
            "ON ai.indexed_sources (knowledge_base_id, index_status)"
        )
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "idx_orch_runs_session_created "
            "ON ai.orchestration_runs (session_id, created_at DESC)"
        )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ai.idx_indexed_sources_kb_status")
    op.execute("DROP INDEX IF EXISTS ai.idx_orch_runs_session_created")
