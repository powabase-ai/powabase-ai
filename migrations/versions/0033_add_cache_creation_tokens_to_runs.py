"""Add cache_creation_tokens (prompt-cache writes) to agent and orchestration runs.

agentic reports cache writes as ``cache_creation_tokens`` in a run's usage,
next to ``cached_tokens`` (cache reads). Anthropic bills writes at 1.25x input
and reads at 0.1x, so a run's net cache saving needs both.

Nullable with no default and no backfill: runs written before this revision
never stored the value, and NULL keeps "not reported" distinct from a reported
zero.

Revision ID: 0033
Revises: 0032
Create Date: 2026-09-23
"""

from alembic import op

revision = "0033"
down_revision = "0032"
branch_labels = None
depends_on = None

_TABLES = ("ai.agent_runs", "ai.orchestration_runs")


def upgrade():
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS cache_creation_tokens INT")


def downgrade():
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS cache_creation_tokens")
