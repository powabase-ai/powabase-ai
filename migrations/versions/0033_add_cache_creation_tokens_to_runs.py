"""Add cache_creation_tokens (prompt-cache writes) to agent and orchestration runs.

agentic reports cache writes as ``cache_creation_tokens`` in a run's usage,
next to ``cached_tokens`` (cache reads). Anthropic bills writes at 1.25x input
and reads at 0.1x, so a run's net cache saving needs both.

Nullable with no default and no backfill: runs written before this revision
never stored the value, so they stay NULL.

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

# Each ALTER takes ACCESS EXCLUSIVE, and migrations run at start-up. Behind a
# long reader (a nightly pg_dump holds ACCESS SHARE for its whole run) an
# unbounded wait would hang the boot and queue every run insert behind the
# waiting ALTER. On timeout the migration fails, start-up exits, and the next
# start runs this revision again. Both settings are transaction-local, so the
# session's own lock_timeout returns at commit.
_LOCK_TIMEOUT = "SET LOCAL lock_timeout = '10s'"
_LOCK_TIMEOUT_DEFAULT = "SET LOCAL lock_timeout TO DEFAULT"


def upgrade():
    op.execute(_LOCK_TIMEOUT)
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS cache_creation_tokens INT")
    op.execute(_LOCK_TIMEOUT_DEFAULT)


def downgrade():
    op.execute(_LOCK_TIMEOUT)
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS cache_creation_tokens")
    op.execute(_LOCK_TIMEOUT_DEFAULT)
