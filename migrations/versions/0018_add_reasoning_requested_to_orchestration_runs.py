"""add reasoning_requested column to orchestration_runs

Revision ID: 0018
Revises: 0017_ai_provider_keys
Create Date: 2026-04-29

Issue #106 Bug B: orchestration replay must surface whether the orchestrator
requested reasoning so the FE pill renders after page refresh. Live-stream
fix already lands the flag on the SSE start event (orchestrations.py); this
migration persists it on the run row so the messages endpoint can return it.

Originally numbered 0017; bumped to 0018 to chain after the
``0017_ai_provider_keys`` migration that landed on main while #106 was in
flight (alembic refuses to upgrade with multiple heads on the same parent).
"""

from alembic import op


revision = "0018"
down_revision = "0017_ai_provider_keys"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE ai.orchestration_runs
            ADD COLUMN IF NOT EXISTS reasoning_requested BOOLEAN
    """)


def downgrade():
    op.execute("""
        ALTER TABLE ai.orchestration_runs
            DROP COLUMN IF EXISTS reasoning_requested
    """)
