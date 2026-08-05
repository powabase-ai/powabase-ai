"""Add sources.error_code — machine-readable failure classification.

error_message stays the human/raw text; error_code is a stable enum-like
string clients can branch on: rate_limited | transient | timeout
(retryable) vs permanent | no_content | internal (not retryable).

Revision ID: 0026
Revises: 0025
Create Date: 2026-08-04
"""

from alembic import op

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE ai.sources ADD COLUMN IF NOT EXISTS error_code TEXT;")


def downgrade():
    op.execute("ALTER TABLE ai.sources DROP COLUMN IF EXISTS error_code;")
