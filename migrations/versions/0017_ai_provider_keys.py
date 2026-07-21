"""ai.ai_provider_keys table

Revision ID: 0017_ai_provider_keys
Revises: 0016
Create Date: 2026-04-23
"""

from alembic import op

revision = "0017_ai_provider_keys"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.ai_provider_keys (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            provider VARCHAR(50) NOT NULL UNIQUE,
            api_key_encrypted TEXT NOT NULL,
            is_valid BOOLEAN NOT NULL DEFAULT true,
            last_validated_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)


def downgrade():
    op.execute("DROP TABLE IF EXISTS ai.ai_provider_keys")
