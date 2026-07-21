"""Add message_citations table for tracking cited context items in agent run responses.

Revision ID: 0009
Revises: 0008b
Create Date: 2026-03-29

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0009"
down_revision = "0008b"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.message_citations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id UUID NOT NULL REFERENCES ai.agent_runs(id) ON DELETE CASCADE,
            citation_key SMALLINT NOT NULL,
            item_id UUID,
            source_id UUID REFERENCES ai.sources(id) ON DELETE SET NULL,
            text_excerpt TEXT,
            meta JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(run_id, citation_key)
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_message_citations_run_id ON ai.message_citations(run_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_message_citations_item_id ON ai.message_citations(item_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_message_citations_source_id ON ai.message_citations(source_id)"
    )


def downgrade():
    op.execute("DROP TABLE IF EXISTS ai.message_citations")
