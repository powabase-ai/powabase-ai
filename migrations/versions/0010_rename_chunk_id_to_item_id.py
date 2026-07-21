"""Rename chunk_id to item_id in message_citations and drop FK constraint.

Items can come from any of 5 content tables (chunks, graph_index_nodes,
page_index_nodes, full_documents, doc2json_documents), so a single FK
to ai.chunks is incorrect.

On new installs, the column is already named item_id (from 0009), so
the rename is guarded with a column existence check.

Revision ID: 0010
Revises: 0009
Create Date: 2026-03-30

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade():
    # Only rename if the old column name exists (production).
    # On new installs, 0009 already creates the column as item_id.
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'ai'
                  AND table_name = 'message_citations'
                  AND column_name = 'chunk_id'
            ) THEN
                ALTER TABLE ai.message_citations
                    DROP CONSTRAINT IF EXISTS message_citations_chunk_id_fkey;
                ALTER TABLE ai.message_citations
                    RENAME COLUMN chunk_id TO item_id;
                DROP INDEX IF EXISTS ai.idx_message_citations_chunk_id;
                CREATE INDEX IF NOT EXISTS idx_message_citations_item_id
                    ON ai.message_citations(item_id);
            END IF;
        END
        $$;
    """)


def downgrade():
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'ai'
                  AND table_name = 'message_citations'
                  AND column_name = 'item_id'
            ) THEN
                DROP INDEX IF EXISTS ai.idx_message_citations_item_id;
                ALTER TABLE ai.message_citations
                    RENAME COLUMN item_id TO chunk_id;
                CREATE INDEX IF NOT EXISTS idx_message_citations_chunk_id
                    ON ai.message_citations(chunk_id);
            END IF;
        END
        $$;
    """)
