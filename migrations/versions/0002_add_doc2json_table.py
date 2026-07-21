"""Add doc2json_documents table for doc2json indexing strategy.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-15

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    # Create doc2json_documents table
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.doc2json_documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            indexed_source_id UUID REFERENCES ai.indexed_sources(id) ON DELETE CASCADE,
            knowledge_base_id UUID NOT NULL REFERENCES ai.knowledge_bases(id) ON DELETE CASCADE,
            source_id UUID NOT NULL REFERENCES ai.sources(id) ON DELETE CASCADE,
            summary TEXT NOT NULL,
            extracted_json JSONB NOT NULL,
            json_schema JSONB NOT NULL,
            window_summaries JSONB DEFAULT '[]',
            extraction_model VARCHAR(255),
            summary_tokens INTEGER,
            input_tokens INTEGER,
            window_size INTEGER,
            window_overlap INTEGER,
            window_count INTEGER,
            meta JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Create indexes
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_ai_doc2json_documents_kb_id
            ON ai.doc2json_documents (knowledge_base_id);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_ai_doc2json_documents_source_id
            ON ai.doc2json_documents (source_id);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_ai_doc2json_documents_indexed_source_id
            ON ai.doc2json_documents (indexed_source_id);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_ai_doc2json_documents_summary_search
            ON ai.doc2json_documents USING gin (to_tsvector('english', summary));
    """)

    # Enable RLS
    op.execute("ALTER TABLE ai.doc2json_documents ENABLE ROW LEVEL SECURITY;")

    # RLS policies
    op.execute("""
        DO $$ BEGIN
            CREATE POLICY service_role_all_doc2json_documents ON ai.doc2json_documents
                FOR ALL TO service_role USING (true) WITH CHECK (true);
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE POLICY auth_read_doc2json_documents ON ai.doc2json_documents
                FOR SELECT TO authenticated USING (true);
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)

    # Update embeddings check constraint to include doc2json_documents
    # (embeddings table may not exist in projects created before it was added to ai_schema.sql)
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'ai' AND table_name = 'embeddings') THEN
                ALTER TABLE ai.embeddings DROP CONSTRAINT IF EXISTS embeddings_item_table_check;
                ALTER TABLE ai.embeddings ADD CONSTRAINT embeddings_item_table_check
                    CHECK (item_table IN ('chunks', 'graph_index_nodes', 'full_documents', 'doc2json_documents'));
            END IF;
        END $$;
    """)


def downgrade():
    # Revert embeddings check constraint
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'ai' AND table_name = 'embeddings') THEN
                ALTER TABLE ai.embeddings DROP CONSTRAINT IF EXISTS embeddings_item_table_check;
                ALTER TABLE ai.embeddings ADD CONSTRAINT embeddings_item_table_check
                    CHECK (item_table IN ('chunks', 'graph_index_nodes', 'full_documents'));
            END IF;
        END $$;
    """)

    # Drop RLS policies
    op.execute("DROP POLICY IF EXISTS auth_read_doc2json_documents ON ai.doc2json_documents;")
    op.execute(
        "DROP POLICY IF EXISTS service_role_all_doc2json_documents ON ai.doc2json_documents;"
    )

    # Drop indexes
    op.execute("DROP INDEX IF EXISTS ai.idx_ai_doc2json_documents_summary_search;")
    op.execute("DROP INDEX IF EXISTS ai.idx_ai_doc2json_documents_indexed_source_id;")
    op.execute("DROP INDEX IF EXISTS ai.idx_ai_doc2json_documents_source_id;")
    op.execute("DROP INDEX IF EXISTS ai.idx_ai_doc2json_documents_kb_id;")

    # Drop table
    op.execute("DROP TABLE IF EXISTS ai.doc2json_documents;")
