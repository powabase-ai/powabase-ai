"""Add attention_required to extraction_status CHECK constraint.

Allows the extraction task to flag sources where non-OCR extraction
produced mostly blank pages, prompting the user to re-extract with OCR.

Revision ID: 0015
Revises: 0014
Create Date: 2026-04-16

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade():
    # Drop the old CHECK constraint and add an updated one.
    # Constraint name follows Postgres convention for inline CHECK on column.
    op.execute("""
        ALTER TABLE ai.sources
        DROP CONSTRAINT IF EXISTS sources_extraction_status_check
    """)
    op.execute("""
        ALTER TABLE ai.sources
        ADD CONSTRAINT sources_extraction_status_check
        CHECK (extraction_status IN (
            'pending', 'extracting', 'extracted',
            'attention_required', 'failed', 'cancelled'
        ))
    """)


def downgrade():
    # Revert any attention_required rows to extracted before tightening
    op.execute("""
        UPDATE ai.sources
        SET extraction_status = 'extracted'
        WHERE extraction_status = 'attention_required'
    """)
    op.execute("""
        ALTER TABLE ai.sources
        DROP CONSTRAINT IF EXISTS sources_extraction_status_check
    """)
    op.execute("""
        ALTER TABLE ai.sources
        ADD CONSTRAINT sources_extraction_status_check
        CHECK (extraction_status IN (
            'pending', 'extracting', 'extracted', 'failed', 'cancelled'
        ))
    """)
