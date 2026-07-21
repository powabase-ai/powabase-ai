"""add content_hash to ai.sources

Revision ID: 0021
Revises: 0020
Create Date: 2026-05-14

Adds a SHA-256 content hash column with a partial unique index for
duplicate-upload detection. Pre-existing rows remain NULL (no backfill).

Renumbered from 0020 -> 0021 (and filename suffix updated) to resolve a
dual-head conflict: this PR and the indexed_sources watchdog PR (#269) were
authored in parallel against revision 0019 and both shipped 0020. The
collision broke prod migrations on 2026-05-15 (alembic_version stuck at
0019, both new columns unapplied). Chained after watchdog's 0020 so the
sequence is now linear: 0019 -> 0020 (watchdog) -> 0021 (this).
"""

from alembic import op


revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE ai.sources ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS sources_content_hash_uniq "
        "ON ai.sources (content_hash) WHERE content_hash IS NOT NULL"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ai.sources_content_hash_uniq")
    op.execute("ALTER TABLE ai.sources DROP COLUMN IF EXISTS content_hash")
