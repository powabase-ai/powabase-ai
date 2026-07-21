"""Move system keys from metadata to auto_metadata.

source.metadata was conflating user-editable tags with BE-managed
system state (extraction_model, source_type, source_url). Move the
three reserved keys to auto_metadata so metadata is purely user-owned.

- extraction_model: upload preference, read by /reextract
- source_type: "url" discriminator, read by /reextract
- source_url: origin URL for URL sources, redundant with origin_url;
  migrated to origin_url in auto_metadata and dropped from metadata

Revision ID: 0015b
Revises: 0015
Create Date: 2026-04-19

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0015b"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade():
    # Order matters: backfill origin_url BEFORE dropping source_url from metadata.
    # Otherwise UPDATE #2 would read NULL because UPDATE #1 already stripped source_url.
    op.execute(r"""
        UPDATE ai.sources SET
          auto_metadata = COALESCE(auto_metadata, '{}'::jsonb) || jsonb_build_object('origin_url', metadata->>'source_url')
        WHERE metadata ? 'source_url' AND NOT (COALESCE(auto_metadata, '{}'::jsonb) ? 'origin_url');
    """)
    # Move extraction_model and source_type from metadata to auto_metadata; drop source_url from metadata.
    op.execute(r"""
        UPDATE ai.sources SET
          auto_metadata = COALESCE(auto_metadata, '{}'::jsonb) || jsonb_strip_nulls(jsonb_build_object(
            'extraction_model', metadata->>'extraction_model',
            'source_type', metadata->>'source_type'
          )),
          metadata = metadata - 'extraction_model' - 'source_type' - 'source_url'
        WHERE metadata ?| array['extraction_model', 'source_type', 'source_url'];
    """)


def downgrade():
    # Move extraction_model and source_type back to metadata; restore source_url from origin_url.
    # 'origin_url' must be in the WHERE array — pure URL sources (no extraction_model / source_type
    # in auto_metadata) would otherwise be skipped and lose the source_url restore.
    op.execute(r"""
        UPDATE ai.sources SET
          metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_strip_nulls(jsonb_build_object(
            'extraction_model', auto_metadata->>'extraction_model',
            'source_type', auto_metadata->>'source_type',
            'source_url', auto_metadata->>'origin_url'
          )),
          auto_metadata = auto_metadata - 'extraction_model' - 'source_type'
        WHERE COALESCE(auto_metadata, '{}'::jsonb) ?| array['extraction_model', 'source_type', 'origin_url'];
    """)
