"""Add ai.list_sources_excluding_kb RPC function.

Used by the Studio "Add sources to knowledge base" modal to list
extracted sources that are NOT already linked to the given KB.
The previous frontend approach loaded every source row and filtered
client-side, which didn't scale past ~30k sources. This function
pushes the filter into Postgres: a NOT EXISTS subquery against the
``(knowledge_base_id, source_id)`` UNIQUE index on ``indexed_sources``
gives O(log N) lookups per source, fast even at hundreds of thousands.

The function returns the same column set the modal currently selects
from ``ai.sources``. Optional ``p_search`` is a case-insensitive
substring match on ``name``. Pagination is via ``p_limit``/``p_offset``;
defaults match the modal's current 50-per-page UX.

Revision ID: 0023
Revises: 0022
Create Date: 2026-06-03
"""

from alembic import op

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(r"""
        CREATE OR REPLACE FUNCTION ai.list_sources_excluding_kb(
            p_kb_id uuid,
            p_search text DEFAULT NULL,
            p_limit int DEFAULT 50,
            p_offset int DEFAULT 0
        )
        RETURNS TABLE (
            id uuid,
            name varchar(255),
            file_type varchar(255),
            storage_path varchar(1024),
            extraction_status varchar(50),
            derivatives jsonb,
            metadata jsonb,
            total_count bigint
        )
        LANGUAGE sql
        STABLE
        SECURITY DEFINER
        SET search_path = ai, pg_temp
        AS $func$
            WITH eligible AS (
                SELECT s.id, s.name, s.file_type, s.storage_path,
                       s.extraction_status, s.derivatives, s.metadata
                FROM ai.sources s
                WHERE s.extraction_status = 'extracted'
                  AND NOT EXISTS (
                      SELECT 1 FROM ai.indexed_sources i
                      WHERE i.source_id = s.id
                        AND i.knowledge_base_id = p_kb_id
                  )
                  AND (
                      p_search IS NULL
                      OR p_search = ''
                      -- Treat the search input as a literal substring.
                      -- ILIKE's % and _ metacharacters and the default
                      -- backslash escape are all escaped so a user typing
                      -- e.g. ``invoice_2024`` matches the literal
                      -- underscore instead of any-single-char. Backslash
                      -- replaced first so the % and _ replacements don't
                      -- recurse on themselves.
                      OR s.name ILIKE
                         '%' ||
                         replace(replace(replace(p_search,
                                                 E'\\', E'\\\\'),
                                         '%', E'\\%'),
                                 '_', E'\\_') ||
                         '%' ESCAPE E'\\'
                  )
            ),
            -- COUNT(*) OVER () computes per-row, so when ``eligible``
            -- yields zero rows the ``counted`` CTE also yields zero
            -- rows — i.e. the RPC returns NO row at all (not a single
            -- row with total_count=0). Callers must read total_count
            -- from rows[0] when present and treat empty result as zero.
            -- The Studio FE handles this at
            -- pages/project/[ref]/knowledge-bases/[kb_id].tsx:
            --   setAddSourceTotalCount(
            --       rows.length > 0 ? Number(rows[0].total_count) : 0
            --   );
            counted AS (
                SELECT *, COUNT(*) OVER ()::bigint AS total_count
                FROM eligible
            )
            SELECT id, name, file_type, storage_path, extraction_status,
                   derivatives, metadata, total_count
            FROM counted
            ORDER BY name ASC, id ASC
            LIMIT GREATEST(p_limit, 0)
            OFFSET GREATEST(p_offset, 0);
        $func$;
    """)
    # Authenticated only — anon never calls Studio APIs.
    op.execute("""
        GRANT EXECUTE ON FUNCTION ai.list_sources_excluding_kb(uuid, text, int, int)
            TO authenticated, service_role;
    """)


def downgrade():
    op.execute("""
        DROP FUNCTION IF EXISTS ai.list_sources_excluding_kb(uuid, text, int, int);
    """)
