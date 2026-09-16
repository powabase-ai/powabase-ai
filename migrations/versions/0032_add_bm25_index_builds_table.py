"""Add ai.bm25_index_builds — persisted outcome of a knowledge base's BM25 move/build.

The move from the DEFAULT partition into a knowledge base's own partition (and
the index build that follows it) can retry several times before it succeeds
or gives up. Nothing survived a restart of that bookkeeping until now: this
table lets the pg tasks record, per (knowledge_base_id, item_table), the
latest status (``queued``, ``moving``, ``building``, ``ready``, ``retrying``,
``failed``), the reason for a retry or failure, and how many attempts it took
-- so a KB's ``bm25_status`` can be reported even after the worker restarts.

Class-B (service-only), same posture as the other backend-only ``ai`` tables
(``ai.ai_provider_keys``, ``ai.message_citations``): RLS enabled with no
policy, so only ``service_role`` (which bypasses RLS) can reach it, and no
``authenticated``/``anon`` grants -- a fresh table receives none by default,
matching the schema-wide Class-B posture migration 0025 established.

Revision ID: 0032
Revises: 0031
Create Date: 2026-09-16
"""

from alembic import op
from sqlalchemy import text

revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    bind.execute(
        text("""
            CREATE TABLE IF NOT EXISTS ai.bm25_index_builds (
                knowledge_base_id UUID NOT NULL
                    REFERENCES ai.knowledge_bases(id) ON DELETE CASCADE,
                item_table TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK (status IN ('queued', 'moving', 'building', 'ready', 'retrying', 'failed')),
                reason TEXT,
                attempts INTEGER,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (knowledge_base_id, item_table)
            )
        """)
    )
    bind.execute(text("ALTER TABLE ai.bm25_index_builds ENABLE ROW LEVEL SECURITY"))


def downgrade():
    bind = op.get_bind()
    bind.execute(text("DROP TABLE IF EXISTS ai.bm25_index_builds CASCADE"))
