"""Revoke the inert anon SELECT grants on the project_copilot tables.

Migration 0027 granted ``SELECT ... TO anon`` on ai.project_copilot_sessions
and ai.project_copilot_messages alongside the authenticated/service_role
grants. Nothing uses them: the copilot routes go through the project-service
(service-role connection), and no RLS policy on either table targets anon —
so today the grant is inert. It is still a hazard: a future anon policy
(e.g. a copy-pasted ``USING (true)``) would silently expose the project's
copilot chat history to unauthenticated PostgREST callers. Drop the grants
so any future anon access has to be added deliberately.

Defensive — REVOKE is naturally idempotent (revoking an absent privilege is
a no-op), and 0027 guarantees both tables exist. The downgrade restores the
grants, returning to 0027's exact final state.

Revision ID: 0028
Revises: 0027
Create Date: 2026-07-16
"""

from alembic import op

revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        REVOKE SELECT ON ai.project_copilot_sessions FROM anon;
        REVOKE SELECT ON ai.project_copilot_messages FROM anon;
    """)


def downgrade():
    op.execute("""
        GRANT SELECT ON ai.project_copilot_sessions TO anon;
        GRANT SELECT ON ai.project_copilot_messages TO anon;
    """)
