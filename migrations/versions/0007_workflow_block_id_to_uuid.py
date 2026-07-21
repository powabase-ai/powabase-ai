"""Convert workflow block IDs from VARCHAR(255) to UUID.

Revision ID: 0007
Revises: 0006b
Create Date: 2026-03-25

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0007"
down_revision = "0006b"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        -- Convert workflow_blocks.id from VARCHAR(255) to UUID
        ALTER TABLE ai.workflow_blocks
            ALTER COLUMN id SET DATA TYPE UUID USING id::uuid;
        ALTER TABLE ai.workflow_blocks
            ALTER COLUMN id SET DEFAULT gen_random_uuid();

        -- Convert edge foreign key columns
        ALTER TABLE ai.workflow_edges
            ALTER COLUMN source_block_id SET DATA TYPE UUID USING source_block_id::uuid;
        ALTER TABLE ai.workflow_edges
            ALTER COLUMN target_block_id SET DATA TYPE UUID USING target_block_id::uuid;

        -- Convert block_logs reference
        ALTER TABLE ai.workflow_block_logs
            ALTER COLUMN block_id SET DATA TYPE UUID USING block_id::uuid;
    """)


def downgrade():
    op.execute("""
        -- Revert block_logs
        ALTER TABLE ai.workflow_block_logs
            ALTER COLUMN block_id SET DATA TYPE VARCHAR(255);

        -- Revert edge columns
        ALTER TABLE ai.workflow_edges
            ALTER COLUMN source_block_id SET DATA TYPE VARCHAR(255);
        ALTER TABLE ai.workflow_edges
            ALTER COLUMN target_block_id SET DATA TYPE VARCHAR(255);

        -- Revert workflow_blocks.id
        ALTER TABLE ai.workflow_blocks
            ALTER COLUMN id DROP DEFAULT;
        ALTER TABLE ai.workflow_blocks
            ALTER COLUMN id SET DATA TYPE VARCHAR(255);
    """)
