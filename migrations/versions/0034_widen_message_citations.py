"""Widen ai.message_citations for run-wide citation units; add a per-server
citation mapping to ai.agent_mcp_servers.

A streaming ReAct run with citations enabled now registers every unit it can
cite under one run-wide key and persists all of them, flagging the ones its
answer cited. A unit is a knowledge-base chunk, an item inside a tool's JSON
result, or a whole tool result.

* ``kind`` is 'kb_chunk', 'tool_item' or 'tool_call'. Every existing row is a
  knowledge-base chunk, so the constant default backfills them.
* ``cited``: existing rows were written only when cited, so the constant
  default ``true`` backfills them. It stays the default because a pod still on
  the previous release during a rolling deploy inserts cited rows without
  naming the column. Current code always writes it explicitly.
* ``tool_name``, ``call_id``, ``title``, ``url`` and ``knowledge_base_id`` are
  nullable. ``source_id`` has been nullable since 0009.
* ``ai.agent_mcp_servers.citation_mapping`` is nullable JSONB. NULL means
  results from that server are never keyed.

Constant defaults keep every ADD COLUMN a catalog-only change (Postgres 11+),
with no table rewrite.

Revision ID: 0034
Revises: 0033
Create Date: 2026-10-06
"""

from alembic import op

revision = "0034"
down_revision = "0033"
branch_labels = None
depends_on = None

# Each ALTER takes ACCESS EXCLUSIVE, and migrations run at start-up; see 0033
# for why the wait is bounded.
_LOCK_TIMEOUT = "SET LOCAL lock_timeout = '10s'"
_LOCK_TIMEOUT_DEFAULT = "SET LOCAL lock_timeout TO DEFAULT"

_CITATION_COLUMNS = (
    ("kind", "TEXT NOT NULL DEFAULT 'kb_chunk'"),
    ("tool_name", "TEXT"),
    ("call_id", "TEXT"),
    ("title", "TEXT"),
    ("url", "TEXT"),
    ("knowledge_base_id", "UUID"),
    ("cited", "BOOLEAN NOT NULL DEFAULT true"),
)


def upgrade():
    op.execute(_LOCK_TIMEOUT)
    for name, definition in _CITATION_COLUMNS:
        op.execute(f"ALTER TABLE ai.message_citations ADD COLUMN IF NOT EXISTS {name} {definition}")
    op.execute("ALTER TABLE ai.agent_mcp_servers ADD COLUMN IF NOT EXISTS citation_mapping JSONB")
    op.execute(_LOCK_TIMEOUT_DEFAULT)


def downgrade():
    op.execute(_LOCK_TIMEOUT)
    # The previous readers return every row as a citation; drop the rows they
    # would misreport before the columns that distinguish them.
    op.execute("DELETE FROM ai.message_citations WHERE kind <> 'kb_chunk' OR NOT cited")
    for name, _definition in reversed(_CITATION_COLUMNS):
        op.execute(f"ALTER TABLE ai.message_citations DROP COLUMN IF EXISTS {name}")
    op.execute("ALTER TABLE ai.agent_mcp_servers DROP COLUMN IF EXISTS citation_mapping")
    op.execute(_LOCK_TIMEOUT_DEFAULT)
