"""Migration 0034: run-wide citation units on ai.message_citations, and a
per-server citation mapping on ai.agent_mcp_servers.

Pins the SQL. The store tier proves it applies to a real database
(tests/test_message_citations_schema_store.py).
"""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def migration_0034():
    versions = Path(__file__).resolve().parents[2] / "migrations" / "versions"
    [path] = versions.glob("0034_*.py")
    spec = importlib.util.spec_from_file_location("mig_0034", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _captured_sql(fn) -> list[str]:
    captured: list[str] = []
    with patch("alembic.op.execute", side_effect=lambda sql, *a, **k: captured.append(str(sql))):
        fn()
    return captured


def test_follows_0033(migration_0034):
    assert migration_0034.revision == "0034"
    assert migration_0034.down_revision == "0033"


def test_upgrade_adds_the_citation_unit_columns(migration_0034):
    sql = _captured_sql(migration_0034.upgrade)
    prefix = "ALTER TABLE ai.message_citations ADD COLUMN IF NOT EXISTS "
    for column in (
        "kind TEXT NOT NULL DEFAULT 'kb_chunk'",
        "tool_name TEXT",
        "call_id TEXT",
        "title TEXT",
        "url TEXT",
        "knowledge_base_id UUID",
        "cited BOOLEAN NOT NULL DEFAULT true",
    ):
        assert prefix + column in sql


def test_upgrade_adds_a_nullable_mapping_to_mcp_servers(migration_0034):
    sql = _captured_sql(migration_0034.upgrade)
    assert (
        "ALTER TABLE ai.agent_mcp_servers ADD COLUMN IF NOT EXISTS citation_mapping JSONB" in sql
    )


@pytest.mark.parametrize("step", ["upgrade", "downgrade"])
def test_bounds_its_lock_wait(migration_0034, step):
    sql = _captured_sql(getattr(migration_0034, step))
    assert sql[0] == "SET LOCAL lock_timeout = '10s'"
    assert sql[-1] == "SET LOCAL lock_timeout TO DEFAULT"


def test_downgrade_first_removes_rows_the_old_readers_would_misreport(migration_0034):
    """Before run-wide keys every row was a cited knowledge-base chunk, and the
    old readers return every row as a citation."""
    sql = _captured_sql(migration_0034.downgrade)
    delete = "DELETE FROM ai.message_citations WHERE kind <> 'kb_chunk' OR NOT cited"
    assert delete in sql
    drops = [i for i, s in enumerate(sql) if "DROP COLUMN" in s]
    assert sql.index(delete) < min(drops)
    for column in ("kind", "tool_name", "call_id", "title", "url", "knowledge_base_id", "cited"):
        assert f"ALTER TABLE ai.message_citations DROP COLUMN IF EXISTS {column}" in sql
    assert "ALTER TABLE ai.agent_mcp_servers DROP COLUMN IF EXISTS citation_mapping" in sql
