"""Prompt-cache writes (`cache_creation_tokens`) survive the usage round-trip.

agentic reports cache writes as `cache_creation_tokens` in `AgentOutput.usage`,
next to `cached_tokens` (cache reads). Anthropic bills writes at 1.25x input,
so a run's net cache saving needs both. These tests pin the unpack (usage dict
-> typed columns), the pack (typed columns -> usage dict) and migration 0033
that adds the column.

A missing key stays NULL rather than becoming 0, matching `cached_tokens`.
agentic's run totals start every key at 0, so runs recorded through agentic
store 0 even when the provider reports no writes; NULL marks rows written
before the column existed, or usage that did not come from agentic's totals.
"""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from agentic_project_service.models.tenant import _pack_usage_from_attrs
from agentic_project_service.services.session import _unpack_usage


class TestUnpackUsage:
    def test_flat_key_is_unpacked(self):
        tokens = _unpack_usage(
            {"prompt_tokens": 100, "cached_tokens": 40, "cache_creation_tokens": 50}
        )
        assert tokens["cache_creation_tokens"] == 50

    def test_nested_prompt_tokens_details_key_is_unpacked(self):
        tokens = _unpack_usage(
            {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 40, "cache_creation_tokens": 50},
            }
        )
        assert tokens["cache_creation_tokens"] == 50
        assert tokens["cached_tokens"] == 40

    def test_flat_key_wins_over_nested(self):
        tokens = _unpack_usage(
            {
                "cache_creation_tokens": 7,
                "prompt_tokens_details": {"cache_creation_tokens": 50},
            }
        )
        assert tokens["cache_creation_tokens"] == 7

    def test_zero_is_kept_as_zero(self):
        tokens = _unpack_usage({"prompt_tokens": 100, "cache_creation_tokens": 0})
        assert tokens["cache_creation_tokens"] == 0

    def test_absent_key_is_none(self):
        tokens = _unpack_usage({"prompt_tokens": 100, "cached_tokens": 40})
        assert tokens["cache_creation_tokens"] is None

    @pytest.mark.parametrize("usage", [None, {}])
    def test_empty_usage_is_none(self, usage):
        assert _unpack_usage(usage)["cache_creation_tokens"] is None


class TestPackUsageFromAttrs:
    def _pack(self, **overrides):
        values = {
            "prompt": None,
            "completion": None,
            "reasoning": None,
            "cached": None,
            "cache_creation": None,
            "total": None,
        }
        values.update(overrides)
        return _pack_usage_from_attrs(**values)

    def test_key_is_packed(self):
        usage = self._pack(prompt=100, cached=40, cache_creation=50, total=120)
        assert usage == {
            "prompt_tokens": 100,
            "cached_tokens": 40,
            "cache_creation_tokens": 50,
            "total_tokens": 120,
        }

    def test_null_key_is_omitted(self):
        usage = self._pack(prompt=100, total=120)
        assert "cache_creation_tokens" not in usage

    def test_only_cache_creation_set_is_not_none(self):
        assert self._pack(cache_creation=50) == {"cache_creation_tokens": 50}

    def test_all_null_is_none(self):
        assert self._pack() is None


@pytest.fixture
def migration_0033():
    """Load 0033 by file path: alembic migrations aren't on sys.path and the
    file name has a leading digit.
    """
    versions = Path(__file__).resolve().parents[2] / "migrations" / "versions"
    [path] = versions.glob("0033_*.py")
    spec = importlib.util.spec_from_file_location("mig_0033", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _captured_sql(fn) -> str:
    captured: list[str] = []
    with patch("alembic.op.execute", side_effect=lambda sql, *a, **k: captured.append(str(sql))):
        fn()
    return "\n".join(captured)


def test_migration_0033_follows_0032(migration_0033):
    assert migration_0033.revision == "0033"
    assert migration_0033.down_revision == "0032"


def test_migration_0033_upgrade_adds_nullable_column_to_both_run_tables(migration_0033):
    sql = _captured_sql(migration_0033.upgrade)
    for table in ("ai.agent_runs", "ai.orchestration_runs"):
        assert f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS cache_creation_tokens INT" in sql
    alters = [s for s in sql.splitlines() if s.startswith("ALTER TABLE")]
    assert len(alters) == 2
    for alter in alters:
        assert "NOT NULL" not in alter
        assert "DEFAULT" not in alter


@pytest.mark.parametrize("step", ["upgrade", "downgrade"])
def test_migration_0033_bounds_its_lock_wait(migration_0033, step):
    """Each ALTER needs ACCESS EXCLUSIVE, and migrations run at start-up: behind
    a long reader, an unbounded wait would hang the boot and queue every run
    insert behind it. The bound is set before the ALTERs and put back after.
    """
    sql = _captured_sql(getattr(migration_0033, step)).splitlines()
    alters = [i for i, s in enumerate(sql) if s.startswith("ALTER TABLE")]
    assert sql[0] == "SET LOCAL lock_timeout = '10s'"
    assert sql[-1] == "SET LOCAL lock_timeout TO DEFAULT"
    assert 0 < min(alters) and max(alters) < len(sql) - 1


def test_migration_0033_downgrade_drops_column_from_both_run_tables(migration_0033):
    sql = _captured_sql(migration_0033.downgrade)
    for table in ("ai.agent_runs", "ai.orchestration_runs"):
        assert f"ALTER TABLE {table} DROP COLUMN IF EXISTS cache_creation_tokens" in sql
