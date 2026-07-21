"""Pin the migration 0024 DELETE filter — it must only touch the two
target keys, not bystanders like MISTRAL_API_KEY or OPENAI_API_KEY.

A typo in the `WHERE key IN (...)` clause would silently destroy
unrelated state across every project's `ai.project_settings` table.

Counterfactual: change the WHERE list to `('EXA_API_KEY', 'MISTRAL_API_KEY')`
→ this test fails.
"""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def migration_module():
    """Load 0024 directly by file path — alembic migrations aren't on
    sys.path and the file name has a leading digit.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "0024_drop_exa_firecrawl_project_settings.py"
    )
    spec = importlib.util.spec_from_file_location("mig_0024", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upgrade_targets_only_exa_and_firecrawl_keys(migration_module):
    """The DELETE WHERE clause must list exactly EXA_API_KEY and
    FIRECRAWL_API_KEY — nothing else.
    """
    captured: list[str] = []

    def fake_execute(sql, *args, **kwargs):
        captured.append(str(sql))

    with patch("alembic.op.execute", side_effect=fake_execute):
        migration_module.upgrade()

    assert len(captured) == 1, "Expected exactly one op.execute call in upgrade()"
    sql = captured[0]
    assert "DELETE FROM ai.project_settings" in sql
    # Bystander keys MUST NOT be in the WHERE clause.
    for bystander in (
        "MISTRAL_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "COHERE_API_KEY",
        "URL_IMPORT_MAX_PAGES",
        "FIRECRAWL_API_BASE",
    ):
        assert bystander not in sql, f"Migration would delete bystander key {bystander!r}"
    # The two targets MUST be in the WHERE clause.
    assert "EXA_API_KEY" in sql
    assert "FIRECRAWL_API_KEY" in sql


def test_downgrade_is_callable_noop(migration_module):
    """Downgrade is documented as no-op (deleted secrets are
    unrecoverable). Verify it's callable and runs no SQL.
    """
    captured: list[str] = []

    def fake_execute(sql, *args, **kwargs):
        captured.append(str(sql))

    with patch("alembic.op.execute", side_effect=fake_execute):
        migration_module.downgrade()  # must not raise

    assert captured == [], "downgrade() should run no SQL"


def test_revision_chains_off_0023(migration_module):
    """Pin the alembic chain — revision is `0024`, parent is `0023`.
    If this breaks, the PR has been rebased onto a different parent and
    coordination notes must be updated.
    """
    assert migration_module.revision == "0024"
    assert migration_module.down_revision == "0023"
