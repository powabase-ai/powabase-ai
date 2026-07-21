"""Tests for the alembic env.py orphan-version self-heal.

Background: PR #144 (April 29, 2026) renamed a migration revision from
``"0017"`` to ``"0018"`` to resolve a same-number collision with
``0017_ai_provider_keys``. Local dev DBs that were stamped at the
original ``"0017"`` BEFORE the rename keep the stale row in
``ai.alembic_version`` indefinitely. On the next project-api restart with
post-rename code, alembic refuses to advance ("Requested revision X
overlaps with other requested revisions Y").

Initial naive fix (commit f7294e9b) silently deleted the orphan row.
That turned out to be wrong: pre-rename DBs had ALSO skipped
``0017_ai_provider_keys``, so deleting the orphan claimed the chain was
complete while the ``ai.ai_provider_keys`` table was missing. Agent
runs then crashed with "relation does not exist".

Revised fix in this module:
  * Run the registered idempotent schema-coherence DDL for known
    orphans BEFORE deleting the row.
  * Raise ``UnknownOrphanVersion`` for unknown orphans rather than
    silent delete — surface as a correctness emergency the operator
    must triage by hand.

These tests cover both paths plus the original no-op states.
"""

import pytest

from agentic_project_service._alembic_self_heal import (
    KNOWN_ORPHANS,
    UnknownOrphanVersion,
    cleanup_orphan_versions,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, scalar_val=None, scalars_val=None):
        self._scalar = scalar_val
        self._scalars = scalars_val

    def scalar(self):
        return self._scalar

    def scalars(self):
        return self._scalars


class _FakeConnection:
    """Records every execute() call so tests can assert ordering of SQL
    operations (DDL apply BEFORE DELETE on the same orphan, for example)."""

    def __init__(self, *, version_table_exists: bool, version_nums: list[str]):
        self._version_table_exists = version_table_exists
        self._version_nums = list(version_nums)
        self.calls: list[tuple[str, dict | None]] = []
        self.deleted: list[str] = []
        self.ddl_executed: list[str] = []

    def execute(self, stmt, params=None):
        sql = str(stmt)
        sql_l = sql.lower().strip()
        self.calls.append((sql_l, params))
        if "information_schema.tables" in sql_l:
            return _FakeResult(scalar_val=self._version_table_exists)
        if "select version_num" in sql_l:
            return _FakeResult(scalars_val=list(self._version_nums))
        if sql_l.startswith("delete from ai.alembic_version"):
            assert params is not None and "v" in params
            self.deleted.append(params["v"])
            self._version_nums = [v for v in self._version_nums if v != params["v"]]
            return _FakeResult()
        # Anything else is the schema-coherence DDL.
        self.ddl_executed.append(sql)
        return _FakeResult()


class _FakeScriptDir:
    def __init__(self, valid_revs: list[str]):
        self._revs = list(valid_revs)
        self.walk_called = 0

    def walk_revisions(self):
        self.walk_called += 1
        from types import SimpleNamespace

        return [SimpleNamespace(revision=r) for r in self._revs]


# ---------------------------------------------------------------------------
# Known-orphan happy path (the April-29 case)
# ---------------------------------------------------------------------------


def test_known_orphan_0017_applies_schema_coherence_ddl_before_deleting():
    """The exact April-29 case: DB has '0017' (orphan known to be the
    pre-rename reasoning_requested revision) and '0019' (valid head).

    Expected behaviour: the registered DDL for '0017' runs FIRST (creating
    ai.ai_provider_keys), then the orphan row is deleted. Valid '0019'
    untouched."""
    conn = _FakeConnection(version_table_exists=True, version_nums=["0017", "0019"])
    sd = _FakeScriptDir(["0001", "0016", "0017_ai_provider_keys", "0018", "0019"])

    cleanup_orphan_versions(conn, sd)

    assert conn.deleted == ["0017"], conn.deleted
    # Exactly one DDL block ran (the registered one for 0017).
    assert len(conn.ddl_executed) == 1
    assert "ai_provider_keys" in conn.ddl_executed[0]
    assert "CREATE TABLE IF NOT EXISTS" in conn.ddl_executed[0]

    # Ordering: DDL apply MUST come before DELETE for that orphan.
    ddl_idx = next(
        i
        for i, (sql, _) in enumerate(conn.calls)
        if "ai_provider_keys" in sql and "create table" in sql
    )
    del_idx = next(
        i
        for i, (sql, _) in enumerate(conn.calls)
        if sql.startswith("delete from ai.alembic_version")
    )
    assert ddl_idx < del_idx, (
        f"schema-coherence DDL (call #{ddl_idx}) must run BEFORE "
        f"DELETE (call #{del_idx}) so a crash mid-cleanup never leaves "
        f"the version row gone but the table still missing"
    )


# ---------------------------------------------------------------------------
# Unknown-orphan: refuses to delete (the correctness emergency path)
# ---------------------------------------------------------------------------


def test_unknown_orphan_raises_without_modifying_anything():
    """An orphan version_num not in KNOWN_ORPHANS halts startup with
    ``UnknownOrphanVersion``. No DELETE, no DDL.

    This is the safeguard for the failure mode that f7294e9b introduced:
    silent deletion of an orphan whose schema impact we hadn't accounted
    for. Failing closed forces the operator to investigate."""
    conn = _FakeConnection(version_table_exists=True, version_nums=["mystery-rev", "0019"])
    sd = _FakeScriptDir(["0017_ai_provider_keys", "0018", "0019"])

    with pytest.raises(UnknownOrphanVersion) as excinfo:
        cleanup_orphan_versions(conn, sd)

    msg = str(excinfo.value)
    assert "mystery-rev" in msg
    assert "KNOWN_ORPHANS" in msg

    # Crucial: no row was deleted, no DDL was applied, before the raise.
    assert conn.deleted == []
    assert conn.ddl_executed == []


def test_unknown_orphan_raises_even_when_mixed_with_known_orphan():
    """Two orphans, one known + one unknown. We must NOT partially
    process the known one before raising — partial work is worse than
    no work when the operator is about to investigate."""
    conn = _FakeConnection(version_table_exists=True, version_nums=["0017", "unknown", "0019"])
    sd = _FakeScriptDir(["0017_ai_provider_keys", "0018", "0019"])

    with pytest.raises(UnknownOrphanVersion):
        cleanup_orphan_versions(conn, sd)

    assert conn.deleted == [], "must not delete the known orphan before raising on the unknown one"
    assert conn.ddl_executed == [], "must not run DDL before raising"


# ---------------------------------------------------------------------------
# No-op paths (production / CI / fresh local dev)
# ---------------------------------------------------------------------------


def test_noop_when_all_revs_have_matching_files():
    """Clean-state path: every DB revision matches a script. No DDL,
    no DELETE, no ScriptDirectory walk skipped. This is the hot path."""
    conn = _FakeConnection(version_table_exists=True, version_nums=["0019"])
    sd = _FakeScriptDir(["0017_ai_provider_keys", "0018", "0019"])

    cleanup_orphan_versions(conn, sd)

    assert conn.deleted == []
    assert conn.ddl_executed == []


def test_noop_when_version_table_does_not_exist():
    """Fresh DB: alembic_version not yet created. Function returns after
    the single EXISTS check; never reads version_nums or walks revs."""
    conn = _FakeConnection(version_table_exists=False, version_nums=[])
    sd = _FakeScriptDir(["0019"])

    cleanup_orphan_versions(conn, sd)

    assert len(conn.calls) == 1
    assert "information_schema.tables" in conn.calls[0][0]
    assert sd.walk_called == 0


def test_noop_when_version_table_is_empty():
    """alembic_version exists but has no rows. We bail before walking
    revisions, so script_dir.walk_revisions is never invoked."""
    conn = _FakeConnection(version_table_exists=True, version_nums=[])
    sd = _FakeScriptDir(["0019"])

    cleanup_orphan_versions(conn, sd)

    assert conn.deleted == []
    assert sd.walk_called == 0


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------


def test_known_orphans_registry_includes_april29_pre_rename_0017():
    """Regression guard against accidentally removing the '0017' entry
    from KNOWN_ORPHANS. If a future refactor drops this, every pre-April-29
    local dev env hits UnknownOrphanVersion on next project-api start."""
    assert "0017" in KNOWN_ORPHANS
    rationale, ddl = KNOWN_ORPHANS["0017"]
    assert "0017_ai_provider_keys" in rationale
    assert "ai_provider_keys" in ddl.lower()
    assert "if not exists" in ddl.lower()  # rerun-safe
