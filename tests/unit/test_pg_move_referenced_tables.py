"""The move takes the referenced tables' locks up front, without queueing.

Adding the partition's foreign keys takes SHARE ROW EXCLUSIVE on
``knowledge_bases``, ``sources`` and ``indexed_sources``. Waited for plainly,
behind one open write of an unrelated row there, it held every writer of the
item table off for the move's whole lock_timeout.
"""

from __future__ import annotations

from agentic_project_service.services import pg_bm25_index as pgb
from tests.unit.test_pg_bm25_lifecycle import _COLUMNS, KB, _FakeConn, _FakeEngine

_REFERENCED = ["ai.indexed_sources", "ai.knowledge_bases", "ai.sources"]


def _run_move(monkeypatch, **conn_kw):
    monkeypatch.setattr(pgb, "_referenced_relations", lambda conn, item_table: list(_REFERENCED))
    conn = _FakeConn(
        foreign_keys=("FOREIGN KEY (source_id) REFERENCES ai.sources(id) ON DELETE CASCADE",),
        **conn_kw,
    )
    pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn), allow_row_move=True)
    return conn


def _nowait_locks(conn, mode):
    return [
        i
        for i, s in enumerate(conn.statements)
        if s.startswith("LOCK TABLE") and f"IN {mode} MODE NOWAIT" in s
    ]


def test_the_move_locks_each_referenced_table_after_validating_and_before_adding_keys(
    monkeypatch,
):
    conn = _run_move(monkeypatch, moved=3)
    validate_at = conn.statements.index(pgb.default_move_check_validate_ddl(KB, "chunks"))
    first_key = next(i for i, s in enumerate(conn.statements) if "ADD FOREIGN KEY" in s)
    locks = [i for i in _nowait_locks(conn, "SHARE ROW EXCLUSIVE") if validate_at < i < first_key]
    assert [conn.statements[i] for i in locks] == [
        f"LOCK TABLE {relation} IN SHARE ROW EXCLUSIVE MODE NOWAIT" for relation in _REFERENCED
    ]


def test_the_pre_flight_probes_the_parent_and_the_referenced_tables_before_the_parent_lock(
    monkeypatch,
):
    conn = _run_move(monkeypatch, moved=3)
    parent_lock_at = conn.statements.index(pgb.partition_lock_parent_ddl("chunks"))
    insert_at = conn.statements.index(pgb.move_rows_sql(KB, "chunks", list(_COLUMNS))[0])
    assert parent_lock_at < insert_at
    probes = [s for s in conn.statements[:parent_lock_at] if s.startswith("LOCK TABLE")]
    assert probes[0] == 'LOCK TABLE ONLY "ai".chunks IN SHARE MODE NOWAIT'
    assert probes[1] == pgb.partition_lock_default_exclusive_ddl("chunks")
    assert probes[2:] == [
        f"LOCK TABLE {relation} IN SHARE ROW EXCLUSIVE MODE NOWAIT" for relation in _REFERENCED
    ]


def test_the_empty_attach_locks_the_referenced_tables_before_adding_keys(monkeypatch):
    conn = _run_move(monkeypatch, default_rows=False)
    first_key = next(i for i, s in enumerate(conn.statements) if "ADD FOREIGN KEY" in s)
    locks = [i for i in _nowait_locks(conn, "SHARE ROW EXCLUSIVE") if i < first_key]
    assert len(locks) == len(_REFERENCED)


def test_the_give_up_names_holders_of_the_referenced_tables():
    sql = pgb._lock_holders_sql()
    assert "confrelid" in sql and "contype = 'f'" in sql
