"""Unit specs for the pg_search index lifecycle and how it is wired up.

Covers ensure/drop against a fake engine, the Celery wrappers, the operator
endpoint, the KB create/patch/delete dispatch points and the bm25_status field.
"""

from __future__ import annotations

import re
import uuid
from unittest.mock import MagicMock, patch

import pytest

from agentic_project_service.routes import knowledge_bases as kb_route
from agentic_project_service.services import pg_bm25_index as pgb

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
HEX = uuid.UUID(KB).hex


@pytest.fixture(autouse=True)
def _clear_caches():
    pgb.reset_pg_bm25_caches()
    yield
    pgb.reset_pg_bm25_caches()


# ---------------------------------------------------------------------------
# A fake engine that answers by statement shape
# ---------------------------------------------------------------------------


#: ``pg_class.relkind`` per relation: every item table already partitioned, its
#: DEFAULT partition in place, and no knowledge base given a partition yet.
_MIGRATED = {t: "p" for t in pgb.PARTITIONED_ITEM_TABLES} | {
    f"{t}_default": "r" for t in pgb.PARTITIONED_ITEM_TABLES
}
#: Every item table still a plain table -- migration 0031 has not run.
_UNMIGRATED = {t: "r" for t in pgb.PARTITIONED_ITEM_TABLES}


def _with_partition(kb_id=None, item_table="chunks"):
    """``_MIGRATED``, plus one knowledge base's partition."""
    return _MIGRATED | {pgb.partition_name(kb_id or KB, item_table): "r"}


class _FakeConn:
    def __init__(
        self,
        *,
        extension=True,
        kb_row=("chunk_embed", "hybrid", "german"),
        indexdef=None,
        indisvalid=True,
        relkinds=None,
        foreign_keys=("FOREIGN KEY (source_id) REFERENCES ai.sources(id) ON DELETE CASCADE",),
        evacuated=(0,),
        attached=True,
    ):
        self.extension = extension
        self.kb_row = kb_row
        self.indexdef = indexdef
        self.indisvalid = indisvalid
        self.relkinds = dict(_MIGRATED if relkinds is None else relkinds)
        self.foreign_keys = list(foreign_keys)
        self.evacuated = list(evacuated)
        self.attached = attached
        self.statements: list[str] = []
        self.options: dict = {}

    # engine.connect().execution_options(...) -> connection
    def execution_options(self, **kw):
        self.options.update(kw)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        sql = getattr(statement, "text", str(statement))
        self.statements.append(sql)
        result = MagicMock()
        result.rowcount = 0
        row = None
        rows: list = []
        if "pg_extension" in sql:
            row = (1,) if self.extension else None
        elif "knowledge_bases" in sql and "pg_class" not in sql:
            row = self.kb_row
        elif "pg_get_indexdef" in sql:
            row = (self.indexdef,) if self.indexdef else None
        elif "relkind" in sql:
            kind = self.relkinds.get((params or {}).get("relname"))
            row = (kind,) if kind else None
        elif "pg_inherits" in sql:
            row = (1,) if self.attached else None
        elif "pg_get_constraintdef" in sql:
            rows = [(fk,) for fk in self.foreign_keys]
        elif "indisvalid" in sql:
            row = (self.indisvalid,)
        elif "moved AS" in sql:
            result.rowcount = self.evacuated.pop(0) if self.evacuated else 0
        result.first.return_value = row
        result.fetchone.return_value = row
        result.scalar.return_value = row[0] if row else None
        result.all.return_value = rows
        return result

    def commit(self):
        pass


class _FakeEngine:
    def __init__(self, conn):
        self.conn = conn

    def connect(self):
        return self.conn

    def begin(self):
        return self.conn


def _ddl(conn) -> list[str]:
    """Index DDL only -- `INCLUDING INDEXES` in the clone is not index DDL."""
    return [
        s for s in conn.statements if re.match(r"\s*(CREATE|DROP) INDEX\b", s, flags=re.IGNORECASE)
    ]


#: Statement shapes that only appear while a partition is being built or torn
#: down: the clone, the row moves, ATTACH/DETACH, the settings mirror, the drop.
_PARTITION_WORK = ("PARTITION", "moved AS", "(LIKE ", "INSERT INTO", "DROP TABLE", "DO $$")


def _partition_ddl(conn) -> list[str]:
    return [s for s in conn.statements if any(marker in s for marker in _PARTITION_WORK)]


# ---------------------------------------------------------------------------
# ensure_bm25_index
# ---------------------------------------------------------------------------


def test_ensure_creates_the_index_on_an_autocommit_connection():
    conn = _FakeConn(relkinds=_with_partition())
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["status"] == "ready"
    assert out["index"] == f"bm25_chunks_{HEX}"
    assert out["partition"] == f"chunks_kb_{HEX}"
    assert conn.options["isolation_level"] == "AUTOCOMMIT"
    assert _ddl(conn) == [pgb.bm25_index_ddl(KB, "chunks", "german")]


def test_ensure_reports_building_while_the_index_is_invalid():
    conn = _FakeConn(relkinds=_with_partition(), indisvalid=False)
    assert pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))["status"] == "building"


# ---------------------------------------------------------------------------
# Partition creation, which is what makes a per-KB index possible
# ---------------------------------------------------------------------------


def test_ensure_creates_the_partition_before_indexing_it():
    """The KB's rows are in DEFAULT, so they have to move out first.

    A partition cannot be attached while the DEFAULT partition still holds a
    row that belongs to it, so the order is fixed: clone, add the local foreign
    keys, evacuate, attach, mirror ownership/grants/RLS -- all in one
    transaction -- and only then build the index (CONCURRENTLY, which cannot
    run inside one).
    """
    conn = _FakeConn(evacuated=(10_000, 4_000, 0))

    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))

    assert out["status"] == "ready"
    assert out["partition_created"] is True
    assert out["rows_moved"] == 14_000
    statements = _partition_ddl(conn)
    assert statements[0] == pgb.partition_create_ddl(KB, "chunks")
    assert statements[-1].strip().startswith("DO $$")
    assert pgb.partition_attach_ddl(KB, "chunks") in statements
    evacuations = [s for s in statements if "moved AS" in s]
    assert evacuations == [pgb.evacuate_batch_sql(KB, "chunks")] * 3
    # Attaching only happens once the evacuation has drained.
    assert statements.index(pgb.partition_attach_ddl(KB, "chunks")) > statements.index(
        evacuations[-1]
    )
    # And the index build comes after the partition exists.
    assert _ddl(conn) == [pgb.bm25_index_ddl(KB, "chunks", "german")]


def test_ensure_copies_the_default_partitions_foreign_keys_onto_the_new_one():
    """``LIKE`` carries the primary key and indexes across, never foreign keys."""
    conn = _FakeConn(
        foreign_keys=(
            "FOREIGN KEY (knowledge_base_id) REFERENCES ai.knowledge_bases(id) ON DELETE CASCADE",
            "FOREIGN KEY (source_id) REFERENCES ai.sources(id) ON DELETE CASCADE",
        )
    )

    pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))

    added = [s for s in conn.statements if "ADD FOREIGN KEY" in s]
    assert added == [
        f'ALTER TABLE "ai".chunks_kb_{HEX} ADD FOREIGN KEY (knowledge_base_id) '
        "REFERENCES ai.knowledge_bases(id) ON DELETE CASCADE",
        f'ALTER TABLE "ai".chunks_kb_{HEX} ADD FOREIGN KEY (source_id) '
        "REFERENCES ai.sources(id) ON DELETE CASCADE",
    ]


def test_ensure_does_not_recreate_a_partition_that_already_exists():
    conn = _FakeConn(relkinds=_with_partition())

    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))

    assert out.get("partition_created") is None
    assert _partition_ddl(conn) == []


def test_ensure_skips_a_table_the_migration_has_not_partitioned_yet():
    """Without a partition the only relation available is the parent, and a
    scored query against a partitioned parent is refused outright -- so this
    KB has to keep the existing keyword path rather than get a half-built one.
    """
    conn = _FakeConn(relkinds=_UNMIGRATED)

    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))

    assert out["status"] == "skipped"
    assert out["reason"] == "table_not_partitioned"
    assert _ddl(conn) == []
    assert _partition_ddl(conn) == []


def test_ensure_skips_a_strategy_whose_table_is_never_partitioned():
    conn = _FakeConn(kb_row=("doc2json", "hybrid", "english"))

    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))

    assert out["status"] == "skipped"
    assert out["reason"] == "table_not_partitionable"
    assert out["item_table"] == "doc2json_documents"
    assert _ddl(conn) == []


def test_ensure_skips_when_the_default_partition_is_missing():
    conn = _FakeConn(relkinds={t: "p" for t in pgb.PARTITIONED_ITEM_TABLES})

    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))

    assert out["status"] == "skipped"
    assert out["reason"] == "default_partition_absent"
    assert _partition_ddl(conn) == []


def test_ensure_refuses_to_evacuate_for_ever():
    """A writer inserting into DEFAULT faster than the batches drain it."""
    conn = _FakeConn(evacuated=[pgb.EVACUATION_BATCH_ROWS] * (pgb._MAX_EVACUATION_BATCHES + 1))

    with pytest.raises(RuntimeError, match="did not drain"):
        pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))


def test_ensure_is_a_no_op_without_the_extension():
    conn = _FakeConn(extension=False)
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out == {"status": "skipped", "reason": "extension_absent"}
    assert _ddl(conn) == []


def test_ensure_skips_a_kb_that_does_not_use_keyword_search():
    conn = _FakeConn(kb_row=("chunk_embed", "vector_search", "english"))
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["status"] == "skipped"
    assert out["reason"] == "retrieval_method"
    assert _ddl(conn) == []


def test_ensure_skips_a_strategy_with_no_keyword_item_table():
    conn = _FakeConn(kb_row=("page_index", "hybrid", "english"))
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["status"] == "skipped"
    assert out["reason"] == "strategy"
    assert _ddl(conn) == []


def test_ensure_skips_a_missing_kb():
    conn = _FakeConn(kb_row=None)
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out == {"status": "skipped", "reason": "kb_not_found"}


def test_ensure_leaves_a_matching_index_alone():
    existing = (
        f"CREATE INDEX bm25_chunks_{HEX} ON ai.chunks_kb_{HEX} USING bm25 "
        "(id, ((text)::pdb.simple('stemmer=german')), source_id, meta) WITH (key_field=id)"
    )
    conn = _FakeConn(relkinds=_with_partition(), indexdef=existing)
    pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert _ddl(conn) == []


def test_ensure_recreates_the_index_when_the_language_changed():
    existing = (
        f"CREATE INDEX bm25_chunks_{HEX} ON ai.chunks_kb_{HEX} USING bm25 "
        "(id, ((text)::pdb.simple('stemmer=english')), source_id, meta) WITH (key_field=id)"
    )
    conn = _FakeConn(
        relkinds=_with_partition(), indexdef=existing, kb_row=("chunk_embed", "hybrid", "german")
    )
    pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert _ddl(conn) == [
        pgb.bm25_drop_ddl(KB, "chunks"),
        pgb.bm25_index_ddl(KB, "chunks", "german"),
    ]
    # The partition is reused; only the index is rebuilt.
    assert _partition_ddl(conn) == []


def test_ensure_refuses_a_non_uuid_kb_id():
    with pytest.raises(ValueError):
        pgb.ensure_bm25_index("not-a-uuid", engine=_FakeEngine(_FakeConn()))


def test_ensure_uses_the_strategys_text_column():
    conn = _FakeConn(
        relkinds=_with_partition(KB, "full_documents"),
        kb_row=("full_document", "full_text", "english"),
    )
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["item_table"] == "full_documents"
    assert "summary::pdb.simple('stemmer=english')" in _ddl(conn)[0]
    assert out["partition"] == f"full_documents_kb_{HEX}"


# ---------------------------------------------------------------------------
# drop_bm25_index
# ---------------------------------------------------------------------------


def test_drop_removes_the_index_for_every_candidate_table():
    conn = _FakeConn()
    pgb.drop_bm25_index(KB, engine=_FakeEngine(conn))
    dropped = _ddl(conn)
    for item_table in pgb.BM25_ITEM_TABLES:
        assert pgb.bm25_drop_ddl(KB, item_table) in dropped


def test_drop_leaves_the_partition_in_place_by_default():
    conn = _FakeConn(relkinds=_with_partition())
    out = pgb.drop_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["partitions"] == []
    assert _partition_ddl(conn) == []


def test_drop_detaches_and_drops_the_partition_when_asked():
    """What a deleted knowledge base needs: no relation left behind.

    Any row still in the partition goes back to DEFAULT before the drop, so a
    partition teardown can never be the thing that loses a row.
    """
    conn = _FakeConn(relkinds=_with_partition())

    out = pgb.drop_bm25_index(KB, engine=_FakeEngine(conn), drop_partitions=True)

    assert out["partitions"] == [f"chunks_kb_{HEX}"]
    assert _partition_ddl(conn) == [
        pgb.partition_detach_ddl(KB, "chunks"),
        f'INSERT INTO "ai".chunks_default SELECT * FROM "ai".chunks_kb_{HEX}',
        pgb.partition_drop_ddl(KB, "chunks"),
    ]


def test_drop_skips_a_partition_that_is_not_there():
    conn = _FakeConn()
    out = pgb.drop_bm25_index(KB, engine=_FakeEngine(conn), drop_partitions=True)
    assert out["partitions"] == []
    assert _partition_ddl(conn) == []


def test_drop_of_a_detached_partition_still_rescues_its_rows():
    conn = _FakeConn(relkinds=_with_partition(), attached=False)

    pgb.drop_bm25_index(KB, engine=_FakeEngine(conn), drop_partitions=True)

    assert _partition_ddl(conn) == [
        f'INSERT INTO "ai".chunks_default SELECT * FROM "ai".chunks_kb_{HEX}',
        pgb.partition_drop_ddl(KB, "chunks"),
    ]


def test_drop_is_a_no_op_without_the_extension():
    conn = _FakeConn(extension=False)
    pgb.drop_bm25_index(KB, engine=_FakeEngine(conn))
    assert _ddl(conn) == []


# ---------------------------------------------------------------------------
# pg_bm25_status (used by the KB detail response)
# ---------------------------------------------------------------------------


def test_status_is_none_without_the_extension():
    conn = _FakeConn(extension=False)
    assert pgb.pg_bm25_status(KB, "chunk_embed", session=conn) is None


def test_status_is_none_for_a_strategy_without_a_keyword_table():
    conn = _FakeConn()
    assert pgb.pg_bm25_status(KB, "page_index", session=conn) is None


@pytest.mark.parametrize(("indisvalid", "expected"), [(True, "ready"), (False, "building")])
def test_status_reflects_index_validity(indisvalid, expected):
    conn = _FakeConn(indisvalid=indisvalid)
    assert pgb.pg_bm25_status(KB, "chunk_embed", session=conn) == expected


def test_status_is_absent_for_a_strategy_whose_table_is_never_partitioned():
    """doc2json keeps the fallback keyword path, so it never has a pg index."""
    assert pgb.pg_bm25_status(KB, "doc2json", session=_FakeConn()) == "absent"


def test_readiness_is_false_for_a_table_that_is_never_partitioned():
    assert pgb.bm25_index_ready(_FakeConn(), KB, "doc2json_documents") is False


def test_status_is_absent_when_there_is_no_index():
    class _NoIndex(_FakeConn):
        def execute(self, statement, params=None):
            sql = getattr(statement, "text", str(statement))
            self.statements.append(sql)
            result = MagicMock()
            row = (1,) if "pg_extension" in sql else None
            result.first.return_value = row
            result.fetchone.return_value = row
            return result

    assert pgb.pg_bm25_status(KB, "chunk_embed", session=_NoIndex()) == "absent"


def test_status_never_raises():
    session = MagicMock()
    session.execute.side_effect = RuntimeError("no app context")
    assert pgb.pg_bm25_status(KB, "chunk_embed", session=session) is None


# ---------------------------------------------------------------------------
# Celery wrappers
# ---------------------------------------------------------------------------


def test_ensure_task_delegates_to_the_service():
    from agentic_project_service.tasks.indexing import ensure_pg_bm25_index

    with patch(
        "agentic_project_service.services.pg_bm25_index.ensure_bm25_index",
        return_value={"status": "building"},
    ) as ensure:
        assert ensure_pg_bm25_index.run(KB) == {"status": "building"}
    ensure.assert_called_once_with(KB)


def test_drop_task_also_removes_the_partition():
    """Its only caller is KB deletion, where the relation must go too."""
    from agentic_project_service.tasks.indexing import drop_pg_bm25_index

    with patch(
        "agentic_project_service.services.pg_bm25_index.drop_bm25_index",
        return_value={"status": "dropped"},
    ) as drop:
        assert drop_pg_bm25_index.run(KB) == {"status": "dropped"}
    drop.assert_called_once_with(KB, drop_partitions=True)


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


def _make_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(kb_route.knowledge_bases_bp)
    return app


_FAKE_JWT = patch(
    "agentic_project_service.auth.decode_jwt",
    return_value={"sub": "user-1", "role": "service_role"},
)


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases.ensure_pg_bm25_index")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
@patch("agentic_project_service.routes.knowledge_bases._pg_search_available", return_value=True)
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
def test_build_endpoint_ensures_the_pg_index_when_available(
    mock_fetch, _avail, mock_bm25s, mock_ensure, _jwt
):
    mock_fetch.return_value = {
        "id": KB,
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    mock_ensure.delay.return_value = MagicMock(id="task-pg")

    with _make_test_app().test_client() as client:
        resp = client.post(
            f"/api/knowledge-bases/{KB}/build-bm25", headers={"Authorization": "Bearer x"}
        )

    assert resp.status_code == 202
    assert resp.get_json() == {"task_id": "task-pg", "knowledge_base_id": KB}
    mock_ensure.delay.assert_called_once_with(KB)
    mock_bm25s.delay.assert_not_called()


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases.ensure_pg_bm25_index")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
@patch("agentic_project_service.routes.knowledge_bases._pg_search_available", return_value=False)
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
def test_build_endpoint_keeps_the_file_index_path_without_the_extension(
    mock_fetch, _avail, mock_bm25s, mock_ensure, _jwt
):
    mock_fetch.return_value = {
        "id": KB,
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    mock_bm25s.delay.return_value = MagicMock(id="task-files")

    with _make_test_app().test_client() as client:
        resp = client.post(
            f"/api/knowledge-bases/{KB}/build-bm25", headers={"Authorization": "Bearer x"}
        )

    assert resp.status_code == 202
    mock_bm25s.delay.assert_called_once_with(KB)
    mock_ensure.delay.assert_not_called()


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases.ensure_pg_bm25_index")
@patch("agentic_project_service.routes.knowledge_bases.db", new_callable=MagicMock)
def test_creating_a_hybrid_kb_dispatches_an_index_build(mock_db, mock_ensure, _jwt):
    with _make_test_app().test_client() as client:
        resp = client.post(
            "/api/knowledge-bases",
            json={"name": "kb", "indexing_config": {"strategy": "chunk_embed"}},
            headers={"Authorization": "Bearer x"},
        )
    assert resp.status_code == 201
    mock_ensure.delay.assert_called_once_with(resp.get_json()["id"])


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases.ensure_pg_bm25_index")
@patch("agentic_project_service.routes.knowledge_bases.db", new_callable=MagicMock)
def test_creating_a_vector_only_kb_dispatches_nothing(mock_db, mock_ensure, _jwt):
    with _make_test_app().test_client() as client:
        resp = client.post(
            "/api/knowledge-bases",
            json={
                "name": "kb",
                "indexing_config": {"strategy": "chunk_embed"},
                "retrieval_config": {"method": "vector_search"},
            },
            headers={"Authorization": "Bearer x"},
        )
    assert resp.status_code == 201
    mock_ensure.delay.assert_not_called()


@_FAKE_JWT
@patch(
    "agentic_project_service.routes.knowledge_bases.get_knowledge_base", return_value=("{}", 200)
)
@patch("agentic_project_service.routes.knowledge_bases.get_setting", return_value=False)
@patch("agentic_project_service.routes.knowledge_bases.ensure_pg_bm25_index")
@patch("agentic_project_service.routes.knowledge_bases._read_existing_retrieval_config")
@patch("agentic_project_service.routes.knowledge_bases.db", new_callable=MagicMock)
def test_changing_ts_language_dispatches_a_rebuild(
    mock_db, mock_read, mock_ensure, _setting, _get, _jwt
):
    mock_read.return_value = {"method": "hybrid", "ts_language": "english"}
    with _make_test_app().test_client() as client:
        client.patch(
            f"/api/knowledge-bases/{KB}",
            json={"retrieval_config": {"method": "hybrid", "ts_language": "german"}},
            headers={"Authorization": "Bearer x"},
        )
    mock_ensure.delay.assert_called_once_with(KB)


@_FAKE_JWT
@patch(
    "agentic_project_service.routes.knowledge_bases.get_knowledge_base", return_value=("{}", 200)
)
@patch("agentic_project_service.routes.knowledge_bases.get_setting", return_value=False)
@patch("agentic_project_service.routes.knowledge_bases.ensure_pg_bm25_index")
@patch("agentic_project_service.routes.knowledge_bases._read_existing_retrieval_config")
@patch("agentic_project_service.routes.knowledge_bases.db", new_callable=MagicMock)
def test_switching_to_hybrid_dispatches_a_rebuild(
    mock_db, mock_read, mock_ensure, _setting, _get, _jwt
):
    mock_read.return_value = {"method": "vector_search"}
    with _make_test_app().test_client() as client:
        client.patch(
            f"/api/knowledge-bases/{KB}",
            json={"retrieval_config": {"method": "hybrid"}},
            headers={"Authorization": "Bearer x"},
        )
    mock_ensure.delay.assert_called_once_with(KB)


@_FAKE_JWT
@patch(
    "agentic_project_service.routes.knowledge_bases.get_knowledge_base", return_value=("{}", 200)
)
@patch("agentic_project_service.routes.knowledge_bases.get_setting", return_value=False)
@patch("agentic_project_service.routes.knowledge_bases.ensure_pg_bm25_index")
@patch("agentic_project_service.routes.knowledge_bases._read_existing_retrieval_config")
@patch("agentic_project_service.routes.knowledge_bases.db", new_callable=MagicMock)
def test_an_unchanged_retrieval_config_dispatches_nothing(
    mock_db, mock_read, mock_ensure, _setting, _get, _jwt
):
    mock_read.return_value = {"method": "hybrid", "ts_language": "german"}
    with _make_test_app().test_client() as client:
        client.patch(
            f"/api/knowledge-bases/{KB}",
            json={"retrieval_config": {"method": "hybrid", "ts_language": "german"}},
            headers={"Authorization": "Bearer x"},
        )
    mock_ensure.delay.assert_not_called()


@_FAKE_JWT
@patch("agentic_project_service.routes.knowledge_bases.drop_pg_bm25_index")
@patch("agentic_project_service.routes.knowledge_bases.db", new_callable=MagicMock)
def test_deleting_a_kb_drops_its_index(mock_db, mock_drop, _jwt):
    mock_db.session.execute.return_value = MagicMock(
        fetchone=lambda: None, __iter__=lambda self: iter([])
    )
    with _make_test_app().test_client() as client:
        resp = client.delete(f"/api/knowledge-bases/{KB}", headers={"Authorization": "Bearer x"})
    assert resp.status_code == 200
    mock_drop.delay.assert_called_once_with(KB)


# ---------------------------------------------------------------------------
# bm25_status reports the pg_search index when there is one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["absent", "building", "ready"])
@patch("agentic_project_service.routes.knowledge_bases.pg_bm25_status")
def test_bm25_status_reports_the_pg_index_state(mock_pg, state):
    mock_pg.return_value = state
    kb = {
        "id": KB,
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    assert kb_route._compute_bm25_status(kb) == state


@patch("agentic_project_service.routes.knowledge_bases.pg_bm25_status")
def test_bm25_status_is_still_none_for_a_vector_only_kb(mock_pg):
    kb = {
        "id": KB,
        "retrieval_config": {"method": "vector_search"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    assert kb_route._compute_bm25_status(kb) is None
    mock_pg.assert_not_called()
