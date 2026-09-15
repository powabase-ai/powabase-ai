"""Unit specs for the pg_search index lifecycle and how it is wired up.

Covers ensure/drop against a fake engine, the Celery wrappers, the operator
endpoint, the KB create/patch/delete dispatch points and the bm25_status field.
"""

from __future__ import annotations

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


class _FakeConn:
    def __init__(self, *, extension=True, kb_row=("chunk_embed", "hybrid", "german"), indexdef=None, indisvalid=True):
        self.extension = extension
        self.kb_row = kb_row
        self.indexdef = indexdef
        self.indisvalid = indisvalid
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
        row = None
        if "pg_extension" in sql:
            row = (1,) if self.extension else None
        elif "knowledge_bases" in sql:
            row = self.kb_row
        elif "pg_get_indexdef" in sql:
            row = (self.indexdef,) if self.indexdef else None
        elif "indisvalid" in sql:
            row = (self.indisvalid,)
        result.first.return_value = row
        result.fetchone.return_value = row
        result.scalar.return_value = row[0] if row else None
        return result

    def commit(self):
        pass


class _FakeEngine:
    def __init__(self, conn):
        self.conn = conn

    def connect(self):
        return self.conn


def _ddl(conn) -> list[str]:
    return [s for s in conn.statements if "INDEX" in s and ("CREATE" in s or "DROP" in s)]


# ---------------------------------------------------------------------------
# ensure_bm25_index
# ---------------------------------------------------------------------------


def test_ensure_creates_the_index_on_an_autocommit_connection():
    conn = _FakeConn()
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["status"] == "ready"
    assert out["index"] == f"bm25_chunks_{HEX}"
    assert conn.options["isolation_level"] == "AUTOCOMMIT"
    assert _ddl(conn) == [pgb.bm25_index_ddl(KB, "chunks", "german")]


def test_ensure_reports_building_while_the_index_is_invalid():
    conn = _FakeConn(indisvalid=False)
    assert pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))["status"] == "building"


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
        f"CREATE INDEX bm25_chunks_{HEX} ON ai.chunks USING bm25 "
        "(id, ((text)::pdb.simple('stemmer=german')), source_id, meta) WITH (key_field=id)"
    )
    conn = _FakeConn(indexdef=existing)
    pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert _ddl(conn) == []


def test_ensure_recreates_the_index_when_the_language_changed():
    existing = (
        f"CREATE INDEX bm25_chunks_{HEX} ON ai.chunks USING bm25 "
        "(id, ((text)::pdb.simple('stemmer=english')), source_id, meta) WITH (key_field=id)"
    )
    conn = _FakeConn(indexdef=existing, kb_row=("chunk_embed", "hybrid", "german"))
    pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert _ddl(conn) == [
        pgb.bm25_drop_ddl(KB, "chunks"),
        pgb.bm25_index_ddl(KB, "chunks", "german"),
    ]


def test_ensure_refuses_a_non_uuid_kb_id():
    with pytest.raises(ValueError):
        pgb.ensure_bm25_index("not-a-uuid", engine=_FakeEngine(_FakeConn()))


def test_ensure_uses_the_strategys_text_column():
    conn = _FakeConn(kb_row=("full_document", "full_text", "english"))
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["item_table"] == "full_documents"
    assert "summary::pdb.simple('stemmer=english')" in _ddl(conn)[0]


# ---------------------------------------------------------------------------
# drop_bm25_index
# ---------------------------------------------------------------------------


def test_drop_removes_the_index_for_every_candidate_table():
    conn = _FakeConn()
    pgb.drop_bm25_index(KB, engine=_FakeEngine(conn))
    dropped = _ddl(conn)
    for item_table in pgb.BM25_ITEM_TABLES:
        assert pgb.bm25_drop_ddl(KB, item_table) in dropped


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


@pytest.mark.parametrize(
    ("indisvalid", "expected"), [(True, "ready"), (False, "building")]
)
def test_status_reflects_index_validity(indisvalid, expected):
    conn = _FakeConn(indisvalid=indisvalid)
    assert pgb.pg_bm25_status(KB, "chunk_embed", session=conn) == expected


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


def test_drop_task_delegates_to_the_service():
    from agentic_project_service.tasks.indexing import drop_pg_bm25_index

    with patch(
        "agentic_project_service.services.pg_bm25_index.drop_bm25_index",
        return_value={"status": "dropped"},
    ) as drop:
        assert drop_pg_bm25_index.run(KB) == {"status": "dropped"}
    drop.assert_called_once_with(KB)


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
        resp = client.post(f"/api/knowledge-bases/{KB}/build-bm25", headers={"Authorization": "Bearer x"})

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
        resp = client.post(f"/api/knowledge-bases/{KB}/build-bm25", headers={"Authorization": "Bearer x"})

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
@patch("agentic_project_service.routes.knowledge_bases.get_knowledge_base", return_value=("{}", 200))
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
@patch("agentic_project_service.routes.knowledge_bases.get_knowledge_base", return_value=("{}", 200))
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
@patch("agentic_project_service.routes.knowledge_bases.get_knowledge_base", return_value=("{}", 200))
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
    mock_db.session.execute.return_value = MagicMock(fetchone=lambda: None, __iter__=lambda self: iter([]))
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
