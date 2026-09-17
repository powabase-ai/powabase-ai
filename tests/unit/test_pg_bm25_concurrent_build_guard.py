"""No bm25 index is built where a concurrent build is not known to be safe.

Stock pg_search 0.25.9 on Postgres 15 and 16 fails a bm25 ``CREATE INDEX
CONCURRENTLY`` while the table takes writes (XX000, the index left INVALID)
or crashes the server; paradedb/paradedb#6211 fixes it, and
``paradedb.version_info()`` cannot tell a build that has the fix from one that
does not. So before any build -- or a move, which only exists to be followed
by one -- the service asks the server whether a concurrent build is safe, and
when nothing says so it builds nothing, moves nothing, and drops no index that
still serves.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_project_service.routes import knowledge_bases as kb_route
from agentic_project_service.services import bm25_build_outcome
from agentic_project_service.services import pg_bm25_index as pgb
from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY
from agentic_project_service.tasks import indexing
from tests.unit.test_pg_bm25_lifecycle import (
    HEX,
    KB,
    _ddl,
    _FakeConn,
    _FakeEngine,
    _partition_ddl,
    _with_partition,
)

SERVICE = "agentic_project_service.services.pg_bm25_index"

_GERMAN = (
    f"CREATE INDEX bm25_chunks_{HEX} ON ai.chunks_kb_{HEX} USING bm25 "
    "(id, ((text)::pdb.simple('stemmer=german')), source_id, meta) WITH (key_field=id)"
)
_ENGLISH = _GERMAN.replace("german", "english")


@pytest.fixture(autouse=True)
def _fresh_warning_state(monkeypatch):
    monkeypatch.setattr(pgb, "_unsafe_build_warning_logged", False)
    monkeypatch.setattr(pgb, "_concurrent_build_override", lambda: False)


class _Session:
    """Answers the safety probe with one row."""

    def __init__(self, row):
        self.row = row
        self.statements: list[str] = []

    def execute(self, statement, params=None):
        self.statements.append(getattr(statement, "text", str(statement)))
        result = MagicMock()
        result.first.return_value = self.row
        return result

    def get_execution_options(self):
        return {"isolation_level": "AUTOCOMMIT"}


# ---------------------------------------------------------------------------
# What counts as safe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row, basis",
    [
        (("on", 150008, "0.25.9"), "marker"),
        (("ON", 160004, "0.25.9"), "marker"),
        ((None, 170002, "0.25.9"), "postgres"),
        ((None, 180000, None), "postgres"),
        ((None, 150008, "0.26.0"), "pg_search"),
        ((None, 150008, "0.26.0-rc.1"), "pg_search"),
        ((None, 160004, "1.2.3"), "pg_search"),
    ],
)
def test_each_safe_condition_on_its_own_is_enough(row, basis):
    safe, why = pgb.concurrent_build_safety(_Session(row))
    assert safe is True
    assert why.startswith(basis), why


@pytest.mark.parametrize(
    "row",
    [
        (None, 150008, "0.25.9"),
        ("off", 150008, "0.25.9"),
        ("", 160004, "0.25.9"),
        (None, 160004, None),
        (None, 150008, "garbled"),
    ],
)
def test_stock_pg_search_on_postgres_15_or_16_is_not_safe(row):
    safe, why = pgb.concurrent_build_safety(_Session(row))
    assert safe is False
    assert why == "unverified"


def test_the_setting_vouches_for_a_build_the_server_cannot_prove(monkeypatch):
    monkeypatch.setattr(pgb, "_concurrent_build_override", lambda: True)
    safe, why = pgb.concurrent_build_safety(_Session((None, 150008, "0.25.9")))
    assert (safe, why) == (True, "setting")


def test_the_probe_reads_the_marker_as_a_missing_ok_setting():
    """``current_setting(name, true)``: a server without the marker answers NULL
    instead of raising."""
    session = _Session((None, 150008, "0.25.9"))
    pgb.concurrent_build_safety(session)
    assert f"current_setting('{pgb.PG_SEARCH_CIC_SAFE_MARKER}', true)" in session.statements[0]
    assert pgb.PG_SEARCH_CIC_SAFE_MARKER == "powabase.pg_search_cic_safe"


def test_the_override_setting_is_registered_off_by_default():
    definition = SETTINGS_REGISTRY["BM25_PG_SEARCH_CONCURRENT_BUILD_SAFE"]
    assert definition.type == "bool"
    assert definition.default is False
    assert definition.advanced is True
    assert definition.category == "knowledge-retrieval"
    assert "6211" in definition.description
    assert pgb.CONCURRENT_BUILD_SAFE_SETTING == "BM25_PG_SEARCH_CONCURRENT_BUILD_SAFE"


def test_the_override_is_off_outside_an_application():
    assert pgb._read_concurrent_build_override() is False


@pytest.mark.parametrize(
    "row", [MagicMock(), (MagicMock(), MagicMock(), MagicMock()), ("on",), None]
)
def test_a_route_never_refuses_on_an_answer_it_cannot_read(row):
    """The request path refuses only on a clear "unsafe"; the task decides otherwise."""
    assert pgb.concurrent_build_known_unsafe(_Session(row)) is False


def test_a_route_refuses_on_a_clear_unsafe_answer():
    assert pgb.concurrent_build_known_unsafe(_Session((None, 150008, "0.25.9"))) is True
    assert pgb.concurrent_build_known_unsafe(_Session(("on", 150008, "0.25.9"))) is False


def test_a_failing_probe_never_refuses_on_the_request_path():
    session = MagicMock()
    session.execute.side_effect = RuntimeError("connection lost")
    assert pgb.concurrent_build_known_unsafe(session) is False


# ---------------------------------------------------------------------------
# ensure_bm25_index when a concurrent build is not safe
# ---------------------------------------------------------------------------


class _UnsafeConn(_FakeConn):
    def __init__(self, **kw):
        kw.setdefault("build_safe", False)
        super().__init__(**kw)


def test_an_unmoved_knowledge_base_is_neither_moved_nor_attached():
    conn = _UnsafeConn(attached=False)
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn), allow_row_move=True)
    assert out["status"] == "unavailable"
    assert out["reason"] == "concurrent_build_unsafe"
    assert _partition_ddl(conn) == []
    assert _ddl(conn) == []


def test_an_empty_knowledge_base_is_not_attached_either():
    """Its partition would retire its file index with no bm25 index to replace it."""
    conn = _UnsafeConn(attached=False, default_rows=False)
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["status"] == "unavailable"
    assert _partition_ddl(conn) == []


def test_a_partition_without_an_index_gets_no_build():
    conn = _UnsafeConn(relkinds=_with_partition())
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["status"] == "unavailable"
    assert _ddl(conn) == []


def test_a_working_index_is_not_dropped_for_a_language_change():
    """A ts_language PATCH on a moved knowledge base: the old index keeps serving."""
    conn = _UnsafeConn(relkinds=_with_partition(), indexdef=_ENGLISH)
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["status"] == "unavailable"
    assert _ddl(conn) == []


def test_an_invalid_index_is_left_for_a_server_that_can_rebuild_it():
    conn = _UnsafeConn(relkinds=_with_partition(), indexdef=_GERMAN, indisvalid=False)
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["status"] == "unavailable"
    assert _ddl(conn) == []


def test_a_matching_ready_index_is_still_reported_ready():
    conn = _UnsafeConn(relkinds=_with_partition(), indexdef=_GERMAN)
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["status"] == "ready"
    assert _ddl(conn) == []


def test_the_extension_and_config_skips_still_come_first():
    conn = _UnsafeConn(extension=False)
    assert pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))["reason"] == "extension_absent"
    conn = _UnsafeConn(kb_row=("chunk_embed", "vector_search", "german"))
    assert pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))["reason"] == "retrieval_method"


def test_the_warning_is_logged_once_per_process(caplog):
    with caplog.at_level(logging.INFO, logger=pgb.logger.name):
        for _ in range(3):
            pgb.ensure_bm25_index(KB, engine=_FakeEngine(_UnsafeConn(attached=False)))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    message = warnings[0].getMessage()
    assert "powabase.pg_search_cic_safe" in message
    assert "BM25_PG_SEARCH_CONCURRENT_BUILD_SAFE" in message


def test_a_safe_server_builds_as_before():
    conn = _FakeConn(relkinds=_with_partition())
    out = pgb.ensure_bm25_index(KB, engine=_FakeEngine(conn))
    assert out["status"] == "ready"
    assert _ddl(conn) == [pgb.bm25_index_ddl(KB, "chunks", "german")]


# ---------------------------------------------------------------------------
# The task and the outcome record
# ---------------------------------------------------------------------------


def test_unavailable_is_a_status_the_outcome_table_accepts():
    assert "unavailable" in bm25_build_outcome.STATUSES
    migration = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "0032_add_bm25_index_builds_table.py"
    ).read_text()
    assert "'unavailable'" in migration


def test_the_task_records_unavailable_with_an_actionable_reason_and_does_not_retry(
    monkeypatch, caplog
):
    recorded: list[tuple] = []
    monkeypatch.setattr(
        indexing,
        "record_bm25_build_outcome",
        lambda bind, kb, table, status, reason=None, attempts=None: recorded.append(
            (status, reason)
        ),
    )
    monkeypatch.setattr(indexing, "db", MagicMock())
    monkeypatch.setattr(pgb, "keyword_item_table", lambda bind, kb_id: "chunks")
    retry = MagicMock()
    monkeypatch.setattr(indexing.ensure_pg_bm25_index, "retry", retry)

    outcome = {"status": "unavailable", "reason": "concurrent_build_unsafe", "item_table": "chunks"}
    with caplog.at_level(logging.INFO), patch(f"{SERVICE}.ensure_bm25_index", return_value=outcome):
        indexing.ensure_pg_bm25_index.run(KB, allow_row_move=True)

    retry.assert_not_called()
    assert [status for status, _ in recorded] == ["queued", "unavailable"]
    reason = recorded[-1][1]
    assert "powabase.pg_search_cic_safe" in reason
    assert "BM25_PG_SEARCH_CONCURRENT_BUILD_SAFE" in reason
    assert "6211" in reason
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


# ---------------------------------------------------------------------------
# POST /build-bm25
# ---------------------------------------------------------------------------


def _make_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(kb_route.knowledge_bases_bp)
    return app


@patch("agentic_project_service.routes.knowledge_bases.db", new=MagicMock())
@patch(
    "agentic_project_service.auth.decode_jwt",
    return_value={"sub": "user-1", "role": "service_role"},
)
@patch("agentic_project_service.routes.knowledge_bases.ensure_pg_bm25_index")
@patch("agentic_project_service.routes.knowledge_bases.build_bm25_for_kb")
@patch.object(kb_route, "_keyword_index_backend", return_value="pg_search")
@patch("agentic_project_service.routes.knowledge_bases._fetch_kb_or_404")
@patch(f"{SERVICE}.concurrent_build_known_unsafe", return_value=True)
def test_build_bm25_refuses_with_409_and_says_why(
    _unsafe, mock_fetch, _backend, mock_bm25s, mock_ensure, _jwt
):
    kb_id = str(uuid.uuid4())
    mock_fetch.return_value = {
        "id": kb_id,
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"strategy": "chunk_embed"},
    }
    with _make_test_app().test_client() as client:
        resp = client.post(
            f"/api/knowledge-bases/{kb_id}/build-bm25", headers={"Authorization": "Bearer x"}
        )

    assert resp.status_code == 409
    body = resp.get_json()
    assert body["code"] == "pg_search_concurrent_build_unsafe"
    assert "powabase.pg_search_cic_safe" in body["error"]
    assert "BM25_PG_SEARCH_CONCURRENT_BUILD_SAFE" in body["error"]
    mock_ensure.delay.assert_not_called()
    mock_bm25s.delay.assert_not_called()
