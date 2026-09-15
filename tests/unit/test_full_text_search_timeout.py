"""full_text_search runs under a savepoint-scoped statement_timeout and turns a
cancelled statement into KeywordSearchTimeout."""

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from agentic_project_service.services import base_vector_store as bvs


class _FakeStore(bvs.BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


class _Canceled(Exception):
    sqlstate = "57014"


class _OtherPgError(Exception):
    sqlstate = "53300"


def _spy_session(search_error: Exception | None = None):
    session = MagicMock()
    log: list[tuple[str, dict | None]] = []

    @contextmanager
    def begin_nested():
        log.append(("SAVEPOINT", None))
        try:
            yield
        except Exception:
            log.append(("ROLLBACK TO SAVEPOINT", None))
            raise
        log.append(("RELEASE SAVEPOINT", None))

    def execute(stmt, params=None):
        sql = stmt.text if hasattr(stmt, "text") else str(stmt)
        log.append((sql, params))
        result = MagicMock()
        if "current_setting('statement_timeout')" in sql:
            result.scalar.return_value = "0"
        elif "corpus_stats" in sql:
            if search_error is not None:
                raise search_error
            result.fetchall.return_value = []
        return result

    session.begin_nested = begin_nested
    session.execute = execute
    return session, log


def _run(session):
    store = _FakeStore(db_session=session, knowledge_base_id="kb-1")
    return asyncio.run(store.full_text_search("anything", top_k=5))


@patch.object(bvs, "_bm25_fallback_timeout_ms", return_value=4321)
def test_search_runs_inside_savepoint_with_timeout_then_restores(_ms):
    session, log = _spy_session()
    assert _run(session) == []

    sqls = [s for s, _ in log]
    sp = sqls.index("SAVEPOINT")
    search = next(i for i, s in enumerate(sqls) if "corpus_stats" in s)
    release = sqls.index("RELEASE SAVEPOINT")
    set_calls = [(i, p) for i, (s, p) in enumerate(log) if "set_config('statement_timeout'" in s]

    assert len(set_calls) == 2
    (set_i, set_p), (restore_i, restore_p) = set_calls
    assert sp < set_i < search < restore_i < release
    assert set_p == {"ms": "4321"}
    assert restore_p == {"ms": "0"}


@patch.object(bvs, "_bm25_fallback_timeout_ms", return_value=4321)
def test_query_canceled_becomes_keyword_search_timeout(_ms):
    err = OperationalError("SELECT ...", {}, _Canceled())
    session, log = _spy_session(search_error=err)

    with pytest.raises(bvs.KeywordSearchTimeout) as exc_info:
        _run(session)

    assert exc_info.value.knowledge_base_id == "kb-1"
    assert exc_info.value.timeout_ms == 4321
    assert not isinstance(exc_info.value, ValueError)
    assert ("ROLLBACK TO SAVEPOINT", None) in log


@patch.object(bvs, "_bm25_fallback_timeout_ms", return_value=4321)
def test_other_operational_errors_propagate_unchanged(_ms):
    err = OperationalError("SELECT ...", {}, _OtherPgError())
    session, _ = _spy_session(search_error=err)

    with pytest.raises(OperationalError):
        _run(session)


def test_timeout_helper_reads_the_setting():
    with patch.object(bvs, "get_setting", return_value=2500) as gs:
        assert bvs._bm25_fallback_timeout_ms() == 2500
    gs.assert_called_once_with("BM25_FALLBACK_TIMEOUT_MS")
