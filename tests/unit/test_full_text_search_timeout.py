"""full_text_search runs under a savepoint-scoped statement_timeout and turns a
cancelled statement into KeywordSearchTimeout."""

import asyncio
import logging
from contextlib import contextmanager
from types import SimpleNamespace
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


def _run(session, query: str = "anything"):
    store = _FakeStore(db_session=session, knowledge_base_id="kb-1")
    return asyncio.run(store.full_text_search(query, top_k=5))


@contextmanager
def _elapsed(seconds: float):
    """Pin the fetch's measured duration.

    Replaces the module's own ``time`` binding rather than time.monotonic
    itself, which asyncio also reads. _fetch_with_timeout reads the clock once
    before the query and once when mapping an error, so two values are enough.
    """
    ticks = iter((0.0, seconds))
    with patch.object(bvs, "time", SimpleNamespace(monotonic=lambda: next(ticks))):
        yield


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

    # Both set_config calls must pass is_local=true: a session-wide timeout
    # would outlive the savepoint and bound unrelated later statements on the
    # same pooled connection.
    for i, _p in set_calls:
        assert log[i][0].strip().endswith(", true)"), log[i][0]


@patch.object(bvs, "_bm25_fallback_timeout_ms", return_value=4321)
def test_corpus_stats_is_materialized(_ms):
    """Inlined, corpus_stats lands inside the per-row Nested Loop and re-runs.

    For a two-term query such as "weather weather" (what the route's query
    builder produces) the planner estimates one matching row and puts the
    whole-KB aggregate on the inner side of the join, so it is re-executed once
    per matching row: quadratic, 39 s at 5,000 rows. MATERIALIZED evaluates it
    once. doc_freqs is deliberately left alone: it already runs once, as an
    InitPlan.
    """
    session, log = _spy_session()
    _run(session, query="weather weather")

    search_sql = next(s for s, _ in log if "corpus_stats" in s)
    assert "WITH corpus_stats AS MATERIALIZED (" in " ".join(search_sql.split())


@patch.object(bvs, "_bm25_fallback_timeout_ms", return_value=4321)
def test_query_canceled_becomes_keyword_search_timeout(_ms):
    err = OperationalError("SELECT ...", {}, _Canceled())
    session, log = _spy_session(search_error=err)

    with _elapsed(4.321), pytest.raises(bvs.KeywordSearchTimeout) as exc_info:
        _run(session)

    assert exc_info.value.knowledge_base_id == "kb-1"
    assert exc_info.value.timeout_ms == 4321
    assert not isinstance(exc_info.value, ValueError)
    assert ("ROLLBACK TO SAVEPOINT", None) in log


@patch.object(bvs, "_bm25_fallback_timeout_ms", return_value=4321)
def test_query_canceled_well_inside_the_budget_propagates(_ms):
    """57014 is also what another session's pg_cancel_backend produces.

    Half a second into a 4321 ms budget our own statement_timeout cannot have
    fired, so calling it a timeout would mislabel a foreign cancellation and
    hybrid would silently swallow it.
    """
    err = OperationalError("SELECT ...", {}, _Canceled())
    session, log = _spy_session(search_error=err)

    with _elapsed(0.5), pytest.raises(OperationalError) as exc_info:
        _run(session)

    assert not isinstance(exc_info.value, bvs.KeywordSearchTimeout)
    assert ("ROLLBACK TO SAVEPOINT", None) in log


@patch.object(bvs, "_bm25_fallback_timeout_ms", return_value=4321)
def test_query_canceled_just_under_the_budget_propagates(_ms):
    """The comparison is exact, with no tolerance widening the blind spot.

    The client clock starts before begin_nested and two extra round trips, so
    measured elapsed strictly exceeds the server's statement time: a genuine
    statement_timeout can never land below the budget.
    """
    err = OperationalError("SELECT ...", {}, _Canceled())
    session, _ = _spy_session(search_error=err)

    with _elapsed(4.320), pytest.raises(OperationalError) as exc_info:
        _run(session)

    assert not isinstance(exc_info.value, bvs.KeywordSearchTimeout)


@patch.object(bvs, "_bm25_fallback_timeout_ms", return_value=4321)
def test_query_canceled_exactly_at_the_budget_is_a_timeout(_ms):
    err = OperationalError("SELECT ...", {}, _Canceled())
    session, _ = _spy_session(search_error=err)

    with _elapsed(4.321), pytest.raises(bvs.KeywordSearchTimeout):
        _run(session)


@patch.object(bvs, "_bm25_fallback_timeout_ms", return_value=4321)
def test_other_operational_errors_propagate_unchanged(_ms):
    err = OperationalError("SELECT ...", {}, _OtherPgError())
    session, log = _spy_session(search_error=err)

    with pytest.raises(OperationalError) as exc_info:
        _run(session)

    assert not isinstance(exc_info.value, bvs.KeywordSearchTimeout)
    # The savepoint was rolled back, so the aborted state is cleared and the
    # caller's session is still usable. That the session really is usable
    # afterwards can only be shown against Postgres — see the live module; a
    # spy would answer any statement regardless.
    assert ("ROLLBACK TO SAVEPOINT", None) in log


@patch.object(bvs, "_bm25_fallback_timeout_ms", return_value=4321)
def test_timeout_logs_one_warning_and_no_error(_ms, caplog):
    """A designed degradation must not page as ERROR, and must say so once.

    full_text_search's generic handler used to log ERROR on the way past, so
    every request against an un-indexed KB produced an error line for expected
    behaviour.
    """
    err = OperationalError("SELECT ...", {}, _Canceled())
    session, _ = _spy_session(search_error=err)

    with caplog.at_level(logging.DEBUG, logger=bvs.logger.name):
        with _elapsed(4.321), pytest.raises(bvs.KeywordSearchTimeout):
            _run(session, query="a moderately long hiking query")

    assert [r.levelname for r in caplog.records if r.levelname == "ERROR"] == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]

    message = warnings[0].getMessage()
    assert "kb-1" in message
    assert "chunks" in message  # _FakeStore.TABLE
    assert "4321" in message
    assert str(len("a moderately long hiking query")) in message


@pytest.fixture(autouse=True)
def _forget_warned_overrides():
    """The warn-once memory is module state; keep these tests independent."""
    bvs._WARNED_TIMEOUT_OVERRIDES.clear()
    yield
    bvs._WARNED_TIMEOUT_OVERRIDES.clear()


def test_timeout_helper_reads_the_setting():
    with patch.object(bvs, "get_setting", return_value=2500) as gs:
        assert bvs._bm25_fallback_timeout_ms() == 2500
    gs.assert_called_once_with("BM25_FALLBACK_TIMEOUT_MS")


@pytest.mark.parametrize(
    "stored,expected",
    [
        (0, 1000),  # a stored 0 means "no timeout" to Postgres: it would disarm the bound
        (500, 1000),
        (-5, 1000),
        (999999, 30000),
        (1000, 1000),
        (30000, 30000),
        (7500, 7500),
    ],
)
def test_timeout_helper_clamps_to_the_registry_range(stored, expected, caplog):
    """get_setting applies no range check -- bounds live on the PUT path only.

    A row written before this setting existed, or by anything other than the
    settings endpoint, can therefore carry any integer.
    """
    with caplog.at_level(logging.WARNING, logger=bvs.logger.name):
        with patch.object(bvs, "get_setting", return_value=stored):
            assert bvs._bm25_fallback_timeout_ms() == expected

    out_of_range = stored != expected
    assert bool([r for r in caplog.records if r.levelname == "WARNING"]) is out_of_range


@pytest.mark.parametrize("stored", [0, 999999])
def test_a_bad_stored_value_warns_once_per_process(stored, caplog):
    """This helper runs on every keyword search.

    Warning each time would put one line per search in the log for as long as
    the bad value sits in project_settings, which buries the first one.
    """
    with caplog.at_level(logging.DEBUG, logger=bvs.logger.name):
        with patch.object(bvs, "get_setting", return_value=stored):
            for _ in range(3):
                bvs._bm25_fallback_timeout_ms()

    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
    # The later occurrences are still traceable, just not at WARNING.
    assert len([r for r in caplog.records if r.levelname == "DEBUG"]) == 2


def test_a_second_distinct_bad_value_warns_again(caplog):
    with caplog.at_level(logging.WARNING, logger=bvs.logger.name):
        for stored in (0, 999999, 0, 999999):
            with patch.object(bvs, "get_setting", return_value=stored):
                bvs._bm25_fallback_timeout_ms()

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2
