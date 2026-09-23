"""Dispatch wiring for the per-knowledge-base vector index.

The service decides *whether* to reconcile an index in the indexing path, so a
project whose knowledge bases are all small never enqueues a task at all. These
pin that decision, that a failure to dispatch cannot fail indexing, and that
the start-up sweep and the knowledge-base delete are wired to the same tasks the
pg_search path uses at those points.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services import pg_bm25_index as pgb
from agentic_project_service.services import pg_vector_index as pvi
from agentic_project_service.tasks import indexing as idx

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


@pytest.fixture
def dispatched(monkeypatch):
    """Record the ensure task's dispatches instead of enqueueing them."""
    calls: list[str] = []
    monkeypatch.setattr(
        idx.ensure_per_kb_vector_index, "delay", lambda kb_id, *a, **k: calls.append(kb_id)
    )
    return calls


def _action(monkeypatch, value):
    monkeypatch.setattr(idx, "_per_kb_vector_index_action", lambda kb_id: value)


# ---------------------------------------------------------------------------
# After a source finishes indexing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["build", "drop"])
def test_a_source_that_crosses_a_threshold_dispatches_a_reconcile(monkeypatch, dispatched, action):
    _action(monkeypatch, action)
    assert idx.dispatch_per_kb_vector_index(KB) is True
    assert dispatched == [KB]


def test_a_source_that_changes_nothing_dispatches_nothing(monkeypatch, dispatched):
    """The point of checking the row count here: no task per indexed source."""
    _action(monkeypatch, None)
    assert idx.dispatch_per_kb_vector_index(KB) is False
    assert dispatched == []


def test_a_broker_failure_does_not_fail_the_indexing_run(monkeypatch):
    """Indexing has already committed; the start-up sweep is the backstop."""
    _action(monkeypatch, "build")

    def boom(kb_id, *a, **k):
        raise RuntimeError("broker down")

    monkeypatch.setattr(idx.ensure_per_kb_vector_index, "delay", boom)
    assert idx.dispatch_per_kb_vector_index(KB) is False


def test_an_unreadable_row_count_is_not_an_error(monkeypatch):
    """A knowledge base whose state cannot be read keeps the index it has."""

    class _Session:
        def connection(self):
            raise RuntimeError("no connection")

        def rollback(self):
            pass

    monkeypatch.setattr(idx.db, "session", _Session(), raising=False)
    assert idx._per_kb_vector_index_action(KB) is None


@pytest.fixture
def indexing_run(monkeypatch):
    """Drive ``_run_index_body`` to its post-commit side effects, recording order.

    Everything the body needs from outside is stubbed; what the test reads is
    the returned event list, in the order the body produced it. Asserting on the
    call itself rather than on the source text is the point: a call site that is
    there but never reached -- the whole block behind a false condition, an
    early return above it -- reads exactly the same in ``inspect.getsource``.
    """
    events: list[str] = []

    session = MagicMock()
    session.commit.side_effect = lambda: events.append("commit")
    session.rollback.side_effect = lambda: events.append("rollback")
    # The auto-enrichment block reads a config row and dispatches a task if it
    # finds one; there is none here.
    session.execute.return_value.fetchone.return_value = None
    fake_db = MagicMock()
    fake_db.session = session
    monkeypatch.setattr(idx, "db", fake_db)

    monkeypatch.setattr(idx, "get_knowledge_base", lambda _id: {"indexing_config": {}})
    monkeypatch.setattr(idx, "get_source", lambda _id: {"name": "a.pdf", "auto_metadata": {}})
    monkeypatch.setattr(idx, "get_storage", lambda: MagicMock())
    monkeypatch.setattr(idx, "get_text_derivative_content", lambda *a, **k: "hello world")
    monkeypatch.setattr(idx, "get_page_texts_from_derivative", lambda *a, **k: None)
    monkeypatch.setattr(idx, "init_accumulator", lambda: MagicMock())
    monkeypatch.setattr(idx, "_fenced_mark_indexed", lambda *a: 1)
    monkeypatch.setattr(idx, "_should_build_bm25_now", lambda _kb: False)
    monkeypatch.setattr(idx, "billing", MagicMock())
    monkeypatch.setattr(
        bvs, "ensure_embedding_index", lambda *a, **k: events.append("shared_index")
    )
    monkeypatch.setattr(
        idx, "dispatch_per_kb_vector_index", lambda kb_id: events.append(f"dispatch:{kb_id}")
    )

    def run(embedding_dim: int | None = 1536) -> list[str]:
        async def fake_run_indexing(*, side_out, **kwargs):
            if embedding_dim is not None:
                side_out["embedding_dim"] = embedding_dim
            return {"artifact_count": 3}

        monkeypatch.setattr(idx, "run_indexing", fake_run_indexing)
        out = idx._run_index_body(
            knowledge_base_id=KB,
            source_id="9f1f4ea2-0000-4000-8000-000000000001",
            indexed_source_id=None,
            task_id="task-1",
            provider_keys={},
        )
        assert out["status"] == "success", out
        return events

    return run


def test_an_indexing_run_dispatches_the_reconcile_after_its_commit(indexing_run):
    """It has to run after the commit, next to the shared index's ensure.

    CREATE INDEX CONCURRENTLY cannot run in a transaction at all, and the rows
    that may have crossed the threshold are only visible once committed.
    """
    events = indexing_run()
    assert f"dispatch:{KB}" in events, events
    assert events.index("commit") < events.index(f"dispatch:{KB}"), events
    assert events.index("shared_index") < events.index(f"dispatch:{KB}"), events


def test_a_run_that_wrote_no_embeddings_still_dispatches_a_reconcile(indexing_run):
    """A re-index that only removes embeddings is exactly when a drop is due.

    Gating this on the run having produced embeddings meant the one shape that
    can take a knowledge base *under* the drop threshold -- reindexing it to a
    strategy that stores none -- was the one shape that never asked for a
    reconcile.
    """
    events = indexing_run(embedding_dim=None)
    assert f"dispatch:{KB}" in events, events
    assert "shared_index" not in events, "no embeddings, so no shared index to ensure"


# ---------------------------------------------------------------------------
# Start-up sweep
# ---------------------------------------------------------------------------


def test_the_start_up_sweep_dispatches_one_reconcile_per_knowledge_base(monkeypatch, dispatched):
    other = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
    monkeypatch.setattr(pvi, "kbs_needing_a_per_kb_index", lambda engine: [KB, other])
    assert idx.dispatch_per_kb_vector_indexes_at_start(object()) == [KB, other]
    assert dispatched == [KB, other]


def test_the_start_up_sweep_never_raises(monkeypatch, dispatched):
    """A start-up must not be stopped by a sweep; a source finishing retries it."""

    def boom(engine):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(pvi, "kbs_needing_a_per_kb_index", boom)
    assert idx.dispatch_per_kb_vector_indexes_at_start(object()) == []
    assert dispatched == []


def test_one_failed_dispatch_does_not_stop_the_others(monkeypatch):
    other = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
    monkeypatch.setattr(pvi, "kbs_needing_a_per_kb_index", lambda engine: [KB, other])
    sent: list[str] = []

    def flaky(kb_id, *a, **k):
        if kb_id == KB:
            raise RuntimeError("broker down")
        sent.append(kb_id)

    monkeypatch.setattr(idx.ensure_per_kb_vector_index, "delay", flaky)
    assert idx.dispatch_per_kb_vector_indexes_at_start(object()) == [other]
    assert sent == [other]


class _SweepConn:
    """Enough of a connection for the sweep: two queries and a rollback."""

    def __init__(self, catalog, counted):
        self._catalog = catalog
        self._counted = counted
        self.rolled_back = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        sql = str(statement)
        if "project_settings" in sql:
            rows = []  # nothing stored: the registry defaults apply
        elif "pg_class" in sql:
            rows = self._catalog
        elif "GROUP BY" in sql:
            rows = self._counted
        else:  # set_config for the sweep's own statement_timeout
            rows = []
        return MagicMock(all=lambda: rows)

    def rollback(self):
        self.rolled_back += 1


class _SweepEngine:
    def __init__(self, catalog, counted):
        self._catalog = catalog
        self._counted = counted
        self.connections: list[_SweepConn] = []

    def connect(self):
        conn = _SweepConn(self._catalog, self._counted)
        self.connections.append(conn)
        return conn


def test_the_sweep_reads_its_settings_without_touching_the_shared_session(monkeypatch):
    """It runs inside the boot's migration transaction.

    ``get_setting`` reads ``ai.project_settings`` through ``db.session``, and on
    a database where that table does not exist yet -- the state at a first boot
    -- the failed read leaves that transaction aborted and takes the boot's own
    next statement with it (observed: "Database migration failed ... current
    transaction is aborted"). So the sweep reads its thresholds on a connection
    of its own, and this fails if it ever reads one through ``get_setting``:
    the sweep swallows every exception, so a settings read that raises would
    come back as "no knowledge base needs anything".
    """

    def forbidden(key):
        raise AssertionError(f"the start-up sweep read {key} through db.session")

    monkeypatch.setattr(pvi, "get_setting", forbidden)
    engine = _SweepEngine(catalog=[], counted=[(KB, 1536, 40_000)])

    assert pvi.kbs_needing_a_per_kb_index(engine) == [KB]
    assert all(c.rolled_back for c in engine.connections), (
        "every connection the sweep opens must be rolled back, boot transaction or not"
    )


def test_the_boot_path_runs_the_sweep(monkeypatch):
    """Not a second mechanism: the same start-up hook, beside the BM25 sweep.

    Driven through ``create_app`` rather than read out of it, because a call
    inside an unreachable branch is indistinguishable from a wired one in the
    source text. Everything the boot block touches outside the sweeps is
    stubbed: no Postgres, no Alembic.
    """
    from agentic_project_service import main

    fake_db = MagicMock()
    monkeypatch.setattr(main, "db", fake_db)
    monkeypatch.setattr(main, "quiet_pg_search_planner_warnings", lambda engine: None)
    monkeypatch.setattr(main, "ensure_pg_search_extension", lambda engine: None)
    # No ai schema yet: the migration branch that skips Alembic entirely, which
    # is still followed by every start-up sweep.
    inspector = MagicMock()
    inspector.get_table_names.return_value = []
    monkeypatch.setattr(main, "inspect", lambda engine: inspector)
    monkeypatch.setattr(pgb, "clear_leftover_move_checks_at_start", lambda engine: None)

    order: list[str] = []
    monkeypatch.setattr(
        idx,
        "dispatch_partition_completion_at_start",
        lambda engine: order.append("bm25") or [],
    )
    monkeypatch.setattr(
        idx,
        "dispatch_per_kb_vector_indexes_at_start",
        lambda engine: order.append(f"vector:{engine is fake_db.engine}") or [],
    )

    main.create_app()

    assert order == ["bm25", "vector:True"], (
        "the boot path must reach the per-knowledge-base vector index sweep, "
        f"on the shared engine, beside the BM25 one; got {order}"
    )


# ---------------------------------------------------------------------------
# Knowledge-base delete
# ---------------------------------------------------------------------------


def _kb_route_app():
    from flask import Flask

    from agentic_project_service.routes import knowledge_bases as routes

    app = Flask(__name__)
    app.register_blueprint(routes.knowledge_bases_bp)
    return app


@pytest.fixture
def kb_route_db(monkeypatch):
    """Stub ``db`` for the knowledge-base routes: no rows, no enrichment table."""
    from agentic_project_service.routes import knowledge_bases as routes

    result = MagicMock()
    result.fetchone.return_value = None
    result.fetchall.return_value = []
    result.__iter__ = lambda self: iter(())
    fake_db = MagicMock()
    fake_db.session.execute.return_value = result
    monkeypatch.setattr(routes, "db", fake_db)
    return fake_db


def test_deleting_a_knowledge_base_drops_its_vector_index(monkeypatch, kb_route_db):
    """The KB row is gone, so nothing else will ever reconcile the index.

    Driven through the route, because the delete path is the only caller and a
    dispatch that is written down but never reached leaves the index orphaned
    just the same.
    """
    from unittest.mock import patch

    dropped: list[str] = []
    monkeypatch.setattr(
        idx.drop_per_kb_vector_index, "delay", lambda kb_id, *a, **k: dropped.append(kb_id)
    )
    monkeypatch.setattr(idx.drop_pg_bm25_index, "delay", lambda kb_id, *a, **k: None)

    with (
        patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"}),
        _kb_route_app().test_client() as client,
    ):
        resp = client.delete(
            f"/api/knowledge-bases/{KB}",
            headers={"Authorization": "Bearer fake.jwt.token"},
        )

    assert resp.status_code == 200, resp.data[:200]
    assert dropped == [KB]


def test_removing_a_source_from_a_knowledge_base_dispatches_a_reconcile(monkeypatch, kb_route_db):
    """Deleting the indexed_sources row CASCADEs its embeddings away.

    Nothing else notices: the indexing path never runs for a removal, so
    without this a knowledge base that has just lost most of its rows keeps a
    partial index it no longer qualifies for -- maintained on every write to the
    embeddings table -- until some pod happens to win the boot lock.
    """
    from unittest.mock import patch

    indexed_source_id = "5b3d0a12-0000-4000-8000-00000000000a"
    kb_route_db.session.execute.return_value.fetchone.return_value = ("indexed", None)

    from agentic_project_service.routes import knowledge_bases as routes

    reconciled: list[str] = []
    monkeypatch.setattr(
        routes, "dispatch_per_kb_vector_index", lambda kb_id: reconciled.append(kb_id) or True
    )

    with (
        patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"}),
        _kb_route_app().test_client() as client,
    ):
        resp = client.delete(
            f"/api/knowledge-bases/{KB}/sources/{indexed_source_id}",
            headers={"Authorization": "Bearer fake.jwt.token"},
        )

    assert resp.status_code == 200, resp.data[:200]
    assert reconciled == [KB]


def test_deleting_a_source_reconciles_every_knowledge_base_it_was_indexed_in(monkeypatch):
    """One source can be indexed in several knowledge bases, and the delete
    CASCADEs its embeddings out of all of them at once."""
    from unittest.mock import patch

    from flask import Flask

    from agentic_project_service.routes import sources as src_routes

    other = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
    source_id = "7c9e6679-7425-40de-944b-e07fc1f90ae7"

    def execute(statement, params=None):
        sql = str(statement)
        result = MagicMock()
        if "SELECT storage_path" in sql:
            result.fetchone.return_value = (None, {})
        elif "knowledge_bases kb" in sql:
            # Two knowledge bases, and the same one twice: one reconcile each.
            result.__iter__ = lambda self: iter([(KB, "first"), (other, "second"), (KB, "first")])
        else:
            result.fetchone.return_value = None
            result.__iter__ = lambda self: iter(())
        return result

    fake_db = MagicMock()
    fake_db.session.execute.side_effect = execute
    monkeypatch.setattr(src_routes, "db", fake_db)
    monkeypatch.setattr(src_routes, "get_storage", lambda: MagicMock())

    reconciled: list[str] = []
    monkeypatch.setattr(
        idx, "dispatch_per_kb_vector_index", lambda kb_id: reconciled.append(kb_id) or True
    )

    app = Flask(__name__)
    app.register_blueprint(src_routes.sources_bp)
    with (
        patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"}),
        app.test_client() as client,
    ):
        resp = client.delete(
            f"/api/sources/{source_id}",
            headers={"Authorization": "Bearer fake.jwt.token"},
        )

    assert resp.status_code == 200, resp.data[:200]
    assert reconciled == [KB, other], "one reconcile per affected knowledge base, deduplicated"
    # The warning the route already returned must survive the added query column.
    assert "first" in resp.get_json()["warning"]


# ---------------------------------------------------------------------------
# What an operator can see
# ---------------------------------------------------------------------------


def _ensure_log(caplog) -> str:
    return "\n".join(
        r.getMessage() for r in caplog.records if "per_kb_vector_index" in r.getMessage()
    )


def test_a_build_attempt_is_logged_as_it_happens(monkeypatch, caplog):
    """The service's progress hook has to be wired to something.

    Without it the only record of a build is the service's own prose line, and
    nothing at all records an attempt that the worker did not survive: to answer
    "does this knowledge base have its index, and if not why", an operator is
    left deriving the index name by hand and grepping.
    """
    import logging

    def fake_ensure(kb_id, engine=None, on_progress=None):
        on_progress("building", dims=1536, rows=61_000)
        return {"status": "ready", "built": [f"hnsw_kb_{KB.replace('-', '')}_1536"], "dropped": []}

    monkeypatch.setattr(pvi, "ensure_per_kb_vector_index", fake_ensure)
    with caplog.at_level(logging.INFO):
        idx.ensure_per_kb_vector_index.run(KB)

    message = _ensure_log(caplog)
    assert "event=building" in message, message
    assert f"kb={KB}" in message, message
    assert "dims=1536" in message, message
    assert "rows=61000" in message, message


def test_every_run_ends_in_one_structured_line(monkeypatch, caplog):
    """Outcome, dimensions and duration in one line, whatever the run did."""
    import logging

    def fake_ensure(kb_id, engine=None, on_progress=None):
        on_progress("dropping")
        return {
            "status": "ready",
            "built": [],
            "dropped": [f"hnsw_kb_{KB.replace('-', '')}_768"],
        }

    monkeypatch.setattr(pvi, "ensure_per_kb_vector_index", fake_ensure)
    with caplog.at_level(logging.INFO):
        idx.ensure_per_kb_vector_index.run(KB)

    summary = [line for line in _ensure_log(caplog).splitlines() if "outcome=" in line]
    assert len(summary) == 1, _ensure_log(caplog)
    line = summary[0]
    assert f"kb={KB}" in line
    assert "outcome=ready" in line
    assert "dims=768" in line, "the dimensions the run touched, not the index name alone"
    assert "dropped=hnsw_kb_" in line
    assert "duration_ms=" in line
    assert "attempt=1" in line


def test_a_failed_build_is_recorded_before_it_is_raised(monkeypatch, caplog):
    """A permanently failing build is exactly the case an operator has to find."""
    import logging

    monkeypatch.setattr(
        pvi,
        "ensure_per_kb_vector_index",
        MagicMock(side_effect=ValueError("dimension 3072 exceeds the HNSW limit")),
    )
    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError):
            idx.ensure_per_kb_vector_index.run(KB)

    message = _ensure_log(caplog)
    assert "outcome=failed" in message, message
    assert "3072" in message, message
    assert "duration_ms=" in message, message


# ---------------------------------------------------------------------------
# The tasks' own retry decisions
# ---------------------------------------------------------------------------


@pytest.fixture
def retry_spy(monkeypatch):
    """Replace both tasks' self.retry with a spy that hands back a Retry."""
    from celery.exceptions import Retry

    spies = {}
    for task in (idx.ensure_per_kb_vector_index, idx.drop_per_kb_vector_index):
        spy = MagicMock(side_effect=Retry("retry"))
        monkeypatch.setattr(task, "retry", spy)
        spies[task.name] = spy
    return spies


def _lost_connection():
    """A transient database failure the pg_search path already classifies as one."""
    from sqlalchemy.exc import OperationalError

    class _Orig(Exception):
        sqlstate = "57P01"

    return OperationalError("SELECT 1", {}, _Orig("terminating connection"))


def test_the_drop_task_retries_a_build_that_is_in_progress(monkeypatch, retry_spy):
    """A contended drop must requeue, not report a drop that did not happen."""
    from celery.exceptions import Retry

    monkeypatch.setattr(
        pvi,
        "drop_per_kb_vector_indexes",
        MagicMock(side_effect=pvi.PerKbVectorIndexBuildInProgress("held")),
    )
    with pytest.raises(Retry):
        idx.drop_per_kb_vector_index.run(KB)
    spy = retry_spy[idx.drop_per_kb_vector_index.name]
    spy.assert_called_once()
    assert spy.call_args.kwargs["countdown"] > 0


def test_the_ensure_task_retries_a_transient_database_failure(monkeypatch, retry_spy):
    from celery.exceptions import Retry

    monkeypatch.setattr(
        pvi, "ensure_per_kb_vector_index", MagicMock(side_effect=_lost_connection())
    )
    with pytest.raises(Retry):
        idx.ensure_per_kb_vector_index.run(KB)
    retry_spy[idx.ensure_per_kb_vector_index.name].assert_called_once()


def test_the_ensure_task_does_not_retry_a_real_failure(monkeypatch, retry_spy):
    """A bad knowledge base id is not going to get better; fail it loudly."""
    monkeypatch.setattr(
        pvi, "ensure_per_kb_vector_index", MagicMock(side_effect=ValueError("not a UUID"))
    )
    with pytest.raises(ValueError):
        idx.ensure_per_kb_vector_index.run(KB)
    retry_spy[idx.ensure_per_kb_vector_index.name].assert_not_called()


def test_the_drop_task_names_the_orphans_when_it_gives_up(monkeypatch, caplog):
    """Nothing comes back to an index whose knowledge base row is already gone.

    So the last retry has to name it: the route's dispatch-failure path says "has
    to be dropped by hand" and the task's exhaustion path must too, or an
    operator has no way to find what is left behind.
    """
    import logging

    from celery.exceptions import Retry

    monkeypatch.setattr(
        pvi,
        "drop_per_kb_vector_indexes",
        MagicMock(side_effect=pvi.PerKbVectorIndexBuildInProgress("held")),
    )
    monkeypatch.setattr(idx, "_orphaned_vector_index_names", lambda kb: ["ai.hnsw_kb_abc_1536"])
    task = idx.drop_per_kb_vector_index
    monkeypatch.setattr(task, "retry", MagicMock(side_effect=Retry("retry")))

    # push_request is how Celery itself supplies a Context; `request` is a
    # read-only property, so it cannot be monkeypatched.
    task.push_request(retries=task.max_retries)
    try:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(pvi.PerKbVectorIndexBuildInProgress):
                task.run(KB)
    finally:
        task.pop_request()
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert "dropped by hand" in message, message
    assert "ai.hnsw_kb_abc_1536" in message, message


def test_the_boot_sweeps_count_is_bounded_short_enough_for_a_start_up():
    """It runs on the boot path, after the migrations, before the lock is released.

    Measured 14 ms over 66,000 embeddings -- about 1.1 s extrapolated to 5.3
    million -- so the ceiling is for a pathological case. This codebase's other
    boot-path bound is 5 s; anything much larger is a start-up that looks hung.
    """
    assert 0 < pvi.SWEEP_TIMEOUT_MS <= 10_000


def test_the_ensure_task_returns_the_services_outcome(monkeypatch, retry_spy):
    outcome = {"status": "ready", "built": ["hnsw_kb_x_1536"], "dropped": []}
    monkeypatch.setattr(pvi, "ensure_per_kb_vector_index", MagicMock(return_value=outcome))
    assert idx.ensure_per_kb_vector_index.run(KB) == outcome
