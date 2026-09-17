"""Routes that dispatch extraction record the task id before the task can run.

The extraction task only runs a source whose ``celery_task_id`` is its own
(or unset). A route that dispatched first and wrote the id afterwards would
let the task start against the previous owner's id and give up as
superseded, and a re-extract that set ``pending`` in one statement and the id
in another would leave a window where the earlier task still looks like the
owner.
"""

import io
from unittest.mock import MagicMock, patch

from agentic_project_service.routes import sources as sources_route
from agentic_project_service.services import billing_port
from tests.support.billing import RecordingBillingAdapter


def _make_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(sources_route.sources_bp)
    return app


def _auth():
    return patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    )


class _Session:
    """Records statements, and whether each came before or after dispatch."""

    def __init__(self, fetchone=None):
        self.events = []
        self.fetchone = fetchone
        self.commit = MagicMock(side_effect=lambda: self.events.append(("commit", None, None)))

    def execute(self, stmt, params=None):
        self.events.append(("sql", str(stmt), params or {}))
        result = MagicMock()
        result.fetchone.return_value = self.fetchone
        return result

    def rollback(self):
        pass


def _dispatch_recorder(session, task_attr):
    def apply_async(*args, **kwargs):
        session.events.append(("dispatch", task_attr, kwargs))
        task = MagicMock()
        task.id = kwargs["task_id"]
        return task

    return apply_async


def _dispatches(session):
    return [(i, kw) for i, (kind, _attr, kw) in enumerate(session.events) if kind == "dispatch"]


def test_reextract_sets_pending_and_the_new_owner_in_one_statement_before_dispatch():
    billing_port.set_billing_adapter(RecordingBillingAdapter())
    session = _Session(fetchone=("src-1", "extracted", {}))

    with (
        patch.object(sources_route.db, "session", session),
        patch.object(
            sources_route.extract_source,
            "apply_async",
            side_effect=_dispatch_recorder(session, "extract_source"),
        ),
        patch.object(sources_route, "get_all_user_provider_keys", return_value={}),
        _auth(),
    ):
        with _make_test_app().test_client() as client:
            resp = client.post(
                "/api/sources/src-1/reextract", json={}, headers={"Authorization": "Bearer x"}
            )

    assert resp.status_code == 200
    ((dispatch_index, kwargs),) = _dispatches(session)
    task_id = kwargs["task_id"]
    assert task_id and resp.get_json()["task_id"] == task_id
    assert kwargs["kwargs"]["reextract_seed"]

    resets = [
        (i, sql, params)
        for i, (kind, sql, params) in enumerate(session.events)
        if kind == "sql" and "extraction_status = 'pending'" in sql
    ]
    ((reset_index, sql, params),) = resets
    assert "celery_task_id = :task_id" in sql
    assert params["task_id"] == task_id
    # Written and committed before the task exists.
    assert reset_index < dispatch_index
    assert any(kind == "commit" for kind, _s, _p in session.events[reset_index:dispatch_index])
    # And nothing rewrites the owner afterwards.
    assert not any(
        "celery_task_id" in sql for _k, sql, _p in session.events[dispatch_index:] if sql
    )


def test_upload_inserts_the_source_with_its_owner_before_dispatch():
    billing_port.set_billing_adapter(RecordingBillingAdapter())
    session = _Session(fetchone=None)
    storage = MagicMock()
    storage.upload.return_value = "sources/x/original/a.pdf"

    with (
        patch.object(sources_route.db, "session", session),
        patch.object(sources_route, "get_storage", return_value=storage),
        patch.object(
            sources_route.extract_source,
            "apply_async",
            side_effect=_dispatch_recorder(session, "extract_source"),
        ),
        patch.object(sources_route, "get_all_user_provider_keys", return_value={}),
        _auth(),
    ):
        with _make_test_app().test_client() as client:
            resp = client.post(
                "/api/sources/upload",
                data={"file": (io.BytesIO(b"%PDF-1.4 x"), "a.pdf")},
                headers={"Authorization": "Bearer x"},
                content_type="multipart/form-data",
            )

    assert resp.status_code == 201, resp.get_json()
    ((dispatch_index, kwargs),) = _dispatches(session)
    task_id = kwargs["task_id"]
    assert resp.get_json()["task_id"] == task_id
    inserts = [
        (i, params)
        for i, (kind, sql, params) in enumerate(session.events)
        if kind == "sql" and sql and "INSERT INTO" in sql
    ]
    ((insert_index, params),) = inserts
    assert params["celery_task_id"] == task_id
    assert insert_index < dispatch_index
    assert not any(
        sql and "celery_task_id" in sql for _k, sql, _p in session.events[dispatch_index:]
    )
