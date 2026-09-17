"""Database-backed tests for the extraction redelivery cap.

The unit tests fake ``get_source`` and the marker writes, so they cannot see
the SQL that the cap depends on: which columns ``get_source`` returns, the
operand order of the JSONB merge that records an interruption, and whether a
later success clears the marker. Each of those, broken, brings back a worker
that is killed by the same file over and over.

A worker being killed is simulated with a ``BaseException`` raised from inside
extraction: the task's ``except Exception`` handlers do not see it, so the
source is left ``extracting`` under the task's id, which is exactly what a
killed process leaves behind.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import fakeredis
import pytest
from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.services.extraction_gate import (
    ExtractionAttempts,
    LargeExtractionGate,
)
from agentic_project_service.tasks import extraction as ext_mod


class _WorkerKilled(BaseException):
    """Stands in for SIGKILL: escapes every ``except Exception``."""


@pytest.fixture
def source_id(app):
    sid = str(uuid.uuid4())
    with app.app_context():
        # Declared by a migration rather than the ORM model the app fixture builds from.
        db.session.execute(text("ALTER TABLE ai.sources ADD COLUMN IF NOT EXISTS error_code TEXT"))
        db.session.execute(
            text("""
                INSERT INTO "ai".sources
                    (id, name, file_type, storage_path, extraction_status, auto_metadata)
                VALUES (:id, 'book.pdf', 'application/pdf', :path, 'pending',
                        CAST(:auto AS jsonb))
            """),
            {
                "id": sid,
                "path": f"sources/{sid}/original/book.pdf",
                "auto": json.dumps({"extraction_model": "auto"}),
            },
        )
        db.session.commit()
    return sid


def _row(source_id):
    return db.session.execute(
        text("""
            SELECT extraction_status, celery_task_id, auto_metadata, error_code, error_message
            FROM "ai".sources WHERE id = :id
        """),
        {"id": source_id},
    ).one()


def _dispatch(source_id, task_id):
    """What the upload and re-extract routes write around ``.delay()``."""
    db.session.execute(
        text("""
            UPDATE "ai".sources
            SET extraction_status = 'pending', error_message = NULL, error_code = NULL,
                celery_task_id = :task_id
            WHERE id = :id
        """),
        {"id": source_id, "task_id": task_id},
    )
    db.session.commit()


@pytest.fixture
def deliver(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    storage = MagicMock()
    storage.object_size.return_value = 1024
    monkeypatch.setattr(ext_mod, "get_storage", lambda: storage)
    monkeypatch.setattr(
        ext_mod,
        "large_extraction_gate",
        LargeExtractionGate(redis_client=redis_client, heartbeat=False),
    )

    observed = []

    def run(source_id, task_id, outcome, worker="worker-a", during=None, auto_metadata=None):
        monkeypatch.setattr(
            ext_mod,
            "extraction_attempts",
            ExtractionAttempts(redis_client=redis_client, incarnation=worker),
        )

        async def fake_run_extraction(*a, **kw):
            observed.append(_row(source_id).auto_metadata)
            if during is not None:
                during()
            if outcome == "killed":
                raise _WorkerKilled()
            return {}, {
                "extraction_method": "fitz",
                "page_count": 1,
                "char_count": 500,
                **(auto_metadata or {}),
            }

        monkeypatch.setattr(ext_mod, "run_extraction", fake_run_extraction)
        ext_mod.extract_source.push_request(id=task_id)
        try:
            return ext_mod.extract_source.run(source_id=source_id, bucket_id="sources")
        except _WorkerKilled:
            # A killed worker never reaches the task's own cleanup, so put back
            # the in-flight record its `finally` just removed.
            db.session.rollback()
            ExtractionAttempts(redis_client=redis_client, incarnation=worker).begin(
                task_id, storage.object_size.return_value, None
            )
            return "killed"
        finally:
            ext_mod.extract_source.pop_request()

    run.redis = redis_client
    run.observed = observed
    return run


@pytest.mark.integration
class TestGetSource:
    def test_returns_the_celery_task_id(self, app, source_id):
        with app.app_context():
            _dispatch(source_id, "task-1")
            source = ext_mod.get_source(source_id)
        assert source["celery_task_id"] == "task-1"


@pytest.mark.integration
class TestRecordExtractionInterruption:
    def test_a_later_task_replaces_an_earlier_tasks_marker(self, app, source_id):
        with app.app_context():
            ext_mod.record_extraction_interruption(source_id, "task-1", 1)
            ext_mod.record_extraction_interruption(source_id, "task-1", 2)
            ext_mod.record_extraction_interruption(source_id, "task-2", 1)
            auto = _row(source_id).auto_metadata
        assert auto["extraction_interrupted_task"] == "task-2"
        assert auto["extraction_interruptions"] == 1
        assert auto["extraction_unattributed_interruptions"] == 0
        assert auto["extraction_model"] == "auto"

    def test_the_unattributed_count_is_stored(self, app, source_id):
        with app.app_context():
            ext_mod.record_extraction_interruption(source_id, "task-1", 1, 3)
            auto = _row(source_id).auto_metadata
        assert auto["extraction_interruptions"] == 1
        assert auto["extraction_unattributed_interruptions"] == 3


@pytest.mark.integration
class TestUpdateSourceExtractionResult:
    def test_success_clears_the_interruption_marker(self, app, source_id):
        with app.app_context():
            ext_mod.record_extraction_interruption(source_id, "task-1", 1, 2)
            applied = ext_mod.update_source_extraction_result(
                source_id, {}, {"extraction_method": "fitz"}
            )
            auto = _row(source_id).auto_metadata
        assert applied is True
        assert "extraction_interruptions" not in auto
        assert "extraction_unattributed_interruptions" not in auto
        assert "extraction_interrupted_task" not in auto
        assert auto["extraction_method"] == "fitz"
        assert auto["extraction_model"] == "auto"

    def test_a_partial_page_image_flag_does_not_outlive_a_complete_run(self, app, source_id):
        with app.app_context():
            ext_mod.update_source_extraction_result(
                source_id, {}, {"extraction_method": "lighton_ocr", "page_images_incomplete": True}
            )
            assert _row(source_id).auto_metadata["page_images_incomplete"] is True
            ext_mod.update_source_extraction_result(
                source_id, {}, {"extraction_method": "lighton_ocr"}
            )
            auto = _row(source_id).auto_metadata
        assert "page_images_incomplete" not in auto

    def test_a_result_from_a_task_that_no_longer_owns_the_source_is_discarded(self, app, source_id):
        with app.app_context():
            _dispatch(source_id, "task-2")
            applied = ext_mod.update_source_extraction_result(
                source_id, {"markdown": []}, {"extraction_method": "fitz"}, task_id="task-1"
            )
            row = _row(source_id)
            owned = ext_mod.update_source_extraction_result(
                source_id, {"markdown": []}, {"extraction_method": "fitz"}, task_id="task-2"
            )
        assert applied is False
        assert row.extraction_status == "pending"
        assert "extraction_method" not in row.auto_metadata
        assert owned is True

    def test_a_cancelled_source_is_left_alone_and_reported(self, app, source_id):
        with app.app_context():
            db.session.execute(
                text("UPDATE ai.sources SET extraction_status = 'cancelled' WHERE id = :id"),
                {"id": source_id},
            )
            db.session.commit()
            applied = ext_mod.update_source_extraction_result(
                source_id, {}, {"extraction_method": "fitz"}
            )
            row = _row(source_id)
        assert applied is False
        assert row.extraction_status == "cancelled"


@pytest.mark.integration
class TestRedeliveryLifecycle:
    def test_a_normal_first_delivery_is_not_counted(self, app, source_id, deliver):
        with app.app_context():
            _dispatch(source_id, "task-1")
            result = deliver(source_id, "task-1", "success")
            row = _row(source_id)
        assert result["status"] == "success"
        assert row.extraction_status == "extracted"
        # Not merely cleared by the success: never recorded while it ran.
        (during,) = deliver.observed
        assert "extraction_interruptions" not in during
        assert "extraction_interruptions" not in row.auto_metadata

    def test_a_file_that_keeps_killing_the_worker_is_failed_then_can_be_re_extracted(
        self, app, source_id, deliver
    ):
        with app.app_context():
            _dispatch(source_id, "task-1")
            assert deliver(source_id, "task-1", "killed") == "killed"
            assert _row(source_id).extraction_status == "extracting"

            assert deliver(source_id, "task-1", "killed", worker="worker-b") == "killed"
            assert _row(source_id).auto_metadata["extraction_interruptions"] == 1

            result = deliver(source_id, "task-1", "success", worker="worker-c")
            row = _row(source_id)
            assert result["status"] == "error"
            assert row.extraction_status == "failed"
            assert row.error_code == "permanent"
            assert row.auto_metadata["extraction_interruptions"] == 2

            # Re-extract: a new task gets a fresh count, and dies once.
            _dispatch(source_id, "task-2")
            assert deliver(source_id, "task-2", "killed", worker="worker-d") == "killed"
            result = deliver(source_id, "task-2", "success", worker="worker-e")
            row = _row(source_id)

        assert result["status"] == "success"
        assert row.extraction_status == "extracted"
        assert "extraction_interruptions" not in row.auto_metadata
        assert "extraction_interrupted_task" not in row.auto_metadata

    def test_a_task_killed_alongside_a_larger_file_is_not_charged(self, app, source_id, deliver):
        with app.app_context():
            _dispatch(source_id, "task-1")
            # The worker this task dies on is also extracting a far larger file.
            ExtractionAttempts(redis_client=deliver.redis, incarnation="worker-a").begin(
                "task-big", 400_000_000, None
            )
            assert deliver(source_id, "task-1", "killed", worker="worker-a") == "killed"
            assert deliver(source_id, "task-1", "killed", worker="worker-b") == "killed"
            row = _row(source_id)
        assert row.extraction_status == "extracting"
        assert row.auto_metadata["extraction_interruptions"] == 0
        assert row.auto_metadata["extraction_unattributed_interruptions"] == 1

    def test_a_re_extract_dispatched_during_a_run_is_not_lost(self, app, source_id, deliver):
        """The first run finishes after the re-extract was dispatched: its
        result is discarded and the re-extract still runs."""
        with app.app_context():
            _dispatch(source_id, "task-1")
            result = deliver(
                source_id,
                "task-1",
                "success",
                during=lambda: _dispatch(source_id, "task-2"),
            )
            row = _row(source_id)
            assert result["status"] == "superseded"
            assert (row.extraction_status, row.celery_task_id) == ("pending", "task-2")

            result = deliver(source_id, "task-2", "success", worker="worker-b")
            row = _row(source_id)
        assert result["status"] == "success"
        assert (row.extraction_status, row.celery_task_id) == ("extracted", "task-2")
