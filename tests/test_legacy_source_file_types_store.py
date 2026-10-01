"""Migration 0034 repairs sources stored under a short file type.

import-from-storage stored ``pdf``, ``docx``, ... instead of a MIME type, so
those files were extracted by the plain-text fallback: a PDF's or an Office
file's raw bytes became its text. Office files never reached a knowledge base
only because those bytes held NUL, which the database refused; with NUL now
cleaned on the way in, the next index of such a source would embed (and bill)
zip bytes. The migration gives every short type its MIME type, records the
old one, and holds each binary source that was "extracted" that way in
``attention_required`` -- which indexing refuses -- until it is re-extracted.

Runs against a real Postgres: the migration is SQL.
"""

import importlib.util
import uuid
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text

from agentic_project_service.db import db

MIGRATION = next((Path(__file__).parents[1] / "migrations" / "versions").glob("0034_*.py"))

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture
def migration():
    spec = importlib.util.spec_from_file_location("mig_0034", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upgrade(migration):
    with db.engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            migration.upgrade()


def _seed(file_type, status, auto_metadata="{}", storage_path="sources/x"):
    source_id = str(uuid.uuid4())
    db.session.execute(
        text(
            "INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status, "
            "auto_metadata) VALUES (:id, :name, :ft, :sp, :st, CAST(:am AS jsonb))"
        ),
        {
            "id": source_id,
            "name": source_id,
            "ft": file_type,
            "sp": storage_path,
            "st": status,
            "am": auto_metadata,
        },
    )
    db.session.commit()
    return source_id


def _row(source_id):
    return db.session.execute(
        text(
            "SELECT file_type, extraction_status, error_message, auto_metadata "
            "FROM ai.sources WHERE id = :id"
        ),
        {"id": source_id},
    ).one()


def test_the_revision_follows_0033(migration):
    assert (migration.revision, migration.down_revision) == ("0034", "0033")


@pytest.mark.parametrize(
    "legacy,mime",
    [
        ("pdf", "application/pdf"),
        ("docx", DOCX),
        ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("xls", "application/vnd.ms-excel"),
        ("pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    ],
)
def test_an_extracted_binary_source_is_held_for_re_extraction(app, migration, legacy, mime):
    with app.app_context():
        source_id = _seed(legacy, "extracted", '{"extraction_method": "text"}')
        _upgrade(migration)
        row = _row(source_id)

    assert row.file_type == mime
    assert row.extraction_status == "attention_required"
    assert "re-extract" in row.error_message.lower()
    assert row.auto_metadata == {
        "extraction_method": "text",
        "legacy_file_type": legacy,
        "reextract_hold": True,
    }


@pytest.mark.parametrize(
    "legacy,mime",
    [("markdown", "text/markdown"), ("text", "text/plain"), ("md", "text/markdown")],
)
def test_a_text_source_keeps_its_extraction(app, migration, legacy, mime):
    """The text fallback was the right extractor for Markdown and plain text."""
    with app.app_context():
        source_id = _seed(legacy, "extracted")
        _upgrade(migration)
        row = _row(source_id)

    assert row.file_type == mime
    assert row.extraction_status == "extracted"
    assert row.error_message is None
    assert row.auto_metadata == {"legacy_file_type": legacy}


@pytest.mark.parametrize("status", ["failed", "pending", "extracting", "attention_required"])
def test_a_binary_source_that_is_not_extracted_keeps_its_status(app, migration, status):
    """Not indexable as it stands; a pending one is extracted with the right
    extractor once the new code runs it."""
    with app.app_context():
        source_id = _seed("docx", status)
        _upgrade(migration)
        row = _row(source_id)

    assert row.file_type == DOCX
    assert row.extraction_status == status


def test_a_source_with_a_mime_type_is_untouched(app, migration):
    with app.app_context():
        source_id = _seed("application/pdf", "extracted")
        before = db.session.execute(
            text("SELECT updated_at FROM ai.sources WHERE id = :id"), {"id": source_id}
        ).scalar_one()
        _upgrade(migration)
        row = _row(source_id)
        after = db.session.execute(
            text("SELECT updated_at FROM ai.sources WHERE id = :id"), {"id": source_id}
        ).scalar_one()

    assert (row.file_type, row.extraction_status, row.auto_metadata) == (
        "application/pdf",
        "extracted",
        {},
    )
    assert after == before


def test_running_it_again_changes_nothing(app, migration):
    with app.app_context():
        ids = [
            _seed("pdf", "extracted"),
            _seed("application/zip", "extracted", '{"extraction_method": "text"}', "s/a.docx"),
            _seed("application/octet-stream", "failed", "{}", "s/b.pdf"),
        ]
        _upgrade(migration)
        first = [_row(i) for i in ids]
        _upgrade(migration)
        second = [_row(i) for i in ids]

    assert second == first


PDF = "application/pdf"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


@pytest.mark.parametrize(
    "stored,path,method,mime",
    [
        ("application/octet-stream", "sources/1/original/report.pdf", "text", PDF),
        ("application/zip", "sources/1/original/letter.docx", "text", DOCX),
        ("application/x-pdf", "sources/1/original/SCAN.PDF", "text", PDF),
        ("text/plain", "sources/1/original/sheet.xlsx", "txt-native", XLSX),
        ("application/msword", "sources/1/original/deck.pptx", "text", PPTX),
    ],
)
def test_an_upload_the_registry_could_not_route_is_held(app, migration, stored, path, method, mime):
    """/upload kept a declared type no extractor is registered under (or a text
    type for a binary file), so the file's bytes were extracted as text."""
    with app.app_context():
        source_id = _seed(stored, "extracted", f'{{"extraction_method": "{method}"}}', path)
        _upgrade(migration)
        row = _row(source_id)

    assert row.file_type == mime
    assert row.extraction_status == "attention_required"
    assert "re-extract" in row.error_message.lower()
    assert row.auto_metadata == {
        "extraction_method": method,
        "legacy_file_type": stored,
        "reextract_hold": True,
    }


def test_a_binary_file_extracted_by_the_text_fallback_under_its_mime_type_is_held(app, migration):
    """A worker still on the old code can finish a short-typed row after this
    migration gave it its MIME type; its text is raw bytes all the same."""
    with app.app_context():
        source_id = _seed(PDF, "extracted", '{"extraction_method": "text"}', "sources/1/a.pdf")
        _upgrade(migration)
        row = _row(source_id)

    assert (row.file_type, row.extraction_status) == (PDF, "attention_required")
    assert row.auto_metadata == {"extraction_method": "text", "reextract_hold": True}


def test_a_failed_upload_under_an_unroutable_type_gets_its_mime_type(app, migration):
    """Not indexable, so not held -- but re-extracting it must pick the real
    extractor, which needs the MIME type."""
    with app.app_context():
        source_id = _seed("application/octet-stream", "failed", "{}", "sources/1/a.docx")
        _upgrade(migration)
        row = _row(source_id)

    assert (row.file_type, row.extraction_status) == (DOCX, "failed")
    assert row.auto_metadata == {"legacy_file_type": "application/octet-stream"}


@pytest.mark.parametrize(
    "stored,path,method",
    [
        (PDF, "sources/1/a.pdf", "fitz"),
        (DOCX, "sources/1/a.docx", "docx-via-pdf"),
        ("application/octet-stream", "sources/1/notes.txt", "text"),
        ("text/plain", "sources/1/notes.txt", "txt-native"),
    ],
)
def test_a_properly_extracted_source_is_untouched(app, migration, stored, path, method):
    with app.app_context():
        source_id = _seed(stored, "extracted", f'{{"extraction_method": "{method}"}}', path)
        _upgrade(migration)
        row = _row(source_id)

    assert (row.file_type, row.extraction_status, row.auto_metadata) == (
        stored,
        "extracted",
        {"extraction_method": method},
    )


def test_a_json_null_auto_metadata_becomes_an_object(app, migration):
    with app.app_context():
        source_id = _seed("pdf", "extracted", "null")
        _upgrade(migration)
        row = _row(source_id)

    assert row.auto_metadata == {"legacy_file_type": "pdf", "reextract_hold": True}


def test_a_held_source_cannot_be_indexed(app, migration, test_knowledge_base):
    from agentic_project_service.routes.knowledge_bases import index_source_into_kb

    with app.app_context():
        source_id = _seed("docx", "extracted")
        _upgrade(migration)
        result = index_source_into_kb(test_knowledge_base["id"], source_id)

    assert result["status_code"] == 400
    assert "attention_required" in result["error"]
