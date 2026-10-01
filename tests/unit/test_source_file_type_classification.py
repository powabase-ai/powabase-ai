"""Upload and import-from-storage classify a file the same way.

Extraction picks its extractor from the source's ``file_type``, looked up by
MIME type. import-from-storage used to store a short name (``pdf``, ``docx``,
...) that no extractor is registered under, so every PDF and Office file it
imported fell through to the plain-text extractor and was "extracted" as its
raw bytes -- while the same file sent to /upload was extracted properly.
"""

import asyncio
import io
from unittest.mock import MagicMock, patch

import pytest
from agentic.ingest import Derivative, ExtractionResult, ExtractorRegistry

from agentic_project_service.routes import sources as sources_route
from agentic_project_service.services import billing_port
from agentic_project_service.services.source_file_types import extraction_mime_type
from tests.support.billing import RecordingBillingAdapter

# Extension -> the extractor that must handle it.
DOCUMENTS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".pptx": "pptx",
    ".md": "text",
    ".txt": "txt",
}


def _app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(sources_route.sources_bp)
    return app


class _Session:
    def __init__(self):
        self.inserts = []

    def execute(self, stmt, params=None):
        if "INSERT INTO" in str(stmt):
            self.inserts.append(params)
        result = MagicMock()
        result.fetchone.return_value = None
        return result

    def commit(self):
        pass

    def rollback(self):
        pass


def _post(route_call):
    """Run one request against the sources blueprint with storage, the
    database and dispatch faked. Returns (response, stored file_type,
    content type the file was stored under)."""
    billing_port.set_billing_adapter(RecordingBillingAdapter())
    session = _Session()
    storage = MagicMock()
    storage.download.return_value = b"file bytes"
    storage.upload.return_value = "sources/x/original/f"
    task = MagicMock()
    task.id = "task-1"

    with (
        patch.object(sources_route.db, "session", session),
        patch.object(sources_route, "get_storage", return_value=storage),
        patch.object(sources_route.extract_source, "apply_async", return_value=task),
        patch.object(sources_route, "get_all_user_provider_keys", return_value={}),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "service_role", "is_service_role": True},
        ),
    ):
        with _app().test_client() as client:
            resp = route_call(client)

    assert resp.status_code == 201, resp.get_json()
    (insert,) = session.inserts
    content_type = storage.upload.call_args.kwargs["content_type"]
    return resp, insert["file_type"], content_type


def _import(ext):
    return _post(
        lambda c: c.post(
            "/api/sources/import-from-storage",
            json={"bucket": "inbox", "path": f"reports/quarterly{ext}"},
            headers={"Authorization": "Bearer x"},
        )
    )


def _upload(ext, declared):
    return _post(
        lambda c: c.post(
            "/api/sources/upload",
            data={"file": (io.BytesIO(b"file bytes"), f"quarterly{ext}", declared)},
            headers={"Authorization": "Bearer x"},
            content_type="multipart/form-data",
        )
    )


@pytest.mark.parametrize("ext,extractor", sorted(DOCUMENTS.items()))
def test_import_from_storage_stores_a_type_extraction_can_route(ext, extractor):
    resp, file_type, content_type = _import(ext)

    assert ExtractorRegistry.default().get_extractor(file_type).name == extractor
    assert content_type == file_type
    assert resp.get_json()["file_type"] == file_type


@pytest.mark.parametrize("ext", sorted(DOCUMENTS))
def test_import_and_upload_classify_the_same_file_identically(ext):
    _, imported, _ = _import(ext)
    _, uploaded_generic, _ = _upload(ext, "application/octet-stream")

    assert imported == uploaded_generic


@pytest.mark.parametrize("ext", sorted(DOCUMENTS))
def test_upload_ignores_a_declared_type_that_disagrees_with_the_extension(ext):
    _, imported, _ = _import(ext)
    _, uploaded, _ = _upload(ext, "text/plain")

    assert imported == uploaded


@pytest.mark.parametrize("ext,extractor", sorted(DOCUMENTS.items()))
def test_upload_without_a_specific_content_type_still_routes(ext, extractor):
    """A client that sends no useful Content-Type (octet-stream) for a PDF or
    a Word file must not get the plain-text extractor either."""
    _, file_type, content_type = _upload(ext, "application/octet-stream")

    assert ExtractorRegistry.default().get_extractor(file_type).name == extractor
    assert content_type == file_type


LEGACY = [
    ("pdf", "pdf"),
    ("docx", "docx"),
    ("xlsx", "xlsx"),
    ("xls", "xlsx"),
    ("pptx", "pptx"),
    ("markdown", "text"),
    ("text", "txt"),
    ("md", "text"),
]


@pytest.mark.parametrize("legacy,extractor", LEGACY)
def test_a_legacy_short_type_maps_to_its_extractor(legacy, extractor):
    assert ExtractorRegistry.default().get_extractor(extraction_mime_type(legacy)).name == extractor


def test_a_mime_type_is_left_as_it_is():
    assert extraction_mime_type("application/pdf") == "application/pdf"
    assert extraction_mime_type("text/csv") == "text/csv"


def test_a_source_imported_before_the_fix_re_extracts_with_its_real_extractor(monkeypatch):
    """Rows already written with a short type (``docx``) pick the Word
    extractor when they are extracted again, not the text fallback."""
    from agentic_project_service.tasks.extraction import run_extraction

    looked_up, handed = [], []

    async def extract(raw):
        handed.append(raw.mime_type)
        return ExtractionResult(
            source_uri=raw.source_uri,
            mime_type=raw.mime_type,
            derivatives=[Derivative(type="text", content="words")],
            extraction_method="docx",
        )

    extractor = MagicMock()
    extractor.name = "docx"
    extractor.extract = extract
    registry = MagicMock()
    registry.get_extractor.side_effect = lambda mime: looked_up.append(mime) or extractor
    monkeypatch.setattr(ExtractorRegistry, "default", classmethod(lambda cls, **kw: registry))

    storage = MagicMock()
    storage.download_to_file.side_effect = lambda path, f: f.write(b"PK docx bytes")
    storage.upload.side_effect = lambda bucket_id, path, **kw: f"{bucket_id}/{path}"
    source = {
        "id": "src-1",
        "name": "quarterly",
        "file_type": "docx",
        "storage_path": "sources/src-1/quarterly.docx",
    }

    asyncio.run(run_extraction(storage, source, "sources", extraction_model="auto"))

    docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert looked_up == [docx]
    assert handed == [docx]


def test_documentation_sources_are_stored_as_markdown(monkeypatch):
    """The docs knowledge base writes its sources itself; they carry the same
    MIME type an uploaded .md file gets."""
    from agentic_project_service.routes import knowledge_bases
    from agentic_project_service.services import docs_refresh
    from agentic_project_service.tasks import extraction as extraction_task

    session = _Session()
    monkeypatch.setattr(extraction_task, "update_source_extraction_result", MagicMock())
    monkeypatch.setattr(knowledge_bases, "index_source_into_kb", MagicMock(return_value={}))
    doc = docs_refresh.DocRecord(key="docs:a.md", title="A", content="# A", content_hash="h")

    docs_refresh._ingest_markdown_source(session, MagicMock(), "kb-1", doc)

    (insert,) = session.inserts
    assert insert["file_type"] == "text/markdown"
