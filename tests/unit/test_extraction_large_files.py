"""Extraction of very large files: bounded memory and no endless redelivery.

A scanned book of 1,500 pages is ~320 MiB of PDF and renders to ~2.6 GiB of
page PNGs. Three things keep one such file from taking a worker down:

* the source is spooled to disk while it downloads, not buffered twice;
* page images are uploaded as the extractor renders them, through the
  ``page_image_sink`` the task passes in ``RawContent.metadata``, instead of
  all being held until extraction returns;
* a task whose worker died mid-extraction (out of memory, most likely) is
  failed after a bounded number of redeliveries rather than looping.
"""

import asyncio
import io
from unittest.mock import MagicMock

import pytest

from agentic.ingest import Derivative, ExtractionError, ExtractionResult
from agentic_project_service.services.storage import StorageError


SOURCE = {
    "id": "src-1",
    "name": "book.pdf",
    "file_type": "application/pdf",
    "storage_path": "sources/src-1/book.pdf",
}


class _Storage:
    def __init__(self, data=b"%PDF-1.4 book", fail_upload_on=None):
        self.data = data
        self.uploads: list[tuple[str, int]] = []
        self.fail_upload_on = fail_upload_on

    def download_to_file(self, storage_path, fileobj):
        assert storage_path == SOURCE["storage_path"]
        for i in range(0, len(self.data), 4):
            fileobj.write(self.data[i : i + 4])
        return len(self.data)

    def download_from_path(self, storage_path):  # pragma: no cover - must not be used
        raise AssertionError("the source must be streamed, not buffered")

    def upload(
        self, bucket_id, path, file_data, content_type="application/octet-stream", upsert=True
    ):
        if self.fail_upload_on and self.fail_upload_on in path:
            raise StorageError("upload refused")
        self.uploads.append((path, len(file_data)))
        return f"{bucket_id}/{path}"


def _use_extractor(monkeypatch, extract):
    from agentic.ingest import ExtractorRegistry

    extractor = MagicMock()
    extractor.name = "pdf"
    extractor.extract = extract
    registry = MagicMock()
    registry.get_extractor.return_value = extractor
    monkeypatch.setattr(ExtractorRegistry, "default", classmethod(lambda cls, **kw: registry))


def _result(derivatives, method="lighton_ocr", pages=2):
    return ExtractionResult(
        source_uri=SOURCE["storage_path"],
        mime_type="application/pdf",
        derivatives=derivatives,
        auto_metadata={"page_count": pages, "char_count": 10},
        extraction_method=method,
    )


def _run(storage):
    from agentic_project_service.tasks.extraction import run_extraction

    return asyncio.run(run_extraction(storage, dict(SOURCE), "sources", extraction_model="auto"))


def test_source_is_streamed_and_handed_to_the_extractor_intact(monkeypatch):
    storage = _Storage(data=b"%PDF-1.4 " + b"x" * 1001)
    seen = {}

    async def extract(raw):
        seen["content"] = raw.content
        return _result([Derivative(type="text", content="t")], method="fitz")

    _use_extractor(monkeypatch, extract)
    _run(storage)

    assert seen["content"] == storage.data


def test_page_images_are_uploaded_as_the_extractor_renders_them(monkeypatch):
    storage = _Storage()
    uploads_seen_by_page = []

    async def extract(raw):
        sink = raw.metadata["page_image_sink"]
        for page in (1, 2):
            sink(Derivative(type="image", content=b"png" * page, format="png", page=page))
            uploads_seen_by_page.append(len(storage.uploads))
        return _result(
            [
                Derivative(type="markdown", content="md"),
                Derivative(type="page_text", content="a", page=1),
                Derivative(type="page_text", content="b", page=2),
            ]
        )

    _use_extractor(monkeypatch, extract)
    derivatives, auto_metadata = _run(storage)

    # Each image was stored before the extractor moved on to the next page.
    assert uploads_seen_by_page == [1, 2]
    assert [r["page"] for r in derivatives["image"]] == [1, 2]
    assert all(
        r["storage_path"].startswith("sources/src-1/derivatives/image/")
        for r in derivatives["image"]
    )
    # Same record shape as images returned in the result: no empty metadata key.
    assert all("metadata" not in r for r in derivatives["image"])
    assert len(derivatives["page_text"]) == 2
    assert auto_metadata["derivative_count"] == 5


def test_a_page_delivered_twice_keeps_one_record(monkeypatch):
    """A method that fails after streaming pages is followed by the next
    method in the chain, which streams them again."""
    storage = _Storage()

    async def extract(raw):
        sink = raw.metadata["page_image_sink"]
        for _attempt in range(2):
            for page in (1, 2):
                sink(Derivative(type="image", content=b"png", format="png", page=page))
        return _result([Derivative(type="markdown", content="md")])

    _use_extractor(monkeypatch, extract)
    derivatives, _ = _run(storage)

    assert [r["page"] for r in derivatives["image"]] == [1, 2]


def test_images_returned_in_the_result_are_still_stored(monkeypatch):
    """An engine that does not know the sink returns images the old way."""
    storage = _Storage()

    async def extract(raw):
        return _result(
            [
                Derivative(type="markdown", content="md"),
                Derivative(type="image", content=b"png", format="png", page=1),
            ]
        )

    _use_extractor(monkeypatch, extract)
    derivatives, _ = _run(storage)

    assert [r["page"] for r in derivatives["image"]] == [1]
    assert any("/derivatives/image/" in path for path, _ in storage.uploads)


def test_storage_failure_inside_the_sink_surfaces_as_storage_error(monkeypatch):
    """The engine may wrap the sink's exception while it tries other methods;
    the task still has to see a StorageError so it retries."""
    storage = _Storage(fail_upload_on="/image/")

    async def extract(raw):
        try:
            raw.metadata["page_image_sink"](
                Derivative(type="image", content=b"png", format="png", page=1)
            )
        except Exception as e:
            raise ExtractionError("All extraction methods failed", extractor_name="pdf") from e

    _use_extractor(monkeypatch, extract)
    with pytest.raises(StorageError, match="upload refused"):
        _run(storage)


# ---------------------------------------------------------------------------
# Storage: download_to_file
# ---------------------------------------------------------------------------


def test_download_to_file_streams_chunks(monkeypatch):
    from agentic_project_service.services import storage as storage_mod

    monkeypatch.setenv("SERVICE_ROLE_KEY", "k")
    client = storage_mod.SupabaseStorage(url="http://storage.test")
    chunks = [b"abc", b"def", b"g"]
    monkeypatch.setattr(
        client, "stream_download", lambda bucket, path, chunk_size=8192: iter(["7", *chunks])
    )
    out = io.BytesIO()

    written = client.download_to_file("sources/a/b.pdf", out)

    assert written == 7
    assert out.getvalue() == b"abcdefg"


def test_download_to_file_rejects_a_path_without_bucket(monkeypatch):
    from agentic_project_service.services import storage as storage_mod

    monkeypatch.setenv("SERVICE_ROLE_KEY", "k")
    client = storage_mod.SupabaseStorage(url="http://storage.test")
    with pytest.raises(StorageError):
        client.download_to_file("no-bucket", io.BytesIO())


# ---------------------------------------------------------------------------
# Redelivery after the worker died mid-extraction
# ---------------------------------------------------------------------------


def _task_source(status, celery_task_id, auto_metadata=None):
    return {
        **SOURCE,
        "extraction_status": status,
        "celery_task_id": celery_task_id,
        "derivatives": {},
        "metadata": {},
        "auto_metadata": auto_metadata or {},
    }


def _run_task(monkeypatch, source, task_id="task-1"):
    from agentic_project_service.tasks import extraction as ext_mod

    statuses = []
    ran = []
    recorded = []
    monkeypatch.setattr(ext_mod, "get_source", lambda _id: source)
    monkeypatch.setattr(ext_mod, "update_source_status", lambda *a, **kw: statuses.append((a, kw)))
    monkeypatch.setattr(
        ext_mod,
        "record_extraction_interruption",
        lambda sid, tid, n: recorded.append((sid, tid, n)),
    )
    monkeypatch.setattr(ext_mod, "update_source_extraction_result", lambda *a, **kw: None)
    monkeypatch.setattr(ext_mod, "get_storage", lambda: MagicMock())

    async def fake_run(*a, **kw):
        ran.append(True)
        return {}, {"extraction_method": "fitz", "page_count": 1, "char_count": 100}

    monkeypatch.setattr(ext_mod, "run_extraction", fake_run)
    ext_mod.extract_source.push_request(id=task_id)
    try:
        result = ext_mod.extract_source.run(source_id="src-1", bucket_id="sources")
    finally:
        ext_mod.extract_source.pop_request()
    return result, statuses, ran, recorded


def test_first_delivery_runs_without_recording_an_interruption(monkeypatch, mock_db_session):
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    result, _statuses, ran, recorded = _run_task(monkeypatch, _task_source("pending", None))

    assert ran == [True]
    assert recorded == []
    assert result["status"] == "success"


def test_redelivery_of_an_interrupted_task_is_recorded_and_retried_once(
    monkeypatch, mock_db_session
):
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    result, _statuses, ran, recorded = _run_task(monkeypatch, _task_source("extracting", "task-1"))

    assert recorded == [("src-1", "task-1", 1)]
    assert ran == [True]
    assert result["status"] == "success"


def test_repeatedly_interrupted_task_fails_without_extracting(monkeypatch, mock_db_session):
    from agentic_project_service.tasks.extraction import MAX_EXTRACTION_INTERRUPTIONS

    source = _task_source(
        "extracting",
        "task-1",
        {
            "extraction_interruptions": MAX_EXTRACTION_INTERRUPTIONS - 1,
            "extraction_interrupted_task": "task-1",
        },
    )
    result, statuses, ran, recorded = _run_task(monkeypatch, source)

    assert ran == []
    assert recorded == [("src-1", "task-1", MAX_EXTRACTION_INTERRUPTIONS)]
    assert result["status"] == "error"
    ((args, kwargs),) = statuses
    assert args[1] == "failed"
    assert "interrupted" in args[2] and "memory" in args[2]
    assert kwargs.get("error_code") == "permanent"


def test_interruptions_of_a_previous_task_do_not_count(monkeypatch, mock_db_session):
    """A re-extract is a new task; an old task's interruptions are history."""
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    source = _task_source(
        "extracting",
        "task-2",
        {"extraction_interruptions": 5, "extraction_interrupted_task": "task-1"},
    )
    result, _statuses, ran, recorded = _run_task(monkeypatch, source, task_id="task-2")

    assert recorded == [("src-1", "task-2", 1)]
    assert ran == [True]
