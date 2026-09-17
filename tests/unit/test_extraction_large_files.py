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
import logging
import tempfile
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
        self.content_types: dict[str, str] = {}
        self.upload_attempts = 0

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
        self.upload_attempts += 1
        if self.fail_upload_on and self.fail_upload_on in path:
            raise StorageError("upload refused")
        self.uploads.append((path, len(file_data)))
        self.content_types[path] = content_type
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


def _run_task(
    monkeypatch,
    source,
    task_id="task-1",
    size=1000,
    gate=None,
    attempts=None,
    run=None,
    retries=0,
):
    from agentic_project_service.tasks import extraction as ext_mod

    statuses = []
    ran = []
    recorded = []
    requeued = []
    monkeypatch.setattr(ext_mod, "get_source", lambda _id: source)
    monkeypatch.setattr(ext_mod, "update_source_status", lambda *a, **kw: statuses.append((a, kw)))
    monkeypatch.setattr(
        ext_mod,
        "record_extraction_interruption",
        lambda sid, tid, n: recorded.append((sid, tid, n)),
    )
    monkeypatch.setattr(ext_mod, "update_source_extraction_result", lambda *a, **kw: True)
    storage = MagicMock()
    storage.object_size.return_value = size
    monkeypatch.setattr(ext_mod, "get_storage", lambda: storage)
    monkeypatch.setattr(ext_mod, "large_extraction_gate", gate or _gate())
    monkeypatch.setattr(ext_mod, "extraction_attempts", attempts or _attempts("worker-b"))
    monkeypatch.setattr(
        ext_mod, "_requeue", lambda task, countdown: requeued.append((task.request.id, countdown))
    )

    async def fake_run(*a, **kw):
        ran.append(True)
        if run is not None:
            return await run()
        return {}, {"extraction_method": "fitz", "page_count": 1, "char_count": 100}

    monkeypatch.setattr(ext_mod, "run_extraction", fake_run)
    _run_task.requeued = requeued
    _run_task.statuses = statuses
    ext_mod.extract_source.push_request(id=task_id, retries=retries)
    try:
        result = ext_mod.extract_source.run(source_id="src-1", bucket_id="sources")
    finally:
        ext_mod.extract_source.pop_request()
    return result, statuses, ran, recorded


_REDIS = {}


def _redis():
    import fakeredis

    return _REDIS.setdefault("client", fakeredis.FakeStrictRedis())


@pytest.fixture(autouse=True)
def _fresh_redis():
    import fakeredis

    _REDIS["client"] = fakeredis.FakeStrictRedis()
    yield
    _REDIS.clear()


def _gate():
    from agentic_project_service.services.extraction_gate import LargeExtractionGate

    return LargeExtractionGate(redis_client=_redis(), heartbeat=False)


def _attempts(incarnation):
    from agentic_project_service.services.extraction_gate import ExtractionAttempts

    return ExtractionAttempts(redis_client=_redis(), incarnation=incarnation)


def test_first_delivery_runs_without_recording_an_interruption(monkeypatch, mock_db_session):
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    # The dispatching route writes celery_task_id right after .delay(), so a
    # normal first delivery already finds its own id on a pending source.
    result, _statuses, ran, recorded = _run_task(monkeypatch, _task_source("pending", "task-1"))

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
    # A whole-worker stop is not always this file's fault: a restart looks the same.
    assert "restart" in args[2]
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


def test_the_interruption_cap_is_two():
    """Pinned by value: tests elsewhere read the constant, so a change to it
    would otherwise pass unnoticed."""
    from agentic_project_service.tasks.extraction import MAX_EXTRACTION_INTERRUPTIONS

    assert MAX_EXTRACTION_INTERRUPTIONS == 2


def test_extracting_under_another_task_id_is_not_an_interruption(monkeypatch, mock_db_session):
    """The source is being extracted by some other task (a re-extract raced
    this delivery): nothing about *this* task died."""
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    result, _statuses, ran, recorded = _run_task(
        monkeypatch, _task_source("extracting", "task-other"), task_id="task-1"
    )

    assert recorded == []
    assert ran == [True]
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Only the plausible cause of a whole-worker kill is charged
# ---------------------------------------------------------------------------


def test_a_smaller_task_killed_alongside_a_larger_one_is_not_charged(monkeypatch, mock_db_session):
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    dead_worker = _attempts("worker-a")
    dead_worker.begin("task-big", 400_000_000, None)
    dead_worker.begin("task-1", 20_000, None)
    # worker-a is killed. task-1 is redelivered to another worker.

    result, statuses, ran, recorded = _run_task(
        monkeypatch, _task_source("extracting", "task-1"), size=20_000
    )

    assert recorded == []
    assert ran == [True]
    assert result["status"] == "success"
    assert all(args[1] != "failed" for args, _ in statuses)


def test_a_bystander_is_never_failed_however_often_the_worker_dies(monkeypatch, mock_db_session):
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    for kill in range(5):
        dead_worker = _attempts(f"worker-{kill}")
        dead_worker.begin(f"task-big-{kill}", 400_000_000, None)
        dead_worker.begin("task-1", 20_000, None)
        source = _task_source(
            "extracting",
            "task-1",
            {"extraction_interruptions": 1, "extraction_interrupted_task": "task-1"},
        )
        result, _statuses, _ran, recorded = _run_task(monkeypatch, source, size=20_000)
        assert recorded == []
        assert result["status"] == "success"


def test_the_largest_task_killed_is_charged(monkeypatch, mock_db_session):
    monkeypatch.setenv("EXTRACTION_LARGE_FILE_BYTES", str(50 * 1024 * 1024))
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    dead_worker = _attempts("worker-a")
    dead_worker.begin("task-1", 400_000_000, "old-slot")
    dead_worker.begin("task-small", 20_000, None)

    result, _statuses, ran, recorded = _run_task(
        monkeypatch, _task_source("extracting", "task-1"), size=400_000_000
    )

    assert recorded == [("src-1", "task-1", 1)]
    assert ran == [True]


def test_a_worker_that_dies_running_only_small_tasks_still_charges_one(
    monkeypatch, mock_db_session
):
    """Something killed it: the largest in flight is charged, so a small file
    that crashes the worker every time is still failed eventually."""
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    dead_worker = _attempts("worker-a")
    dead_worker.begin("task-1", 30_000, None)
    dead_worker.begin("task-smaller", 20_000, None)

    _result, _statuses, _ran, recorded = _run_task(
        monkeypatch, _task_source("extracting", "task-1"), size=30_000
    )

    assert recorded == [("src-1", "task-1", 1)]


def test_the_attempt_is_recorded_while_running_and_forgotten_after(monkeypatch, mock_db_session):
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    attempts = _attempts("worker-b")
    seen = {}

    async def run():
        seen["during"] = _attempts("observer").previous("task-1")
        return {}, {"extraction_method": "fitz", "page_count": 1, "char_count": 100}

    _run_task(monkeypatch, _task_source("pending", "task-1"), size=1234, attempts=attempts, run=run)

    assert seen["during"] is not None and seen["during"].size == 1234
    assert _attempts("observer").previous("task-1") is None


def test_the_attempt_is_forgotten_after_an_extraction_error(monkeypatch, mock_db_session):
    async def run():
        raise ValueError("unreadable")

    result, _statuses, _ran, _recorded = _run_task(
        monkeypatch, _task_source("pending", "task-1"), run=run
    )

    assert result["status"] == "error"
    assert _attempts("observer").previous("task-1") is None


# ---------------------------------------------------------------------------
# At most N large extractions at once, across every worker of the project
# ---------------------------------------------------------------------------

LARGE = 60 * 1024 * 1024
SMALL = 1024 * 1024


@pytest.fixture
def large_limits(monkeypatch):
    monkeypatch.setenv("EXTRACTION_LARGE_FILE_BYTES", str(50 * 1024 * 1024))
    monkeypatch.setenv("EXTRACTION_LARGE_MAX_CONCURRENT", "1")


def _slot_is_free():
    probe = _gate().try_acquire("probe")
    if probe is None:
        return False
    probe.release()
    return True


def test_a_large_file_holds_a_slot_while_it_extracts_and_releases_it(
    monkeypatch, mock_db_session, large_limits
):
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    held = {}

    async def run():
        held["during"] = not _slot_is_free()
        return {}, {"extraction_method": "fitz", "page_count": 1, "char_count": 100}

    result, _s, _r, _rec = _run_task(
        monkeypatch, _task_source("pending", "task-1"), size=LARGE, run=run
    )

    assert result["status"] == "success"
    assert held["during"] is True
    assert _slot_is_free()


def test_the_slot_is_released_when_extraction_fails(monkeypatch, mock_db_session, large_limits):
    async def run():
        raise ValueError("unreadable")

    result, _s, _r, _rec = _run_task(
        monkeypatch, _task_source("pending", "task-1"), size=LARGE, run=run
    )

    assert result["status"] == "error"
    assert _slot_is_free()


def test_the_slot_is_released_when_the_task_raises(monkeypatch, mock_db_session, large_limits):
    async def run():
        raise StorageError("storage down")

    with pytest.raises(StorageError):
        _run_task(monkeypatch, _task_source("pending", "task-1"), size=LARGE, run=run)

    assert _slot_is_free()


def test_the_slot_is_released_when_the_interruption_cap_fails_the_source(
    monkeypatch, mock_db_session, large_limits
):
    source = _task_source(
        "extracting",
        "task-1",
        {"extraction_interruptions": 1, "extraction_interrupted_task": "task-1"},
    )
    result, _s, ran, _rec = _run_task(monkeypatch, source, size=LARGE)

    assert result["status"] == "error"
    assert ran == []
    assert _slot_is_free()


def test_a_large_file_waits_by_requeueing_when_every_slot_is_taken(
    monkeypatch, mock_db_session, large_limits
):
    from agentic_project_service.tasks.extraction import LARGE_EXTRACTION_REQUEUE_SECONDS

    busy = _gate().try_acquire("someone-else")
    assert busy is not None

    result, statuses, ran, recorded = _run_task(
        monkeypatch, _task_source("pending", "task-1"), size=LARGE
    )

    assert result["status"] == "deferred"
    assert ran == []
    assert statuses == []
    assert recorded == []
    assert _run_task.requeued == [("task-1", LARGE_EXTRACTION_REQUEUE_SECONDS)]
    assert LARGE_EXTRACTION_REQUEUE_SECONDS > 0


def test_waiting_for_a_slot_is_not_counted_as_an_interruption(
    monkeypatch, mock_db_session, large_limits
):
    """A redelivered large task that has to wait must not be charged on every
    wake-up: the source stays `extracting` under its id while it waits."""
    _gate().try_acquire("someone-else")
    source = _task_source(
        "extracting",
        "task-1",
        {"extraction_interruptions": 1, "extraction_interrupted_task": "task-1"},
    )
    for _ in range(3):
        result, statuses, ran, recorded = _run_task(monkeypatch, source, size=LARGE)
        assert result["status"] == "deferred"
        assert (statuses, ran, recorded) == ([], [], [])


def test_a_small_file_is_never_gated(monkeypatch, mock_db_session, large_limits):
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    _gate().try_acquire("someone-else")  # every large slot is taken

    result, _s, ran, _rec = _run_task(monkeypatch, _task_source("pending", "task-1"), size=SMALL)

    assert result["status"] == "success"
    assert ran == [True]
    assert _run_task.requeued == []


def test_a_small_file_does_not_touch_the_gate(monkeypatch, mock_db_session, large_limits):
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    gate = MagicMock()

    result, _s, ran, _rec = _run_task(
        monkeypatch, _task_source("pending", "task-1"), size=SMALL, gate=gate
    )

    assert result["status"] == "success"
    gate.try_acquire.assert_not_called()


def test_small_files_keep_flowing_while_large_ones_queue(
    monkeypatch, mock_db_session, large_limits
):
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    running_large = _gate().try_acquire("large-running")
    assert running_large is not None

    outcomes = []
    for i in range(5):
        size = LARGE if i % 2 else SMALL
        result, _s, _r, _rec = _run_task(
            monkeypatch, _task_source("pending", f"task-{i}"), task_id=f"task-{i}", size=size
        )
        outcomes.append((size, result["status"]))

    assert [status for size, status in outcomes if size == SMALL] == ["success"] * 3
    assert [status for size, status in outcomes if size == LARGE] == ["deferred"] * 2


def test_a_duplicate_delivery_of_a_task_still_running_waits_instead_of_counting(
    monkeypatch, mock_db_session, large_limits
):
    """The broker redelivers a message after its visibility timeout even when
    the first delivery is still running. The running attempt's slot is live,
    so this delivery neither runs nor charges an interruption."""
    monkeypatch.setenv("EXTRACTION_LARGE_MAX_CONCURRENT", "2")
    running = _gate().try_acquire("task-1")
    _attempts("worker-a").begin("task-1", LARGE, running.token)

    result, statuses, ran, recorded = _run_task(
        monkeypatch, _task_source("extracting", "task-1"), size=LARGE
    )

    assert result["status"] == "deferred"
    assert (statuses, ran, recorded) == ([], [], [])


def test_requeue_keeps_the_task_id_and_does_not_spend_a_retry(monkeypatch):
    from celery.canvas import Signature

    from agentic_project_service.tasks import extraction as ext_mod

    sent = []
    monkeypatch.setattr(
        Signature, "apply_async", lambda self, *a, **kw: sent.append(dict(self.options))
    )
    ext_mod.extract_source.push_request(
        id="task-1", retries=2, args=["src-1", "sources"], kwargs={"extraction_model": "auto"}
    )
    try:
        ext_mod._requeue(ext_mod.extract_source, 45)
    finally:
        ext_mod.extract_source.pop_request()

    (options,) = sent
    assert options["task_id"] == "task-1"
    assert options["retries"] == 2
    assert options["countdown"] == 45


def test_a_file_whose_size_is_unknown_is_gated(monkeypatch, mock_db_session, large_limits):
    _gate().try_acquire("someone-else")

    result, _s, ran, _rec = _run_task(monkeypatch, _task_source("pending", "task-1"), size=None)

    assert result["status"] == "deferred"
    assert ran == []


# ---------------------------------------------------------------------------
# Page-image storage failures
# ---------------------------------------------------------------------------


def test_after_one_page_image_fails_to_store_later_pages_fail_without_uploading(monkeypatch):
    storage = _Storage(fail_upload_on="image_page1_")
    raised = []

    async def extract(raw):
        sink = raw.metadata["page_image_sink"]
        for page in (1, 2, 3):
            try:
                sink(Derivative(type="image", content=b"png", format="png", page=page))
            except StorageError as e:
                raised.append((page, e))
        raise ExtractionError("aborted", extractor_name="pdf") from raised[-1][1]

    _use_extractor(monkeypatch, extract)
    with pytest.raises(StorageError):
        _run(storage)

    assert [page for page, _ in raised] == [1, 2, 3]
    assert storage.upload_attempts == 1
    assert storage.uploads == []


def test_page_image_failure_is_reported_as_such(monkeypatch):
    from agentic_project_service.tasks.extraction import PageImageStorageError

    storage = _Storage(fail_upload_on="/image/")

    async def extract(raw):
        raw.metadata["page_image_sink"](
            Derivative(type="image", content=b"png", format="png", page=7)
        )

    _use_extractor(monkeypatch, extract)
    with pytest.raises(PageImageStorageError) as info:
        _run(storage)

    assert isinstance(info.value, StorageError)
    assert "page image" in str(info.value).lower()
    assert "upload refused" in str(info.value)


def test_an_earlier_recovered_sink_error_is_not_blamed_for_a_later_failure(monkeypatch):
    """The storage error must be in the failure's own cause chain; otherwise the
    real failure is reported and not retried as a storage problem."""
    storage = _Storage(fail_upload_on="image_page1_")

    async def extract(raw):
        try:
            raw.metadata["page_image_sink"](
                Derivative(type="image", content=b"png", format="png", page=1)
            )
        except StorageError:
            pass  # the engine moved on without page images

        raise ExtractionError("Mistral OCR rejected the document", extractor_name="pdf")

    _use_extractor(monkeypatch, extract)
    with pytest.raises(ExtractionError, match="rejected"):
        _run(storage)


def test_a_sink_failure_in_the_implicit_context_is_still_found(monkeypatch):
    storage = _Storage(fail_upload_on="/image/")

    async def extract(raw):
        try:
            raw.metadata["page_image_sink"](
                Derivative(type="image", content=b"png", format="png", page=1)
            )
        except StorageError:
            raise RuntimeError("page image sink failed; extraction aborted")

    _use_extractor(monkeypatch, extract)
    with pytest.raises(StorageError, match="upload refused"):
        _run(storage)


def test_a_page_image_failure_fails_the_source_with_a_clear_error_and_retries(
    monkeypatch, mock_db_session
):
    """Retried through the task's bounded retry budget, never re-run forever."""
    from agentic_project_service.tasks import extraction as ext_mod

    async def run():
        raise ext_mod.PageImageStorageError(
            "Storing the page image of page 3 failed: upload refused"
        )

    # Called directly, Task.retry re-raises the exception instead of enqueueing.
    with pytest.raises(ext_mod.PageImageStorageError):
        _run_task(monkeypatch, _task_source("pending", "task-1"), run=run)

    failed = [args for args, _kw in _run_task.statuses if args[1] == "failed"]
    assert len(failed) == 1
    assert "page image of page 3 failed: upload refused" in failed[0][2]
    assert ext_mod.extract_source.max_retries == 3


# ---------------------------------------------------------------------------
# Result shape details
# ---------------------------------------------------------------------------


def test_streamed_pages_are_listed_in_page_order(monkeypatch):
    storage = _Storage()

    async def extract(raw):
        sink = raw.metadata["page_image_sink"]
        for page in (3, 1, 2):
            sink(Derivative(type="image", content=b"png", format="png", page=page))
        return _result([Derivative(type="markdown", content="md")], pages=3)

    _use_extractor(monkeypatch, extract)
    derivatives, _ = _run(storage)

    assert [r["page"] for r in derivatives["image"]] == [1, 2, 3]


def test_a_jpg_page_is_stored_as_image_jpeg(monkeypatch):
    storage = _Storage()

    async def extract(raw):
        raw.metadata["page_image_sink"](
            Derivative(type="image", content=b"jpg", format="jpg", page=1)
        )
        raw.metadata["page_image_sink"](
            Derivative(type="image", content=b"png", format="png", page=2)
        )
        return _result([Derivative(type="markdown", content="md")])

    _use_extractor(monkeypatch, extract)
    _run(storage)

    by_page = {
        path.rsplit("/", 1)[-1].split("_")[1]: (path.rsplit(".", 1)[-1], ct)
        for path, ct in storage.content_types.items()
    }
    assert by_page["page1"] == ("jpg", "image/jpeg")
    assert by_page["page2"] == ("png", "image/png")


_REAL_TEMPORARY_FILE = tempfile.TemporaryFile


class _TrackingTempFiles:
    def __init__(self):
        self.files = []

    def __call__(self, *a, **kw):
        f = _REAL_TEMPORARY_FILE(*a, **kw)
        self.files.append(f)
        return f


def test_the_download_spool_is_closed_after_extraction(monkeypatch):
    from agentic_project_service.tasks import extraction as ext_mod

    tracker = _TrackingTempFiles()
    monkeypatch.setattr(ext_mod.tempfile, "TemporaryFile", tracker)

    open_during_extraction = []

    async def extract(raw):
        open_during_extraction.extend(not f.closed for f in tracker.files)
        return _result([Derivative(type="text", content="t")], method="fitz")

    _use_extractor(monkeypatch, extract)
    _run(_Storage())

    assert len(tracker.files) == 1
    # Read back and closed before extraction starts: the bytes are in memory
    # by then, so the disk copy would only be a second copy.
    assert open_during_extraction == [False]
    assert tracker.files[0].closed


def test_the_download_spool_is_closed_when_the_download_fails(monkeypatch):
    from agentic_project_service.tasks import extraction as ext_mod

    tracker = _TrackingTempFiles()
    monkeypatch.setattr(ext_mod.tempfile, "TemporaryFile", tracker)
    storage = _Storage()

    def broken_download(storage_path, fileobj):
        fileobj.write(b"partial")
        raise StorageError("connection reset")

    storage.download_to_file = broken_download
    _use_extractor(monkeypatch, None)

    with pytest.raises(StorageError):
        _run(storage)

    assert len(tracker.files) == 1
    assert tracker.files[0].closed


# ---------------------------------------------------------------------------
# An engine that ignores the sink must not go unnoticed
# ---------------------------------------------------------------------------


def test_an_engine_that_does_not_stream_page_images_is_warned_about(monkeypatch, caplog):
    from agentic_project_service.tasks import extraction as ext_mod

    monkeypatch.setattr(ext_mod, "engine_streams_page_images", lambda: False)
    monkeypatch.setattr(ext_mod, "_engine_sink_warning_logged", False)

    async def extract(raw):
        return _result([Derivative(type="text", content="t")], method="fitz")

    _use_extractor(monkeypatch, extract)
    with caplog.at_level(logging.WARNING, logger=ext_mod.logger.name):
        _run(_Storage())

    assert "page_image_sink" in caplog.text
    assert "powabase-agentic" in caplog.text


def test_the_engine_warning_repeats_for_large_files_only(monkeypatch, caplog):
    from agentic_project_service.tasks import extraction as ext_mod

    monkeypatch.setenv("EXTRACTION_LARGE_FILE_BYTES", "100")
    monkeypatch.setattr(ext_mod, "engine_streams_page_images", lambda: False)
    monkeypatch.setattr(ext_mod, "_engine_sink_warning_logged", False)

    async def extract(raw):
        return _result([Derivative(type="text", content="t")], method="fitz")

    _use_extractor(monkeypatch, extract)
    with caplog.at_level(logging.WARNING, logger=ext_mod.logger.name):
        _run(_Storage(data=b"small"))
        _run(_Storage(data=b"small"))
        _run(_Storage(data=b"x" * 101))

    assert caplog.text.count("ignores page_image_sink") == 2


def test_an_engine_that_streams_page_images_is_not_warned_about(monkeypatch, caplog):
    from agentic_project_service.tasks import extraction as ext_mod

    monkeypatch.setattr(ext_mod, "engine_streams_page_images", lambda: True)

    async def extract(raw):
        return _result([Derivative(type="text", content="t")], method="fitz")

    _use_extractor(monkeypatch, extract)
    with caplog.at_level(logging.WARNING, logger=ext_mod.logger.name):
        _run(_Storage())

    assert "page_image_sink" not in caplog.text


def test_engine_capability_is_detected_from_the_installed_engine(monkeypatch):
    import types

    from agentic_project_service.tasks import extraction as ext_mod

    streaming = types.ModuleType("agentic.ingest.extractor.pdf")
    streaming._page_image_sink = lambda raw: None
    monkeypatch.setitem(__import__("sys").modules, "agentic.ingest.extractor.pdf", streaming)
    ext_mod.engine_streams_page_images.cache_clear()
    try:
        assert ext_mod.engine_streams_page_images() is True
        legacy = types.ModuleType("agentic.ingest.extractor.pdf")
        monkeypatch.setitem(__import__("sys").modules, "agentic.ingest.extractor.pdf", legacy)
        ext_mod.engine_streams_page_images.cache_clear()
        assert ext_mod.engine_streams_page_images() is False
    finally:
        ext_mod.engine_streams_page_images.cache_clear()


# ---------------------------------------------------------------------------
# Derivatives a re-extract no longer produces are removed
# ---------------------------------------------------------------------------


def _run_task_with_derivatives(monkeypatch, old, new, applied=True, delete_error=None):
    from agentic_project_service.tasks import extraction as ext_mod

    storage = MagicMock()
    storage.object_size.return_value = 1000
    if delete_error:
        storage.delete.side_effect = delete_error
    source = {**_task_source("pending", "task-1"), "derivatives": old}
    monkeypatch.setattr(ext_mod, "get_source", lambda _id: source)
    monkeypatch.setattr(ext_mod, "update_source_status", lambda *a, **kw: None)
    monkeypatch.setattr(ext_mod, "update_source_extraction_result", lambda *a, **kw: applied)
    monkeypatch.setattr(ext_mod, "get_storage", lambda: storage)
    monkeypatch.setattr(ext_mod, "large_extraction_gate", _gate())
    monkeypatch.setattr(ext_mod, "extraction_attempts", _attempts("worker-b"))

    async def fake_run(*a, **kw):
        return new, {"extraction_method": "fitz", "page_count": 1, "char_count": 100}

    monkeypatch.setattr(ext_mod, "run_extraction", fake_run)
    ext_mod.extract_source.push_request(id="task-1")
    try:
        result = ext_mod.extract_source.run(source_id="src-1", bucket_id="sources")
    finally:
        ext_mod.extract_source.pop_request()
    return result, storage


OLD_DERIVATIVES = {
    "markdown": [{"storage_path": "sources/src-1/derivatives/markdown/content.md"}],
    "image": [
        {"storage_path": "sources/src-1/derivatives/image/image_page1_1.png", "page": 1},
        {"storage_path": "sources/src-1/derivatives/image/image_page2_2.png", "page": 2},
    ],
}
NEW_DERIVATIVES = {
    "markdown": [{"storage_path": "sources/src-1/derivatives/markdown/content.md"}],
    "image": [
        {"storage_path": "sources/src-1/derivatives/image/image_page1.png", "page": 1},
        {"storage_path": "sources/src-1/derivatives/image/image_page2.png", "page": 2},
    ],
}


def test_derivatives_the_new_extraction_replaced_are_deleted(monkeypatch, mock_db_session):
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    result, storage = _run_task_with_derivatives(monkeypatch, OLD_DERIVATIVES, NEW_DERIVATIVES)

    assert result["status"] == "success"
    storage.delete.assert_called_once()
    bucket, paths = storage.delete.call_args.args
    assert bucket == "sources"
    assert sorted(paths) == [
        "src-1/derivatives/image/image_page1_1.png",
        "src-1/derivatives/image/image_page2_2.png",
    ]


def test_nothing_is_deleted_when_the_result_was_not_written(monkeypatch, mock_db_session):
    """Cancelled between the check and the write: the old derivatives are
    still the source's current ones."""
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    _result, storage = _run_task_with_derivatives(
        monkeypatch, OLD_DERIVATIVES, NEW_DERIVATIVES, applied=False
    )

    storage.delete.assert_not_called()


def test_only_this_sources_derivatives_are_ever_deleted(monkeypatch, mock_db_session):
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    old = {
        "image": [
            {"storage_path": "sources/src-2/derivatives/image/other.png"},
            {"storage_path": "sources/src-1/original/book.pdf"},
            {"storage_path": "elsewhere/src-1/derivatives/image/x.png"},
            {"storage_path": None},
        ]
    }
    _result, storage = _run_task_with_derivatives(monkeypatch, old, {})

    storage.delete.assert_not_called()


def test_a_failed_cleanup_does_not_fail_the_extraction(monkeypatch, mock_db_session):
    mock_db_session.execute.return_value.scalar.return_value = "extracting"
    result, _storage = _run_task_with_derivatives(
        monkeypatch, OLD_DERIVATIVES, NEW_DERIVATIVES, delete_error=StorageError("nope")
    )

    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Storage: object_size
# ---------------------------------------------------------------------------


def _client(monkeypatch):
    from agentic_project_service.services import storage as storage_mod

    monkeypatch.setenv("SERVICE_ROLE_KEY", "k")
    return storage_mod.SupabaseStorage(url="http://storage.test")


def test_object_size_reads_the_length_without_the_body(monkeypatch):
    client = _client(monkeypatch)
    state = {"body_read": False, "closed": False}

    def stream(bucket, path, chunk_size=8192):
        assert (bucket, path) == ("sources", "a/b.pdf")
        try:
            yield "367000000"
            state["body_read"] = True
            yield b"x"
        finally:
            state["closed"] = True

    opened = []  # held here so garbage collection cannot close it for the code

    def tracked(bucket, path, chunk_size=8192):
        gen = stream(bucket, path, chunk_size)
        opened.append(gen)
        return gen

    monkeypatch.setattr(client, "stream_download", tracked)

    assert client.object_size("sources/a/b.pdf") == 367000000
    assert len(opened) == 1
    assert state == {"body_read": False, "closed": True}


@pytest.mark.parametrize("header", [None, "", "abc"])
def test_object_size_is_none_without_a_usable_length(monkeypatch, header):
    client = _client(monkeypatch)

    def stream(bucket, path, chunk_size=8192):
        yield header

    monkeypatch.setattr(client, "stream_download", stream)

    assert client.object_size("sources/a/b.pdf") is None


def test_object_size_rejects_a_path_without_bucket(monkeypatch):
    with pytest.raises(StorageError):
        _client(monkeypatch).object_size("no-bucket")
