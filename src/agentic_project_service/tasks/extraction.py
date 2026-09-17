"""Source extraction Celery task.

Downloads source files from storage, extracts content using the agentic
ingest module, and stores derivatives back to storage.
"""

import asyncio
import functools
import importlib
import json
import logging
import tempfile

from ..celery import celery_app
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import text

from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
from ..services.extraction_gate import (
    ExtractionAttempts,
    LargeExtractionGate,
    PreviousAttempt,
    is_large_file,
)
from ..services.storage import (
    StorageError,
    SupabaseStorage,
    get_derivative_storage_path,
    get_storage,
)

logger = logging.getLogger(__name__)

# Extraction methods that perform cloud OCR (billed per page). Standard cloud
# OCR bills as ``ocr_pages``; LlamaParse bills against the higher-priced,
# separate ``advanced_ocr`` category (see _ADVANCED_OCR_EXTRACTION_METHODS).
# Non-OCR methods (fitz, pdfplumber, opendataloader, txt-native, etc.) are
# not billed at the extraction layer — they are CPU-only and bundled into
# the indexing charge that follows.
_OCR_EXTRACTION_METHODS: frozenset[str] = frozenset({"mistral_ocr", "paddleocr_vl", "lighton_ocr"})
# Cloud OCR methods billed at the advanced-OCR rate (separate catalog action).
_ADVANCED_OCR_EXTRACTION_METHODS: frozenset[str] = frozenset({"llamaparse_ocr"})

# Times a task may be found to have died mid-extraction before its source is
# failed instead of extracted again. A worker killed while extracting (out of
# memory, most likely) never acks the message (acks_late +
# reject_on_worker_lost), so the broker redelivers it — and a file that killed
# the worker once will usually kill it again, taking every other task running
# in that process with it each time.
MAX_EXTRACTION_INTERRUPTIONS = 2

# Deaths of a task that was not the likely cause (a larger file was in flight
# beside it) are counted separately, against a much higher cap: never counting
# them would let a real killer loop forever behind a stale in-flight record.
MAX_UNATTRIBUTED_INTERRUPTIONS = 5

# How long a large-file extraction waits before trying for a slot again.
LARGE_EXTRACTION_REQUEUE_SECONDS = 60

large_extraction_gate = LargeExtractionGate()
extraction_attempts = ExtractionAttempts()


class PageImageStorageError(StorageError):
    """A page image handed over by the extractor could not be stored."""


@functools.cache
def engine_streams_page_images() -> bool:
    """Whether the installed engine hands page images to ``page_image_sink``.

    Releases up to 0.3.1 ignore the sink and return every page image in the
    result, so a long scan still holds all of them in memory at once.
    """
    try:
        pdf = importlib.import_module("agentic.ingest.extractor.pdf")
    except Exception:
        return False
    return callable(getattr(pdf, "_page_image_sink", None))


_engine_sink_warning_logged = False


def _warn_if_engine_ignores_sink(source_id: str, size: int) -> None:
    """Once per process, and for every large file: an engine that ignores the
    sink leaves large scans without their memory bound, which must be visible."""
    global _engine_sink_warning_logged
    if engine_streams_page_images():
        return
    if _engine_sink_warning_logged and not is_large_file(size):
        return
    _engine_sink_warning_logged = True
    logger.warning(
        f"Source {source_id}: the installed engine (powabase-agentic {_engine_version()}) "
        f"ignores page_image_sink, so every page image of this {size}-byte file is held "
        f"in memory until extraction returns. Upgrade powabase-agentic to a release "
        f"that streams page images."
    )


def _engine_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("powabase-agentic")
    except PackageNotFoundError:
        return "unknown"


def resolve_api_key_for_model(
    model: str,
    provider_keys: dict[str, str] | None,
) -> str | None:
    """Resolve the correct API key from *provider_keys* for *model*.

    Delegates to the canonical resolver in services.ai_provider_keys_resolver.
    Signature kept stable so tasks/indexing.py callers don't change.
    """
    from ..services.ai_provider_keys_resolver import resolve_api_key_for_model as _resolve

    return _resolve(model, provider_keys or {})


def get_source(source_id: str) -> dict | None:
    """Get a source record from the database."""
    result = db.session.execute(
        text(f"""
            SELECT id, name, file_type, storage_path, extraction_status,
                   derivatives, metadata, auto_metadata, celery_task_id
            FROM "{AI_SCHEMA}".sources
            WHERE id = :id
        """),
        {"id": source_id},
    )

    row = result.fetchone()
    if not row:
        return None

    return {
        "id": str(row[0]),
        "name": row[1],
        "file_type": row[2],
        "storage_path": row[3],
        "extraction_status": row[4],
        "derivatives": row[5] or {},
        "metadata": row[6] or {},
        "auto_metadata": row[7] or {},
        "celery_task_id": row[8],
    }


def update_source_status(
    source_id: str,
    status: str,
    error_message: str | None = None,
    celery_task_id: str | None = None,
    error_code: str | None = None,
) -> None:
    """Update source extraction status."""
    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".sources
            SET extraction_status = :status,
                error_message = :error_message,
                error_code = :error_code,
                celery_task_id = :celery_task_id,
                updated_at = NOW()
            WHERE id = :id
        """),
        {
            "id": source_id,
            "status": status,
            "error_message": error_message,
            "error_code": error_code,
            "celery_task_id": celery_task_id,
        },
    )
    db.session.commit()


def record_extraction_interruption(
    source_id: str, task_id: str, count: int, unattributed: int = 0
) -> None:
    """Persist how many times *task_id* has died mid-extraction of this source:
    *count* as the likely cause, *unattributed* beside a larger file."""
    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".sources
            SET auto_metadata = COALESCE(auto_metadata, '{{}}'::jsonb) || CAST(:marker AS jsonb),
                updated_at = NOW()
            WHERE id = :id
        """),
        {
            "id": source_id,
            "marker": json.dumps(
                {
                    "extraction_interruptions": count,
                    "extraction_unattributed_interruptions": unattributed,
                    "extraction_interrupted_task": task_id,
                }
            ),
        },
    )
    db.session.commit()


def update_source_extraction_result(
    source_id: str,
    derivatives: dict,
    auto_metadata: dict,
    status: str = "extracted",
    error_message: str | None = None,
    task_id: str | None = None,
) -> bool:
    """Update source with extraction results.

    Returns False, writing nothing, when the source was cancelled or, given
    *task_id*, when it now belongs to another task: a re-extract dispatched
    during this run owns it, and this run's result must not stand in for it.

    Clears the interruption marker and the partial page-image flag: both
    describe an earlier run.
    """
    result = db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".sources
            SET extraction_status = :status,
                derivatives = CAST(:derivatives AS jsonb),
                auto_metadata = (COALESCE(auto_metadata, '{{}}'::jsonb)
                                 - 'extraction_interruptions'
                                 - 'extraction_unattributed_interruptions'
                                 - 'extraction_interrupted_task'
                                 - 'page_images_incomplete')
                                || CAST(:auto_metadata AS jsonb),
                error_message = :error_message,
                updated_at = NOW()
            WHERE id = :id
              AND extraction_status != 'cancelled'
              AND (CAST(:task_id AS text) IS NULL OR celery_task_id = CAST(:task_id AS text))
        """),
        {
            "id": source_id,
            "task_id": task_id,
            "derivatives": json.dumps(derivatives),
            "auto_metadata": json.dumps(auto_metadata),
            "status": status,
            "error_message": error_message,
        },
    )
    db.session.commit()
    return result.rowcount > 0


def _delete_replaced_derivatives(
    storage: SupabaseStorage, bucket_id: str, source_id: str, old: dict, new: dict
) -> None:
    """Remove stored derivatives of this source that the new result no longer lists.

    Best effort: a leftover file wastes storage but breaks nothing, since
    readers only follow the paths in the source's current derivatives.
    """
    prefix = f"{bucket_id}/{source_id}/derivatives/"

    def paths(derivatives) -> set[str]:
        if not isinstance(derivatives, dict):
            return set()
        return {
            record["storage_path"]
            for records in derivatives.values()
            if isinstance(records, list)
            for record in records
            if isinstance(record, dict) and isinstance(record.get("storage_path"), str)
        }

    try:
        stale = sorted(p for p in paths(old) - paths(new) if p.startswith(prefix))
        if not stale:
            return
        storage.delete(bucket_id, [p[len(bucket_id) + 1 :] for p in stale])
        logger.info(f"Deleted {len(stale)} replaced derivatives of source {source_id}")
    except Exception as e:
        logger.warning(f"Could not delete replaced derivatives of source {source_id}: {e}")


def _sink_failure_behind(exc: BaseException, sink_errors: list) -> StorageError | None:
    """The sink error that *exc* was raised from, if any."""
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if any(current is err for err in sink_errors):
            return current
        current = current.__cause__ or current.__context__
    return None


async def run_extraction(
    storage: SupabaseStorage,
    source: dict,
    bucket_id: str,
    extraction_model: str | None = None,
    provider_keys: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    """Run the actual extraction asynchronously."""
    from agentic.ingest import ExtractorRegistry, RawContent

    source_id = source["id"]
    storage_path = source["storage_path"]
    file_type = source["file_type"]
    filename = source["name"]

    logger.info(f"Downloading source {source_id} from {storage_path}")
    # Spooled to disk and read back once: buffering the response in memory
    # holds the body twice while it is joined.
    with tempfile.TemporaryFile() as spool:
        storage.download_to_file(storage_path, spool)
        spool.seek(0)
        raw_bytes = spool.read()
    logger.info(f"Downloaded {len(raw_bytes)} bytes")

    raw_content = RawContent(
        content=raw_bytes,
        mime_type=file_type,
        source_uri=storage_path,
        filename=filename,
    )

    # Pass extraction model preference so PDFExtractor can read it
    raw_content.metadata["extraction_model"] = extraction_model or "auto"

    # Page images are stored as the extractor renders them instead of being
    # returned all at once: a long scanned PDF renders to gigabytes of PNG.
    # Keyed by page, because a method that fails part-way is followed by the
    # next one in the chain, which delivers the same pages again. An engine
    # that does not know the sink ignores it and returns images in the result,
    # which the loop below still stores.
    streamed_images: dict[int, dict] = {}
    sink_errors: list[PageImageStorageError] = []

    def page_image_sink(deriv) -> None:
        if sink_errors:
            # Storage already failed once: every later page would fail too, and
            # an engine that keeps rendering or OCRing pages is paying for work
            # that cannot be kept.
            err = PageImageStorageError(
                f"Not storing the page image of page {deriv.page} after an earlier "
                f"failure ({sink_errors[0]})"
            )
            sink_errors.append(err)
            raise err from sink_errors[0]
        ext = deriv.format or "png"
        deriv_path = get_derivative_storage_path(
            source_id, "image", f"image_page{deriv.page}.{ext}"
        )
        try:
            full_path = storage.upload(
                bucket_id=bucket_id,
                path=deriv_path,
                file_data=deriv.content,
                content_type=f"image/{ext}" if ext != "jpg" else "image/jpeg",
            )
        except StorageError as e:
            err = PageImageStorageError(f"Storing the page image of page {deriv.page} failed: {e}")
            sink_errors.append(err)
            raise err from e
        record = {"storage_path": full_path, "format": deriv.format, "page": deriv.page}
        if deriv.metadata:
            record["metadata"] = deriv.metadata
        streamed_images[deriv.page] = record

    raw_content.metadata["page_image_sink"] = page_image_sink
    _warn_if_engine_ignores_sink(source_id, len(raw_bytes))

    registry = ExtractorRegistry.default(provider_keys=provider_keys)

    try:
        extractor = registry.get_extractor(file_type)
        logger.info(f"Using extractor: {extractor.name} for {file_type}")
    except KeyError:
        logger.warning(f"No extractor for {file_type}, using fallback text extractor")
        from agentic.ingest import TextExtractor

        extractor = TextExtractor()

    logger.info(f"Starting extraction for source {source_id}")
    try:
        result = await extractor.extract(raw_content)
    except Exception as e:
        # The engine may have wrapped a page-image storage failure; surface it
        # as the StorageError the task retries on. Only when it is in this
        # failure's own chain: an earlier sink error the engine recovered from
        # did not cause this one.
        cause = _sink_failure_behind(e, sink_errors)
        if cause is not None and cause is not e:
            # A new exception: re-raising `cause` from `e` would link the
            # chain back to itself, since `e` already leads to `cause`.
            raise PageImageStorageError(str(cause)) from e
        raise
    del raw_content, raw_bytes
    logger.info(
        f"Extraction complete: {len(result.derivatives)} derivatives, "
        f"method: {result.extraction_method}"
    )

    derivatives = {}
    if streamed_images:
        derivatives["image"] = [streamed_images[page] for page in sorted(streamed_images)]

    for i, deriv in enumerate(result.derivatives):
        if deriv.type == "text":
            deriv_filename = "content.txt"
        elif deriv.type == "markdown":
            deriv_filename = "content.md"
        elif deriv.type == "html":
            deriv_filename = "content.html"
        elif deriv.type == "page_text":
            deriv_filename = f"page_{deriv.page}.txt"
        elif deriv.type == "image":
            ext = deriv.format or "png"
            page_suffix = f"_page{deriv.page}" if deriv.page else ""
            deriv_filename = f"image{page_suffix}_{i}.{ext}"
        else:
            deriv_filename = f"{deriv.type}_{i}.bin"

        deriv_path = get_derivative_storage_path(source_id, deriv.type, deriv_filename)

        if deriv.is_text():
            content_bytes = deriv.get_text().encode("utf-8")
            content_type = "text/plain"
            if deriv.type == "markdown":
                content_type = "text/markdown"
            elif deriv.type == "html":
                content_type = "text/html"
        else:
            content_bytes = (
                deriv.content if isinstance(deriv.content, bytes) else deriv.content.encode("utf-8")
            )
            if deriv.type == "image":
                fmt = deriv.format or "png"
                content_type = f"image/{fmt}" if fmt != "jpg" else "image/jpeg"
            else:
                content_type = (
                    f"application/{deriv.format}" if deriv.format else "application/octet-stream"
                )

        full_path = storage.upload(
            bucket_id=bucket_id,
            path=deriv_path,
            file_data=content_bytes,
            content_type=content_type,
        )

        logger.info(f"Stored derivative {deriv.type} at {full_path}")

        deriv_record = {
            "storage_path": full_path,
            "format": deriv.format,
        }
        if deriv.page:
            deriv_record["page"] = deriv.page
        if deriv.metadata:
            deriv_record["metadata"] = deriv.metadata

        if deriv.type not in derivatives:
            derivatives[deriv.type] = []
        derivatives[deriv.type].append(deriv_record)

    if result.auto_metadata.get("page_images_incomplete"):
        logger.warning(
            f"Source {source_id}: page images are incomplete; rendering failed after "
            f"{len(streamed_images)} pages had been stored"
        )

    auto_metadata = {
        **result.auto_metadata,
        "extraction_method": result.extraction_method,
        "extracted_at": result.extracted_at.isoformat(),
        "derivative_count": len(result.derivatives) + len(streamed_images),
        "stats": result.stats,
    }

    return derivatives, auto_metadata


def _requeue(task, countdown: int, source_size: int | None) -> None:
    """Send this delivery again later, as the same task id and retry count:
    waiting for a slot is neither a retry nor a failure, and occupies no
    worker thread while it waits. The file size goes with it, so a waiting
    task does not ask storage again on every wake-up."""
    kwargs = {**(task.request.kwargs or {}), "source_size": source_size}
    task.signature_from_request(task.request, kwargs=kwargs, countdown=countdown).apply_async()


def _count_interruption(
    source: dict, task_id: str, previous: PreviousAttempt | None
) -> dict | None:
    """Record that *task_id*'s last delivery died mid-extraction.

    Returns the task result when the source has now been failed. A task that
    was not the plausible cause of the kill (a larger file was in flight in the
    same worker) is counted separately against MAX_UNATTRIBUTED_INTERRUPTIONS:
    a whole worker stopping takes every task in it down, and the others did
    nothing wrong.
    """
    source_id = source["id"]
    auto_metadata = source.get("auto_metadata") or {}
    count = unattributed = 0
    if auto_metadata.get("extraction_interrupted_task") == task_id:
        count = int(auto_metadata.get("extraction_interruptions") or 0)
        unattributed = int(auto_metadata.get("extraction_unattributed_interruptions") or 0)
    if previous is not None and not previous.plausible_cause:
        unattributed += 1
        record_extraction_interruption(source_id, task_id, count, unattributed)
        if unattributed < MAX_UNATTRIBUTED_INTERRUPTIONS:
            logger.warning(
                f"Source {source_id}: extraction task {task_id} was interrupted while a "
                f"larger file ({previous.largest_in_flight} bytes, against {previous.size}) "
                f"was being extracted in the same worker; not counting it against this "
                f"file ({unattributed}/{MAX_UNATTRIBUTED_INTERRUPTIONS}), extracting again"
            )
            return None
        interruptions = unattributed
    else:
        count += 1
        record_extraction_interruption(source_id, task_id, count, unattributed)
        if count < MAX_EXTRACTION_INTERRUPTIONS:
            logger.warning(
                f"Source {source_id}: extraction task {task_id} was interrupted "
                f"({count}/{MAX_EXTRACTION_INTERRUPTIONS}); extracting again"
            )
            return None
        interruptions = count
    message = (
        f"Extraction was interrupted {interruptions} times before finishing: the worker "
        f"stopped while processing this file, for example because it ran out of "
        f"memory or was restarted. Not retrying automatically; re-extract to try again."
    )
    logger.error(f"Source {source_id}: {message}")
    update_source_status(source_id, "failed", message, task_id, error_code="permanent")
    return {"status": "error", "source_id": source_id, "error": message}


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
@billing.task_context
def extract_source(
    self,
    source_id: str,
    bucket_id: str,
    extraction_model: str | None = None,
    provider_keys: dict[str, str] | None = None,
    reextract_seed: str | None = None,
    billing_idempotency_key: str | None = None,
    billing_org_id: str | None = None,
    billing_project_id: str | None = None,
    source_size: int | None = None,
):
    """
    Extract content from a source file.

    Args:
        source_id: The source UUID
        bucket_id: The storage bucket ID
        extraction_model: Optional extraction method override
        provider_keys: Optional dict of provider→api_key for extraction services
        reextract_seed: Per-call idempotency-key tail set ONLY by the reextract
            route (a uuid4 generated before dispatch, stable across this call's
            Celery retries). Appended to the charge's idempotency_parts so each
            reextract of a source produces a distinct ledger row, while the
            upload path (seed=None) keeps the stable per-source key. Absence =
            an upload/import dispatch.
        billing_idempotency_key, billing_org_id, billing_project_id: VESTIGIAL —
            retained for deploy-compat only. Billing now flows through the
            billing port, which derives identity from the adapter (org/project)
            and recomputes the key from this task's own args (see the charge
            below). These params are unused, but an in-flight task enqueued
            before the port migration still carries them; keeping them avoids a
            TypeError on a cross-deploy retry.
        source_size: set only when this task requeued itself to wait for a
            large-extraction slot: the size it already read from storage.

    Returns:
        Dict with extraction results or error info
    """
    task_id = self.request.id
    logger.info(f"Starting extraction task {task_id} for source {source_id}")

    slot = None
    forget_attempt = False
    try:
        source = get_source(source_id)
        if not source:
            logger.error(f"Source {source_id} not found")
            return {"status": "error", "error": "Source not found"}

        if source["extraction_status"] == "cancelled":
            logger.info(f"Source {source_id} already cancelled, skipping")
            return {"status": "cancelled", "source_id": source_id}

        if source["extraction_status"] == "extracted":
            logger.info(f"Source {source_id} already extracted, skipping")
            return {"status": "skipped", "reason": "already_extracted"}

        storage = get_storage()
        size = source_size
        if size is None:
            size = storage.object_size(source["storage_path"])

        # Every exit from a run of this task leaves the source in some other
        # status, so finding it still `extracting` under this task's own id
        # means an earlier delivery died mid-run and this is the redelivery.
        redelivered = bool(
            task_id
            and source["extraction_status"] == "extracting"
            and source.get("celery_task_id") == task_id
        )
        previous = extraction_attempts.previous(task_id) if redelivered else None
        if (
            previous is not None
            and previous.slot_token
            and large_extraction_gate.is_live(previous.slot_token)
        ):
            # The broker redelivers after its visibility timeout even when the
            # first delivery is still running. That run holds a live slot, so
            # nothing died: wait for it rather than run twice or charge it.
            logger.warning(
                f"Source {source_id}: task {task_id} is still extracting in another "
                f"worker; checking again in {LARGE_EXTRACTION_REQUEUE_SECONDS}s"
            )
            _requeue(self, LARGE_EXTRACTION_REQUEUE_SECONDS, size)
            return {"status": "deferred", "source_id": source_id, "reason": "still_running"}

        if is_large_file(size):
            slot = large_extraction_gate.try_acquire(task_id)
            if slot is None:
                logger.info(
                    f"Source {source_id}: all large-file extraction slots are taken "
                    f"({size} bytes); trying again in {LARGE_EXTRACTION_REQUEUE_SECONDS}s"
                )
                _requeue(self, LARGE_EXTRACTION_REQUEUE_SECONDS, size)
                return {"status": "deferred", "source_id": source_id, "reason": "slots_busy"}

        if redelivered:
            failure = _count_interruption(source, task_id, previous)
            if failure is not None:
                forget_attempt = True
                return failure

        extraction_attempts.begin(task_id, size, slot.token if slot else None)
        forget_attempt = True
        update_source_status(source_id, "extracting", celery_task_id=task_id)

        derivatives, auto_metadata = asyncio.run(
            run_extraction(
                storage,
                source,
                bucket_id,
                extraction_model=extraction_model,
                provider_keys=provider_keys,
            )
        )

        # Check if cancelled while extraction was running
        current_status = db.session.execute(
            text(f'SELECT extraction_status FROM "{AI_SCHEMA}".sources WHERE id = :id'),
            {"id": source_id},
        ).scalar()
        if current_status == "cancelled":
            logger.info(f"Source {source_id} cancelled during extraction, discarding results")
            return {"status": "cancelled", "source_id": source_id}

        # Detect cloud→local fallback
        status = "extracted"
        warning_msg = None
        method = auto_metadata.get("extraction_method", "")
        requested = auto_metadata.get("requested_method")
        fallback_reason = auto_metadata.get("fallback_reason")
        if requested:
            status = "attention_required"
            reason_detail = f": {fallback_reason}" if fallback_reason else ""
            warning_msg = (
                f"Requested method '{requested}' failed{reason_detail}. "
                f"Fell back to '{method}'. Consider fixing the API key or choosing another method."
            )
            logger.warning(f"Source {source_id}: {warning_msg}")

        # Detect blank-page extraction from non-OCR methods
        # (skip if we already flagged a fallback — that's more specific)
        page_count = auto_metadata.get("page_count", 0)
        char_count = auto_metadata.get("char_count", 0)

        if (
            not warning_msg
            and method in ("fitz", "pdfplumber", "opendataloader")
            and page_count >= 3
        ):
            avg_chars = char_count / page_count if page_count else 0
            if avg_chars < 50:
                status = "attention_required"
                warning_msg = (
                    f"Non-OCR extraction yielded ~{int(avg_chars)} chars/page "
                    f"across {page_count} pages. The PDF may contain scanned "
                    f"images. Consider re-extracting with OCR."
                )
                logger.warning(f"Source {source_id}: {warning_msg}")

        if auto_metadata.get("page_images_incomplete"):
            incomplete_msg = (
                "Page images are incomplete: rendering failed part-way, so some pages "
                "have no stored image. Re-extract to try again."
            )
            status = "attention_required"
            warning_msg = f"{warning_msg} {incomplete_msg}" if warning_msg else incomplete_msg

        if not update_source_extraction_result(
            source_id, derivatives, auto_metadata, status, warning_msg, task_id=task_id
        ):
            logger.info(
                f"Source {source_id}: cancelled or re-dispatched to another task during "
                f"extraction task {task_id}; discarding its results"
            )
            return {"status": "superseded", "source_id": source_id}
        _delete_replaced_derivatives(
            storage, bucket_id, source_id, source.get("derivatives"), derivatives
        )

        # Bill OCR when OCR was performed. Non-OCR extraction (fitz, pdfplumber,
        # opendataloader, txt-native, ...) is CPU-only and not separately billed
        # at the extraction layer. The billing-configured check now lives in the
        # adapter (no-op when unconfigured), so the guard is just the method.
        if method in (_OCR_EXTRACTION_METHODS | _ADVANCED_OCR_EXTRACTION_METHODS):
            actual_pages = max(1, int(page_count or 1))
            # ACTUAL method's billed category — advanced_ocr only when the engine
            # that actually ran was LlamaParse; a fallback to Mistral bills the
            # cheaper ocr_pages.
            ocr_action = (
                "advanced_ocr" if method in _ADVANCED_OCR_EXTRACTION_METHODS else "ocr_pages"
            )
            # REQUESTED action — drives the idempotency KEY, which must stay
            # stable across retries even when the ACTUAL method varies (e.g.
            # LlamaParse falling back to Mistral). Computed from extraction_model
            # the same way the route does.
            # mirrors routes.sources._extraction_billing_action
            requested_action = "advanced_ocr" if extraction_model == "llamaparse" else "ocr_pages"
            # The reextract route supplies a per-call seed so its key is distinct
            # per call; the upload path (seed=None) keeps the stable per-source key.
            idempotency_parts = (source_id, reextract_seed) if reextract_seed else (source_id,)
            # billing.charge never raises; ChargeOutcome reports outcome. A
            # post-success 402 is bounded over-serve per spec line 54.
            billing.charge(
                action=ocr_action,
                idempotency_action=requested_action,
                idempotency_parts=idempotency_parts,
                ref_type="extraction",
                ref_id=source_id,
                quantity=actual_pages,
                metadata={"extraction_method": method},
            )

        logger.info(f"Extraction complete for source {source_id}")

        return {
            "status": "success",
            "source_id": source_id,
            "derivative_types": list(derivatives.keys()),
            "auto_metadata": auto_metadata,
        }

    except SoftTimeLimitExceeded:
        logger.warning(f"Extraction cancelled/timed out for source {source_id}")
        db.session.rollback()
        # Check if cancelled by user (cancel endpoint sets status before revoking)
        current = db.session.execute(
            text(f'SELECT extraction_status FROM "{AI_SCHEMA}".sources WHERE id = :id'),
            {"id": source_id},
        ).scalar()
        if current == "cancelled":
            return {"status": "cancelled", "source_id": source_id}
        update_source_status(source_id, "failed", "Extraction timed out", task_id)
        return {"status": "error", "source_id": source_id, "error": "Extraction timed out"}

    except StorageError as e:
        logger.error(f"Storage error during extraction: {e}")
        update_source_status(source_id, "failed", str(e), task_id)
        raise self.retry(exc=e) from e

    except Exception as e:
        logger.exception(f"Extraction failed for source {source_id}")
        update_source_status(source_id, "failed", str(e), task_id)

        return {
            "status": "error",
            "source_id": source_id,
            "error": str(e),
        }

    finally:
        if forget_attempt:
            extraction_attempts.end(task_id)
        if slot is not None:
            slot.release()
