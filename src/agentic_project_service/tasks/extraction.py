"""Source extraction Celery task.

Downloads source files from storage, extracts content using the agentic
ingest module, and stores derivatives back to storage.
"""

import asyncio
import json
import logging

from ..celery import celery_app
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import text

from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
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
                   derivatives, metadata, auto_metadata
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
    }


def update_source_status(
    source_id: str,
    status: str,
    error_message: str | None = None,
    celery_task_id: str | None = None,
) -> None:
    """Update source extraction status."""
    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".sources
            SET extraction_status = :status,
                error_message = :error_message,
                celery_task_id = :celery_task_id,
                updated_at = NOW()
            WHERE id = :id
        """),
        {
            "id": source_id,
            "status": status,
            "error_message": error_message,
            "celery_task_id": celery_task_id,
        },
    )
    db.session.commit()


def update_source_extraction_result(
    source_id: str,
    derivatives: dict,
    auto_metadata: dict,
    status: str = "extracted",
    error_message: str | None = None,
) -> None:
    """Update source with extraction results."""
    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".sources
            SET extraction_status = :status,
                derivatives = CAST(:derivatives AS jsonb),
                auto_metadata = COALESCE(auto_metadata, '{{}}'::jsonb) || CAST(:auto_metadata AS jsonb),
                error_message = :error_message,
                updated_at = NOW()
            WHERE id = :id
              AND extraction_status != 'cancelled'
        """),
        {
            "id": source_id,
            "derivatives": json.dumps(derivatives),
            "auto_metadata": json.dumps(auto_metadata),
            "status": status,
            "error_message": error_message,
        },
    )
    db.session.commit()


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
    raw_bytes = storage.download_from_path(storage_path)
    logger.info(f"Downloaded {len(raw_bytes)} bytes")

    raw_content = RawContent(
        content=raw_bytes,
        mime_type=file_type,
        source_uri=storage_path,
        filename=filename,
    )

    # Pass extraction model preference so PDFExtractor can read it
    raw_content.metadata["extraction_model"] = extraction_model or "auto"

    registry = ExtractorRegistry.default(provider_keys=provider_keys)

    try:
        extractor = registry.get_extractor(file_type)
        logger.info(f"Using extractor: {extractor.name} for {file_type}")
    except KeyError:
        logger.warning(f"No extractor for {file_type}, using fallback text extractor")
        from agentic.ingest import TextExtractor

        extractor = TextExtractor()

    logger.info(f"Starting extraction for source {source_id}")
    result = await extractor.extract(raw_content)
    logger.info(
        f"Extraction complete: {len(result.derivatives)} derivatives, "
        f"method: {result.extraction_method}"
    )

    derivatives = {}

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

    auto_metadata = {
        **result.auto_metadata,
        "extraction_method": result.extraction_method,
        "extracted_at": result.extracted_at.isoformat(),
        "derivative_count": len(result.derivatives),
        "stats": result.stats,
    }

    return derivatives, auto_metadata


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

    Returns:
        Dict with extraction results or error info
    """
    task_id = self.request.id
    logger.info(f"Starting extraction task {task_id} for source {source_id}")

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

        update_source_status(source_id, "extracting", celery_task_id=task_id)

        storage = get_storage()

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

        update_source_extraction_result(source_id, derivatives, auto_metadata, status, warning_msg)

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
