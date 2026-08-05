"""URL source extraction Celery task.

Scrapes a single URL via Firecrawl, stores HTML/markdown/text derivatives,
downloads inline images, and updates the source record.
"""

import logging
import os
import random
import re
from datetime import datetime, timezone

import httpx
from ..celery import celery_app
from celery.exceptions import Retry, SoftTimeLimitExceeded
from sqlalchemy import text

from ..db import AI_SCHEMA, db
from ..services import billing_port as billing
from ..services.external_api import PERMANENT_STATUSES, parse_retry_after
from ..services.rate_limit import external_limiter
from ..services.settings_registry import get_setting
from ..services.storage import (
    StorageError,
    get_derivative_storage_path,
    get_source_storage_path,
    get_storage,
)
from .extraction import (
    get_source,
    update_source_extraction_result,
    update_source_status,
)

logger = logging.getLogger(__name__)

# Per-cause retry budgets (tracked in the task's `retry_counts` kwarg, not
# Celery's single request.retries counter — see the task docstring).
# rate_limit covers pacing deferrals and 429 re-queues; worst case is
# ~15 min of pacing, or ~80 min of 429s at the 300s Retry-After cap.
RATE_LIMIT_MAX_RETRIES = 15
# ~4 hours at countdown=600 for the operator to fix a missing platform key.
MISSING_KEY_MAX_RETRIES = 24

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^)]+)\)")


def _guess_image_ext(content_type: str | None, url: str) -> str:
    """Determine image file extension from Content-Type or URL."""
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        mime_map = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/gif": "gif",
            "image/webp": "webp",
            "image/tiff": "tiff",
            "image/svg+xml": "svg",
        }
        ext = mime_map.get(ct)
        if ext:
            return ext

    # Fallback: parse URL path
    path = url.split("?")[0].split("#")[0]
    if "." in path.split("/")[-1]:
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in ("png", "jpg", "jpeg", "gif", "webp", "tiff", "svg"):
            return "jpg" if ext == "jpeg" else ext

    return "png"  # safe default


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
@billing.no_billing_context
def extract_url_source(
    self,
    source_id: str,
    bucket_id: str,
    url: str,
    provider_keys: dict[str, str] | None = None,
    reextract_seed: str | None = None,
    billing_idempotency_key: str | None = None,
    billing_org_id: str | None = None,
    billing_project_id: str | None = None,
    retry_counts: dict[str, int] | None = None,
):
    """Scrape a single URL and store derivatives.

    Args:
        source_id: The source UUID
        bucket_id: The storage bucket ID
        url: The URL to scrape
        provider_keys: Optional provider API keys (unused, for signature compat)
        reextract_seed: Per-call idempotency-key tail set ONLY by the reextract
            route (a uuid4 generated before dispatch, stable across this call's
            Celery retries). Appended to the charge's idempotency_parts so each
            reextract produces a distinct ledger row; the import path (seed=None)
            keeps the stable per-source key.
        billing_idempotency_key, billing_org_id, billing_project_id: VESTIGIAL —
            retained for deploy-compat only. Billing now flows through the
            billing port (identity from the adapter, key recomputed from this
            task's own args). Unused, but an in-flight task enqueued before the
            port migration still carries them; keeping them avoids a TypeError on
            a cross-deploy retry.
        retry_counts: Per-cause retry counters ({"missing_key" | "rate_limit" |
            "transient": n}), threaded through each re-queue via retry(kwargs=...)
            so the causes have independent budgets — Celery's single
            ``request.retries`` counter cannot distinguish them (a run of pacing
            deferrals would otherwise eat the transient budget and vice versa).
            Every retry site passes ``max_retries=None`` and does its own
            pre-retry budget check, because ``Task.retry(exc=...)`` re-raises
            ``exc`` itself on exhaustion, never MaxRetriesExceededError.
    """
    task_id = self.request.id
    counts = dict(retry_counts or {})
    logger.info("Starting URL extraction task %s for source %s: %s", task_id, source_id, url)

    def _requeue_kwargs() -> dict:
        return {**(self.request.kwargs or {}), "retry_counts": counts}

    try:
        source = get_source(source_id)
        if not source:
            logger.error("Source %s not found", source_id)
            return {"status": "error", "error": "Source not found"}

        if source["extraction_status"] == "cancelled":
            logger.info("Source %s already cancelled, skipping", source_id)
            return {"status": "cancelled", "source_id": source_id}

        if source["extraction_status"] == "extracted":
            logger.info("Source %s already extracted, skipping", source_id)
            return {"status": "skipped", "reason": "already_extracted"}

        update_source_status(source_id, "extracting", celery_task_id=task_id)

        firecrawl_key = os.environ.get("FIRECRAWL_API_KEY", "")
        if not firecrawl_key:
            # Mark the source `extracting` (it stays visible as a retry-able
            # state in Studio) and ask Celery to retry. A plain `return`
            # would leave the source permanently `failed` for every URL
            # submitted during a misconfig window. countdown=600 with a
            # budget of 24 gives the operator ~4 hours to fix the missing
            # key and roll the affected pods.
            used = counts.get("missing_key", 0)
            if used >= MISSING_KEY_MAX_RETRIES:
                logger.error(
                    "FIRECRAWL_API_KEY still missing after %d retries — failing source %s",
                    used,
                    source_id,
                )
                update_source_status(
                    source_id,
                    "failed",
                    "FIRECRAWL_API_KEY missing from pod env",
                    task_id,
                    error_code="internal",
                )
                return {
                    "status": "error",
                    "source_id": source_id,
                    "error": "FIRECRAWL_API_KEY missing from pod env",
                }
            logger.error("FIRECRAWL_API_KEY missing from pod env — platform misconfiguration")
            counts["missing_key"] = used + 1
            raise self.retry(
                exc=RuntimeError("FIRECRAWL_API_KEY missing from pod env"),
                countdown=600,
                max_retries=None,
                kwargs=_requeue_kwargs(),
            )

        # 1. Scrape via Firecrawl
        firecrawl_base = get_setting("FIRECRAWL_API_BASE").rstrip("/")

        rate_limit = get_setting("FIRECRAWL_RATE_LIMIT_PER_MINUTE")
        wait = external_limiter.try_acquire("firecrawl", rate_limit)
        if wait is not None:
            used = counts.get("rate_limit", 0)
            if used >= RATE_LIMIT_MAX_RETRIES:
                logger.error(
                    "Firecrawl pacing budget exhausted for source %s after %d attempts",
                    source_id,
                    used,
                )
                update_source_status(
                    source_id,
                    "failed",
                    "Scraping provider rate limit exceeded after retries",
                    task_id,
                    error_code="rate_limited",
                )
                return {
                    "status": "error",
                    "source_id": source_id,
                    "error": "Firecrawl rate limit pacing budget exhausted",
                }
            logger.info("Firecrawl pacing: deferring source %s by %.1fs", source_id, wait)
            counts["rate_limit"] = used + 1
            raise self.retry(
                exc=RuntimeError("Firecrawl rate limit pacing"),
                countdown=wait + random.uniform(0, 5),
                max_retries=None,
                kwargs=_requeue_kwargs(),
            )

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{firecrawl_base}/scrape",
                headers={"Authorization": f"Bearer {firecrawl_key}"},
                json={"url": url, "formats": ["markdown", "html"]},
            )
            resp.raise_for_status()
            scrape_data = resp.json().get("data", {})

        markdown_content = scrape_data.get("markdown", "")
        html_content = scrape_data.get("html", "")
        if not markdown_content and not html_content:
            update_source_status(
                source_id,
                "failed",
                f"No content returned for URL: {url}",
                task_id,
                error_code="no_content",
            )
            return {
                "status": "error",
                "source_id": source_id,
                "error": "No content returned from scrape",
            }
        page_title = scrape_data.get("metadata", {}).get("title", "")

        storage = get_storage()
        storage.ensure_bucket(bucket_id)
        derivatives: dict[str, list] = {}

        # 2. Store HTML as original
        if html_content:
            original_path = get_source_storage_path(source_id, "page.html")
            full_path = storage.upload(
                bucket_id=bucket_id,
                path=original_path,
                file_data=html_content.encode("utf-8"),
                content_type="text/html",
            )
            # Update the source's storage_path to match actual upload
            db.session.execute(
                text(f"""
                    UPDATE "{AI_SCHEMA}".sources
                    SET storage_path = :path, updated_at = NOW()
                    WHERE id = :id
                """),
                {"path": full_path, "id": source_id},
            )
            db.session.commit()

        # 3. Store markdown derivative
        if markdown_content:
            md_path = get_derivative_storage_path(source_id, "markdown", "content.md")
            md_full = storage.upload(
                bucket_id=bucket_id,
                path=md_path,
                file_data=markdown_content.encode("utf-8"),
                content_type="text/markdown",
            )
            derivatives.setdefault("markdown", []).append({"storage_path": md_full, "format": None})

        # 4. Store plain text derivative
        text_content = (
            _strip_markdown_to_text(markdown_content) if markdown_content else html_content or ""
        )
        if text_content:
            txt_path = get_derivative_storage_path(source_id, "text", "content.txt")
            txt_full = storage.upload(
                bucket_id=bucket_id,
                path=txt_path,
                file_data=text_content.encode("utf-8"),
                content_type="text/plain",
            )
            derivatives.setdefault("text", []).append({"storage_path": txt_full, "format": None})

        # 5. Extract and download images
        max_images = get_setting("URL_IMPORT_MAX_IMAGES_PER_PAGE")
        max_image_size_mb = get_setting("URL_IMPORT_IMAGE_MAX_SIZE_MB")
        max_image_bytes = max_image_size_mb * 1024 * 1024

        image_urls: list[str] = []
        seen_urls: set[str] = set()
        for _alt, img_url in _IMAGE_RE.findall(markdown_content):
            if img_url not in seen_urls:
                seen_urls.add(img_url)
                image_urls.append(img_url)
            if len(image_urls) >= max_images:
                break

        with httpx.Client(timeout=10) as img_client:
            for i, img_url in enumerate(image_urls):
                try:
                    img_resp = img_client.get(img_url, follow_redirects=True)
                    img_resp.raise_for_status()

                    if len(img_resp.content) > max_image_bytes:
                        logger.warning(
                            "Image too large (%d bytes), skipping: %s",
                            len(img_resp.content),
                            img_url,
                        )
                        continue

                    ct = img_resp.headers.get("content-type")
                    ext = _guess_image_ext(ct, img_url)
                    img_filename = f"image_page1_{i}.{ext}"
                    img_path = get_derivative_storage_path(source_id, "image", img_filename)

                    content_type = ct.split(";")[0].strip() if ct else f"image/{ext}"
                    img_full = storage.upload(
                        bucket_id=bucket_id,
                        path=img_path,
                        file_data=img_resp.content,
                        content_type=content_type,
                    )

                    derivatives.setdefault("image", []).append(
                        {
                            "storage_path": img_full,
                            "format": ext,
                            "page": 1,
                            "metadata": {"original_url": img_url},
                        }
                    )

                except Exception:
                    logger.warning("Failed to download image %s", img_url, exc_info=True)

        # 6. Update source name to page title if available
        if page_title:
            db.session.execute(
                text(f"""
                    UPDATE "{AI_SCHEMA}".sources
                    SET name = :name, updated_at = NOW()
                    WHERE id = :id
                """),
                {"name": page_title, "id": source_id},
            )
            db.session.commit()

        # 7. Build auto_metadata and finalize
        auto_metadata = {
            "extraction_method": "firecrawl_url",
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "derivative_count": sum(len(v) for v in derivatives.values()),
            "origin_url": url,
            "stats": {
                "markdown_chars": len(markdown_content),
                "html_chars": len(html_content),
                "image_count": len(derivatives.get("image", [])),
            },
        }
        if page_title:
            auto_metadata["page_title"] = page_title

        # Check if cancelled while extraction was running
        current_status = db.session.execute(
            text(f'SELECT extraction_status FROM "{AI_SCHEMA}".sources WHERE id = :id'),
            {"id": source_id},
        ).scalar()
        if current_status == "cancelled":
            logger.info("Source %s cancelled during extraction, discarding results", source_id)
            return {"status": "cancelled", "source_id": source_id}

        update_source_extraction_result(source_id, derivatives, auto_metadata)

        # Bill web_scrape (one page per URL). The billing-configured check now
        # lives in the adapter (no-op when unconfigured), so this is
        # unconditional. web_scrape is fixed on both sides — no idempotency_action
        # override needed. The reextract route supplies a per-call seed so its
        # key is distinct per call; the import path (seed=None) keeps the stable
        # per-source key. billing.charge never raises; a post-success 402 is
        # bounded over-serve per spec line 54.
        billing.charge(
            action="web_scrape",
            idempotency_parts=(source_id, reextract_seed) if reextract_seed else (source_id,),
            ref_type="extraction",
            ref_id=source_id,
            quantity=1,
            metadata={"url": url},
        )

        logger.info("URL extraction complete for source %s", source_id)
        return {
            "status": "success",
            "source_id": source_id,
            "derivative_types": list(derivatives.keys()),
            "auto_metadata": auto_metadata,
        }

    except SoftTimeLimitExceeded:
        logger.warning("URL extraction cancelled/timed out for source %s", source_id)
        db.session.rollback()
        current = db.session.execute(
            text(f'SELECT extraction_status FROM "{AI_SCHEMA}".sources WHERE id = :id'),
            {"id": source_id},
        ).scalar()
        if current == "cancelled":
            return {"status": "cancelled", "source_id": source_id}
        update_source_status(
            source_id, "failed", "Extraction timed out", task_id, error_code="timeout"
        )
        return {"status": "error", "source_id": source_id, "error": "Extraction timed out"}

    except StorageError as e:
        logger.error("Storage error during URL extraction: %s", e)
        update_source_status(source_id, "failed", str(e), task_id, error_code="transient")
        used = counts.get("transient", 0)
        if used >= self.max_retries:
            logger.error(
                "Storage retry budget exhausted for source %s after %d attempts", source_id, used
            )
            return {"status": "error", "source_id": source_id, "error": str(e)}
        counts["transient"] = used + 1
        raise self.retry(exc=e, max_retries=None, kwargs=_requeue_kwargs()) from e

    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        # Celery's Task.retry(exc=...) re-raises `exc` itself once the retry
        # budget is exhausted — it does NOT raise MaxRetriesExceededError
        # (that only happens when retry() is called with no `exc`). So each
        # branch below checks its per-cause counter against the budget BEFORE
        # calling retry(), and every retry passes max_retries=None so Celery's
        # own (shared-counter) exhaustion path can never preempt these checks.
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        if status_code == 429:
            delay = parse_retry_after(e.response.headers.get("Retry-After")) + random.uniform(0, 5)
            used = counts.get("rate_limit", 0)
            if used >= RATE_LIMIT_MAX_RETRIES:
                logger.error(
                    "Firecrawl 429 for source %s — retry budget exhausted after %d attempts",
                    source_id,
                    used,
                )
                update_source_status(
                    source_id,
                    "failed",
                    f"Scraping provider rate limit exceeded after retries: {e}",
                    task_id,
                    error_code="rate_limited",
                )
                return {"status": "error", "source_id": source_id, "error": str(e)}
            logger.warning("Firecrawl 429 for source %s — re-queueing in %.1fs", source_id, delay)
            counts["rate_limit"] = used + 1
            raise self.retry(exc=e, countdown=delay, max_retries=None, kwargs=_requeue_kwargs())
        if status_code in PERMANENT_STATUSES:
            logger.error(
                "Firecrawl permanent error %s for source %s: %s", status_code, source_id, e
            )
            update_source_status(source_id, "failed", str(e), task_id, error_code="permanent")
            return {"status": "error", "source_id": source_id, "error": str(e)}
        # 5xx / transport errors: bounded retry on the task's default budget.
        used = counts.get("transient", 0)
        if used >= self.max_retries:
            logger.error(
                "Firecrawl transient error for source %s — retry budget exhausted: %s",
                source_id,
                e,
            )
            update_source_status(source_id, "failed", str(e), task_id, error_code="transient")
            return {"status": "error", "source_id": source_id, "error": str(e)}
        counts["transient"] = used + 1
        raise self.retry(exc=e, max_retries=None, kwargs=_requeue_kwargs())

    except Retry:
        # Celery's self.retry() raises Retry; re-raise so the Celery worker
        # re-enqueues the task instead of the generic `except Exception`
        # below swallowing it and marking the source permanently failed.
        raise

    except Exception as e:
        logger.exception("URL extraction failed for source %s", source_id)
        update_source_status(source_id, "failed", str(e), task_id, error_code="internal")
        return {"status": "error", "source_id": source_id, "error": str(e)}


def _strip_markdown_to_text(md: str) -> str:
    """Rough markdown-to-plain-text conversion."""
    text_out = md
    # Remove images
    text_out = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text_out)
    # Remove links but keep text
    text_out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text_out)
    # Remove headers markup
    text_out = re.sub(r"^#{1,6}\s+", "", text_out, flags=re.MULTILINE)
    # Remove bold/italic
    text_out = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text_out)
    text_out = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text_out)
    # Remove code blocks
    text_out = re.sub(r"```[\s\S]*?```", "", text_out)
    text_out = re.sub(r"`([^`]+)`", r"\1", text_out)
    # Remove horizontal rules
    text_out = re.sub(r"^[-*_]{3,}\s*$", "", text_out, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text_out = re.sub(r"\n{3,}", "\n\n", text_out)
    return text_out.strip()
