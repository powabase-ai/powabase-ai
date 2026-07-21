"""Metadata enrichment Celery task.

Runs LLM-based metadata extraction for all items in a knowledge base.
"""

import asyncio
import logging

from ..celery import celery_app
from sqlalchemy import text

from ..db import db, AI_SCHEMA
from ..services import billing_port as billing
from ..services.metadata_enricher import MetadataEnricher

from agentic.knowledge.model_config import (
    METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS,
    METADATA_ENRICHMENT_DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)


def _classify_enrichment_failure(exc: BaseException) -> str:
    """Convert an enrichment-run exception into a user-actionable error_message.

    Specific LLM provider errors (rate-limit, auth) get templated messages
    that set ops-side expectations (platform team gets paged via Task 19;
    the customer is told the job will not auto-retry). Other errors return
    a generic user-facing message — the full traceback is captured via
    logger.error(..., exc_info=True) at the caller site for ops visibility.

    Important 4 from PR #440 review: do NOT leak raw tracebacks to the
    user-facing error_message field (shown in Studio); tracebacks contain
    file paths, line numbers, and internal class names.
    """
    try:
        import litellm.exceptions as llme
    except ImportError:
        llme = None

    if llme is not None and isinstance(exc, llme.RateLimitError):
        return (
            "Enrichment was rate-limited by the LLM provider. The platform team has "
            "been notified. If this is your first time running enrichment on a large "
            "knowledge base, try splitting it into smaller chunks. The job will not "
            "automatically retry; trigger it again once the upstream issue is resolved."
        )
    if llme is not None and isinstance(exc, llme.AuthenticationError):
        return (
            "The LLM provider rejected the API key used for enrichment. The platform "
            "team has been notified. The job will not automatically retry."
        )
    # Fallback: do NOT leak the raw traceback to the user-facing error_message.
    # The full traceback is captured via logger.error(..., exc_info=True) at
    # the caller site for ops visibility.
    return (
        "Enrichment failed with an unexpected error. The platform team has "
        "been notified. Check the ops logs for the full stack trace."
    )


def _get_enrichment_config(kb_id: str) -> dict | None:
    """Fetch enrichment config for a KB, or None."""
    result = db.session.execute(
        text(
            f"SELECT id, fields, llm_model, max_tokens, use_multimodal, "
            f"metadata_table_name, status "
            f'FROM "{AI_SCHEMA}".enrichment_configs '
            f"WHERE knowledge_base_id = :kb_id"
        ),
        {"kb_id": kb_id},
    )
    row = result.fetchone()
    if not row:
        return None
    return {
        "id": str(row[0]),
        "fields": row[1] or [],
        "llm_model": row[2],
        "max_tokens": row[3] or METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS,
        "use_multimodal": bool(row[4]) if row[4] is not None else False,
        "metadata_table_name": row[5],
        "status": row[6],
    }


def _get_kb_strategy(kb_id: str) -> str:
    """Get the indexing strategy for a KB."""
    result = db.session.execute(
        text(f'SELECT indexing_config FROM "{AI_SCHEMA}".knowledge_bases WHERE id = :id'),
        {"id": kb_id},
    )
    row = result.fetchone()
    if not row:
        return "chunk_embed"
    config = row[0] or {}
    return config.get("strategy", "chunk_embed")


def _update_enrichment_status(
    config_id: str,
    status: str,
    enriched_count: int | None = None,
    total_count: int | None = None,
    error_message: str | None = None,
    celery_task_id: str | None = None,
) -> None:
    """Update enrichment config status."""
    updates = ["status = :status"]
    params: dict = {"id": config_id, "status": status}

    if enriched_count is not None:
        updates.append("enriched_count = :enriched_count")
        params["enriched_count"] = enriched_count
    if total_count is not None:
        updates.append("total_count = :total_count")
        params["total_count"] = total_count
    updates.append("error_message = :error_message")
    params["error_message"] = error_message  # None → SQL NULL via SQLAlchemy
    if celery_task_id is not None:
        updates.append("celery_task_id = :celery_task_id")
        params["celery_task_id"] = celery_task_id

    db.session.execute(
        text(f'UPDATE "{AI_SCHEMA}".enrichment_configs SET {", ".join(updates)} WHERE id = :id'),
        params,
    )
    db.session.commit()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)
@billing.task_context
def enrich_knowledge_base(
    self,
    knowledge_base_id: str,
    incremental: bool = False,
    retry_failed: bool = False,
    billing_idempotency_key: str | None = None,
    billing_org_id: str | None = None,
    billing_project_id: str | None = None,
):
    """
    Run metadata enrichment for all items in a knowledge base.

    Args:
        knowledge_base_id: UUID of the knowledge base
        incremental: If True, only enrich items without existing results
        retry_failed: If True, only re-enrich items that previously failed
        billing_idempotency_key, billing_org_id, billing_project_id: VESTIGIAL —
            retained for deploy-compat only. Per-batch metadata_enrichment
            charging now flows through the billing port
            (``billing.per_batch_callback``), which is enabled by default and
            ctx-gated by the adapter; these params are no longer read.
    """
    task_id = self.request.id
    logger.info(
        "Starting enrichment task %s for KB %s (incremental=%s)",
        task_id,
        knowledge_base_id,
        incremental,
    )

    config = _get_enrichment_config(knowledge_base_id)
    if not config:
        logger.warning("No enrichment config for KB %s", knowledge_base_id)
        return {"status": "skipped", "reason": "no enrichment config"}

    if not config["fields"]:
        logger.warning("Empty fields in enrichment config for KB %s", knowledge_base_id)
        return {"status": "skipped", "reason": "no fields defined"}

    try:
        # Atomic claim: only proceed if not already being processed
        claimed = db.session.execute(
            text(
                f'UPDATE "{AI_SCHEMA}".enrichment_configs '
                f"SET status = 'enriching', celery_task_id = :task_id, error_message = NULL "
                f"WHERE knowledge_base_id = :kb_id "
                f"AND (status != 'enriching' OR celery_task_id IS NULL) "
                f"RETURNING id"
            ),
            {"kb_id": knowledge_base_id, "task_id": task_id},
        ).fetchone()
        db.session.commit()

        if not claimed:
            logger.info("Enrichment already in progress for KB %s, skipping", knowledge_base_id)
            return {"status": "skipped", "reason": "already enriching"}

        strategy = _get_kb_strategy(knowledge_base_id)
        model = config["llm_model"] or METADATA_ENRICHMENT_DEFAULT_MODEL

        on_batch_complete = billing.per_batch_callback(
            config_id=config["id"],
            action="metadata_enrichment",
        )

        enricher = MetadataEnricher(db.session, knowledge_base_id)
        result = asyncio.run(
            enricher.run_enrichment(
                fields=config["fields"],
                model=model,
                strategy=strategy,
                table_name=config["metadata_table_name"],
                incremental=incremental,
                retry_failed=retry_failed,
                max_tokens=config.get("max_tokens", METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS),
                use_multimodal=config.get("use_multimodal", False),
                on_batch_complete=on_batch_complete,
            )
        )

        # Use DB-based counts for accurate final status regardless of run mode
        ok_count, failed_count = enricher.count_by_status(config["metadata_table_name"])
        total_items = enricher.count_total_items(strategy)

        if failed_count == 0:
            final_status = "completed"
        elif ok_count == 0:
            final_status = "failed"
        else:
            final_status = "completed_with_errors"
        error_msg = None
        if result["errors"]:
            error_msg = f"{len(result['errors'])} items failed: {result['errors'][0]}"

        _update_enrichment_status(
            config["id"],
            status=final_status,
            enriched_count=ok_count,
            total_count=total_items,
            error_message=error_msg,
        )

        logger.info(
            "Enrichment complete for KB %s: %d/%d (failed=%d)",
            knowledge_base_id,
            ok_count,
            total_items,
            failed_count,
        )

        # Per-batch charging is wired via on_batch_complete (billing.per_batch_callback).
        # The old end-of-job charge was removed in #437 fix Task 18 to give the runtime
        # mid-flow backpressure when a customer runs out of credits.

        return {
            "status": "success",
            "enriched_count": ok_count,
            "total_count": total_items,
        }

    except Exception as exc:
        logger.error("Enrichment failed for KB %s", knowledge_base_id, exc_info=True)
        if config:
            _update_enrichment_status(
                config["id"],
                status="failed",
                error_message=_classify_enrichment_failure(exc),
            )
        return {
            "status": "error",
            "error": str(exc),
        }
