"""
Query Enrichment Service.

Uses an LLM to rewrite a user query into two variants optimised for retrieval:
- enriched_query: semantically rich restatement for vector embedding
- keyword_query: OR-joined keywords/synonyms for BM25 full-text search

Session history is optionally threaded through so the LLM can resolve
conversational references (e.g. "what about pricing?" → "AWS cloud pricing").
"""

import logging
import time
from typing import Any

from agentic.knowledge.indexing._pageindex_lib.utils import extract_json
from agentic.knowledge.model_config import (
    QUERY_ENRICHMENT_DEFAULT_MODEL,
    QUERY_ENRICHMENT_MAX_TOKENS,
    QUERY_ENRICHMENT_TEMPERATURE,
)
from agentic.llm.routing import maybe_route_through_responses, reasoning_call_kwargs

from . import billing_port as billing
from .llm_call import with_llm_key
from .run_context import (
    get_run_id,
    new_request_id,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a search query optimizer. Given a user question, return a JSON object with exactly two fields:

"enriched_query": A self-contained sentence rephrasing the question for semantic search. \
Resolve pronouns using session history if provided.

"keywords": Search terms and synonyms joined with OR for keyword search. \
Wrap multi-word phrases in double quotes.

Return ONLY valid JSON. No markdown, no code fences, no explanation.
Example:
{"enriched_query": "What are the benefits of cloud computing?", "keywords": "benefits OR advantages OR \\"cloud computing\\" OR scalability"}
"""


def enrich_query(
    query: str,
    retrieval_method: str,
    session_history: list[dict[str, Any]] | None = None,
    model: str | None = None,
    request_id: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, str]:
    """
    Enrich a user query for improved retrieval.

    Args:
        request_id: Optional natural identifier for the originating retrieval
            request, used to derive the billing idempotency key. When omitted,
            a UUID4 is generated so retries that pass the same id remain
            idempotent but two unrelated calls do not collide.

    Returns:
        {"enriched_query": "...", "keyword_query": "..."}
    """
    # Short-circuit for single-char/empty queries when there's no session context
    if len(query.strip()) < 2 and not session_history:
        return {"enriched_query": query, "keyword_query": query}

    model = model or QUERY_ENRICHMENT_DEFAULT_MODEL

    # Prefer the caller-supplied request_id, then the agent-run id bound in
    # context, then a uuid4 fallback. Spec line 132: deterministic key on
    # agent_run replay.
    rid = request_id or get_run_id() or new_request_id()
    # Billing: pre-op balance check (free-tier hard cap). Goes through the
    # billing port — a no-op when no cloud billing adapter is registered
    # (unit tests, local dev, system calls, OSS build).
    billing.check_balance(estimated_cost=1)

    # Build user message with optional session context
    parts: list[str] = []

    if session_history:
        # Include last 3 message pairs for context, truncated
        recent = session_history[-6:]
        context_lines: list[str] = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))[:200]
            context_lines.append(f"{role}: {content}")
        if context_lines:
            parts.append("Session context:\n" + "\n".join(context_lines))

    parts.append(f"User query: {query}")
    user_message = "\n\n".join(parts)

    try:
        import litellm

        t0 = time.time()
        routed_model = maybe_route_through_responses(model, reasoning_effort)
        reasoning_kwargs = reasoning_call_kwargs(reasoning_effort, routed_model)
        with with_llm_key(model) as api_key:
            response = litellm.completion(
                model=routed_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=QUERY_ENRICHMENT_TEMPERATURE,
                max_tokens=QUERY_ENRICHMENT_MAX_TOKENS,
                drop_params=True,
                api_key=api_key,
                **reasoning_kwargs,
            )
        elapsed = time.time() - t0

        raw = (response.choices[0].message.content or "").strip()
        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            parsed = {}

        enriched_query = parsed.get("enriched_query") or ""
        keyword_query = parsed.get("keywords") or ""

        # Detect silent parse/extraction failure
        error = None
        if not enriched_query or not keyword_query:
            error = (
                f"LLM returned unusable response (missing enriched_query or keywords). "
                f"Raw response: {raw[:500]}"
            )
            logger.warning("Query enrichment parse failure: %s", error)
            enriched_query = enriched_query or query
            keyword_query = keyword_query or query

        logger.info(
            "Query enrichment completed in %.2fs | model=%s | enriched_query=%r | keyword_query=%r",
            elapsed,
            model,
            enriched_query[:100],
            keyword_query[:100],
        )

        result = {
            "enriched_query": enriched_query,
            "keyword_query": keyword_query,
        }
        if error:
            result["error"] = error

        # Post-charge on successful op (best-effort; never fails the op).
        billing.charge(
            action="query_enrichment",
            quantity=1,
            ref_type="retrieval",
            ref_id=rid,
            idempotency_parts=(rid,),
        )
        return result

    except Exception as e:
        logger.exception("Query enrichment failed, falling back to original query")
        return {
            "enriched_query": query,
            "keyword_query": query,
            "error": f"Query enrichment failed: {e}",
        }
