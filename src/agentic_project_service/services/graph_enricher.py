"""
Graph Enricher — Referenced Nodes Enrichment.

Handles Stage 2 of the GraphIndex indexing pipeline: identifying
cross-references between sections within a document.
"""

import asyncio
import json
import logging
from typing import Callable, Literal

import litellm
import litellm.exceptions as llme
from agentic.knowledge.model_config import (
    GRAPHINDEX_ENRICHMENT_BATCH_SIZE,
    GRAPHINDEX_ENRICHMENT_MAX_CONCURRENT,
    GRAPHINDEX_ENRICHMENT_MAX_INPUT_CHARS,
    GRAPHINDEX_ENRICHMENT_MAX_JSON_RETRIES,
    GRAPHINDEX_ENRICHMENT_MAX_TOKENS,
    GRAPHINDEX_ENRICHMENT_TOC_INCLUDE_SUMMARIES,
)
from agentic.llm.routing import (
    maybe_route_through_responses,
    reasoning_call_kwargs,
)

from .graph_index_store import GraphIndexStore
from .llm_call import with_llm_key

logger = logging.getLogger(__name__)

# After this many consecutive trippable errors from the same provider within one
# ToC's enrichment, short-circuit remaining nodes. Mirrors metadata_enricher.
_CIRCUIT_BREAKER_THRESHOLD = 5


def _provider_for_model(model: str) -> str:
    """Return the LiteLLM provider name for a model string."""
    try:
        _, provider, _, _ = litellm.get_llm_provider(model)
        return provider
    except Exception:
        return model.split("/")[0] if "/" in model else "unknown"


_REFERENCED_NODES_SCHEMA = {
    "type": "object",
    "properties": {
        "referenced_nodes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["referenced_nodes"],
    "additionalProperties": False,
}


def _build_toc_context(nodes: list[dict], include_summaries: bool = False) -> str:
    """Format all nodes as a ToC string for the LLM prompt."""
    lines = []
    for node in nodes:
        title = node.get("title", "")
        indent = "  " * node.get("depth", 0)
        line = f"{indent}[{node['node_id']}] {title}"
        if include_summaries:
            meta = node.get("meta") or {}
            summary = meta.get("summary", "")
            if summary:
                line += f" — {summary}"
        lines.append(line)
    return "\n".join(lines)


async def _enrich_single_node(
    node: dict,
    toc_context: str,
    valid_node_ids: set[str],
    children_map: dict[str, set[str]],
    id_to_title: dict[str, str],
    model: str,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> tuple[list[str], str | None]:
    """Call LLM for one node, return validated list of referenced node_ids and optional error."""
    current_node_id = node["node_id"]
    current_title = node.get("title", "")
    node_text = node.get("text", "")

    # Build excluded IDs: self, parent, and children
    parent_id = node.get("parent_node_id")
    child_ids = children_map.get(current_node_id, set())
    excluded_ids = {current_node_id}
    if parent_id:
        excluded_ids.add(parent_id)
    excluded_ids.update(child_ids)

    # Parent context line for the prompt
    if parent_id and parent_id in id_to_title:
        parent_line = f'\nIts parent section is [{parent_id}] "{id_to_title[parent_id]}".'
    else:
        parent_line = ""

    # Truncate very long node text
    if (
        GRAPHINDEX_ENRICHMENT_MAX_INPUT_CHARS > 0
        and len(node_text) > GRAPHINDEX_ENRICHMENT_MAX_INPUT_CHARS
    ):
        node_text = node_text[:GRAPHINDEX_ENRICHMENT_MAX_INPUT_CHARS] + "\n... [truncated]"

    system_prompt = (
        "You are analyzing a document section to identify explicit "
        "cross-references to other sections in the same document. "
        "Only flag references where the text directly mentions, cites, "
        "or depends on another specific section."
    )

    user_prompt = f"""Document table of contents (with section IDs):
{toc_context}

You are analyzing section [{current_node_id}] "{current_title}".{parent_line}

Based **only** on the text of **this** section, identify other sections that this text **explicitly references**. Valid references include:
- Direct mentions: "see Section X", " ""as described in...", "refer to...", "in Appendix A", etc.
- Named references to titles or topics defined in other sections; may include its own parent section.
- Explicit dependencies: "building on [section name]", "following from [section name]", "as shown in [section name]", etc.

Do NOT include:
- Parent or child sections (structural relationships are already captured)
- Sections that merely discuss similar topics without being explicitly referenced in the text

Note that this section may be preceded or followed by other sections and you should only consider the text of this section, not the surrounding context.
Text of section [{current_node_id}] and surrounding context:
---
{node_text}
---

Return ONLY a JSON object: {{"referenced_nodes": ["0003", "0005"]}} and do not return any other text.
If no references found, return: {{"referenced_nodes": []}} instead of an empty string."""

    # Use structured JSON schema output when the model supports it
    if litellm.supports_response_schema(model=model):
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "referenced_nodes",
                "strict": True,
                "schema": _REFERENCED_NODES_SCHEMA,
            },
        }
    else:
        response_format = {"type": "json_object"}

    raw = ""
    finish_reason = None
    usage_str = "usage=n/a"
    try:
        for attempt in range(1, GRAPHINDEX_ENRICHMENT_MAX_JSON_RETRIES + 1):
            routed_model = maybe_route_through_responses(model, reasoning_effort)
            extra_kwargs = reasoning_call_kwargs(reasoning_effort, routed_model)
            with with_llm_key(routed_model) as resolved_key:
                effective_key = api_key or resolved_key
                response = await litellm.acompletion(
                    model=routed_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=GRAPHINDEX_ENRICHMENT_MAX_TOKENS,
                    response_format=response_format,
                    drop_params=True,
                    num_retries=0,  # LiteLLM-level retry cap; outer JSON loop handles parse failures
                    max_retries=0,  # OpenAI SDK-level cap (separate from LiteLLM's num_retries)
                    timeout=60,  # bound in-flight memory if one call hangs
                    metadata={"stage": "enrichment"},
                    **({"api_key": effective_key} if effective_key else {}),
                    **extra_kwargs,
                )

            raw = (response.choices[0].message.content or "").strip()
            finish_reason = response.choices[0].finish_reason
            usage = getattr(response, "usage", None)
            usage_str = (
                f"prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}"
                if usage
                else "usage=n/a"
            )

            if finish_reason == "length":
                logger.warning(
                    "Node %s: response truncated (finish_reason=length, %s, model=%s)",
                    current_node_id,
                    usage_str,
                    model,
                )

            if not raw:
                msg = (
                    f"Node {current_node_id}: LLM returned empty content "
                    f"(finish_reason={finish_reason}, {usage_str}, model={model})"
                )
                logger.warning(msg)
                return [], msg

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                if attempt < GRAPHINDEX_ENRICHMENT_MAX_JSON_RETRIES:
                    delay = 2 ** (attempt - 1)  # 1s, 2s
                    logger.warning(
                        "Node %s: invalid JSON on attempt %d/%d, retrying in %ds (raw=%s)",
                        current_node_id,
                        attempt,
                        GRAPHINDEX_ENRICHMENT_MAX_JSON_RETRIES,
                        delay,
                        raw[:200],
                    )
                    await asyncio.sleep(delay)
                    continue
                msg = (
                    f"Node {current_node_id}: invalid JSON after "
                    f"{GRAPHINDEX_ENRICHMENT_MAX_JSON_RETRIES} attempts "
                    f"(finish_reason={finish_reason}, {usage_str}, len={len(raw)}, raw={raw[:200]})"
                )
                logger.warning(msg)
                return [], msg

            refs = parsed.get("referenced_nodes", [])

            # Validate: must be strings, must exist in ToC, exclude self/parent/children
            validated = [
                r
                for r in refs
                if isinstance(r, str) and r in valid_node_ids and r not in excluded_ids
            ]
            return validated, None

        # Should not reach here, but satisfy the type checker
        return [], f"Node {current_node_id}: exhausted retries without result"

    except (
        llme.RateLimitError,
        llme.AuthenticationError,
        llme.ServiceUnavailableError,
        llme.InternalServerError,
        llme.BadGatewayError,
        llme.APIConnectionError,
        llme.Timeout,
    ):
        # Let the circuit breaker in enrich_referenced_nodes see and count these.
        raise
    except Exception as e:
        msg = (
            f"Node {current_node_id} enrichment failed ({type(e).__name__}): {e}"
            f" (finish_reason={finish_reason}, {usage_str})"
        )
        logger.warning(msg)
        return [], msg


async def enrich_referenced_nodes(
    db_session,
    knowledge_base_id: str,
    toc_id: str,
    model: str,
    retry_failed: bool = False,
    indexed_source_id: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    on_batch_complete: Callable[[int, list[str]], Literal["continue", "abort"]] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Run referenced_nodes enrichment for all nodes in one ToC.

    For each node:
    1. Build prompt with full document ToC + node full text
    2. LLM identifies cross-references -> returns list of node_ids
    3. Validate returned node_ids against actual node_ids in the ToC
    4. Store as meta["referenced_nodes"]

    The caller is responsible for committing the transaction.

    Returns:
        {node_id: [referenced_node_id, ...], ...}
    """
    store = GraphIndexStore(
        db_session=db_session,
        knowledge_base_id=knowledge_base_id,
    )

    all_nodes = store.get_all_nodes_for_toc(toc_id)
    if not all_nodes:
        logger.warning(
            "No nodes found for toc %s in KB %s, skipping enrichment",
            toc_id,
            knowledge_base_id,
        )
        return {}, []

    # Full ToC context is always needed (even when retrying a subset)
    valid_node_ids = {n["node_id"] for n in all_nodes}
    toc_context = _build_toc_context(
        all_nodes, include_summaries=GRAPHINDEX_ENRICHMENT_TOC_INCLUDE_SUMMARIES
    )

    # Narrow processing scope to a single source when requested
    if indexed_source_id:
        nodes_to_process = [
            n for n in all_nodes if str(n.get("indexed_source_id")) == indexed_source_id
        ]
    else:
        nodes_to_process = all_nodes

    # When retrying, only process nodes that previously failed
    if retry_failed:
        nodes = [n for n in nodes_to_process if n.get("enrichment_error")]
        if not nodes:
            return {}, []
        logger.info(
            "Retrying %d failed nodes (of %d total) for toc %s",
            len(nodes),
            len(all_nodes),
            toc_id,
        )
    else:
        nodes = nodes_to_process

    logger.info(
        "Starting graph enrichment for toc %s: %d nodes (model=%s)",
        toc_id,
        len(nodes),
        model,
    )

    # Build parent→children map and id→title map for exclusion/context
    # Always use all_nodes so reference resolution works even when retrying a subset
    children_map: dict[str, set[str]] = {}
    for n in all_nodes:
        pid = n.get("parent_node_id")
        if pid:
            children_map.setdefault(pid, set()).add(n["node_id"])

    id_to_title = {n["node_id"]: n.get("title", "") for n in all_nodes}

    sem = asyncio.Semaphore(GRAPHINDEX_ENRICHMENT_MAX_CONCURRENT)
    results: dict[str, list[str]] = {}
    errors: list[str] = []
    node_errors: dict[str, str] = {}

    provider = _provider_for_model(model)
    tripped_providers: set[str] = set()
    consecutive_errors: dict[str, int] = {}
    tripped_provider_cause: dict[str, type] = {}

    async def _process_node_safe(node: dict) -> tuple[str, list[str], str | None]:
        try:
            return await _process_node(node)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001
            msg = f"Node {node['node_id']} enrichment task failed ({type(exc).__name__}): {exc}"
            logger.warning(msg)
            return node["node_id"], [], str(exc)[:500]

    async def _process_node(node: dict) -> tuple[str, list[str], str | None]:
        async with sem:
            if provider in tripped_providers:
                cause_cls = tripped_provider_cause.get(provider, llme.RateLimitError)
                msg = (
                    f"Circuit breaker tripped for {provider} after "
                    f"{_CIRCUIT_BREAKER_THRESHOLD} consecutive errors. "
                    f"Resolve the upstream issue and re-run."
                )
                try:
                    raise cause_cls(msg, model=model, llm_provider=provider)
                except TypeError:
                    raise cause_cls(msg)  # noqa: B904
            try:
                refs, error = await _enrich_single_node(
                    node=node,
                    toc_context=toc_context,
                    valid_node_ids=valid_node_ids,
                    children_map=children_map,
                    id_to_title=id_to_title,
                    model=model,
                    api_key=api_key,
                    reasoning_effort=reasoning_effort,
                )
            except (
                llme.RateLimitError,
                llme.AuthenticationError,
                llme.ServiceUnavailableError,
                llme.InternalServerError,
                llme.BadGatewayError,
                llme.APIConnectionError,
                llme.Timeout,
            ) as _trip_exc:
                consecutive_errors[provider] = consecutive_errors.get(provider, 0) + 1
                if (
                    consecutive_errors[provider] >= _CIRCUIT_BREAKER_THRESHOLD
                    and provider not in tripped_providers
                ):
                    tripped_providers.add(provider)
                    tripped_provider_cause[provider] = type(_trip_exc)
                    _quota_causes = (llme.RateLimitError, llme.AuthenticationError)
                    if isinstance(_trip_exc, _quota_causes):
                        logger.error(
                            "PLATFORM_LLM_QUOTA_EXHAUSTED provider=%s model=%s kb=%s "
                            "consecutive_errors=%d — investigate quota / spend cap / billing "
                            "on the platform LLM account.",
                            provider,
                            model,
                            knowledge_base_id,
                            consecutive_errors[provider],
                            extra={"alert": "platform_llm_quota_exhausted"},
                        )
                    else:
                        logger.error(
                            "PLATFORM_LLM_PROVIDER_DEGRADED provider=%s model=%s kb=%s "
                            "consecutive_errors=%d — investigate upstream provider status "
                            "(likely outage or network issue).",
                            provider,
                            model,
                            knowledge_base_id,
                            consecutive_errors[provider],
                            extra={"alert": "platform_llm_provider_degraded"},
                        )
                raise
            consecutive_errors[provider] = 0  # reset on success
            return node["node_id"], refs, error

    for batch_start in range(0, len(nodes), GRAPHINDEX_ENRICHMENT_BATCH_SIZE):
        batch = nodes[batch_start : batch_start + GRAPHINDEX_ENRICHMENT_BATCH_SIZE]
        batch_ok_count = 0

        # Stream results as each node completes — O(batch) peak retained memory
        # instead of O(total) with a single gather().
        tasks = [asyncio.create_task(_process_node_safe(n)) for n in batch]
        for coro in asyncio.as_completed(tasks):
            node_id, refs, error = await coro  # only CancelledError can raise; let it propagate
            results[node_id] = refs
            if error:
                errors.append(error)
                node_errors[node_id] = error
            else:
                batch_ok_count += 1

        # Persist this batch's results before invoking the callback so already
        # completed work is durable even if the callback aborts.
        for node in batch:
            nid = node["node_id"]
            meta = dict(node.get("meta") or {})
            meta["referenced_nodes"] = results.get(nid, [])
            meta.pop("_enrichment_error", None)
            store.update_node_meta(toc_id, nid, meta)
            if nid in node_errors:
                store.update_node_enrichment_error(toc_id, nid, node_errors[nid])
            elif nid not in results:
                store.update_node_enrichment_error(
                    toc_id, nid, "Enrichment task raised an unhandled exception"
                )
            else:
                store.update_node_enrichment_error(toc_id, nid, None)
        db_session.flush()

        if on_batch_complete is not None:
            batch_item_ids = [n["node_id"] for n in batch]
            if on_batch_complete(batch_ok_count, batch_item_ids) == "abort":
                logger.warning(
                    "Graph enrichment aborted by on_batch_complete after batch at offset %d (toc=%s)",
                    batch_start,
                    toc_id,
                )
                break

    ref_count = sum(len(v) for v in results.values())
    logger.info(
        "Graph enrichment complete for toc %s: %d nodes, %d total references",
        toc_id,
        len(nodes),
        ref_count,
    )

    return results, errors
