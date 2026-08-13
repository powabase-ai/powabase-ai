"""Internal docs-RAG search endpoint (Project Copilot).

This endpoint lives on the singleton "system docs" project — the one place per
deployment that holds the hidden, full_document-indexed Powabase documentation KB.
Every project's Project Copilot reaches it through its ``search_docs`` tool over
internal HTTP.

Auth is a shared bearer token (``DOCS_SEARCH_TOKEN``), sent as the
``X-Docs-Search-Token`` header and compared in constant time. We use a token
rather than the per-project JWT auth (``routes/internal.py``) because the caller
is a *different* project's service, which does not hold this docs project's
service_role_key. Fail-closed: when the token env is unset/empty, every call is
rejected (fail-closed on an unset shared secret, as with our other internal
endpoints).

``DOCS_KB_ID`` names the docs KB to search. When unset, this service isn't acting
as the docs project and the endpoint returns 503.
"""

import hmac
import logging
import os
import time

import redis
from flask import Blueprint, jsonify, request
from sqlalchemy import text

from ..db import AI_SCHEMA, db
from ..services.knowledge_search import EmptyKnowledgeBaseError, search_knowledge_base

logger = logging.getLogger(__name__)

internal_docs_bp = Blueprint("internal_docs", __name__, url_prefix="/api/internal/docs")

# Default number of documents to retrieve when the caller doesn't specify.
_DEFAULT_TOP_K = 8
# Guard against an unbounded query driving an unbounded embedding cost.
_MAX_QUERY_LEN = 4000

# Rate-limit against token-burn: this endpoint is reachable directly at :5000
# by other projects' copilots in K8s (bypassing Kong entirely — see the module
# docstring), so the only limit that covers both K8s and docker is here in the
# handler. Fixed window, per CALLING project (via the X-Caller-Project header
# — see _rate_limited), per minute.
_RATE_LIMIT_PER_MIN = 120

# Coarse GLOBAL backstop, same fixed window, summed across ALL callers
# regardless of the X-Caller-Project header. The per-caller limit above is
# evadable by a compromised token-holder that simply rotates the header value
# on every request (each rotation gets a fresh per-caller bucket) — this tier
# bounds total fleet spend on the endpoint even under that attack. Higher than
# the per-caller cap since it's meant to catch fleet-wide abuse, not normal
# multi-project traffic.
_RATE_LIMIT_GLOBAL_PER_MIN = 600

# Atomic INCR+EXPIRE: a bare INCR followed by a separate EXPIRE leaks the key
# with no TTL forever if the process dies between the two calls. One EVAL
# makes "increment and arm the TTL" a single atomic step (mirrors the
# compare-and-delete Lua in tasks/docs_refresh.py).
_RATE_LIMIT_LUA = (
    "local count = redis.call('INCR', KEYS[1]) "
    "if tonumber(count) == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end "
    "return count"
)


# Cached client — reused across calls instead of building a fresh
# ConnectionPool on every request.
_redis_client: redis.Redis | None = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"))
    return _redis_client


def _rate_limited() -> bool:
    """Fixed-window Redis rate limit: per-CALLER tier + a coarse GLOBAL backstop.

    This handler runs on the singleton docs project's own pod, so its own
    PROJECT_REF is a constant, not the caller's identity — every other
    project's Project Copilot shares that same pod. The caller instead sends
    its own PROJECT_REF as the ``X-Caller-Project`` header (see
    ``services/project_copilot.py``'s ``search_docs``), and THAT is what keys
    the per-caller limit, giving true per-caller isolation (one noisy/compromised
    project can't 429 docs-search for everyone else). A request with no
    header (e.g. a bare curl) falls back to a shared ``"_unknown"`` bucket —
    still bounded, just not isolated from other header-less callers.

    The per-caller tier alone is evadable: a compromised token-holder can
    rotate ``X-Caller-Project`` on every request and get a fresh bucket each
    time, driving unbounded embedding spend. A second, coarse GLOBAL counter
    (keyed only by epoch-minute, summed across every caller value) closes that
    gap and bounds total fleet spend on the endpoint regardless of the header.
    A request is rejected if EITHER tier is over its cap.

    Fail OPEN: any Redis error (in either tier) allows the request through
    (logged, not denied) — this is an internal tool endpoint already gated by
    token auth plus the query/top_k caps, so availability wins over strictness
    here. Note that means both cost ceilings below are VOID for the duration
    of a Redis outage; the token gate and the query-length/top_k caps are the
    only limits still in force until Redis recovers.
    """
    try:
        r = _get_redis()
        caller = request.headers.get("X-Caller-Project", "") or "_unknown"
        epoch_minute = int(time.time() // 60)
        key = f"docs_search_rl:{caller}:{epoch_minute}"
        count = r.eval(_RATE_LIMIT_LUA, 1, key, 60)
        global_key = f"docs_search_rl:_global:{epoch_minute}"
        global_count = r.eval(_RATE_LIMIT_LUA, 1, global_key, 60)
        return count > _RATE_LIMIT_PER_MIN or global_count > _RATE_LIMIT_GLOBAL_PER_MIN
    except Exception:
        logger.warning("docs_search rate limit check failed; failing open", exc_info=True)
        return False


def _token_ok() -> bool:
    """Constant-time compare of the request token against ``DOCS_SEARCH_TOKEN``.

    Fail-closed: an unset/empty env returns False so the endpoint rejects every
    call until the deployment provisions the secret.
    """
    expected = os.environ.get("DOCS_SEARCH_TOKEN", "")
    provided = request.headers.get("X-Docs-Search-Token", "")
    if not expected or not provided:
        return False
    return hmac.compare_digest(expected, provided)


def _format_item(item, meta_by_source: dict[str, dict]) -> dict:
    """Shape a RetrievedItem into the copilot-facing result.

    The full_document retrieval path doesn't surface the source's title/URL on the
    item itself, so we look them up from the ingested source's ``auto_metadata``
    (keyed by ``source_id``) — that's where docs_refresh records them — and fall
    back to the item's own ``meta`` for other strategies.
    """
    item_meta = getattr(item, "meta", None) or {}
    src_meta = meta_by_source.get(str(getattr(item, "source_id", "")) or "", {})
    return {
        "text": item.text,
        "score": item.score,
        "source_id": getattr(item, "source_id", None),
        "title": src_meta.get("title") or item_meta.get("title") or item_meta.get("name"),
        "url": src_meta.get("url") or item_meta.get("url") or item_meta.get("source_url"),
    }


@internal_docs_bp.route("/search", methods=["POST"])
def docs_search():
    """Search the hidden docs KB and return ranked, citable chunks."""
    if not _token_ok():
        return jsonify({"error": "unauthorized"}), 401

    kb_id = os.environ.get("DOCS_KB_ID", "")
    if not kb_id:
        return jsonify({"error": "docs search is not configured on this service"}), 503

    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    if len(query) > _MAX_QUERY_LEN:
        return jsonify({"error": f"query exceeds {_MAX_QUERY_LEN} chars"}), 400

    try:
        top_k = int(body.get("top_k") or _DEFAULT_TOP_K)
    except (TypeError, ValueError):
        top_k = _DEFAULT_TOP_K
    top_k = max(1, min(top_k, 20))

    if _rate_limited():
        return jsonify({"error": "rate limited"}), 429

    try:
        items = search_knowledge_base(db.session, kb_id, query, top_k=top_k)
    except EmptyKnowledgeBaseError:
        # Genuinely empty / not-yet-indexed docs KB — return no results, not a
        # 500, so the copilot degrades gracefully while indexing catches up.
        # Flag it as `kb_not_ready` (distinct from a real search that simply
        # matched nothing) so search_docs can surface a grounding-degraded
        # notice instead of silently answering from parametric knowledge.
        return jsonify({"results": [], "kb_not_ready": True})
    except ValueError:
        # A *different* ValueError (bad retrieval_config, embedding-dim mismatch,
        # pgvector parse failure): the RAG subsystem is broken, not empty. Don't
        # mask it as "no results" — log it and 500 so it's visible in monitoring
        # (search_docs degrades to "temporarily unavailable" on non-200).
        logger.exception("docs_search retrieval error (not empty-KB) for kb=%s", kb_id)
        return jsonify({"error": "docs search failed"}), 500
    except Exception:
        logger.exception("docs_search failed for kb=%s", kb_id)
        return jsonify({"error": "docs search failed"}), 500

    # Look up title/url from the ingested sources for citations.
    source_ids = list({str(it.source_id) for it in items if getattr(it, "source_id", None)})
    meta_by_source: dict[str, dict] = {}
    if source_ids:
        rows = db.session.execute(
            text(
                f'SELECT id, auto_metadata FROM "{AI_SCHEMA}".sources '
                "WHERE id::text = ANY(:ids)"
            ),
            {"ids": source_ids},
        ).fetchall()
        meta_by_source = {str(r[0]): (r[1] or {}) for r in rows}

    return jsonify({"results": [_format_item(it, meta_by_source) for it in items]})
