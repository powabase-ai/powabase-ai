"""Standard LLM call pattern for all PS code: BYOK-first, charge-otherwise.

Every ``litellm.completion`` / ``litellm.acompletion`` call site in the
project service should go through ``with_llm_key()`` so the v1.5 billing
invariant holds uniformly:

* If the project has a BYOK key for the provider, that key is used. The
  user pays the upstream provider; BillingLogger's BYOK-skip gate fires
  and no ``llm_call`` charge is posted.
* If no BYOK key exists, ``litellm`` falls back to the platform env key.
  BillingLogger posts an ``llm_call`` charge to recoup the platform's
  upstream cost (raw_cost × markup).

The contextmanager:
1. Resolves the BYOK key for the model (or returns None to let litellm
   read its env).
2. Enters ``billing.llm_call_scope()`` so the BYOK-skip gate is armed.

The caller must pass the yielded ``api_key`` as the ``api_key=`` kwarg
to ``litellm.(a)completion`` — otherwise the platform key is used even
when a BYOK key exists, and the user gets double-charged.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

from .ai_provider_keys_resolver import (
    get_all_user_provider_keys,
    resolve_api_key_for_model,
)
from . import billing_port as billing


# Per-task cache of the BYOK provider→key dict. ``with_llm_key``
# reads this when set; ``cached_byok_resolver`` sets it for the duration
# of a containing block. Outside the cache scope the contextvar is None
# and we fall back to the per-call DB read (existing behavior).
#
# Why opt-in: the enrichment / indexing loops iterate over hundreds of
# items, each firing one or more litellm calls. Without the cache,
# ``get_all_user_provider_keys`` issues a DB query (plus its self-heal
# DELETE+commit side-effect) per call — an N+1 against the project DB
# that's wasteful and noisy. Single-shot callers (one-off agent runs,
# the copilot turn) keep the per-call behavior so a freshly-saved
# BYOK row is picked up without restarting the worker.
_cached_byok: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "cached_byok_providers", default=None
)


@contextmanager
def cached_byok_resolver():
    """Cache the BYOK provider→key map for the duration of this block.

    Call from any task body that's about to iterate over many items
    (Celery indexing tasks, metadata enrichment, graph enrichment).
    All inner ``with_llm_key`` calls reuse the same snapshot.

    A single DB hit per task, instead of one per item × retry attempt.
    """
    tok = _cached_byok.set(get_all_user_provider_keys())
    try:
        yield
    finally:
        _cached_byok.reset(tok)


@contextmanager
def with_llm_key(model: str):
    """Resolve the operator's key for ``model`` and open the billing metered scope.

    Yields the resolved API key (or ``None`` -> litellm reads the platform env var).
    BYOK resolution is RESERVE (ships in OSS); the metered scope is a no-op in OSS
    and arms the recoup gate in cloud (see billing_port / CloudBillingAdapter).

    Uses the ``cached_byok_resolver`` snapshot when set (per-task);
    otherwise reads from the DB per call (per-request semantics).
    """
    cached = _cached_byok.get()
    provider_keys = cached if cached is not None else get_all_user_provider_keys()
    api_key = resolve_api_key_for_model(model, provider_keys)
    with billing.llm_call_scope():
        yield api_key
