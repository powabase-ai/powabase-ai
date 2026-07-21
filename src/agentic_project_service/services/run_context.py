"""Run correlation + idempotency helpers. RESERVE — ships in OSS; zero
charging logic (no prices, no billing I/O). The run-id contextvar tags logs
+ supplies ref_id; the cloud billing adapter reads it to build idempotency
keys.
"""

import contextvars
import hashlib
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass


# Carries the current run_id across the agent loop / Celery task / route
# handler so downstream billing call sites can derive deterministic
# idempotency keys without each layer having to plumb the id explicitly.
#
# Propagation rules (CPython 3.13, verified empirically):
#
#   * asyncio.Task — inherits via copy_context() at Task construction.
#     ``asyncio.run`` + ``await``ed coroutines see the parent's binding.
#   * raw threading.Thread — does NOT inherit. Worker callable runs in
#     the worker's own (empty default) context. Bind run_id inside the
#     worker function body via set_run_id, not before the Thread(...)
#     call. See routes/agents.py:_finish_run_in_background and the
#     nested ``def run_agent()`` worker inside ``run_agent_stream`` for
#     the pattern (there is also a top-level ``def run_agent(agent_id)``
#     sync handler — different function, no worker thread).
#   * concurrent.futures.ThreadPoolExecutor — does NOT inherit either.
#     stdlib's _WorkItem.run invokes ``fn(*args, **kwargs)`` directly,
#     not via Context.run. The submitter must capture the context and
#     submit ``ctx.run``: see agentic.agent.agent.Agent._run_step and
#     agentic.orchestration.strategies.ParallelEngine for the pattern.
#   * threading.Thread + asyncio.to_thread — DOES propagate (to_thread
#     copies the current context for the worker).
run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)


# Per-run sequence counters keyed by (action, args_hash). Used by tool_registry
# to disambiguate repeated same-args tool calls within a single run so each
# gets a distinct idempotency key (e.g. parallel tool_use blocks calling the
# same tool with identical args). Without this, billing dedupes on
# UNIQUE(org_id, idempotency_key) and silently under-charges every duplicate.
# Lifecycle is glued to set_run_id/reset_run_id so the dict is always a fresh
# allocation per run and never leaks to the next request reusing the worker.
call_seqs_var: contextvars.ContextVar[dict[tuple[str, str], int] | None] = contextvars.ContextVar(
    "call_seqs", default=None
)


# Sentinel token type carrying both the run_id and call_seqs Tokens so
# callers reset both with a single reset_run_id(token) call.
@dataclass(frozen=True)
class _RunIdToken:
    run_id_token: contextvars.Token
    call_seqs_token: contextvars.Token


def set_run_id(run_id: str | None) -> _RunIdToken:
    """Bind run_id to the current execution context. Also allocates a fresh
    per-run call-sequence dict so repeated same-args tool calls within this
    run produce distinct idempotency keys. Returns a token the caller MUST
    pair with reset_run_id() in a finally block.

    Where to call: inside a single sync request handler / Celery task body
    OR inside a raw threading.Thread target's function body (raw Threads
    don't inherit contextvars). asyncio coroutines inherit the bind from
    the spawning context. ThreadPoolExecutor children do NOT inherit —
    the submitter must wrap submissions in
    ``contextvars.copy_context().run`` for the binding to flow through
    (see agentic.agent.agent + agentic.orchestration.strategies).

    Nested-binding semantics: outer wins. A workflow_run that contains an
    orchestration block that contains a sub-agent ReAct loop calls
    set_run_id once at the workflow layer; nothing inside re-binds, so
    every internal tool/retrieval bills against the workflow's exec_id —
    making the workflow the billed unit. Tests/agent paths that need a
    distinct per-leaf id can re-bind, but the default is composition by
    outer-most run_id."""
    run_token = run_id_var.set(run_id)
    # Always allocate a NEW dict — never mutate an inherited one — so a
    # parent context's counters don't leak into a child run.
    seqs_token = call_seqs_var.set({} if run_id is not None else None)
    return _RunIdToken(run_id_token=run_token, call_seqs_token=seqs_token)


def reset_run_id(token: _RunIdToken) -> None:
    """Restore the previous run_id + call-seqs bindings. Pair with set_run_id."""
    run_id_var.reset(token.run_id_token)
    call_seqs_var.reset(token.call_seqs_token)


def get_run_id() -> str | None:
    """Return the run_id bound to the current execution context, or None."""
    return run_id_var.get()


@contextmanager
def run_scope(run_id: str):
    """Tag every ``llm_call`` ledger row inside this block with ``run_id``.

    Sets ``run_id_var`` on entry, restores the previous value on
    exit. Use a stable, joinable string (e.g. ``f"indexed_source:{uuid}"``
    or ``f"kb_metadata:{item_id}"``) so the ledger row's
    ``metadata->>'run_id'`` can be joined back to the source-of-truth
    table for ops auditing.

    Reset on exit is critical: Celery's default prefork pool reuses
    worker processes across tasks, so a bare ``.set()`` leaks the value
    into the next task picked up by that worker.
    """
    tok = run_id_var.set(run_id)
    try:
        yield
    finally:
        run_id_var.reset(tok)


_call_seq_lock = threading.Lock()


def next_call_seq(action: str, args_hash: str) -> int | None:
    """Return the next monotonically-increasing sequence number for an
    (action, args_hash) tuple within the current run's call-seqs map.

    Returns None when no run_id is bound — caller should fall back to a
    non-deterministic id (uuid4) in that case.

    Thread-safe via a module-level lock. ContextVar values are SHARED by
    reference across copies (copy_context is shallow), so when the agent
    loop runs concurrent tool handlers in a ThreadPoolExecutor wrapped
    with ``contextvars.copy_context().run``, all workers see the same
    underlying dict. The read-modify-write below is not atomic without
    the lock; concurrent workers would otherwise race and assign the
    same seq value to two distinct calls, defeating the disambiguation
    purpose.

    Ordering caveat: under parallel execution, the SEQ value assigned
    to each parallel call is non-deterministic — whichever worker
    acquires the lock first gets seq=N. On retry, the assignment may
    differ. Each parallel call still gets a unique key (no
    under-billing), but parallel paths are not retry-deterministic at
    this granularity. The deterministic-retry guarantee holds for
    sequential paths only. Threading LLM-provided ``tool_call_id`` is
    a future enhancement that would close this gap."""
    seqs = call_seqs_var.get()
    if seqs is None:
        return None
    key = (action, args_hash)
    with _call_seq_lock:
        seqs[key] = seqs.get(key, 0) + 1
        return seqs[key]


def make_idempotency_key(*parts: str) -> str:
    """Build a stable idempotency key from natural identifiers.

    Hashing keeps the key bounded to 64 hex chars regardless of input length
    and avoids leaking org_id/project_id values verbatim into billing's
    idempotency_keys table.
    """
    raw = ":".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:64]


def new_request_id() -> str:
    """Random request id when no natural identifier is available."""
    return str(uuid.uuid4())
