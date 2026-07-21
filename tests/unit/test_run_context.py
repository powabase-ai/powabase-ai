"""Tests for run_context — run correlation + idempotency helpers.

RESERVE (ships in OSS): run_id_var / set_run_id / reset_run_id / get_run_id /
run_scope / call_seqs_var / next_call_seq / make_idempotency_key /
new_request_id. No charging logic — see test_billing_context.py for the
BillingContext / BYOK-tracking symbols, now in billing_cloud/identity.py (CUT).
"""

import pytest

from agentic_project_service.services import run_context as rc


def test_make_idempotency_key_is_deterministic():
    """Same parts → same key. Different parts → different key."""
    k1 = rc.make_idempotency_key("org-1", "vector_search", "req-1")
    k2 = rc.make_idempotency_key("org-1", "vector_search", "req-1")
    k3 = rc.make_idempotency_key("org-1", "vector_search", "req-2")
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 64


def test_new_request_id_unique():
    """new_request_id returns a UUID4-ish string; two calls don't collide."""
    a = rc.new_request_id()
    b = rc.new_request_id()
    assert a != b
    assert len(a) > 0


def test_get_run_id_defaults_to_none():
    """No set_run_id call yet → get_run_id() reads the ContextVar default."""
    assert rc.get_run_id() is None


def test_set_run_id_binds_and_reset_run_id_restores():
    """set_run_id binds run_id_var; reset_run_id restores the prior (None)
    binding — the pair is the primitive contract every route/task wraps."""
    assert rc.get_run_id() is None
    token = rc.set_run_id("run-abc")
    try:
        assert rc.get_run_id() == "run-abc"
    finally:
        rc.reset_run_id(token)
    assert rc.get_run_id() is None


def test_reset_run_id_restores_prior_nested_binding():
    """Nested set_run_id/reset_run_id pairs restore the OUTER binding, not
    None — proves reset_run_id replays the captured Token rather than
    always clearing back to the ContextVar's default. Matches the
    documented 'outer wins' composition semantics in set_run_id."""
    outer = rc.set_run_id("run-outer")
    try:
        assert rc.get_run_id() == "run-outer"
        inner = rc.set_run_id("run-inner")
        try:
            assert rc.get_run_id() == "run-inner"
        finally:
            rc.reset_run_id(inner)
        assert rc.get_run_id() == "run-outer"
    finally:
        rc.reset_run_id(outer)
    assert rc.get_run_id() is None


def test_next_call_seq_is_none_without_a_bound_run():
    """No run_id bound → call_seqs_var is None → next_call_seq returns None
    so callers fall back to a non-deterministic uuid4 id."""
    assert rc.get_run_id() is None
    assert rc.next_call_seq("web_search", "abc123") is None


def test_next_call_seq_is_monotonic_within_a_run():
    """Repeated calls for the SAME (action, args_hash) increment 1, 2, 3..."""
    token = rc.set_run_id("run-seq")
    try:
        assert rc.next_call_seq("web_search", "hash-a") == 1
        assert rc.next_call_seq("web_search", "hash-a") == 2
        assert rc.next_call_seq("web_search", "hash-a") == 3
    finally:
        rc.reset_run_id(token)


def test_next_call_seq_tracks_distinct_keys_independently():
    """Different (action, args_hash) tuples get their own counters — so two
    parallel tool_use blocks calling DIFFERENT tools (or the same tool with
    different args) don't share a sequence."""
    token = rc.set_run_id("run-seq-2")
    try:
        assert rc.next_call_seq("web_search", "hash-a") == 1
        assert rc.next_call_seq("web_search", "hash-b") == 1
        assert rc.next_call_seq("web_search", "hash-a") == 2
    finally:
        rc.reset_run_id(token)


def test_set_run_id_allocates_a_fresh_seq_dict_per_run():
    """A new set_run_id binding — even replaying the SAME run_id string
    (retry semantics) — starts its call-seq counters at a fresh dict, not
    one inherited/leaked from the previous binding. Load-bearing for
    retry-determinism: replaying a run must reproduce the same seq=1,2,...
    sequence, not continue from where the prior run's counters left off."""
    token1 = rc.set_run_id("run-replay")
    try:
        assert rc.next_call_seq("web_search", "hash-a") == 1
        assert rc.next_call_seq("web_search", "hash-a") == 2
    finally:
        rc.reset_run_id(token1)

    token2 = rc.set_run_id("run-replay")  # same run_id, fresh bind (retry)
    try:
        assert rc.next_call_seq("web_search", "hash-a") == 1, (
            "retry of the same run_id must restart the seq counter at 1, "
            "not continue from the prior binding's state"
        )
    finally:
        rc.reset_run_id(token2)


def test_run_scope_sets_and_resets_run_id():
    """run_scope is the ergonomic set+reset wrapper around set_run_id/
    reset_run_id — get_run_id() reads the bound value inside the block and
    reverts to the prior value on exit."""
    assert rc.get_run_id() is None
    with rc.run_scope("run-scoped"):
        assert rc.get_run_id() == "run-scoped"
    assert rc.get_run_id() is None


def test_run_scope_resets_on_exception():
    """The reset must fire even when the block raises — Celery's prefork
    pool reuses worker processes across tasks, so a leaked binding would
    poison the next task picked up by that worker."""
    assert rc.get_run_id() is None
    with pytest.raises(ValueError):
        with rc.run_scope("run-error"):
            assert rc.get_run_id() == "run-error"
            raise ValueError("boom")
    assert rc.get_run_id() is None
