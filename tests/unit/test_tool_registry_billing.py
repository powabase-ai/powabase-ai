"""Tests for tool_registry billing wiring.

Each tool handler/execute is wrapped at load time. Tests directly invoke
the wrapper helpers because constructing a full agent + DB fixture for
load_all_tools_for_agent is heavyweight; the wrappers are the contract we
care about.
"""

import json
from unittest.mock import MagicMock

import pytest
from werkzeug.exceptions import HTTPException

from agentic_project_service.services import billing_port, tool_registry
from tests.support.billing import RecordingBillingAdapter


@pytest.fixture
def billing_env(monkeypatch):
    monkeypatch.setenv("BILLING_ORG_ID", "org-1")
    monkeypatch.setenv("PROJECT_ID", "proj-1")
    monkeypatch.delenv("BILLING_PLAN_TIER", raising=False)
    yield


def test_resolve_tool_billing_action_for_known_tools():
    assert tool_registry._resolve_tool_billing_action("code_execute") == ("agent_tool_code_execute")
    assert tool_registry._resolve_tool_billing_action("web_search") == "web_search"
    # web_search_deep is a billing ACTION (resolved from a web_search call's
    # search_type), not a tool — a literal tool_name lookup falls through to the
    # generic agent_tool_call default (there is no BuiltinTool by that name).
    assert tool_registry._resolve_tool_billing_action("web_search_deep") == "agent_tool_call"
    # web_scrape must be in the map — without it the builtin bills as the
    # generic agent_tool_call (1 credit) rather than web_scrape (5 credits),
    # silently under-charging every page scrape.
    assert tool_registry._resolve_tool_billing_action("web_scrape") == "web_scrape"


def test_resolve_tool_billing_action_defaults_to_agent_tool_call():
    """Anything not in the map bills as the generic agent_tool_call."""
    assert tool_registry._resolve_tool_billing_action("database_query") == "agent_tool_call"
    assert tool_registry._resolve_tool_billing_action("storage_read") == "agent_tool_call"
    assert tool_registry._resolve_tool_billing_action("some_unknown") == "agent_tool_call"


def test_resolve_web_search_deep_via_search_type():
    """web_search with search_type='deep' bills the pricier web_search_deep
    action — the deep tiers are values of search_type, so the resolver must
    inspect arguments."""
    assert (
        tool_registry._resolve_tool_billing_action("web_search", {"search_type": "deep"})
        == "web_search_deep"
    )


def test_resolve_web_search_deep_reasoning_via_search_type():
    """search_type='deep-reasoning' bills the priciest tier, distinct from
    plain deep."""
    assert (
        tool_registry._resolve_tool_billing_action("web_search", {"search_type": "deep-reasoning"})
        == "web_search_deep_reasoning"
    )


def test_resolve_web_search_standard_for_non_deep_search_types():
    """auto / neural / keyword / absent / no-args → standard web_search rate."""
    for st in ("auto", "neural", "keyword"):
        assert (
            tool_registry._resolve_tool_billing_action("web_search", {"search_type": st})
            == "web_search"
        )
    assert tool_registry._resolve_tool_billing_action("web_search", {"query": "x"}) == "web_search"
    assert tool_registry._resolve_tool_billing_action("web_search", {}) == "web_search"
    # Backward-compat: arguments is optional; single-arg call still resolves.
    assert tool_registry._resolve_tool_billing_action("web_search") == "web_search"


def test_resolve_deep_tiers_only_apply_to_web_search():
    """A stray deep search_type on some other tool must NOT reroute billing."""
    assert (
        tool_registry._resolve_tool_billing_action("web_scrape", {"search_type": "deep"})
        == "web_scrape"
    )
    assert (
        tool_registry._resolve_tool_billing_action(
            "database_query", {"search_type": "deep-reasoning"}
        )
        == "agent_tool_call"
    )


def test_wrap_handler_bills_deep_tier_by_search_type(billing_env, recording_billing):
    """End-to-end through the billing wrapper: each deep search_type posts its
    own charge action — deep → web_search_deep, deep-reasoning →
    web_search_deep_reasoning."""
    handler = MagicMock(return_value="results")
    wrapped = tool_registry._wrap_handler_with_billing(handler, "web_search")

    cases = {
        "deep": "web_search_deep",
        "deep-reasoning": "web_search_deep_reasoning",
    }
    for i, (search_type, expected_action) in enumerate(cases.items()):
        wrapped({"query": "anthropic", "search_type": search_type}, None)
        assert recording_billing.charges[i]["action"] == expected_action


def test_load_all_tools_for_agent_bills_override_deep_search_as_deep(
    billing_env, recording_billing, monkeypatch
):
    """A UI-pinned deep search mode (config_override) must bill the deep tier.

    This calls the PRODUCTION ``load_all_tools_for_agent`` (not a local
    re-implementation), which composes the override OUTSIDE the billing wrapper
    so the forced ``search_type`` is present when the wrapper resolves the
    action. Reverting that composition order (override inside, billing outside)
    makes this test red — it pins the real revenue-leak fix.
    """
    import types

    # One builtin web_search assignment, deep-reasoning forced via override.
    assignment = types.SimpleNamespace(
        tool_type="builtin",
        tool_name="web_search",
        config_override={"search_type": "deep-reasoning"},
    )

    class _Query:
        def filter_by(self, **_):
            return self

        def all(self):
            return [assignment]

    # Replace the whole AgentTool with a fake — patching the real
    # `AgentTool.query` attribute fails because monkeypatch reads its current
    # value (a Flask-SQLAlchemy descriptor) which needs an app context.
    class _FakeAgentTool:
        query = _Query()

    monkeypatch.setattr(tool_registry, "AgentTool", _FakeAgentTool)
    monkeypatch.setattr(tool_registry, "_get_flask_app", lambda: None)
    monkeypatch.setattr(tool_registry, "_ensure_app_context", lambda h, app: h)
    # KB + MCP tool builders query the DB (need an app context); out of scope.
    monkeypatch.setattr(tool_registry, "build_kb_tools_for_agent", lambda *a, **k: {})
    monkeypatch.setattr(tool_registry, "build_mcp_tools_for_agent", lambda *a, **k: {})

    # Stub the real Exa handler: return 200 and record the args it actually saw.
    seen: dict = {}

    def fake_web_search(arguments, context):
        seen.update(arguments)
        return json.dumps([{"ok": True}])

    monkeypatch.setitem(tool_registry.BUILTIN_HANDLERS, "web_search", fake_web_search)

    tools = tool_registry.load_all_tools_for_agent("agent-x", db_session=None)
    handler = tools["web_search"].handler

    # No search_type in the call args — only the config_override forces it.
    handler({"query": "anthropic"}, None)

    # Override injected search_type BEFORE the handler ran...
    assert seen.get("search_type") == "deep-reasoning"
    # ...and BEFORE billing resolved the action → priciest tier, not standard.
    assert recording_billing.charges[0]["action"] == "web_search_deep_reasoning"


def test_wrap_handler_calls_check_balance_then_handler_then_charge(billing_env, recording_billing):
    """Order matters: balance check FIRST, handler runs, charge posts."""
    seen_at_handler_time = {}

    def handler(arguments, context):
        # Snapshot the recording adapter's state from INSIDE the handler —
        # proves the balance check already landed and the charge has not.
        seen_at_handler_time["balance_checks"] = list(recording_billing.balance_checks)
        seen_at_handler_time["charges"] = list(recording_billing.charges)
        return "ok"

    wrapped = tool_registry._wrap_handler_with_billing(handler, "web_search")
    result = wrapped({"query": "hi"}, None)

    assert result == "ok"
    assert seen_at_handler_time["balance_checks"] == [1]
    assert seen_at_handler_time["charges"] == []
    assert len(recording_billing.charges) == 1


def test_wrap_handler_uses_correct_action_for_code_execute(billing_env, recording_billing):
    """code_execute → action=agent_tool_code_execute."""
    handler = MagicMock(return_value="ok")
    wrapped = tool_registry._wrap_handler_with_billing(handler, "code_execute")

    wrapped({"language": "python", "code": "1+1"}, None)

    assert recording_billing.balance_checks == [1]
    assert len(recording_billing.charges) == 1
    charge = recording_billing.charges[0]
    assert charge["action"] == "agent_tool_code_execute"
    assert charge["quantity"] == 1
    assert charge["ref_type"] == "tool_call"
    # idempotency_parts is the (step_id,) tail the cloud adapter combines with
    # org_id to build the real key; step_id == ref_id for tool calls.
    assert charge["idempotency_parts"] == (charge["ref_id"],)


def test_wrap_handler_uses_web_search_action_for_web_search(billing_env, recording_billing):
    handler = MagicMock(return_value="results")
    wrapped = tool_registry._wrap_handler_with_billing(handler, "web_search")

    wrapped({"query": "anthropic"}, None)

    assert recording_billing.charges[0]["action"] == "web_search"


def test_wrap_handler_uses_generic_action_for_unknown_tool(billing_env, recording_billing):
    handler = MagicMock(return_value="ok")
    wrapped = tool_registry._wrap_handler_with_billing(handler, "database_query")

    wrapped({"sql": "select 1"}, None)

    assert recording_billing.charges[0]["action"] == "agent_tool_call"


def test_wrap_handler_propagates_payment_required(billing_env):
    """402 from billing.check_balance bubbles up; handler never runs."""
    handler = MagicMock(return_value="ok")
    wrapped = tool_registry._wrap_handler_with_billing(handler, "web_search")

    rec = RecordingBillingAdapter(raise_402=True)
    billing_port.set_billing_adapter(rec)

    with pytest.raises(HTTPException) as exc_info:
        wrapped({"query": "x"}, None)
    assert exc_info.value.code == 402

    handler.assert_not_called()
    assert rec.charges == []


def test_wrap_handler_fallback_uuid4_when_no_run_id_bound(billing_env, recording_billing):
    """When no run_id is bound (non-agent caller, e.g. internal route doing
    an ad-hoc tool invocation), the wrapper falls back to uuid4 so each call
    still gets a distinct key. Exercises the fallback branch in
    ``_derive_tool_idempotency_inputs``."""
    from agentic_project_service.services import run_context

    # Pre-condition: no run_id bound in the current context.
    assert run_context.get_run_id() is None

    handler = MagicMock(return_value="ok")
    wrapped = tool_registry._wrap_handler_with_billing(handler, "web_search")

    wrapped({"query": "x"}, None)
    wrapped({"query": "y"}, None)

    assert len(recording_billing.charges) == 2
    parts0 = recording_billing.charges[0]["idempotency_parts"]
    parts1 = recording_billing.charges[1]["idempotency_parts"]
    ref0 = recording_billing.charges[0]["ref_id"]
    ref1 = recording_billing.charges[1]["ref_id"]
    assert parts0 != parts1
    assert ref0 != ref1


def test_wrap_handler_duplicate_same_args_within_run_get_distinct_keys(
    billing_env, recording_billing
):
    """Critical anti-pattern guard: parallel / repeated same-args tool calls
    within ONE run must each produce a DISTINCT idempotency_parts tuple so
    every charge lands. Reviewer follow-up C4: args_hash alone dedupes via
    UNIQUE(org_id, idempotency_key) → silent under-billing of every dup."""
    from agentic_project_service.services import run_context

    handler = MagicMock(return_value="ok")
    wrapped = tool_registry._wrap_handler_with_billing(handler, "web_search")

    token = run_context.set_run_id("run_dup_test")
    try:
        wrapped({"query": "weather in Paris"}, None)
        wrapped({"query": "weather in Paris"}, None)
    finally:
        run_context.reset_run_id(token)

    assert len(recording_billing.charges) == 2
    parts = [c["idempotency_parts"] for c in recording_billing.charges]
    assert (
        parts[0] != parts[1]
    ), "duplicate same-args calls within one run must produce DISTINCT idempotency_parts"


def test_wrap_handler_same_run_replayed_produces_identical_keys(billing_env, recording_billing):
    """Retry semantics: replaying the same run (same run_id, fresh
    set_run_id binding, same call sequence with same args) produces the
    same idempotency_parts so the cloud adapter's key rejects every retried
    charge via the UNIQUE(org_id, idempotency_key) index. Without this, a
    user-triggered retry double-charges every internal tool call."""
    from agentic_project_service.services import run_context

    handler = MagicMock(return_value="ok")
    wrapped = tool_registry._wrap_handler_with_billing(handler, "web_search")

    # Run 1 — bind run_id, fire two same-args calls, reset.
    token = run_context.set_run_id("run_retry_test")
    try:
        wrapped({"query": "weather"}, None)
        wrapped({"query": "weather"}, None)
    finally:
        run_context.reset_run_id(token)

    # Run 2 (retry) — same run_id, fresh bind, same sequence.
    token2 = run_context.set_run_id("run_retry_test")
    try:
        wrapped({"query": "weather"}, None)
        wrapped({"query": "weather"}, None)
    finally:
        run_context.reset_run_id(token2)

    parts = [c["idempotency_parts"] for c in recording_billing.charges]
    assert len(parts) == 4
    assert (
        parts[0:2] == parts[2:4]
    ), "retry of same run with same call sequence must produce identical idempotency_parts"


def test_wrap_handler_different_run_ids_get_different_keys(billing_env, recording_billing):
    """Two unrelated agent_runs charging the same tool with the same args
    must NOT collide on each other — they're different ops."""
    from agentic_project_service.services import run_context

    handler = MagicMock(return_value="ok")
    wrapped = tool_registry._wrap_handler_with_billing(handler, "web_search")

    for run in ("run_aaa", "run_bbb"):
        token = run_context.set_run_id(run)
        try:
            wrapped({"query": "same args"}, None)
        finally:
            run_context.reset_run_id(token)

    parts = [c["idempotency_parts"] for c in recording_billing.charges]
    assert len(parts) == 2
    assert parts[0] != parts[1]


def test_wrap_handler_parallel_pool_submit_via_copy_context_sees_run_id(
    billing_env, recording_billing
):
    """Reviewer C4 round 2 (parallel-pool path) — verifies the wiring
    contract between this PS package and the agentic library's
    ThreadPoolExecutor sites.

    Asserts three things explicitly so a regression in either layer fails
    this test, not just the disjointness check it had before:

      A. Workers in the positive arm OBSERVE the bound run_id via
         get_run_id() at call time (spy inside the handler).
      B. Workers in the negative arm OBSERVE get_run_id() == None
         (proves the contextvar binding is genuinely lost without
         copy_context wrapping — defends against false-positive where
         both arms silently use uuid4 fallback).
      C. The positive-arm idempotency_parts match the EXACT deterministic
         ref_id computed from the documented formula
         (run_id:action:args_hash:seq), not just "two distinct tuples" —
         which uuid4 fallback also produces. The final sha256(org_id, ...)
         wrapping is the cloud adapter's job (pinned by
         test_billing_cloud_adapter.py), not tested again here.

    The pool.submit pattern below uses ``contextvars.copy_context().run``
    inline per submission (NOT a shared parent_ctx) — matches the
    production fix at agentic.agent.agent:801 and
    services/context_handler.py:637. A shared parent_ctx would
    deterministically fail with "cannot enter context: ... is already
    entered" when two submissions overlap, regardless of how fast the
    handler is.
    """
    import contextvars
    import hashlib
    import json
    from concurrent.futures import ThreadPoolExecutor

    from agentic_project_service.services import run_context

    bound_run_id = "run_pool_test"

    # Spy handlers — capture get_run_id() seen by the worker thread.
    captured_run_ids_positive: list[str | None] = []
    captured_run_ids_negative: list[str | None] = []

    def handler_positive(args, ctx):
        captured_run_ids_positive.append(run_context.get_run_id())
        return "ok"

    def handler_negative(args, ctx):
        captured_run_ids_negative.append(run_context.get_run_id())
        return "ok"

    wrapped_pos = tool_registry._wrap_handler_with_billing(handler_positive, "web_search")
    wrapped_neg = tool_registry._wrap_handler_with_billing(handler_negative, "web_search")

    token = run_context.set_run_id(bound_run_id)
    try:
        # Positive arm: fresh copy_context().run PER submission. Matches
        # production. A single shared ctx would fail by re-entry.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(contextvars.copy_context().run, wrapped_pos, {"q": "x"}, None)
                for _ in range(2)
            ]
            for f in futures:
                f.result()
        positive_parts = [c["idempotency_parts"] for c in recording_billing.charges]

        # Negative-control arm: NO copy_context wrapping. Worker enters
        # its own empty default context; get_run_id() returns None;
        # _derive_tool_idempotency_inputs falls back to uuid4.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(wrapped_neg, {"q": "x"}, None) for _ in range(2)]
            for f in futures:
                f.result()
        negative_parts = [
            c["idempotency_parts"] for c in recording_billing.charges[len(positive_parts) :]
        ]
    finally:
        run_context.reset_run_id(token)

    # === Assertion A: workers in positive arm SAW the bound run_id ===
    assert captured_run_ids_positive == [bound_run_id, bound_run_id], (
        f"positive-arm workers must see get_run_id() == {bound_run_id!r} "
        f"(production fix at agent.py + context_handler.py + tool_registry.py "
        f"depends on this); got {captured_run_ids_positive!r}"
    )

    # === Assertion B: workers in negative arm SAW None ===
    assert captured_run_ids_negative == [None, None], (
        f"negative-arm workers must see get_run_id() == None — proves the "
        f"contextvar binding is genuinely lost without copy_context. If "
        f"this fails, the positive arm's success is a false-positive; "
        f"got {captured_run_ids_negative!r}"
    )

    # === Assertion C: positive-arm idempotency_parts match the deterministic formula ===
    # Recompute the expected ref_id from the spec formula. If production
    # silently regresses to uuid4 (e.g. someone removes copy_context.run
    # from agent.py), this set comparison fails — disjointness alone
    # cannot catch that regression.
    args_blob = json.dumps({"q": "x"}, sort_keys=True, default=repr)
    args_hash = hashlib.sha256(args_blob.encode()).hexdigest()[:16]

    def _expected_ref_id(seq: int) -> str:
        return f"{bound_run_id}:web_search:{args_hash}:{seq}"

    expected_parts = {(_expected_ref_id(1),), (_expected_ref_id(2),)}
    assert set(positive_parts) == expected_parts, (
        f"positive-arm idempotency_parts must match deterministic spec formula. "
        f"Expected {expected_parts}, got {set(positive_parts)}"
    )

    # Negative-arm parts must be uuid4-derived and must differ from the
    # deterministic set.
    assert len(negative_parts) == 2
    assert set(negative_parts).isdisjoint(expected_parts)


def test_wrap_tool_execute_skips_knowledge_search_tool(billing_env):
    """KnowledgeSearchTool/BuiltinTool aren't wrapped here (handled elsewhere).

    Verifies behaviour by checking that ``execute`` is not present as an
    instance attribute (Python looks up the dataclass-defined method on
    the class). If the wrapper rebound ``execute`` on the instance, this
    attribute would appear in ``__dict__``.
    """
    from agentic.agent.tools import BuiltinTool, KnowledgeSearchTool

    builtin = BuiltinTool(name="x", description="x", input_schema={}, handler=lambda *_: "")
    kb = KnowledgeSearchTool(name="ks", description="")

    tool_registry._wrap_tool_execute_with_billing(builtin)
    tool_registry._wrap_tool_execute_with_billing(kb)

    # No instance-level ``execute`` attribute = wrapper was skipped.
    assert "execute" not in builtin.__dict__
    assert "execute" not in kb.__dict__


def test_wrap_tool_execute_wraps_custom_tool(billing_env, recording_billing):
    """CustomTool.execute gets wrapped to fire billing around the original call."""
    from agentic.agent.tools import CustomTool

    tool = CustomTool(
        name="my_tool",
        description="d",
        input_schema={"type": "object", "properties": {}},
        endpoint="http://example.com",
    )

    sentinel = MagicMock(return_value="custom-result")
    tool.execute = sentinel  # type: ignore[assignment]

    tool_registry._wrap_tool_execute_with_billing(tool)

    out = tool.execute({"foo": "bar"}, None)

    assert out == "custom-result"
    sentinel.assert_called_once()
    assert recording_billing.balance_checks == [1]
    assert len(recording_billing.charges) == 1
    assert recording_billing.charges[0]["action"] == "agent_tool_call"


def test_wrap_tool_execute_propagates_402(billing_env):
    """Wrapped custom-tool execute propagates a 402 without running."""
    from agentic.agent.tools import CustomTool

    tool = CustomTool(
        name="my_tool",
        description="d",
        input_schema={"type": "object", "properties": {}},
        endpoint="http://example.com",
    )
    sentinel = MagicMock(return_value="x")
    tool.execute = sentinel  # type: ignore[assignment]

    tool_registry._wrap_tool_execute_with_billing(tool)

    rec = RecordingBillingAdapter(raise_402=True)
    billing_port.set_billing_adapter(rec)

    with pytest.raises(HTTPException) as exc_info:
        tool.execute({}, None)
    assert exc_info.value.code == 402

    sentinel.assert_not_called()
    assert rec.charges == []


def test_wrap_handler_skips_post_charge_when_handler_marks_platform_error(
    billing_env, recording_billing
):
    """Platform-misconfig errors carry a `_platform_error: true` marker in
    the result JSON. The wrapper must NOT debit the tenant for the
    platform's own misconfiguration — e.g., EXA_API_KEY missing from pod
    env when web_search fires.
    """

    def handler(args, ctx):
        return json.dumps(
            {
                "error": "Web search is currently unavailable. Please try again later.",
                "_platform_error": True,
            }
        )

    wrapped = tool_registry._wrap_handler_with_billing(handler, "web_search")
    result = wrapped({"query": "anything"}, MagicMock())

    assert recording_billing.charges == []
    # The wrapper strips the internal `_platform_error` marker before
    # returning so the LLM never sees the signaling key in its tool-result
    # message. The user-facing `error` content is preserved.
    parsed = json.loads(result)
    assert "_platform_error" not in parsed
    assert "currently unavailable" in parsed["error"]


def test_wrap_handler_still_bills_on_handler_error_without_platform_marker(
    billing_env, recording_billing
):
    """Generic handler errors (bad input, transient API failure) MUST still
    bill — the platform-error skip is opt-in via the marker, not a blanket
    skip-on-any-error. Prevents tenants from gaming the billing surface by
    crafting inputs that error.
    """

    def handler(args, ctx):
        # Error from a handler that ISN'T a platform misconfiguration —
        # e.g., upstream API returned 4xx for the tenant's query.
        return json.dumps({"error": "Exa returned 400: invalid query"})

    wrapped = tool_registry._wrap_handler_with_billing(handler, "web_search")
    wrapped({"query": "anything"}, MagicMock())

    assert len(recording_billing.charges) == 1


def test_wrap_handler_bills_normally_on_non_json_result(billing_env, recording_billing):
    """Handlers that return non-JSON strings (e.g., code_execute returning
    stdout) must continue to bill. The platform-error marker check is
    JSON-only.
    """

    def handler(args, ctx):
        return "plain text result from code_execute"

    wrapped = tool_registry._wrap_handler_with_billing(handler, "code_execute")
    wrapped({"code": "print(1)"}, MagicMock())

    assert len(recording_billing.charges) == 1
