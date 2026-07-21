"""Tests for BYOK-only fail-fast at agent/orchestration/workflow run entry.

Covers:

  * ``platform_supports`` — provider→pod-env lookup (factor 2 of AI-on-us).
  * ``check_model_available`` — abort 400 when neither BYOK nor platform key
    is available for the model's provider. Reads BYOK presence via
    ``list_byok_providers()`` (a direct DB query) — independent of billing.
  * Route-level fail-fast wiring in agents / orchestrations / workflows — the
    handler aborts 400 before touching the DB / LLM when the model is BYOK-only
    and no project key is configured.

The helpers live in ``services/llm_availability.py``; the route wiring is in
``routes/{agents,orchestrations,workflows}.py``. Tests follow the existing
``test_agents_route_billing.py`` pattern (lightweight Flask app, patch the
balance_check + auth so the route reaches the model check before any DB).
"""

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from agentic_project_service.routes import agents as agents_route
from agentic_project_service.routes import orchestrations as orchestrations_route
from agentic_project_service.routes import webhooks as webhooks_route
from agentic_project_service.routes import workflows as workflows_route
from agentic_project_service.services import llm_availability
from agentic_project_service.services.llm_availability import (
    check_model_available,
    platform_supports,
)
from agentic_project_service.tasks import scheduler as scheduler_task


@pytest.fixture
def no_platform_keys(monkeypatch):
    """Strip all platform env keys so platform_supports returns False everywhere."""
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(env, raising=False)


@pytest.fixture
def billing_env(monkeypatch):
    monkeypatch.setenv("BILLING_ORG_ID", "org-1")
    monkeypatch.setenv("PROJECT_ID", "proj-1")
    monkeypatch.delenv("BILLING_PLAN_TIER", raising=False)
    yield


# ---------------------------------------------------------------------------
# platform_supports
# ---------------------------------------------------------------------------


def test_platform_supports_returns_true_when_env_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform")
    assert platform_supports("openai") is True


def test_platform_supports_returns_false_when_env_missing(no_platform_keys):
    assert platform_supports("openai") is False
    assert platform_supports("anthropic") is False


def test_platform_supports_returns_false_for_unknown_provider(no_platform_keys):
    """Unknown providers are not in _PROVIDER_ENV → never platform-supported."""
    assert platform_supports("totally-fake-provider") is False


# ---------------------------------------------------------------------------
# check_model_available
# ---------------------------------------------------------------------------


def test_check_model_available_proceeds_when_byok_present(no_platform_keys):
    """BYOK key for the provider → proceed even with no platform key."""
    with patch.object(
        llm_availability, "list_byok_providers", return_value=frozenset({"openrouter"})
    ):
        # Must run inside a Flask app context for abort() to be reachable —
        # but proceed path never aborts so we don't need one.
        check_model_available("openrouter/meta-llama/llama-4-maverick")


def test_check_model_available_proceeds_when_platform_supports(monkeypatch):
    """No BYOK, but platform env key set → AI-on-us proceed path."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-platform")
    with patch.object(llm_availability, "list_byok_providers", return_value=frozenset()):
        check_model_available("anthropic/claude-sonnet-4-6")


def test_check_model_available_aborts_400_when_no_key_anywhere(no_platform_keys):
    """No BYOK + no platform key → abort 400."""
    app = Flask(__name__)
    with patch.object(llm_availability, "list_byok_providers", return_value=frozenset()):
        with app.test_request_context("/"):
            with pytest.raises(Exception) as exc_info:
                check_model_available("openrouter/some-model")
            # Werkzeug HTTPException carries code=400 on abort(400).
            assert getattr(exc_info.value, "code", None) == 400
            description = getattr(exc_info.value, "description", "") or str(exc_info.value)
            assert "BYOK" in description
            assert "openrouter" in description


def test_check_model_available_handles_litellm_get_provider_failure(no_platform_keys):
    """When litellm.get_llm_provider raises, we fall back to splitting on '/'."""
    app = Flask(__name__)
    with patch.object(llm_availability, "list_byok_providers", return_value=frozenset()):
        with patch(
            "litellm.get_llm_provider",
            side_effect=RuntimeError("model not recognized"),
        ):
            with app.test_request_context("/"):
                with pytest.raises(Exception) as exc_info:
                    check_model_available("openrouter/foo-bar")
                assert getattr(exc_info.value, "code", None) == 400
                # The fallback parses "openrouter" from "openrouter/foo-bar".
                description = getattr(exc_info.value, "description", "") or str(exc_info.value)
                assert "openrouter" in description


# ---------------------------------------------------------------------------
# Broken-session degrade — pins the try/except around list_byok_providers()
# ---------------------------------------------------------------------------
#
# check_model_available's rewrite DB-queries directly (list_byok_providers()),
# unlike the old contextvar read (current_byok_providers.get()) which never
# touched the DB. A broken session (e.g. a PendingRollbackError left by a
# prior aborted handler) must degrade to "no BYOK key seen", NOT 500 the run
# entry — mirroring the swallow in _set_billing_byok (main.py).


def test_check_model_available_degrades_to_400_when_list_byok_providers_raises(
    no_platform_keys,
):
    """A broken DB session (list_byok_providers raises) must degrade to the
    platform/400 path, NOT propagate as a 500.

    Counterfactual: delete the try/except around ``list_byok_providers()`` in
    check_model_available — the raw Exception from the patched
    list_byok_providers propagates instead of the werkzeug BadRequest(400),
    so ``exc_info.value.code`` is None instead of 400.
    """
    app = Flask(__name__)
    with patch.object(
        llm_availability, "list_byok_providers", side_effect=Exception("db session broken")
    ):
        with app.test_request_context("/"):
            with pytest.raises(Exception) as exc_info:
                check_model_available("openrouter/some-model")
            assert getattr(exc_info.value, "code", None) == 400
            description = getattr(exc_info.value, "description", "") or str(exc_info.value)
            assert "BYOK" in description


# ---------------------------------------------------------------------------
# Provider-alias contract — applies symmetrically across BYOK and platform-key.
# ---------------------------------------------------------------------------
#
# litellm classifies `gemini/<x>` as provider "gemini", but our BYOK rows + the
# platform env are stored under "google" (GOOGLE_API_KEY). The BYOK side
# already aliases gemini→google; the platform-supports side must do the same,
# otherwise GOOGLE_API_KEY is invisible to the gemini path and the route
# false-positive 400s "Add an API key for gemini" even though litellm would
# happily fall back GEMINI_API_KEY → GOOGLE_API_KEY at call time.
#
# Counterfactual: revert the alias in check_model_available (pass `provider`
# instead of `byok_provider` to platform_supports) → these tests fail.


def test_check_model_available_proceeds_for_gemini_when_google_platform_key_set(
    monkeypatch,
):
    """`gemini/<x>` with no BYOK but GOOGLE_API_KEY set → proceed (alias)."""
    # Strip the env litellm classifies the prefix to, so the only path that
    # can satisfy the check is the gemini→google alias being honored.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "platform-google-key")
    with patch.object(llm_availability, "list_byok_providers", return_value=frozenset()):
        # Proceed path does not abort — no Flask request context needed.
        check_model_available("gemini/gemini-3.1-pro-preview")


def test_check_model_available_proceeds_for_vertex_ai_when_google_platform_key_set(
    monkeypatch,
):
    """`vertex_ai/<x>` shares the same google alias path as gemini/."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "platform-google-key")
    with patch.object(llm_availability, "list_byok_providers", return_value=frozenset()):
        check_model_available("vertex_ai/gemini-2.5-flash")


def test_check_model_available_still_aborts_for_gemini_when_no_google_key(
    no_platform_keys,
):
    """The alias only HELPS when GOOGLE_API_KEY is actually set; without it
    AND without BYOK, the route must still abort 400 (regression guard so
    the alias fix doesn't silently swallow real misconfigurations)."""
    app = Flask(__name__)
    with patch.object(llm_availability, "list_byok_providers", return_value=frozenset()):
        with app.test_request_context("/"):
            with pytest.raises(Exception) as exc_info:
                check_model_available("gemini/gemini-3.1-pro-preview")
            assert getattr(exc_info.value, "code", None) == 400


# ---------------------------------------------------------------------------
# Route-level fail-fast — agents / orchestrations / workflows
# ---------------------------------------------------------------------------


def _make_test_app():
    """Mount the three blueprints under test on a fresh Flask app."""
    app = Flask(__name__)
    app.register_blueprint(agents_route.agents_bp)
    app.register_blueprint(orchestrations_route.orchestrations_bp)
    app.register_blueprint(workflows_route.workflows_bp)
    return app


def _auth_headers():
    return {"Authorization": "Bearer fake"}


def test_agents_route_imports_check_model_available():
    """Wiring contract: routes/agents.py imports check_model_available."""
    assert hasattr(agents_route, "check_model_available")


def test_orchestrations_route_imports_check_model_available():
    """Wiring contract: routes/orchestrations.py imports check_model_available."""
    assert hasattr(orchestrations_route, "check_model_available")


def test_workflows_route_imports_check_model_available():
    """Wiring contract: routes/workflows.py imports check_model_available."""
    assert hasattr(workflows_route, "check_model_available")


def test_run_agent_fail_fast_when_model_byok_only_and_no_key(billing_env, no_platform_keys):
    """openrouter model + no BYOK key + no platform key → 400 fail-fast.

    The balance check goes through the billing port (a no-op without a
    registered cloud adapter) and needs no patching. We patch the DB fetch
    so the route reaches check_model_available, then assert the 400.
    """
    app = _make_test_app()

    # Stub the agent fetch so the route believes the agent exists and is
    # using an openrouter model that has no key. We patch via db.session.execute
    # to control the SELECT result.
    fake_agent_row = (
        "agent-1",
        "Test Agent",
        "openrouter/meta-llama/llama-4-maverick",
        "",
        {},
    )

    with (
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
        patch.object(agents_route.db.session, "execute") as mock_exec,
    ):
        result = mock_exec.return_value
        result.fetchone.return_value = fake_agent_row
        result.scalar.return_value = 0
        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run",
                json={"message": "hi"},
                headers=_auth_headers(),
            )

    assert resp.status_code == 400
    # Flask's abort(400, description=...) renders the description into the
    # HTML body by default; we check the raw body for the actionable message.
    body_text = resp.get_data(as_text=True)
    assert "BYOK" in body_text or "openrouter" in body_text


def test_run_agent_stream_fail_fast_when_model_byok_only(billing_env, no_platform_keys):
    """Streaming endpoint also fails fast BEFORE entering the SSE generator."""
    app = _make_test_app()

    fake_agent_row = (
        "agent-1",
        "Test Agent",
        "openrouter/meta-llama/llama-4-maverick",
        "",
        {},
    )

    with (
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
        patch.object(agents_route.db.session, "execute") as mock_exec,
    ):
        result = mock_exec.return_value
        result.fetchone.return_value = fake_agent_row
        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run/stream",
                json={"message": "hi"},
                headers=_auth_headers(),
            )

    assert resp.status_code == 400


def test_run_agent_proceeds_when_provider_is_platform_supported(billing_env, monkeypatch):
    """Anthropic has a platform env key → check_model_available proceeds.

    We assert that check_model_available was called and returned without
    aborting (i.e. the route progressed past the gate). Downstream of the
    gate the agent run will fail in this lightweight test environment (no
    DB session, no agentic.Agent.run) — that's fine; we only care the
    400-with-BYOK message did NOT fire.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-platform")
    app = _make_test_app()

    fake_agent_row = (
        "agent-1",
        "Test Agent",
        "anthropic/claude-sonnet-4-6",
        "",
        {},
    )

    with (
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
        patch.object(
            agents_route, "check_model_available", wraps=check_model_available
        ) as mock_check,
        patch.object(agents_route.db.session, "execute") as mock_exec,
    ):
        result = mock_exec.return_value
        result.fetchone.return_value = fake_agent_row
        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run",
                json={"message": "hi"},
                headers=_auth_headers(),
            )

    # check_model_available was invoked at least once and did not abort 400
    # with the BYOK message. (Anything downstream of the gate may still 500
    # in this minimal harness, but the gate itself proceeded.)
    mock_check.assert_called()
    if resp.status_code == 400:
        body = resp.get_json() or {}
        description = body.get("description") or body.get("message") or ""
        assert "BYOK" not in description


def test_run_orchestration_stream_fail_fast_when_sub_agent_byok_only(billing_env, monkeypatch):
    """Orchestration's sub-agent uses a BYOK-only model with no key → 400 fail-fast.

    The orchestrator's own model (anthropic) has a platform env key and would
    proceed, but the orchestration references a sub-agent (entity_type='agent')
    whose model is openrouter-only with no BYOK key — that should fail-fast at
    run entry with 400, BEFORE the SSE stream begins, NOT mid-loop when the
    sub-agent's LLM call eventually fails.

    The route enumerates sub-agents via a helper (``_load_sub_agent_models``)
    so this test can stub the enumeration without wiring up Flask-SQLAlchemy.
    """
    # Orchestrator can run (platform key present for its provider)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-platform")
    # Sub-agent's openrouter provider has NO platform key
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    app = _make_test_app()

    # Fake the orchestration row: its own model is anthropic (resolves).
    fake_orch_row = MagicMock()
    fake_orch_row.settings = {}
    fake_orch_row.orchestrator_config = {"model": "anthropic/claude-sonnet-4-6"}

    with (
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
        patch.object(orchestrations_route.db.session, "get", return_value=fake_orch_row),
        patch.object(
            orchestrations_route,
            "_load_sub_agent_models",
            return_value=["openrouter/meta-llama/llama-4-maverick"],
        ),
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/orchestrations/orch-1/run/stream",
                json={"message": "hi"},
                headers=_auth_headers(),
            )

    # 400 fail-fast — BEFORE the SSE stream begins. The streaming endpoint
    # would otherwise return 200 + a `text/event-stream` body and surface
    # this as a stream `event: error` deep in the generator.
    assert resp.status_code == 400
    body_text = resp.get_data(as_text=True)
    # Error message identifies the sub-agent's openrouter provider, not the
    # orchestrator's anthropic one — the orchestrator passes the check.
    assert "openrouter" in body_text
    assert "BYOK" in body_text


def test_run_orchestration_stream_proceeds_when_all_sub_agents_have_keys(billing_env, monkeypatch):
    """All sub-agent models resolve (platform key present) → gate proceeds.

    Asserts check_model_available was invoked for both the orchestrator AND
    the sub-agent without aborting 400 with the BYOK message. Anything past
    the gate may 500 in this minimal harness (no DB/Agent/litellm wiring) —
    we only care the gate itself proceeded.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-platform")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform")
    app = _make_test_app()

    fake_orch_row = MagicMock()
    fake_orch_row.settings = {}
    fake_orch_row.orchestrator_config = {"model": "anthropic/claude-sonnet-4-6"}

    with (
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
        patch.object(orchestrations_route.db.session, "get", return_value=fake_orch_row),
        patch.object(
            orchestrations_route,
            "_load_sub_agent_models",
            return_value=["openai/gpt-4o"],
        ),
        patch.object(
            orchestrations_route,
            "check_model_available",
            wraps=check_model_available,
        ) as mock_check,
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/orchestrations/orch-1/run/stream",
                json={"message": "hi"},
                headers=_auth_headers(),
            )

    # check_model_available invoked for orchestrator AND each sub-agent.
    assert mock_check.call_count >= 2
    called_models = [c.args[0] for c in mock_check.call_args_list]
    assert "anthropic/claude-sonnet-4-6" in called_models
    assert "openai/gpt-4o" in called_models
    # If the gate proceeds and downstream errors, the response is NOT a 400
    # with the BYOK message.
    if resp.status_code == 400:
        body = resp.get_data(as_text=True)
        assert "BYOK" not in body


# ---------------------------------------------------------------------------
# Workflow execute (sync + stream) — behavioural gate for direct /execute path
# ---------------------------------------------------------------------------
#
# Above we have an import-contract test
# (`test_workflows_route_imports_check_model_available`) but no behavioural
# test that asserts the route ACTUALLY aborts 400 when an agent block's
# model is BYOK-only and no key is configured. The webhook + scheduler
# tests below cover the indirect paths into workflow execution, but the
# direct ``POST /api/workflows/<id>/execute`` + ``/execute/stream`` paths
# need their own coverage — a refactor that nuked the in-route enumeration
# loop while keeping the ``check_model_available`` import would pass the
# ``hasattr`` test today.


@pytest.mark.parametrize("endpoint", ["execute", "execute/stream"])
def test_run_workflow_fail_fast_when_block_byok_only(
    billing_env, no_platform_keys, endpoint, recording_billing
):
    """``/execute`` and ``/execute/stream`` both 400 before workflow_executions
    insert when any agent block's model has no BYOK + no platform key.

    Parametrized over both endpoints so a refactor that adds the gate to
    one but forgets the other is caught. Patches ``load_blocks`` to inject
    a BYOK-only agent block and ``build_workflow_from_db`` so the gate
    runs before any actual workflow construction. ``charge_workflow_blocks``
    is patched, and ``recording_billing`` is asserted, so a regression that
    lets the handler reach the billing call would also be visible.
    """
    app = _make_test_app()

    blocks_data = [
        {
            "id": "block-1",
            "type": "agent",
            "config": {"model": "openrouter/meta-llama/llama-4-maverick"},
        },
    ]

    fake_wf = MagicMock()
    fake_wf.arun_detailed = MagicMock()
    fake_wf.astream = MagicMock()

    with (
        patch.object(llm_availability, "list_byok_providers", return_value=frozenset()),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
        patch.object(workflows_route, "build_workflow_from_db", return_value=fake_wf),
        patch.object(workflows_route, "load_blocks", return_value=blocks_data),
        patch.object(workflows_route, "load_edges", return_value=[]),
        patch.object(workflows_route, "charge_workflow_blocks") as mock_block_charge,
        patch.object(workflows_route.db.session, "execute") as mock_exec,
        patch.object(workflows_route.db.session, "commit"),
    ):
        with app.test_client() as client:
            resp = client.post(
                f"/api/workflows/wf-1/{endpoint}",
                json={"variables": {}},
                headers=_auth_headers(),
            )

    # 400 fail-fast — gate fires BEFORE the workflow_executions INSERT.
    assert resp.status_code == 400, f"endpoint={endpoint} resp.body={resp.get_data(as_text=True)}"
    body_text = resp.get_data(as_text=True)
    assert "BYOK" in body_text
    assert "openrouter" in body_text
    # arun_detailed / astream were never invoked (we never opened the engine).
    fake_wf.arun_detailed.assert_not_called()
    fake_wf.astream.assert_not_called()
    # No billing emitted.
    assert recording_billing.charges == []
    mock_block_charge.assert_not_called()
    # The INSERT INTO workflow_executions was never reached (gate is BEFORE it).
    insert_calls = [
        c
        for c in mock_exec.call_args_list
        if c.args
        and "INSERT" in str(c.args[0]).upper()
        and "WORKFLOW_EXECUTIONS" in str(c.args[0]).upper()
    ]
    assert not insert_calls, f"workflow_executions INSERT reached before gate: endpoint={endpoint}"


# ---------------------------------------------------------------------------
# Webhook + scheduler — bypass paths that skipped the gate before this fix
# ---------------------------------------------------------------------------
#
# Without this gate, a webhook-triggered or scheduled workflow whose agent
# block referenced a BYOK-only model would proceed all the way to the LLM
# call deep inside wf.arun_detailed and surface as LiteLLM's generic
# "Missing API Key" — masking the actionable "no project key for <provider>"
# message that the /execute path already surfaces at run entry.


def test_run_workflow_webhook_fail_fast_when_block_byok_only(
    billing_env, no_platform_keys, recording_billing
):
    """Webhook trigger must enumerate agent-block models and 400 before
    creating workflow_executions / arun_detailed when any block model has
    neither a BYOK key nor a platform key.

    Mirrors the agent-block enumeration already in routes/workflows.py:515
    (the synchronous /execute path). Without this enumeration, a public
    webhook URL routes BYOK-only block models straight into the LLM call
    where LiteLLM raises a generic "Missing API Key" deep in the engine.
    """
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(webhooks_route.webhooks_bp)

    # Webhook flow: 1st db.session.execute returns the webhook block row,
    # 2nd returns the workflow state row, then load_blocks is patched.
    # arun_detailed must NOT be reached — we patch build_workflow_from_db
    # and assert it's never called.
    def session_side(*args, **kwargs):
        if not hasattr(session_side, "calls"):
            session_side.calls = 0
        session_side.calls += 1
        if session_side.calls == 1:
            r = MagicMock()
            r.fetchone.return_value = (
                "block-1",
                "22222222-2222-2222-2222-222222222222",
                {"webhook_secret": "sekret"},
            )
            return r
        elif session_side.calls == 2:
            r = MagicMock()
            r.fetchone.return_value = ("deployed", None)
            return r
        else:
            # The agent_id → model lookup inside the enumeration may also
            # hit db.session.execute; return None so the loop falls through
            # gracefully when block already has cfg["model"] inline.
            r = MagicMock()
            r.fetchone.return_value = None
            return r

    mock_session = MagicMock()
    mock_session.execute.side_effect = session_side

    blocks_data = [
        {
            "id": "block-1",
            "type": "webhook",
            "config": {},
        },
        {
            "id": "block-2",
            "type": "agent",
            "config": {"model": "openrouter/meta-llama/llama-4-maverick"},
        },
    ]

    # Build returns a workflow object whose arun_detailed would otherwise run.
    # If the gate fires correctly, arun_detailed is never awaited.
    fake_wf = MagicMock()
    fake_wf.arun_detailed = MagicMock()

    with (
        patch.object(llm_availability, "list_byok_providers", return_value=frozenset()),
        patch("agentic_project_service.routes.webhooks.db.session", mock_session),
        patch.object(webhooks_route, "_workflow_pre_check"),
        patch.object(webhooks_route, "load_blocks", return_value=blocks_data),
        patch.object(webhooks_route, "load_edges", return_value=[]),
        patch.object(webhooks_route, "build_workflow_from_db", return_value=fake_wf),
        patch.object(webhooks_route, "charge_workflow_blocks") as mock_block_charge,
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/webhooks/11111111-1111-1111-1111-111111111111",
                json={"payload": "x"},
                headers={"Authorization": "Bearer sekret"},
            )

    # 400 fail-fast — arun_detailed never awaited, no charge posted.
    assert resp.status_code == 400
    body_text = resp.get_data(as_text=True)
    assert "BYOK" in body_text
    assert "openrouter" in body_text
    fake_wf.arun_detailed.assert_not_called()
    assert recording_billing.charges == []
    mock_block_charge.assert_not_called()


def test_scheduler_tick_fail_fast_when_block_byok_only(billing_env, no_platform_keys):
    """The scheduler's _execute_scheduled_workflow must call check_model_available
    on each agent block's resolved model BEFORE invoking wf.arun_detailed.

    Without the gate, every scheduled tick of a BYOK-only-block workflow
    would burn through the LLM call only to hit LiteLLM's generic error;
    worse, the workflow_executions row would be created and left in
    'running' indefinitely.
    """
    from werkzeug.exceptions import BadRequest

    blocks_data = [
        {
            "id": "block-1",
            "type": "agent",
            "config": {"model": "openrouter/meta-llama/llama-4-maverick"},
        },
    ]

    # Mock the workflow object so it would otherwise run if reached.
    fake_wf = MagicMock()
    fake_wf.arun_detailed = MagicMock()

    with (
        patch.object(llm_availability, "list_byok_providers", return_value=frozenset()),
        patch.object(scheduler_task, "check_model_available") as mock_check,
        patch(
            "agentic_project_service.routes._workflow_helpers.build_workflow_from_db",
            return_value=fake_wf,
        ),
        patch(
            "agentic_project_service.routes._workflow_helpers.load_blocks",
            return_value=blocks_data,
        ),
        patch(
            "agentic_project_service.routes._workflow_helpers.load_edges",
            return_value=[],
        ),
        patch.object(scheduler_task.db.session, "execute"),
        patch.object(scheduler_task.db.session, "commit"),
    ):
        # Configure check_model_available to raise the abort like the real one.
        mock_check.side_effect = BadRequest(
            "Model openrouter/meta-llama/llama-4-maverick requires BYOK"
        )

        # The gate fires BEFORE the workflow_executions INSERT and before
        # the try-block that would otherwise mark the row 'failed'. The
        # abort therefore propagates out of _execute_scheduled_workflow,
        # which is fine: the calling scheduler_tick swallows it with its
        # outer try/except.
        with pytest.raises(BadRequest):
            scheduler_task._execute_scheduled_workflow("wf-1")

    # check_model_available was called on the BYOK-only block model — the
    # entire point of this fix.
    mock_check.assert_called_with("openrouter/meta-llama/llama-4-maverick")
    # arun_detailed must NOT be invoked once the gate fires.
    fake_wf.arun_detailed.assert_not_called()


def test_scheduler_tick_imports_check_model_available():
    """Wiring contract: tasks/scheduler.py imports check_model_available."""
    assert hasattr(scheduler_task, "check_model_available")


def test_webhooks_route_imports_check_model_available():
    """Wiring contract: routes/webhooks.py imports check_model_available."""
    assert hasattr(webhooks_route, "check_model_available")


# ---------------------------------------------------------------------------
# Scheduler tight-loop fix — _maybe_execute_scheduled must NOT let a BadRequest
# from check_model_available skip the last_scheduled_at UPDATE. Without this,
# every 30s tick re-classifies the workflow as due and re-aborts forever.
# ---------------------------------------------------------------------------


def test_scheduler_maybe_execute_advances_last_scheduled_at_when_check_model_aborts(
    billing_env, no_platform_keys
):
    """A workflow with a BYOK-only model that has no BYOK key must NOT cause
    a tight loop. ``_maybe_execute_scheduled`` must advance last_scheduled_at
    and swallow the BadRequest so the next 30s tick does not immediately
    re-classify the same workflow as due.

    Counterfactual: remove the try/except in _maybe_execute_scheduled and the
    UPDATE call is never issued (BadRequest unwinds past it).
    """
    from datetime import UTC, datetime, timedelta

    from werkzeug.exceptions import BadRequest

    # Schedule with an interval that long-since elapsed — workflow is overdue.
    sched = {"type": "interval", "interval_seconds": 60, "enabled": True}
    long_ago = datetime.now(UTC) - timedelta(hours=1)

    with (
        patch.object(llm_availability, "list_byok_providers", return_value=frozenset()),
        patch.object(scheduler_task, "_execute_scheduled_workflow") as mock_exec,
        patch.object(scheduler_task.db.session, "execute") as mock_db_exec,
        patch.object(scheduler_task.db.session, "commit") as mock_commit,
    ):
        # Simulate the real failure: check_model_available aborts deep inside
        # _execute_scheduled_workflow, which propagates BadRequest upward.
        mock_exec.side_effect = BadRequest(
            "Model openrouter/meta-llama/llama-4-maverick requires BYOK. "
            "Add an API key for openrouter in Settings → LLM Provider Keys."
        )

        # Must NOT raise — _maybe_execute_scheduled handles the abort.
        scheduler_task._maybe_execute_scheduled(
            wf_id="11111111-1111-1111-1111-111111111111",
            sched=sched,
            run_count=0,
            last_scheduled_at=long_ago,
        )

    # _execute_scheduled_workflow was invoked (we got past the due-check).
    mock_exec.assert_called_once()

    # The UPDATE on last_scheduled_at MUST have fired (this is the tight-loop fix).
    update_calls = [
        c
        for c in mock_db_exec.call_args_list
        if c.args
        and "UPDATE" in str(c.args[0]).upper()
        and "LAST_SCHEDULED_AT" in str(c.args[0]).upper()
    ]
    assert update_calls, (
        "last_scheduled_at UPDATE was never issued — every 30s tick will "
        "re-classify this workflow as due and re-abort (tight loop)."
    )
    # And commit was called at least once so the UPDATE actually lands.
    mock_commit.assert_called()


def test_scheduler_maybe_execute_writes_failed_execution_row_when_check_model_aborts(
    billing_env, no_platform_keys
):
    """When check_model_available aborts, the user must see a failed
    workflow_executions row in the UI with an actionable error message —
    not silence. The row uses the ai.workflow_executions schema
    (status='failed', error TEXT) defined in ai_schema.sql.
    """
    from datetime import UTC, datetime, timedelta

    from werkzeug.exceptions import BadRequest

    sched = {"type": "interval", "interval_seconds": 60, "enabled": True}
    long_ago = datetime.now(UTC) - timedelta(hours=1)
    wf_id = "11111111-1111-1111-1111-111111111111"

    with (
        patch.object(llm_availability, "list_byok_providers", return_value=frozenset()),
        patch.object(scheduler_task, "_execute_scheduled_workflow") as mock_exec,
        patch.object(scheduler_task.db.session, "execute") as mock_db_exec,
        patch.object(scheduler_task.db.session, "commit"),
    ):
        mock_exec.side_effect = BadRequest(
            "Model openrouter/meta-llama/llama-4-maverick requires BYOK. "
            "Add an API key for openrouter in Settings → LLM Provider Keys."
        )
        scheduler_task._maybe_execute_scheduled(
            wf_id=wf_id,
            sched=sched,
            run_count=0,
            last_scheduled_at=long_ago,
        )

    # An INSERT INTO workflow_executions with status='failed' must have fired.
    failed_insert_calls = [
        c
        for c in mock_db_exec.call_args_list
        if c.args
        and "INSERT" in str(c.args[0]).upper()
        and "WORKFLOW_EXECUTIONS" in str(c.args[0]).upper()
    ]
    assert failed_insert_calls, "Expected a workflow_executions INSERT for the failed run"

    # The inserted row carries the BYOK error message in the 'error' column
    # (matches schema: ai.workflow_executions.error TEXT). The status is 'failed'.
    insert_call = failed_insert_calls[0]
    params = insert_call.args[1] if len(insert_call.args) > 1 else insert_call.kwargs
    assert params.get("wid") == wf_id
    assert params.get("status") == "failed"
    error_text = params.get("error") or ""
    assert "BYOK" in error_text
    assert "openrouter" in error_text


def test_scheduler_maybe_execute_propagates_non_badrequest_exceptions(
    billing_env, no_platform_keys
):
    """The try/except in _maybe_execute_scheduled must be narrow:
    only BadRequest (from check_model_available) is swallowed. Unrelated
    runtime errors must still propagate so the outer scheduler-tick
    try/except can log them with full context.

    Counterfactual: a broad ``except Exception`` would catch everything and
    advance last_scheduled_at on any failure, hiding real bugs.
    """
    from datetime import UTC, datetime, timedelta

    sched = {"type": "interval", "interval_seconds": 60, "enabled": True}
    long_ago = datetime.now(UTC) - timedelta(hours=1)

    with (
        patch.object(scheduler_task, "_execute_scheduled_workflow") as mock_exec,
        patch.object(scheduler_task.db.session, "execute"),
        patch.object(scheduler_task.db.session, "commit"),
    ):
        mock_exec.side_effect = RuntimeError("unrelated downstream failure")

        # RuntimeError must propagate — only BadRequest is the swallowed case.
        with pytest.raises(RuntimeError, match="unrelated downstream failure"):
            scheduler_task._maybe_execute_scheduled(
                wf_id="22222222-2222-2222-2222-222222222222",
                sched=sched,
                run_count=0,
                last_scheduled_at=long_ago,
            )


def test_scheduler_advances_when_failed_row_insert_raises(billing_env, no_platform_keys, caplog):
    """N14: if the failed-row INSERT itself raises (DB transient blip, FK
    violation, schema drift), the UPDATE that advances last_scheduled_at
    MUST still fire. Otherwise the tight loop returns through a different
    path — round-2 N1 fix wrote the row, this round closes the second
    failure mode where the row write itself dies.

    Counterfactual: remove the inner try/except around the INSERT and the
    UPDATE is never reached (OperationalError unwinds past it).
    """
    from datetime import UTC, datetime, timedelta

    from sqlalchemy.exc import OperationalError
    from werkzeug.exceptions import BadRequest

    sched = {"type": "interval", "interval_seconds": 60, "enabled": True}
    long_ago = datetime.now(UTC) - timedelta(hours=1)
    wf_id = "33333333-3333-3333-3333-333333333333"

    # First execute call = the failed-row INSERT (must raise).
    # Second execute call = the last_scheduled_at UPDATE (must succeed).
    def execute_side_effect(*args, **kwargs):
        execute_side_effect.calls += 1
        sql_text = str(args[0]).upper() if args else ""
        if execute_side_effect.calls == 1:
            # The INSERT INTO workflow_executions — raise simulated DB blip.
            assert (
                "INSERT" in sql_text and "WORKFLOW_EXECUTIONS" in sql_text
            ), f"Expected first execute to be the failed-row INSERT, got: {sql_text[:120]}"
            raise OperationalError("INSERT", {}, Exception("simulated DB blip"))
        # Subsequent calls (the UPDATE) succeed.
        return MagicMock()

    execute_side_effect.calls = 0

    with (
        patch.object(llm_availability, "list_byok_providers", return_value=frozenset()),
        patch.object(scheduler_task, "_execute_scheduled_workflow") as mock_exec,
        patch.object(scheduler_task.db.session, "execute", side_effect=execute_side_effect),
        patch.object(scheduler_task.db.session, "commit"),
        patch.object(scheduler_task.db.session, "rollback") as mock_rollback,
        caplog.at_level("WARNING"),
    ):
        mock_exec.side_effect = BadRequest(
            "Model openrouter/meta-llama/llama-4-maverick requires BYOK"
        )

        # Must NOT raise — the inner try/except around the INSERT
        # swallows the OperationalError and falls through to the UPDATE.
        scheduler_task._maybe_execute_scheduled(
            wf_id=wf_id,
            sched=sched,
            run_count=0,
            last_scheduled_at=long_ago,
        )

    # Both calls fired: 1st = INSERT (raised), 2nd = UPDATE.
    assert execute_side_effect.calls == 2, (
        f"Expected exactly 2 execute calls (INSERT then UPDATE); got {execute_side_effect.calls}. "
        "If only 1, the OperationalError unwound past the UPDATE — tight loop."
    )
    # Rollback was called on the broken session so the UPDATE could proceed.
    mock_rollback.assert_called_once()
    # The warning log was emitted with the expected event name.
    assert any(
        "scheduled_failed_row_insert_failed" in rec.message for rec in caplog.records
    ), "Expected WARNING log 'scheduled_failed_row_insert_failed'"
