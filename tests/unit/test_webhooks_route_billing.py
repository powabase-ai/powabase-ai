"""Tests for billing wiring in routes/webhooks.py.

The webhook endpoint runs workflows directly (no delegation to the
workflows.py /execute route), so it needs its own balance-check and
billing.charge (services/billing_port.py) wiring, verified via a
RecordingBillingAdapter (tests/support/billing.py). Tests patch helpers +
heavy collaborators rather than exercising a real DB-backed workflow.

Note: webhook tests must reach past the auth+state gates before billing
fires. The implementation places the balance check AFTER token validation
+ state check, so we patch DB access to return a deployed webhook with a
matching secret.
"""

from unittest.mock import MagicMock, patch

from agentic_project_service.routes import webhooks as webhooks_route


def _make_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(webhooks_route.webhooks_bp)
    return app


def _valid_webhook_id():
    """A real UUID — webhooks route 400s on non-UUID webhook ids."""
    return "11111111-1111-1111-1111-111111111111"


def _valid_workflow_uuid():
    return "22222222-2222-2222-2222-222222222222"


# ---------------------------------------------------------------------------
# trigger_webhook — balance check fires after auth/state, before workflow build
# ---------------------------------------------------------------------------
#
# Note: the test that drove _workflow_pre_check's 402 path via the private
# services.billing_cloud.balance_cache.PaymentRequired exception has been
# removed (that module is excluded from this OSS build). The remaining
# tests below cover the auth/state gating and the success/failure charge
# paths through the OSS-shipped billing_port facade.


def test_trigger_webhook_unauthorized_skips_billing(recording_billing):
    """Bad webhook secret returns 401 BEFORE the balance check runs.

    This is the security property: an attacker spamming a public webhook
    URL with a bad secret can't burn the project's free-tier credits.
    """
    app = _make_test_app()

    def session_side(*args, **kwargs):
        r = MagicMock()
        r.fetchone.return_value = (
            "block-1",
            _valid_workflow_uuid(),
            {"webhook_secret": "correct-secret"},
        )
        return r

    mock_session = MagicMock()
    mock_session.execute.side_effect = session_side

    with (
        patch("agentic_project_service.routes.webhooks.db.session", mock_session),
        patch.object(webhooks_route, "_workflow_pre_check") as mock_check,
    ):
        with app.test_client() as client:
            resp = client.post(
                f"/api/webhooks/{_valid_webhook_id()}",
                json={"payload": "x"},
                headers={"Authorization": "Bearer wrong-secret"},
            )

    assert resp.status_code == 401
    mock_check.assert_not_called()
    assert recording_billing.charges == []


def test_trigger_webhook_invalid_uuid_skips_billing(recording_billing):
    """Malformed webhook id 400s before billing fires."""
    app = _make_test_app()
    with patch.object(webhooks_route, "_workflow_pre_check") as mock_check:
        with app.test_client() as client:
            resp = client.post(
                "/api/webhooks/not-a-uuid",
                json={},
                headers={"Authorization": "Bearer s"},
            )

    assert resp.status_code == 400
    mock_check.assert_not_called()
    assert recording_billing.charges == []


# ---------------------------------------------------------------------------
# Success path: billing.charge fires on successful webhook-triggered run
# ---------------------------------------------------------------------------


def _mock_async_arun_detailed(block_outputs=None):
    block_outputs = block_outputs or {"b1": {"output": "result"}}

    async def _run(**_):
        return block_outputs, []

    wf = MagicMock()
    wf.arun_detailed = _run
    return wf


def test_trigger_webhook_posts_charge_on_success(recording_billing):
    """A successful webhook-triggered run posts workflow_run + per-block charges."""
    app = _make_test_app()
    block_outputs = {"b1": {"output": "ok"}}
    blocks_data = [{"id": "b1", "type": "code"}]

    def session_side(*args, **kwargs):
        if not hasattr(session_side, "calls"):
            session_side.calls = 0
        session_side.calls += 1
        if session_side.calls == 1:
            # webhook block lookup
            r = MagicMock()
            r.fetchone.return_value = (
                "block-1",
                _valid_workflow_uuid(),
                {"webhook_secret": "sekret"},
            )
            return r
        elif session_side.calls == 2:
            # workflow state lookup
            r = MagicMock()
            r.fetchone.return_value = ("deployed", None)
            return r
        else:
            # all subsequent execute() calls: INSERT/UPDATE — return empty
            r = MagicMock()
            r.fetchone.return_value = None
            return r

    mock_session = MagicMock()
    mock_session.execute.side_effect = session_side

    with (
        patch("agentic_project_service.routes.webhooks.db.session", mock_session),
        patch.object(webhooks_route, "_workflow_pre_check"),
        patch.object(webhooks_route, "charge_workflow_blocks") as mock_block_charge,
        patch.object(
            webhooks_route,
            "build_workflow_from_db",
            return_value=_mock_async_arun_detailed(block_outputs),
        ),
        patch.object(webhooks_route, "load_blocks", return_value=blocks_data),
        patch.object(webhooks_route, "load_edges", return_value=[]),
        patch.object(webhooks_route, "make_agent_run_recorder", return_value=(lambda _: None, {})),
        patch.object(webhooks_route, "make_services", return_value={}),
        patch.object(webhooks_route, "build_block_logs", return_value=[]),
        patch.object(webhooks_route, "persist_block_logs", return_value=True),
    ):
        with app.test_client() as client:
            resp = client.post(
                f"/api/webhooks/{_valid_webhook_id()}",
                json={"payload": "x"},
                headers={"Authorization": "Bearer sekret"},
            )

    assert resp.status_code == 200
    assert len(recording_billing.charges) == 1
    charge = recording_billing.charges[0]
    assert charge["action"] == "workflow_run"
    assert charge["quantity"] == 1
    assert charge["ref_type"] == "workflow_run"
    assert charge["metadata"].get("trigger") == "webhook"
    body = resp.get_json()
    assert charge["ref_id"] == body["execution_id"]
    assert charge["idempotency_parts"] == (body["execution_id"],)
    # charge_workflow_blocks fired with the same block outputs
    mock_block_charge.assert_called_once()
    block_kwargs = mock_block_charge.call_args.kwargs
    assert block_kwargs["block_outputs"] == block_outputs


def test_trigger_webhook_skips_charge_on_failure(recording_billing):
    """A webhook run that raises inside arun_detailed does NOT post a charge."""
    app = _make_test_app()

    async def _raise(**_):
        raise RuntimeError("boom")

    wf = MagicMock()
    wf.arun_detailed = _raise

    def session_side(*args, **kwargs):
        if not hasattr(session_side, "calls"):
            session_side.calls = 0
        session_side.calls += 1
        if session_side.calls == 1:
            r = MagicMock()
            r.fetchone.return_value = (
                "block-1",
                _valid_workflow_uuid(),
                {"webhook_secret": "sekret"},
            )
            return r
        elif session_side.calls == 2:
            r = MagicMock()
            r.fetchone.return_value = ("deployed", None)
            return r
        else:
            r = MagicMock()
            r.fetchone.return_value = None
            return r

    mock_session = MagicMock()
    mock_session.execute.side_effect = session_side

    with (
        patch("agentic_project_service.routes.webhooks.db.session", mock_session),
        patch.object(webhooks_route, "_workflow_pre_check"),
        patch.object(webhooks_route, "charge_workflow_blocks") as mock_block_charge,
        patch.object(webhooks_route, "build_workflow_from_db", return_value=wf),
        patch.object(webhooks_route, "load_blocks", return_value=[]),
        patch.object(webhooks_route, "load_edges", return_value=[]),
        patch.object(webhooks_route, "make_agent_run_recorder", return_value=(lambda _: None, {})),
        patch.object(webhooks_route, "make_services", return_value={}),
    ):
        with app.test_client() as client:
            resp = client.post(
                f"/api/webhooks/{_valid_webhook_id()}",
                json={"payload": "x"},
                headers={"Authorization": "Bearer sekret"},
            )

    assert resp.status_code == 500
    assert recording_billing.charges == []
    mock_block_charge.assert_not_called()
