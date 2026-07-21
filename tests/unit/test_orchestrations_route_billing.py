"""Tests for billing wiring in routes/orchestrations.py.

Tests the pre-op balance check on /run/stream via a RecordingBillingAdapter
(tests/support/billing.py) registered on the billing port
(services/billing_port.py). The orchestration runner is heavyweight (DB,
sessions, supervisor threads, delegated agent persistence), so tests verify
the pre-dispatch contract rather than driving a full run.

"orchestration_run charge fires with the right action/idempotency after a
successful run" mirrors the run_agent charge pattern pinned behaviorally in
test_finish_run_background.py; the route-level call itself is a single
``billing.charge(...)`` guarded only by run status (see routes/agents.py's
charge sites for the same shape).
"""

from unittest.mock import patch

from agentic_project_service.routes import orchestrations as orch_route
from agentic_project_service.services import billing_port
from tests.support.billing import RecordingBillingAdapter


def _make_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(orch_route.orchestrations_bp)
    return app


def _auth_headers():
    return {"Authorization": "Bearer fake"}


# ---------------------------------------------------------------------------
# Module wiring + constants
# ---------------------------------------------------------------------------


def test_orchestration_run_estimated_cost_constant():
    """Same heuristic as agent/workflow runs — base + ~50 internal ops."""
    assert orch_route._ORCHESTRATION_RUN_ESTIMATED_COST == 20_000


# ---------------------------------------------------------------------------
# run_orchestration_stream — pre-stream balance check
# ---------------------------------------------------------------------------


def test_run_orchestration_stream_balance_check_fires_pre_stream():
    """Balance check is performed BEFORE entering the SSE generator so 402
    propagates as a regular HTTP error response, not via a stream."""
    app = _make_test_app()
    rec = RecordingBillingAdapter(raise_402=True)
    billing_port.set_billing_adapter(rec)

    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/orchestrations/orch-1/run/stream",
                json={"message": "hi"},
                headers=_auth_headers(),
            )

    assert resp.status_code == 402
    assert rec.balance_checks == [20_000]
    # 402 fired before any orchestration ran → no charge.
    assert rec.charges == []


def test_run_orchestration_stream_propagates_503():
    """503 from balance check propagates to caller."""
    app = _make_test_app()
    rec = RecordingBillingAdapter(raise_503=True)
    billing_port.set_billing_adapter(rec)

    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/orchestrations/orch-1/run/stream",
                json={"message": "hi"},
                headers=_auth_headers(),
            )

    assert resp.status_code == 503
    assert rec.charges == []


def test_run_orchestration_stream_short_circuits_before_billing_when_no_message(
    recording_billing,
):
    """Empty body returns 400 before billing fires."""
    app = _make_test_app()
    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/orchestrations/orch-1/run/stream",
                json={},
                headers=_auth_headers(),
            )

    assert resp.status_code == 400
    assert recording_billing.balance_checks == []
    assert recording_billing.charges == []
