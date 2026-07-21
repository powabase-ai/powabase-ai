"""Tests for billing wiring in routes/agents.py.

The run_agent / run_agent_stream endpoints are heavyweight (DB, session,
Agent.run, etc.), so these tests patch auth and drive billing through a
RecordingBillingAdapter (tests/support/billing.py) registered on the billing
port (services/billing_port.py), rather than driving a full end-to-end run.
The goal is to verify:

  * billing.check_balance() fires with the route's estimated_cost BEFORE any
    expensive work
  * a 402/503 raised by the adapter's check_balance() propagates as the
    corresponding HTTP error, and no charge is posted
  * routes that short-circuit before dispatch (validation failures) never
    touch the billing port at all

"agent_run charge fires with the right action/ref/idempotency after a
successful run" is covered by test_finish_run_background.py, which can
drive a completed run to its billing charge without the full route's
DB/Agent machinery.
"""

from unittest.mock import patch

from agentic_project_service.routes import agents as agents_route
from agentic_project_service.services import billing_port
from tests.support.billing import RecordingBillingAdapter


def test_agent_run_estimated_cost_constant():
    """Pre-op estimate is the spec heuristic (base + ~50 internal ops)."""
    assert agents_route._AGENT_RUN_ESTIMATED_COST == 1_000


def _make_test_app():
    """Create a minimal Flask app with the agents blueprint for test_client use."""
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(agents_route.agents_bp)
    return app


def _auth_headers():
    return {"Authorization": "Bearer fake"}


# ---------------------------------------------------------------------------
# run_agent (non-streaming) — wired billing path
# ---------------------------------------------------------------------------


def test_run_agent_balance_check_fires_before_run():
    """billing.check_balance() fires with the route's estimated_cost early,
    before any DB/agent work. A 402 from the adapter aborts the request with
    no charge posted."""
    app = _make_test_app()
    rec = RecordingBillingAdapter(raise_402=True)
    billing_port.set_billing_adapter(rec)

    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run",
                json={"message": "hi"},
                headers=_auth_headers(),
            )

    assert resp.status_code == 402
    assert rec.balance_checks == [1_000]
    # Run never started → no charge posted.
    assert rec.charges == []


def test_run_agent_propagates_503_from_balance_check():
    """When billing is unreachable, the route surfaces 503 to the caller."""
    app = _make_test_app()
    rec = RecordingBillingAdapter(raise_503=True)
    billing_port.set_billing_adapter(rec)

    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run",
                json={"message": "hi"},
                headers=_auth_headers(),
            )

    assert resp.status_code == 503
    assert rec.charges == []


def test_run_agent_short_circuits_before_billing_when_no_message(recording_billing):
    """The `message is required` 400 returns before billing fires."""
    app = _make_test_app()
    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run",
                json={},  # missing message
                headers=_auth_headers(),
            )

    assert resp.status_code == 400
    assert recording_billing.balance_checks == []
    assert recording_billing.charges == []


# ---------------------------------------------------------------------------
# run_agent_stream — wired billing path (pre-stream check)
# ---------------------------------------------------------------------------


def test_run_agent_stream_balance_check_fires_pre_stream():
    """Streaming endpoint: balance check is performed BEFORE entering the SSE
    generator so 402 propagates as a normal HTTP error (not as a stream)."""
    app = _make_test_app()
    rec = RecordingBillingAdapter(raise_402=True)
    billing_port.set_billing_adapter(rec)

    with patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/agents/agent-1/run/stream",
                json={"message": "hi"},
                headers=_auth_headers(),
            )

    assert resp.status_code == 402
    assert rec.balance_checks == [1_000]
    assert rec.charges == []
