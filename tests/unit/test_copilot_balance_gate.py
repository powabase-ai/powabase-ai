"""The copilot credit gate short-circuits on BYOK and otherwise defers to billing_port.

Two directions matter and they fail differently: failing CLOSED is loud (a user
sees a 402 they shouldn't), failing OPEN is silent and costs money. Both are pinned.

RecordingBillingAdapter (tests/support/billing.py:18) records every check_balance
estimated_cost in `.balance_checks`, so these assert on what the port was
actually asked, not on a mock's call args.
"""

from unittest.mock import patch

import pytest
from werkzeug.exceptions import HTTPException

from agentic_project_service.routes import copilot as copilot_routes
from agentic_project_service.services import billing_port
from tests.support.billing import RecordingBillingAdapter


def test_byok_project_skips_the_balance_check(recording_billing):
    """A BYOK turn bills the user's own provider, so credits are irrelevant."""
    with patch.object(copilot_routes, "project_has_byok_for_model", return_value=True):
        copilot_routes._gate_copilot_turn("claude-opus-4-8")
    assert recording_billing.balance_checks == []


def test_ai_on_us_project_defers_to_the_billing_port(recording_billing):
    """No BYOK key -> the port decides, with the deliberate 1-credit estimate."""
    with patch.object(copilot_routes, "project_has_byok_for_model", return_value=False):
        copilot_routes._gate_copilot_turn("claude-opus-4-8")
    assert recording_billing.balance_checks == [
        copilot_routes._COPILOT_TURN_ESTIMATED_CREDITS
    ]


def test_the_port_error_is_not_swallowed():
    """A 402 from the adapter must reach the caller, not be caught and logged.

    Sets the adapter directly rather than via `recording_billing`, because 402
    is a constructor flag. The autouse `_billing_adapter_isolation` fixture
    restores the previous adapter afterwards.
    """
    billing_port.set_billing_adapter(RecordingBillingAdapter(raise_402=True))
    with patch.object(copilot_routes, "project_has_byok_for_model", return_value=False):
        with pytest.raises(HTTPException) as exc:
            copilot_routes._gate_copilot_turn("claude-opus-4-8")
    assert exc.value.code == 402
