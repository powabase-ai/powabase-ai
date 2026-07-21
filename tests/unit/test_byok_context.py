"""Tests for current_byok_providers contextvar + list_byok_providers helper.

current_byok_providers lives in services/billing_cloud/identity.py (CUT — the
cloud-only BYOK contextvar). list_byok_providers lives in services/llm_availability.py
(moved there so it works independent of billing — see that module's docstring).

Fixture-adaptation note: the v1.5 plan body sketched these tests with
``test_db_session`` and ``sample_project`` fixtures. Neither exists in this
project (PS is single-project; there is no Project ORM model in tenant.py).
The unit conftest is deliberately no-DB (overrides parent ``db_cleanup`` and
mocks the Flask app). We mock ``db.session.query`` directly here — same
pattern used by ``tests/test_ai_provider_keys_resolver.py``.
"""

from unittest.mock import MagicMock, patch

from agentic_project_service.services.billing_cloud.identity import current_byok_providers
from agentic_project_service.services.llm_availability import list_byok_providers
from agentic_project_service.services import billing_port, llm_call
from tests.support.billing import RecordingBillingAdapter


def test_current_byok_providers_default_is_empty_frozenset():
    assert current_byok_providers.get() == frozenset()


def test_current_byok_providers_isolate_via_token():
    token = current_byok_providers.set(frozenset({"openai"}))
    assert current_byok_providers.get() == frozenset({"openai"})
    current_byok_providers.reset(token)
    assert current_byok_providers.get() == frozenset()


def test_list_byok_providers_returns_frozen_set():
    """list_byok_providers reads only the provider column; does not decrypt."""
    row_openai = MagicMock()
    row_openai.provider = "openai"
    row_anthropic = MagicMock()
    row_anthropic.provider = "anthropic"

    with patch("agentic_project_service.services.llm_availability.db") as mock_db:
        mock_db.session.query.return_value.filter_by.return_value.all.return_value = [
            row_openai,
            row_anthropic,
        ]
        result = list_byok_providers(project_id="proj-1")

    assert isinstance(result, frozenset)
    assert result == frozenset({"openai", "anthropic"})


def test_list_byok_providers_skips_invalid_rows():
    """Query filters on is_valid=True so invalid rows never reach the result."""
    row_openai = MagicMock()
    row_openai.provider = "openai"

    with patch("agentic_project_service.services.llm_availability.db") as mock_db:
        # Only the valid row comes back — filter_by(is_valid=True) is the gate.
        mock_db.session.query.return_value.filter_by.return_value.all.return_value = [
            row_openai,
        ]
        result = list_byok_providers(project_id="proj-1")

        # Assert the filter is keyed on is_valid=True (so invalid rows are excluded).
        mock_db.session.query.return_value.filter_by.assert_called_once_with(is_valid=True)

    assert result == frozenset({"openai"})


def test_list_byok_providers_does_not_decrypt_or_delete():
    """Regression: read-only — no decrypt call, no self-heal DELETE."""
    row_openai = MagicMock()
    row_openai.provider = "openai"

    with patch("agentic_project_service.services.llm_availability.db") as mock_db:
        mock_db.session.query.return_value.filter_by.return_value.all.return_value = [
            row_openai,
        ]
        with patch("agentic_project_service.services.encryption.decrypt_api_key") as mock_decrypt:
            result = list_byok_providers(project_id="proj-1")

        # Decrypt must not be called — list_byok_providers reads only the provider column.
        mock_decrypt.assert_not_called()
        # No DELETE side effect: the self-heal path lives in
        # ai_provider_keys_resolver, not here. Confirm we never called .delete().
        mock_db.session.query.return_value.filter_by.return_value.delete.assert_not_called()
        mock_db.session.begin_nested.assert_not_called()

    assert "openai" in result


# ---------------------------------------------------------------------------
# with_llm_key (llm_call.py) — Task 5: split of with_byok_and_recoup
# ---------------------------------------------------------------------------


def test_with_llm_key_resolves_byok_and_enters_port_scope():
    rec = RecordingBillingAdapter()
    billing_port.set_billing_adapter(rec)
    with (
        patch.object(llm_call, "get_all_user_provider_keys", return_value={"openai": "sk-x"}),
        patch.object(llm_call, "resolve_api_key_for_model", return_value="sk-x") as res,
    ):
        with llm_call.with_llm_key("openai/gpt-5") as key:
            assert key == "sk-x"
    res.assert_called_once()
    assert rec.llm_scopes == 1  # port llm_call_scope entered (recoup armed in cloud)


def test_with_llm_key_yields_none_when_no_byok():
    billing_port.set_billing_adapter(RecordingBillingAdapter())
    with (
        patch.object(llm_call, "get_all_user_provider_keys", return_value={}),
        patch.object(llm_call, "resolve_api_key_for_model", return_value=None),
    ):
        with llm_call.with_llm_key("openai/gpt-5") as key:
            assert key is None


def test_with_byok_and_recoup_is_gone():
    assert not hasattr(llm_call, "with_byok_and_recoup")
