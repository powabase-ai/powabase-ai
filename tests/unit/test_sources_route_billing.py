"""Tests for billing wiring in routes/sources.py.

Covers the balance-check-before-state-mutation contract for routes that
queue Celery tasks. The reextract route in particular must NOT mutate the
source's extraction_status to 'pending' before the balance check passes —
otherwise a 402/503 from the balance check leaves the source stuck in
'pending' with no Celery task to clear it.

Reference pattern: upload_source, import_url, import_from_storage all
balance-check BEFORE any state mutation.

Balance checks are routed through the billing port (services/billing_port.py)
via a RecordingBillingAdapter (tests/support/billing.py) — see Task 9 of the
OSS-edition migration.
"""

from unittest.mock import MagicMock, patch

import pytest

from agentic_project_service.routes import sources as sources_route
from agentic_project_service.services import billing_port
from tests.support.billing import RecordingBillingAdapter


@pytest.fixture
def billing_env(monkeypatch):
    """Set env vars so get_billing_context() returns a non-None context."""
    monkeypatch.setenv("BILLING_ORG_ID", "org-1")
    monkeypatch.setenv("PROJECT_ID", "proj-1")
    monkeypatch.delenv("BILLING_PLAN_TIER", raising=False)
    yield


@pytest.fixture
def no_billing_env(monkeypatch):
    monkeypatch.delenv("BILLING_ORG_ID", raising=False)
    monkeypatch.delenv("PROJECT_ID", raising=False)
    monkeypatch.delenv("BILLING_PROJECT_ID", raising=False)
    yield


def _make_test_app():
    """Create a minimal Flask app with the sources blueprint for test_client use."""
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(sources_route.sources_bp)
    return app


def _auth_headers():
    return {"Authorization": "Bearer fake"}


# ---------------------------------------------------------------------------
# reextract — balance check MUST fire before any state mutation.
# ---------------------------------------------------------------------------


def test_reextract_balance_check_blocks_status_mutation(billing_env):
    """When billing.check_balance raises 402, the source's extraction_status
    must NOT be flipped to 'pending', and the Celery task must NOT dispatch.

    Bug it prevents: pre-fix flow was UPDATE→commit→balance_check→dispatch,
    so a 402 left the row stuck in 'pending' forever."""
    app = _make_test_app()
    rec = RecordingBillingAdapter(raise_402=True)
    billing_port.set_billing_adapter(rec)

    # Track every db.session.execute call. We assert no UPDATE ... SET
    # extraction_status = 'pending' was sent.
    update_calls: list[str] = []

    def _record_execute(stmt, params=None):
        sql = str(stmt)
        update_calls.append(sql)
        # Return a result that satisfies the SELECT fetchone() path.
        result = MagicMock()
        # Mirror the fixed route's SELECT id, extraction_status, auto_metadata.
        result.fetchone.return_value = ("src-1", "extracted", {})
        return result

    fake_session = MagicMock()
    fake_session.execute.side_effect = _record_execute
    fake_session.commit = MagicMock()

    # Capture Celery dispatch attempts.
    with (
        patch.object(sources_route.db, "session", fake_session),
        patch.object(sources_route.extract_source, "delay") as mock_extract,
        patch.object(sources_route.extract_url_source, "delay") as mock_extract_url,
        patch.object(sources_route, "get_all_user_provider_keys", return_value={}),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/sources/src-1/reextract",
                json={},
                headers=_auth_headers(),
            )

    # 402 propagates.
    assert resp.status_code == 402, f"expected 402, got {resp.status_code}: {resp.data}"
    # Balance check fired, with the expected ocr_pages estimate.
    assert rec.balance_checks == [
        sources_route._EXTRACTION_ESTIMATED_PAGES * sources_route._OCR_UNIT_CREDITS
    ]
    # Celery task never dispatched.
    mock_extract.assert_not_called()
    mock_extract_url.assert_not_called()

    # CRITICAL: no UPDATE SET extraction_status = 'pending' was issued.
    pending_updates = [
        sql
        for sql in update_calls
        if "UPDATE" in sql.upper() and "extraction_status = 'pending'" in sql.lower()
    ]
    assert pending_updates == [], (
        f"Bug recurrence: an UPDATE to extraction_status='pending' was issued "
        f"BEFORE the balance check passed. Source would be stuck in 'pending'. "
        f"Offending statements: {pending_updates}"
    )


def test_reextract_balance_check_propagates_503(billing_env):
    """When billing is unreachable, the route returns 503 and does NOT
    mutate state or dispatch the Celery task."""
    app = _make_test_app()
    rec = RecordingBillingAdapter(raise_503=True)
    billing_port.set_billing_adapter(rec)
    update_calls: list[str] = []

    def _record_execute(stmt, params=None):
        update_calls.append(str(stmt))
        result = MagicMock()
        result.fetchone.return_value = ("src-1", "extracted", {})
        return result

    fake_session = MagicMock()
    fake_session.execute.side_effect = _record_execute

    with (
        patch.object(sources_route.db, "session", fake_session),
        patch.object(sources_route.extract_source, "delay") as mock_extract,
        patch.object(sources_route, "get_all_user_provider_keys", return_value={}),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/sources/src-1/reextract",
                json={},
                headers=_auth_headers(),
            )

    assert resp.status_code == 503
    assert rec.balance_checks == [
        sources_route._EXTRACTION_ESTIMATED_PAGES * sources_route._OCR_UNIT_CREDITS
    ]
    mock_extract.assert_not_called()
    # No 'pending' status mutation.
    pending_updates = [
        sql
        for sql in update_calls
        if "UPDATE" in sql.upper() and "extraction_status = 'pending'" in sql.lower()
    ]
    assert pending_updates == [], "503 from balance check must not leave the source in 'pending'."


def test_reextract_dispatches_task_when_balance_check_passes(billing_env):
    """Happy path: when billing.check_balance returns (no raise), the
    route DOES mutate status to 'pending' and DOES dispatch the Celery task."""
    app = _make_test_app()
    rec = RecordingBillingAdapter()
    billing_port.set_billing_adapter(rec)
    update_calls: list[str] = []

    def _record_execute(stmt, params=None):
        update_calls.append(str(stmt))
        result = MagicMock()
        result.fetchone.return_value = ("src-1", "extracted", {})
        return result

    fake_session = MagicMock()
    fake_session.execute.side_effect = _record_execute

    fake_task = MagicMock()
    fake_task.id = "task-xyz"

    with (
        patch.object(sources_route.db, "session", fake_session),
        patch.object(sources_route.extract_source, "delay", return_value=fake_task) as mock_extract,
        patch.object(sources_route, "get_all_user_provider_keys", return_value={}),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/sources/src-1/reextract",
                json={},
                headers=_auth_headers(),
            )

    assert resp.status_code == 200
    # Balance check fired before dispatch.
    assert rec.balance_checks == [
        sources_route._EXTRACTION_ESTIMATED_PAGES * sources_route._OCR_UNIT_CREDITS
    ]
    mock_extract.assert_called_once()
    # State mutation DID occur — confirms the route still completes the
    # workflow after the balance check passes.
    pending_updates = [
        sql
        for sql in update_calls
        if "UPDATE" in sql.upper() and "extraction_status = 'pending'" in sql.lower()
    ]
    assert (
        pending_updates
    ), "After balance check passed, the route should set extraction_status='pending'"


def test_reextract_checks_balance_unconditionally(no_billing_env):
    """After the port migration the reextract route no longer gates the balance
    check on ``get_billing_context()`` — it computes the estimate and calls
    ``billing.check_balance`` UNCONDITIONALLY, delegating the skip-when-
    unconfigured to the adapter (no-op adapter in OSS; cloud adapter no-ops when
    the env is unset — pinned by
    tests/unit/test_billing_cloud_adapter.py::test_check_balance_is_noop_when_unconfigured).
    The RecordingBillingAdapter records every call regardless of env, so even
    with no billing env the check fires and the task still dispatches."""
    app = _make_test_app()
    rec = RecordingBillingAdapter()
    billing_port.set_billing_adapter(rec)

    def _record_execute(stmt, params=None):
        result = MagicMock()
        result.fetchone.return_value = ("src-1", "extracted", {})
        return result

    fake_session = MagicMock()
    fake_session.execute.side_effect = _record_execute

    fake_task = MagicMock()
    fake_task.id = "task-xyz"

    with (
        patch.object(sources_route.db, "session", fake_session),
        patch.object(sources_route.extract_source, "delay", return_value=fake_task) as mock_extract,
        patch.object(sources_route, "get_all_user_provider_keys", return_value={}),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/sources/src-1/reextract",
                json={},
                headers=_auth_headers(),
            )

    assert resp.status_code == 200
    assert rec.balance_checks == [
        sources_route._EXTRACTION_ESTIMATED_PAGES * sources_route._OCR_UNIT_CREDITS
    ]
    mock_extract.assert_called_once()


def test_reextract_threads_per_call_seed_to_task(billing_env):
    """The reextract route generates a fresh per-call ``reextract_seed`` and
    passes it to the Celery task (both the PDF and URL dispatch paths), so the
    task can build a 4-part per-call idempotency key — distinct per reextract,
    preserving the pre-port 'one ledger row per reextract' behavior."""
    app = _make_test_app()
    billing_port.set_billing_adapter(RecordingBillingAdapter())

    def _record_execute(stmt, params=None):
        result = MagicMock()
        # PDF source (no origin_url) → extract_source path.
        result.fetchone.return_value = ("src-1", "extracted", {})
        return result

    fake_session = MagicMock()
    fake_session.execute.side_effect = _record_execute

    fake_task = MagicMock()
    fake_task.id = "task-xyz"

    with (
        patch.object(sources_route.db, "session", fake_session),
        patch.object(sources_route.extract_source, "delay", return_value=fake_task) as mock_extract,
        patch.object(sources_route, "get_all_user_provider_keys", return_value={}),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ),
    ):
        with app.test_client() as client:
            resp = client.post(
                "/api/sources/src-1/reextract",
                json={},
                headers=_auth_headers(),
            )

    assert resp.status_code == 200
    mock_extract.assert_called_once()
    seed = mock_extract.call_args.kwargs.get("reextract_seed")
    assert seed, "reextract must pass a non-empty per-call reextract_seed to the task"
    # billing_kwargs stash is gone — the route no longer passes an idempotency key.
    assert "billing_idempotency_key" not in mock_extract.call_args.kwargs
