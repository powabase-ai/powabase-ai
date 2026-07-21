"""Tests for the run approval endpoint and run registry."""

from agentic.execution.context import ExecutionContext

from agentic_project_service.services.run_registry import (
    get_active_run_context,
    register_run,
    unregister_run,
)


# ---------------------------------------------------------------------------
# Unit tests — registry functions directly
# ---------------------------------------------------------------------------


def test_registry_flow():
    ctx = ExecutionContext(execution_id="test")
    register_run("run_123", ctx)
    assert get_active_run_context("run_123") is ctx
    unregister_run("run_123")
    assert get_active_run_context("run_123") is None


def test_registry_get_missing_returns_none():
    assert get_active_run_context("nonexistent_run") is None


def test_registry_unregister_missing_is_safe():
    # Should not raise
    unregister_run("run_does_not_exist")


def test_registry_overwrite():
    ctx1 = ExecutionContext(execution_id="a")
    ctx2 = ExecutionContext(execution_id="b")
    register_run("run_x", ctx1)
    register_run("run_x", ctx2)
    assert get_active_run_context("run_x") is ctx2
    unregister_run("run_x")


# ---------------------------------------------------------------------------
# Integration tests — approve endpoint
# ---------------------------------------------------------------------------


class TestApproveEndpoint:
    def test_approve_unknown_run_returns_404(self, client, mock_auth, auth_headers):
        """Calling approve with a run_id that has no active context → 404."""
        resp = client.post(
            "/api/agents/runs/nonexistent_run_id/approve",
            json={"approved": True},
            headers=auth_headers,
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert "error" in data

    def test_approve_requires_auth(self, client):
        """Approve endpoint rejects unauthenticated requests."""
        resp = client.post(
            "/api/agents/runs/some_run_id/approve",
            json={"approved": True},
        )
        assert resp.status_code == 401

    def test_approve_active_run_resumes(self, client, mock_auth, auth_headers):
        """Approve endpoint calls set_approval_decision on the context and returns 200."""
        ctx = ExecutionContext(execution_id="run_test_approve")
        register_run("run_test_approve", ctx)

        try:
            resp = client.post(
                "/api/agents/runs/run_test_approve/approve",
                json={"approved": True, "tool_call_id": "tc_abc"},
                headers=auth_headers,
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["status"] == "resumed"
        finally:
            unregister_run("run_test_approve")
