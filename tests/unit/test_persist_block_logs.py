"""Tests for persist_block_logs — error propagation and return value."""

import json
from unittest.mock import MagicMock, patch


class TestPersistBlockLogs:
    """persist_block_logs must surface failures to callers."""

    def test_returns_true_on_success(self):
        """Caller should know when logs were persisted successfully."""
        from agentic_project_service.routes._workflow_helpers import (
            persist_block_logs,
        )

        with patch("agentic_project_service.routes._workflow_helpers.db") as mock_db:
            mock_db.session = MagicMock()
            result = persist_block_logs(
                "exec-1",
                [
                    {
                        "block_id": "b1",
                        "block_type": "agent",
                        "block_name": "My Agent",
                        "status": "success",
                        "duration_ms": 100.0,
                        "config": {},
                        "output": {"output": "hello"},
                        "input": None,
                    },
                ],
            )
            assert result is True

    def test_returns_false_on_db_error(self):
        """When DB insert fails, caller must know logs were NOT persisted."""
        from agentic_project_service.routes._workflow_helpers import (
            persist_block_logs,
        )

        with patch("agentic_project_service.routes._workflow_helpers.db") as mock_db:
            mock_db.session.execute.side_effect = RuntimeError("connection lost")
            result = persist_block_logs(
                "exec-1",
                [
                    {
                        "block_id": "b1",
                        "block_type": "agent",
                        "block_name": "My Agent",
                        "status": "success",
                        "duration_ms": 100.0,
                        "config": {},
                        "output": {"output": "hello"},
                        "input": None,
                    },
                ],
            )
            assert result is False

    def test_returns_true_for_empty_list(self):
        """Empty block logs list should succeed (nothing to insert)."""
        from agentic_project_service.routes._workflow_helpers import (
            persist_block_logs,
        )

        with patch("agentic_project_service.routes._workflow_helpers.db") as mock_db:
            mock_db.session = MagicMock()
            result = persist_block_logs("exec-1", [])
            assert result is True

    def test_circular_input_does_not_crash(self):
        """When a block log's input contains a circular reference,
        persist_block_logs must still succeed (safe_json handles it)."""
        from agentic_project_service.routes._workflow_helpers import (
            persist_block_logs,
        )

        # Build a circular reference in the input dict
        circular = {"key": "value"}
        circular["self"] = circular  # circular reference

        with patch("agentic_project_service.routes._workflow_helpers.db") as mock_db:
            mock_db.session = MagicMock()
            result = persist_block_logs(
                "exec-1",
                [
                    {
                        "block_id": "b1",
                        "block_type": "code",
                        "block_name": "Function",
                        "status": "success",
                        "duration_ms": 50.0,
                        "config": {},
                        "output": {"output": "ok"},
                        "input": circular,
                    },
                ],
            )
            assert result is True
