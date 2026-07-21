"""Tests for build_block_logs helper — constructing log entries from execution data."""


class TestBuildBlockLogs:
    """build_block_logs should produce correct log entries from block outputs."""

    def test_basic_block_log_entry(self):
        from agentic_project_service.routes._workflow_helpers import build_block_logs

        blocks_data = [
            {"id": "b1", "type": "agent", "name": "My Agent", "config": {"model": "gpt-4"}},
        ]
        output = {"b1": {"output": "hello"}}
        duration_map = {"b1": 150.0}
        edges_data = []

        logs = build_block_logs(output, blocks_data, edges_data, duration_map)

        assert len(logs) == 1
        assert logs[0]["block_id"] == "b1"
        assert logs[0]["block_type"] == "agent"
        assert logs[0]["block_name"] == "My Agent"
        assert logs[0]["status"] == "success"
        assert logs[0]["duration_ms"] == 150.0
        assert logs[0]["output"] == {"output": "hello"}
        assert logs[0]["input"] is None
        assert logs[0]["config"] == {"model": "gpt-4"}

    def test_block_with_error_status(self):
        from agentic_project_service.routes._workflow_helpers import build_block_logs

        blocks_data = [
            {"id": "b1", "type": "agent", "name": "Broken", "config": {}},
        ]
        output = {"b1": {"error": "timeout"}}

        logs = build_block_logs(output, blocks_data, [], {})

        assert logs[0]["status"] == "error"

    def test_upstream_input_included(self):
        from agentic_project_service.routes._workflow_helpers import build_block_logs

        blocks_data = [
            {"id": "b1", "type": "starter", "name": "Start", "config": {}},
            {"id": "b2", "type": "agent", "name": "Agent", "config": {}},
        ]
        output = {
            "b1": {"output": "input data"},
            "b2": {"output": "result"},
        }
        edges_data = [
            {"source": "b1", "target": "b2", "sourceHandle": "output", "targetHandle": "input"},
        ]

        logs = build_block_logs(output, blocks_data, edges_data, {})

        b2_log = next(l for l in logs if l["block_id"] == "b2")
        assert b2_log["input"] == {"b1": {"output": "input data"}}

    def test_missing_block_in_blocks_data(self):
        """If output contains a block ID not in blocks_data, use block_id as name."""
        from agentic_project_service.routes._workflow_helpers import build_block_logs

        blocks_data = []
        output = {"unknown_block": {"output": "data"}}

        logs = build_block_logs(output, blocks_data, [], {})

        assert len(logs) == 1
        assert logs[0]["block_name"] == "unknown_block"
        assert logs[0]["block_type"] == ""

    def test_empty_output(self):
        from agentic_project_service.routes._workflow_helpers import build_block_logs

        logs = build_block_logs({}, [], [], {})
        assert logs == []
