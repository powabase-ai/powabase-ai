"""Tests for copilot workflow state sanitization."""

import json

from agentic_project_service.services.copilot import (
    _sanitize_workflow_state,
    _truncate_config,
)
from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY

# Sanitization limits used to be module-level constants on copilot.py.
# They were moved to the settings registry so operators can tune them
# per-project from the UI. The SUT now reads them via `get_setting(...)`
# at call time. Tests read the defaults directly off the registry —
# `get_setting()` would return these same values in the absence of a
# project_settings override, but going through it requires a Flask app
# context that unit tests don't have.
_MAX_BLOCK_NAME_LEN = SETTINGS_REGISTRY["MAX_BLOCK_NAME_LEN"].default
_MAX_CONFIG_DEPTH = SETTINGS_REGISTRY["MAX_CONFIG_DEPTH"].default
_MAX_CONFIG_VALUE_LEN = SETTINGS_REGISTRY["MAX_CONFIG_VALUE_LEN"].default
_MAX_TOTAL_STATE_LEN = SETTINGS_REGISTRY["MAX_TOTAL_STATE_LEN"].default


class TestTruncateConfig:
    """Unit tests for _truncate_config helper."""

    def test_short_values_unchanged(self):
        config = {"key": "short value", "num": 42}
        assert _truncate_config(config) == config

    def test_long_string_truncated(self):
        long_val = "x" * (_MAX_CONFIG_VALUE_LEN + 500)
        result = _truncate_config({"prompt": long_val})
        assert len(result["prompt"]) < len(long_val)
        assert result["prompt"].endswith("... [truncated]")

    def test_nested_dict_truncated(self):
        config = {"nested": {"deep": "y" * (_MAX_CONFIG_VALUE_LEN + 100)}}
        result = _truncate_config(config)
        assert result["nested"]["deep"].endswith("... [truncated]")

    def test_non_string_values_preserved(self):
        config = {"count": 42, "enabled": True, "items": [1, 2, 3]}
        assert _truncate_config(config) == config

    def test_deeply_nested_dict_capped(self):
        """Nesting beyond _MAX_CONFIG_DEPTH is replaced with _truncated marker."""
        d = {"leaf": "value"}
        for _ in range(_MAX_CONFIG_DEPTH + 5):
            d = {"nested": d}
        result = _truncate_config(d)
        # Walk down to the depth limit
        node = result
        for _ in range(_MAX_CONFIG_DEPTH):
            node = node["nested"]
        assert node == {"_truncated": True}


class TestSanitizeWorkflowState:
    """Tests for _sanitize_workflow_state."""

    def test_normal_state_unchanged(self):
        state = {
            "nodes": [
                {
                    "id": "starter_1",
                    "data": {"name": "Input", "config": {"input": {"query": "string"}}},
                }
            ],
            "edges": [{"source": "starter_1", "target": "agent_1"}],
        }
        result = _sanitize_workflow_state(state)
        assert result["nodes"][0]["data"]["name"] == "Input"
        assert result["nodes"][0]["data"]["config"] == {"input": {"query": "string"}}

    def test_edges_preserved(self):
        edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]
        state = {"nodes": [], "edges": edges}
        result = _sanitize_workflow_state(state)
        assert result["edges"] == edges

    def test_long_block_name_truncated(self):
        long_name = "A" * (_MAX_BLOCK_NAME_LEN + 50)
        state = {
            "nodes": [{"id": "n1", "data": {"name": long_name}}],
            "edges": [],
        }
        result = _sanitize_workflow_state(state)
        assert len(result["nodes"][0]["data"]["name"]) == _MAX_BLOCK_NAME_LEN

    def test_long_config_value_truncated(self):
        long_val = "x" * (_MAX_CONFIG_VALUE_LEN + 500)
        state = {
            "nodes": [{"id": "n1", "data": {"name": "Test", "config": {"prompt": long_val}}}],
            "edges": [],
        }
        result = _sanitize_workflow_state(state)
        assert result["nodes"][0]["data"]["config"]["prompt"].endswith("... [truncated]")

    def test_total_size_cap(self):
        # Create a state that exceeds the total size cap
        huge_config = {"big_field": "z" * min(_MAX_CONFIG_VALUE_LEN, 1000)}
        nodes = [
            {"id": f"n{i}", "data": {"name": f"Node {i}", "config": huge_config}}
            for i in range(200)
        ]
        state = {"nodes": nodes, "edges": []}
        result = _sanitize_workflow_state(state)
        serialized = json.dumps(result, indent=2)
        # Should be within the cap (configs get replaced with _truncated marker)
        assert (
            len(serialized) <= _MAX_TOTAL_STATE_LEN + 1000
        )  # small margin for final serialization

    def test_empty_state(self):
        result = _sanitize_workflow_state({})
        assert result == {"nodes": [], "edges": []}

    def test_none_state(self):
        result = _sanitize_workflow_state(None)
        assert result == {"nodes": [], "edges": []}

    def test_node_without_data(self):
        state = {"nodes": [{"id": "n1"}], "edges": []}
        result = _sanitize_workflow_state(state)
        assert result["nodes"][0] == {"id": "n1"}

    def test_non_dict_config_not_crashed(self):
        """Config that is a list or None should not crash _truncate_config."""
        state = {
            "nodes": [
                {"id": "n1", "data": {"name": "Test", "config": None}},
                {"id": "n2", "data": {"name": "Test2", "config": [1, 2, 3]}},
            ],
            "edges": [],
        }
        result = _sanitize_workflow_state(state)
        assert result["nodes"][0]["data"]["config"] is None
        assert result["nodes"][1]["data"]["config"] == [1, 2, 3]

    def test_does_not_mutate_original_state(self):
        """Sanitization must not modify the caller's state dict."""
        original_name = "A" * (_MAX_BLOCK_NAME_LEN + 50)
        original_config = {"prompt": "x" * (_MAX_CONFIG_VALUE_LEN + 500)}
        state = {
            "nodes": [{"id": "n1", "data": {"name": original_name, "config": original_config}}],
            "edges": [],
        }
        _sanitize_workflow_state(state)
        # Original must be untouched
        assert state["nodes"][0]["data"]["name"] == original_name
        assert len(state["nodes"][0]["data"]["config"]["prompt"]) == _MAX_CONFIG_VALUE_LEN + 500
