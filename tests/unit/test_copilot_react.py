"""Tests for the ReAct agent-based copilot implementation."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agentic.agent.tools import BuiltinTool


def _copilot_get_setting(key):
    """Stub for get_setting that returns production defaults without DB access.

    Covers all keys called by run_copilot_chat and its callees (_sanitize_workflow_state,
    _truncate_config).  Defaults taken from settings_registry.py SettingDef defaults.
    """
    defaults = {
        "COPILOT_TEMPERATURE": 0.7,
        "COPILOT_MAX_STEPS": 25,
        "COPILOT_REASONING_EFFORT": "medium",
        "MAX_TOTAL_STATE_LEN": 50_000,
        "MAX_CONFIG_VALUE_LEN": 2_000,
        "MAX_CONFIG_DEPTH": 10,
    }
    return defaults.get(key, None)


# ---------------------------------------------------------------------------
# Test build_copilot_tools
# ---------------------------------------------------------------------------


class TestBuildCopilotTools:
    """Tests for build_copilot_tools()."""

    def test_returns_dict_with_eight_tools(self):
        from agentic_project_service.services.copilot import build_copilot_tools

        diff_acc = [None]
        tools = build_copilot_tools(diff_acc)

        assert isinstance(tools, dict)
        assert len(tools) == 8
        expected_names = {
            "modify_workflow",
            "get_block_info",
            "get_db_schema",
            "list_project_assets",
            "get_asset_details",
            "execute_public_sql",
            "get_workflow_run_logs",
            "manage_project_asset",
        }
        assert set(tools.keys()) == expected_names

    def test_all_values_are_builtin_tools(self):
        from agentic_project_service.services.copilot import build_copilot_tools

        diff_acc = [None]
        tools = build_copilot_tools(diff_acc)

        for name, tool in tools.items():
            assert isinstance(tool, BuiltinTool), f"{name} is not a BuiltinTool"

    def test_read_only_metadata(self):
        from agentic_project_service.services.copilot import build_copilot_tools

        diff_acc = [None]
        tools = build_copilot_tools(diff_acc)

        assert tools["get_block_info"].is_read_only is True
        assert tools["get_db_schema"].is_read_only is True
        assert tools["list_project_assets"].is_read_only is True
        assert tools["get_asset_details"].is_read_only is True
        assert tools["get_workflow_run_logs"].is_read_only is True
        assert tools["modify_workflow"].is_read_only is False
        assert tools["execute_public_sql"].is_read_only is False
        assert tools["manage_project_asset"].is_read_only is False

    def test_all_tools_not_concurrency_safe(self):
        """All copilot tools must be concurrency_safe=False because the
        Agent framework's ThreadPoolExecutor doesn't propagate Flask's
        thread-local app context to worker threads."""
        from agentic_project_service.services.copilot import build_copilot_tools

        diff_acc = [None]
        tools = build_copilot_tools(diff_acc)

        for name, tool in tools.items():
            assert tool.is_concurrency_safe is False, (
                f"{name} must not be concurrency_safe — Flask app context "
                "won't propagate to ThreadPoolExecutor worker threads"
            )

    def test_each_tool_has_input_schema(self):
        from agentic_project_service.services.copilot import build_copilot_tools

        diff_acc = [None]
        tools = build_copilot_tools(diff_acc)

        for name, tool in tools.items():
            assert isinstance(tool.input_schema, dict), f"{name} missing input_schema"
            assert "type" in tool.input_schema, f"{name} schema missing 'type'"


# ---------------------------------------------------------------------------
# Test modify_workflow handler
# ---------------------------------------------------------------------------


class TestModifyWorkflowHandler:
    """Tests for the modify_workflow tool handler."""

    def test_accumulates_single_diff(self):
        from agentic_project_service.services.copilot import build_copilot_tools

        diff_acc = [None]
        tools = build_copilot_tools(diff_acc)
        handler = tools["modify_workflow"]

        result = handler.execute(
            {
                "add_blocks": [
                    {
                        "id": "agent_1",
                        "type": "agent",
                        "name": "A1",
                        "position": {"x": 0, "y": 0},
                        "config": {"model": "gpt-5.2", "system_prompt": "test", "input": "hi"},
                    },
                ],
            },
            None,  # context
        )

        assert diff_acc[0] is not None
        assert "add_blocks" in diff_acc[0]
        assert len(diff_acc[0]["add_blocks"]) == 1
        parsed = json.loads(result)
        assert parsed["status"] == "ok"

    def test_accumulates_multiple_diffs(self):
        from agentic_project_service.services.copilot import build_copilot_tools

        diff_acc = [None]
        tools = build_copilot_tools(diff_acc)
        handler = tools["modify_workflow"]

        handler.execute(
            {
                "add_blocks": [
                    {
                        "id": "agent_1",
                        "type": "agent",
                        "name": "A1",
                        "position": {"x": 0, "y": 0},
                        "config": {"model": "gpt-5.2", "system_prompt": "test", "input": "hi"},
                    },
                ],
            },
            None,
        )
        handler.execute(
            {
                "add_blocks": [
                    {
                        "id": "agent_2",
                        "type": "agent",
                        "name": "A2",
                        "position": {"x": 0, "y": 200},
                        "config": {"model": "gpt-5.2", "system_prompt": "test2", "input": "hi2"},
                    },
                ],
                "add_edges": [{"source": "agent_1", "target": "agent_2"}],
            },
            None,
        )

        assert len(diff_acc[0]["add_blocks"]) == 2
        assert "add_edges" in diff_acc[0]

    def test_returns_warnings_on_incomplete_config(self):
        from agentic_project_service.services.copilot import build_copilot_tools

        diff_acc = [None]
        tools = build_copilot_tools(diff_acc)
        handler = tools["modify_workflow"]

        result = handler.execute(
            {
                "add_blocks": [
                    {
                        "id": "agent_1",
                        "type": "agent",
                        "name": "Oops",
                        "position": {"x": 0, "y": 0},
                        "config": {},
                    },  # missing model, system_prompt, input
                ],
            },
            None,
        )

        parsed = json.loads(result)
        assert "warnings" in parsed
        assert len(parsed["warnings"]) > 0
        assert "action_required" in parsed

    def test_invalid_diff_returns_error(self):
        from agentic_project_service.services.copilot import build_copilot_tools

        diff_acc = [None]
        tools = build_copilot_tools(diff_acc)
        handler = tools["modify_workflow"]

        result = handler.execute({}, None)
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert diff_acc[0] is None


# ---------------------------------------------------------------------------
# Test read-only tool handlers delegate correctly
# ---------------------------------------------------------------------------


class TestReadOnlyToolHandlers:
    """Verify read-only tools delegate to existing resolver functions."""

    @patch("agentic_project_service.services.copilot.resolve_get_block_info")
    def test_get_block_info_delegates(self, mock_resolve):
        from agentic_project_service.services.copilot import build_copilot_tools

        mock_resolve.return_value = '{"type": "agent"}'
        tools = build_copilot_tools([None])

        result = tools["get_block_info"].execute({"block_type": "agent"}, None)
        mock_resolve.assert_called_once_with("agent")
        assert result == '{"type": "agent"}'

    @patch("agentic_project_service.services.copilot.resolve_get_db_schema")
    def test_get_db_schema_delegates(self, mock_resolve):
        from agentic_project_service.services.copilot import build_copilot_tools

        mock_resolve.return_value = '{"tables": []}'
        tools = build_copilot_tools([None])

        result = tools["get_db_schema"].execute({"table_name": "users"}, None)
        mock_resolve.assert_called_once_with("users")
        assert result == '{"tables": []}'

    @patch("agentic_project_service.services.copilot.resolve_get_db_schema")
    def test_get_db_schema_no_table(self, mock_resolve):
        from agentic_project_service.services.copilot import build_copilot_tools

        mock_resolve.return_value = '{"tables": []}'
        tools = build_copilot_tools([None])

        tools["get_db_schema"].execute({}, None)
        mock_resolve.assert_called_once_with(None)

    @patch("agentic_project_service.services.copilot.resolve_list_project_assets")
    def test_list_project_assets_delegates(self, mock_resolve):
        from agentic_project_service.services.copilot import build_copilot_tools

        mock_resolve.return_value = '{"agents": []}'
        tools = build_copilot_tools([None])

        tools["list_project_assets"].execute({"asset_type": "agents"}, None)
        mock_resolve.assert_called_once_with("agents")

    @patch("agentic_project_service.services.copilot.resolve_execute_public_sql")
    def test_execute_public_sql_delegates(self, mock_resolve):
        from agentic_project_service.services.copilot import build_copilot_tools

        mock_resolve.return_value = '{"status": "ok"}'
        tools = build_copilot_tools([None])

        tools["execute_public_sql"].execute({"sql": "SELECT 1"}, None)
        mock_resolve.assert_called_once_with("SELECT 1")

    @patch("agentic_project_service.services.copilot.resolve_get_asset_details")
    def test_get_asset_details_delegates(self, mock_resolve):
        from agentic_project_service.services.copilot import build_copilot_tools

        mock_resolve.return_value = '{"agent": {}}'
        tools = build_copilot_tools([None])

        tools["get_asset_details"].execute({"asset_type": "agent", "asset_id": "abc-123"}, None)
        mock_resolve.assert_called_once_with("agent", "abc-123")

    @patch("agentic_project_service.services.copilot.resolve_get_workflow_run_logs")
    def test_get_workflow_run_logs_delegates(self, mock_resolve):
        from agentic_project_service.services.copilot import build_copilot_tools

        mock_resolve.return_value = '{"executions": []}'
        tools = build_copilot_tools([None])

        tools["get_workflow_run_logs"].execute(
            {"workflow_id": "wf-123", "execution_id": "exec-456"}, None
        )
        mock_resolve.assert_called_once_with("wf-123", "exec-456")

    @patch("agentic_project_service.services.copilot.resolve_get_workflow_run_logs")
    def test_get_workflow_run_logs_without_execution_id(self, mock_resolve):
        from agentic_project_service.services.copilot import build_copilot_tools

        mock_resolve.return_value = '{"executions": []}'
        tools = build_copilot_tools([None])

        tools["get_workflow_run_logs"].execute({"workflow_id": "wf-123"}, None)
        mock_resolve.assert_called_once_with("wf-123", None)

    @patch("agentic_project_service.services.copilot.resolve_manage_project_asset")
    def test_manage_project_asset_delegates(self, mock_resolve):
        from agentic_project_service.services.copilot import build_copilot_tools

        mock_resolve.return_value = '{"status": "created"}'
        tools = build_copilot_tools([None])

        tools["manage_project_asset"].execute(
            {"action": "create", "asset_type": "agent", "config": {"name": "Test"}},
            None,
        )
        mock_resolve.assert_called_once_with(
            action="create",
            asset_type="agent",
            asset_id=None,
            config={"name": "Test"},
        )


# ---------------------------------------------------------------------------
# Test run_copilot_chat
# ---------------------------------------------------------------------------


_COPILOT_TEST_API_KEY_SENTINEL = "sk-copilot-test-sentinel"


@patch(
    "agentic_project_service.services.copilot.resolve_api_key_or_raise_for_drop",
    return_value=_COPILOT_TEST_API_KEY_SENTINEL,
)
@patch(
    "agentic_project_service.services.copilot.get_setting",
    side_effect=_copilot_get_setting,
)
class TestRunCopilotChat:
    """Tests for run_copilot_chat().

    Class-level patches:
    - resolve_api_key_or_raise_for_drop → returns _COPILOT_TEST_API_KEY_SENTINEL
      (prevents DB lookup for BYOK key AND provides a sentinel value that
      Agent mock assertions can pin via equality, not `ANY`)
    - get_setting → stub returning production defaults (prevents DB lookup for
      project_settings overrides; COPILOT_TEMPERATURE=0.7, COPILOT_MAX_STEPS=25)

    Both were introduced by CRIT-1 (copilot.py:2289) which added api_key= to the
    Agent constructor, causing every existing test to hit the DB and raise
    RuntimeError("Working outside of application context.").

    IMP-NEW-6 + R3-F1: Agent mock assertions pin api_key equality to the sentinel.
    A regression that hardcodes api_key=None (or drops the resolver call entirely)
    would fail these tests — the api_key=ANY pattern accepted that regression
    silently. Mirrors the strong sentinel pattern used in
    test_agent_block_execute_passes_api_key_to_agent (commit b4c439e4).
    """

    @patch("agentic_project_service.services.copilot.Agent")
    @patch("agentic_project_service.services.copilot.get_copilot_model")
    def test_creates_agent_with_correct_params(
        self, mock_get_model, MockAgent, _mock_get_setting, _mock_resolver
    ):
        from agentic_project_service.services.copilot import run_copilot_chat, SYSTEM_PROMPT

        mock_get_model.return_value = "gpt-5.2"

        mock_output = MagicMock()
        mock_output.content = "Hello!"
        mock_output.is_failed.return_value = False
        MockAgent.return_value.run.return_value = mock_output

        content, diff = run_copilot_chat(
            messages=[{"role": "user", "content": "Add a starter block"}],
            workflow_state={"nodes": [], "edges": []},
        )

        # Reasoning is enabled by default for the copilot
        # (COPILOT_REASONING_EFFORT="medium"). For reasoning-capable
        # models, temperature is dropped to None so Anthropic doesn't
        # reject ``temperature != 1`` with extended thinking. gpt-5.2
        # supports reasoning per litellm.supports_reasoning, so this
        # exercises the drop path.
        MockAgent.assert_called_once_with(
            model="gpt-5.2",
            system_prompt=SYSTEM_PROMPT,
            temperature=None,
            api_key=_COPILOT_TEST_API_KEY_SENTINEL,  # R3-F1: exact equality (not ANY)
            reasoning_effort="medium",
        )
        assert content == "Hello!"

    @patch("agentic_project_service.services.copilot.Agent")
    @patch("agentic_project_service.services.copilot.get_copilot_model")
    def test_keeps_temperature_for_non_reasoning_model(
        self, mock_get_model, MockAgent, _mock_get_setting, _mock_resolver
    ):
        """Models without reasoning support keep the configured
        temperature; the reasoning_effort is still forwarded (Agent
        silently drops it via litellm.supports_reasoning)."""
        from agentic_project_service.services.copilot import (
            run_copilot_chat,
            SYSTEM_PROMPT,
        )

        # gpt-4.1-mini does NOT support reasoning per litellm.supports_reasoning.
        mock_get_model.return_value = "gpt-4.1-mini"

        mock_output = MagicMock()
        mock_output.content = "Hi"
        mock_output.is_failed.return_value = False
        MockAgent.return_value.run.return_value = mock_output

        run_copilot_chat(
            messages=[{"role": "user", "content": "ping"}],
            workflow_state={"nodes": [], "edges": []},
        )

        MockAgent.assert_called_once_with(
            model="gpt-4.1-mini",
            system_prompt=SYSTEM_PROMPT,
            temperature=0.7,  # preserved — no reasoning to conflict with
            api_key=_COPILOT_TEST_API_KEY_SENTINEL,
            reasoning_effort="medium",
        )

    @patch("agentic_project_service.services.copilot.Agent")
    @patch("agentic_project_service.services.copilot.get_copilot_model")
    def test_keeps_temperature_when_reasoning_effort_unset(
        self, mock_get_model, MockAgent, _mock_get_setting, _mock_resolver
    ):
        """If an operator disables reasoning by setting
        COPILOT_REASONING_EFFORT to None / empty, the temperature gate
        does not fire and the configured temperature is preserved even
        for reasoning-capable models."""
        from agentic_project_service.services.copilot import (
            run_copilot_chat,
            SYSTEM_PROMPT,
        )

        mock_get_model.return_value = "gpt-5.2"

        mock_output = MagicMock()
        mock_output.content = "Hi"
        mock_output.is_failed.return_value = False
        MockAgent.return_value.run.return_value = mock_output

        # Override only COPILOT_REASONING_EFFORT for this call.
        def _stub(key):
            if key == "COPILOT_REASONING_EFFORT":
                return None
            return _copilot_get_setting(key)

        with patch(
            "agentic_project_service.services.copilot.get_setting",
            side_effect=_stub,
        ):
            run_copilot_chat(
                messages=[{"role": "user", "content": "ping"}],
                workflow_state={"nodes": [], "edges": []},
            )

        MockAgent.assert_called_once_with(
            model="gpt-5.2",
            system_prompt=SYSTEM_PROMPT,
            temperature=0.7,
            api_key=_COPILOT_TEST_API_KEY_SENTINEL,
            reasoning_effort=None,
        )

    @patch("agentic_project_service.services.copilot.Agent")
    @patch("agentic_project_service.services.copilot.get_copilot_model")
    def test_passes_tools_and_max_steps(
        self, mock_get_model, MockAgent, _mock_get_setting, _mock_resolver
    ):
        from agentic_project_service.services.copilot import run_copilot_chat

        mock_get_model.return_value = "gpt-5.2"

        mock_output = MagicMock()
        mock_output.content = "Done"
        mock_output.is_failed.return_value = False
        MockAgent.return_value.run.return_value = mock_output

        run_copilot_chat(
            messages=[{"role": "user", "content": "test"}],
            workflow_state={"nodes": [], "edges": []},
        )

        call_kwargs = MockAgent.return_value.run.call_args
        assert call_kwargs.kwargs["max_steps"] == 25
        assert isinstance(call_kwargs.kwargs["tools"], dict)
        assert len(call_kwargs.kwargs["tools"]) == 8

    @patch("agentic_project_service.services.copilot.Agent")
    @patch("agentic_project_service.services.copilot.get_copilot_model")
    def test_custom_model_override(
        self, mock_get_model, MockAgent, _mock_get_setting, _mock_resolver
    ):
        from agentic_project_service.services.copilot import run_copilot_chat

        mock_output = MagicMock()
        mock_output.content = "Done"
        mock_output.is_failed.return_value = False
        MockAgent.return_value.run.return_value = mock_output

        run_copilot_chat(
            messages=[{"role": "user", "content": "test"}],
            workflow_state={"nodes": [], "edges": []},
            model="claude-sonnet-4-20250514",
        )

        # Should NOT call get_copilot_model when model is provided
        mock_get_model.assert_not_called()
        MockAgent.assert_called_once()
        assert MockAgent.call_args.kwargs["model"] == "claude-sonnet-4-20250514"
        # R3-F1: pin exact equality to the resolver sentinel, not just "key present".
        # A regression that hardcodes api_key=None would pass "in kwargs" but fail this.
        assert MockAgent.call_args.kwargs["api_key"] == _COPILOT_TEST_API_KEY_SENTINEL

    @patch("agentic_project_service.services.copilot.Agent")
    @patch("agentic_project_service.services.copilot.get_copilot_model")
    def test_on_event_callback_forwarded(
        self, mock_get_model, MockAgent, _mock_get_setting, _mock_resolver
    ):
        from agentic_project_service.services.copilot import run_copilot_chat

        mock_get_model.return_value = "gpt-5.2"

        mock_output = MagicMock()
        mock_output.content = "Done"
        mock_output.is_failed.return_value = False
        MockAgent.return_value.run.return_value = mock_output

        callback = MagicMock()
        run_copilot_chat(
            messages=[{"role": "user", "content": "test"}],
            workflow_state={"nodes": [], "edges": []},
            on_event=callback,
        )

        # ExecutionContext should have been created with the callback
        call_kwargs = MockAgent.return_value.run.call_args
        ctx = call_kwargs.kwargs["context"]
        assert ctx is not None
        assert ctx.on_event is callback

    @patch("agentic_project_service.services.copilot.Agent")
    @patch("agentic_project_service.services.copilot.get_copilot_model")
    def test_workflow_state_injected_as_trailing_system_message(
        self, mock_get_model, MockAgent, _mock_get_setting, _mock_resolver
    ):
        from agentic_project_service.services.copilot import run_copilot_chat

        mock_get_model.return_value = "gpt-5.2"

        mock_output = MagicMock()
        mock_output.content = "Done"
        mock_output.is_failed.return_value = False
        MockAgent.return_value.run.return_value = mock_output

        run_copilot_chat(
            messages=[{"role": "user", "content": "test"}],
            workflow_state={
                "nodes": [
                    {
                        "id": "x",
                        "name": "X",
                        "type": "starter",
                        "position": {"x": 0, "y": 0},
                        "config": {},
                    }
                ],
                "edges": [],
            },
        )

        call_kwargs = MockAgent.return_value.run.call_args
        input_messages = call_kwargs.kwargs["input"]
        # Last message should be system with workflow state
        last_msg = input_messages[-1]
        assert last_msg["role"] == "system"
        assert "Current workflow state" in last_msg["content"]
        assert "DATA only" in last_msg["content"]

    @patch("agentic_project_service.services.copilot.Agent")
    @patch("agentic_project_service.services.copilot.get_copilot_model")
    def test_returns_none_diff_when_no_modify_calls(
        self, mock_get_model, MockAgent, _mock_get_setting, _mock_resolver
    ):
        from agentic_project_service.services.copilot import run_copilot_chat

        mock_get_model.return_value = "gpt-5.2"

        mock_output = MagicMock()
        mock_output.content = "Just explaining things."
        mock_output.is_failed.return_value = False
        MockAgent.return_value.run.return_value = mock_output

        content, diff = run_copilot_chat(
            messages=[{"role": "user", "content": "What blocks are available?"}],
            workflow_state={"nodes": [], "edges": []},
        )

        assert content == "Just explaining things."
        assert diff is None

    @patch("agentic_project_service.services.copilot.Agent")
    @patch("agentic_project_service.services.copilot.get_copilot_model")
    def test_failed_agent_raises(
        self, mock_get_model, MockAgent, _mock_get_setting, _mock_resolver
    ):
        from agentic_project_service.services.copilot import run_copilot_chat

        mock_get_model.return_value = "gpt-5.2"

        mock_output = MagicMock()
        mock_output.is_failed.return_value = True
        mock_output.error = "Rate limit exceeded"
        mock_output.content = None
        MockAgent.return_value.run.return_value = mock_output

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            run_copilot_chat(
                messages=[{"role": "user", "content": "test"}],
                workflow_state={"nodes": [], "edges": []},
            )


# ---------------------------------------------------------------------------
# Test event translation for SSE bridge (route-level logic)
# ---------------------------------------------------------------------------


class TestEventTranslation:
    """Tests for the SSE event translation constants used in the route.

    STATUS_MESSAGES and TOOL_STATUS live in services/copilot_config.py and are
    imported into routes/copilot.py without underscore prefix.  Previous test
    imports used _STATUS_MESSAGES / _TOOL_STATUS (wrong names → ImportError).
    """

    def test_status_messages_defined(self):
        from agentic_project_service.services.copilot_config import STATUS_MESSAGES

        assert "step_started" in STATUS_MESSAGES
        assert "tool_result" in STATUS_MESSAGES
        assert STATUS_MESSAGES["step_started"] == "Thinking..."

    def test_tool_status_messages_defined(self):
        from agentic_project_service.services.copilot_config import TOOL_STATUS

        assert "modify_workflow" in TOOL_STATUS
        assert "get_block_info" in TOOL_STATUS
        assert "get_db_schema" in TOOL_STATUS
        assert "list_project_assets" in TOOL_STATUS
        assert "get_asset_details" in TOOL_STATUS
        assert "execute_public_sql" in TOOL_STATUS
        assert "get_workflow_run_logs" in TOOL_STATUS
        assert "manage_project_asset" in TOOL_STATUS


# ---------------------------------------------------------------------------
# Test SSE streaming flow (route-level generate() function)
# ---------------------------------------------------------------------------


def _parse_sse_events(response_data: bytes) -> list[dict]:
    """Parse SSE lines from response data into a list of dicts."""
    events = []
    for line in response_data.decode().strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def _make_test_app():
    """Create a Flask app with the copilot blueprint for testing."""
    from flask import Flask
    from agentic_project_service.routes.copilot import copilot_bp

    app = Flask(__name__)
    app.register_blueprint(copilot_bp)
    return app


# Auth header + decode_jwt patch so require_auth passes
_AUTH_HEADERS = {"Authorization": "Bearer fake-token"}


class TestSSEStreamingFlow:
    """Integration tests for the threading/queue/SSE bridge in the chat route."""

    @patch("agentic_project_service.routes.copilot.run_copilot_chat")
    @patch("agentic_project_service.routes.copilot.db")
    @patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    )
    def test_successful_chat_emits_status_and_complete(self, _mock_jwt, mock_db, mock_chat):
        """A happy-path chat should emit status events then a complete event."""
        app = _make_test_app()

        mock_session = MagicMock()
        mock_db.session = mock_session

        # The route calls execute() multiple times:
        #   1. fetchone() — session lookup (returns workflow_id)
        #   2. (insert user message — no fetch)
        #   3. fetchall() — message history
        #   4. (update session timestamp — no fetch)
        #   5. (insert assistant message — no fetch)
        session_row = MagicMock()
        session_row.__getitem__ = lambda self, idx: "wf-123"

        mock_exec = MagicMock()
        mock_exec.fetchone.return_value = session_row
        mock_exec.fetchall.return_value = [("user", "hello")]
        mock_session.execute.return_value = mock_exec

        def fake_chat(messages, workflow_state, on_event=None):
            if on_event:
                on_event({"type": "step_started"})
                on_event(
                    {
                        "type": "tool_call",
                        "tool_name": "get_block_info",
                        "arguments": {},
                    }
                )
            return ("Here is your answer", None)

        mock_chat.side_effect = fake_chat

        with app.test_client() as client:
            resp = client.post(
                "/api/copilot/sessions/sess-1/chat",
                json={
                    "message": "hello",
                    "workflow_state": {"nodes": [], "edges": []},
                },
                headers=_AUTH_HEADERS,
            )

        assert resp.status_code == 200
        assert resp.content_type.startswith("text/event-stream")

        events = _parse_sse_events(resp.data)
        event_types = [e["event"] for e in events]

        assert "status" in event_types
        assert "tool_call" in event_types
        assert event_types[-1] == "complete"

        complete = events[-1]
        assert complete["content"] == "Here is your answer"
        assert complete["workflow_diff"] is None

    @patch("agentic_project_service.routes.copilot.run_copilot_chat")
    @patch("agentic_project_service.routes.copilot.db")
    @patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "user-1", "role": "authenticated"},
    )
    def test_agent_error_emits_error_event(self, _mock_jwt, mock_db, mock_chat):
        """If run_copilot_chat raises, an error SSE event should be emitted."""
        app = _make_test_app()

        mock_session = MagicMock()
        mock_db.session = mock_session
        session_row = MagicMock()
        session_row.__getitem__ = lambda self, idx: "wf-123"
        mock_exec = MagicMock()
        mock_exec.fetchone.return_value = session_row
        mock_exec.fetchall.return_value = [("user", "hello")]
        mock_session.execute.return_value = mock_exec

        mock_chat.side_effect = RuntimeError("LLM exploded")

        with app.test_client() as client:
            resp = client.post(
                "/api/copilot/sessions/sess-1/chat",
                json={
                    "message": "hello",
                    "workflow_state": {"nodes": [], "edges": []},
                },
                headers=_AUTH_HEADERS,
            )

        events = _parse_sse_events(resp.data)
        event_types = [e["event"] for e in events]

        assert "error" in event_types
        error_event = next(e for e in events if e["event"] == "error")
        assert "LLM exploded" in error_event["error"]

    def test_queue_timeout_raises_empty(self):
        """Verify that q.get(timeout=...) raises Empty when nothing is enqueued."""
        import queue as queue_mod
        from queue import Empty

        q = queue_mod.Queue()
        with pytest.raises(Empty):
            q.get(timeout=0.01)
