from agentic_project_service.routes.agents import _preresponse_edited


class TestAgentPreResponseEdited:
    def test_true_when_preresponse_modified(self):
        events = [
            {
                "type": "hook_result",
                "hook_event": "PreResponse",
                "modified": True,
                "blocked": False,
            },
        ]
        assert _preresponse_edited(events) is True

    def test_true_when_preresponse_blocked(self):
        # A PreResponse *block* must also trigger reconciliation, else the raw
        # answer silently persists under streaming.
        events = [
            {
                "type": "hook_result",
                "hook_event": "PreResponse",
                "modified": False,
                "blocked": True,
            },
        ]
        assert _preresponse_edited(events) is True

    def test_false_when_no_modification(self):
        events = [
            {
                "type": "hook_result",
                "hook_event": "PreResponse",
                "modified": False,
                "blocked": False,
            },
        ]
        assert _preresponse_edited(events) is False

    def test_false_when_other_event_modified(self):
        """A modification at a non-PreResponse event must not trigger
        reconciliation — otherwise every tool-modifying run would needlessly
        discard the streamed buffer."""
        events = [
            {"type": "hook_result", "hook_event": "PreToolUse", "modified": True, "blocked": False},
            {
                "type": "hook_result",
                "hook_event": "PostToolUse",
                "modified": True,
                "blocked": False,
            },
        ]
        assert _preresponse_edited(events) is False

    def test_false_when_empty(self):
        assert _preresponse_edited([]) is False
