from agentic_project_service.routes.orchestrations import _preresponse_edited


class TestPreResponseEdited:
    def test_true_when_preresponse_modified(self):
        events = [
            {"type": "delegation_started"},
            {
                "type": "hook_result",
                "hook_event": "PreResponse",
                "modified": True,
                "blocked": False,
            },
        ]
        assert _preresponse_edited(events) is True

    def test_true_when_preresponse_blocked(self):
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
            # A modification at a DIFFERENT event must not count as a
            # PreResponse edit. Uses `hook_event` (the real key) — with the
            # pre-rename `event` key this decoy was inert.
            {"type": "hook_result", "hook_event": "PreToolUse", "modified": True, "blocked": False},
        ]
        assert _preresponse_edited(events) is False

    def test_false_when_no_hook_results(self):
        assert _preresponse_edited([{"type": "delegation_started"}]) is False
