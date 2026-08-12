"""Launch-claim guard: the model must never claim a walkthrough launch it
didn't perform.

Two layers under test:

1. ``looks_like_guide_launch_claim`` — a Python mirror of the Studio's
   ``launch-claim.ts`` detector (same phrasings, same negation rules), so both
   ends of the contract classify the same texts the same way.
2. ``run_project_copilot_chat``'s retry guard — when the assistant's final text
   claims a launch but ``play_guide`` was never called, the turn is re-run once
   with a corrective nudge instead of streaming the false claim to the user.
   (The Studio's ``guide_launch_missing`` telemetry observes this case; this
   guard is what actually repairs it.)
"""

from unittest.mock import MagicMock

from agentic_project_service.services import project_copilot as pc
from agentic_project_service.services.guide_launch_claim import looks_like_guide_launch_claim


# ---------------------------------------------------------------------------
# Detector — mirrors launch-claim.test.ts case for case
# ---------------------------------------------------------------------------


def test_matches_common_launch_claim_phrasings():
    assert looks_like_guide_launch_claim("I've launched the walkthrough for you.")
    assert looks_like_guide_launch_claim("Starting the walkthrough now — follow along!")
    assert looks_like_guide_launch_claim("I'll highlight the New table button in the walkthrough.")
    assert looks_like_guide_launch_claim("The walkthrough is now on screen.")
    assert looks_like_guide_launch_claim("I've launched it for you.")


def test_does_not_match_ordinary_replies():
    assert not looks_like_guide_launch_claim("You can create a table from the editor.")
    assert not looks_like_guide_launch_claim("Here is the SQL you asked for.")


def test_does_not_match_walkthrough_mention_with_no_launch_verb():
    assert not looks_like_guide_launch_claim("The walkthrough covers connecting your coding agent.")


def test_does_not_match_negated_launch_phrasing():
    assert not looks_like_guide_launch_claim("I can't launch the walkthrough right now.")
    assert not looks_like_guide_launch_claim("Sorry, I cannot start the walkthrough for that.")
    assert not looks_like_guide_launch_claim("I won't open the walkthrough until you confirm.")
    assert not looks_like_guide_launch_claim(
        "I'm unable to launch the walkthrough for you right now."
    )
    assert not looks_like_guide_launch_claim(
        "I'll show you — actually there's no walkthrough for this yet."
    )


def test_handles_none_and_empty_text():
    assert not looks_like_guide_launch_claim(None)
    assert not looks_like_guide_launch_claim("")


def test_verb_and_walkthrough_must_share_a_sentence_within_60_chars():
    # Verb in one sentence, "walkthrough" in the next — not a claim.
    assert not looks_like_guide_launch_claim(
        "Let's start with tables. A walkthrough exists for that."
    )
    # More than 60 chars between verb and "walkthrough" — too far to be a claim.
    filler = "x" * 70
    assert not looks_like_guide_launch_claim(f"I'll show {filler} walkthrough")


# ---------------------------------------------------------------------------
# Retry guard in run_project_copilot_chat
# ---------------------------------------------------------------------------


def _agent_output(content, failed=False):
    out = MagicMock()
    out.content = content
    out.is_failed.return_value = failed
    out.error = "boom" if failed else None
    return out


def _run_chat_with_mocked_agent(mocker, run_side_effect):
    mocker.patch.object(pc, "resolve_api_key_or_raise_for_drop", return_value="key")
    agent_cls = mocker.patch.object(pc, "Agent")
    agent_cls.return_value.run.side_effect = run_side_effect
    result = pc.run_project_copilot_chat([{"role": "user", "content": "help me connect"}])
    return result, agent_cls.return_value.run


def test_launch_claim_without_play_guide_retries_with_corrective_nudge(mocker):
    """First run claims a launch but never calls play_guide → the service re-runs
    the turn once, feeding back the offending reply plus a corrective user
    message; the retry's play_guide call and content are what get returned."""

    def side_effect(*args, **kwargs):
        if run_mock.call_count == 1:
            return _agent_output("I've launched the create-table walkthrough for you!")
        # The corrective retry actually calls the tool this time.
        kwargs["tools"]["play_guide"].handler({"sequence_id": "create-table"}, None)
        return _agent_output("Launching the create-table walkthrough now.")

    mocker.patch.object(pc, "resolve_api_key_or_raise_for_drop", return_value="key")
    agent_cls = mocker.patch.object(pc, "Agent")
    run_mock = agent_cls.return_value.run
    run_mock.side_effect = side_effect

    content, guide_id, notice = pc.run_project_copilot_chat(
        [{"role": "user", "content": "help me create a table"}]
    )

    assert run_mock.call_count == 2
    assert guide_id == "create-table"
    assert content == "Launching the create-table walkthrough now."

    # The retry input carries the offending assistant reply + a corrective user
    # nudge appended to the original window (so the model sees what it claimed).
    retry_input = run_mock.call_args_list[1].kwargs["input"]
    assert retry_input[-2]["role"] == "assistant"
    assert "I've launched" in retry_input[-2]["content"]
    assert retry_input[-1]["role"] == "user"
    assert "play_guide" in retry_input[-1]["content"]


def test_no_retry_when_play_guide_was_called(mocker):
    """A genuine launch (text claim + tool call) must not trigger the retry."""

    def side_effect(*args, **kwargs):
        kwargs["tools"]["play_guide"].handler({"sequence_id": "connect"}, None)
        return _agent_output("I've launched the connect walkthrough.")

    (content, guide_id, _), run_mock = _run_chat_with_mocked_agent(mocker, side_effect)
    assert run_mock.call_count == 1
    assert guide_id == "connect"


def test_no_retry_when_text_makes_no_launch_claim(mocker):
    (content, guide_id, _), run_mock = _run_chat_with_mocked_agent(
        mocker, lambda *a, **k: _agent_output("Tables hold structured data.")
    )
    assert run_mock.call_count == 1
    assert guide_id is None
    assert content == "Tables hold structured data."


def test_failed_retry_falls_back_to_original_reply(mocker):
    """If the corrective retry itself fails, keep the original reply rather than
    failing a turn that already produced an answer (the Studio's telemetry still
    observes the claim-without-launch)."""
    outputs = [
        _agent_output("I've launched the connect walkthrough."),
        _agent_output(None, failed=True),
    ]
    (content, guide_id, _), run_mock = _run_chat_with_mocked_agent(
        mocker, lambda *a, **k: outputs.pop(0)
    )
    assert run_mock.call_count == 2
    assert guide_id is None
    assert content == "I've launched the connect walkthrough."
