"""Launch-claim repair guard: the model must never ASSERT a walkthrough launch
it didn't perform — and the repair must never fire on anything weaker than an
assertion.

Two layers under test:

1. ``looks_like_guide_launch_assertion`` — the strict predicate driving the
   corrective retry. Unlike the Studio's recall-tuned ``launch-claim.ts``
   telemetry detector (where a false positive costs an analytics row), a false
   positive HERE mutates the turn: the retry nudge instructs the model to call
   ``play_guide``, which would convert a permission question ("Would you like
   me to start the walkthrough?") into an unrequested launch — the exact
   clarifying-question behaviour the system prompt solicits. So questions,
   offers, ability/conditional phrasing, and negations must never match.
2. ``run_project_copilot_chat``'s repair path — bounded retry, correction
   suffix when a false claim would otherwise be delivered, retry re-check,
   reasoning suppression, and the structured outcome log.
"""

from unittest.mock import MagicMock

from agentic_project_service.services import project_copilot as pc
from agentic_project_service.services.guide_launch_claim import looks_like_guide_launch_assertion


# ---------------------------------------------------------------------------
# Detector — strict assertions only
# ---------------------------------------------------------------------------


def test_matches_first_person_launch_assertions():
    assert looks_like_guide_launch_assertion("I've launched the walkthrough for you.")
    assert looks_like_guide_launch_assertion("I've launched the create-table walkthrough for you!")
    assert looks_like_guide_launch_assertion("I have started the connect walkthrough.")
    assert looks_like_guide_launch_assertion("I'm opening the walkthrough on your screen.")
    assert looks_like_guide_launch_assertion("I'll launch the walkthrough now.")
    assert looks_like_guide_launch_assertion("I've launched it for you.")


def test_matches_subjectless_progressive_and_state_assertions():
    assert looks_like_guide_launch_assertion("Starting the walkthrough now — follow along!")
    assert looks_like_guide_launch_assertion("Launching the connect walkthrough…")
    assert looks_like_guide_launch_assertion("The walkthrough is now on screen.")
    assert looks_like_guide_launch_assertion("The create-table walkthrough is launching.")


def test_never_matches_questions():
    assert not looks_like_guide_launch_assertion("Would you like me to start the walkthrough?")
    assert not looks_like_guide_launch_assertion("Shall I launch the connect walkthrough?")
    assert not looks_like_guide_launch_assertion("Want me to open the walkthrough for this?")
    assert not looks_like_guide_launch_assertion(
        "Tables hold structured data. Should I start the walkthrough?"
    )


def test_never_matches_offers_or_ability_statements():
    assert not looks_like_guide_launch_assertion(
        "I can launch the create-table walkthrough whenever you're ready."
    )
    assert not looks_like_guide_launch_assertion("I could show you the walkthrough for that.")
    assert not looks_like_guide_launch_assertion(
        "I'll launch the walkthrough if you'd like — just say the word."
    )
    assert not looks_like_guide_launch_assertion(
        "Let me know if you want the walkthrough and I'll start it."
    )


def test_never_matches_negations_or_denials():
    assert not looks_like_guide_launch_assertion("I can't launch the walkthrough right now.")
    assert not looks_like_guide_launch_assertion("Sorry, I cannot start the walkthrough for that.")
    assert not looks_like_guide_launch_assertion("I won't open the walkthrough until you confirm.")
    assert not looks_like_guide_launch_assertion(
        "I'm unable to launch the walkthrough for you right now."
    )
    assert not looks_like_guide_launch_assertion("There's no walkthrough for this yet.")
    assert not looks_like_guide_launch_assertion("I haven't launched the walkthrough yet.")


def test_never_matches_descriptions_without_an_assertion():
    assert not looks_like_guide_launch_assertion("You can create a table from the editor.")
    assert not looks_like_guide_launch_assertion(
        "The walkthrough covers connecting your coding agent."
    )
    assert not looks_like_guide_launch_assertion(
        "You can start the walkthrough from the copilot panel."
    )


def test_assertion_must_be_contained_in_one_sentence():
    # Assertion verb in one sentence, "walkthrough" in the next — not a claim.
    assert not looks_like_guide_launch_assertion(
        "Let's start with tables. A walkthrough exists for that."
    )
    filler = "x" * 70
    assert not looks_like_guide_launch_assertion(f"I'm showing {filler} walkthrough")


def test_handles_none_and_empty_text():
    assert not looks_like_guide_launch_assertion(None)
    assert not looks_like_guide_launch_assertion("")


# ---------------------------------------------------------------------------
# Repair path in run_project_copilot_chat
# ---------------------------------------------------------------------------


def _agent_output(content, failed=False):
    out = MagicMock()
    out.content = content
    out.is_failed.return_value = failed
    out.error = "boom" if failed else None
    return out


def _run_chat_with_mocked_agent(mocker, run_side_effect, on_event=None):
    mocker.patch.object(pc, "resolve_api_key_or_raise_for_drop", return_value="key")
    agent_cls = mocker.patch.object(pc, "Agent")
    agent_cls.return_value.run.side_effect = run_side_effect
    result = pc.run_project_copilot_chat(
        [{"role": "user", "content": "help me connect"}], on_event=on_event
    )
    return result, agent_cls.return_value.run


def test_launch_claim_without_play_guide_retries_with_corrective_nudge(mocker):
    """First run asserts a launch but never calls play_guide → the service
    re-runs the turn once (bounded), feeding back the offending reply plus a
    corrective user message; the retry's play_guide call and content win."""

    def side_effect(*args, **kwargs):
        if run_mock.call_count == 1:
            return _agent_output("I've launched the create-table walkthrough for you!")
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

    retry_kwargs = run_mock.call_args_list[1].kwargs
    retry_input = retry_kwargs["input"]
    assert retry_input[-2]["role"] == "assistant"
    assert "I've launched" in retry_input[-2]["content"]
    assert retry_input[-1]["role"] == "user"
    assert "play_guide" in retry_input[-1]["content"]
    # The nudge asks for the previous reply back, not a fresh (shorter) answer.
    assert "previous reply" in retry_input[-1]["content"]
    # Bounded: the repair is a tool call + a rewrite, not a fresh research loop.
    assert retry_kwargs["max_steps"] == pc._REPAIR_MAX_STEPS
    assert pc._REPAIR_MAX_STEPS < pc.PROJECT_COPILOT_MAX_STEPS


def test_no_retry_on_permission_question(mocker):
    """The regression that motivated the strict predicate: a clarifying
    question about launching must be delivered as-is, never converted into an
    unrequested launch."""
    (content, guide_id, _), run_mock = _run_chat_with_mocked_agent(
        mocker,
        lambda *a, **k: _agent_output("Would you like me to start the walkthrough?"),
    )
    assert run_mock.call_count == 1
    assert guide_id is None
    assert content == "Would you like me to start the walkthrough?"


def test_no_retry_when_play_guide_was_called(mocker):
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


def test_failed_retry_appends_correction_instead_of_delivering_false_claim(mocker):
    """When the retry errors, the backend KNOWS the original claim is false —
    it must not stream it verbatim. The original reply is kept but a visible
    correction is appended."""
    outputs = [
        _agent_output("I've launched the connect walkthrough."),
        _agent_output(None, failed=True),
    ]
    (content, guide_id, _), run_mock = _run_chat_with_mocked_agent(
        mocker, lambda *a, **k: outputs.pop(0)
    )
    assert run_mock.call_count == 2
    assert guide_id is None
    assert content.startswith("I've launched the connect walkthrough.")
    assert "not actually launched" in content


def test_still_claiming_retry_gets_correction_and_no_third_attempt(mocker):
    """A retry that claims-without-launching again is corrected, not retried
    again and not delivered unmarked."""
    outputs = [
        _agent_output("I've launched the connect walkthrough."),
        _agent_output("I've launched the connect walkthrough, take a look!"),
    ]
    (content, guide_id, _), run_mock = _run_chat_with_mocked_agent(
        mocker, lambda *a, **k: outputs.pop(0)
    )
    assert run_mock.call_count == 2
    assert guide_id is None
    assert content.startswith("I've launched the connect walkthrough, take a look!")
    assert "not actually launched" in content


def test_repair_skipped_when_turn_already_ate_the_time_budget(mocker):
    """The route streams for at most 300s and the turn lock TTLs at 360s — a
    repair on top of an already-long turn would blow both (stream/DB divergence
    plus the documented double-charge race). Past the budget the repair is
    skipped and the claim corrected textually instead."""
    clock = iter([0.0, pc._REPAIR_TIME_BUDGET_SECONDS + 5.0])
    mocker.patch.object(pc.time, "monotonic", side_effect=lambda: next(clock))
    (content, guide_id, _), run_mock = _run_chat_with_mocked_agent(
        mocker, lambda *a, **k: _agent_output("I've launched the connect walkthrough.")
    )
    assert run_mock.call_count == 1  # no retry attempted
    assert guide_id is None
    assert "not actually launched" in content


def test_retry_suppresses_reasoning_but_keeps_tool_events(mocker):
    """The retry shares the panel's live-rendered reasoning buffer — streaming
    the model reasoning about its own false claim would be user-visible. The
    retry context must drop reasoning_delta and keep tool_call events."""
    seen = []

    def side_effect(*args, **kwargs):
        ctx = kwargs["context"]
        if run_mock.call_count == 1:
            ctx.on_event({"type": "reasoning_delta", "delta": "thinking about docs"})
            return _agent_output("I've launched the connect walkthrough.")
        ctx.on_event({"type": "reasoning_delta", "delta": "oops I never called it"})
        ctx.on_event({"type": "tool_call", "tool_name": "play_guide", "arguments": {}})
        kwargs["tools"]["play_guide"].handler({"sequence_id": "connect"}, None)
        return _agent_output("Launching the connect walkthrough now.")

    mocker.patch.object(pc, "resolve_api_key_or_raise_for_drop", return_value="key")
    agent_cls = mocker.patch.object(pc, "Agent")
    run_mock = agent_cls.return_value.run
    run_mock.side_effect = side_effect

    pc.run_project_copilot_chat(
        [{"role": "user", "content": "help me connect"}], on_event=seen.append
    )

    deltas = [e for e in seen if e.get("type") == "reasoning_delta"]
    assert [e["delta"] for e in deltas] == ["thinking about docs"]  # retry's delta dropped
    assert any(e.get("type") == "tool_call" for e in seen)  # tool events still surface


def test_repair_outcome_is_logged_structured(mocker, caplog):
    """Successful repairs mask the Studio's guide_launch_missing telemetry, so
    the backend log line is the measurement of the underlying prompt defect —
    it must carry a parseable outcome."""
    import logging

    def side_effect(*args, **kwargs):
        if run_mock.call_count == 1:
            return _agent_output("I've launched the connect walkthrough.")
        kwargs["tools"]["play_guide"].handler({"sequence_id": "connect"}, None)
        return _agent_output("Launching the connect walkthrough now.")

    mocker.patch.object(pc, "resolve_api_key_or_raise_for_drop", return_value="key")
    agent_cls = mocker.patch.object(pc, "Agent")
    run_mock = agent_cls.return_value.run
    run_mock.side_effect = side_effect

    with caplog.at_level(logging.WARNING):
        pc.run_project_copilot_chat([{"role": "user", "content": "help me connect"}])

    repair_lines = [r.message for r in caplog.records if "launch_claim_repair" in r.message]
    assert any("outcome=launched" in line for line in repair_lines)
