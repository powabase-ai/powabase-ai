"""What a streamed run holds while it runs: its run-gate slot and its
liveness record, until the response and every thread carrying the run on
have let go."""

import threading
from unittest.mock import MagicMock

from agentic_project_service.services import run_lifetime as lt
from agentic_project_service.services.run_lifetime import NO_RELEASE, RunLifetime, SharedRelease


def test_released_once_when_the_last_holder_lets_go():
    release = MagicMock()
    shared = SharedRelease(release)
    first, second = shared.hold(), shared.hold()
    first()
    release.assert_not_called()
    second()
    release.assert_called_once()


def test_letting_go_twice_counts_once():
    release = MagicMock()
    shared = SharedRelease(release)
    first, second = shared.hold(), shared.hold()
    first()
    first()
    release.assert_not_called()
    second()
    release.assert_called_once()


def test_a_hold_after_the_release_is_a_no_op():
    release = MagicMock()
    shared = SharedRelease(release)
    shared.hold()()
    late = shared.hold()
    late()
    release.assert_called_once()


def test_a_failing_release_does_not_raise(caplog):
    shared = SharedRelease(MagicMock(side_effect=RuntimeError("redis down")))
    shared.hold()()
    assert "release failed" in caplog.text


def test_no_release_never_releases():
    NO_RELEASE.hold()()
    NO_RELEASE.hold()()


def test_concurrent_holders_release_exactly_once():
    release = MagicMock()
    shared = SharedRelease(release)
    lets = [shared.hold() for _ in range(50)]
    threads = [threading.Thread(target=let_go) for let_go in lets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    release.assert_called_once()


def test_a_lifetime_keeps_the_slot_and_the_liveness_record_while_a_thread_carries_on(
    monkeypatch,
):
    registered, unregistered = [], []
    monkeypatch.setattr(lt, "register_live_run", registered.append)
    monkeypatch.setattr(lt, "unregister_run", unregistered.append)
    slot_release = MagicMock()
    slot = SharedRelease(slot_release)
    response_hold = slot.hold()  # what admit_run holds for the response

    life = RunLifetime(slot)
    life.mark_live("run_1")
    assert registered == ["run_1"]
    carry_on = life.carry_on()  # a disconnect hands the run to a thread
    life.end()  # the generator ends
    response_hold()  # the response is closed
    assert unregistered == [] and not slot_release.called

    carry_on()  # the thread finishes
    assert unregistered == ["run_1"]
    slot_release.assert_called_once()


def test_without_a_thread_the_generator_end_drops_the_liveness_record(monkeypatch):
    unregistered = []
    monkeypatch.setattr(lt, "register_live_run", lambda run_id: None)
    monkeypatch.setattr(lt, "unregister_run", unregistered.append)
    life = RunLifetime(NO_RELEASE)
    life.mark_live("run_1")
    life.end()
    assert unregistered == ["run_1"]


def test_ending_before_marking_live_is_harmless(monkeypatch):
    monkeypatch.setattr(lt, "unregister_run", MagicMock(side_effect=AssertionError))
    life = RunLifetime(NO_RELEASE)
    life.end()
    life.carry_on()()


def test_an_agent_streams_background_finish_lets_go_however_it_ends():
    import pytest

    from agentic_project_service.routes.agents import _carried_on

    let_go = MagicMock()
    assert _carried_on(let_go, lambda f, *a: f(*a), lambda x: x * 2, 21) == 42
    let_go.assert_called_once()

    let_go = MagicMock()

    def boom():
        raise RuntimeError("finish failed")

    with pytest.raises(RuntimeError):
        _carried_on(let_go, lambda f: f(), boom)
    let_go.assert_called_once()


class _UnstartableThread:
    def start(self):
        raise RuntimeError("can't start new thread")


def test_a_thread_that_cannot_start_lets_go_of_its_hold():
    """Under thread exhaustion start() raises; the hold carry_on() took must
    not stay taken (the slot would be renewed for hours)."""
    release = MagicMock()
    life = RunLifetime(SharedRelease(release))
    let_go = life.carry_on()
    try:
        lt.start_carried_on(_UnstartableThread(), let_go)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the start error must propagate")
    release.assert_called_once()


def test_a_started_thread_keeps_its_hold():
    release = MagicMock()
    life = RunLifetime(SharedRelease(release))
    let_go = life.carry_on()
    done = threading.Event()
    lt.start_carried_on(threading.Thread(target=done.set), let_go)
    assert done.wait(2)
    release.assert_not_called()
    let_go()
    release.assert_called_once()


def test_mark_live_twice_keeps_one_hold(monkeypatch):
    """A second mark_live must not replace the first hold (which would then
    never be let go, and the liveness record never removed)."""
    live, gone = [], []
    monkeypatch.setattr(lt, "register_live_run", live.append)
    monkeypatch.setattr(lt, "unregister_run", gone.append)
    life = RunLifetime(NO_RELEASE)
    life.mark_live("run_1")
    first = life._live
    life.mark_live("run_1")
    assert life._live is first and live == ["run_1"]
    carried = life.carry_on()
    life.end()
    assert gone == []  # the carried-on thread still holds it
    carried()
    assert gone == ["run_1"]
