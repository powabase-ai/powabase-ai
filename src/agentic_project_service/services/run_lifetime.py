"""What a streamed run holds while it runs, and when it lets go.

A streamed run can outlive its HTTP response: when the client disconnects,
the agent and orchestration stream routes hand the run to a thread that
finishes it. Two things must last until the run really ends, not until the
response closes:

* its MAX_CONCURRENT_RUNS slot (services/run_gate.py), so the gate caps real
  concurrent runs rather than open connections;
* its liveness record in the run registry, so the stale-run cleanup never
  fails a run that is still going.

``SharedRelease`` runs a release once, after every holder has let go.
``RunLifetime`` ties the two together for one streamed run.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from functools import partial

from .run_registry import register_live_run, unregister_run

logger = logging.getLogger(__name__)


def _noop() -> None:
    return None


class SharedRelease:
    """Calls ``release`` once, when the last holder lets go."""

    def __init__(self, release: Callable[[], None] | None):
        self._release = release
        self._holds = 0
        self._released = release is None
        self._lock = threading.Lock()

    def hold(self) -> Callable[[], None]:
        """Take a hold. Returns its let-go callable; calling it again is a no-op.
        A hold taken after the release is a no-op too."""
        with self._lock:
            if self._released:
                return _noop
            self._holds += 1
        state = {"let_go": False}

        def let_go() -> None:
            with self._lock:
                if state["let_go"]:
                    return
                state["let_go"] = True
                self._holds -= 1
                fire = self._holds == 0 and not self._released
                if fire:
                    self._released = True
            if fire:
                try:
                    self._release()
                except Exception:
                    logger.warning("Run lifetime: release failed", exc_info=True)

        return let_go


NO_RELEASE = SharedRelease(None)


class RunLifetime:
    """The run-gate slot and the liveness record of one streamed run.

    The streaming generator is one holder of the liveness record (``end()``
    in its ``finally``); admit_run holds the slot for the response. A thread
    that carries the run on after a disconnect calls ``carry_on()`` before it
    starts, is started with ``start_carried_on``, and calls the returned
    callable when it finishes.
    """

    def __init__(self, slot: SharedRelease):
        self._slot = slot
        self._live: SharedRelease | None = None
        self._generator_hold: Callable[[], None] = _noop

    def mark_live(self, run_id: str) -> None:
        register_live_run(run_id)
        self._live = SharedRelease(partial(unregister_run, run_id))
        self._generator_hold = self._live.hold()

    def carry_on(self) -> Callable[[], None]:
        slot_hold = self._slot.hold()
        live_hold = self._live.hold() if self._live is not None else _noop

        def let_go() -> None:
            live_hold()
            slot_hold()

        return let_go

    def end(self) -> None:
        self._generator_hold()


def start_carried_on(thread: threading.Thread, let_go: Callable[[], None]) -> None:
    """Start a thread that carries a run on, holding what ``carry_on()`` took.

    The thread lets go when it finishes; if it never starts (``RuntimeError:
    can't start new thread`` under load), let go here instead, or the slot
    would be renewed until its hold ceiling.
    """
    try:
        thread.start()
    except BaseException:
        let_go()
        raise
