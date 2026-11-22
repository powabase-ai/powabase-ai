"""A streamed orchestration that its client abandons keeps its gate slot and
its liveness record until the worker running it actually ends."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import fakeredis
import pytest
from flask import Flask

from agentic_project_service.routes import orchestrations as orchestrations_route
from agentic_project_service.services import hook_loader
from agentic_project_service.services import orchestration as orch_service
from agentic_project_service.services import run_gate as gate_mod
from agentic_project_service.services import run_lifetime as lifetime_mod
from agentic_project_service.services.run_gate import RunGate

ORCH_ID = "4a8b2d3f-6c8e-4f22-8b4d-9e3a7c5f2b81"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setenv("PROJECT_REF", "ref-a")
    g = RunGate(redis_client=fakeredis.FakeStrictRedis(), renew_in_background=False)
    monkeypatch.setattr(gate_mod, "default_gate", g)
    monkeypatch.setattr(gate_mod, "get_setting", lambda key: 1)
    return g


class _BlockingOrchestration:
    def __init__(self):
        self.orchestrator_config: dict = {}
        self.settings: dict = {}
        self.running = threading.Event()
        self.finish = threading.Event()
        self.done = threading.Event()

    def run(self, input, context, **kwargs):
        context.on_event({"type": "step", "step": 1})
        self.running.set()
        self.finish.wait(5)
        self.done.set()
        return SimpleNamespace()


def test_an_abandoned_orchestration_holds_its_slot_and_liveness_until_it_ends(gate, monkeypatch):
    orchestration = _BlockingOrchestration()
    live, gone = [], []
    monkeypatch.setattr(lifetime_mod, "register_live_run", live.append)
    monkeypatch.setattr(lifetime_mod, "unregister_run", gone.append)
    fake_session = MagicMock()
    fake_session.get.return_value = None  # skips the view's model checks
    app = Flask(__name__)
    app.register_blueprint(orchestrations_route.orchestrations_bp)
    with (
        patch.object(orchestrations_route.db, "session", fake_session),
        patch.object(orch_service, "build_orchestration", return_value=(None, orchestration)),
        patch.object(hook_loader, "load_hooks_for_orchestration", return_value=[]),
        patch.object(
            orch_service,
            "get_or_create_orchestration_session",
            return_value=("sess-uuid", "orch_sess_1", True),
        ),
        patch.object(
            orch_service, "create_orchestration_run", return_value=("run-uuid", "orch_run_1")
        ),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "service", "role": "service_role", "is_service_role": True},
        ),
        app.test_client() as c,
    ):
        resp = c.post(
            f"/api/orchestrations/{ORCH_ID}/run/stream",
            json={"message": "hi"},
            headers={"Authorization": "Bearer fake"},
            buffered=False,
        )
        body = iter(resp.response)
        assert b'"event": "start"' in next(body)
        assert b'"step"' in next(body)
        assert orchestration.running.is_set()
        resp.close()  # the client goes away while the worker runs

        assert gate.active_runs() == 1
        assert live == ["orch_run_1"] and gone == []
        assert (
            c.post(
                f"/api/orchestrations/{ORCH_ID}/run/stream",
                json={"message": "hi"},
                headers={"Authorization": "Bearer fake"},
            ).status_code
            == 429
        )

        orchestration.finish.set()
        assert orchestration.done.wait(5)
        for _ in range(100):
            if gone:
                break
            threading.Event().wait(0.02)
    assert gone == ["orch_run_1"]
    assert gate.active_runs() == 0
