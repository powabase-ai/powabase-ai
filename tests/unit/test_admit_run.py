"""@admit_run: the request-scoped MAX_CONCURRENT_RUNS slot."""

from unittest.mock import MagicMock, patch

import fakeredis
import pytest
from flask import Flask, Response

from agentic_project_service.services import run_gate as gate_mod
from agentic_project_service.services.run_gate import RunGate

AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"
ORCH_ID = "4a8b2d3f-6c8e-4f22-8b4d-9e3a7c5f2b81"
WORKFLOW_ID = "5b9c3e4a-7d9f-4a33-9c5e-af4b8d6a3c92"
HEADERS = {"Authorization": "Bearer fake"}


def _app():
    app = Flask(__name__)

    @app.route("/run", methods=["POST"])
    @gate_mod.admit_run
    def run():
        return {"ok": True}

    @app.route("/tuple", methods=["POST"])
    @gate_mod.admit_run
    def tuple_():
        return {"error": "bad"}, 400

    @app.route("/stream", methods=["POST"])
    @gate_mod.admit_run
    def stream():
        def gen():
            yield "data: 1\n\n"
            yield "data: 2\n\n"

        return Response(gen(), mimetype="text/event-stream")

    @app.route("/boom", methods=["POST"])
    @gate_mod.admit_run
    def boom():
        raise RuntimeError("boom")

    return app


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setenv("PROJECT_REF", "ref-a")
    g = RunGate(redis_client=fakeredis.FakeStrictRedis(), renew_in_background=False)
    monkeypatch.setattr(gate_mod, "default_gate", g)
    return g


def _limit(monkeypatch, n):
    settings = {"MAX_CONCURRENT_RUNS": n, "RUN_STALE_AFTER_SECONDS": 1800}
    monkeypatch.setattr(gate_mod, "get_setting", settings.__getitem__)


def _service_role():
    return patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": "service", "role": "service_role", "is_service_role": True},
    )


def test_over_the_limit_is_429_with_retry_after(gate, monkeypatch):
    _limit(monkeypatch, 1)
    gate.try_acquire(1)
    with _app().test_client() as c:
        resp = c.post("/run")
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "2"
    assert resp.get_json() == {
        "error": "Too many concurrent runs",
        "active_runs": 1,
        "max_concurrent_runs": 1,
    }


def test_a_plain_response_frees_its_slot_on_close(gate, monkeypatch):
    """The WSGI server closes every response once written (the Werkzeug test
    client only when asked, hence the explicit close)."""
    _limit(monkeypatch, 1)
    with _app().test_client() as c:
        for path, status in (("/run", 200), ("/tuple", 400), ("/run", 200)):
            resp = c.post(path)
            assert resp.status_code == status
            assert gate.active_runs() == 1
            resp.close()
            assert gate.active_runs() == 0


def test_the_slot_is_held_while_streaming_and_freed_on_close(gate, monkeypatch):
    _limit(monkeypatch, 1)
    with _app().test_client() as c:
        resp = c.post("/stream", buffered=False)
        assert gate.active_runs() == 1
        assert c.post("/run").status_code == 429
        assert resp.get_data() == b"data: 1\n\ndata: 2\n\n"
        resp.close()
    assert gate.active_runs() == 0


def test_a_client_disconnect_frees_the_slot(gate, monkeypatch):
    _limit(monkeypatch, 1)
    with _app().test_client() as c:
        resp = c.post("/stream", buffered=False)
        assert gate.active_runs() == 1
        resp.close()  # the stream was never read to its end
    assert gate.active_runs() == 0


def test_an_exception_frees_the_slot(gate, monkeypatch):
    _limit(monkeypatch, 1)
    with _app().test_client() as c:
        assert c.post("/boom").status_code == 500
    assert gate.active_runs() == 0


def test_limit_zero_never_touches_redis(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(
        gate_mod, "default_gate", RunGate(redis_client=client, renew_in_background=False)
    )
    _limit(monkeypatch, 0)
    with _app().test_client() as c:
        assert c.post("/run").status_code == 200
    client.register_script.assert_not_called()


def test_unreadable_settings_admit(gate, monkeypatch):
    def broken(key):
        raise RuntimeError("no app context")

    monkeypatch.setattr(gate_mod, "get_setting", broken)
    with _app().test_client() as c:
        assert c.post("/run").status_code == 200


def test_every_run_entry_point_is_gated():
    from agentic_project_service.routes import agents, orchestrations, workflows

    for view in (
        agents.run_agent,
        agents.run_agent_stream,
        orchestrations.run_orchestration_stream,
        workflows.execute_workflow,
        workflows.execute_workflow_stream,
    ):
        assert getattr(view, "_admits_runs", False), f"{view.__name__} is not behind @admit_run"


def test_the_approve_route_is_not_gated():
    """Resuming a paused run is not a new run; a full project must still
    accept the decision that lets a held run finish."""
    from agentic_project_service.routes import agents

    assert not getattr(agents.approve_run, "_admits_runs", False)


def _full(gate, monkeypatch):
    _limit(monkeypatch, 1)
    gate.try_acquire(1)


def test_agent_run_429_comes_before_the_balance_check(gate, monkeypatch):
    from agentic_project_service.routes import agents as agents_route

    _full(gate, monkeypatch)
    app = Flask(__name__)
    app.register_blueprint(agents_route.agents_bp)
    with (
        patch.object(agents_route.billing, "check_balance") as check_balance,
        _service_role(),
        app.test_client() as c,
    ):
        for path in (f"/api/agents/{AGENT_ID}/run", f"/api/agents/{AGENT_ID}/run/stream"):
            resp = c.post(path, json={"message": "hi"}, headers=HEADERS)
            assert resp.status_code == 429, path
            assert resp.mimetype == "application/json"
    check_balance.assert_not_called()


def test_orchestration_run_429_comes_before_the_balance_check(gate, monkeypatch):
    from agentic_project_service.routes import orchestrations as orchestrations_route

    _full(gate, monkeypatch)
    app = Flask(__name__)
    app.register_blueprint(orchestrations_route.orchestrations_bp)
    with (
        patch.object(orchestrations_route.billing, "check_balance") as check_balance,
        _service_role(),
        app.test_client() as c,
    ):
        resp = c.post(
            f"/api/orchestrations/{ORCH_ID}/run/stream", json={"message": "hi"}, headers=HEADERS
        )
    assert resp.status_code == 429
    check_balance.assert_not_called()


def test_workflow_execute_429_comes_before_the_balance_check(gate, monkeypatch):
    from agentic_project_service.routes import workflows as workflows_route
    from agentic_project_service.services import rate_limit as rl

    _full(gate, monkeypatch)
    monkeypatch.setattr(rl, "external_limiter", MagicMock(try_acquire=lambda *a: None))
    monkeypatch.setattr(rl, "get_setting", lambda key: 120)
    app = Flask(__name__)
    app.register_blueprint(workflows_route.workflows_bp)
    with (
        patch.object(workflows_route.billing, "check_balance") as check_balance,
        _service_role(),
        app.test_client() as c,
    ):
        for path in (
            f"/api/workflows/{WORKFLOW_ID}/execute",
            f"/api/workflows/{WORKFLOW_ID}/execute/stream",
        ):
            resp = c.post(path, json={}, headers=HEADERS)
            assert resp.status_code == 429, path
            assert resp.get_json()["error"] == "Too many concurrent runs"
    check_balance.assert_not_called()


def test_auth_answers_before_the_gate(gate, monkeypatch):
    """A caller that may not run anything is told so, and takes no slot."""
    from agentic_project_service.routes import workflows as workflows_route

    _full(gate, monkeypatch)
    app = Flask(__name__)
    app.register_blueprint(workflows_route.workflows_bp)
    with app.test_client() as c:
        resp = c.post(f"/api/workflows/{WORKFLOW_ID}/execute", json={})
    assert resp.status_code == 401
    assert gate.active_runs() == 1


def _raising_close_app():
    app = Flask(__name__)

    @app.route("/stream", methods=["POST"])
    @gate_mod.admit_run
    def stream():
        def gen():
            try:
                yield "data: 1\n\n"
                yield "data: 2\n\n"
            finally:
                raise RuntimeError("cleanup failed")

        return Response(gen(), mimetype="text/event-stream")

    return app


def test_a_stream_whose_close_raises_still_frees_its_slot(gate, monkeypatch):
    """Werkzeug runs call_on_close callbacks only after the body's close()
    returns, so the release must not depend on that close succeeding."""
    _limit(monkeypatch, 1)
    with _raising_close_app().test_client() as c:
        resp = c.post("/stream", buffered=False)
        assert gate.active_runs() == 1
        with pytest.raises(RuntimeError, match="cleanup failed"):
            resp.close()  # a disconnect: the generator is closed mid-stream
    assert gate.active_runs() == 0
    assert not gate._held


def test_a_plain_response_keeps_its_content_length(gate, monkeypatch):
    _limit(monkeypatch, 1)
    with _app().test_client() as c:
        resp = c.post("/run")
        assert resp.headers.get("Content-Length") == str(len(resp.get_data()))
        resp.close()
    assert gate.active_runs() == 0


def test_a_run_carried_on_after_the_response_keeps_its_slot(gate, monkeypatch):
    """A stream route that hands the run to a thread on disconnect takes a hold
    on the slot first, so the gate caps runs, not open connections."""
    import threading

    _limit(monkeypatch, 1)
    finish = threading.Event()
    finished = threading.Event()
    app = Flask(__name__)

    @app.route("/stream", methods=["POST"])
    @gate_mod.admit_run
    def stream():
        slot = gate_mod.current_slot()

        def gen():
            try:
                yield "data: start\n\n"
                yield "data: more\n\n"
            except GeneratorExit:
                let_go = slot.hold()

                def carry_on():
                    finish.wait(5)
                    let_go()
                    finished.set()

                threading.Thread(target=carry_on, daemon=True).start()
                raise

        return Response(gen(), mimetype="text/event-stream")

    with app.test_client() as c:
        resp = c.post("/stream", buffered=False)
        next(resp.response)  # the client reads "start", then disconnects
        resp.close()
        assert gate.active_runs() == 1
        assert c.post("/stream").status_code == 429
    finish.set()
    assert finished.wait(5)
    assert gate.active_runs() == 0
    assert not gate._held


def test_current_slot_outside_a_gated_request_is_a_no_op():
    gate_mod.current_slot().hold()()
    app = Flask(__name__)
    with app.test_request_context():
        gate_mod.current_slot().hold()()


# --- webhook-triggered workflow runs ---

WEBHOOK_ID = "11111111-1111-1111-1111-111111111111"


def _webhook_session(statements):
    """Answers the block lookup (secret 's3cret') and an armed, undeployed
    workflow; records every statement."""

    def execute(statement, params=None):
        sql = str(statement)
        statements.append(sql)
        result = MagicMock()
        if "workflow_blocks" in sql:
            result.fetchone.return_value = ("block-1", WORKFLOW_ID, {"webhook_secret": "s3cret"})
        elif "SELECT state" in sql:
            from datetime import UTC, datetime, timedelta

            result.fetchone.return_value = ("internal", datetime.now(UTC) + timedelta(minutes=5))
        return result

    session = MagicMock()
    session.execute.side_effect = execute
    return session


def test_a_webhook_run_is_gated_after_its_secret_and_before_its_arm(gate, monkeypatch):
    from agentic_project_service.routes import webhooks as webhooks_route

    _full(gate, monkeypatch)
    app = Flask(__name__)
    app.register_blueprint(webhooks_route.webhooks_bp)
    statements: list[str] = []
    with (
        patch.object(webhooks_route.db, "session", _webhook_session(statements)),
        app.test_client() as c,
    ):
        refused = c.post(
            f"/api/webhooks/{WEBHOOK_ID}", json={}, headers={"Authorization": "Bearer s3cret"}
        )
        unauthorized = c.post(
            f"/api/webhooks/{WEBHOOK_ID}", json={}, headers={"Authorization": "Bearer nope"}
        )
    assert refused.status_code == 429
    assert refused.headers["Retry-After"] == "2"
    assert unauthorized.status_code == 401  # a bad secret is told so, not 429
    # A refused run leaves the single-use arm in place.
    assert not any("webhook_armed_until = NULL" in sql for sql in statements)
    assert gate.active_runs() == 1


def test_the_webhook_run_path_is_gated():
    from agentic_project_service.routes import webhooks as webhooks_route

    assert getattr(webhooks_route._run_triggered_workflow, "_admits_runs", False)
