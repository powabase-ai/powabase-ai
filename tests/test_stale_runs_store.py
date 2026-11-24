"""The stale-run sweep and the session list's metadata, against Postgres.

The unit tier checks these statements only by substring against mocks: the
sweep's candidate SELECT (an interval built from a bound number of seconds),
its ``id = ANY(CAST(:ids AS uuid[]))`` UPDATE, and the session list's
``GROUP BY s.metadata``. A cast, a column or a grouping that Postgres rejects
would pass there and fail on every project.
"""

import uuid

import fakeredis
import pytest
from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.services.run_registry import RunRegistry
from agentic_project_service.tasks import cleanup


@pytest.fixture
def broker(monkeypatch):
    """The Redis the sweep reads liveness records from, and its clock."""
    monkeypatch.setenv("PROJECT_REF", "ref-store")
    client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr(cleanup, "broker_client", lambda: client)
    clock = {"now": 1_000_000}
    monkeypatch.setattr(cleanup, "_redis_now", lambda c: clock["now"])
    return client, clock


def _run(table: str, run_id: str, started_seconds_ago: int, status: str = "running") -> str:
    row_id = str(uuid.uuid4())
    db.session.execute(
        text(
            f'INSERT INTO "ai".{table} (id, run_id, status, started_at) '
            "VALUES (:id, :run_id, :status, NOW() - (:ago * INTERVAL '1 second'))"
        ),
        {"id": row_id, "run_id": run_id, "status": status, "ago": started_seconds_ago},
    )
    db.session.commit()
    return row_id


def _row(table: str, row_id: str):
    return db.session.execute(
        text(f'SELECT status, error, completed_at FROM "ai".{table} WHERE id = :id'),
        {"id": row_id},
    ).one()


def test_the_sweep_fails_only_old_runs_without_a_live_process(app, broker):
    client, clock = broker
    with app.app_context():
        dead = _run("agent_runs", "run_dead", 4000)
        live = _run("agent_runs", "run_live", 4000)
        young = _run("agent_runs", "run_young", 60)
        done = _run("agent_runs", "run_done", 4000, status="completed")
        orch_dead = _run("orchestration_runs", "orch_run_dead", 4000)
        orch_live = _run("orchestration_runs", "orch_run_live", 4000)
        client.set(RunRegistry.alive_key("run_live"), "pod-a")
        client.set(RunRegistry.alive_key("orch_run_live"), "pod-a")

        # The default cutoff (1800 s), read through get_setting_strict.
        assert cleanup.run_stale_cleanup() == {"agent_runs": 0, "orchestration_runs": 0}
        clock["now"] += cleanup.STALE_TICK_INTERVAL
        assert cleanup.run_stale_cleanup() == {"agent_runs": 1, "orchestration_runs": 1}

        for table, row_id in (("agent_runs", dead), ("orchestration_runs", orch_dead)):
            status, error, completed_at = _row(table, row_id)
            assert (status, error) == ("failed", cleanup.INTERRUPTED_ERROR)
            assert completed_at is not None
        for table, row_id in (
            ("agent_runs", live),
            ("agent_runs", young),
            ("orchestration_runs", orch_live),
        ):
            assert _row(table, row_id)[0] == "running"
        assert _row("agent_runs", done)[0] == "completed"


def test_an_override_of_the_cutoff_is_honoured(app, broker):
    _, clock = broker
    with app.app_context():
        db.session.execute(
            text(
                'INSERT INTO "ai".project_settings (key, value) '
                "VALUES ('RUN_STALE_AFTER_SECONDS', '7200')"
            )
        )
        db.session.commit()
        try:
            row_id = _run("agent_runs", "run_long", 4000)
            cleanup.run_stale_cleanup()
            clock["now"] += cleanup.STALE_TICK_INTERVAL
            assert cleanup.run_stale_cleanup() == {"agent_runs": 0, "orchestration_runs": 0}
            assert _row("agent_runs", row_id)[0] == "running"
        finally:
            db.session.execute(
                text("DELETE FROM \"ai\".project_settings WHERE key = 'RUN_STALE_AFTER_SECONDS'")
            )
            db.session.commit()


def test_the_session_list_groups_by_metadata_and_returns_it(app, client, mock_auth, auth_headers):
    agent_id = str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                'INSERT INTO "ai".agents (id, name, model, system_prompt) '
                "VALUES (:id, 'meta', 'gpt-4o-mini', '')"
            ),
            {"id": agent_id},
        )
        sessions = {}
        for name, metadata, runs in (
            ("sess_a", '{"topic": "billing", "n": 1}', 2),
            ("sess_b", "{}", 1),
            ("sess_c", '{"topic": "billing", "n": 1}', 0),
        ):
            session_uuid = str(uuid.uuid4())
            sessions[name] = session_uuid
            db.session.execute(
                text(
                    'INSERT INTO "ai".agent_sessions (id, session_id, agent_id, metadata) '
                    "VALUES (:id, :sid, :agent_id, CAST(:metadata AS jsonb))"
                ),
                {"id": session_uuid, "sid": name, "agent_id": agent_id, "metadata": metadata},
            )
            for i in range(runs):
                db.session.execute(
                    text(
                        'INSERT INTO "ai".agent_runs (id, session_id, run_id, status) '
                        "VALUES (:id, :session_id, :run_id, 'completed')"
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "session_id": session_uuid,
                        "run_id": f"run_{name}_{i}",
                    },
                )
        db.session.commit()

    resp = client.get(f"/api/agents/{agent_id}/sessions", headers=auth_headers)
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    by_id = {s["session_id"]: s for s in body["sessions"]}
    assert body["total"] == 3
    # Two sessions with equal metadata stay two rows: grouping is by session.
    assert by_id["sess_a"]["metadata"] == {"topic": "billing", "n": 1}
    assert by_id["sess_a"]["run_count"] == 2
    assert by_id["sess_c"]["metadata"] == {"topic": "billing", "n": 1}
    assert by_id["sess_c"]["run_count"] == 0
    assert by_id["sess_b"]["metadata"] == {}

    resp = client.get(f"/api/agents/{agent_id}/sessions?min_runs=1", headers=auth_headers)
    assert resp.status_code == 200
    assert sorted(s["session_id"] for s in resp.get_json()["sessions"]) == ["sess_a", "sess_b"]
