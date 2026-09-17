"""Recovering waiting extraction deliveries lost with a killed worker.

A task waiting for a large-extraction slot, or for a retry, sits in a worker
as a countdown message that the broker only hands back after its visibility
timeout (the task time limit, 6 h by default). Each wake-up refreshes a record
of the wait in Redis; a record nobody refreshed for WAITING_STALE_SECONDS
belongs to a message that died with its worker, and is dispatched again under
the same task id.
"""

from unittest.mock import MagicMock

import fakeredis
import pytest

from agentic_project_service.services.extraction_gate import (
    ExtractionAttempts,
    LargeExtractionGate,
    WaitingExtractions,
)
from agentic_project_service.tasks import extraction as ext_mod

DISPATCH = {
    "args": ["src-1", "sources"],
    "kwargs": {"extraction_model": "auto", "source_size": 400_000_000},
    "retries": 1,
}


@pytest.fixture
def env(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    waiting = WaitingExtractions(redis_client=client)
    gate = LargeExtractionGate(redis_client=client, heartbeat=False)
    attempts = ExtractionAttempts(redis_client=client, incarnation="worker-b")
    monkeypatch.setattr(ext_mod, "waiting_extractions", waiting)
    monkeypatch.setattr(ext_mod, "large_extraction_gate", gate)
    monkeypatch.setattr(ext_mod, "extraction_attempts", attempts)
    sent = []
    monkeypatch.setattr(ext_mod.extract_source, "apply_async", lambda *a, **kw: sent.append(kw))
    sources = {}
    monkeypatch.setattr(ext_mod, "get_source", lambda sid: sources.get(sid))
    offset = {"s": 0}
    real_time = client.time
    monkeypatch.setattr(
        client, "time", lambda: (lambda t: (int(t[0] + offset["s"]), t[1]))(real_time())
    )
    ns = MagicMock()
    ns.client, ns.waiting, ns.gate, ns.attempts = client, waiting, gate, attempts
    ns.sent, ns.sources, ns.offset = sent, sources, offset
    return ns


def _source(status="pending", owner="task-1"):
    return {"id": "src-1", "extraction_status": status, "celery_task_id": owner}


def _age(env):
    env.offset["s"] += ext_mod.WAITING_STALE_SECONDS + 5


def test_a_stale_wait_is_dispatched_again_under_the_same_task_id(env):
    env.sources["src-1"] = _source()
    env.waiting.touch("task-1", "src-1", DISPATCH)
    _age(env)

    result = ext_mod.recover_stranded_extractions.run()

    assert result["recovered"] == 1
    assert env.sent == [
        {
            "args": DISPATCH["args"],
            "kwargs": DISPATCH["kwargs"],
            "task_id": "task-1",
            "retries": 1,
        }
    ]
    # The clock restarts, so the next sweep does not dispatch it again.
    assert env.waiting.stale(ext_mod.WAITING_STALE_SECONDS) == []


def test_a_redelivered_task_waiting_as_extracting_is_recovered_too(env):
    env.sources["src-1"] = _source(status="extracting")
    env.waiting.touch("task-1", "src-1", DISPATCH)
    _age(env)

    assert ext_mod.recover_stranded_extractions.run()["recovered"] == 1


def test_a_recent_wait_is_left_alone(env):
    env.sources["src-1"] = _source()
    env.waiting.touch("task-1", "src-1", DISPATCH)

    assert ext_mod.recover_stranded_extractions.run()["recovered"] == 0
    assert env.sent == []


@pytest.mark.parametrize(
    "source",
    [
        None,
        _source(owner="task-2"),
        _source(status="extracted"),
        _source(status="failed"),
        _source(status="cancelled"),
    ],
    ids=["deleted", "re-dispatched", "extracted", "failed", "cancelled"],
)
def test_a_wait_whose_source_moved_on_is_dropped(env, source):
    if source is not None:
        env.sources["src-1"] = source
    env.waiting.touch("task-1", "src-1", DISPATCH)
    _age(env)

    assert ext_mod.recover_stranded_extractions.run()["recovered"] == 0
    assert env.sent == []
    assert env.waiting.get("task-1") is None


def test_a_task_whose_attempt_still_holds_a_live_lease_is_not_dispatched(env):
    env.sources["src-1"] = _source(status="extracting")
    slot = env.gate.try_acquire("task-1")
    env.attempts.begin("task-1", 400_000_000, slot.token)
    env.waiting.touch("task-1", "src-1", DISPATCH)
    _age(env)

    assert ext_mod.recover_stranded_extractions.run()["recovered"] == 0
    assert env.sent == []


def test_every_delivery_clears_its_wait_record(env, monkeypatch, mock_db_session):
    """Only a message that never ran again keeps a stale record."""
    env.sources["src-1"] = {
        **_source(status="extracted"),
        "name": "a.pdf",
        "file_type": "application/pdf",
        "storage_path": "sources/src-1/a.pdf",
    }
    env.waiting.touch("task-1", "src-1", DISPATCH)
    ext_mod.extract_source.push_request(id="task-1")
    try:
        ext_mod.extract_source.run(source_id="src-1", bucket_id="sources")
    finally:
        ext_mod.extract_source.pop_request()

    assert env.waiting.get("task-1") is None


def test_the_sweep_reschedules_itself_while_waits_remain(env, monkeypatch):
    scheduled = []
    monkeypatch.setattr(
        ext_mod.recover_stranded_extractions,
        "apply_async",
        lambda *a, **kw: scheduled.append(kw),
    )
    env.sources["src-1"] = _source()
    env.waiting.touch("task-1", "src-1", DISPATCH)

    ext_mod.recover_stranded_extractions.run()

    assert scheduled == [{"countdown": ext_mod.WAITING_STALE_SECONDS + 60}]


def test_worker_start_schedules_one_sweep_across_workers(env, monkeypatch):
    scheduled = []
    monkeypatch.setattr(
        ext_mod.recover_stranded_extractions,
        "apply_async",
        lambda *a, **kw: scheduled.append(kw),
    )

    ext_mod.schedule_stranded_extraction_recovery()
    ext_mod.schedule_stranded_extraction_recovery()  # a second worker starting

    assert scheduled == [{"countdown": ext_mod.WAITING_STALE_SECONDS + 60}]


def test_worker_ready_triggers_the_schedule(monkeypatch):
    from agentic_project_service import celery as celery_mod

    called = []
    monkeypatch.setattr(ext_mod, "schedule_stranded_extraction_recovery", lambda: called.append(1))
    celery_mod.seed_stranded_extraction_recovery()

    assert called == [1]
