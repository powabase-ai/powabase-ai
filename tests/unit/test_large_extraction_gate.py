"""The admission limit on concurrent extractions of large source files, and the
record of what was in flight when a worker died.

Runs against fakeredis with Lua enabled, the same way the rate limiter's tests
do: the semaphore is a server-side script, so a client-side fake of individual
commands would not exercise it.
"""

import logging
import time
from unittest.mock import MagicMock

import fakeredis
import pytest

from agentic_project_service.services import extraction_gate as gate_mod
from agentic_project_service.services.extraction_gate import (
    DEFAULT_LARGE_FILE_BYTES,
    DEFAULT_MAX_CONCURRENT_LARGE,
    ExtractionAttempts,
    LargeExtractionGate,
    is_large_file,
    large_file_threshold,
    max_concurrent_large_extractions,
)


class _Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def gate(client, clock):
    return LargeExtractionGate(redis_client=client, clock=clock, heartbeat=False)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("EXTRACTION_LARGE_FILE_BYTES", raising=False)
        monkeypatch.delenv("EXTRACTION_LARGE_MAX_CONCURRENT", raising=False)
        assert DEFAULT_LARGE_FILE_BYTES == 52428800
        assert DEFAULT_MAX_CONCURRENT_LARGE == 1
        assert large_file_threshold() == 52428800
        assert max_concurrent_large_extractions() == 1

    def test_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_LARGE_FILE_BYTES", "1000")
        monkeypatch.setenv("EXTRACTION_LARGE_MAX_CONCURRENT", "3")
        assert large_file_threshold() == 1000
        assert max_concurrent_large_extractions() == 3

    @pytest.mark.parametrize("value", ["", "lots", "-5", "0"])
    def test_invalid_values_fall_back_to_the_default(self, monkeypatch, value):
        monkeypatch.setenv("EXTRACTION_LARGE_FILE_BYTES", value)
        monkeypatch.setenv("EXTRACTION_LARGE_MAX_CONCURRENT", value)
        assert large_file_threshold() == 52428800
        assert max_concurrent_large_extractions() == 1

    def test_large_means_over_the_threshold(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_LARGE_FILE_BYTES", "100")
        assert not is_large_file(0)
        assert not is_large_file(100)
        assert is_large_file(101)

    def test_a_file_of_unknown_size_is_treated_as_large(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_LARGE_FILE_BYTES", "100")
        assert is_large_file(None)


# ---------------------------------------------------------------------------
# Semaphore
# ---------------------------------------------------------------------------


class TestAcquire:
    def test_takes_slots_up_to_the_limit(self, gate, monkeypatch):
        monkeypatch.setenv("EXTRACTION_LARGE_MAX_CONCURRENT", "2")
        first = gate.try_acquire("task-1")
        second = gate.try_acquire("task-2")
        assert first is not None and second is not None
        assert first.token != second.token
        assert gate.try_acquire("task-3") is None

    def test_default_limit_is_one(self, gate, monkeypatch):
        monkeypatch.delenv("EXTRACTION_LARGE_MAX_CONCURRENT", raising=False)
        assert gate.try_acquire("task-1") is not None
        assert gate.try_acquire("task-2") is None

    def test_a_redelivered_task_does_not_share_its_old_slot(self, gate, monkeypatch):
        """Tokens are per attempt, so a duplicate delivery of a task that is
        still running cannot ride on the running attempt's slot."""
        monkeypatch.setenv("EXTRACTION_LARGE_MAX_CONCURRENT", "2")
        first = gate.try_acquire("task-1")
        second = gate.try_acquire("task-1")
        assert first is not None and second is not None
        assert first.token != second.token
        assert gate.try_acquire("task-2") is None
        first.release()
        assert gate.is_live(second.token)

    def test_release_frees_the_slot(self, gate):
        slot = gate.try_acquire("task-1")
        slot.release()
        assert gate.try_acquire("task-2") is not None

    def test_release_is_idempotent(self, gate, monkeypatch):
        monkeypatch.setenv("EXTRACTION_LARGE_MAX_CONCURRENT", "2")
        slot = gate.try_acquire("task-1")
        other = gate.try_acquire("task-2")
        slot.release()
        slot.release()
        assert gate.is_live(other.token)

    def test_projects_are_independent(self, gate, monkeypatch):
        monkeypatch.setenv("PROJECT_REF", "proj-a")
        assert gate.try_acquire("task-1") is not None
        monkeypatch.setenv("PROJECT_REF", "proj-b")
        assert gate.try_acquire("task-1") is not None

    def test_keys_carry_the_project_ref(self, gate, client, monkeypatch):
        monkeypatch.setenv("PROJECT_REF", "proj-a")
        gate.try_acquire("task-1")
        assert any(b"proj-a" in key for key in client.keys("*"))

    def test_the_slot_key_expires_on_its_own(self, gate, client):
        gate.try_acquire("task-1")
        (key,) = client.keys("*large*")
        assert 0 < client.ttl(key) <= 2 * gate_mod.LEASE_SECONDS


class TestLease:
    def test_a_crashed_holder_stops_blocking_once_its_lease_expires(self, gate, clock):
        crashed = gate.try_acquire("task-1")
        # Never released: the process holding it died.
        assert gate.try_acquire("task-2") is None
        clock.now += gate_mod.LEASE_SECONDS - 1
        assert gate.try_acquire("task-2") is None
        clock.now += 2
        assert gate.try_acquire("task-2") is not None
        assert not gate.is_live(crashed.token)

    def test_renewal_keeps_a_long_extraction_holding_its_slot(self, gate, clock):
        slot = gate.try_acquire("task-1")
        for _ in range(10):
            clock.now += gate_mod.LEASE_SECONDS - 10
            assert gate.renew(slot.token)
        assert gate.is_live(slot.token)
        assert gate.try_acquire("task-2") is None

    def test_an_expired_lease_is_not_renewed(self, gate, clock):
        slot = gate.try_acquire("task-1")
        clock.now += gate_mod.LEASE_SECONDS + 1
        assert not gate.renew(slot.token)
        assert not gate.is_live(slot.token)

    def test_the_heartbeat_renews_the_lease_and_stops_on_release(self, client, clock, monkeypatch):
        monkeypatch.setattr(gate_mod, "RENEW_SECONDS", 0.01)
        gate = LargeExtractionGate(redis_client=client, clock=clock, heartbeat=True)
        renewed = []
        real_renew = gate.renew

        def counting_renew(token):
            renewed.append(token)
            return real_renew(token)

        gate.renew = counting_renew
        slot = gate.try_acquire("task-1")
        deadline = time.monotonic() + 2
        while not renewed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert renewed and renewed[0] == slot.token
        heartbeat = slot._heartbeat
        assert heartbeat is not None and heartbeat.is_alive()
        slot.release()
        assert not heartbeat.is_alive()


class TestRedisUnavailable:
    def test_acquire_fails_open_with_a_warning(self, clock, caplog):
        broken = MagicMock()
        broken.register_script.side_effect = ConnectionError("redis down")
        gate = LargeExtractionGate(redis_client=broken, clock=clock, heartbeat=False)
        with caplog.at_level(logging.WARNING):
            slot = gate.try_acquire("task-1")
        assert slot is not None
        assert slot.token is None
        slot.release()
        assert "unavailable" in caplog.text

    def test_liveness_is_unknown_so_reported_not_live(self, clock):
        broken = MagicMock()
        broken.register_script.side_effect = ConnectionError("redis down")
        gate = LargeExtractionGate(redis_client=broken, clock=clock, heartbeat=False)
        assert gate.is_live("anything") is False


# ---------------------------------------------------------------------------
# What was in flight when a worker died
# ---------------------------------------------------------------------------


@pytest.fixture
def attempts_factory(client):
    def make(incarnation):
        return ExtractionAttempts(redis_client=client, incarnation=incarnation)

    return make


class TestExtractionAttempts:
    def test_nothing_is_known_about_a_task_that_never_started(self, attempts_factory):
        assert attempts_factory("worker-a").previous("task-1") is None

    def test_a_finished_attempt_leaves_no_record(self, attempts_factory):
        attempts = attempts_factory("worker-a")
        attempts.begin("task-1", 10, "slot-token")
        attempts.end("task-1")
        assert attempts.previous("task-1") is None

    def test_the_largest_file_in_flight_is_the_plausible_cause(self, attempts_factory):
        dead = attempts_factory("worker-a")
        dead.begin("big", 400_000_000, "slot-token")
        dead.begin("small-1", 20_000, None)
        dead.begin("small-2", 30_000, None)
        # worker-a is killed; nothing calls end(). The redeliveries run elsewhere.
        survivor = attempts_factory("worker-b")

        big = survivor.previous("big")
        assert big.slot_token == "slot-token"
        assert big.size == 400_000_000
        assert big.plausible_cause

        for task_id in ("small-1", "small-2"):
            previous = survivor.previous(task_id)
            assert previous.largest_in_flight == 400_000_000
            assert not previous.plausible_cause

    def test_a_task_alone_in_its_worker_is_the_plausible_cause(self, attempts_factory):
        dead = attempts_factory("worker-a")
        dead.begin("only", 10, None)
        assert attempts_factory("worker-b").previous("only").plausible_cause

    def test_equal_sizes_are_all_plausible(self, attempts_factory):
        dead = attempts_factory("worker-a")
        dead.begin("a", 10, None)
        dead.begin("b", 10, None)
        survivor = attempts_factory("worker-b")
        assert survivor.previous("a").plausible_cause
        assert survivor.previous("b").plausible_cause

    def test_unknown_size_counts_as_the_largest(self, attempts_factory):
        dead = attempts_factory("worker-a")
        dead.begin("unknown", None, None)
        dead.begin("big", 400_000_000, None)
        survivor = attempts_factory("worker-b")
        assert survivor.previous("unknown").plausible_cause
        assert not survivor.previous("big").plausible_cause

    def test_tasks_that_finished_before_the_kill_are_not_suspects(self, attempts_factory):
        dead = attempts_factory("worker-a")
        dead.begin("big-but-done", 400_000_000, None)
        dead.begin("small", 10, None)
        dead.end("big-but-done")
        assert attempts_factory("worker-b").previous("small").plausible_cause

    def test_incarnations_do_not_mix(self, attempts_factory):
        """A big file running on a healthy worker is no suspect for a kill of
        another worker."""
        attempts_factory("worker-b").begin("big-elsewhere", 400_000_000, None)
        attempts_factory("worker-a").begin("small", 10, None)
        assert attempts_factory("worker-c").previous("small").plausible_cause

    def test_the_latest_attempt_of_a_task_wins(self, attempts_factory):
        first = attempts_factory("worker-a")
        first.begin("big", 400_000_000, None)
        first.begin("task", 10, None)
        second = attempts_factory("worker-b")
        second.begin("task", 10, None)  # redelivered, then killed again, alone
        assert attempts_factory("worker-c").previous("task").plausible_cause

    def test_records_expire_on_their_own(self, attempts_factory, client):
        attempts_factory("worker-a").begin("task", 10, None)
        keys = client.keys("*")
        assert keys
        assert all(client.ttl(key) > 0 for key in keys)

    def test_the_record_outlives_the_broker_redelivery_delay(self, attempts_factory, client):
        """Redelivery waits for the broker's visibility timeout, which is the
        task time limit; a record gone by then would charge every task."""
        attempts_factory("worker-a").begin("task", 10, None)
        assert all(client.ttl(key) > 21600 for key in client.keys("*"))

    def test_redis_failures_lose_the_record_but_never_raise(self):
        broken = MagicMock()
        broken.pipeline.side_effect = ConnectionError("redis down")
        broken.get.side_effect = ConnectionError("redis down")
        attempts = ExtractionAttempts(redis_client=broken, incarnation="worker-a")
        attempts.begin("task", 10, None)
        attempts.end("task")
        assert attempts.previous("task") is None

    def test_the_default_incarnation_is_unique_per_process(self):
        assert ExtractionAttempts(redis_client=MagicMock()).incarnation == (
            ExtractionAttempts(redis_client=MagicMock()).incarnation
        )
        assert gate_mod.process_incarnation() == gate_mod.process_incarnation()
