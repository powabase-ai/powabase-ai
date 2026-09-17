"""The admission limit on concurrent extractions of large source files, and the
record of what was in flight when a worker died.

Runs against fakeredis with Lua enabled, the same way the rate limiter's tests
do: the semaphore is a server-side script, so a client-side fake of individual
commands would not exercise it. Set ``TEST_REDIS_URL`` (a database this suite
may flush, e.g. ``redis://localhost:6379/15``) to run every test here against a
real Redis as well.
"""

import logging
import os
import socket
import threading
import time
from unittest.mock import MagicMock

import fakeredis
import pytest
import redis

from agentic_project_service.services import extraction_gate as gate_mod
from agentic_project_service.services.extraction_gate import (
    WAITING_STALE_SECONDS,
    WaitingExtractions,
    DEFAULT_LARGE_FILE_BYTES,
    DEFAULT_MAX_CONCURRENT_LARGE,
    ExtractionAttempts,
    LargeExtractionGate,
    LargeExtractionSlot,
    Renewal,
    is_large_file,
    large_file_threshold,
    max_concurrent_large_extractions,
    max_hold_seconds,
)

_REAL_REDIS_URL = os.getenv("TEST_REDIS_URL")


@pytest.fixture(params=["fakeredis", "redis"])
def client(request):
    if request.param == "fakeredis":
        yield fakeredis.FakeStrictRedis()
        return
    if not _REAL_REDIS_URL:
        pytest.skip("TEST_REDIS_URL not set")
    real = redis.from_url(_REAL_REDIS_URL)
    real.flushdb()
    yield real
    real.flushdb()


@pytest.fixture
def gate(client):
    return LargeExtractionGate(redis_client=client, heartbeat=False)


@pytest.fixture
def short_lease(monkeypatch):
    """Leases short enough to watch expire."""
    monkeypatch.setattr(gate_mod, "LEASE_SECONDS", 0.5)
    monkeypatch.setattr(gate_mod, "RENEW_SECONDS", 0.1)


def _wait_until(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


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

    def test_lease_timing_is_pinned(self):
        """A lease of minutes frees a killed holder's slot quickly; renewing
        several times per lease survives a few failed renewals in a row."""
        assert gate_mod.LEASE_SECONDS == 300
        assert gate_mod.RENEW_SECONDS == 60
        assert gate_mod.RENEW_SECONDS * 3 <= gate_mod.LEASE_SECONDS

    def test_a_slot_is_held_a_little_less_than_the_task_time_limit(self, monkeypatch):
        """The broker redelivers an unfinished task after the task time limit.
        The lease must have fully lapsed by then, so the redelivery is judged
        as an interruption instead of first waiting on a lease that is about
        to vanish and then running beside the original."""
        monkeypatch.delenv("CELERY_TASK_TIME_LIMIT", raising=False)
        assert max_hold_seconds() == 21600 - 300 - 60
        monkeypatch.setenv("CELERY_TASK_TIME_LIMIT", "7200")
        assert max_hold_seconds() == 7200 - 300 - 60
        assert max_hold_seconds() + gate_mod.LEASE_SECONDS < 7200


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

    def test_lease_times_come_from_redis_not_the_workers_clock(self, gate, monkeypatch):
        """Two pods whose clocks disagree by more than a lease must not prune
        each other's live slots."""
        fast = MagicMock(wraps=time)
        fast.time.return_value = time.time() + 100_000
        slow = MagicMock(wraps=time)
        slow.time.return_value = time.time() - 100_000

        monkeypatch.setattr(gate_mod, "time", slow)
        first = gate.try_acquire("task-1")
        monkeypatch.setattr(gate_mod, "time", fast)
        assert gate.try_acquire("task-2") is None
        assert gate.is_live(first.token)
        assert gate.renew(first.token) is Renewal.RENEWED


class TestLease:
    def test_a_crashed_holder_stops_blocking_once_its_lease_expires(self, gate, short_lease):
        crashed = gate.try_acquire("task-1")
        # Never released: the process holding it died.
        assert gate.try_acquire("task-2") is None
        time.sleep(0.7)
        assert gate.try_acquire("task-2") is not None
        assert not gate.is_live(crashed.token)

    def test_renewal_keeps_a_long_extraction_holding_its_slot(self, gate, short_lease):
        slot = gate.try_acquire("task-1")
        for _ in range(4):
            time.sleep(0.3)
            assert gate.renew(slot.token) is Renewal.RENEWED
        assert gate.is_live(slot.token)
        assert gate.try_acquire("task-2") is None

    def test_an_expired_lease_is_reported_lost_and_not_revived(self, gate, short_lease):
        slot = gate.try_acquire("task-1")
        time.sleep(0.7)
        assert gate.renew(slot.token) is Renewal.LOST
        assert not gate.is_live(slot.token)

    def test_a_redis_error_is_not_mistaken_for_a_lost_lease(self):
        broken = MagicMock()
        broken.register_script.side_effect = ConnectionError("redis down")
        gate = LargeExtractionGate(redis_client=broken, heartbeat=False)
        assert gate.renew("token") is Renewal.ERROR

    def test_the_heartbeat_renews_the_lease_and_stops_on_release(self, client, short_lease):
        gate = LargeExtractionGate(redis_client=client, heartbeat=True)
        renewed = []
        real_renew = gate.renew

        def counting_renew(token):
            renewed.append(token)
            return real_renew(token)

        gate.renew = counting_renew
        slot = gate.try_acquire("task-1")
        assert _wait_until(lambda: len(renewed) >= 2)
        assert renewed[0] == slot.token
        heartbeat = slot._heartbeat
        assert heartbeat is not None and heartbeat.is_alive()
        slot.release()
        assert not heartbeat.is_alive()

    def test_a_failed_renewal_does_not_let_a_second_holder_in(self, client, short_lease):
        """A Redis blip on one renewal must not end the heartbeat: the lease
        would lapse while the first extraction is still running."""
        gate = LargeExtractionGate(redis_client=client, heartbeat=True)
        calls = []
        real_script = gate._script

        def flaky_script(name, source):
            script = real_script(name, source)
            if name != "renew":
                return script

            def run(*a, **kw):
                calls.append(name)
                if len(calls) == 1:
                    raise redis.exceptions.ConnectionError("connection reset")
                return script(*a, **kw)

            return run

        gate._script = flaky_script
        holder = gate.try_acquire("task-1")
        try:
            other = LargeExtractionGate(redis_client=client, heartbeat=False)
            deadline = time.monotonic() + 1.5  # three leases
            while time.monotonic() < deadline:
                assert other.try_acquire("task-2") is None
                time.sleep(0.05)
            assert len(calls) >= 3
            assert holder._heartbeat.is_alive()
        finally:
            holder.release()

    def test_the_heartbeat_stops_when_the_lease_is_gone(self, client, short_lease):
        gate = LargeExtractionGate(redis_client=client, heartbeat=True)
        slot = gate.try_acquire("task-1")
        client.delete(gate._key())
        assert _wait_until(lambda: not slot._heartbeat.is_alive())
        slot.release()

    def test_a_hung_holder_frees_its_slot_after_the_task_time_limit(
        self, client, short_lease, monkeypatch
    ):
        monkeypatch.setattr(gate_mod, "max_hold_seconds", lambda: 0.3)
        gate = LargeExtractionGate(redis_client=client, heartbeat=True)
        hung = gate.try_acquire("task-1")  # never released
        other = LargeExtractionGate(redis_client=client, heartbeat=False)
        assert other.try_acquire("task-2") is None
        assert _wait_until(lambda: not hung._heartbeat.is_alive(), timeout=2)
        assert _wait_until(lambda: other.try_acquire("task-2") is not None, timeout=2)
        hung.release()


class TestRedisUnavailable:
    def test_acquire_fails_open_with_a_warning_and_a_metric(self, caplog):
        broken = MagicMock()
        broken.register_script.side_effect = ConnectionError("redis down")
        gate = LargeExtractionGate(redis_client=broken, heartbeat=False)
        before = gate_mod.fail_open_count("acquire")
        with caplog.at_level(logging.WARNING):
            slot = gate.try_acquire("task-1")
        assert slot is not None
        assert slot.token is None
        slot.release()
        assert "unavailable" in caplog.text
        assert gate_mod.fail_open_count("acquire") == before + 1

    def test_liveness_is_unknown_so_reported_not_live(self):
        broken = MagicMock()
        broken.register_script.side_effect = ConnectionError("redis down")
        gate = LargeExtractionGate(redis_client=broken, heartbeat=False)
        assert gate.is_live("anything") is False


class _SilentServer:
    """Accepts connections and never answers: a half-open Redis."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.conns = []
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.conns.append(conn)

    def close(self):
        for conn in self.conns:
            conn.close()
        self.sock.close()


class TestBrokerClient:
    def test_the_client_has_socket_timeouts(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            gate_mod.redis, "from_url", lambda url, **kw: seen.update(kw) or MagicMock()
        )
        gate_mod._broker_client()
        assert seen["socket_timeout"] == gate_mod.REDIS_SOCKET_TIMEOUT_SECONDS
        assert seen["socket_connect_timeout"] == gate_mod.REDIS_SOCKET_TIMEOUT_SECONDS
        assert 0 < gate_mod.REDIS_SOCKET_TIMEOUT_SECONDS < gate_mod.RENEW_SECONDS

    def test_release_and_end_cannot_hang_on_an_unresponsive_redis(self, monkeypatch):
        server = _SilentServer()
        monkeypatch.setattr(gate_mod, "REDIS_SOCKET_TIMEOUT_SECONDS", 0.3)
        monkeypatch.setenv("CELERY_BROKER_URL", f"redis://127.0.0.1:{server.port}/0")
        finished = []

        def cleanup():
            gate = LargeExtractionGate(heartbeat=False)
            LargeExtractionSlot(gate, "token", heartbeat=False).release()
            ExtractionAttempts(incarnation="worker-a").end("task-1")
            finished.append(True)

        worker = threading.Thread(target=cleanup, daemon=True)
        try:
            worker.start()
            worker.join(timeout=5)
            assert finished == [True]
        finally:
            server.close()


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


# ---------------------------------------------------------------------------
# Deliveries sitting in a worker, waiting to run
# ---------------------------------------------------------------------------


class _ServerClock:
    """Shifts what the store believes Redis's clock says, to age records."""

    def __init__(self, client, monkeypatch):
        self.offset = 0.0
        real_time = client.time

        def shifted():
            seconds, micros = real_time()
            return int(seconds + self.offset), micros

        monkeypatch.setattr(client, "time", shifted)


class TestWaitingExtractions:
    DISPATCH = {"args": ["src-1", "sources"], "kwargs": {"source_size": 10}, "retries": 1}

    def test_the_stale_threshold_is_pinned(self):
        assert WAITING_STALE_SECONDS == 600
        assert WAITING_STALE_SECONDS >= 5 * 60  # several requeue intervals

    def test_a_recorded_wait_can_be_read_back_and_cleared(self, client):
        waiting = WaitingExtractions(redis_client=client)
        waiting.touch("task-1", "src-1", self.DISPATCH)
        assert waiting.get("task-1") == {"source_id": "src-1", "dispatch": self.DISPATCH}
        waiting.clear("task-1")
        assert waiting.get("task-1") is None
        assert waiting.stale(0) == []

    def test_only_waits_not_refreshed_recently_are_stale(self, client, monkeypatch):
        clock = _ServerClock(client, monkeypatch)
        waiting = WaitingExtractions(redis_client=client)
        waiting.touch("old", "src-1", self.DISPATCH)
        clock.offset = WAITING_STALE_SECONDS + 5
        waiting.touch("fresh", "src-2", self.DISPATCH)
        assert [task_id for task_id, _ in waiting.stale(WAITING_STALE_SECONDS)] == ["old"]

    def test_touching_again_restarts_the_clock(self, client, monkeypatch):
        clock = _ServerClock(client, monkeypatch)
        waiting = WaitingExtractions(redis_client=client)
        waiting.touch("task-1", "src-1", self.DISPATCH)
        clock.offset = WAITING_STALE_SECONDS + 5
        waiting.touch("task-1", "src-1", self.DISPATCH)
        assert waiting.stale(WAITING_STALE_SECONDS) == []

    def test_projects_are_independent(self, client, monkeypatch):
        waiting = WaitingExtractions(redis_client=client)
        monkeypatch.setenv("PROJECT_REF", "proj-a")
        waiting.touch("task-1", "src-1", self.DISPATCH)
        monkeypatch.setenv("PROJECT_REF", "proj-b")
        assert waiting.get("task-1") is None
        assert waiting.stale(0) == []

    def test_records_expire_on_their_own(self, client):
        WaitingExtractions(redis_client=client).touch("task-1", "src-1", self.DISPATCH)
        assert all(client.ttl(key) > 0 for key in client.keys("*"))

    def test_redis_failures_never_raise(self):
        broken = MagicMock()
        broken.pipeline.side_effect = ConnectionError("redis down")
        broken.time.side_effect = ConnectionError("redis down")
        broken.get.side_effect = ConnectionError("redis down")
        broken.zrangebyscore.side_effect = ConnectionError("redis down")
        waiting = WaitingExtractions(redis_client=broken)
        waiting.touch("task-1", "src-1", self.DISPATCH)
        waiting.clear("task-1")
        assert waiting.get("task-1") is None
        assert waiting.stale(0) == []
