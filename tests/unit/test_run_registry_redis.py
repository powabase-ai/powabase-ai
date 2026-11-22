"""Pending approvals across processes.

Two RunRegistry objects on one Redis stand in for two gunicorn workers or two
pods. The ExecutionContext never leaves the registry that registered it; Redis
records who owns the run and carries the decision.

Runs against fakeredis (one ``FakeServer`` shared by both "processes"). Set
``TEST_REDIS_URL`` (a database this suite may flush, e.g.
``redis://localhost:6379/15``) to run every test here against a real Redis as
well.
"""

import os
import threading
import time
from unittest.mock import MagicMock

import fakeredis
import pytest
import redis
from agentic.execution.context import ExecutionContext

from agentic_project_service.services import run_registry as rr
from agentic_project_service.services.run_registry import (
    OWNER_TTL_SECONDS,
    Delivery,
    RemoteRun,
    RunRegistry,
)

_REAL_REDIS_URL = os.getenv("TEST_REDIS_URL")


@pytest.fixture(autouse=True)
def _ref(monkeypatch):
    monkeypatch.setenv("PROJECT_REF", "ref-a")


@pytest.fixture(params=["fakeredis", "redis"])
def connect(request):
    """A factory of clients that all talk to the same Redis."""
    if request.param == "fakeredis":
        server = fakeredis.FakeServer()
        yield lambda: fakeredis.FakeStrictRedis(server=server)
        return
    if not _REAL_REDIS_URL:
        pytest.skip("TEST_REDIS_URL not set")
    real = redis.from_url(_REAL_REDIS_URL)
    real.flushdb()
    yield lambda: redis.from_url(_REAL_REDIS_URL)
    real.flushdb()


@pytest.fixture
def real_redis():
    """Only for what fakeredis cannot do (CLIENT KILL)."""
    if not _REAL_REDIS_URL:
        pytest.skip("TEST_REDIS_URL not set")
    real = redis.from_url(_REAL_REDIS_URL)
    real.flushdb()
    yield lambda: redis.from_url(_REAL_REDIS_URL)
    real.flushdb()


_open: list[RunRegistry] = []


@pytest.fixture(autouse=True)
def _close_registries():
    """A registry left subscribed would hear the next test's messages."""
    yield
    while _open:
        _open.pop().close()


def _registry(connect, name, start_thread=False):
    registry = RunRegistry(redis_client=connect(), instance=name, start_thread=start_thread)
    _open.append(registry)
    return registry


def _settle(registry):
    """Drive a thread-less registry until its Redis side has caught up."""
    for _ in range(50):
        registry.step(timeout=0.01)
        if (
            registry._subscribed.is_set()
            and not registry._dirty
            and not registry._removed
            and not registry._alive_dirty
            and not registry._alive_removed
        ):
            return
    raise AssertionError("registry did not settle")


def _broken_client():
    broken = MagicMock()
    for name in ("hset", "expire", "delete", "hgetall", "pubsub", "publish", "pipeline"):
        getattr(broken, name).side_effect = redis.ConnectionError("down")
    return broken


class TestOwnerHash:
    def test_register_records_instance_and_owner_with_ttl(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        a.register("run_1", ExecutionContext(execution_id="run_1"), owner_user_id="user-1")
        _settle(a)
        r = connect()
        assert r.hgetall("runs:owner:ref-a:run_1") == {
            b"instance": b"pod-a:1:aaaa",
            b"owner": b"user-1",
        }
        assert 0 < r.ttl("runs:owner:ref-a:run_1") <= OWNER_TTL_SECONDS
        assert r.get("runs:alive:ref-a:run_1") == b"pod-a:1:aaaa"
        assert 0 < r.ttl("runs:alive:ref-a:run_1") <= OWNER_TTL_SECONDS

    def test_unregister_deletes_the_hash_and_the_local_context(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        a.register("run_1", ExecutionContext(execution_id="run_1"))
        _settle(a)
        assert connect().exists("runs:owner:ref-a:run_1")
        a.unregister("run_1")
        _settle(a)
        assert not connect().exists("runs:owner:ref-a:run_1")
        assert not connect().exists("runs:alive:ref-a:run_1")
        assert a.context("run_1") is None

    def test_another_registry_finds_the_owner(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        a.register("run_1", ExecutionContext(execution_id="run_1"), owner_user_id=None)
        _settle(a)
        b = _registry(connect, "pod-b:2:bbbb")
        assert b.lookup_remote("run_1") == RemoteRun(instance="pod-a:1:aaaa", owner_user_id=None)
        assert b.lookup_remote("run_missing") is None

    def test_keys_are_project_scoped(self, connect, monkeypatch):
        a = _registry(connect, "pod-a")
        a.register("run_1", ExecutionContext(execution_id="run_1"))
        _settle(a)
        monkeypatch.setenv("PROJECT_REF", "ref-b")
        assert _registry(connect, "pod-b").lookup_remote("run_1") is None

    def test_redis_down_keeps_the_local_registration(self):
        a = RunRegistry(redis_client=_broken_client(), instance="pod-a", start_thread=False)
        ctx = ExecutionContext(execution_id="run_1")
        a.register("run_1", ctx, owner_user_id="user-1")
        assert a.step() is False  # fails open, schedules a resubscribe
        assert a.context("run_1") is ctx
        assert a.owner("run_1") == "user-1"
        with pytest.raises(rr.RegistryUnavailable):
            a.lookup_remote("run_1")  # Redis cannot say: not the same as "no such run"
        a.unregister("run_1")
        assert a.context("run_1") is None

    def test_an_unusable_broker_url_stays_process_local(self, monkeypatch):
        monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
        a = RunRegistry(instance="pod-a", start_thread=False)
        ctx = ExecutionContext(execution_id="run_1")
        a.register("run_1", ctx, owner_user_id="user-1")
        assert a.context("run_1") is ctx
        assert a.lookup_remote("run_1") is None


class TestInstanceId:
    def test_generated_id_is_host_pid_and_suffix(self, monkeypatch):
        monkeypatch.setenv("HOSTNAME", "pod-x")
        instance = RunRegistry(start_thread=False).instance_id()
        host, pid, suffix = instance.split(":")
        assert host == "pod-x"
        assert pid == str(os.getpid())
        assert len(suffix) == 8

    def test_a_forked_child_gets_its_own_id(self, monkeypatch):
        registry = RunRegistry(start_thread=False)
        parent = registry.instance_id()
        child_pid = os.getpid() + 1
        monkeypatch.setattr(rr.os, "getpid", lambda: child_pid)
        assert registry.instance_id() != parent


def _wait_until(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class TestForwarding:
    def _waiting_context(self, run_id):
        ctx = ExecutionContext(execution_id=run_id)
        ctx.reset_approval()  # what the approval hook does before it waits
        return ctx

    def test_approval_posted_to_b_resumes_the_run_on_a(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        b = _registry(connect, "pod-b:2:bbbb")
        ctx = self._waiting_context("run_1")
        a.register("run_1", ctx, owner_user_id="user-1")
        _settle(a)

        remote = b.lookup_remote("run_1")
        result = {}
        sender = threading.Thread(
            target=lambda: result.update(ok=b.deliver_remote("run_1", remote, {"approved": True}))
        )
        sender.start()
        assert a.step(timeout=1.0) is True
        sender.join(timeout=5)
        assert result == {"ok": Delivery.RESUMED}  # acknowledged by the owner
        assert ctx.get_approval_decision() == {"approved": True}
        assert ctx.get_approval_event().is_set()

    def test_the_listener_thread_resumes_the_run(self, connect):
        a = _registry(connect, "pod-a:1:aaaa", start_thread=True)
        b = _registry(connect, "pod-b:2:bbbb")
        ctx = self._waiting_context("run_1")
        a.register("run_1", ctx)
        assert _wait_until(lambda: b.lookup_remote("run_1") is not None)
        assert (
            b.deliver_remote("run_1", b.lookup_remote("run_1"), {"approved": False})
            == Delivery.RESUMED
        )
        assert ctx.get_approval_event().wait(timeout=3.0)
        assert ctx.get_approval_decision() == {"approved": False}

    def test_delivery_to_a_gone_process_reports_false(self, connect):
        b = _registry(connect, "pod-b:2:bbbb")
        gone = RemoteRun(instance="pod-dead:9:dead", owner_user_id=None)
        assert b.deliver_remote("run_1", gone, {"approved": True}) == Delivery.GONE

    def test_a_late_approval_for_an_ended_run_is_not_delivered(self, connect):
        a = _registry(connect, "pod-a:1:aaaa", start_thread=True)
        b = _registry(connect, "pod-b:2:bbbb")
        a.register("run_1", self._waiting_context("run_1"))
        assert _wait_until(lambda: b.lookup_remote("run_1") is not None)
        a.unregister("run_1")
        late = RemoteRun(instance="pod-a:1:aaaa", owner_user_id=None)
        # a still listens, and answers that the run is not there any more.
        started = time.monotonic()
        assert b.deliver_remote("run_1", late, {"approved": True}) == Delivery.GONE
        assert time.monotonic() - started < 1.0

    def test_a_subscriber_that_never_acknowledges_is_not_a_delivery(self, connect, monkeypatch):
        """A deaf owner (Redis still counts its subscription) must not read as resumed."""
        monkeypatch.setattr(rr, "ACK_TIMEOUT_SECONDS", 0.3)
        deaf = connect().pubsub()
        deaf.subscribe(RunRegistry.channel("pod-a:1:aaaa"))
        assert _wait_until(
            lambda: (deaf.get_message(timeout=0.05) or {}).get("type") == "subscribe"
        )
        b = _registry(connect, "pod-b:2:bbbb")
        assert (
            b.deliver_remote("run_1", RemoteRun("pod-a:1:aaaa", "u"), {"approved": True})
            == Delivery.UNCONFIRMED
        )
        deaf.close()

    def test_a_failing_resume_keeps_the_subscription_and_answers_not_resumed(self, connect, caplog):
        """An error applying one decision is that decision's problem, not a
        broken channel: log it, say not resumed, keep listening."""
        a = _registry(connect, "pod-a:1:aaaa", start_thread=True)
        b = _registry(connect, "pod-b:2:bbbb")
        broken = MagicMock()
        broken.set_approval_decision.side_effect = RuntimeError("hook state")
        a.register("run_bad", broken)
        ok = self._waiting_context("run_ok")
        a.register("run_ok", ok)
        assert _wait_until(lambda: b.lookup_remote("run_ok") is not None)
        pubsub = a._pubsub
        with caplog.at_level("ERROR", logger=rr.logger.name):
            assert (
                b.deliver_remote("run_bad", b.lookup_remote("run_bad"), {"approved": True})
                == Delivery.GONE
            )
        assert any(
            r.levelname == "ERROR" and r.exc_info and "run_bad" in r.getMessage()
            for r in caplog.records
        )
        assert a._pubsub is pubsub  # not torn down
        assert (
            b.deliver_remote("run_ok", b.lookup_remote("run_ok"), {"approved": True})
            == Delivery.RESUMED
        )

    def test_a_malformed_message_is_ignored(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        ctx = self._waiting_context("run_1")
        a.register("run_1", ctx)
        _settle(a)
        connect().publish(a.channel("pod-a:1:aaaa"), "not json")
        assert a.step(timeout=0.3) is False
        assert ctx.get_approval_decision() is None

    def test_subscription_exists_before_the_owner_hash(self, connect):
        """No window where another process sees the owner but nobody listens."""
        a = _registry(connect, "pod-a:1:aaaa")
        a.register("run_1", self._waiting_context("run_1"))
        _settle(a)
        assert connect().publish(a.channel("pod-a:1:aaaa"), "probe") == 1

    def test_no_subscription_means_no_owner_hash(self, connect, monkeypatch):
        a = _registry(connect, "pod-a:1:aaaa")
        a.register("run_1", self._waiting_context("run_1"))
        monkeypatch.setattr(a, "_connect", MagicMock(side_effect=redis.ConnectionError("down")))
        assert a.step() is False
        assert not connect().exists("runs:owner:ref-a:run_1")
        monkeypatch.undo()
        monkeypatch.setenv("PROJECT_REF", "ref-a")
        a._retry_at = 0.0
        _settle(a)
        assert connect().exists("runs:owner:ref-a:run_1")

    def test_channels_are_project_scoped(self, connect, monkeypatch):
        a = _registry(connect, "pod-a:1:aaaa")
        a.register("run_1", self._waiting_context("run_1"))
        _settle(a)
        monkeypatch.setenv("PROJECT_REF", "ref-b")
        b = _registry(connect, "pod-b:2:bbbb")
        assert (
            b.deliver_remote("run_1", RemoteRun("pod-a:1:aaaa", None), {"approved": True})
            == Delivery.GONE
        )

    def test_redis_down_delivery_is_unavailable_not_gone(self):
        b = RunRegistry(redis_client=_broken_client(), instance="pod-b", start_thread=False)
        assert (
            b.deliver_remote("run_1", RemoteRun("pod-a", None), {"approved": True})
            == Delivery.UNAVAILABLE
        )

    def test_an_ack_read_error_after_the_publish_is_unconfirmed(self, connect, monkeypatch):
        """Once published, the run may resume: never report it unavailable."""
        b = _registry(connect, "pod-b:2:bbbb")
        monkeypatch.setattr(b._client, "publish", lambda channel, payload: 1)
        monkeypatch.setattr(b._client, "get", MagicMock(side_effect=redis.ConnectionError("down")))
        assert (
            b.deliver_remote("run_1", RemoteRun("pod-a", None), {"approved": True})
            == Delivery.UNCONFIRMED
        )

    def test_redis_down_listener_thread_keeps_retrying(self, monkeypatch):
        monkeypatch.setattr(rr, "RESUBSCRIBE_BACKOFF_SECONDS", 0.01)
        broken = _broken_client()
        a = RunRegistry(redis_client=broken, instance="pod-a")
        _open.append(a)
        a.register("run_1", ExecutionContext(execution_id="run_1"))
        assert a.context("run_1") is not None
        assert _wait_until(lambda: broken.pubsub.call_count >= 3)

    def test_close_stops_listening(self, connect):
        a = _registry(connect, "pod-a:1:aaaa", start_thread=True)
        a.register("run_1", self._waiting_context("run_1"))
        assert _wait_until(a._subscribed.is_set)
        a.close()
        assert connect().publish(a.channel("pod-a:1:aaaa"), "probe") == 0


def test_module_functions_keep_their_contract():
    ctx = ExecutionContext(execution_id="run_mod")
    rr.register_run("run_mod", ctx, owner_user_id="user-1")
    assert rr.get_active_run_context("run_mod") is ctx
    assert rr.get_active_run_owner("run_mod") == "user-1"
    rr.unregister_run("run_mod")
    assert rr.get_active_run_context("run_mod") is None


class _SleepingRedis:
    """A Redis that hangs on every call (a blackhole with a 2 s socket timeout)."""

    def __init__(self, seconds=2.0):
        self.seconds = seconds

    def __getattr__(self, name):
        def call(*args, **kwargs):
            time.sleep(self.seconds)
            raise redis.TimeoutError(f"{name} timed out")

        return call


class TestNeverWaitsForRedis:
    """Run start and end sit on the request path: they must not wait on Redis."""

    def test_register_and_unregister_return_immediately(self):
        a = RunRegistry(redis_client=_SleepingRedis(), instance="pod-a")
        _open.append(a)
        ctx = ExecutionContext(execution_id="run_1")
        started = time.monotonic()
        a.register("run_1", ctx, owner_user_id="user-1")
        assert time.monotonic() - started < 0.05
        assert a.context("run_1") is ctx
        started = time.monotonic()
        a.unregister("run_1")
        assert time.monotonic() - started < 0.05
        assert a.context("run_1") is None

    def test_concurrent_registers_do_not_queue_behind_redis(self):
        a = RunRegistry(redis_client=_SleepingRedis(), instance="pod-a")
        _open.append(a)
        took = {}

        def go(i):
            started = time.monotonic()
            a.register(f"run_{i}", ExecutionContext(execution_id=f"run_{i}"))
            took[i] = time.monotonic() - started

        threads = [threading.Thread(target=go, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert max(took.values()) < 0.05


class _Deaf:
    """A subscription whose connection died silently: nothing arrives, no PONG."""

    def __init__(self, inner):
        self.inner = inner

    def get_message(self, timeout=0.0):
        time.sleep(min(timeout or 0.0, 0.05))

    def ping(self):
        pass

    def close(self):
        self.inner.close()


@pytest.fixture
def fast_liveness(monkeypatch):
    monkeypatch.setattr(rr, "OWNER_TTL_SECONDS", 1)
    monkeypatch.setattr(rr, "HEARTBEAT_SECONDS", 0.2)
    monkeypatch.setattr(rr, "PONG_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(rr, "ACK_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(rr, "RESUBSCRIBE_BACKOFF_SECONDS", 0.05)


class TestLiveness:
    """The owner hash lives only as long as a listener that answers PINGs."""

    def _waiting(self, run_id):
        ctx = ExecutionContext(execution_id=run_id)
        ctx.reset_approval()
        return ctx

    def test_owner_ttl_is_short_and_refreshed_by_the_heartbeat(self):
        assert rr.OWNER_TTL_SECONDS == 60
        assert rr.HEARTBEAT_SECONDS == 15
        assert rr.PONG_TIMEOUT_SECONDS == 3
        assert rr.OWNER_TTL_SECONDS > 3 * rr.HEARTBEAT_SECONDS

    def test_the_heartbeat_keeps_a_waiting_run_reachable(self, connect, fast_liveness):
        a = _registry(connect, "pod-a:1:aaaa", start_thread=True)
        a.register("run_1", self._waiting("run_1"))
        assert _wait_until(lambda: connect().exists("runs:owner:ref-a:run_1"))
        time.sleep(2.5)  # well past the 1 s TTL
        assert connect().exists("runs:owner:ref-a:run_1")

    def test_the_owner_hash_expires_when_the_listener_stops(self, connect, fast_liveness):
        a = _registry(connect, "pod-a:1:aaaa", start_thread=True)
        a.register("run_1", self._waiting("run_1"))
        assert _wait_until(lambda: connect().exists("runs:owner:ref-a:run_1"))
        a._stopped.set()  # the listener dies; the run stays registered locally
        a._thread.join(timeout=3)
        assert a.context("run_1") is not None
        assert _wait_until(lambda: not connect().exists("runs:owner:ref-a:run_1"), timeout=3)

    def test_a_lost_owner_hash_is_written_again(self, connect, fast_liveness):
        """Redis lost its data (restart without persistence, failover)."""
        a = _registry(connect, "pod-a:1:aaaa", start_thread=True)
        a.register("run_1", self._waiting("run_1"))
        assert _wait_until(lambda: connect().exists("runs:owner:ref-a:run_1"))
        connect().flushdb()
        assert _wait_until(lambda: connect().exists("runs:owner:ref-a:run_1"))

    def test_a_silently_dead_subscription_is_replaced(self, connect, fast_liveness):
        a = _registry(connect, "pod-a:1:aaaa", start_thread=True)
        b = _registry(connect, "pod-b:2:bbbb")
        ctx = self._waiting("run_1")
        a.register("run_1", ctx, owner_user_id="u")
        assert _wait_until(lambda: b.lookup_remote("run_1") is not None)

        deaf = _Deaf(a._pubsub)
        a._pubsub = deaf
        time.sleep(0.15)  # let the listener's current read on the live one finish
        connect().delete("runs:owner:ref-a:run_1")
        # While deaf: Redis may still count the subscriber, but nothing acknowledges.
        assert (
            b.deliver_remote("run_1", RemoteRun("pod-a:1:aaaa", "u"), {"approved": True})
            == Delivery.UNCONFIRMED
        )
        assert ctx.get_approval_decision() is None

        # A missing PONG drops it; the new subscription re-asserts the owner hash.
        assert _wait_until(lambda: a._pubsub not in (None, deaf) and a._subscribed.is_set())
        assert _wait_until(lambda: b.lookup_remote("run_1") is not None)
        assert (
            b.deliver_remote("run_1", b.lookup_remote("run_1"), {"approved": True})
            == Delivery.RESUMED
        )
        assert ctx.get_approval_event().wait(timeout=1)

    def test_a_killed_subscription_reconnects(self, real_redis, fast_liveness):
        a = _registry(real_redis, "pod-a:1:aaaa", start_thread=True)
        b = _registry(real_redis, "pod-b:2:bbbb")
        ctx = self._waiting("run_1")
        a.register("run_1", ctx, owner_user_id="u")
        assert _wait_until(lambda: b.lookup_remote("run_1") is not None)
        real_redis().client_kill_filter(_type="pubsub")
        assert _wait_until(
            lambda: (
                b.deliver_remote("run_1", RemoteRun("pod-a:1:aaaa", "u"), {"approved": True})
                == Delivery.RESUMED
            ),
            timeout=5,
        )
        assert ctx.get_approval_decision() == {"approved": True}


class TestWhatIsForwarded:
    def test_only_the_decision_fields_go_to_redis(self, connect, monkeypatch):
        """The hook reads ``approved`` and ``reason``; nothing else is published."""
        import json

        monkeypatch.setattr(rr, "ACK_TIMEOUT_SECONDS", 0.2)
        listener = connect().pubsub()
        listener.subscribe(RunRegistry.channel("pod-a:1:aaaa"))
        assert _wait_until(
            lambda: (listener.get_message(timeout=0.05) or {}).get("type") == "subscribe"
        )
        b = _registry(connect, "pod-b:2:bbbb")
        body = {"approved": True, "reason": "r" * 5000, "padding": "x" * 1_000_000}
        b.deliver_remote("run_1", RemoteRun("pod-a:1:aaaa", None), body)
        published = None
        deadline = time.monotonic() + 3
        while published is None and time.monotonic() < deadline:
            message = listener.get_message(timeout=0.05)
            if message is not None and message.get("type") == "message":
                published = message
        listener.close()
        payload = json.loads(published["data"])
        assert payload["decision"] == {"approved": True, "reason": "r" * rr.REASON_MAX_CHARS}
        assert rr.REASON_MAX_CHARS == 2000

    def test_a_decision_without_a_reason_stays_without_one(self):
        assert rr.wire_decision({"approved": False, "extra": 1}) == {"approved": False}
        assert rr.wire_decision({"approved": True, "reason": None}) == {"approved": True}
        assert rr.wire_decision({"approved": True, "reason": 42}) == {
            "approved": True,
            "reason": "42",
        }


class TestOwnerHashWriteFailures:
    """A failing hash write (OOM under noeviction) is not a dead subscription."""

    def test_a_failing_write_keeps_the_subscription_and_is_retried(
        self, connect, monkeypatch, caplog
    ):
        import json

        a = _registry(connect, "pod-a:1:aaaa")
        ctx = ExecutionContext(execution_id="run_1")
        ctx.reset_approval()
        a.register("run_1", ctx)
        real_pipeline = a._client.pipeline

        def oom_pipeline(*args, **kwargs):
            pipe = real_pipeline(*args, **kwargs)
            pipe.execute = MagicMock(
                side_effect=redis.ResponseError(
                    "OOM command not allowed when used memory > 'maxmemory'."
                )
            )
            return pipe

        monkeypatch.setattr(a._client, "pipeline", oom_pipeline)
        with caplog.at_level("WARNING", logger=rr.logger.name):
            for _ in range(5):
                a.step(timeout=0.01)
                a._write_retry_at = a._alive_retry_at = 0.0  # skip the retry delays
        pubsub = a._pubsub
        assert pubsub is not None
        assert a._subscribed.is_set()
        assert not connect().exists("runs:owner:ref-a:run_1")
        warnings = [r for r in caplog.records if r.name == rr.logger.name]
        assert len(warnings) == 1  # rate-limited, across owner and liveness writes
        assert "could not write" in warnings[0].getMessage()

        # The subscription still carries decisions.
        connect().publish(
            a.channel("pod-a:1:aaaa"),
            json.dumps({"run_id": "run_1", "decision": {"approved": True}}),
        )
        assert a.step(timeout=1.0) is True
        assert a._pubsub is pubsub

        # Once Redis accepts writes again, the pending hash is written.
        monkeypatch.setattr(a._client, "pipeline", real_pipeline)
        a._write_retry_at = a._alive_retry_at = 0.0
        a.step(timeout=0.01)
        assert connect().exists("runs:owner:ref-a:run_1")
        assert connect().exists("runs:alive:ref-a:run_1")
        assert a._pubsub is pubsub


class TestLiveOnlyRuns:
    """Runs without an approval gate (plain agent streams, orchestrations)
    publish only a liveness record, so the stale-run cleanup can tell a run
    that is alive from one whose process is gone. They are never approval
    targets."""

    def test_a_live_run_has_a_liveness_record_with_a_ttl_and_no_owner_hash(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        a.register_live("orch_run_1")
        _settle(a)
        r = connect()
        assert r.get("runs:alive:ref-a:orch_run_1") == b"pod-a:1:aaaa"
        assert 0 < r.ttl("runs:alive:ref-a:orch_run_1") <= OWNER_TTL_SECONDS
        assert not r.exists("runs:owner:ref-a:orch_run_1")

    def test_a_live_run_is_not_an_approval_target(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        a.register_live("orch_run_1")
        _settle(a)
        assert a.context("orch_run_1") is None
        assert _registry(connect, "pod-b:2:bbbb").lookup_remote("orch_run_1") is None

    def test_unregister_deletes_a_live_runs_record(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        a.register_live("orch_run_1")
        _settle(a)
        a.unregister("orch_run_1")
        _settle(a)
        assert not connect().exists("runs:alive:ref-a:orch_run_1")

    def test_the_heartbeat_keeps_a_live_run_alive(self, connect, fast_liveness):
        a = _registry(connect, "pod-a:1:aaaa", start_thread=True)
        a.register_live("orch_run_1")
        assert _wait_until(lambda: connect().exists("runs:alive:ref-a:orch_run_1"))
        time.sleep(2.5)  # well past the 1 s TTL
        assert connect().exists("runs:alive:ref-a:orch_run_1")

    def test_a_lost_live_record_is_rewritten_at_the_next_refresh(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        a.register_live("orch_run_1")
        _settle(a)
        connect().delete("runs:alive:ref-a:orch_run_1")
        a._alive_due = 0.0  # the refresh is due
        _settle(a)
        assert connect().exists("runs:alive:ref-a:orch_run_1")

    def test_a_run_registered_both_ways_stays_approvable(self, connect):
        """register then register_live (or the reverse) must not make a run
        that waits for an approval unreachable."""
        a = _registry(connect, "pod-a:1:aaaa")
        a.register("run_1", ExecutionContext(execution_id="run_1"), owner_user_id="u")
        a.register_live("run_1")
        _settle(a)
        b = _registry(connect, "pod-b:2:bbbb")
        assert b.lookup_remote("run_1") == RemoteRun("pod-a:1:aaaa", "u")


class TestLivenessWithoutTheSubscription:
    """The liveness record says the process is alive, not that it can take
    approvals, so it is written with plain commands even when the approvals
    subscription cannot be made (pub/sub refused by an ACL or a proxy) or has
    dropped."""

    def _no_pubsub(self, registry, monkeypatch):
        monkeypatch.setattr(
            registry, "_connect", MagicMock(side_effect=redis.ConnectionError("no pubsub"))
        )

    def test_records_are_written_while_the_subscription_fails(self, connect, monkeypatch):
        a = _registry(connect, "pod-a:1:aaaa")
        self._no_pubsub(a, monkeypatch)
        a.register("run_1", ExecutionContext(execution_id="run_1"))
        a.register_live("orch_run_1")
        assert a.step() is False
        r = connect()
        assert r.get("runs:alive:ref-a:run_1") == b"pod-a:1:aaaa"
        assert r.get("runs:alive:ref-a:orch_run_1") == b"pod-a:1:aaaa"
        assert not r.exists("runs:owner:ref-a:run_1")  # not reachable for approvals

    def test_they_are_refreshed_while_the_subscription_fails(self, connect, monkeypatch):
        a = _registry(connect, "pod-a:1:aaaa")
        self._no_pubsub(a, monkeypatch)
        a.register_live("orch_run_1")
        a.step()
        connect().delete("runs:alive:ref-a:orch_run_1")
        a._alive_due = 0.0
        a._retry_at = 0.0
        a.step()
        assert connect().exists("runs:alive:ref-a:orch_run_1")

    def test_unregister_deletes_the_record_without_a_subscription(self, connect, monkeypatch):
        a = _registry(connect, "pod-a:1:aaaa")
        self._no_pubsub(a, monkeypatch)
        a.register_live("orch_run_1")
        a.step()
        a.unregister("orch_run_1")
        a._retry_at = 0.0
        a.step()
        assert not connect().exists("runs:alive:ref-a:orch_run_1")

    def test_the_listener_thread_writes_them_while_the_subscription_fails(
        self, connect, monkeypatch, fast_liveness
    ):
        a = _registry(connect, "pod-a:1:aaaa")
        self._no_pubsub(a, monkeypatch)
        a._start_thread = True
        a.register_live("orch_run_1")
        assert _wait_until(lambda: connect().exists("runs:alive:ref-a:orch_run_1"))
        time.sleep(2.5)  # past the 1 s TTL: refreshed though never subscribed
        assert connect().exists("runs:alive:ref-a:orch_run_1")

    def test_a_failing_write_is_retried(self, connect, monkeypatch):
        a = _registry(connect, "pod-a:1:aaaa")
        self._no_pubsub(a, monkeypatch)
        real_pipeline = a._client.pipeline

        def oom_pipeline(*args, **kwargs):
            pipe = real_pipeline(*args, **kwargs)
            pipe.execute = MagicMock(side_effect=redis.ResponseError("OOM"))
            return pipe

        monkeypatch.setattr(a._client, "pipeline", oom_pipeline)
        a.register_live("orch_run_1")
        a.step()
        assert not connect().exists("runs:alive:ref-a:orch_run_1")
        monkeypatch.setattr(a._client, "pipeline", real_pipeline)
        a._alive_retry_at = 0.0
        a.step()
        assert connect().exists("runs:alive:ref-a:orch_run_1")


def test_register_live_run_module_function():
    reg = RunRegistry(redis_client=fakeredis.FakeStrictRedis(), instance="x", start_thread=False)
    original = rr._default
    rr._default = reg
    try:
        rr.register_live_run("run_live")
        assert "run_live" in reg._live
        rr.unregister_run("run_live")
        assert "run_live" not in reg._live
    finally:
        rr._default = original
