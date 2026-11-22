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
import time
from unittest.mock import MagicMock

import fakeredis
import pytest
import redis
from agentic.execution.context import ExecutionContext

from agentic_project_service.services import run_registry as rr
from agentic_project_service.services.run_registry import (
    OWNER_TTL_SECONDS,
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


def _broken_client():
    broken = MagicMock()
    for name in ("hset", "expire", "delete", "hgetall", "pubsub", "publish", "pipeline"):
        getattr(broken, name).side_effect = redis.ConnectionError("down")
    return broken


class TestOwnerHash:
    def test_register_records_instance_and_owner_with_ttl(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        a.register("run_1", ExecutionContext(execution_id="run_1"), owner_user_id="user-1")
        r = connect()
        assert r.hgetall("runs:owner:ref-a:run_1") == {
            b"instance": b"pod-a:1:aaaa",
            b"owner": b"user-1",
        }
        assert 0 < r.ttl("runs:owner:ref-a:run_1") <= OWNER_TTL_SECONDS

    def test_unregister_deletes_the_hash_and_the_local_context(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        a.register("run_1", ExecutionContext(execution_id="run_1"))
        a.unregister("run_1")
        assert not connect().exists("runs:owner:ref-a:run_1")
        assert a.context("run_1") is None

    def test_another_registry_finds_the_owner(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        a.register("run_1", ExecutionContext(execution_id="run_1"), owner_user_id=None)
        b = _registry(connect, "pod-b:2:bbbb")
        assert b.lookup_remote("run_1") == RemoteRun(instance="pod-a:1:aaaa", owner_user_id=None)
        assert b.lookup_remote("run_missing") is None

    def test_keys_are_project_scoped(self, connect, monkeypatch):
        _registry(connect, "pod-a").register("run_1", ExecutionContext(execution_id="run_1"))
        monkeypatch.setenv("PROJECT_REF", "ref-b")
        assert _registry(connect, "pod-b").lookup_remote("run_1") is None

    def test_redis_down_keeps_the_local_registration(self):
        a = RunRegistry(redis_client=_broken_client(), instance="pod-a", start_thread=False)
        ctx = ExecutionContext(execution_id="run_1")
        a.register("run_1", ctx, owner_user_id="user-1")
        assert a.context("run_1") is ctx
        assert a.owner("run_1") == "user-1"
        assert a.lookup_remote("run_1") is None
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

        remote = b.lookup_remote("run_1")
        assert b.deliver_remote("run_1", remote, {"approved": True}) is True
        assert a.poll_once(timeout=1.0) is True
        assert ctx.get_approval_decision() == {"approved": True}
        assert ctx.get_approval_event().is_set()

    def test_the_listener_thread_resumes_the_run(self, connect):
        a = _registry(connect, "pod-a:1:aaaa", start_thread=True)
        b = _registry(connect, "pod-b:2:bbbb")
        ctx = self._waiting_context("run_1")
        a.register("run_1", ctx)
        assert b.deliver_remote("run_1", b.lookup_remote("run_1"), {"approved": False}) is True
        assert ctx.get_approval_event().wait(timeout=3.0)
        assert ctx.get_approval_decision() == {"approved": False}

    def test_delivery_to_a_gone_process_reports_false(self, connect):
        b = _registry(connect, "pod-b:2:bbbb")
        gone = RemoteRun(instance="pod-dead:9:dead", owner_user_id=None)
        assert b.deliver_remote("run_1", gone, {"approved": True}) is False

    def test_a_late_approval_for_an_ended_run_is_dropped(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        b = _registry(connect, "pod-b:2:bbbb")
        a.register("run_1", self._waiting_context("run_1"))
        a.unregister("run_1")
        late = RemoteRun(instance="pod-a:1:aaaa", owner_user_id=None)
        assert b.deliver_remote("run_1", late, {"approved": True}) is True  # a still listens
        assert a.poll_once(timeout=0.3) is False

    def test_a_malformed_message_is_ignored(self, connect):
        a = _registry(connect, "pod-a:1:aaaa")
        ctx = self._waiting_context("run_1")
        a.register("run_1", ctx)
        connect().publish(a.channel("pod-a:1:aaaa"), "not json")
        assert a.poll_once(timeout=0.3) is False
        assert ctx.get_approval_decision() is None

    def test_subscription_exists_before_the_owner_hash(self, connect):
        """No window where another process sees the owner but nobody listens."""
        a = _registry(connect, "pod-a:1:aaaa")
        a.register("run_1", self._waiting_context("run_1"))
        assert connect().publish(a.channel("pod-a:1:aaaa"), "probe") == 1

    def test_channels_are_project_scoped(self, connect, monkeypatch):
        a = _registry(connect, "pod-a:1:aaaa")
        a.register("run_1", self._waiting_context("run_1"))
        monkeypatch.setenv("PROJECT_REF", "ref-b")
        b = _registry(connect, "pod-b:2:bbbb")
        assert (
            b.deliver_remote("run_1", RemoteRun("pod-a:1:aaaa", None), {"approved": True}) is False
        )

    def test_redis_down_delivery_is_false(self):
        b = RunRegistry(redis_client=_broken_client(), instance="pod-b", start_thread=False)
        assert b.deliver_remote("run_1", RemoteRun("pod-a", None), {"approved": True}) is False

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
        a.close()
        assert connect().publish(a.channel("pod-a:1:aaaa"), "probe") == 0


def test_module_functions_keep_their_contract():
    ctx = ExecutionContext(execution_id="run_mod")
    rr.register_run("run_mod", ctx, owner_user_id="user-1")
    assert rr.get_active_run_context("run_mod") is ctx
    assert rr.get_active_run_owner("run_mod") == "user-1"
    rr.unregister_run("run_mod")
    assert rr.get_active_run_context("run_mod") is None
