"""Redis clients for request-path state on the Celery broker's Redis.

The approvals registry, the run gate, the rate limiter and the stale-run
cleanup talk to Redis while a request or a task waits, and all of them are
advisory: a Redis error must fail open. A refused connection fails fast on its
own, but a Redis that accepts connections and never answers (node loss, a
dropped route, a long fork) would block for the kernel's TCP timeouts. These
clients bound every connect and read and never retry, so one call costs at
most about a second (REDIS_CONNECT_TIMEOUT_SECONDS + REDIS_SOCKET_TIMEOUT_SECONDS)
before it fails. A request that makes several calls can wait for each of them.

The broker Redis is shared by every project, so a process keeps one client
(one connection pool) per URL and timeouts, shared by all of those components:
its connections number about the threads using Redis at once, plus the
registry's one pub/sub connection. redis-py's pool opens fresh connections in
a forked child.
"""

import os
import threading

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

REDIS_CONNECT_TIMEOUT_SECONDS = 0.5
REDIS_SOCKET_TIMEOUT_SECONDS = 1.0

_clients: dict[tuple[str, float, float], redis.Redis] = {}
_clients_lock = threading.Lock()


def broker_url() -> str:
    return os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")


def broker_client(
    socket_timeout: float = REDIS_SOCKET_TIMEOUT_SECONDS,
    connect_timeout: float = REDIS_CONNECT_TIMEOUT_SECONDS,
) -> redis.Redis:
    """This process's client on the broker Redis with these timeouts, built on
    first use; builds no connection. Raises ValueError for a URL that is not
    a Redis URL (``memory://`` in the unit tests), every time it is asked."""
    url = broker_url()
    key = (url, socket_timeout, connect_timeout)
    client = _clients.get(key)
    if client is not None:
        return client
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            client = _clients[key] = redis.from_url(
                url,
                socket_connect_timeout=connect_timeout,
                socket_timeout=socket_timeout,
                socket_keepalive=True,
                retry=Retry(NoBackoff(), 0),
            )
        return client
