"""Redis clients for request-path state on the Celery broker's Redis.

The approvals registry and the rate limiter talk to Redis while a request
waits, and both are advisory: a Redis error must fail open. A refused
connection fails fast on its own, but a Redis that accepts connections and
never answers (node loss, a dropped route, a long fork) would block a request
for the kernel's TCP timeouts. These clients bound every connect and read and
never retry, so the worst case is about a second before failing open.
"""

import os

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

REDIS_CONNECT_TIMEOUT_SECONDS = 0.5
REDIS_SOCKET_TIMEOUT_SECONDS = 1.0


def broker_url() -> str:
    return os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")


def broker_client() -> redis.Redis:
    """A client on the broker Redis with bounded timeouts. Raises ValueError
    for a URL that is not a Redis URL (``memory://`` in the unit tests)."""
    return redis.from_url(
        broker_url(),
        socket_connect_timeout=REDIS_CONNECT_TIMEOUT_SECONDS,
        socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
        socket_keepalive=True,
        retry=Retry(NoBackoff(), 0),
    )
