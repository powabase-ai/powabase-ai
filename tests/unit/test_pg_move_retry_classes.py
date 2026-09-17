"""Which failures of a move or an index build are worth retrying.

The move is one transaction and the index build is re-entrant, so anything
that interrupted them without the request being wrong -- a lock conflict, a
statement cancelled by a server-side timeout, a lost connection, pg_search's
own concurrent build failing -- is retried by the task instead of failing it
for good.
"""

from __future__ import annotations

import psycopg
import pytest
from sqlalchemy.exc import DBAPIError, OperationalError

from agentic_project_service.services import pg_bm25_index as pgb


def _wrapped(sqlstate, message="boom", *, invalidated=False):
    class _Orig(Exception):
        pass

    orig = _Orig(message)
    orig.sqlstate = sqlstate
    return OperationalError("stmt", {}, orig, connection_invalidated=invalidated)


@pytest.mark.parametrize(
    "sqlstate",
    ["55P03", "40P01", "40001", "57014", "57P01", "57P02", "57P03", "08000", "08003", "08006"],
)
def test_contention_cancellation_and_connection_loss_are_transient(sqlstate):
    assert pgb.is_transient_db_error(_wrapped(sqlstate)) is True


def test_a_connection_lost_on_the_client_side_is_transient():
    """psycopg reports a dropped connection with no SQLSTATE at all."""
    assert pgb.is_transient_db_error(_wrapped(None, invalidated=True)) is True
    lost = DBAPIError.instance(
        "stmt",
        {},
        psycopg.OperationalError("server closed the connection unexpectedly"),
        psycopg.Error,
        connection_invalidated=True,
    )
    assert pgb.is_transient_db_error(lost) is True


@pytest.mark.parametrize("sqlstate", ["23505", "42P01", "XX000", None])
def test_real_errors_are_not_transient(sqlstate):
    assert pgb.is_transient_db_error(_wrapped(sqlstate)) is False


def test_a_failed_concurrent_bm25_build_is_transient():
    """pg_search 0.25.9's CREATE INDEX CONCURRENTLY can die with XX000 under
    concurrent writes, leaving an INVALID index the next ensure repairs."""
    failure = pgb.Bm25IndexBuildFailed("buffer 903 is not owned by resource owner")
    assert pgb.is_transient_db_error(failure) is True


def test_lock_conflicts_are_recognised_wrapped_or_bare():
    bare = psycopg.errors.LockNotAvailable("canceling statement due to lock timeout")
    assert pgb.is_lock_conflict(bare) is True
    assert pgb.is_lock_conflict(_wrapped("55P03")) is True
    assert pgb.is_lock_conflict(_wrapped("40P01")) is True
    assert pgb.is_lock_conflict(_wrapped("57014")) is False
