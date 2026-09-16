"""A knowledge base's partition move against the transactions around it.

The move holds SHARE on the partitioned parent for its whole transaction, and
needs ACCESS EXCLUSIVE on the DEFAULT partition twice: once on a second
connection to put up its temporary check, and once for the ATTACH. These tests
drive real transactions against both of those steps -- a transaction that read
DEFAULT and then writes through the parent, a reader left idle in its
transaction, a long read that starts part-way through the move -- and pin who
gives way, what is left behind, and how long everyone else waited.

Same database and scratch schema as ``test_pg_bm25_live``.
"""

from __future__ import annotations

import threading

import pytest
from sqlalchemy import text

from agentic_project_service.services import pg_bm25_index as pgb
from tests.pg_search import test_pg_bm25_live as live
from tests.pg_search.test_pg_bm25_live import (
    KB_A,
    KB_A_DOCS,
    KB_B,
    KB_B_DOCS,
    KB_C,
    SCHEMA,
    SOURCE_1,
    _hold_the_move_open,
    _move_check_names,
    _rows_in,
    _seed,
)

# The live module's fixtures: its database, and the scratch schema every test
# here starts from (autouse there, so re-bound here to stay autouse).
migration = live.migration
engine = live.engine
scratch_schema = live.scratch_schema
session = live.session


def _kb_rows(session, kb_id) -> int:
    return _rows_in(session, "chunks", kb_id)


def _insert(conn, kb_id, body):
    conn.execute(
        text(
            f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text) "
            "VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), :body)"
        ),
        {"kb": kb_id, "src": SOURCE_1, "body": body},
    )


def _first_line(exc: BaseException) -> str:
    return str(getattr(exc, "orig", exc)).splitlines()[0]


# ---------------------------------------------------------------------------
# A transaction that reads DEFAULT and then writes through the parent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stall_seconds", [0.3, 1.5])
def test_a_transaction_that_read_default_then_writes_is_never_the_deadlock_victim(
    engine, session, monkeypatch, stall_seconds
):
    """The app transaction commits; the move gives way and succeeds on retry.

    What the re-index cleanup does: SELECT the chunk ids (ACCESS SHARE on
    DEFAULT), then write through the parent, which waits on the move's SHARE
    lock. If the move then *waited* for ACCESS EXCLUSIVE on DEFAULT, that is a
    lock cycle, and Postgres aborts whichever side runs its deadlock check
    first -- the app whenever it began waiting less than ``deadlock_timeout``
    before the ATTACH (the 0.3 s case), the move otherwise (1.5 s). The move
    must never wait in that queue, so it is the move that gives up, both times.
    """
    monkeypatch.setattr(pgb, "DEFAULT_EXCLUSIVE_LOCK_WAIT_SECONDS", 0.5, raising=False)
    _seed(session, KB_A, 2_000)
    moving = threading.Event()
    _hold_the_move_open(monkeypatch, moving, stall_seconds)
    app: dict = {}

    def app_transaction():
        moving.wait(timeout=30)
        with engine.connect() as conn:
            try:
                conn.execute(
                    text(
                        f"SELECT count(*) FROM {SCHEMA}.chunks "
                        "WHERE knowledge_base_id = CAST(:kb AS uuid)"
                    ),
                    {"kb": KB_B},
                ).scalar()
                _insert(conn, KB_B, "rando pendant le deplacement")
                conn.execute(
                    text(
                        f"DELETE FROM {SCHEMA}.chunks WHERE knowledge_base_id = CAST(:kb AS uuid)"
                    ),
                    {"kb": KB_C},
                )
                conn.commit()
                app["committed"] = True
            except Exception as exc:
                conn.rollback()
                app["error"] = _first_line(exc)

    thread = threading.Thread(target=app_transaction, daemon=True)
    thread.start()
    first_attempt: BaseException | None = None
    try:
        pgb.create_partition(engine, KB_A, "chunks")
    except Exception as exc:
        first_attempt = exc
    thread.join(timeout=60)

    assert app == {"committed": True}
    if first_attempt is not None:
        assert pgb.is_transient_db_error(first_attempt), _first_line(first_attempt)
        assert _move_check_names(session) == []
        assert pgb.create_partition(engine, KB_A, "chunks")["rows_moved"] == 2_000 + len(KB_A_DOCS)
    partition = pgb.partition_name(KB_A, "chunks")
    assert _rows_in(session, partition) == 2_000 + len(KB_A_DOCS)
    assert _rows_in(session, "chunks_default", KB_A) == 0
    assert _kb_rows(session, KB_B) == len(KB_B_DOCS) + 1
    assert _kb_rows(session, KB_C) == 0
