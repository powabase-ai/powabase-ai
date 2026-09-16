"""I12: a write that races a partition move must leave the source claimable.

A write of a knowledge base's rows that Postgres routed to the DEFAULT
partition just as that knowledge base's partition was attached fails with
SQLSTATE 23514 -- "violates partition constraint", or the move's temporary
``bm25_move_<kb>`` check. Nothing about the source is wrong, so it must be
re-queued within the attempts bound, not marked ``failed`` with nothing to
re-queue it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import psycopg
import pytest
from sqlalchemy.exc import IntegrityError

from agentic_project_service.services import pg_bm25_index as pgb
from agentic_project_service.tasks import indexing

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
SRC = "11111111-1111-4111-8111-111111111111"
IS_ID = "22222222-2222-4222-8222-222222222222"


def _integrity(message: str) -> IntegrityError:
    return IntegrityError("INSERT INTO ai.chunks ...", {}, psycopg.errors.CheckViolation(message))


RACES = [
    'new row for relation "chunks_default" violates partition constraint',
    'new row for relation "chunks_default" violates check constraint '
    '"bm25_move_3f2504e04f8911d39a0c0305e82c3301"',
]


@pytest.mark.parametrize("message", RACES)
def test_a_partition_move_race_is_recognised(message):
    assert pgb.is_partition_move_race(_integrity(message)) is True


def test_an_ordinary_check_violation_is_not_a_race():
    assert (
        pgb.is_partition_move_race(
            _integrity('new row for relation "chunks" violates check constraint "tokens_positive"')
        )
        is False
    )
    assert pgb.is_partition_move_race(RuntimeError("violates partition constraint")) is False


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(indexing, "db", MagicMock())
    monkeypatch.setattr(indexing, "get_knowledge_base", lambda _id: {"indexing_config": {}})
    monkeypatch.setattr(indexing, "get_source", lambda _id: {"extraction_status": "extracted"})
    monkeypatch.setattr(indexing, "_claim_indexed_source", lambda *_a: 1)
    requeue = MagicMock()
    failed = MagicMock()
    monkeypatch.setattr(indexing, "_handle_storage_error", requeue)
    monkeypatch.setattr(indexing, "_fenced_mark_failed", failed)
    return monkeypatch, requeue, failed


@pytest.mark.parametrize("message", RACES)
def test_index_source_requeues_a_source_whose_write_raced_the_attach(harness, message):
    monkeypatch, requeue, failed = harness

    def _raise(**_kwargs):
        raise _integrity(message)

    monkeypatch.setattr(indexing, "_run_index_body", _raise)

    out = indexing.index_source.run(KB, SRC, indexed_source_id=IS_ID)

    failed.assert_not_called()
    requeue.assert_called_once()
    assert requeue.call_args.kwargs["indexed_source_id"] == IS_ID
    assert "partition" in requeue.call_args.kwargs["cause"]
    assert out["status"] == "retrying_or_failed"


def test_index_source_still_fails_a_real_check_violation(harness):
    monkeypatch, requeue, failed = harness

    def _raise(**_kwargs):
        raise _integrity(
            'new row for relation "chunks" violates check constraint "tokens_positive"'
        )

    monkeypatch.setattr(indexing, "_run_index_body", _raise)

    indexing.index_source.run(KB, SRC, indexed_source_id=IS_ID)

    requeue.assert_not_called()
    failed.assert_called_once()
