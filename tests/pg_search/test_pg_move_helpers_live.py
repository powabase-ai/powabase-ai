"""The public probes other layers ask before starting or reporting a move.

``partition_exists`` and ``kb_has_rows_in_default`` are read on request paths
(a knowledge base PATCH, its detail response), so they must answer from the
catalog and the DEFAULT partition cheaply, never raise, and never leave the
caller's transaction aborted.

Same database and scratch schema as ``test_pg_bm25_live``.
"""

from __future__ import annotations

import uuid

from sqlalchemy import text

from agentic_project_service.services import pg_bm25_index as pgb
from tests.pg_search import test_pg_bm25_live as live
from tests.pg_search.test_pg_bm25_live import KB_A, KB_B, SCHEMA

migration = live.migration
engine = live.engine
scratch_schema = live.scratch_schema
session = live.session

KB_NOBODY = str(uuid.UUID("00000000-0000-4000-8000-00000000abcd"))


def test_partition_exists_is_true_only_for_an_attached_partition(engine, session):
    assert pgb.partition_exists(session, KB_A, "chunks") is False

    # An unattached clone is a move that did not finish, not a partition.
    with engine.connect() as conn:
        conn.execute(text(pgb.partition_create_ddl(KB_A, "chunks")))
        conn.commit()
    assert pgb.partition_exists(session, KB_A, "chunks") is False

    pgb.ensure_bm25_index(KB_A, engine=engine)
    assert pgb.partition_exists(session, KB_A, "chunks") is True
    assert pgb.partition_exists(session, KB_B, "chunks") is False
    assert pgb.partition_exists(session, KB_A, "full_documents") is False
    session.rollback()


def test_partition_exists_accepts_an_engine_and_a_connection(engine):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    assert pgb.partition_exists(engine, KB_A, "chunks") is True
    with engine.connect() as conn:
        assert pgb.partition_exists(conn, KB_A, "chunks") is True


def test_kb_has_rows_in_default_follows_the_rows(engine, session):
    assert pgb.kb_has_rows_in_default(session, KB_A, "chunks") is True
    assert pgb.kb_has_rows_in_default(session, KB_NOBODY, "chunks") is False
    session.rollback()

    pgb.ensure_bm25_index(KB_A, engine=engine)

    assert pgb.kb_has_rows_in_default(session, KB_A, "chunks") is False
    assert pgb.kb_has_rows_in_default(session, KB_B, "chunks") is True
    session.rollback()


def test_the_probes_never_raise_and_leave_the_callers_transaction_usable(engine, session):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP TABLE {SCHEMA}.chunks CASCADE"))

    assert pgb.kb_has_rows_in_default(session, KB_A, "chunks") is None
    assert pgb.partition_exists(session, "not-a-uuid", "chunks") is False
    assert pgb.partition_exists(session, KB_A, "doc2json_documents") is False
    assert pgb.kb_has_rows_in_default(session, KB_A, "doc2json_documents") is None
    assert session.execute(text("SELECT 1")).scalar() == 1
    session.rollback()
