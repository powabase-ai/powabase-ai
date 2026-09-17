"""A knowledge base PATCH never starts a table-wide move, checked for real.

The PATCH route may dispatch the pg_search index build only when that build
cannot move rows out of the item table's DEFAULT partition: the knowledge
base already has its own partition, or it has no rows in DEFAULT. Here the
real route (Flask test client) asks the real catalog and DEFAULT probes;
only the Celery dispatch is replaced -- by running the build it would have
queued, synchronously, while the request is still in flight. That build
needs locks on DEFAULT, so it also proves the request is no longer holding
the lock its probe of DEFAULT took.

Same database and scratch schema as ``test_pg_bm25_live``.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from flask import Flask, jsonify
from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.routes import knowledge_bases as kb_route
from agentic_project_service.services import pg_bm25_index as pgb
from tests.pg_search import test_pg_bm25_live as live
from tests.pg_search.test_pg_bm25_live import KB_A, KB_A_DOCS, KB_B, KB_B_DOCS, SCHEMA

migration = live.migration
engine = live.engine
scratch_schema = live.scratch_schema
session = live.session

KB_EMPTY = "9a3c1f2e-5b7d-4e8f-a1b2-c3d4e5f6a7b8"


@pytest.fixture
def client(engine, monkeypatch):
    """The knowledge-base blueprint on an app bound to the scratch database."""
    monkeypatch.setattr(kb_route, "AI_SCHEMA", SCHEMA)
    with engine.connect() as conn:
        # The PATCH stamps updated_at; the scratch table has no other use for it.
        conn.execute(
            text(f"ALTER TABLE {SCHEMA}.knowledge_bases ADD COLUMN updated_at timestamptz")
        )
        conn.commit()
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = engine.url.render_as_string(hide_password=False)
    db.init_app(app)
    app.register_blueprint(kb_route.knowledge_bases_bp)
    with (
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "service_role"},
        ),
        # The detail response reads tables the scratch schema does not have;
        # the PATCH's own decision is what is under test.
        patch.object(
            kb_route, "get_knowledge_base", side_effect=lambda kb_id: jsonify({"id": kb_id})
        ),
    ):
        yield app.test_client()
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


def _set_method(engine, kb_id, method, *, create=False):
    config = json.dumps({"method": method, "ts_language": "german"})
    with engine.connect() as conn:
        if create:
            conn.execute(
                text(
                    f"INSERT INTO {SCHEMA}.knowledge_bases (id, indexing_config, retrieval_config) "
                    "VALUES (CAST(:id AS uuid), CAST(:ix AS jsonb), CAST(:rx AS jsonb))"
                ),
                {"id": kb_id, "ix": json.dumps({"strategy": "chunk_embed"}), "rx": config},
            )
        else:
            conn.execute(
                text(
                    f"UPDATE {SCHEMA}.knowledge_bases SET retrieval_config = CAST(:rx AS jsonb) "
                    "WHERE id = CAST(:id AS uuid)"
                ),
                {"id": kb_id, "rx": config},
            )
        conn.commit()


def _locks_on_default_held_by_others(engine) -> list[tuple[int, str]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT pid, mode FROM pg_locks "
                f"WHERE relation = '{SCHEMA}.chunks_default'::regclass "
                "AND pid <> pg_backend_pid()"
            )
        ).all()
        conn.rollback()
    return [tuple(row) for row in rows]


def _patch_to_hybrid(client, engine, kb_id):
    """PATCH the KB to hybrid; the ensure it dispatches, if any, runs inline."""
    dispatched: list[dict] = []

    def run_the_ensure(dispatched_kb_id):
        # The route swallows a failed dispatch, so record the failure here.
        record = {
            "kb_id": dispatched_kb_id,
            "locks_on_default": _locks_on_default_held_by_others(engine),
        }
        dispatched.append(record)
        try:
            record["outcome"] = pgb.ensure_bm25_index(dispatched_kb_id, engine=engine)
        except Exception as exc:
            record["outcome"] = {"status": f"raised {type(exc).__name__}: {exc}"}

    with patch.object(kb_route, "ensure_pg_bm25_index") as ensure:
        ensure.delay.side_effect = run_the_ensure
        resp = client.patch(
            f"/api/knowledge-bases/{kb_id}",
            json={"retrieval_config": {"method": "hybrid", "ts_language": "german"}},
            headers={"Authorization": "Bearer fake"},
        )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp.get_json(), dispatched


def test_a_kb_with_rows_in_default_dispatches_nothing_and_gets_a_note(client, engine, session):
    _set_method(engine, KB_A, "vector_search")

    body, dispatched = _patch_to_hybrid(client, engine, KB_A)

    assert dispatched == []
    assert f"POST /api/knowledge-bases/{KB_A}/build-bm25" in body["bm25_note"]
    assert "blocks writes" in body["bm25_note"]
    # Nothing moved, and the request left no lock on DEFAULT behind.
    assert pgb.partition_exists(session, KB_A, "chunks") is False
    assert live._rows_in(session, "chunks_default", KB_A) == len(KB_A_DOCS)
    assert _locks_on_default_held_by_others(engine) == []


def test_a_kb_whose_partition_exists_dispatches_the_ensure(client, engine, session):
    pgb.ensure_bm25_index(KB_B, engine=engine)
    assert pgb.partition_exists(session, KB_B, "chunks") is True
    session.rollback()
    _set_method(engine, KB_B, "vector_search")

    body, dispatched = _patch_to_hybrid(client, engine, KB_B)

    assert "bm25_note" not in body
    assert [d["kb_id"] for d in dispatched] == [KB_B]
    assert dispatched[0]["locks_on_default"] == []
    assert dispatched[0]["outcome"]["status"] == "ready"
    assert live._rows_in(session, pgb.partition_name(KB_B, "chunks")) == len(KB_B_DOCS)


def test_a_new_empty_kb_dispatches_the_ensure_and_gets_its_partition(client, engine, session):
    """The fast path: no rows to move. Its attach needs ACCESS EXCLUSIVE on
    DEFAULT, which a lock the request's DEFAULT probe still held would refuse."""
    _set_method(engine, KB_EMPTY, "vector_search", create=True)

    body, dispatched = _patch_to_hybrid(client, engine, KB_EMPTY)

    assert "bm25_note" not in body
    assert [d["kb_id"] for d in dispatched] == [KB_EMPTY]
    assert dispatched[0]["locks_on_default"] == []
    assert dispatched[0]["outcome"]["status"] == "ready"
    assert pgb.partition_exists(session, KB_EMPTY, "chunks") is True
    session.rollback()
