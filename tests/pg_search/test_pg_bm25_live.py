"""Database-backed checks for the pg_search BM25 path.

Unit specs show the emitted SQL is shaped right. Only a Postgres with
``pg_search`` installed shows the partitions get built, the index lands on the
partition, two knowledge bases are scored independently with their own
stemmers, the filters push down, DML is visible without a reindex, and an
adversarial query answers instead of raising.

Runs against ``PG_SEARCH_TEST_DATABASE_URL`` if set, otherwise ``DATABASE_URL``,
and skips with a reason when that server cannot offer the extension -- unless
``PG_SEARCH_REQUIRED=1``, where a missing extension fails instead, so a CI job
built to run these cannot pass by skipping them. Every test
works in a scratch schema of its own, so nothing here touches the ``ai`` schema.
The scratch schema is built unpartitioned and then converted by the real
migration, so these tests run against the shape a migrated database has.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import threading
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session

from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services import pg_bm25_index as pgb

SCHEMA = "bm25_live_test"

KB_A = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
KB_B = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
KB_C = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

SOURCE_1 = "11111111-1111-4111-8111-111111111111"
SOURCE_2 = "22222222-2222-4222-8222-222222222222"

# German for KB A, French for KB B: each partition carries its own stemmer, so
# the two must fold their own inflections and neither the other's.
KB_A_DOCS = [
    (SOURCE_1, "Die Wanderung des Bergführers wurde abgesagt"),
    (SOURCE_1, "Mehrere Wanderungen fanden am Wochenende statt"),
    (SOURCE_2, "Die Brücke wurde nach drei Jahren erneuert"),
]
KB_B_DOCS = [
    (SOURCE_1, "La randonnée en montagne a été magnifique"),
    (SOURCE_1, "Plusieurs randonnées ont eu lieu ce week-end"),
]
KB_C_DOCS = [(SOURCE_1, "Eine Wanderung in einer dritten Wissensbasis")]


class _ChunkStore(bvs.BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


class _NodeStore(bvs.BasePgVectorStore):
    TABLE = "graph_index_nodes"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


class _FullDocumentStore(bvs.BasePgVectorStore):
    TABLE = "full_documents"
    TEXT_COL = "summary"
    SEARCH_TEXT_COL = "summary"


@pytest.fixture(scope="module")
def migration():
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "0031_partition_bm25_item_tables.py"
    )
    spec = importlib.util.spec_from_file_location("mig_0031_live", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def engine():
    dsn = os.environ.get("PG_SEARCH_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        if os.environ.get("PG_SEARCH_REQUIRED") == "1":
            pytest.fail("PG_SEARCH_REQUIRED=1 but no PG_SEARCH_TEST_DATABASE_URL or DATABASE_URL")
        pytest.skip("no PG_SEARCH_TEST_DATABASE_URL or DATABASE_URL to test pg_search against")
    eng = create_engine(dsn)
    where = eng.url.render_as_string(hide_password=True)
    try:
        with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            # pg_search hard-requires vector ("required extension "vector" is not
            # installed"); without this every test here skipped on a fresh
            # database, and a skip reads as green.
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_search"))
    except Exception as exc:
        eng.dispose()
        reason = f"pg_search is not available on {where}: {str(exc).splitlines()[0]}"
        if os.environ.get("PG_SEARCH_REQUIRED") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    yield eng
    eng.dispose()


_KB_LANGUAGES = {KB_A: "german", KB_B: "french", KB_C: "german"}


@pytest.fixture(autouse=True)
def scratch_schema(engine, migration, monkeypatch):
    """The item tables as a migrated database has them: partitioned parents.

    ``pg_bm25_index`` writes its DDL against ``AI_SCHEMA``; pointing that at a
    scratch schema is what lets these tests exercise the real statements
    without touching a real table. The tables are created unpartitioned and
    converted by revision 0031, so the starting state is the real one: every
    knowledge base's rows in the DEFAULT partition, nobody indexed yet.
    """
    monkeypatch.setattr(pgb, "AI_SCHEMA", SCHEMA)
    pgb.reset_pg_bm25_caches()
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))
        conn.execute(
            text(f"""
                CREATE TABLE {SCHEMA}.knowledge_bases (
                    id uuid PRIMARY KEY,
                    indexing_config jsonb,
                    retrieval_config jsonb
                )
            """)
        )
        conn.execute(
            text(f"""
                CREATE TABLE {SCHEMA}.chunks (
                    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    knowledge_base_id uuid NOT NULL
                        REFERENCES {SCHEMA}.knowledge_bases(id) ON DELETE CASCADE,
                    indexed_source_id uuid,
                    source_id uuid,
                    text text,
                    meta jsonb DEFAULT '{{}}'::jsonb
                )
            """)
        )
        conn.execute(
            text(f"""
                CREATE TABLE {SCHEMA}.full_documents (
                    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    knowledge_base_id uuid NOT NULL
                        REFERENCES {SCHEMA}.knowledge_bases(id) ON DELETE CASCADE,
                    source_id uuid,
                    summary text,
                    meta jsonb DEFAULT '{{}}'::jsonb
                )
            """)
        )
        conn.execute(
            text(f"""
                CREATE TABLE {SCHEMA}.graph_index_nodes (
                    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    knowledge_base_id uuid NOT NULL
                        REFERENCES {SCHEMA}.knowledge_bases(id) ON DELETE CASCADE,
                    source_id uuid,
                    title text,
                    text text,
                    meta jsonb DEFAULT '{{}}'::jsonb
                )
            """)
        )
        for kb_id, language in _KB_LANGUAGES.items():
            conn.execute(
                text(f"""
                    INSERT INTO {SCHEMA}.knowledge_bases (id, indexing_config, retrieval_config)
                    VALUES (CAST(:id AS uuid), CAST(:ix AS jsonb), CAST(:rx AS jsonb))
                """),
                {
                    "id": kb_id,
                    "ix": json.dumps({"strategy": "chunk_embed"}),
                    "rx": json.dumps({"method": "hybrid", "ts_language": language}),
                },
            )
        for kb_id, docs in ((KB_A, KB_A_DOCS), (KB_B, KB_B_DOCS), (KB_C, KB_C_DOCS)):
            for source_id, body in docs:
                conn.execute(
                    text(f"""
                        INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text, meta)
                        VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), :body,
                                CAST(:meta AS jsonb))
                    """),
                    {
                        "kb": kb_id,
                        "src": source_id,
                        "body": body,
                        "meta": json.dumps({"lang": "de" if kb_id != KB_B else "fr"}),
                    },
                )
        migration.partition_item_tables(conn, schema=SCHEMA)
    yield
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    pgb.reset_pg_bm25_caches()


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


def _store(session, kb_id=KB_A, cls=_ChunkStore):
    return cls(db_session=session, knowledge_base_id=kb_id, schema=SCHEMA)


def _search(session, query, kb_id=KB_A, cls=_ChunkStore, **kwargs):
    """Search, then close the read transaction.

    Every helper here releases the session's transaction before returning. A
    read left open holds an ACCESS SHARE lock on the DEFAULT partition, which
    would block the ATTACH PARTITION inside ``create_partition`` (and a
    DROP INDEX CONCURRENTLY) until the test itself timed out -- these tests
    drive both sides of that lock from one process.
    """
    try:
        return asyncio.run(_store(session, kb_id, cls).pg_bm25_search(query, **kwargs))
    finally:
        session.rollback()


def _indexdef(session, kb_id=KB_A, item_table="chunks") -> str | None:
    try:
        row = session.execute(
            text("SELECT indexdef FROM pg_indexes WHERE schemaname = :s AND indexname = :n"),
            {"s": SCHEMA, "n": pgb.bm25_index_name(kb_id, item_table)},
        ).first()
        return row[0] if row else None
    finally:
        session.rollback()


def _set_strategy(session, kb_id, strategy):
    session.execute(
        text(
            f"UPDATE {SCHEMA}.knowledge_bases SET indexing_config = CAST(:ix AS jsonb) "
            "WHERE id = CAST(:id AS uuid)"
        ),
        {"ix": json.dumps({"strategy": strategy}), "id": kb_id},
    )
    session.commit()


def _set_language(session, kb_id, language):
    session.execute(
        text(
            f"UPDATE {SCHEMA}.knowledge_bases SET retrieval_config = CAST(:rx AS jsonb) "
            "WHERE id = CAST(:id AS uuid)"
        ),
        {"rx": json.dumps({"method": "hybrid", "ts_language": language}), "id": kb_id},
    )
    session.commit()


def _rows_in(session, relation, kb_id=None) -> int:
    sql = f"SELECT count(*) FROM {SCHEMA}.{relation}"
    params: dict = {}
    if kb_id:
        sql += " WHERE knowledge_base_id = CAST(:kb AS uuid)"
        params["kb"] = kb_id
    try:
        return session.execute(text(sql), params).scalar()
    finally:
        session.rollback()


# ---------------------------------------------------------------------------
# Partition creation
# ---------------------------------------------------------------------------


def test_ensure_moves_the_kbs_rows_out_of_default_into_its_own_partition(engine, session):
    partition = pgb.partition_name(KB_A, "chunks")
    assert _rows_in(session, "chunks_default", KB_A) == 3

    result = pgb.ensure_bm25_index(KB_A, engine=engine)

    assert result["status"] == "ready"
    assert result["partition"] == partition
    assert result["partition_created"] is True
    assert result["rows_moved"] == 3
    # Rows moved, none lost, and the parent still sees all of them.
    assert _rows_in(session, "chunks_default", KB_A) == 0
    assert _rows_in(session, partition) == 3
    assert _rows_in(session, "chunks", KB_A) == 3
    assert _rows_in(session, "chunks") == len(KB_A_DOCS) + len(KB_B_DOCS) + len(KB_C_DOCS)
    # Every other knowledge base is untouched.
    assert _rows_in(session, "chunks_default", KB_B) == 2


def test_the_new_partition_carries_the_default_partitions_key_and_foreign_keys(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    partition = pgb.partition_name(KB_A, "chunks")

    constraints = {
        row[0]: row[1]
        for row in session.execute(
            text(
                "SELECT contype, pg_get_constraintdef(oid) FROM pg_constraint "
                f"WHERE conrelid = '{SCHEMA}.{partition}'::regclass"
            )
        ).all()
    }
    assert constraints["p"] == "PRIMARY KEY (id)"
    assert "REFERENCES" in constraints["f"]

    # And the local key really enforces uniqueness inside the partition.
    existing = session.execute(text(f"SELECT id FROM {SCHEMA}.{partition} LIMIT 1")).scalar()
    with pytest.raises(Exception, match="duplicate key value"):
        session.execute(
            text(f"""
                INSERT INTO {SCHEMA}.chunks (id, knowledge_base_id, source_id, text)
                VALUES (CAST(:id AS uuid), CAST(:kb AS uuid), CAST(:src AS uuid), 'dup')
            """),
            {"id": str(existing), "kb": KB_A, "src": SOURCE_1},
        )
    session.rollback()


def test_the_new_partition_is_reachable_by_the_same_roles_as_the_parent(engine, session):
    """A fresh partition starts with no privileges and RLS off.

    Without mirroring, either the roles that can read the parent cannot read the
    relation the search path names, or the partition is readable past the rules
    the parent enforces. Both are mirrored from the parent instead.
    """
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(
            text("DO $$ BEGIN CREATE ROLE bm25_live_reader; EXCEPTION WHEN OTHERS THEN END $$")
        )
        conn.execute(text(f"GRANT SELECT, INSERT ON {SCHEMA}.chunks TO bm25_live_reader"))
        conn.execute(text(f"ALTER TABLE {SCHEMA}.chunks ENABLE ROW LEVEL SECURITY"))

    pgb.ensure_bm25_index(KB_A, engine=engine)
    partition = pgb.partition_name(KB_A, "chunks")

    def _privileges(relation):
        try:
            return {
                row[0]
                for row in session.execute(
                    text(
                        "SELECT privilege_type FROM information_schema.role_table_grants "
                        "WHERE table_schema = :s AND table_name = :t AND grantee = :g"
                    ),
                    {"s": SCHEMA, "t": relation, "g": "bm25_live_reader"},
                ).all()
            }
        finally:
            session.rollback()

    assert _privileges(partition) == _privileges("chunks") == {"SELECT", "INSERT"}
    try:
        assert (
            session.execute(
                text(
                    f"SELECT relrowsecurity FROM pg_class WHERE oid = '{SCHEMA}.{partition}'::regclass"
                )
            ).scalar()
            is True
        )
    finally:
        session.rollback()


def test_a_policy_gated_role_can_read_the_new_partition_by_name(engine, session):
    """B5: the partition gets the parent's RLS policies, not just the RLS flag.

    Mirrors the self-hosted grant (``FOR SELECT TO authenticated``) with a role
    that has no BYPASSRLS. The search path names the partition, and Postgres
    applies only the queried relation's policies -- so with the flag copied and
    no policy, this role would read zero rows.
    """
    role = "bm25_live_authenticated"
    partition = pgb.partition_name(KB_A, "chunks")
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(
            text(
                f"DO $$ BEGIN CREATE ROLE {role} NOLOGIN NOBYPASSRLS; "
                "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
        )
        conn.execute(text(f"GRANT USAGE ON SCHEMA {SCHEMA} TO {role}"))
        conn.execute(text(f"GRANT SELECT ON {SCHEMA}.chunks TO {role}"))
        conn.execute(text(f"ALTER TABLE {SCHEMA}.chunks ENABLE ROW LEVEL SECURITY"))
        conn.execute(
            text(
                f"CREATE POLICY auth_read_chunks ON {SCHEMA}.chunks "
                f"FOR SELECT TO {role} USING (true)"
            )
        )
        conn.execute(
            text(
                f"CREATE POLICY no_foreign_kb ON {SCHEMA}.chunks AS RESTRICTIVE "
                f"FOR SELECT TO {role} USING (knowledge_base_id <> '{KB_B}'::uuid)"
            )
        )

    pgb.create_partition(engine, KB_A, "chunks")

    try:
        with engine.connect() as conn:
            conn.execute(text(f"SET ROLE {role}"))
            assert conn.execute(text(f"SELECT count(*) FROM {SCHEMA}.{partition}")).scalar() == 3
            conn.rollback()
        policies = {
            (row[0], row[1], row[2])
            for row in session.execute(
                text(
                    "SELECT policyname, permissive, cmd FROM pg_policies "
                    "WHERE schemaname = :s AND tablename = :t"
                ),
                {"s": SCHEMA, "t": partition},
            ).all()
        }
        session.rollback()
        assert policies == {
            ("auth_read_chunks", "PERMISSIVE", "SELECT"),
            ("no_foreign_kb", "RESTRICTIVE", "SELECT"),
        }
    finally:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
            conn.execute(text(f"DROP OWNED BY {role}"))
            conn.execute(text(f"DROP ROLE IF EXISTS {role}"))


def _seed(session, kb_id, count, prefix="Wanderung Nummer"):
    """Bulk-insert ``count`` rows for one KB. One statement, so tests stay quick."""
    session.execute(
        text(f"""
            INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text)
            SELECT CAST(:kb AS uuid), CAST(:src AS uuid), :prefix || ' ' || g
            FROM generate_series(1, :count) g
        """),
        {"kb": kb_id, "src": SOURCE_1, "prefix": prefix, "count": count},
    )
    session.commit()


def test_partition_creation_moves_every_row_in_one_go(engine, session):
    _seed(session, KB_A, 500)

    move = pgb.create_partition(engine, KB_A, "chunks")

    assert move["rows_moved"] == 503
    assert move["writes_blocked_seconds"] >= 0
    assert _rows_in(session, "chunks", KB_A) == 503
    assert _rows_in(session, "chunks_default", KB_A) == 0
    assert _rows_in(session, pgb.partition_name(KB_A, "chunks")) == 503
    analyzed = session.execute(
        text(
            "SELECT last_analyze IS NOT NULL FROM pg_stat_user_tables "
            "WHERE schemaname = :s AND relname = :r"
        ),
        {"s": SCHEMA, "r": pgb.partition_name(KB_A, "chunks")},
    ).scalar()
    session.rollback()
    assert analyzed is True


def _hold_the_move_open(monkeypatch, moving: threading.Event, seconds: float) -> None:
    """Stall ``create_partition`` after its rows have moved, before ATTACH.

    The ATTACH statement is built right before it runs, inside the move's
    transaction, so wrapping the builder is a seam that needs no test hook in
    the service. Restored by the monkeypatch fixture at teardown.
    """
    real_attach = pgb.partition_attach_ddl

    def _stalled(*args, **kwargs):
        moving.set()
        time.sleep(seconds)
        return real_attach(*args, **kwargs)

    monkeypatch.setattr(pgb, "partition_attach_ddl", _stalled)


def test_updates_and_deletes_through_the_parent_during_a_move_are_honoured(
    engine, session, monkeypatch
):
    """The B1 regression: writes issued mid-move were silently lost.

    Written exactly as the app writes -- a per-row ``UPDATE ... WHERE id`` and a
    ``DELETE ... WHERE indexed_source_id`` through the parent -- while the move
    is between moving the rows and attaching the partition. Every update must be
    present afterwards and no deleted row may come back.
    """
    doomed_source = str(uuid.uuid4())
    session.execute(
        text(f"""
            INSERT INTO {SCHEMA}.chunks (knowledge_base_id, indexed_source_id, source_id, text)
            SELECT CAST(:kb AS uuid),
                   CASE WHEN g % 10 = 0 THEN CAST(:doomed AS uuid) ELSE NULL END,
                   CAST(:src AS uuid), 'Wanderweg Abschnitt ' || g
            FROM generate_series(1, 2000) g
        """),
        {"kb": KB_A, "doomed": doomed_source, "src": SOURCE_1},
    )
    session.commit()
    targets = [
        str(row[0])
        for row in session.execute(
            text(
                f"SELECT id FROM {SCHEMA}.chunks WHERE knowledge_base_id = CAST(:kb AS uuid) "
                "AND indexed_source_id IS NULL ORDER BY id LIMIT 40"
            ),
            {"kb": KB_A},
        ).all()
    ]
    session.rollback()

    moving = threading.Event()
    _hold_the_move_open(monkeypatch, moving, 1.0)
    update_counts: list[int] = []
    delete_counts: list[int] = []
    errors: list[str] = []

    def updater():
        moving.wait(timeout=30)
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            for chunk_id in targets:
                try:
                    update_counts.append(
                        conn.execute(
                            text(
                                f"UPDATE {SCHEMA}.chunks SET text = 'Wegsperrung wegen Sturm' "
                                "WHERE id = CAST(:id AS uuid)"
                            ),
                            {"id": chunk_id},
                        ).rowcount
                    )
                except Exception as exc:
                    errors.append(str(exc).splitlines()[0])

    def deleter():
        moving.wait(timeout=30)
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            try:
                delete_counts.append(
                    conn.execute(
                        text(
                            f"DELETE FROM {SCHEMA}.chunks "
                            "WHERE indexed_source_id = CAST(:source AS uuid)"
                        ),
                        {"source": doomed_source},
                    ).rowcount
                )
            except Exception as exc:
                errors.append(str(exc).splitlines()[0])

    threads = [threading.Thread(target=f, daemon=True) for f in (updater, deleter)]
    for thread in threads:
        thread.start()
    pgb.create_partition(engine, KB_A, "chunks")
    for thread in threads:
        thread.join(timeout=60)

    assert errors == []
    assert update_counts == [1] * len(targets)
    assert delete_counts == [200]
    updated = session.execute(
        text(
            f"SELECT count(*) FROM {SCHEMA}.chunks WHERE text = 'Wegsperrung wegen Sturm' "
            "AND id = ANY(CAST(:ids AS uuid[]))"
        ),
        {"ids": "{" + ",".join(targets) + "}"},
    ).scalar()
    session.rollback()
    assert updated == len(targets)
    resurrected = session.execute(
        text(f"SELECT count(*) FROM {SCHEMA}.chunks WHERE indexed_source_id = CAST(:s AS uuid)"),
        {"s": doomed_source},
    ).scalar()
    session.rollback()
    assert resurrected == 0
    assert _rows_in(session, "chunks", KB_A) == len(KB_A_DOCS) + 1800
    assert _rows_in(session, "chunks_default", KB_A) == 0


def test_a_reader_through_the_parent_sees_every_row_throughout_the_move(
    engine, session, monkeypatch
):
    """The move is one transaction, so a reader never sees it half-done.

    And reads are not blocked while the rows move: the stall below holds the
    transaction open for a second, and every read completes well inside it.
    """
    _seed(session, KB_A, 2000)
    moving = threading.Event()
    _hold_the_move_open(monkeypatch, moving, 1.0)
    counts: list[int] = []
    latencies: list[float] = []
    stop = threading.Event()

    def reader():
        moving.wait(timeout=30)
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            while not stop.is_set():
                issued = time.monotonic()
                counts.append(
                    conn.execute(
                        text(
                            f"SELECT count(*) FROM {SCHEMA}.chunks "
                            "WHERE knowledge_base_id = CAST(:kb AS uuid)"
                        ),
                        {"kb": KB_A},
                    ).scalar()
                )
                latencies.append(time.monotonic() - issued)
                time.sleep(0.01)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        pgb.create_partition(engine, KB_A, "chunks")
        time.sleep(0.1)
    finally:
        stop.set()
        thread.join(timeout=30)

    assert len(counts) >= 10, counts
    assert set(counts) == {2003}
    assert max(latencies) < 0.5, max(latencies)


def test_a_concurrent_insert_during_the_move_waits_and_then_lands(engine, session, monkeypatch):
    """Inserts through the parent wait for the move and then land in the partition.

    They wait on the parent's lock, so they are planned after the ATTACH commits
    and routed straight into the new partition -- none fails, none is lost.
    """
    moving = threading.Event()
    _hold_the_move_open(monkeypatch, moving, 0.5)
    landed: list[int] = []
    failed: list[str] = []

    def writer():
        moving.wait(timeout=30)
        for n in range(20):
            try:
                with engine.begin() as conn:
                    conn.execute(
                        text(f"""
                            INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text)
                            VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), :body)
                        """),
                        {"kb": KB_A, "src": SOURCE_1, "body": f"Wanderung spaet {n}"},
                    )
                landed.append(n)
            except Exception as exc:
                failed.append(str(exc).splitlines()[0])

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    pgb.create_partition(engine, KB_A, "chunks")
    thread.join(timeout=30)

    assert failed == []
    assert len(landed) == 20
    partition = pgb.partition_name(KB_A, "chunks")
    assert _rows_in(session, "chunks", KB_A) == len(KB_A_DOCS) + 20
    assert _rows_in(session, partition) == len(KB_A_DOCS) + 20
    assert _rows_in(session, "chunks_default", KB_A) == 0


def test_two_knowledge_bases_can_be_indexed_at_the_same_time(engine, session):
    """The deadlock this serialisation exists to remove.

    Observed on a real project: two ``build-bm25`` calls for two knowledge bases
    whose rows both still sat in the shared DEFAULT partition failed with
    ``deadlock detected`` -- each held SHARE on that partition and then asked for
    the ACCESS EXCLUSIVE its own ATTACH needs. Serialised per item table, the
    second caller waits and both finish.
    """
    _seed(session, KB_A, 400, prefix="Wanderung A")
    _seed(session, KB_B, 400, prefix="rando B")

    results: dict[str, dict] = {}
    errors: list[str] = []
    start = threading.Barrier(2, timeout=30)

    def build(kb_id):
        try:
            start.wait()
            results[kb_id] = pgb.ensure_bm25_index(kb_id, engine=engine)
        except Exception as exc:
            errors.append(f"{kb_id}: {str(exc).splitlines()[0]}")

    threads = [threading.Thread(target=build, args=(kb,), daemon=True) for kb in (KB_A, KB_B)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert errors == []
    assert [results[kb]["status"] for kb in (KB_A, KB_B)] == ["ready", "ready"]
    assert _rows_in(session, pgb.partition_name(KB_A, "chunks")) == 403
    assert _rows_in(session, pgb.partition_name(KB_B, "chunks")) == 402
    assert _rows_in(session, "chunks_default", KB_A) == 0
    assert _rows_in(session, "chunks_default", KB_B) == 0
    assert _rows_in(session, "chunks") == 403 + 402 + len(KB_C_DOCS)
    # Both indexes exist and answer only their own knowledge base.
    assert _indexdef(session, KB_A) is not None
    assert _indexdef(session, KB_B) is not None
    assert len(_search(session, "rando", kb_id=KB_B, top_k=5)) == 5
    assert _search(session, "rando", kb_id=KB_A, top_k=5) == []


def test_a_second_caller_is_told_to_retry_rather_than_blocking_for_ever(
    engine, session, monkeypatch
):
    """With no patience left, contention is a retryable outcome, not a 500."""
    monkeypatch.setattr(pgb, "PARTITION_BUILD_LOCK_WAIT_SECONDS", 0.0)
    relation = pgb.partition_build_lock_relation("chunks")
    with engine.connect() as holder:
        holder.execute(text(pgb.partition_build_lock_sql()), {"relation": relation})
        holder.commit()
        try:
            out = pgb.ensure_bm25_index(KB_A, engine=engine)
        finally:
            # Released by hand, and this is the point: the lock is session
            # scoped, so returning the connection to the pool does NOT drop it.
            holder.execute(text(pgb.partition_build_unlock_sql()), {"relation": relation})
            holder.commit()

    assert out["status"] == "skipped"
    assert out["reason"] == "partition_build_in_progress"
    assert _indexdef(session) is None
    # Nothing half-done: the knowledge base's rows never moved.
    assert _rows_in(session, "chunks_default", KB_A) == 3
    # And the next caller is not locked out by the one that just declined.
    assert pgb.ensure_bm25_index(KB_A, engine=engine)["status"] == "ready"


def test_a_continuous_writer_is_held_for_the_move_and_loses_nothing(engine, session):
    """The price of the atomic move, measured: writers wait, and only that long.

    A writer inserting continuously for the same knowledge base is blocked while
    the rows move and resumes when the move commits. No write fails and none is
    lost, and no single write waited much longer than the window the move
    itself reports.
    """
    _seed(session, KB_A, 20_000)

    landed: list[float] = []
    failed: list[str] = []
    stop = threading.Event()

    def writer():
        n = 0
        while not stop.is_set():
            issued = time.monotonic()
            try:
                with engine.begin() as conn:
                    conn.execute(
                        text(f"""
                            INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text)
                            VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), :body)
                        """),
                        {"kb": KB_A, "src": SOURCE_1, "body": f"Wanderung spaet {n}"},
                    )
                landed.append(time.monotonic() - issued)
            except Exception as exc:
                failed.append(str(exc).splitlines()[0])
            n += 1
            time.sleep(0.005)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    time.sleep(0.05)
    try:
        move = pgb.create_partition(engine, KB_A, "chunks")
        time.sleep(0.05)
    finally:
        stop.set()
        thread.join(timeout=30)

    assert failed == []
    assert len(landed) >= 5
    # The longest write wait is the move's own blocking window, plus slack.
    assert max(landed) < move["writes_blocked_seconds"] + 0.5, (max(landed), move)
    expected = len(KB_A_DOCS) + 20_000 + len(landed)
    assert _rows_in(session, "chunks", KB_A) == expected
    assert _rows_in(session, "chunks_default", KB_A) == 0
    assert _rows_in(session, pgb.partition_name(KB_A, "chunks")) == expected
    assert move["rows_moved"] >= 20_000 + len(KB_A_DOCS)


def _move_check_names(session) -> list[str]:
    try:
        return [
            row[0]
            for row in session.execute(
                text(
                    "SELECT conname FROM pg_constraint "
                    f"WHERE conrelid = '{SCHEMA}.chunks_default'::regclass "
                    "AND conname LIKE 'bm25\\_move\\_%' ORDER BY conname"
                )
            ).all()
        ]
    finally:
        session.rollback()


def test_the_attach_scans_neither_the_new_partition_nor_default(engine, session):
    """What keeps the ATTACH itself short, pinned by Postgres' own debug notices.

    ATTACH PARTITION proves two things before it commits, each by a full scan
    under ACCESS EXCLUSIVE unless an existing *validated* constraint already
    implies it: that the new partition holds only its own knowledge base (the
    CHECK on the clone), and that DEFAULT holds none of it (a temporary
    ``CHECK (knowledge_base_id <> kb)`` on DEFAULT, validated under the move's
    SHARE lock). With both in place Postgres says so at DEBUG1 and never logs
    ``verifying table``. A timing bound could not tell these apart on a test
    sized DEFAULT; the notice can.
    """
    _seed(session, KB_B, 2_000, prefix="rando B")
    _seed(session, KB_A, 2_000)
    notices: list[str] = []
    debug_engine = create_engine(
        engine.url, connect_args={"options": "-c client_min_messages=debug1"}
    )

    @event.listens_for(debug_engine, "connect")
    def _capture(dbapi_connection, _record):
        dbapi_connection.add_notice_handler(lambda diag: notices.append(diag.message_primary))

    try:
        move = pgb.create_partition(debug_engine, KB_A, "chunks")
    finally:
        debug_engine.dispose()

    partition = pgb.partition_name(KB_A, "chunks")
    assert move["rows_moved"] == 2_003
    assert (
        f'partition constraint for table "{partition}" is implied by existing constraints'
        in notices
    ), notices
    assert (
        'updated partition constraint for default partition "chunks_default" is implied '
        "by existing constraints" in notices
    ), notices
    # Nothing is scanned by the ATTACH: no "verifying table" after its first notice.
    attach_notices = notices[
        notices.index(
            f'partition constraint for table "{partition}" is implied by existing constraints'
        ) :
    ]
    assert not [n for n in attach_notices if n.startswith("verifying table")], attach_notices
    # The CHECK on the clone stays; the temporary one on DEFAULT is gone.
    constraints = [
        row[0]
        for row in session.execute(
            text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                f"WHERE conrelid = '{SCHEMA}.{partition}'::regclass AND contype = 'c'"
            )
        ).all()
    ]
    session.rollback()
    assert constraints == [f"CHECK ((knowledge_base_id = '{KB_A}'::uuid))"]
    assert _move_check_names(session) == []
    # Rows of the knowledge base can be written through the parent again.
    _seed(session, KB_A, 5)
    assert _rows_in(session, partition) == 2_008


def test_an_unrelated_check_on_default_does_not_stop_the_kb_check(engine, session):
    """I13: the clone inherits every CHECK on DEFAULT; its own kb check must
    still be added, or the ATTACH would scan the new partition."""
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(
            text(
                f"ALTER TABLE {SCHEMA}.chunks_default "
                "ADD CONSTRAINT chunks_text_not_huge CHECK (length(text) < 1000000)"
            )
        )

    pgb.create_partition(engine, KB_A, "chunks")

    partition = pgb.partition_name(KB_A, "chunks")
    names = {
        row[0]
        for row in session.execute(
            text(
                "SELECT conname FROM pg_constraint "
                f"WHERE conrelid = '{SCHEMA}.{partition}'::regclass AND contype = 'c'"
            )
        ).all()
    }
    session.rollback()
    assert f"{partition}_kb_check" in names


def test_a_failed_move_does_not_leave_the_default_check_behind(engine, session, monkeypatch):
    """The temporary CHECK refuses this KB's rows in DEFAULT; a failed move
    must not leave it there, or every later write for the KB would fail."""

    def _explode(*_args, **_kwargs):
        raise RuntimeError("simulated failure before the attach")

    real_attach = pgb.partition_attach_ddl
    monkeypatch.setattr(pgb, "partition_attach_ddl", _explode)
    with pytest.raises(RuntimeError, match="simulated failure"):
        pgb.create_partition(engine, KB_A, "chunks")
    monkeypatch.setattr(pgb, "partition_attach_ddl", real_attach)

    assert _move_check_names(session) == []
    _seed(session, KB_A, 5)
    assert _rows_in(session, "chunks_default", KB_A) == len(KB_A_DOCS) + 5


def test_a_leftover_default_check_from_a_crashed_move_is_cleared_by_the_next_move(engine, session):
    """A worker killed mid-move cannot run its cleanup. The next move on the
    table (under the same build lock, so no move is in flight) clears it."""
    kb_b_hex = uuid.UUID(KB_B).hex
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(
            text(
                f"ALTER TABLE {SCHEMA}.chunks_default ADD CONSTRAINT bm25_move_{kb_b_hex} "
                f"CHECK (knowledge_base_id <> '{KB_B}') NOT VALID"
            )
        )

    pgb.create_partition(engine, KB_A, "chunks")

    assert _move_check_names(session) == []
    _seed(session, KB_B, 5, prefix="rando B")
    assert _rows_in(session, "chunks_default", KB_B) == len(KB_B_DOCS) + 5


def test_the_move_gives_up_on_its_lock_instead_of_stalling_writers(engine, session, monkeypatch):
    """The move's lock wait is bounded (mutation M04).

    A writer transaction left open on the parent holds ROW EXCLUSIVE, which the
    move's SHARE lock conflicts with. Waiting on it unbounded would queue every
    other writer of the table behind the move for as long as that transaction
    lives. The move must time out, roll back whole, drop its DEFAULT check and
    release the build lock so a retry can succeed.
    """
    monkeypatch.setattr(pgb, "MOVE_LOCK_TIMEOUT_MS", 200)
    outcome: dict = {}

    def move():
        try:
            outcome["result"] = pgb.create_partition(engine, KB_A, "chunks")
        except Exception as exc:
            outcome["error"] = exc

    holder = engine.connect()
    try:
        holder.execute(
            text(
                f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text) "
                f"VALUES ('{KB_B}', '{SOURCE_1}', 'offene Transaktion')"
            )
        )
        thread = threading.Thread(target=move, daemon=True)
        thread.start()
        thread.join(timeout=10)
        stalled = thread.is_alive()
    finally:
        holder.rollback()
        holder.close()
    thread.join(timeout=30)

    assert not stalled, "the move waited on its lock without a timeout"
    assert pgb.is_transient_db_error(outcome["error"]), outcome
    assert _rows_in(session, "chunks_default", KB_A) == len(KB_A_DOCS)
    assert _rows_in(session, pgb.partition_name(KB_A, "chunks")) == 0
    assert _move_check_names(session) == []
    assert pgb.create_partition(engine, KB_A, "chunks")["rows_moved"] == len(KB_A_DOCS)


def test_a_crashed_move_is_resumed_with_no_row_lost_or_duplicated(engine, session, monkeypatch):
    """A failure between moving the rows and the ATTACH rolls the move back whole.

    The copy, the delete and the ATTACH are one transaction, so the crash leaves
    every row where it started and the clone empty and unattached. ``ensure``
    has to notice the unattached clone and finish the job.
    """
    _seed(session, KB_A, 400)

    def _explode(*_args, **_kwargs):
        raise RuntimeError("simulated crash just before the attach")

    # Restored by hand, not with monkeypatch.undo(): undo() would also revert the
    # fixture's AI_SCHEMA patch and send the rest of this test at the real schema.
    real_attach = pgb.partition_attach_ddl
    monkeypatch.setattr(pgb, "partition_attach_ddl", _explode)
    with pytest.raises(RuntimeError, match="simulated crash"):
        pgb.ensure_bm25_index(KB_A, engine=engine)
    monkeypatch.setattr(pgb, "partition_attach_ddl", real_attach)

    partition = pgb.partition_name(KB_A, "chunks")
    assert _rows_in(session, partition) == 0, "the move was not rolled back whole"
    assert _rows_in(session, "chunks_default", KB_A) == 403
    assert _rows_in(session, "chunks", KB_A) == 403
    assert _indexdef(session) is None

    result = pgb.ensure_bm25_index(KB_A, engine=engine)

    assert result["status"] == "ready"
    assert result["partition_created"] is True
    assert _rows_in(session, partition) == 403
    assert _rows_in(session, "chunks_default", KB_A) == 0
    assert _rows_in(session, "chunks", KB_A) == 403
    assert _indexdef(session) is not None
    assert len(_search(session, "Wanderung", top_k=5)) == 5


def test_ensure_repairs_an_index_whose_concurrent_build_failed(engine, session, caplog):
    """B3: an INVALID bm25 index with no build running is dropped and rebuilt.

    ``indisvalid = false`` is exactly what a cancelled, killed or failed
    CREATE INDEX CONCURRENTLY leaves behind.
    """
    assert pgb.ensure_bm25_index(KB_A, engine=engine)["status"] == "ready"
    name = pgb.bm25_index_name(KB_A, "chunks")
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(
            text(
                f"UPDATE pg_index SET indisvalid = false WHERE indexrelid = '{SCHEMA}.{name}'::regclass"
            )
        )
    assert pgb.bm25_index_state(session, KB_A, "chunks") == "building"
    session.rollback()
    pgb.reset_pg_bm25_caches()

    with caplog.at_level("WARNING"):
        result = pgb.ensure_bm25_index(KB_A, engine=engine)

    assert result["status"] == "ready"
    assert pgb.bm25_index_state(session, KB_A, "chunks") == "ready"
    session.rollback()
    assert name in caplog.text
    assert len(_search(session, "Wanderung", top_k=5)) == 2


def test_a_kb_with_no_strategy_key_is_indexed_as_chunk_embed(engine, session):
    """Search treats a missing strategy as chunk_embed, so the build must too."""
    session.execute(
        text(
            f"UPDATE {SCHEMA}.knowledge_bases SET indexing_config = '{{}}'::jsonb "
            "WHERE id = CAST(:id AS uuid)"
        ),
        {"id": KB_A},
    )
    session.commit()

    result = pgb.ensure_bm25_index(KB_A, engine=engine)

    assert result["status"] == "ready"
    assert result["item_table"] == "chunks"


def test_drop_partition_refuses_to_wait_for_ever_behind_a_long_reader(engine, session, monkeypatch):
    """I2: DETACH wants ACCESS EXCLUSIVE on the parent; a reader holding its
    transaction open must make it give up (retryable), not stall the table."""
    pgb.ensure_bm25_index(KB_A, engine=engine)
    monkeypatch.setattr(pgb, "MOVE_LOCK_TIMEOUT_MS", 200)

    with engine.connect() as reader:
        reader.execute(text(f"SELECT count(*) FROM {SCHEMA}.chunks")).scalar()
        started = time.monotonic()
        with pytest.raises(Exception, match="lock timeout") as caught:
            pgb.drop_partition(engine, KB_A, "chunks")
        waited = time.monotonic() - started
        reader.rollback()

    assert pgb.is_transient_db_error(caught.value)
    assert waited < 5
    # Nothing changed, and the build lock was released for the retry.
    assert _rows_in(session, pgb.partition_name(KB_A, "chunks")) == 3
    assert pgb.drop_partition(engine, KB_A, "chunks") is True
    assert _rows_in(session, "chunks", KB_A) == 3


def test_ensure_is_idempotent_and_does_not_rebuild_the_partition(engine, session):
    first = pgb.ensure_bm25_index(KB_A, engine=engine)
    created = _indexdef(session)
    assert created is not None
    assert "USING bm25" in created
    assert "'stemmer=german'" in created

    second = pgb.ensure_bm25_index(KB_A, engine=engine)

    assert second["status"] == "ready"
    assert second.get("partition_created") is None
    assert second["partition"] == first["partition"]
    assert _indexdef(session) == created
    assert _rows_in(session, "chunks", KB_A) == 3


def test_the_index_lands_on_the_partition_not_the_parent(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)

    row = session.execute(
        text("SELECT tablename FROM pg_indexes WHERE schemaname = :s AND indexname = :n"),
        {"s": SCHEMA, "n": pgb.bm25_index_name(KB_A, "chunks")},
    ).first()

    assert row[0] == pgb.partition_name(KB_A, "chunks")


def test_ensure_skips_a_table_the_migration_has_not_partitioned(engine, session):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        migration_undone = f"{SCHEMA}.chunks"
        conn.execute(
            text(f"ALTER TABLE {migration_undone} DETACH PARTITION {SCHEMA}.chunks_default")
        )
        conn.execute(text(f"DROP TABLE {migration_undone}"))
        conn.execute(text(f"ALTER TABLE {SCHEMA}.chunks_default RENAME TO chunks"))

    result = pgb.ensure_bm25_index(KB_A, engine=engine)

    assert result["status"] == "skipped"
    assert result["reason"] == "table_not_partitioned"
    assert _indexdef(session) is None


def test_ensure_rebuilds_when_ts_language_changes_and_keeps_the_partition(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    assert "'stemmer=german'" in _indexdef(session)

    _set_language(session, KB_A, "english")

    assert pgb.ensure_bm25_index(KB_A, engine=engine)["status"] == "ready"
    rebuilt = _indexdef(session)
    assert "'stemmer=english'" in rebuilt
    assert "'stemmer=german'" not in rebuilt
    assert _rows_in(session, pgb.partition_name(KB_A, "chunks")) == 3


def test_drop_removes_the_index_and_leaves_the_partition(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    assert _indexdef(session) is not None

    result = pgb.drop_bm25_index(KB_A, engine=engine)

    assert result["status"] == "dropped"
    assert result["partitions"] == []
    assert _indexdef(session) is None
    assert pgb.bm25_index_state(session, KB_A, "chunks") == "absent"
    assert _rows_in(session, pgb.partition_name(KB_A, "chunks")) == 3


def test_drop_with_partitions_returns_the_rows_to_default(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    partition = pgb.partition_name(KB_A, "chunks")

    result = pgb.drop_bm25_index(KB_A, engine=engine, drop_partitions=True)

    assert result["partitions"] == [partition]
    assert (
        session.execute(text("SELECT to_regclass(:r)"), {"r": f"{SCHEMA}.{partition}"}).scalar()
        is None
    )
    assert _rows_in(session, "chunks", KB_A) == 3
    assert _rows_in(session, "chunks_default", KB_A) == 3


# ---------------------------------------------------------------------------
# Two knowledge bases at once -- the case the unpartitioned design could not do
# ---------------------------------------------------------------------------


def test_two_knowledge_bases_each_get_their_own_index_and_score_independently(engine, session):
    assert pgb.ensure_bm25_index(KB_A, engine=engine)["status"] == "ready"
    assert pgb.ensure_bm25_index(KB_B, engine=engine)["status"] == "ready"

    assert _indexdef(session, KB_A) is not None
    assert _indexdef(session, KB_B) is not None
    assert pgb.bm25_index_ready(session, KB_A, "chunks") is True
    assert pgb.bm25_index_ready(session, KB_B, "chunks") is True

    german = _search(session, "Wanderung", kb_id=KB_A, top_k=10)
    french = _search(session, "randonnée", kb_id=KB_B, top_k=10)

    assert len(german) == 2
    assert {item.knowledge_base_id for item in german} == {KB_A}
    assert len(french) == 2
    assert {item.knowledge_base_id for item in french} == {KB_B}
    # Neither sees the other's rows.
    assert _search(session, "randonnée", kb_id=KB_A, top_k=10) == []
    assert _search(session, "Wanderung", kb_id=KB_B, top_k=10) == []


def test_a_third_index_does_not_disturb_the_first_two(engine, session):
    """The old design's failure: building KB B's index broke KB A's scores."""
    pgb.ensure_bm25_index(KB_A, engine=engine)
    before = [item.score for item in _search(session, "Wanderung", kb_id=KB_A, top_k=10)]

    pgb.ensure_bm25_index(KB_B, engine=engine)
    pgb.ensure_bm25_index(KB_C, engine=engine)

    after = [item.score for item in _search(session, "Wanderung", kb_id=KB_A, top_k=10)]
    assert after == before
    assert len(_search(session, "Wanderung", kb_id=KB_C, top_k=10)) == 1


def test_each_partition_stems_in_its_own_language(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    pgb.ensure_bm25_index(KB_B, engine=engine)

    assert "'stemmer=german'" in _indexdef(session, KB_A)
    assert "'stemmer=french'" in _indexdef(session, KB_B)

    # German folds singular and plural, from either direction.
    assert len(_search(session, "Wanderung", kb_id=KB_A, top_k=10)) == 2
    assert len(_search(session, "Wanderungen", kb_id=KB_A, top_k=10)) == 2
    assert len(_search(session, "Bergführer", kb_id=KB_A, top_k=10)) == 1
    # French does the same for its own inflections.
    assert len(_search(session, "randonnée", kb_id=KB_B, top_k=10)) == 2
    assert len(_search(session, "randonnées", kb_id=KB_B, top_k=10)) == 2
    assert len(_search(session, "montagnes", kb_id=KB_B, top_k=10)) == 1


def test_a_kb_without_a_partition_falls_back_instead_of_erroring(engine, session):
    """The property the whole routing rests on.

    KB B has no partition and no index, so its keyword leg must reach the
    fallback. A query against the partitioned parent would have been refused
    by pg_search outright.
    """
    pgb.ensure_bm25_index(KB_A, engine=engine)

    assert pgb.bm25_index_ready(session, KB_B, "chunks") is False
    store = _store(session, KB_B)
    items = asyncio.run(store.bm25s_search("randonnée", top_k=5))

    assert isinstance(items, list)
    assert {item.knowledge_base_id for item in items} <= {KB_B}
    # And the session is still usable: nothing was aborted.
    assert session.execute(text("SELECT 1")).scalar() == 1


def test_a_scored_query_against_the_parent_is_refused_by_pg_search(engine, session):
    """Why the search path names the partition. Pinned so a later pg_search
    version that lifts this shows up as a failure here, not as a silent
    opportunity nobody notices."""
    pgb.ensure_bm25_index(KB_A, engine=engine)

    with pytest.raises(Exception, match="does not contain a .USING bm25. index"):
        session.execute(
            text(
                f"SELECT id, pdb.score(id) FROM {SCHEMA}.chunks "
                f"WHERE knowledge_base_id = '{KB_A}' AND text ||| 'Wanderung' "
                "ORDER BY pdb.score(id) DESC LIMIT 5"
            )
        ).fetchall()
    session.rollback()


# ---------------------------------------------------------------------------
# The other two partitioned tables
# ---------------------------------------------------------------------------


def test_graph_index_nodes_indexes_title_and_text_together(engine, session):
    _set_strategy(session, KB_A, "graph_index")
    session.execute(
        text(f"""
            INSERT INTO {SCHEMA}.graph_index_nodes (knowledge_base_id, source_id, title, text)
            VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), 'Titel Wanderung',
                    'Bergführers Pflichten'),
                   (CAST(:kb AS uuid), CAST(:src AS uuid), 'Anderer Titel', 'Brücke')
        """),
        {"kb": KB_A, "src": SOURCE_1},
    )
    session.execute(
        text(f"""
            INSERT INTO {SCHEMA}.graph_index_nodes (knowledge_base_id, source_id, title, text)
            VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), 'Titel', 'in einer anderen Basis')
        """),
        {"kb": KB_C, "src": SOURCE_1},
    )
    session.commit()

    result = pgb.ensure_bm25_index(KB_A, engine=engine)

    assert result["status"] == "ready"
    assert result["partition"] == pgb.partition_name(KB_A, "graph_index_nodes")
    # The alias is what makes an indexed expression legal at all.
    assert "'alias=bm25_text'" in _indexdef(session, KB_A, "graph_index_nodes")
    # A term from the title matches, and so does one from the text.
    assert len(_search(session, "Titel", cls=_NodeStore, top_k=5)) == 2
    assert len(_search(session, "Bergführer", cls=_NodeStore, top_k=5)) == 1
    # And KB C's node never appears.
    assert _search(session, "Basis", cls=_NodeStore, top_k=5) == []


def test_full_documents_indexes_the_summary(engine, session):
    _set_strategy(session, KB_A, "full_document")
    session.execute(
        text(f"""
            INSERT INTO {SCHEMA}.full_documents (knowledge_base_id, source_id, summary)
            VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), 'Zusammenfassung der Wanderungen'),
                   (CAST(:kb AS uuid), CAST(:src AS uuid), 'Bericht über die Brücke')
        """),
        {"kb": KB_A, "src": SOURCE_1},
    )
    session.commit()

    result = pgb.ensure_bm25_index(KB_A, engine=engine)

    assert result["status"] == "ready"
    assert result["item_table"] == "full_documents"
    assert result["partition"] == pgb.partition_name(KB_A, "full_documents")
    assert "(summary)::pdb.simple('stemmer=german')" in _indexdef(session, KB_A, "full_documents")
    items = _search(session, "Wanderung", cls=_FullDocumentStore, top_k=5)
    assert len(items) == 1
    assert "Zusammenfassung" in items[0].text


# ---------------------------------------------------------------------------
# Search behaviour on a partition
# ---------------------------------------------------------------------------


def test_search_returns_scored_rows_ordered_by_score(engine, session):
    """Best match first, with scores that actually differ (mutation M36).

    The seed documents score identically for "Wanderung", so an ascending sort
    passed the old assertion. Here one row repeats the term and must lead.
    """
    session.execute(
        text(f"""
            INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text)
            VALUES (CAST(:kb AS uuid), CAST(:src AS uuid),
                    'Wanderung Wanderung Wanderung am Grat, Wanderung bei Nebel')
        """),
        {"kb": KB_A, "src": SOURCE_1},
    )
    session.commit()
    pgb.ensure_bm25_index(KB_A, engine=engine)

    items = _search(session, "Wanderung", top_k=5)

    assert len(items) == 3
    assert all(item.score > 0 for item in items)
    assert items[0].text.startswith("Wanderung Wanderung Wanderung")
    assert items[0].score > items[-1].score
    assert [item.score for item in items] == sorted((item.score for item in items), reverse=True)
    assert {item.knowledge_base_id for item in items} == {KB_A}


def test_top_k_bounds_the_result_set(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)

    assert len(_search(session, "Wanderung", top_k=1)) == 1


def test_source_ids_filter_restricts_results(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)

    both = _search(session, "Wanderung", top_k=10)
    assert {item.source_id for item in both} == {SOURCE_1}

    assert _search(session, "Wanderung", top_k=10, source_ids=[SOURCE_2]) == []

    bruecke = _search(session, "Brücke", top_k=10, source_ids=[SOURCE_2])
    assert len(bruecke) == 1
    assert bruecke[0].source_id == SOURCE_2


def test_metadata_filter_and_item_ids_restrict_results(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)

    assert len(_search(session, "Wanderung", top_k=10, filter_metadata={"lang": "de"})) == 2
    assert _search(session, "Wanderung", top_k=10, filter_metadata={"lang": "fr"}) == []

    one = _search(session, "Wanderung", top_k=1)[0]
    restricted = _search(session, "Wanderung", top_k=10, item_ids={one.item_id})
    assert [item.item_id for item in restricted] == [one.item_id]


def test_inserts_and_deletes_through_the_parent_are_visible_without_a_rebuild(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    assert len(_search(session, "Brücke", top_k=10)) == 1

    new_id = str(uuid.uuid4())
    session.execute(
        text(f"""
            INSERT INTO {SCHEMA}.chunks (id, knowledge_base_id, source_id, text)
            VALUES (CAST(:id AS uuid), CAST(:kb AS uuid), CAST(:src AS uuid),
                    'Die Brücke wird erneut gebaut')
        """),
        {"id": new_id, "kb": KB_A, "src": SOURCE_1},
    )
    session.commit()
    # Written through the parent, routed into the partition, indexed there.
    assert _rows_in(session, pgb.partition_name(KB_A, "chunks")) == 4
    assert len(_search(session, "Brücke", top_k=10)) == 2

    session.execute(
        text(f"DELETE FROM {SCHEMA}.chunks WHERE id = CAST(:id AS uuid)"), {"id": new_id}
    )
    session.commit()
    assert len(_search(session, "Brücke", top_k=10)) == 1


def test_a_cascading_delete_of_the_kb_empties_its_partition(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    partition = pgb.partition_name(KB_A, "chunks")
    assert _rows_in(session, partition) == 3

    session.execute(
        text(f"DELETE FROM {SCHEMA}.knowledge_bases WHERE id = CAST(:id AS uuid)"), {"id": KB_A}
    )
    session.commit()

    assert _rows_in(session, partition) == 0
    assert _search(session, "Wanderung", top_k=10) == []


def test_without_a_stemmer_the_inflection_no_longer_matches(engine, session):
    """The stemmer is what does the work, not the tokenizer.

    Rebuilt for a language pg_search cannot stem, the same index keeps working
    but stops matching German inflections -- so a KB configured that way gets a
    plain keyword index rather than no index at all.
    """
    _set_language(session, KB_A, "hindi")

    assert pgb.ensure_bm25_index(KB_A, engine=engine)["status"] == "ready"
    assert "stemmer" not in _indexdef(session)
    assert len(_search(session, "Wanderungen", top_k=10)) == 1
    assert _search(session, "Bergführer", top_k=10) == []


# ---------------------------------------------------------------------------
# Query-string safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adversarial",
    [
        "it's a quote",
        'a "quoted" phrase',
        "field:value",
        ":",
        "back\\slash",
        "\\",
        "plus+minus-tilde~caret^",
        "(parens) [brackets] {braces}",
        "Wanderung (",
        "AND OR NOT",
        "/slashes/",
        "' OR 1=1; DROP TABLE chunks; --",
        "Wander\x00ung",
        "Wanderung^2",
        "[a TO z]",
        '{"k": "v"}',
        "a ||| b",
        "a" * 50_000,
        "😀 ünïcode",
    ],
)
def test_an_adversarial_query_answers_instead_of_raising(engine, session, adversarial):
    pgb.ensure_bm25_index(KB_A, engine=engine)

    items = _search(session, adversarial, top_k=5)

    assert isinstance(items, list)
    # The session is still usable: nothing was aborted mid-transaction.
    assert session.execute(text("SELECT 1")).scalar() == 1


def test_a_blank_query_never_reaches_the_database(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)

    assert _search(session, "   \n ", top_k=5) == []
    assert _search(session, "", top_k=5) == []


def test_the_nul_byte_is_stripped_before_it_reaches_psycopg(engine, session):
    """psycopg refuses a text parameter containing NUL, so it must not get one."""
    pgb.ensure_bm25_index(KB_A, engine=engine)

    items = _search(session, "Wanderung\x00", top_k=5)

    assert len(items) == 2


# ---------------------------------------------------------------------------
# Routing through bm25s_search
# ---------------------------------------------------------------------------


def test_bm25s_search_uses_the_pg_index_when_it_is_ready(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    store = _store(session)

    items = asyncio.run(store.bm25s_search("Wanderung", top_k=5))

    assert len(items) == 2
    assert all(item.score > 0 for item in items)


def test_a_stale_ready_cache_degrades_to_the_fallback_inside_one_transaction(engine, session):
    """The savepoint around the scored query is load-bearing (mutation M31).

    The readiness cache can say "ready" for up to its TTL after the index is
    gone (dropped by a worker in another process, say). The scored query then
    fails; without the savepoint that failure aborts the caller's transaction
    and the keyword fallback dies with "current transaction is aborted".
    """
    pgb.ensure_bm25_index(KB_A, engine=engine)
    assert pgb.bm25_index_ready(session, KB_A, "chunks") is True  # cached: ready
    session.rollback()
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP INDEX {SCHEMA}.{pgb.bm25_index_name(KB_A, 'chunks')}"))

    # One open transaction across the whole search, as a request has.
    session.execute(text("SELECT 1"))
    items = asyncio.run(_store(session).bm25s_search("Wanderung", top_k=5))

    assert {item.knowledge_base_id for item in items} == {KB_A}
    assert len(items) >= 1
    assert session.execute(text("SELECT 1")).scalar() == 1
    session.rollback()
