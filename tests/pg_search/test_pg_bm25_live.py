"""Database-backed checks for the pg_search BM25 path.

Unit specs show the emitted SQL is shaped right. Only a Postgres with
``pg_search`` installed shows the partitions get built, the index lands on the
partition, two knowledge bases are scored independently with their own
stemmers, the filters push down, DML is visible without a reindex, and an
adversarial query answers instead of raising.

Runs against ``PG_SEARCH_TEST_DATABASE_URL`` if set, otherwise ``DATABASE_URL``,
and skips with a reason when that server cannot offer the extension. Every test
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
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
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
    (SOURCE_1, "Die Beschwerde des Werkeigentümers wurde abgewiesen"),
    (SOURCE_1, "Mehrere Beschwerden gingen beim Gericht ein"),
    (SOURCE_2, "Verjährung der Forderung nach drei Jahren"),
]
KB_B_DOCS = [
    (SOURCE_1, "La réclamation du propriétaire a été rejetée"),
    (SOURCE_1, "Plusieurs réclamations sont arrivées au tribunal"),
]
KB_C_DOCS = [(SOURCE_1, "Eine Beschwerde in einer dritten Wissensbasis")]


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
        pytest.skip("no PG_SEARCH_TEST_DATABASE_URL or DATABASE_URL to test pg_search against")
    eng = create_engine(dsn)
    where = eng.url.render_as_string(hide_password=True)
    try:
        with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_search"))
    except Exception as exc:
        eng.dispose()
        pytest.skip(f"pg_search is not available on {where}: {str(exc).splitlines()[0]}")
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


def test_partition_creation_is_batched_and_loses_nothing(engine, session, monkeypatch):
    """Many small statements, one transaction, same rows at the end."""
    for n in range(500):
        session.execute(
            text(f"""
                INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text)
                VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), :body)
            """),
            {"kb": KB_A, "src": SOURCE_1, "body": f"Beschwerde Nummer {n}"},
        )
    session.commit()
    monkeypatch.setattr(pgb, "EVACUATION_BATCH_ROWS", 100)

    moved = pgb.create_partition(engine, KB_A, "chunks")

    assert moved == 503
    assert _rows_in(session, "chunks", KB_A) == 503
    assert _rows_in(session, "chunks_default", KB_A) == 0
    assert _rows_in(session, pgb.partition_name(KB_A, "chunks")) == 503


def test_a_reader_never_sees_the_rows_missing_while_they_move(engine, session, monkeypatch):
    """The move is one transaction, so every count a reader takes is the full one."""
    for n in range(2000):
        session.execute(
            text(f"""
                INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text)
                VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), :body)
            """),
            {"kb": KB_A, "src": SOURCE_1, "body": f"Beschwerde Nummer {n}"},
        )
    session.commit()
    monkeypatch.setattr(pgb, "EVACUATION_BATCH_ROWS", 50)

    observed: list[int] = []
    stop = threading.Event()

    def reader():
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            while not stop.is_set():
                observed.append(
                    conn.execute(
                        text(
                            f"SELECT count(*) FROM {SCHEMA}.chunks "
                            f"WHERE knowledge_base_id = '{KB_A}'"
                        )
                    ).scalar()
                )

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        pgb.create_partition(engine, KB_A, "chunks")
    finally:
        stop.set()
        thread.join(timeout=10)

    assert observed, "the reader never got a chance to look"
    assert set(observed) == {2003}


def test_a_concurrent_insert_during_the_move_either_lands_or_fails_loudly(engine, session):
    """No row is ever silently dropped -- but an in-flight write can fail.

    The SHARE lock holds off writers that *start* during the move. A writer
    already inside its INSERT when the ATTACH commits is a case Postgres itself
    cannot thread: its tuple routing already resolved to the DEFAULT partition,
    and after the attach that partition's constraint excludes this knowledge
    base, so the statement fails with a partition-constraint violation. That is
    documented Postgres behaviour for ATTACH PARTITION on a table with a DEFAULT
    partition, and it is a *retryable error*, not a lost row.

    Pinned here so the failure mode stays the known one. In practice the race is
    avoidable: a knowledge base created after this feature has its partition
    dispatched at creation, before it has any rows to move.
    """
    landed: list[int] = []
    failed: list[str] = []
    ready = threading.Event()

    def writer():
        ready.wait(timeout=10)
        for n in range(20):
            try:
                with engine.begin() as conn:
                    conn.execute(
                        text(f"""
                            INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text)
                            VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), :body)
                        """),
                        {"kb": KB_A, "src": SOURCE_1, "body": f"Beschwerde spaet {n}"},
                    )
                landed.append(n)
            except Exception as exc:
                failed.append(str(exc).splitlines()[0])

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    ready.set()
    pgb.create_partition(engine, KB_A, "chunks")
    thread.join(timeout=30)

    # Every attempt is accounted for: committed, or refused with an error.
    assert len(landed) + len(failed) == 20
    # At most the one statement in flight across the ATTACH can fail, and only
    # with the documented, retryable partition-constraint violation.
    assert len(failed) <= 1, failed
    assert all("violates partition constraint" in message for message in failed), failed
    # Everything that did commit is visible through the parent, and nowhere else.
    expected = len(KB_A_DOCS) + len(landed)
    assert _rows_in(session, "chunks", KB_A) == expected
    assert (
        _rows_in(session, "chunks_default", KB_A)
        + _rows_in(session, pgb.partition_name(KB_A, "chunks"))
        == expected
    )


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

    german = _search(session, "Beschwerde", kb_id=KB_A, top_k=10)
    french = _search(session, "réclamation", kb_id=KB_B, top_k=10)

    assert len(german) == 2
    assert {item.knowledge_base_id for item in german} == {KB_A}
    assert len(french) == 2
    assert {item.knowledge_base_id for item in french} == {KB_B}
    # Neither sees the other's rows.
    assert _search(session, "réclamation", kb_id=KB_A, top_k=10) == []
    assert _search(session, "Beschwerde", kb_id=KB_B, top_k=10) == []


def test_a_third_index_does_not_disturb_the_first_two(engine, session):
    """The old design's failure: building KB B's index broke KB A's scores."""
    pgb.ensure_bm25_index(KB_A, engine=engine)
    before = [item.score for item in _search(session, "Beschwerde", kb_id=KB_A, top_k=10)]

    pgb.ensure_bm25_index(KB_B, engine=engine)
    pgb.ensure_bm25_index(KB_C, engine=engine)

    after = [item.score for item in _search(session, "Beschwerde", kb_id=KB_A, top_k=10)]
    assert after == before
    assert len(_search(session, "Beschwerde", kb_id=KB_C, top_k=10)) == 1


def test_each_partition_stems_in_its_own_language(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    pgb.ensure_bm25_index(KB_B, engine=engine)

    assert "'stemmer=german'" in _indexdef(session, KB_A)
    assert "'stemmer=french'" in _indexdef(session, KB_B)

    # German folds singular and plural, from either direction.
    assert len(_search(session, "Beschwerde", kb_id=KB_A, top_k=10)) == 2
    assert len(_search(session, "Beschwerden", kb_id=KB_A, top_k=10)) == 2
    assert len(_search(session, "Werkeigentümer", kb_id=KB_A, top_k=10)) == 1
    # French does the same for its own inflections.
    assert len(_search(session, "réclamation", kb_id=KB_B, top_k=10)) == 2
    assert len(_search(session, "réclamations", kb_id=KB_B, top_k=10)) == 2
    assert len(_search(session, "propriétaires", kb_id=KB_B, top_k=10)) == 1


def test_a_kb_without_a_partition_falls_back_instead_of_erroring(engine, session):
    """The property the whole routing rests on.

    KB B has no partition and no index, so its keyword leg must reach the
    fallback. A query against the partitioned parent would have been refused
    by pg_search outright.
    """
    pgb.ensure_bm25_index(KB_A, engine=engine)

    assert pgb.bm25_index_ready(session, KB_B, "chunks") is False
    store = _store(session, KB_B)
    items = asyncio.run(store.bm25s_search("réclamation", top_k=5))

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
                f"WHERE knowledge_base_id = '{KB_A}' AND text ||| 'Beschwerde' "
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
            VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), 'Titel Beschwerde',
                    'Werkeigentümers Pflichten'),
                   (CAST(:kb AS uuid), CAST(:src AS uuid), 'Anderer Titel', 'Verjährung')
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
    assert len(_search(session, "Werkeigentümer", cls=_NodeStore, top_k=5)) == 1
    # And KB C's node never appears.
    assert _search(session, "Basis", cls=_NodeStore, top_k=5) == []


def test_full_documents_indexes_the_summary(engine, session):
    _set_strategy(session, KB_A, "full_document")
    session.execute(
        text(f"""
            INSERT INTO {SCHEMA}.full_documents (knowledge_base_id, source_id, summary)
            VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), 'Zusammenfassung der Beschwerden'),
                   (CAST(:kb AS uuid), CAST(:src AS uuid), 'Bericht über die Verjährung')
        """),
        {"kb": KB_A, "src": SOURCE_1},
    )
    session.commit()

    result = pgb.ensure_bm25_index(KB_A, engine=engine)

    assert result["status"] == "ready"
    assert result["item_table"] == "full_documents"
    assert result["partition"] == pgb.partition_name(KB_A, "full_documents")
    assert "(summary)::pdb.simple('stemmer=german')" in _indexdef(session, KB_A, "full_documents")
    items = _search(session, "Beschwerde", cls=_FullDocumentStore, top_k=5)
    assert len(items) == 1
    assert "Zusammenfassung" in items[0].text


# ---------------------------------------------------------------------------
# Search behaviour on a partition
# ---------------------------------------------------------------------------


def test_search_returns_scored_rows_ordered_by_score(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)

    items = _search(session, "Beschwerde", top_k=5)

    assert len(items) == 2
    assert all(item.score > 0 for item in items)
    assert [item.score for item in items] == sorted((item.score for item in items), reverse=True)
    assert {item.knowledge_base_id for item in items} == {KB_A}
    assert all("Beschwerde" in item.text for item in items)


def test_top_k_bounds_the_result_set(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)

    assert len(_search(session, "Beschwerde", top_k=1)) == 1


def test_source_ids_filter_restricts_results(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)

    both = _search(session, "Beschwerde", top_k=10)
    assert {item.source_id for item in both} == {SOURCE_1}

    assert _search(session, "Beschwerde", top_k=10, source_ids=[SOURCE_2]) == []

    verjaehrung = _search(session, "Verjährung", top_k=10, source_ids=[SOURCE_2])
    assert len(verjaehrung) == 1
    assert verjaehrung[0].source_id == SOURCE_2


def test_metadata_filter_and_item_ids_restrict_results(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)

    assert len(_search(session, "Beschwerde", top_k=10, filter_metadata={"lang": "de"})) == 2
    assert _search(session, "Beschwerde", top_k=10, filter_metadata={"lang": "fr"}) == []

    one = _search(session, "Beschwerde", top_k=1)[0]
    restricted = _search(session, "Beschwerde", top_k=10, item_ids={one.item_id})
    assert [item.item_id for item in restricted] == [one.item_id]


def test_inserts_and_deletes_through_the_parent_are_visible_without_a_rebuild(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    assert len(_search(session, "Verjährung", top_k=10)) == 1

    new_id = str(uuid.uuid4())
    session.execute(
        text(f"""
            INSERT INTO {SCHEMA}.chunks (id, knowledge_base_id, source_id, text)
            VALUES (CAST(:id AS uuid), CAST(:kb AS uuid), CAST(:src AS uuid),
                    'Die Verjährung beginnt erneut')
        """),
        {"id": new_id, "kb": KB_A, "src": SOURCE_1},
    )
    session.commit()
    # Written through the parent, routed into the partition, indexed there.
    assert _rows_in(session, pgb.partition_name(KB_A, "chunks")) == 4
    assert len(_search(session, "Verjährung", top_k=10)) == 2

    session.execute(
        text(f"DELETE FROM {SCHEMA}.chunks WHERE id = CAST(:id AS uuid)"), {"id": new_id}
    )
    session.commit()
    assert len(_search(session, "Verjährung", top_k=10)) == 1


def test_a_cascading_delete_of_the_kb_empties_its_partition(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    partition = pgb.partition_name(KB_A, "chunks")
    assert _rows_in(session, partition) == 3

    session.execute(
        text(f"DELETE FROM {SCHEMA}.knowledge_bases WHERE id = CAST(:id AS uuid)"), {"id": KB_A}
    )
    session.commit()

    assert _rows_in(session, partition) == 0
    assert _search(session, "Beschwerde", top_k=10) == []


def test_without_a_stemmer_the_inflection_no_longer_matches(engine, session):
    """The stemmer is what does the work, not the tokenizer.

    Rebuilt for a language pg_search cannot stem, the same index keeps working
    but stops matching German inflections -- so a KB configured that way gets a
    plain keyword index rather than no index at all.
    """
    _set_language(session, KB_A, "hindi")

    assert pgb.ensure_bm25_index(KB_A, engine=engine)["status"] == "ready"
    assert "stemmer" not in _indexdef(session)
    assert len(_search(session, "Beschwerden", top_k=10)) == 1
    assert _search(session, "Werkeigentümer", top_k=10) == []


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
        "Beschwerde (",
        "AND OR NOT",
        "/slashes/",
        "' OR 1=1; DROP TABLE chunks; --",
        "Besch\x00werde",
        "Beschwerde^2",
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

    items = _search(session, "Beschwerde\x00", top_k=5)

    assert len(items) == 2


# ---------------------------------------------------------------------------
# Routing through bm25s_search
# ---------------------------------------------------------------------------


def test_bm25s_search_uses_the_pg_index_when_it_is_ready(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    store = _store(session)

    items = asyncio.run(store.bm25s_search("Beschwerde", top_k=5))

    assert len(items) == 2
    assert all(item.score > 0 for item in items)
