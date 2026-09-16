"""Database-backed checks for the copy-free partitioning migration.

The migration turns each bm25-backed item table into a partitioned parent whose
DEFAULT partition *is* the original table, renamed. Nothing is copied, so the
only way to know it preserved the rows, the constraints, the grants and the RLS
posture is to run it against a real Postgres and look.

Runs against ``PG_SEARCH_TEST_DATABASE_URL`` if set, otherwise ``DATABASE_URL``.
Everything happens in a scratch schema, so the ``ai`` schema is never touched.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

SCHEMA = "bm25_migration_test"

KB_A = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
KB_B = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
SOURCE = "11111111-1111-4111-8111-111111111111"


@pytest.fixture(scope="module")
def migration():
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "0031_partition_bm25_item_tables.py"
    )
    spec = importlib.util.spec_from_file_location("mig_0031", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def engine():
    dsn = os.environ.get("PG_SEARCH_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("no PG_SEARCH_TEST_DATABASE_URL or DATABASE_URL to test against")
    eng = create_engine(dsn)
    yield eng
    eng.dispose()


def _create_unpartitioned(conn, rows_per_kb: int = 3) -> None:
    """The schema as it is before the migration: three plain item tables."""
    conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))
    conn.execute(text(f"CREATE TABLE {SCHEMA}.knowledge_bases (id uuid PRIMARY KEY)"))
    conn.execute(text(f"CREATE TABLE {SCHEMA}.sources (id uuid PRIMARY KEY)"))
    conn.execute(text(f"CREATE TABLE {SCHEMA}.graph_index_toc (id uuid PRIMARY KEY)"))
    for kb_id in (KB_A, KB_B):
        conn.execute(
            text(f"INSERT INTO {SCHEMA}.knowledge_bases VALUES (CAST(:id AS uuid))"),
            {"id": kb_id},
        )
    conn.execute(text(f"INSERT INTO {SCHEMA}.sources VALUES (CAST(:id AS uuid))"), {"id": SOURCE})
    conn.execute(
        text(f"INSERT INTO {SCHEMA}.graph_index_toc VALUES (CAST(:id AS uuid))"), {"id": SOURCE}
    )

    conn.execute(
        text(f"""
            CREATE TABLE {SCHEMA}.chunks (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                knowledge_base_id uuid NOT NULL
                    REFERENCES {SCHEMA}.knowledge_bases(id) ON DELETE CASCADE,
                source_id uuid NOT NULL REFERENCES {SCHEMA}.sources(id) ON DELETE CASCADE,
                text text NOT NULL,
                meta jsonb DEFAULT '{{}}'::jsonb,
                created_at timestamptz DEFAULT now()
            )
        """)
    )
    conn.execute(text(f"CREATE INDEX idx_chunks_kb ON {SCHEMA}.chunks(knowledge_base_id)"))
    conn.execute(text(f"CREATE INDEX idx_chunks_meta ON {SCHEMA}.chunks USING gin(meta)"))
    conn.execute(text(f"ALTER TABLE {SCHEMA}.chunks ENABLE ROW LEVEL SECURITY"))

    conn.execute(
        text(f"""
            CREATE TABLE {SCHEMA}.full_documents (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                knowledge_base_id uuid NOT NULL
                    REFERENCES {SCHEMA}.knowledge_bases(id) ON DELETE CASCADE,
                source_id uuid NOT NULL REFERENCES {SCHEMA}.sources(id) ON DELETE CASCADE,
                summary text NOT NULL,
                meta jsonb DEFAULT '{{}}'::jsonb
            )
        """)
    )
    conn.execute(
        text(f"""
            CREATE TABLE {SCHEMA}.graph_index_nodes (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                toc_id uuid NOT NULL
                    REFERENCES {SCHEMA}.graph_index_toc(id) ON DELETE CASCADE,
                knowledge_base_id uuid NOT NULL
                    REFERENCES {SCHEMA}.knowledge_bases(id) ON DELETE CASCADE,
                source_id uuid NOT NULL REFERENCES {SCHEMA}.sources(id) ON DELETE CASCADE,
                node_id text NOT NULL,
                title text,
                text text NOT NULL,
                meta jsonb DEFAULT '{{}}'::jsonb,
                CONSTRAINT graph_index_nodes_toc_id_node_id_key UNIQUE (toc_id, node_id)
            )
        """)
    )

    for kb_id in (KB_A, KB_B):
        for n in range(rows_per_kb):
            conn.execute(
                text(f"""
                    INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text)
                    VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), :body)
                """),
                {"kb": kb_id, "src": SOURCE, "body": f"Wanderung {kb_id} {n}"},
            )
            conn.execute(
                text(f"""
                    INSERT INTO {SCHEMA}.full_documents (knowledge_base_id, source_id, summary)
                    VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), :body)
                """),
                {"kb": kb_id, "src": SOURCE, "body": f"Zusammenfassung {n}"},
            )
            conn.execute(
                text(f"""
                    INSERT INTO {SCHEMA}.graph_index_nodes
                        (toc_id, knowledge_base_id, source_id, node_id, title, text)
                    VALUES (CAST(:toc AS uuid), CAST(:kb AS uuid), CAST(:src AS uuid),
                            :node, :title, :body)
                """),
                {
                    "toc": SOURCE,
                    "kb": kb_id,
                    "src": SOURCE,
                    "node": f"{kb_id}-{n}",
                    "title": f"Titel {n}",
                    "body": f"Abschnitt {n}",
                },
            )


@pytest.fixture
def prefilled(engine):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        _create_unpartitioned(conn)
    yield
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))


def _relkind(conn, relname: str) -> str | None:
    row = conn.execute(
        text(
            "SELECT relkind FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = :s AND c.relname = :r"
        ),
        {"s": SCHEMA, "r": relname},
    ).first()
    return row[0] if row else None


def _count(conn, relation: str, kb_id: str | None = None) -> int:
    sql = f"SELECT count(*) FROM {SCHEMA}.{relation}"
    params: dict = {}
    if kb_id:
        sql += " WHERE knowledge_base_id = CAST(:kb AS uuid)"
        params["kb"] = kb_id
    return conn.execute(text(sql), params).scalar()


# ---------------------------------------------------------------------------
# The conversion itself
# ---------------------------------------------------------------------------


def test_conversion_partitions_every_table_and_keeps_every_row(migration, engine, prefilled):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        before = {t: _count(conn, t) for t in ("chunks", "full_documents", "graph_index_nodes")}
        assert before == {"chunks": 6, "full_documents": 6, "graph_index_nodes": 6}

        migration.partition_item_tables(conn, schema=SCHEMA)

        for table, rows in before.items():
            assert _relkind(conn, table) == "p", table
            assert _relkind(conn, f"{table}_default") == "r", table
            assert _count(conn, table) == rows, table
            # Reads through the parent still answer per knowledge base.
            assert _count(conn, table, KB_A) == rows // 2, table


def test_the_renamed_table_is_attached_as_the_default_partition(migration, engine, prefilled):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        migration.partition_item_tables(conn, schema=SCHEMA)

        row = conn.execute(
            text(f"""
                SELECT c.relname, pg_get_expr(c.relpartbound, c.oid)
                FROM pg_class c
                JOIN pg_inherits i ON i.inhrelid = c.oid
                WHERE i.inhparent = '{SCHEMA}.chunks'::regclass
            """)
        ).all()

        assert [(r[0], r[1]) for r in row] == [("chunks_default", "DEFAULT")]


def test_the_parent_has_no_primary_key_and_the_default_partition_keeps_its_own(
    migration, engine, prefilled
):
    """The whole point of the shape: ``id`` alone stays a legal key.

    A unique constraint on a partitioned parent must contain every partitioning
    column, so a parent-level ``PRIMARY KEY (id)`` is impossible. Declaring
    none on the parent leaves each partition's key local, which is all
    ``key_field = 'id'`` needs.
    """
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        migration.partition_item_tables(conn, schema=SCHEMA)

        parent = conn.execute(
            text(f"SELECT conname FROM pg_constraint WHERE conrelid = '{SCHEMA}.chunks'::regclass")
        ).all()
        assert parent == []

        child = {
            r[0]: r[1]
            for r in conn.execute(
                text(
                    "SELECT contype, pg_get_constraintdef(oid) FROM pg_constraint "
                    f"WHERE conrelid = '{SCHEMA}.chunks_default'::regclass"
                )
            ).all()
        }
        assert child["p"] == "PRIMARY KEY (id)"
        assert "f" in child, "the foreign keys came along with the rename"


def test_the_default_partition_keeps_its_indexes_and_a_duplicate_id_is_still_refused(
    migration, engine, prefilled
):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        migration.partition_item_tables(conn, schema=SCHEMA)

        indexes = {
            r[0]
            for r in conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE schemaname = :s AND tablename = :t"),
                {"s": SCHEMA, "t": "chunks_default"},
            ).all()
        }
        assert {"chunks_pkey", "idx_chunks_kb", "idx_chunks_meta"} <= indexes

        existing = conn.execute(text(f"SELECT id FROM {SCHEMA}.chunks LIMIT 1")).scalar()
        with pytest.raises(Exception, match="duplicate key value"):
            conn.execute(
                text(f"""
                    INSERT INTO {SCHEMA}.chunks (id, knowledge_base_id, source_id, text)
                    VALUES (CAST(:id AS uuid), CAST(:kb AS uuid), CAST(:src AS uuid), 'dup')
                """),
                {"id": str(existing), "kb": KB_A, "src": SOURCE},
            )


def test_conversion_mirrors_rls_and_grants_onto_the_parent(migration, engine, prefilled):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(
            text("DO $$ BEGIN CREATE ROLE bm25_mig_reader; EXCEPTION WHEN OTHERS THEN END $$")
        )
        conn.execute(text(f"GRANT SELECT, INSERT ON {SCHEMA}.chunks TO bm25_mig_reader"))

        migration.partition_item_tables(conn, schema=SCHEMA)

        row = conn.execute(
            text(
                "SELECT relrowsecurity, relowner::regrole::text, relacl::text "
                f"FROM pg_class WHERE oid = '{SCHEMA}.chunks'::regclass"
            )
        ).first()
        assert row[0] is True, "RLS must survive the conversion"
        assert "bm25_mig_reader" in row[2]

        privileges = {
            r[0]
            for r in conn.execute(
                text(
                    "SELECT privilege_type FROM information_schema.role_table_grants "
                    "WHERE table_schema = :s AND table_name = 'chunks' AND grantee = :g"
                ),
                {"s": SCHEMA, "g": "bm25_mig_reader"},
            ).all()
        }
        assert privileges == {"SELECT", "INSERT"}


def test_conversion_writes_and_reads_still_work_through_the_parent(migration, engine, prefilled):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        migration.partition_item_tables(conn, schema=SCHEMA)

        new_id = conn.execute(
            text(f"""
                INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text)
                VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), 'nach der Migration')
                RETURNING id
            """),
            {"kb": KB_A, "src": SOURCE},
        ).scalar()
        assert _count(conn, "chunks", KB_A) == 4

        conn.execute(
            text(f"DELETE FROM {SCHEMA}.chunks WHERE id = CAST(:id AS uuid)"),
            {"id": str(new_id)},
        )
        assert _count(conn, "chunks", KB_A) == 3

        # The foreign key still cascades from the knowledge base.
        conn.execute(
            text(f"DELETE FROM {SCHEMA}.knowledge_bases WHERE id = CAST(:id AS uuid)"),
            {"id": KB_B},
        )
        assert _count(conn, "chunks") == 3


def test_conversion_is_idempotent(migration, engine, prefilled):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        migration.partition_item_tables(conn, schema=SCHEMA)
        first = conn.execute(
            text(
                "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = :s ORDER BY c.relname"
            ),
            {"s": SCHEMA},
        ).all()

        migration.partition_item_tables(conn, schema=SCHEMA)

        assert _count(conn, "chunks") == 6
        assert _relkind(conn, "chunks") == "p"
        assert (
            conn.execute(
                text(
                    "SELECT c.relname FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = :s ORDER BY c.relname"
                ),
                {"s": SCHEMA},
            ).all()
            == first
        )


def test_conversion_skips_a_table_that_does_not_exist(migration, engine, prefilled):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP TABLE {SCHEMA}.graph_index_nodes"))

        migration.partition_item_tables(conn, schema=SCHEMA)

        assert _relkind(conn, "graph_index_nodes") is None
        assert _relkind(conn, "chunks") == "p"


def test_conversion_works_on_an_empty_table(migration, engine, prefilled):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DELETE FROM {SCHEMA}.chunks"))

        migration.partition_item_tables(conn, schema=SCHEMA)

        assert _relkind(conn, "chunks") == "p"
        assert _count(conn, "chunks") == 0


def test_conversion_drops_a_leftover_bm25_index_from_the_unpartitioned_design(
    migration, engine, prefilled
):
    """A per-KB partial index on the old table has no place in the new shape.

    Left in place it would occupy the DEFAULT partition's single bm25 slot and
    still answer only one knowledge base.
    """
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_search"))
        except Exception as exc:  # pragma: no cover - server without the extension
            pytest.skip(f"pg_search unavailable: {str(exc).splitlines()[0]}")
        conn.execute(
            text(
                f"CREATE INDEX bm25_chunks_legacy ON {SCHEMA}.chunks "
                "USING bm25 (id, text, source_id, meta) WITH (key_field = 'id') "
                f"WHERE knowledge_base_id = '{KB_A}'"
            )
        )

        migration.partition_item_tables(conn, schema=SCHEMA)

        left = conn.execute(
            text(
                "SELECT count(*) FROM pg_index i JOIN pg_class ic ON ic.oid = i.indexrelid "
                "JOIN pg_am am ON am.oid = ic.relam JOIN pg_class tc ON tc.oid = i.indrelid "
                "JOIN pg_namespace n ON n.oid = tc.relnamespace "
                "WHERE n.nspname = :s AND am.amname = 'bm25'"
            ),
            {"s": SCHEMA},
        ).scalar()
        assert left == 0


# ---------------------------------------------------------------------------
# Downgrade
# ---------------------------------------------------------------------------


def test_downgrade_restores_a_plain_table_with_every_row(migration, engine, prefilled):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        migration.partition_item_tables(conn, schema=SCHEMA)
        conn.execute(
            text(f"""
                CREATE TABLE {SCHEMA}.chunks_kb_extra (
                    LIKE {SCHEMA}.chunks_default INCLUDING DEFAULTS INCLUDING INDEXES
                )
            """)
        )
        conn.execute(
            text(f"""
                WITH moved AS (
                    DELETE FROM {SCHEMA}.chunks_default
                    WHERE knowledge_base_id = CAST(:kb AS uuid) RETURNING *
                ) INSERT INTO {SCHEMA}.chunks_kb_extra SELECT * FROM moved
            """),
            {"kb": KB_A},
        )
        conn.execute(
            text(
                f"ALTER TABLE {SCHEMA}.chunks ATTACH PARTITION {SCHEMA}.chunks_kb_extra "
                f"FOR VALUES IN ('{KB_A}')"
            )
        )
        assert _count(conn, "chunks") == 6

        migration.unpartition_item_tables(conn, schema=SCHEMA)

        assert _relkind(conn, "chunks") == "r"
        assert _relkind(conn, "chunks_default") is None
        assert _relkind(conn, "chunks_kb_extra") is None
        assert _count(conn, "chunks") == 6
        assert _count(conn, "chunks", KB_A) == 3
