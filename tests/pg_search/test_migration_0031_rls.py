"""Row-level security survives the partitioning migration.

Postgres applies only the *queried* relation's policies. After revision 0031
the name every reader uses (``ai.chunks``) belongs to a brand-new partitioned
parent, so unless the migration copies the original table's policies onto it,
a role that is subject to RLS reads an empty table -- while the renamed
DEFAULT partition, still carrying the old policies, holds every row.

These tests set up the policies a self-hosted schema grants
(``FOR SELECT TO authenticated USING (true)``), run the migration, and read
through the parent as a role that does not bypass RLS.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from sqlalchemy import create_engine

from tests.pg_search.test_partition_migration import (
    KB_A,
    SCHEMA,
    SOURCE,
    _create_unpartitioned,
    database_url_or_skip,
    load_revision,
)

TABLES = ("chunks", "full_documents", "graph_index_nodes")

# Cluster-wide names, so they are specific to this module.
PROBE_ROLE = "bm25_mig_rls_probe"
READER_ROLE = "authenticated"


@pytest.fixture(scope="module")
def migration():
    return load_revision("0031_partition_bm25_item_tables.py", "mig_0031_rls")


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(database_url_or_skip())
    yield eng
    eng.dispose()


def _policies(conn, relname: str) -> list[tuple]:
    return [
        tuple(r)
        for r in conn.execute(
            text(
                "SELECT policyname, permissive, roles::text[], cmd, qual, with_check "
                "FROM pg_policies WHERE schemaname = :s AND tablename = :t "
                "ORDER BY policyname"
            ),
            {"s": SCHEMA, "t": relname},
        ).all()
    ]


def _count_as_probe(conn, relation: str) -> int:
    conn.execute(text(f"SET ROLE {PROBE_ROLE}"))
    try:
        return conn.execute(text(f"SELECT count(*) FROM {SCHEMA}.{relation}")).scalar()
    finally:
        conn.execute(text("RESET ROLE"))


@pytest.fixture
def self_host_policies(engine):
    """The unpartitioned tables, readable by ``authenticated`` through policies."""
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        _create_unpartitioned(conn)
        conn.execute(
            text(
                f"DO $$ BEGIN CREATE ROLE {READER_ROLE} NOLOGIN; "
                "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
        )
        conn.execute(
            text(
                f"DO $$ BEGIN CREATE ROLE {PROBE_ROLE} NOLOGIN NOBYPASSRLS; "
                "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
        )
        conn.execute(text(f"GRANT {READER_ROLE} TO {PROBE_ROLE}"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA {SCHEMA} TO {READER_ROLE}"))
        for table in TABLES:
            conn.execute(text(f"ALTER TABLE {SCHEMA}.{table} ENABLE ROW LEVEL SECURITY"))
            conn.execute(text(f"GRANT SELECT, INSERT ON {SCHEMA}.{table} TO {READER_ROLE}"))
            conn.execute(
                text(
                    f"CREATE POLICY auth_read_{table} ON {SCHEMA}.{table} "
                    f"FOR SELECT TO {READER_ROLE} USING (true)"
                )
            )
        # Every shape a policy can take, so the copy is checked field by field.
        conn.execute(
            text(
                f"CREATE POLICY auth_insert_own_kb ON {SCHEMA}.chunks "
                f"FOR INSERT TO {READER_ROLE}, public "
                f"WITH CHECK (knowledge_base_id = '{KB_A}'::uuid)"
            )
        )
        conn.execute(
            text(
                f"CREATE POLICY restrict_nonempty ON {SCHEMA}.chunks AS RESTRICTIVE "
                f"FOR ALL TO {READER_ROLE} USING (length(text) > 0) "
                "WITH CHECK (length(text) > 0)"
            )
        )
    yield
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))


def test_rls_subject_role_reads_the_same_rows_through_the_parent(
    migration, engine, self_host_policies
):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        before = {t: _count_as_probe(conn, t) for t in TABLES}
        assert before == {t: 6 for t in TABLES}

        migration.partition_item_tables(conn, schema=SCHEMA)

        after = {t: _count_as_probe(conn, t) for t in TABLES}
        assert after == before


def test_every_policy_is_copied_onto_the_parent_and_kept_on_the_default_partition(
    migration, engine, self_host_policies
):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        original = {t: _policies(conn, t) for t in TABLES}
        assert len(original["chunks"]) == 3

        migration.partition_item_tables(conn, schema=SCHEMA)

        for table in TABLES:
            assert _policies(conn, table) == original[table], table
            assert _policies(conn, f"{table}_default") == original[table], table


def test_copied_with_check_policy_still_governs_writes_through_the_parent(
    migration, engine, self_host_policies
):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        migration.partition_item_tables(conn, schema=SCHEMA)

        conn.execute(text(f"SET ROLE {PROBE_ROLE}"))
        try:
            conn.execute(
                text(
                    f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text) "
                    "VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), 'erlaubt')"
                ),
                {"kb": KB_A, "src": SOURCE},
            )
            with pytest.raises(Exception, match="row-level security"):
                conn.execute(
                    text(
                        f"INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text) "
                        "VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), '')"
                    ),
                    {"kb": KB_A, "src": SOURCE},
                )
        finally:
            conn.execute(text("RESET ROLE"))


def test_downgrade_keeps_the_rows_readable_by_an_rls_subject_role(
    migration, engine, self_host_policies
):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        original = {t: _policies(conn, t) for t in TABLES}
        before = {t: _count_as_probe(conn, t) for t in TABLES}

        migration.partition_item_tables(conn, schema=SCHEMA)
        migration.unpartition_item_tables(conn, schema=SCHEMA)

        assert {t: _count_as_probe(conn, t) for t in TABLES} == before
        for table in TABLES:
            assert _policies(conn, table) == original[table], table
