"""Revision 0031 leaves the new parents with planner statistics.

Autovacuum never analyses a partitioned parent (Postgres 15), so without an
explicit ANALYZE the planner has no statistics for statements that name the
parent -- which is every statement the application issues.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from tests.pg_search.test_partition_migration import (
    SCHEMA,
    _create_unpartitioned,
    database_url_or_skip,
    load_revision,
)

TABLES = ("chunks", "full_documents", "graph_index_nodes")


@pytest.fixture(scope="module")
def migration():
    return load_revision("0031_partition_bm25_item_tables.py", "mig_0031_analyze")


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(database_url_or_skip())
    yield eng
    eng.dispose()


@pytest.fixture
def prefilled(engine):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        _create_unpartitioned(conn)
    yield
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))


def _inherited_stat_rows(conn, table: str) -> int:
    return conn.execute(
        text(
            "SELECT count(*) FROM pg_statistic "
            f"WHERE starelid = '{SCHEMA}.{table}'::regclass AND stainherit"
        )
    ).scalar()


def test_every_parent_is_analysed(migration, engine, prefilled):
    with engine.begin() as conn:
        migration.partition_item_tables(conn, schema=SCHEMA)

    with engine.connect() as conn:
        for table in TABLES:
            assert _inherited_stat_rows(conn, table) > 0, table
