"""Unit specs for the per-knowledge-base partition layer.

``pg_search`` allows one ``USING bm25`` index per relation, so every knowledge
base can only have its own index if every knowledge base has its own relation.
The item tables are therefore partitioned ``BY LIST (knowledge_base_id)``, and
these specs pin the part that is pure string work: which tables are
partitioned, what a partition is called, and the exact DDL that creates,
attaches, fills, detaches and indexes one.
"""

from __future__ import annotations

import re
import uuid

import pytest

from agentic_project_service.services import pg_bm25_index as pgb

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
KB2 = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
KB_HEX = uuid.UUID(KB).hex


# ---------------------------------------------------------------------------
# Which tables are partitioned
# ---------------------------------------------------------------------------


def test_the_three_bm25_backed_tables_are_partitioned():
    assert pgb.PARTITIONED_ITEM_TABLES == frozenset(
        {"chunks", "full_documents", "graph_index_nodes"}
    )


def test_doc2json_documents_can_carry_bm25_text_but_is_not_partitioned():
    """It keeps the fallback keyword path, so it is never given a partition."""
    assert "doc2json_documents" in pgb.BM25_ITEM_TABLES
    assert "doc2json_documents" not in pgb.PARTITIONED_ITEM_TABLES


def test_every_partitioned_table_is_a_bm25_item_table():
    assert pgb.PARTITIONED_ITEM_TABLES <= pgb.BM25_ITEM_TABLES


# ---------------------------------------------------------------------------
# Partition naming
# ---------------------------------------------------------------------------


def test_partition_name_is_deterministic_and_derived_from_the_kb_uuid():
    assert pgb.partition_name(KB, "chunks") == f"chunks_kb_{KB_HEX}"
    assert pgb.partition_name(KB, "chunks") == pgb.partition_name(KB, "chunks")
    assert pgb.partition_name(KB, "chunks") != pgb.partition_name(KB2, "chunks")
    assert pgb.partition_name(KB, "chunks") != pgb.partition_name(KB, "full_documents")


def test_partition_name_accepts_a_uuid_object_and_any_uuid_spelling():
    braced = "{3F2504E0-4F89-11D3-9A0C-0305E82C3301}"
    assert pgb.partition_name(uuid.UUID(KB), "chunks") == pgb.partition_name(KB, "chunks")
    assert pgb.partition_name(braced, "chunks") == pgb.partition_name(KB, "chunks")


def test_partition_name_is_a_bare_lowercase_identifier():
    """No dashes, no quoting needed, no case-folding surprises."""
    for item_table in pgb.PARTITIONED_ITEM_TABLES:
        name = pgb.partition_name(KB, item_table)
        assert re.fullmatch(r"[a-z][a-z0-9_]*", name), name


def test_partition_name_fits_the_postgres_identifier_limit_for_every_table():
    for item_table in pgb.PARTITIONED_ITEM_TABLES:
        name = pgb.partition_name(KB, item_table)
        assert len(name.encode("utf-8")) <= 63, (item_table, name)


def test_partition_name_rejects_anything_that_is_not_a_uuid():
    for bad in ["kb'; DROP TABLE ai.chunks; --", "", None, "chunks_kb_x"]:
        with pytest.raises(ValueError):
            pgb.partition_name(bad, "chunks")


def test_partition_name_rejects_a_table_that_is_not_partitioned():
    with pytest.raises(ValueError):
        pgb.partition_name(KB, "doc2json_documents")
    with pytest.raises(ValueError):
        pgb.partition_name(KB, "not_an_item_table")


def test_default_partition_name_is_the_table_plus_default():
    assert pgb.default_partition_name("chunks") == "chunks_default"
    assert pgb.default_partition_name("graph_index_nodes") == "graph_index_nodes_default"
    with pytest.raises(ValueError):
        pgb.default_partition_name("doc2json_documents")


# ---------------------------------------------------------------------------
# Index DDL — now on the partition, and no longer partial
# ---------------------------------------------------------------------------


def test_index_ddl_targets_the_partition_and_has_no_predicate():
    ddl = pgb.bm25_index_ddl(KB, "chunks", "german")

    assert f'ON "ai".chunks_kb_{KB_HEX} ' in ddl
    assert "WHERE" not in ddl
    assert "knowledge_base_id" not in ddl
    assert ddl.startswith(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS bm25_chunks_{KB_HEX} ")
    assert "USING bm25 (id, (text::pdb.simple('stemmer=german')), source_id, meta)" in ddl
    assert "WITH (key_field = 'id')" in ddl


def test_index_ddl_for_the_expression_table_keeps_its_alias():
    ddl = pgb.bm25_index_ddl(KB, "graph_index_nodes", "french")

    assert f'ON "ai".graph_index_nodes_kb_{KB_HEX} ' in ddl
    assert "'alias=bm25_text'" in ddl
    assert "'stemmer=french'" in ddl
    assert "WHERE" not in ddl


def test_index_ddl_for_full_documents_indexes_the_summary():
    ddl = pgb.bm25_index_ddl(KB, "full_documents", None)

    assert f'ON "ai".full_documents_kb_{KB_HEX} ' in ddl
    assert "(summary::pdb.simple)" in ddl
    assert "stemmer" not in ddl


def test_index_ddl_refuses_an_unpartitioned_table():
    """An index on ``doc2json_documents`` has nowhere to live."""
    with pytest.raises(ValueError):
        pgb.bm25_index_ddl(KB, "doc2json_documents", "english")


def test_drop_index_ddl_is_unchanged_and_concurrent():
    assert pgb.bm25_drop_ddl(KB, "chunks") == (
        f'DROP INDEX CONCURRENTLY IF EXISTS "ai".bm25_chunks_{KB_HEX}'
    )


# ---------------------------------------------------------------------------
# Partition lifecycle DDL
# ---------------------------------------------------------------------------


def test_create_partition_ddl_clones_the_default_partition():
    ddl = pgb.partition_create_ddl(KB, "chunks")

    assert ddl == (
        f'CREATE TABLE IF NOT EXISTS "ai".chunks_kb_{KB_HEX} '
        '(LIKE "ai".chunks_default '
        "INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING INDEXES "
        "INCLUDING STORAGE INCLUDING COMMENTS)"
    )


def test_attach_partition_ddl_binds_exactly_this_kb():
    ddl = pgb.partition_attach_ddl(KB, "chunks")

    assert ddl == (
        f'ALTER TABLE "ai".chunks ATTACH PARTITION "ai".chunks_kb_{KB_HEX} FOR VALUES IN (\'{KB}\')'
    )


def test_detach_and_drop_partition_ddl():
    assert pgb.partition_detach_ddl(KB, "full_documents") == (
        f'ALTER TABLE "ai".full_documents DETACH PARTITION "ai".full_documents_kb_{KB_HEX}'
    )
    assert pgb.partition_drop_ddl(KB, "full_documents") == (
        f'DROP TABLE IF EXISTS "ai".full_documents_kb_{KB_HEX}'
    )


def test_evacuation_sql_moves_a_bounded_batch_and_binds_its_values():
    sql = pgb.evacuate_batch_sql(KB, "chunks")

    # Identifiers are interpolated (validated); values are bound.
    assert f'"ai".chunks_kb_{KB_HEX}' in sql
    assert '"ai".chunks_default' in sql
    assert ":kb" in sql
    assert "LIMIT :batch" in sql
    assert KB not in sql, "the KB id must be bound here, not interpolated"
    assert sql.count("DELETE") == 1
    assert sql.count("INSERT") == 1
    assert "RETURNING *" in sql


def test_the_partition_carries_a_check_constraint_matching_its_bound():
    """What lets the ATTACH skip its validation scan of the new partition.

    Postgres skips the scan when the table being attached already has a CHECK
    constraint that implies the partition constraint. Without it, ATTACH reads
    every row of the partition while holding ACCESS EXCLUSIVE -- exactly the
    window this design is trying to keep short.
    """
    ddl = pgb.partition_check_ddl(KB, "chunks")

    assert ddl == (
        f'ALTER TABLE "ai".chunks_kb_{KB_HEX} '
        f"ADD CONSTRAINT chunks_kb_{KB_HEX}_kb_check "
        f"CHECK (knowledge_base_id = '{KB}')"
    )


def test_the_check_constraint_name_fits_the_identifier_limit():
    for item_table in pgb.PARTITIONED_ITEM_TABLES:
        ddl = pgb.partition_check_ddl(KB, item_table)
        name = ddl.split("ADD CONSTRAINT ")[1].split(" ")[0]
        assert len(name.encode("utf-8")) <= 63, (item_table, name)


def test_check_ddl_refuses_an_unpartitioned_table():
    with pytest.raises(ValueError):
        pgb.partition_check_ddl(KB, "doc2json_documents")


# ---------------------------------------------------------------------------
# Serialising partition builds per item table
# ---------------------------------------------------------------------------


def test_the_build_lock_is_a_try_lock_keyed_on_the_qualified_table():
    """Two moves out of the same DEFAULT partition deadlock each other.

    Each holds SHARE on ``<table>_default`` and then asks to upgrade to the
    ACCESS EXCLUSIVE its own ATTACH needs, so each waits for the other's SHARE:
    ``deadlock detected``, observed on a real project. A Postgres advisory lock
    keyed on the item table serialises the whole move instead, and the *try*
    form is what makes the wait bounded -- a caller that cannot get it is told
    to retry rather than left blocking.
    """
    sql = pgb.partition_build_lock_sql()

    assert "pg_try_advisory_lock" in sql
    assert "hashtextextended(:relation, 0)" in sql
    assert "ai" not in sql, "the relation is bound, not interpolated"


def test_the_build_lock_is_released_by_name_not_by_transaction():
    """It has to outlive the transactions, because the move is several of them."""
    assert "pg_advisory_unlock" in pgb.partition_build_unlock_sql()
    assert "hashtextextended(:relation, 0)" in pgb.partition_build_unlock_sql()


def test_the_build_lock_relation_is_schema_qualified():
    assert pgb.partition_build_lock_relation("chunks") == "ai.chunks"
    with pytest.raises(ValueError):
        pgb.partition_build_lock_relation("doc2json_documents")


def test_a_contended_build_is_a_named_error_not_a_bare_exception():
    assert issubclass(pgb.PartitionBuildInProgress, Exception)


def test_the_cutover_batch_is_smaller_than_the_bulk_batch():
    """The cutover batches run under the lock, so they are sized to be quick."""
    assert 0 < pgb.CUTOVER_BATCH_ROWS < pgb.EVACUATION_BATCH_ROWS


def test_the_default_partition_is_locked_against_writers_not_readers():
    """SHARE, because a writer during the move breaks the ATTACH outright.

    A row inserted into DEFAULT after the evacuation drained but before the
    ATTACH makes Postgres refuse the attach ("updated partition constraint for
    default partition would be violated by some row") and the whole move rolls
    back. SHARE conflicts with ROW EXCLUSIVE, so it holds writers off the
    DEFAULT partition for the duration while every reader carries on.
    """
    sql = pgb.partition_lock_default_ddl("chunks")

    assert sql == 'LOCK TABLE "ai".chunks_default IN SHARE MODE'
    with pytest.raises(ValueError):
        pgb.partition_lock_default_ddl("doc2json_documents")


def test_evacuation_sql_refuses_an_unpartitioned_table():
    with pytest.raises(ValueError):
        pgb.evacuate_batch_sql(KB, "doc2json_documents")


def test_mirror_relation_settings_sql_copies_owner_grants_and_rls():
    sql = pgb.mirror_relation_settings_sql('"ai".chunks', '"ai".chunks_kb_x')

    assert "OWNER TO" in sql
    assert "GRANT" in sql
    assert "ROW LEVEL SECURITY" in sql
    assert '"ai".chunks' in sql
    assert '"ai".chunks_kb_x' in sql
