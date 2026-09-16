"""Partition the BM25-backed item tables by knowledge base.

``pg_search`` allows exactly one ``USING bm25`` index per relation, and a
scored query has to name a relation that carries one. A single keyword index
per item table therefore serves a single knowledge base. Partitioning each
table ``BY LIST (knowledge_base_id)`` makes that limit per *partition*, so
every knowledge base can have its own index -- with its own stemmer.

The conversion copies nothing. For each table:

1. rename it to ``<table>_default``, which keeps its heap, its
   ``PRIMARY KEY (id)``, its foreign keys, its indexes, its RLS flag and its
   grants exactly as they were;
2. create a partitioned parent under the original name with identical columns,
   types, defaults and NOT NULLs (``LIKE``), and **no primary key** -- a unique
   constraint on a partitioned table must contain every partitioning column, so
   a parent-level ``PRIMARY KEY (id)`` is impossible. Leaving the parent
   without one keeps each partition's key local, which is all ``id`` needs to
   stay the BM25 key field, and avoids widening every key and foreign key to
   ``(id, knowledge_base_id)``;
3. attach the renamed table as the DEFAULT partition, so every knowledge base
   that has not been given a partition of its own keeps reading and writing
   exactly where it did before;
4. mirror ownership, grants and the RLS flag onto the parent, since ``LIKE``
   copies none of them.

Every step is catalog-only, so the migration's cost does not scale with the
number of rows. It is idempotent (a table already partitioned is left alone),
safe on an empty table, and a no-op for a table that does not exist yet.

Any ``USING bm25`` index left on the table by the unpartitioned design is
dropped first: as an index on the DEFAULT partition it would occupy that
partition's single BM25 slot while still only answering one knowledge base.

Revision ID: 0031
Revises: 0030
Create Date: 2026-09-15
"""

import logging

from alembic import op

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

# Kept in step with ``pg_bm25_index.PARTITIONED_ITEM_TABLES``, but spelled out
# here so this revision keeps meaning what it meant when it was written.
ITEM_TABLES = ("chunks", "full_documents", "graph_index_nodes")

PARTITION_KEY = "knowledge_base_id"


def _relkind(bind, schema: str, relname: str) -> str | None:
    row = bind.execute(
        _sql(
            "SELECT relkind FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = :schema AND c.relname = :relname"
        ),
        {"schema": schema, "relname": relname},
    ).first()
    return row[0] if row else None


def _sql(statement: str):
    from sqlalchemy import text

    return text(statement)


def _mirror_settings(bind, schema: str, source: str, target: str) -> None:
    """Copy ownership, GRANTs and the RLS flag from one relation to another.

    Server-side, so every identifier is quoted by ``format()`` rather than
    here. Policies are deliberately not copied: reads through the parent keep
    applying the parent's policies, and the DEFAULT partition still carries the
    ones it always had.
    """
    bind.execute(
        _sql(f"""
            DO $$
            DECLARE
                src oid := '"{schema}"."{source}"'::regclass;
                owner text := (SELECT pg_get_userbyid(relowner) FROM pg_class WHERE oid = src);
                rls boolean := (SELECT relrowsecurity FROM pg_class WHERE oid = src);
                g record;
            BEGIN
                EXECUTE format('ALTER TABLE "{schema}"."{target}" OWNER TO %I', owner);
                IF rls THEN
                    EXECUTE 'ALTER TABLE "{schema}"."{target}" ENABLE ROW LEVEL SECURITY';
                END IF;
                FOR g IN
                    SELECT a.grantee::regrole AS role,
                           string_agg(a.privilege_type, ', ') AS privileges
                    FROM pg_class c, aclexplode(c.relacl) a
                    WHERE c.oid = src AND a.grantee <> 0
                    GROUP BY a.grantee
                LOOP
                    EXECUTE format('GRANT %s ON TABLE "{schema}"."{target}" TO %s',
                                   g.privileges, g.role);
                END LOOP;
            END $$;
        """)
    )


def _drop_bm25_indexes(bind, schema: str, relname: str) -> None:
    for row in bind.execute(
        _sql(
            "SELECT ic.relname FROM pg_index i "
            "JOIN pg_class ic ON ic.oid = i.indexrelid "
            "JOIN pg_am am ON am.oid = ic.relam "
            "JOIN pg_class tc ON tc.oid = i.indrelid "
            "JOIN pg_namespace n ON n.oid = tc.relnamespace "
            "WHERE n.nspname = :schema AND tc.relname = :relname AND am.amname = 'bm25'"
        ),
        {"schema": schema, "relname": relname},
    ).all():
        logger.info("Dropping pre-partition BM25 index %s.%s", schema, row[0])
        bind.execute(_sql(f'DROP INDEX "{schema}"."{row[0]}"'))


def _partition_one(bind, schema: str, table: str) -> None:
    default = f"{table}_default"
    kind = _relkind(bind, schema, table)

    if kind is None:
        logger.info(
            "%s.%s does not exist; nothing to partition (a project created after this "
            "revision gets the table from the schema bootstrap and is partitioned on its "
            "first boot)",
            schema,
            table,
        )
        return
    if kind == "p":
        logger.info("%s.%s is already partitioned; leaving it alone", schema, table)
        return
    if _relkind(bind, schema, default) is not None:
        raise RuntimeError(
            f'"{schema}"."{table}" is a plain table while "{schema}"."{default}" also '
            "exists; a previous run of this revision did not finish. Resolve the two "
            "relations by hand before retrying."
        )

    _drop_bm25_indexes(bind, schema, table)

    bind.execute(_sql(f'ALTER TABLE "{schema}"."{table}" RENAME TO "{default}"'))
    bind.execute(
        _sql(
            f'CREATE TABLE "{schema}"."{table}" '
            f'(LIKE "{schema}"."{default}" '
            "INCLUDING DEFAULTS INCLUDING GENERATED INCLUDING STORAGE INCLUDING COMMENTS) "
            f"PARTITION BY LIST ({PARTITION_KEY})"
        )
    )
    bind.execute(
        _sql(f'ALTER TABLE "{schema}"."{table}" ATTACH PARTITION "{schema}"."{default}" DEFAULT')
    )
    _mirror_settings(bind, schema, default, table)
    logger.info(
        "Partitioned %s.%s BY LIST (%s); the original table is now its DEFAULT partition",
        schema,
        table,
        PARTITION_KEY,
    )


def partition_item_tables(bind, schema: str = "ai", tables=ITEM_TABLES) -> None:
    """Convert each item table into a partitioned parent over its own rows."""
    for table in tables:
        _partition_one(bind, schema, table)


def _unpartition_one(bind, schema: str, table: str) -> None:
    default = f"{table}_default"
    if _relkind(bind, schema, table) != "p":
        logger.info("%s.%s is not partitioned; nothing to undo", schema, table)
        return

    partitions = [
        row[0]
        for row in bind.execute(
            _sql(
                "SELECT c.relname FROM pg_class c "
                "JOIN pg_inherits i ON i.inhrelid = c.oid "
                f'WHERE i.inhparent = \'"{schema}"."{table}"\'::regclass '
                "ORDER BY c.relname"
            )
        ).all()
    ]
    if default not in partitions:
        raise RuntimeError(
            f'"{schema}"."{table}" has no DEFAULT partition named "{default}"; '
            "cannot fold it back into a single table automatically."
        )

    _drop_bm25_indexes(bind, schema, default)
    for partition in partitions:
        if partition == default:
            continue
        _drop_bm25_indexes(bind, schema, partition)
        bind.execute(
            _sql(f'ALTER TABLE "{schema}"."{table}" DETACH PARTITION "{schema}"."{partition}"')
        )
        bind.execute(
            _sql(f'INSERT INTO "{schema}"."{default}" SELECT * FROM "{schema}"."{partition}"')
        )
        bind.execute(_sql(f'DROP TABLE "{schema}"."{partition}"'))

    bind.execute(_sql(f'ALTER TABLE "{schema}"."{table}" DETACH PARTITION "{schema}"."{default}"'))
    bind.execute(_sql(f'DROP TABLE "{schema}"."{table}"'))
    bind.execute(_sql(f'ALTER TABLE "{schema}"."{default}" RENAME TO "{table}"'))
    logger.info("Folded %s.%s back into a single unpartitioned table", schema, table)


def unpartition_item_tables(bind, schema: str = "ai", tables=ITEM_TABLES) -> None:
    """Fold every partition back into one plain table, keeping every row."""
    for table in tables:
        _unpartition_one(bind, schema, table)


def upgrade():
    partition_item_tables(op.get_bind())


def downgrade():
    unpartition_item_tables(op.get_bind())
