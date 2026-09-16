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
4. mirror ownership, grants, the RLS flag and every row-level security
   policy onto the parent, since ``LIKE`` copies none of them. The policies
   matter most: Postgres applies only the policies of the relation a statement
   names, so a parent with RLS enabled and no policy of its own would return
   no rows at all to a role subject to RLS (``authenticated`` on a self-hosted
   install, say), while the DEFAULT partition still held every one. The
   DEFAULT partition keeps its own policies too, for anything that names it
   directly.

Every conversion step is catalog-only, so the migration's cost does not scale
with the number of rows; the one read of data is the closing ``ANALYZE`` of
each parent, which samples a bounded number of rows (autovacuum never analyses
a partitioned parent in Postgres 15, so without it the planner would have no
statistics for any statement that names the parent). It is idempotent (a
table already partitioned is left alone), safe on an empty table, and a no-op
for a table that does not exist yet.

The renames need ACCESS EXCLUSIVE on each table. Migrations run at start-up, so
every lock wait is bounded by ``LOCK_TIMEOUT_MS``: behind a long reader (a
nightly ``pg_dump``) the revision fails with ``MigrationLockTimeout``, logs the
sessions holding the table, rolls the whole transaction back, and runs again on
the next start instead of hanging the boot.

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

# How long any one statement of this revision waits for a table lock. The
# rename needs ACCESS EXCLUSIVE, and migrations run at application start: an
# unbounded wait behind a long reader (a nightly pg_dump holds ACCESS SHARE on
# every table for its whole run) would hang the boot, and queue every query of
# the table behind the waiting rename for just as long.
LOCK_TIMEOUT_MS = 10_000


class MigrationLockTimeout(RuntimeError):
    """A table lock this revision needs was not granted within LOCK_TIMEOUT_MS."""


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
    here. Policies are copied separately by ``_copy_policies``.
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


def _copy_policies(bind, schema: str, source: str, target: str) -> None:
    """Recreate every row-level security policy of ``source`` on ``target``.

    Postgres applies only the policies of the relation a statement names. A
    query through the partitioned parent therefore never consults the
    policies still sitting on the DEFAULT partition, and a parent with RLS
    enabled but no policy of its own denies every row to every role that does
    not bypass RLS. Each policy keeps its name, command, roles,
    permissive/restrictive mode, USING and WITH CHECK expressions. A policy
    whose name already exists on the target is left alone.
    """
    bind.execute(
        _sql(f"""
            DO $$
            DECLARE
                src oid := '"{schema}"."{source}"'::regclass;
                tgt oid := '"{schema}"."{target}"'::regclass;
                p record;
                roles text;
                statement text;
            BEGIN
                FOR p IN
                    SELECT pol.polname, pol.polpermissive, pol.polcmd, pol.polroles,
                           pg_get_expr(pol.polqual, pol.polrelid) AS qual,
                           pg_get_expr(pol.polwithcheck, pol.polrelid) AS with_check
                    FROM pg_policy pol
                    WHERE pol.polrelid = src
                      AND NOT EXISTS (
                          SELECT 1 FROM pg_policy existing
                          WHERE existing.polrelid = tgt AND existing.polname = pol.polname
                      )
                    ORDER BY pol.polname
                LOOP
                    SELECT string_agg(
                               CASE WHEN r = 0 THEN 'PUBLIC'
                                    ELSE quote_ident(pg_get_userbyid(r)) END,
                               ', ')
                      INTO roles
                      FROM unnest(p.polroles) AS r;
                    statement := format(
                        'CREATE POLICY %I ON "{schema}"."{target}" AS %s FOR %s TO %s',
                        p.polname,
                        CASE WHEN p.polpermissive THEN 'PERMISSIVE' ELSE 'RESTRICTIVE' END,
                        CASE p.polcmd WHEN 'r' THEN 'SELECT' WHEN 'a' THEN 'INSERT'
                                      WHEN 'w' THEN 'UPDATE' WHEN 'd' THEN 'DELETE'
                                      ELSE 'ALL' END,
                        roles
                    );
                    IF p.qual IS NOT NULL THEN
                        statement := statement || ' USING (' || p.qual || ')';
                    END IF;
                    IF p.with_check IS NOT NULL THEN
                        statement := statement || ' WITH CHECK (' || p.with_check || ')';
                    END IF;
                    EXECUTE statement;
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
    _copy_policies(bind, schema, default, table)
    # Autovacuum never analyses a partitioned parent, and every application
    # statement names the parent; without this the planner has no statistics
    # for any of them. Samples the partitions, so it reads data, not just the
    # catalog -- a bounded sample per table, not a scan.
    bind.execute(_sql(f'ANALYZE "{schema}"."{table}"'))
    logger.info(
        "Partitioned %s.%s BY LIST (%s); the original table is now its DEFAULT partition",
        schema,
        table,
        PARTITION_KEY,
    )


def _lock_holders(bind, schema: str, table: str) -> list[str]:
    """Best-effort description of the sessions holding a lock on ``table``.

    Read on a separate connection: the migration's own transaction is already
    aborted by the time this is useful.
    """
    try:
        with bind.engine.connect() as probe:
            rows = probe.execute(
                _sql(
                    "SELECT DISTINCT a.pid, a.state, left(a.query, 120) "
                    "FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid "
                    "WHERE l.granted AND l.relation = to_regclass(:rel) "
                    "AND a.pid <> pg_backend_pid()"
                ),
                {"rel": f'"{schema}"."{table}"'},
            ).all()
    except Exception:  # noqa: BLE001 - diagnostics only, never mask the real error
        return []
    return [f"pid {pid} ({state}): {query}" for pid, state, query in rows]


def _run_with_lock_timeout(bind, schema: str, tables, step) -> None:
    """Run ``step(bind, schema, table)`` per table with a bounded lock wait.

    The bound is transaction-local and put back afterwards, so revisions that
    run after this one in the same transaction keep the server's setting. A
    timeout rolls back the whole migration transaction; nothing is half-done,
    and the next start runs this revision again.
    """
    from sqlalchemy.exc import DBAPIError

    previous = bind.execute(_sql("SELECT current_setting('lock_timeout')")).scalar()
    bind.execute(
        _sql("SELECT set_config('lock_timeout', :value, true)"),
        {"value": f"{LOCK_TIMEOUT_MS}ms"},
    )
    for table in tables:
        try:
            step(bind, schema, table)
        except DBAPIError as exc:
            if getattr(exc.orig, "sqlstate", None) != "55P03":
                raise
            holders = _lock_holders(bind, schema, table)
            message = (
                f"Timed out after {LOCK_TIMEOUT_MS} ms waiting for a lock on "
                f"{schema}.{table}: another session holds it (a long-running read "
                "or a pg_dump, typically). The migration transaction is rolled back "
                "and nothing was changed; this revision runs again on the next start."
            )
            logger.error(
                "%s Sessions holding a lock on it: %s", message, "; ".join(holders) or "unknown"
            )
            raise MigrationLockTimeout(message) from exc
    bind.execute(
        _sql("SELECT set_config('lock_timeout', :value, true)"),
        {"value": previous},
    )


def partition_item_tables(bind, schema: str = "ai", tables=ITEM_TABLES) -> None:
    """Convert each item table into a partitioned parent over its own rows."""
    _run_with_lock_timeout(bind, schema, tables, _partition_one)


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
    _run_with_lock_timeout(bind, schema, tables, _unpartition_one)


def upgrade():
    partition_item_tables(op.get_bind())


def downgrade():
    unpartition_item_tables(op.get_bind())
