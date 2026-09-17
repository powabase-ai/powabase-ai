"""What the move does while writers are held off, and what it leaves for later.

Measured on the production schema, maintaining the clone's secondary indexes
and foreign keys row by row during the copy is what made the move block
writers for about a minute per million rows -- almost all of it the
full-text GIN index. So the copy goes into a bare heap; the primary key (and
any other constraint-backed index) is built in bulk inside the move, the
foreign keys are added ``NOT VALID`` inside it, and everything else happens
after the commit without blocking writes: the plain secondary indexes are
built ``CONCURRENTLY`` (after the bm25 index, which is what serves the
knowledge base's keyword search) and the foreign keys are validated.

Same database and scratch schema as ``test_pg_bm25_live``.
"""

from __future__ import annotations

from sqlalchemy import event, text

from agentic_project_service.services import pg_bm25_index as pgb
from tests.pg_search import test_pg_bm25_live as live
from tests.pg_search.test_pg_bm25_live import KB_A, KB_A_DOCS, SCHEMA

migration = live.migration
engine = live.engine
scratch_schema = live.scratch_schema
session = live.session


def _add_production_like_indexes(engine):
    """The shape the real item tables have beyond the key: plain btrees and a GIN."""
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"CREATE INDEX chunks_src_idx ON {SCHEMA}.chunks_default (source_id)"))
        conn.execute(
            text(
                f"CREATE INDEX chunks_fts_idx ON {SCHEMA}.chunks_default "
                "USING gin (to_tsvector('english'::regconfig, text))"
            )
        )
        conn.execute(
            text(
                f"CREATE UNIQUE INDEX chunks_src_text_uq ON {SCHEMA}.chunks_default "
                "(source_id, md5(text))"
            )
        )


def _index_shapes(session, relation) -> set[tuple[bool, str]]:
    """(unique, definition after the relation name) for every index of a relation."""
    rows = session.execute(
        text(
            "SELECT i.indisunique, i.indisvalid, pg_get_indexdef(i.indexrelid), "
            "i.indrelid::regclass::text FROM pg_index i "
            f"WHERE i.indrelid = '{SCHEMA}.{relation}'::regclass"
        )
    ).all()
    session.rollback()
    shapes = set()
    for unique, valid, definition, regclass in rows:
        assert valid, definition
        if " USING bm25 " in definition:
            continue
        shapes.add((unique, definition.split(f" ON {regclass} ", 1)[1]))
    return shapes


def _foreign_keys(session, relation) -> list[tuple[str, bool]]:
    rows = session.execute(
        text(
            "SELECT pg_get_constraintdef(oid), convalidated FROM pg_constraint "
            f"WHERE conrelid = '{SCHEMA}.{relation}'::regclass AND contype = 'f' ORDER BY 1"
        )
    ).all()
    session.rollback()
    return [tuple(row) for row in rows]


def test_the_partition_ends_up_with_every_index_and_key_of_default(engine, session):
    _add_production_like_indexes(engine)

    outcome = pgb.ensure_bm25_index(KB_A, engine=engine)

    assert outcome["status"] == "ready"
    partition = pgb.partition_name(KB_A, "chunks")
    assert _index_shapes(session, partition) == _index_shapes(session, "chunks_default")
    assert [fk for fk, _ in _foreign_keys(session, partition)] == [
        fk for fk, _ in _foreign_keys(session, "chunks_default")
    ]
    assert all(valid for _, valid in _foreign_keys(session, partition))


def test_the_rows_are_copied_into_a_clone_with_no_secondary_index_or_foreign_key(engine, session):
    """What the clone looks like, to every other session, when the copy starts."""
    _add_production_like_indexes(engine)
    partition = pgb.partition_name(KB_A, "chunks")
    seen: dict = {}

    def before_execute(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith(f'INSERT INTO "{SCHEMA}".{partition} '):
            with engine.connect() as other:
                seen["indexes"] = other.execute(
                    text(
                        "SELECT count(*) FROM pg_index "
                        f"WHERE indrelid = to_regclass('{SCHEMA}.{partition}')"
                    )
                ).scalar()
                seen["foreign_keys"] = other.execute(
                    text(
                        "SELECT count(*) FROM pg_constraint "
                        f"WHERE conrelid = to_regclass('{SCHEMA}.{partition}') AND contype = 'f'"
                    )
                ).scalar()

    event.listen(engine, "before_cursor_execute", before_execute)
    try:
        moved = pgb.create_partition(engine, KB_A, "chunks")
    finally:
        event.remove(engine, "before_cursor_execute", before_execute)

    assert moved["rows_moved"] == len(KB_A_DOCS)
    assert seen == {"indexes": 0, "foreign_keys": 0}


def test_the_key_and_foreign_keys_are_in_place_when_the_move_commits(engine, session):
    """Nothing is ever attached without its key, and a delete of the knowledge
    base cascades into the partition from the moment it is attached."""
    _add_production_like_indexes(engine)
    pgb.create_partition(engine, KB_A, "chunks")
    partition = pgb.partition_name(KB_A, "chunks")

    constraints = session.execute(
        text(
            "SELECT contype, convalidated FROM pg_constraint "
            f"WHERE conrelid = '{SCHEMA}.{partition}'::regclass AND contype IN ('p', 'f') "
            "ORDER BY 1"
        )
    ).all()
    session.rollback()
    assert [c for c, _ in constraints] == ["f", "p"]

    session.execute(
        text(f"DELETE FROM {SCHEMA}.knowledge_bases WHERE id = CAST(:kb AS uuid)"), {"kb": KB_A}
    )
    session.commit()
    assert session.execute(text(f"SELECT count(*) FROM {SCHEMA}.{partition}")).scalar() == 0
    session.rollback()


def test_ensure_completes_a_partition_whose_indexes_or_key_validation_never_happened(
    engine, session
):
    """The work after the commit is re-entrant: a worker killed after the move
    committed leaves a partition that ensure finishes on its next run."""
    _add_production_like_indexes(engine)
    pgb.ensure_bm25_index(KB_A, engine=engine)
    partition = pgb.partition_name(KB_A, "chunks")
    expected = _index_shapes(session, "chunks_default")
    secondary = (
        session.execute(
            text(
                "SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                f"WHERE i.indrelid = '{SCHEMA}.{partition}'::regclass AND NOT i.indisprimary "
                "AND pg_get_indexdef(i.indexrelid) NOT LIKE '% USING bm25 %'"
            )
        )
        .scalars()
        .all()
    )
    fk = session.execute(
        text(
            "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint "
            f"WHERE conrelid = '{SCHEMA}.{partition}'::regclass AND contype = 'f'"
        )
    ).first()
    session.rollback()
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for name in secondary:
            conn.execute(text(f"DROP INDEX {SCHEMA}.{name}"))
        conn.execute(text(f"ALTER TABLE {SCHEMA}.{partition} DROP CONSTRAINT {fk[0]}"))
        conn.execute(text(f"ALTER TABLE {SCHEMA}.{partition} ADD {fk[1]} NOT VALID"))
    assert _index_shapes(session, partition) != expected

    outcome = pgb.ensure_bm25_index(KB_A, engine=engine)

    assert outcome["status"] == "ready"
    assert _index_shapes(session, partition) == expected
    assert all(valid for _, valid in _foreign_keys(session, partition))


def test_an_unfinished_partition_is_reported_and_found_by_the_start_up_sweep(engine, session):
    """Between the move's commit and the end of its post-commit work, the
    partition serves keyword search but is not finished: the status must say
    so, and a sweep at start-up must find it if the worker died there."""
    _add_production_like_indexes(engine)
    pgb.create_partition(engine, KB_A, "chunks")

    assert pgb.partition_completion_pending(engine, KB_A, "chunks") is True
    assert pgb.partition_completion_pending(session, KB_A, "chunks") is True
    session.rollback()
    assert (KB_A, "chunks") in pgb.partitions_needing_completion(engine)

    progress: list[str] = []
    outcome = pgb.ensure_bm25_index(KB_A, engine=engine, on_progress=progress.append)

    assert outcome["status"] == "ready"
    assert progress == ["building", "completing"]
    assert pgb.partition_completion_pending(engine, KB_A, "chunks") is False
    assert pgb.partitions_needing_completion(engine) == []
    # A finished partition says so on the next run without recording anything.
    again: list[str] = []
    pgb.ensure_bm25_index(KB_A, engine=engine, on_progress=again.append)
    assert again == []


def test_a_role_with_the_schema_on_its_search_path_can_move(engine, session):
    """With the schema on the search path, pg_get_indexdef names DEFAULT without
    its schema, and reading an index's definition by the qualified name failed
    every move for good."""
    from sqlalchemy import create_engine

    _add_production_like_indexes(engine)
    on_path = create_engine(engine.url, connect_args={"options": f"-c search_path={SCHEMA},public"})
    try:
        outcome = pgb.ensure_bm25_index(KB_A, engine=on_path, allow_row_move=True)
    finally:
        on_path.dispose()

    assert outcome["status"] == "ready", outcome
    partition = pgb.partition_name(KB_A, "chunks")
    assert _index_shapes(session, partition) == _index_shapes(session, "chunks_default")
