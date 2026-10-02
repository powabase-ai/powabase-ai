"""Database-backed tests for the graph_index stores' retrieval-path queries.

The expansion unit tests fake the store, so nothing there exercises these
queries — and they are the kind that unit fakes cannot vouch for: a window
function that pages per document, a count that has to survive that paging,
a knowledge_base_id filter whose absence is invisible until one project
has two knowledge bases, and which partitions a lookup's plan reads.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from agentic.knowledge.models import RetrievedItem
from agentic_project_service.db import db
from agentic_project_service.services import base_toc_store
from agentic_project_service.services.graph_index_node_store import GraphIndexNodeStore
from agentic_project_service.services.graph_index_store import GraphIndexStore


@pytest.fixture
def toc_with_nodes(app, test_source, test_knowledge_base):
    """Insert one ToC and a handful of nodes, deliberately out of order.

    ``indexed_source_id`` is left NULL — it is nullable on both tables, and
    the outline query neither selects nor joins on it.
    """
    toc_id = str(uuid.uuid4())
    ids = {"kb_id": test_knowledge_base["id"], "source_id": test_source["id"]}
    with app.app_context():
        db.session.execute(
            text("""
                INSERT INTO "ai".graph_index_toc
                    (id, knowledge_base_id, source_id,
                     doc_name, doc_description, structure)
                VALUES (:id, :kb_id, :sid, :doc_name, '', CAST('[]' AS jsonb))
                """),
            {
                "id": toc_id,
                "kb_id": ids["kb_id"],
                "sid": ids["source_id"],
                "doc_name": "Master Services Agreement",
            },
        )
        # Inserted 0003, 0001, 0002 so row order differs from document order.
        for node_id, title, depth in [
            ("0003", "Scope of Indemnity", 1),
            ("0001", "Definitions", 0),
            ("0002", "Indemnification", 0),
        ]:
            db.session.execute(
                text("""
                    INSERT INTO "ai".graph_index_nodes
                        (id, toc_id, knowledge_base_id,
                         source_id, node_id, title, depth, text, meta)
                    VALUES (:id, :toc_id, :kb_id, :sid, :node_id,
                            :title, :depth, :text, CAST('{}' AS jsonb))
                    """),
                {
                    "id": str(uuid.uuid4()),
                    "toc_id": toc_id,
                    "kb_id": ids["kb_id"],
                    "sid": ids["source_id"],
                    "node_id": node_id,
                    "title": title,
                    "depth": depth,
                    "text": f"body of {node_id}",
                },
            )
        db.session.commit()
    return {"toc_id": toc_id, **ids}


@pytest.mark.integration
class TestGetTocOutline:
    def test_returns_nodes_in_document_order_with_document_metadata(self, app, toc_with_nodes):
        with app.app_context():
            store = GraphIndexStore(
                db_session=db.session, knowledge_base_id=toc_with_nodes["kb_id"]
            )
            outlines = store.get_toc_outline([toc_with_nodes["toc_id"]], 200)

        outline = outlines[toc_with_nodes["toc_id"]]
        assert [n["node_id"] for n in outline["nodes"]] == ["0001", "0002", "0003"]
        assert outline["nodes"][2] == {
            "node_id": "0003",
            "title": "Scope of Indemnity",
            "depth": 1,
        }
        assert outline["doc_name"] == "Master Services Agreement"
        assert outline["source_id"] == toc_with_nodes["source_id"]

    def test_limit_pages_the_query_but_total_counts_every_section(self, app, toc_with_nodes):
        """The renderer's "N more sections" marker is only honest if the count
        survives paging."""
        with app.app_context():
            store = GraphIndexStore(
                db_session=db.session, knowledge_base_id=toc_with_nodes["kb_id"]
            )
            outlines = store.get_toc_outline([toc_with_nodes["toc_id"]], 2)

        outline = outlines[toc_with_nodes["toc_id"]]
        assert [n["node_id"] for n in outline["nodes"]] == ["0001", "0002"]
        assert outline["total_nodes"] == 3

    def test_unknown_toc_id_returns_no_entry(self, app, toc_with_nodes):
        with app.app_context():
            store = GraphIndexStore(
                db_session=db.session, knowledge_base_id=toc_with_nodes["kb_id"]
            )
            outlines = store.get_toc_outline([str(uuid.uuid4())], 200)

        assert outlines == {}

    def test_a_toc_from_another_knowledge_base_is_not_returned(self, app, toc_with_nodes):
        """The toc_id alone identifies the row; the outline item built from it
        is stamped with the *searching* KB's id, so a toc belonging to another
        knowledge base would reach the model mislabelled as this one's.
        Dropping the filter passes every other test in this file."""
        with app.app_context():
            store = GraphIndexStore(db_session=db.session, knowledge_base_id=str(uuid.uuid4()))
            outlines = store.get_toc_outline([toc_with_nodes["toc_id"]], 200)

        assert outlines == {}

    def test_empty_selection_does_not_query(self, app, toc_with_nodes):
        with app.app_context():
            store = GraphIndexStore(
                db_session=db.session, knowledge_base_id=toc_with_nodes["kb_id"]
            )
            assert store.get_toc_outline([], 200) == {}


# ---------------------------------------------------------------------------
# Node lookups on a partitioned graph_index_nodes read only the KB's partition
# ---------------------------------------------------------------------------

PRUNE_SCHEMA = "gi_prune_test"
KB_MINE = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
KB_OTHER = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
PART_MINE = "graph_index_nodes_kb_mine"
PART_OTHER = "graph_index_nodes_kb_other"


class _RecordingSession:
    """A real session that also remembers each statement it was asked to run."""

    def __init__(self, session):
        self._session = session
        self.statements: list[tuple[str, dict]] = []

    def execute(self, statement, params=None):
        self.statements.append((statement.text, dict(params or {})))
        return self._session.execute(statement, params)

    def __getattr__(self, name):
        return getattr(self._session, name)


@pytest.fixture
def partitioned_nodes(app, monkeypatch):
    """graph_index_nodes as a migrated pg_search database has it: partitioned
    by knowledge base, one partition per KB. Two KBs, each with one ToC holding
    a parent section and its child, under the *same* node_ids -- so a lookup
    that loses its KB or toc scoping shows up as a wrong row, not only a plan.
    """
    monkeypatch.setattr(base_toc_store, "AI_SCHEMA", PRUNE_SCHEMA)
    tocs = {KB_MINE: str(uuid.uuid4()), KB_OTHER: str(uuid.uuid4())}
    node_ids: dict[str, dict[str, str]] = {KB_MINE: {}, KB_OTHER: {}}
    with app.app_context():
        with db.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {PRUNE_SCHEMA} CASCADE"))
            conn.execute(text(f"CREATE SCHEMA {PRUNE_SCHEMA}"))
            conn.execute(
                text(f"""
                    CREATE TABLE {PRUNE_SCHEMA}.graph_index_toc (
                        id uuid PRIMARY KEY,
                        knowledge_base_id uuid NOT NULL,
                        source_id uuid,
                        doc_name text,
                        doc_description text
                    )
                """)
            )
            conn.execute(
                text(f"""
                    CREATE TABLE {PRUNE_SCHEMA}.graph_index_nodes (
                        id uuid NOT NULL DEFAULT gen_random_uuid(),
                        toc_id uuid NOT NULL,
                        knowledge_base_id uuid NOT NULL,
                        source_id uuid NOT NULL,
                        node_id text NOT NULL,
                        title text,
                        text text NOT NULL,
                        depth int,
                        parent_node_id text,
                        line_num int,
                        meta jsonb DEFAULT '{{}}'::jsonb,
                        PRIMARY KEY (knowledge_base_id, id)
                    ) PARTITION BY LIST (knowledge_base_id)
                """)
            )
            for partition, kb in ((PART_MINE, KB_MINE), (PART_OTHER, KB_OTHER)):
                conn.execute(
                    text(
                        f"CREATE TABLE {PRUNE_SCHEMA}.{partition} "
                        f"PARTITION OF {PRUNE_SCHEMA}.graph_index_nodes "
                        f"FOR VALUES IN ('{kb}')"
                    )
                )
            conn.execute(
                text(
                    f"CREATE TABLE {PRUNE_SCHEMA}.graph_index_nodes_default "
                    f"PARTITION OF {PRUNE_SCHEMA}.graph_index_nodes DEFAULT"
                )
            )
            source_id = str(uuid.uuid4())
            for kb, toc_id in tocs.items():
                conn.execute(
                    text(
                        f"INSERT INTO {PRUNE_SCHEMA}.graph_index_toc "
                        "(id, knowledge_base_id, source_id, doc_name, doc_description) "
                        "VALUES (:id, :kb, :sid, :name, '')"
                    ),
                    {"id": toc_id, "kb": kb, "sid": source_id, "name": f"doc of {kb}"},
                )
                for node_id, parent in (("0001", None), ("0002", "0001")):
                    row_id = str(uuid.uuid4())
                    node_ids[kb][node_id] = row_id
                    conn.execute(
                        text(
                            f"INSERT INTO {PRUNE_SCHEMA}.graph_index_nodes "
                            "(id, toc_id, knowledge_base_id, source_id, node_id, title, "
                            " text, depth, parent_node_id) "
                            "VALUES (:id, :toc, :kb, :sid, :node_id, :title, :text, :depth, "
                            " :parent)"
                        ),
                        {
                            "id": row_id,
                            "toc": toc_id,
                            "kb": kb,
                            "sid": source_id,
                            "node_id": node_id,
                            "title": f"section {node_id}",
                            "text": f"body {node_id} of {kb}",
                            "depth": 0 if parent is None else 1,
                            "parent": parent,
                        },
                    )
    yield {"tocs": tocs, "node_ids": node_ids}
    with app.app_context():
        with db.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {PRUNE_SCHEMA} CASCADE"))


def _partitions_read(session, statements) -> set[str]:
    """Every partition the plans of these statements touch."""
    assert statements, "the lookup ran no statement"
    touched: set[str] = set()
    for sql, params in statements:
        plan = "\n".join(
            row[0] for row in session.execute(text(f"EXPLAIN (COSTS OFF) {sql}"), params)
        )
        touched.update(p for p in (PART_MINE, PART_OTHER, "graph_index_nodes_default") if p in plan)
    return touched


@pytest.mark.integration
class TestNodeLookupsReadOnlyTheKbsPartition:
    """A lookup by toc or row id alone cannot be pruned: Postgres plans every
    knowledge base's partition and opens all of their indexes, and the pooled
    connection keeps that relcache for good. On a project with 512 graph_index
    knowledge bases that was ~65 MB per connection -- enough to OOM-kill a
    512 MiB Postgres over and over. Naming the KB keeps it to one partition."""

    def test_referenced_nodes_are_fetched_from_the_kbs_partition_only(self, app, partitioned_nodes):
        toc = partitioned_nodes["tocs"][KB_MINE]
        with app.app_context():
            session = _RecordingSession(db.session)
            store = GraphIndexStore(db_session=session, knowledge_base_id=KB_MINE)
            nodes = store.get_nodes_by_ids([(toc, "0001"), (toc, "0002")])

            assert {key: n["text"] for key, n in nodes.items()} == {
                (toc, "0001"): f"body 0001 of {KB_MINE}",
                (toc, "0002"): f"body 0002 of {KB_MINE}",
            }
            assert _partitions_read(db.session, session.statements) == {PART_MINE}

    def test_children_are_fetched_from_the_kbs_partition_only(self, app, partitioned_nodes):
        toc = partitioned_nodes["tocs"][KB_MINE]
        with app.app_context():
            session = _RecordingSession(db.session)
            store = GraphIndexStore(db_session=session, knowledge_base_id=KB_MINE)
            children = store.get_children_by_parent_ids([(toc, "0001"), (toc, "0009")])

            assert [c["node_id"] for c in children[(toc, "0001")]] == ["0002"]
            assert list(children) == [(toc, "0001")]
            assert _partitions_read(db.session, session.statements) == {PART_MINE}

    def test_another_kbs_toc_is_not_read_through_this_kbs_store(self, app, partitioned_nodes):
        other_toc = partitioned_nodes["tocs"][KB_OTHER]
        with app.app_context():
            store = GraphIndexStore(db_session=db.session, knowledge_base_id=KB_MINE)
            assert store.get_nodes_by_ids([(other_toc, "0001")]) == {}
            assert store.get_children_by_parent_ids([(other_toc, "0001")]) == {}

    def test_search_results_are_resolved_from_the_kbs_partition_only(self, app, partitioned_nodes):
        toc = partitioned_nodes["tocs"][KB_MINE]
        row_id = partitioned_nodes["node_ids"][KB_MINE]["0002"]
        with app.app_context():
            session = _RecordingSession(db.session)
            store = GraphIndexNodeStore(
                db_session=session, knowledge_base_id=KB_MINE, schema=PRUNE_SCHEMA
            )
            item = RetrievedItem(item_id=row_id, text="body", score=1.0, source_id="s", meta={})
            (resolved,) = store._resolve_results([item])

            assert resolved.meta == {
                "title": "section 0002",
                "node_id": "0002",
                "toc_id": toc,
                "depth": 1,
                "doc_name": f"doc of {KB_MINE}",
                "doc_description": "",
            }
            assert _partitions_read(db.session, session.statements) == {PART_MINE}
