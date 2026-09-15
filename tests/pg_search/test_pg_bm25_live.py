"""Database-backed checks for the pg_search BM25 path.

Unit specs show the emitted SQL is shaped right. Only a Postgres with
``pg_search`` installed shows the index builds, the scores are the KB's own,
the filters push down, DML is visible without a reindex, German inflections
match, and an adversarial query answers instead of raising.

Runs against ``PG_SEARCH_TEST_DATABASE_URL`` if set, otherwise ``DATABASE_URL``,
and skips with a reason when that server cannot offer the extension. Every test
works in a scratch schema of its own, so nothing here touches the ``ai`` schema.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services import pg_bm25_index as pgb

SCHEMA = "bm25_live_test"

KB_A = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
KB_B = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"

SOURCE_1 = "11111111-1111-4111-8111-111111111111"
SOURCE_2 = "22222222-2222-4222-8222-222222222222"

# German so the stemmer has something to do; "Beschwerde"/"Beschwerden" and
# "Werkeigentümer"/"Werkeigentümers" are the inflection pairs under test.
KB_A_DOCS = [
    (SOURCE_1, "Die Beschwerde des Werkeigentümers wurde abgewiesen"),
    (SOURCE_1, "Mehrere Beschwerden gingen beim Gericht ein"),
    (SOURCE_2, "Verjährung der Forderung nach drei Jahren"),
]
KB_B_DOCS = [(SOURCE_1, "Eine Beschwerde in einer anderen Wissensbasis")]


class _ChunkStore(bvs.BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


class _NodeStore(bvs.BasePgVectorStore):
    TABLE = "graph_index_nodes"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


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


@pytest.fixture(autouse=True)
def scratch_schema(engine, monkeypatch):
    """A schema shaped like the real item tables, plus the KB rows ensure reads.

    ``pg_bm25_index`` writes its DDL against ``AI_SCHEMA``; pointing that at a
    scratch schema is what lets these tests exercise the real statements
    without touching a real table.
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
                    knowledge_base_id uuid NOT NULL,
                    source_id uuid,
                    text text,
                    meta jsonb DEFAULT '{{}}'::jsonb
                )
            """)
        )
        conn.execute(
            text(f"""
                CREATE TABLE {SCHEMA}.graph_index_nodes (
                    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    knowledge_base_id uuid NOT NULL,
                    source_id uuid,
                    title text,
                    text text,
                    meta jsonb DEFAULT '{{}}'::jsonb
                )
            """)
        )
        for kb_id in (KB_A, KB_B):
            conn.execute(
                text(f"""
                    INSERT INTO {SCHEMA}.knowledge_bases (id, indexing_config, retrieval_config)
                    VALUES (CAST(:id AS uuid), CAST(:ix AS jsonb), CAST(:rx AS jsonb))
                """),
                {
                    "id": kb_id,
                    "ix": json.dumps({"strategy": "chunk_embed"}),
                    "rx": json.dumps({"method": "hybrid", "ts_language": "german"}),
                },
            )
        for kb_id, docs in ((KB_A, KB_A_DOCS), (KB_B, KB_B_DOCS)):
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
                        "meta": json.dumps({"lang": "de"}),
                    },
                )
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
    return asyncio.run(_store(session, kb_id, cls).pg_bm25_search(query, **kwargs))


def _indexdef(session, kb_id=KB_A, item_table="chunks") -> str | None:
    row = session.execute(
        text("SELECT indexdef FROM pg_indexes WHERE schemaname = :s AND indexname = :n"),
        {"s": SCHEMA, "n": pgb.bm25_index_name(kb_id, item_table)},
    ).first()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------


def test_ensure_builds_a_ready_index_and_is_idempotent(engine, session):
    first = pgb.ensure_bm25_index(KB_A, engine=engine)
    assert first == {
        "status": "ready",
        "index": pgb.bm25_index_name(KB_A, "chunks"),
        "item_table": "chunks",
    }
    created = _indexdef(session)
    assert created is not None
    assert "USING bm25" in created
    assert "'stemmer=german'" in created

    second = pgb.ensure_bm25_index(KB_A, engine=engine)
    assert second["status"] == "ready"
    assert _indexdef(session) == created


def test_ensure_rebuilds_when_ts_language_changes(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    assert "'stemmer=german'" in _indexdef(session)

    session.execute(
        text(
            f"UPDATE {SCHEMA}.knowledge_bases "
            "SET retrieval_config = CAST(:rx AS jsonb) WHERE id = CAST(:id AS uuid)"
        ),
        {"rx": json.dumps({"method": "hybrid", "ts_language": "english"}), "id": KB_A},
    )
    session.commit()

    assert pgb.ensure_bm25_index(KB_A, engine=engine)["status"] == "ready"
    rebuilt = _indexdef(session)
    assert "'stemmer=english'" in rebuilt
    assert "'stemmer=german'" not in rebuilt


def test_drop_removes_the_index(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)
    assert _indexdef(session) is not None

    result = pgb.drop_bm25_index(KB_A, engine=engine)

    assert result["status"] == "dropped"
    assert _indexdef(session) is None
    assert pgb.bm25_index_state(session, KB_A, "chunks") == "absent"


def test_a_second_kb_on_the_same_table_is_declined(engine, session):
    """One bm25 index per table, enforced here rather than by pg_search.

    The second KB must not get an index: see
    ``test_a_second_index_breaks_scored_queries_on_the_first`` for what
    building one would cost the first KB. Declining has to be quiet — the
    second KB is still searchable through the existing keyword path, and no
    Celery retry could ever clear the condition.
    """
    assert pgb.ensure_bm25_index(KB_A, engine=engine)["status"] == "ready"

    second = pgb.ensure_bm25_index(KB_B, engine=engine)

    assert second["status"] == "skipped"
    assert second["reason"] == "table_index_conflict"
    assert _indexdef(session, KB_B) is None
    # The first KB's scored search is untouched.
    assert len(_search(session, "Beschwerde", kb_id=KB_A, top_k=10)) == 2


def test_a_second_index_breaks_scored_queries_on_the_first(engine, session):
    """Why the rule above is enforced by us, not left to the extension.

    pg_search refuses a second bm25 index only in the non-concurrent build
    path: a plain CREATE INDEX is rejected, while the CONCURRENTLY form a
    concurrent build has to use creates it. With two present, unscored
    matching still works for both KBs, but the older index can no longer be
    scored — and scores are the whole point. The search path survives it by
    falling back, which is the property pinned here.
    """
    pgb.ensure_bm25_index(KB_A, engine=engine)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(pgb.bm25_index_ddl(KB_B, "chunks", "german")))

    with pytest.raises(Exception, match="Unsupported query shape"):
        _search(session, "Beschwerde", kb_id=KB_A, top_k=10)

    # The session survived it — the failed statement was rolled back to a
    # savepoint, not left as an aborted transaction.
    assert session.execute(text("SELECT 1")).scalar() == 1

    # So bm25s_search answers anyway, by falling back to the tsvector path
    # instead of surfacing the failure to the caller.
    store = _store(session)
    assert isinstance(asyncio.run(store.bm25s_search("Beschwerde", top_k=5)), list)


def test_expression_indexed_table_builds_and_answers(engine, session):
    session.execute(
        text(
            f"UPDATE {SCHEMA}.knowledge_bases SET indexing_config = CAST(:ix AS jsonb) "
            "WHERE id = CAST(:id AS uuid)"
        ),
        {"ix": json.dumps({"strategy": "graph_index"}), "id": KB_A},
    )
    session.execute(
        text(f"""
            INSERT INTO {SCHEMA}.graph_index_nodes (knowledge_base_id, source_id, title, text)
            VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), 'Titel Beschwerde', 'Werkeigentümers')
        """),
        {"kb": KB_A, "src": SOURCE_1},
    )
    session.commit()

    result = pgb.ensure_bm25_index(KB_A, engine=engine)

    assert result == {
        "status": "ready",
        "index": pgb.bm25_index_name(KB_A, "graph_index_nodes"),
        "item_table": "graph_index_nodes",
    }
    # The alias is what makes an indexed expression legal at all.
    assert "'alias=bm25_text'" in _indexdef(session, KB_A, "graph_index_nodes")
    titles = _search(session, "Titel", cls=_NodeStore, top_k=5)
    assert len(titles) == 1


# ---------------------------------------------------------------------------
# Search
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


def test_scores_stay_inside_the_indexed_knowledge_base(engine, session):
    """KB B's rows never appear in KB A's results, and vice versa.

    The second half is the trap this path exists to avoid: with only KB A's
    partial index present, a KB B query does not error, it silently returns
    nothing — which is why readiness is checked per knowledge base before this
    path is ever used.
    """
    pgb.ensure_bm25_index(KB_A, engine=engine)

    a_items = _search(session, "Beschwerde", kb_id=KB_A, top_k=10)
    assert {item.knowledge_base_id for item in a_items} == {KB_A}
    assert all("anderen Wissensbasis" not in item.text for item in a_items)

    assert pgb.bm25_index_ready(session, KB_A, "chunks") is True
    assert pgb.bm25_index_ready(session, KB_B, "chunks") is False
    assert _search(session, "Beschwerde", kb_id=KB_B, top_k=10) == []


def test_source_ids_filter_restricts_results(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)

    both = _search(session, "Beschwerde", top_k=10)
    assert {item.source_id for item in both} == {SOURCE_1}

    only_two = _search(session, "Beschwerde", top_k=10, source_ids=[SOURCE_2])
    assert only_two == []

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


def test_inserts_and_deletes_are_visible_without_a_rebuild(engine, session):
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
    assert len(_search(session, "Verjährung", top_k=10)) == 2

    session.execute(
        text(f"DELETE FROM {SCHEMA}.chunks WHERE id = CAST(:id AS uuid)"), {"id": new_id}
    )
    session.commit()
    assert len(_search(session, "Verjährung", top_k=10)) == 1


def test_german_stemming_matches_inflections(engine, session):
    pgb.ensure_bm25_index(KB_A, engine=engine)

    # Singular query reaches the plural document and back again.
    assert len(_search(session, "Beschwerde", top_k=10)) == 2
    assert len(_search(session, "Beschwerden", top_k=10)) == 2
    # And a query stripped of its genitive still finds it.
    assert len(_search(session, "Werkeigentümer", top_k=10)) == 1


def test_without_a_stemmer_the_inflection_no_longer_matches(engine, session):
    """The stemmer is what does the work, not the tokenizer.

    Rebuilt for a language pg_search cannot stem, the same index keeps working
    but stops matching German inflections — so a KB configured that way gets a
    plain keyword index rather than no index at all.
    """
    session.execute(
        text(
            f"UPDATE {SCHEMA}.knowledge_bases SET retrieval_config = CAST(:rx AS jsonb) "
            "WHERE id = CAST(:id AS uuid)"
        ),
        {"rx": json.dumps({"method": "hybrid", "ts_language": "hindi"}), "id": KB_A},
    )
    session.commit()

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
