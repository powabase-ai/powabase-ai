"""Database-backed proof that a metadata filter cannot widen a search.

The unit specs show the statement text no longer carries a filter's keys. Only
Postgres shows what that is worth: that a crafted key cannot reach another
knowledge base's rows, and that its truth value cannot be read off the result.

Both used to be possible. The filter was assembled one key at a time with the
key interpolated into the statement as the name of its own bind parameter, and
``filter_metadata`` is caller data (``POST /api/knowledge-bases/<id>/search``),
so a key closing the cast and adding a predicate of its own was appended to the
WHERE clause. The searches below therefore assert on rows, not on SQL: a key
that changes which rows come back is the whole defect.

Needs only pgvector, runs against ``DATABASE_URL`` (or
``PG_SEARCH_TEST_DATABASE_URL``), and works in a scratch schema of its own, so
nothing here touches the ``ai`` schema or needs the ``app`` fixture.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from agentic_project_service.services.base_vector_store import BasePgVectorStore

SCHEMA = "metadata_filter_live_test"

KB_A = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
KB_B = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
SOURCE = "11111111-1111-4111-8111-111111111111"

DIMS = 4
EMBEDDING = [1.0, 0.0, 0.0, 0.0]

# Three rows per knowledge base. Text is shared so a keyword search matches in
# both, and meta differs so a legitimate filter has something to select on.
DOCS = {
    KB_A: [
        ("alpha one hiking notes", {"lang": "de", "page": 1}),
        ("alpha two hiking notes", {"lang": "de", "page": 2}),
        ("alpha three hiking notes", {"lang": "en", "page": 3}),
    ],
    KB_B: [
        ("beta one hiking notes", {"lang": "de", "page": 1}),
        ("beta two hiking notes", {"lang": "fr", "page": 2}),
        ("beta three hiking notes", {"lang": "fr", "page": 3}),
    ],
}

# The keys a caller can no longer smuggle into the statement. Each pairs with a
# benign "lang" key, which used to supply the bind the interpolated name needed.
CROSS_KB_KEY = f"lang AS jsonb) OR c.knowledge_base_id = '{KB_B}' --"
ORACLE_TRUE_KEY = "lang AS jsonb) OR (SELECT count(*) FROM pg_authid) > 0 --"
ORACLE_FALSE_KEY = "lang AS jsonb) OR (SELECT count(*) FROM pg_authid) < 0 --"
# A key with no SQL in it at all, for comparison: this is what a crafted key is
# supposed to behave like, an ordinary filter that matches nothing.
INERT_KEY = "no row has this key"


class _ChunkStore(BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


@pytest.fixture(autouse=True)
def db_cleanup():
    """No-op override of the parent conftest's ai-schema truncation."""
    yield


@pytest.fixture(scope="module")
def engine():
    dsn = os.environ.get("PG_SEARCH_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("no DATABASE_URL to test the metadata filter against")
    eng = create_engine(dsn)
    try:
        with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as exc:
        eng.dispose()
        pytest.skip(f"pgvector is not available: {str(exc).splitlines()[0]}")
    yield eng
    eng.dispose()


@pytest.fixture(scope="module", autouse=True)
def scratch_schema(engine):
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))
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
                CREATE TABLE {SCHEMA}.embeddings (
                    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    item_id uuid NOT NULL,
                    item_table text NOT NULL DEFAULT 'chunks',
                    knowledge_base_id uuid NOT NULL,
                    dims smallint NOT NULL,
                    embedding vector NOT NULL
                )
            """)
        )
        for kb_id, docs in DOCS.items():
            for body, meta in docs:
                item_id = conn.execute(
                    text(f"""
                        INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text, meta)
                        VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), :body, CAST(:meta AS jsonb))
                        RETURNING id
                    """),
                    {"kb": kb_id, "src": SOURCE, "body": body, "meta": json.dumps(meta)},
                ).scalar()
                conn.execute(
                    text(f"""
                        INSERT INTO {SCHEMA}.embeddings
                            (item_id, knowledge_base_id, dims, embedding)
                        VALUES (CAST(:item AS uuid), CAST(:kb AS uuid), :dims,
                                CAST(:emb AS vector))
                    """),
                    {
                        "item": str(item_id),
                        "kb": kb_id,
                        "dims": DIMS,
                        "emb": json.dumps(EMBEDDING),
                    },
                )
    yield
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


def _vector(session, kb_id=KB_A, **kwargs):
    store = _ChunkStore(db_session=session, knowledge_base_id=kb_id, schema=SCHEMA)
    try:
        return asyncio.run(store.vector_search(embedding=EMBEDDING, top_k=10, **kwargs))
    finally:
        session.rollback()


def _keyword(session, kb_id=KB_A, **kwargs):
    store = _ChunkStore(db_session=session, knowledge_base_id=kb_id, schema=SCHEMA)
    try:
        return asyncio.run(store.full_text_search("hiking", top_k=10, **kwargs))
    finally:
        session.rollback()


SEARCHES = [pytest.param(_vector, id="vector_search"), pytest.param(_keyword, id="full_text")]


def _texts(items) -> set[str]:
    return {item.text for item in items}


# ---------------------------------------------------------------------------
# Positive controls: the rows the injection went after really are there
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("search", SEARCHES)
def test_each_knowledge_base_answers_only_with_its_own_rows(search, session):
    assert _texts(search(session, kb_id=KB_A)) == {t for t, _ in DOCS[KB_A]}
    assert _texts(search(session, kb_id=KB_B)) == {t for t, _ in DOCS[KB_B]}


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("search", SEARCHES)
def test_a_crafted_filter_key_cannot_reach_another_knowledge_base(search, session):
    """A store scoped to KB_A must not return KB_B's rows, whatever the key says."""
    leaked = search(session, kb_id=KB_A, filter_metadata={"lang": "de", CROSS_KB_KEY: 1})
    assert _texts(leaked) & {t for t, _ in DOCS[KB_B]} == set(), (
        f"a filter key returned another knowledge base's rows: {_texts(leaked)}"
    )
    # And the key did nothing beyond being a filter nothing matches.
    inert = search(session, kb_id=KB_A, filter_metadata={"lang": "de", INERT_KEY: 1})
    assert _texts(leaked) == _texts(inert) == set()


@pytest.mark.parametrize("search", SEARCHES)
def test_a_filter_key_cannot_answer_a_question_about_the_database(search, session):
    """The two keys differ only in a predicate that is true one way and false the
    other. If the result differs with them, one bit of the database's state has
    been read through a search request, and the read can be repeated."""
    yes = search(session, kb_id=KB_A, filter_metadata={"lang": "de", ORACLE_TRUE_KEY: 1})
    no = search(session, kb_id=KB_A, filter_metadata={"lang": "de", ORACLE_FALSE_KEY: 1})
    assert _texts(yes) == _texts(no) == set()


@pytest.mark.parametrize("search", SEARCHES)
def test_a_filter_key_cannot_drop_the_knowledge_base_predicate(search, session):
    """`OR 1=1` is the shortest form of the same defect: it would return the whole
    table, both knowledge bases included."""
    rows = search(
        session, kb_id=KB_A, filter_metadata={"lang": "de", "lang AS jsonb) OR 1=1 --": 1}
    )
    assert _texts(rows) == set()


# ---------------------------------------------------------------------------
# Legitimate filters still select the rows they used to
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("search", SEARCHES)
def test_a_filter_selects_by_value(search, session):
    assert _texts(search(session, filter_metadata={"lang": "de"})) == {
        "alpha one hiking notes",
        "alpha two hiking notes",
    }
    assert _texts(search(session, filter_metadata={"lang": "en"})) == {"alpha three hiking notes"}
    assert search(session, filter_metadata={"lang": "fr"}) == []


@pytest.mark.parametrize("search", SEARCHES)
def test_several_keys_are_all_required(search, session):
    assert _texts(search(session, filter_metadata={"lang": "de", "page": 2})) == {
        "alpha two hiking notes"
    }
    # page 3 is the "en" row, so the two keys together match nothing.
    assert search(session, filter_metadata={"lang": "de", "page": 3}) == []


@pytest.mark.parametrize("search", SEARCHES)
def test_a_numeric_value_matches_the_number_it_was_stored_as(search, session):
    assert _texts(search(session, filter_metadata={"page": 1})) == {"alpha one hiking notes"}
    assert search(session, filter_metadata={"page": 99}) == []


@pytest.mark.parametrize("search", SEARCHES)
def test_an_empty_filter_filters_nothing(search, session):
    for empty in (None, {}):
        assert _texts(search(session, filter_metadata=empty)) == {t for t, _ in DOCS[KB_A]}


ODD_KEY_TEXT = "alpha four hiking notes"
ODD_KEYS = {
    "doc.lang": "de",
    "two words": "yes",
    "Sprache": "Übersetzung",
    "outer": {"inner": "value"},
}


@pytest.fixture
def odd_key_row(engine):
    """One extra KB_A row carrying keys that are not identifiers.

    Added and removed per test rather than written into the module fixture, so
    every other test here still sees exactly the rows DOCS declares and the file
    does not depend on collection order.
    """
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        item_id = conn.execute(
            text(f"""
                INSERT INTO {SCHEMA}.chunks (knowledge_base_id, source_id, text, meta)
                VALUES (CAST(:kb AS uuid), CAST(:src AS uuid), :body, CAST(:meta AS jsonb))
                RETURNING id
            """),
            {"kb": KB_A, "src": SOURCE, "body": ODD_KEY_TEXT, "meta": json.dumps(ODD_KEYS)},
        ).scalar()
        conn.execute(
            text(f"""
                INSERT INTO {SCHEMA}.embeddings (item_id, knowledge_base_id, dims, embedding)
                VALUES (CAST(:item AS uuid), CAST(:kb AS uuid), :dims, CAST(:emb AS vector))
            """),
            {"item": str(item_id), "kb": KB_A, "dims": DIMS, "emb": json.dumps(EMBEDDING)},
        )
    yield
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(
            text(f"DELETE FROM {SCHEMA}.embeddings WHERE item_id = CAST(:item AS uuid)"),
            {"item": str(item_id)},
        )
        conn.execute(
            text(f"DELETE FROM {SCHEMA}.chunks WHERE id = CAST(:item AS uuid)"),
            {"item": str(item_id)},
        )


@pytest.mark.parametrize("search", SEARCHES)
def test_keys_that_are_not_identifiers_are_matched_as_written(search, session, odd_key_row):
    """Nested, dotted and spaced keys are ordinary jsonb keys.

    A dot or a space ended the bind-parameter name and the rest became SQL, so
    these keys could not work while the key was part of the statement. (A
    non-ASCII key like ``Sprache`` did work before — SQLAlchemy's bind-name
    pattern is unicode-aware — and is here because it must keep working.) They
    are the reason to bind the filter rather than screen the keys.
    """
    for filter_metadata in (
        {"doc.lang": "de"},
        {"two words": "yes"},
        {"Sprache": "Übersetzung"},
        {"outer": {"inner": "value"}},
    ):
        assert _texts(search(session, filter_metadata=filter_metadata)) == {ODD_KEY_TEXT}, (
            filter_metadata
        )

    assert search(session, filter_metadata={"outer": {"inner": "other"}}) == []


# ---------------------------------------------------------------------------
# A filter that is not an object
# ---------------------------------------------------------------------------


def test_containment_against_a_non_object_is_false_rather_than_an_error(session):
    """Why a non-object filter has to be refused in Python.

    Postgres does not complain about `jsonb @> <array|string|number|boolean>`; it
    answers false. So binding one would answer the request with no rows, no error
    and no log line — which on the agent path reads as "nothing relevant".
    """
    for literal in ('["a", "b"]', '"hello"', "42", "true"):
        answer = session.execute(
            text("SELECT CAST(:meta AS jsonb) @> CAST(:filter AS jsonb)"),
            {"meta": '{"lang": "de"}', "filter": literal},
        ).scalar()
        assert answer is False, literal
    session.rollback()


@pytest.mark.parametrize("search", SEARCHES)
@pytest.mark.parametrize(
    "bad", [["premium"], "premium", 42, True], ids=["list", "str", "int", "bool"]
)
def test_a_filter_that_is_not_an_object_is_refused(search, session, bad):
    with pytest.raises(ValueError, match="filter_metadata must be a JSON object"):
        search(session, filter_metadata=bad)
