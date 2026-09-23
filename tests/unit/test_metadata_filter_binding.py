"""A metadata filter is bound, never written into the statement.

``filter_metadata`` arrives as a JSON object on the search request, and both its
keys and its values are caller data. The filter used to be assembled one key at
a time with the key interpolated into the statement as the name of its own bind
parameter, so a key carrying SQL of its own landed in the WHERE clause: it could
widen a knowledge-base-scoped search to another knowledge base, and its truth
value was readable from whether rows came back.

These specs pin the statement text for the three searches that filter on
metadata in SQL. The companion live file proves what the rows do.
"""

import asyncio
import json
import uuid
from unittest.mock import MagicMock

import pytest

from agentic_project_service.services import pg_bm25_index as pgb
from agentic_project_service.services.base_vector_store import BasePgVectorStore

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
OTHER_KB = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"

# A key that used to close the CAST, OR in a predicate of its own and comment
# away the rest of the line. The benign key beside it supplies the bind the
# interpolated name used to need.
CROSS_KB_KEY = f"lang AS jsonb) OR c.knowledge_base_id = '{OTHER_KB}' --"
# A key whose subquery is true, so rows come back; negate it and they do not.
# That difference is one bit of whatever the database role can read.
ORACLE_KEY = "lang AS jsonb) OR (SELECT count(*) FROM pg_authid) > 0 --"


class _ChunkStore(BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


def _spy_session():
    """A session that records every statement and returns no rows."""
    session = MagicMock()
    calls: list[tuple[str, dict]] = []

    def execute(statement, params=None):
        calls.append((getattr(statement, "text", str(statement)), params or {}))
        result = MagicMock()
        result.fetchall.return_value = []
        result.__iter__ = lambda self: iter([])
        return result

    session.execute = execute
    session.calls = calls
    return session


def _search_sql(method: str, **kwargs) -> tuple[str, dict]:
    """Run one search against a spy session, return its main statement."""
    session = _spy_session()
    store = _ChunkStore(db_session=session, knowledge_base_id=KB)
    if method == "vector_search":
        asyncio.run(store.vector_search(embedding=[0.0] * 8, top_k=5, **kwargs))
    elif method == "full_text_search":
        asyncio.run(store.full_text_search("hiking", top_k=5, **kwargs))
    elif method == "pg_bm25_search":
        asyncio.run(store.pg_bm25_search("hiking", top_k=5, **kwargs))
    else:  # pragma: no cover - guards a typo in a parametrisation
        raise AssertionError(f"unknown method {method}")
    main = [(sql, params) for sql, params in session.calls if "c.meta @>" in sql]
    assert main, f"no statement with a metadata filter; statements: {session.calls}"
    return main[-1]


_METHODS = ["vector_search", "full_text_search", "pg_bm25_search"]


@pytest.fixture(autouse=True)
def _bm25_partition_exists(monkeypatch):
    """pg_bm25_search needs no readiness probe here; it is called directly."""
    monkeypatch.setattr(pgb, "partition_name", lambda kb, table: f"{table}_kb_{uuid.UUID(kb).hex}")


# ---------------------------------------------------------------------------
# The regression: a crafted key cannot reach the statement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _METHODS)
def test_a_crafted_filter_key_never_reaches_the_statement(method):
    sql, params = _search_sql(method, filter_metadata={"lang": "de", CROSS_KB_KEY: 1})
    assert OTHER_KB not in sql, f"a filter key put another knowledge base in the SQL:\n{sql}"
    assert " OR " not in sql, f"a filter key put a disjunction in the SQL:\n{sql}"
    assert "--" not in sql, f"a filter key put a comment in the SQL:\n{sql}"
    # The key travels as data instead, inside the one bound value.
    assert CROSS_KB_KEY in json.loads(params["filter_metadata"])


@pytest.mark.parametrize("method", _METHODS)
def test_a_boolean_oracle_key_never_reaches_the_statement(method):
    sql, params = _search_sql(method, filter_metadata={"lang": "de", ORACLE_KEY: 1})
    assert "pg_authid" not in sql, f"a filter key put a subquery in the SQL:\n{sql}"
    assert "SELECT count" not in sql, f"a filter key put a subquery in the SQL:\n{sql}"
    assert ORACLE_KEY in json.loads(params["filter_metadata"])


@pytest.mark.parametrize("method", _METHODS)
def test_the_filter_is_one_bound_jsonb_whatever_its_keys(method):
    """One parameter, one clause — the statement text does not vary with the filter."""
    one_key, one_params = _search_sql(method, filter_metadata={"lang": "de"})
    three_keys, three_params = _search_sql(
        method, filter_metadata={"lang": "de", "year": 2026, "kind": "note"}
    )
    assert one_key == three_keys
    assert one_key.count("c.meta @>") == 1
    assert "CAST(:filter_metadata AS jsonb)" in one_key
    assert [p for p in one_params if p.startswith("filter_")] == ["filter_metadata"]
    assert json.loads(one_params["filter_metadata"]) == {"lang": "de"}
    assert json.loads(three_params["filter_metadata"]) == {
        "lang": "de",
        "year": 2026,
        "kind": "note",
    }


# ---------------------------------------------------------------------------
# Legitimate filters still work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filter_metadata",
    [
        pytest.param({"lang": "de"}, id="plain"),
        pytest.param({"page": 3}, id="numeric"),
        pytest.param({"reviewed": True}, id="boolean"),
        pytest.param({"tag": None}, id="null"),
        pytest.param({"tags": ["a", "b"]}, id="list"),
        pytest.param({"outer": {"inner": "value"}}, id="nested"),
        pytest.param({"doc.lang": "de"}, id="dotted"),
        pytest.param({"two words": "yes"}, id="spaced"),
        pytest.param({"Sprache": "Übersetzung"}, id="unicode"),
        pytest.param({"a-b": 1}, id="hyphenated"),
        pytest.param({"x": "it's"}, id="quote-in-value"),
    ],
)
@pytest.mark.parametrize("method", _METHODS)
def test_an_ordinary_filter_is_bound_verbatim(method, filter_metadata):
    sql, params = _search_sql(method, filter_metadata=filter_metadata)
    assert "CAST(:filter_metadata AS jsonb)" in sql
    assert json.loads(params["filter_metadata"]) == filter_metadata


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("empty", [None, {}])
def test_an_empty_filter_adds_no_clause_and_binds_nothing(method, empty):
    session = _spy_session()
    store = _ChunkStore(db_session=session, knowledge_base_id=KB)
    if method == "vector_search":
        asyncio.run(store.vector_search(embedding=[0.0] * 8, top_k=5, filter_metadata=empty))
    elif method == "full_text_search":
        asyncio.run(store.full_text_search("hiking", top_k=5, filter_metadata=empty))
    else:
        asyncio.run(store.pg_bm25_search("hiking", top_k=5, filter_metadata=empty))
    for sql, params in session.calls:
        assert "c.meta @>" not in sql
        assert "filter_metadata" not in params


# ---------------------------------------------------------------------------
# The helper itself. Imported inside the tests so the specs above, which are the
# regression, fail on their own assertions rather than on a collection error.
# ---------------------------------------------------------------------------


def test_the_clause_is_a_containment_test_on_one_bound_parameter():
    from agentic_project_service.services.base_vector_store import metadata_filter_clause

    sql, params = metadata_filter_clause({"lang": "de", "page": 3})
    assert sql == " AND c.meta @> CAST(:filter_metadata AS jsonb)"
    assert params == {"filter_metadata": '{"lang": "de", "page": 3}'}


@pytest.mark.parametrize("empty", [None, {}])
def test_the_clause_is_empty_for_an_empty_filter(empty):
    from agentic_project_service.services.base_vector_store import metadata_filter_clause

    assert metadata_filter_clause(empty) == ("", {})
