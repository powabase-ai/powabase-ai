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
from unittest.mock import MagicMock, patch

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


def _run_search(session, method: str, **kwargs) -> None:
    """Drive one of the three searches that filter on metadata in SQL."""
    store = _ChunkStore(db_session=session, knowledge_base_id=KB)
    if method == "vector_search":
        asyncio.run(store.vector_search(embedding=[0.0] * 8, top_k=5, **kwargs))
    elif method == "full_text_search":
        asyncio.run(store.full_text_search("hiking", top_k=5, **kwargs))
    elif method == "pg_bm25_search":
        asyncio.run(store.pg_bm25_search("hiking", top_k=5, **kwargs))
    else:  # pragma: no cover - guards a typo in a parametrisation
        raise AssertionError(f"unknown method {method}")


def _search_sql(method: str, **kwargs) -> tuple[str, dict]:
    """Run one search against a spy session, return its main statement."""
    session = _spy_session()
    _run_search(session, method, **kwargs)
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
@pytest.mark.parametrize("crafted", [CROSS_KB_KEY, ORACLE_KEY], ids=["cross-kb", "oracle"])
def test_a_crafted_filter_key_never_reaches_the_statement(method, crafted):
    """The statement is the one an ordinary two-key filter produces, byte for byte.

    Asserting the crafted text is absent is weaker than it looks — a statement
    that happens not to contain ``OR`` today would pass it. Equality against a
    benign filter of the same size cannot pass by accident: the key material has
    to be somewhere, and the only place left is the bound value.
    """
    sql, params = _search_sql(method, filter_metadata={"lang": "de", crafted: 1})
    benign, _ = _search_sql(method, filter_metadata={"lang": "de", "tier": 1})
    assert sql == benign, f"a filter key changed the statement:\n{sql}"
    assert crafted not in sql
    assert OTHER_KB not in sql
    # The key travels as data instead, inside the one bound value.
    assert crafted in json.loads(params["filter_metadata"])


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
    _run_search(session, method, filter_metadata=empty)
    for sql, params in session.calls:
        assert "c.meta @>" not in sql
        assert "filter_metadata" not in params


# ---------------------------------------------------------------------------
# A filter that is not an object is refused, not quietly answered with nothing
# ---------------------------------------------------------------------------

# Every one of these is truthy, so it reaches the clause builder. Bound as jsonb
# they would all be valid SQL and all false — `jsonb @> <array|string|number>`
# does not error — so the search would answer 200 with no rows and no log line.
NON_OBJECT_FILTERS = [
    pytest.param(["a", "b"], id="list"),
    pytest.param("hello", id="string"),
    pytest.param(42, id="int"),
    pytest.param(3.5, id="float"),
    pytest.param(True, id="bool"),
    pytest.param(("a",), id="tuple"),
]


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("bad", NON_OBJECT_FILTERS)
def test_a_filter_that_is_not_an_object_is_refused(method, bad):
    """A malformed filter must fail loudly rather than silently return nothing.

    An empty result set is indistinguishable from "nothing matched", so a typo in
    a stored knowledge-base config would make that knowledge base invisible for
    as long as the config stood. ValueError is what the search route turns into a
    400 — see test_the_route_answers_400_for_a_filter_that_is_not_an_object.
    """
    with pytest.raises(ValueError, match="filter_metadata must be a JSON object"):
        _search_sql(method, filter_metadata=bad)


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("bad", NON_OBJECT_FILTERS)
def test_a_filter_that_is_not_an_object_never_reaches_the_database(method, bad):
    """The refusal happens before the search statement is executed."""
    session = _spy_session()
    with pytest.raises(ValueError):
        _run_search(session, method, filter_metadata=bad)
    assert [sql for sql, _ in session.calls if "c.meta @>" in sql] == []


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("falsy", [[], "", 0, False], ids=["empty-list", "empty-str", "0", "False"])
def test_a_falsy_filter_still_adds_no_clause(method, falsy):
    """The guard sits after the falsy short-circuit, on purpose.

    These have never added a clause and have never raised, so the guard must not
    start rejecting them: the set of inputs that filter nothing is unchanged.
    """
    session = _spy_session()
    _run_search(session, method, filter_metadata=falsy)
    for sql, params in session.calls:
        assert "c.meta @>" not in sql
        assert "filter_metadata" not in params


def test_the_route_answers_400_for_a_filter_that_is_not_an_object():
    """The helper's ValueError reaches the caller as a 400, not a 500.

    The search route already has `except ValueError`; this drives the real route
    with the real exception the real helper raises, so the two halves are pinned
    together rather than each assumed.
    """
    from flask import Flask

    from agentic_project_service.routes import knowledge_bases as kb_route
    from agentic_project_service.services.base_vector_store import metadata_filter_clause

    app = Flask(__name__)
    app.register_blueprint(kb_route.knowledge_bases_bp)
    kb_id = str(uuid.uuid4())

    with (
        patch("agentic_project_service.auth.decode_jwt", return_value={"role": "authenticated"}),
        patch("agentic_project_service.routes.knowledge_bases.db"),
        patch(
            "agentic_project_service.services.knowledge_search.search_knowledge_base",
            side_effect=lambda **kw: metadata_filter_clause(kw["filter_metadata"]),
        ),
        app.test_client() as client,
    ):
        resp = client.post(
            f"/api/knowledge-bases/{kb_id}/search",
            headers={"Authorization": "Bearer fake.jwt.token"},
            json={"query": "hiking", "filter_metadata": ["premium"]},
        )

    assert resp.status_code == 400, resp.data[:300]
    assert "filter_metadata must be a JSON object" in resp.get_json()["error"]
    assert "list" in resp.get_json()["error"]


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
