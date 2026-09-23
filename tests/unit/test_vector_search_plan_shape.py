"""The shape ``vector_search`` must emit for a prepared statement to keep the index.

``tests/unit/test_base_vector_store_hnsw_cast.py`` pins the distance expression.
These specs pin the other half: which values reach the server as literals, and
what the store does about the one value it cannot turn into a literal -- the
metadata filter.

Both properties are invisible to a single execution. They only matter once
psycopg has prepared the statement and PostgreSQL has adopted its generic plan,
which is a live-Postgres measurement (``tests/pg_search``). What a unit spec can
do is pin the text and the session settings that measurement showed to be the
difference, so a later edit that quietly reverts either one fails here first.
"""

import asyncio
from unittest.mock import MagicMock

from agentic_project_service.services.base_vector_store import BasePgVectorStore

_KB_ID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


class _FakeStore(BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


def _capture(**kwargs) -> list[tuple[str, dict]]:
    """Every statement ``vector_search`` executes, with the parameters it binds."""
    session = MagicMock()
    captured: list[tuple[str, dict]] = []

    def spy_execute(text_obj, params=None):
        sql = text_obj.text if hasattr(text_obj, "text") else str(text_obj)
        captured.append((sql, dict(params or {})))
        return iter([])

    session.execute = spy_execute
    store = _FakeStore(db_session=session, knowledge_base_id=_KB_ID)
    kwargs.setdefault("embedding", [0.0] * 1536)
    kwargs.setdefault("top_k", 10)
    asyncio.run(store.vector_search(**kwargs))
    assert captured, "vector_search did not execute any SQL"
    return captured


def _search(statements: list[tuple[str, dict]]) -> tuple[str, dict]:
    searches = [pair for pair in statements if "ORDER BY" in pair[0]]
    assert searches, f"no search query executed; statements: {statements}"
    return searches[0]


def _settings(statements: list[tuple[str, dict]], guc: str) -> list[int]:
    """Indices of the statements that set ``guc``."""
    return [i for i, (sql, _) in enumerate(statements) if guc in sql]


# ---------------------------------------------------------------------------
# The knowledge base id, on both sides of the join
# ---------------------------------------------------------------------------


def test_both_knowledge_base_predicates_reach_sql_as_literals():
    """Interpolating only the embeddings side is not enough, and that was measured.

    The partial index's predicate is on ``ai.embeddings``, so the embeddings-side
    literal is the one that makes the index *matchable*. But with the item-table
    id still bound, the generic plan has no row estimate for the item side, and
    it costs a hash join plus an exact sort below the ordered index scan it would
    otherwise drive -- so the index is matchable and not chosen. Measured through
    the real driver on a pooled connection: ~0.98 ms while the custom plan held,
    ~135 ms from the execution the generic plan was adopted on, for the life of
    the connection.
    """
    sql, _ = _search(_capture())
    normalized = "".join(sql.split())
    assert f"c.knowledge_base_id='{_KB_ID}'" in normalized, (
        "the item-table knowledge_base_id must be a literal too, or a generic "
        f"plan prices the ordered index scan out:\n{sql}"
    )
    assert f"e.knowledge_base_id='{_KB_ID}'" in normalized, (
        f"the embeddings-side predicate must stay a literal:\n{sql}"
    )


def test_no_knowledge_base_id_is_left_bound():
    """No leftover ``:kb_id``: a bound copy of a literal value is the defect itself."""
    sql, params = _search(_capture())
    assert ":kb_id" not in sql, f"the knowledge base id must not be bound:\n{sql}"
    assert "kb_id" not in params, f"the knowledge base id must not be bound: {params}"


def test_a_bad_knowledge_base_id_is_rejected_on_both_sides():
    """Two interpolations, one gate. Neither may reach SQL unvalidated."""
    session = MagicMock()
    session.execute = lambda *a, **k: iter([])
    store = _FakeStore(db_session=session, knowledge_base_id="not-a-uuid")
    try:
        asyncio.run(store.vector_search(embedding=[0.0] * 8, top_k=1))
    except ValueError as exc:
        assert "knowledge_base_id" in str(exc), exc
    else:
        raise AssertionError("a non-UUID knowledge base id must raise")


# ---------------------------------------------------------------------------
# The metadata filter, which cannot be a literal
# ---------------------------------------------------------------------------


def test_a_filtered_search_asks_for_a_custom_plan():
    """The fourth bound value, and the only one that has to stay bound.

    A generic plan has no selectivity estimate for ``meta @> $n``, so it prices
    the ordered index scan out and falls back to a bitmap scan plus an exact
    sort. Measured: 5.6 ms with the index, 628.8 ms without. The filter value
    cannot be interpolated -- it is caller data -- so the fix is to make this one
    execution plan against the value it actually has.
    """
    statements = _capture(filter_metadata={"tag": "a"})
    forced = _settings(statements, "plan_cache_mode")
    assert forced, (
        "a filtered search must ask for a custom plan, or it loses the partial "
        f"index once the statement is prepared; statements: {statements}"
    )
    normalized = "".join(statements[forced[0]][0].split()).lower()
    assert "setlocalplan_cache_mode" in normalized, statements[forced[0]][0]
    assert "force_custom_plan" in normalized, statements[forced[0]][0]


def test_the_custom_plan_request_precedes_the_search():
    """``SET LOCAL`` only reaches a statement that runs after it, same transaction."""
    statements = _capture(filter_metadata={"tag": "a"})
    first_set = _settings(statements, "plan_cache_mode")[0]
    search_at = next(i for i, (sql, _) in enumerate(statements) if "ORDER BY" in sql)
    assert first_set < search_at, f"plan_cache_mode set after the search: {statements}"


def test_an_unfiltered_search_leaves_the_plan_cache_alone():
    """The unfiltered shape is fully provable from literals, so it keeps its
    cached generic plan -- which is the point of the literals, and worth one
    fewer round trip and one fewer replan per search."""
    statements = _capture()
    assert not _settings(statements, "plan_cache_mode"), (
        f"nothing should touch plan_cache_mode without a filter: {statements}"
    )


def test_an_empty_filter_is_not_a_filter():
    """``filter_metadata={}`` adds no predicate, so it must not cost a replan."""
    statements = _capture(filter_metadata={})
    assert not _settings(statements, "plan_cache_mode"), statements


def test_the_custom_plan_request_is_transaction_scoped():
    """Session-level would follow the connection back into the pool and make
    every later search on it replan."""
    statements = _capture(filter_metadata={"tag": "a"})
    sql = statements[_settings(statements, "plan_cache_mode")[0]][0]
    assert "SET LOCAL" in sql, f"the setting must not outlive the transaction:\n{sql}"


def test_the_custom_plan_request_follows_the_argument_not_the_sql_text():
    """Pinned against how the filter is *compiled*.

    The clause the filter becomes is being rewritten to bind the whole filter as
    one jsonb instead of one parameter per key. That changes the SQL and changes
    nothing about why this setting is needed, so the decision keys off the
    argument: any non-empty filter, whatever it compiles to.
    """
    for filter_metadata in ({"tag": "a"}, {"a": 1, "b": 2}, {"nested": {"x": [1, 2]}}):
        statements = _capture(filter_metadata=filter_metadata)
        assert _settings(statements, "plan_cache_mode"), (
            f"no custom plan requested for {filter_metadata}: {statements}"
        )


def test_a_filter_combined_with_other_predicates_still_asks_for_a_custom_plan():
    statements = _capture(
        filter_metadata={"tag": "a"},
        item_ids={"3f2504e0-4f89-11d3-9a0c-0305e82c3302"},
        source_ids=["3f2504e0-4f89-11d3-9a0c-0305e82c3303"],
    )
    assert _settings(statements, "plan_cache_mode"), statements
