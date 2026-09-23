"""The shape ``vector_search`` must emit for the planner to keep the index.

``tests/unit/test_base_vector_store_hnsw_cast.py`` pins the distance expression.
These specs pin the rest of what the measurements in ``tests/pg_search`` found to
be the difference between using the per-knowledge-base partial HNSW index and
not: which values reach the server as literals, what the store does about the
one value it cannot turn into a literal (the metadata filter), and the session
settings it asks for around the search.

None of it is visible in a single execution's answer. Which is why it is pinned
here as text: a later edit that quietly reverts one of them fails here first,
before the live suite has to catch it with a scan counter.
"""

import asyncio
from unittest.mock import MagicMock

from agentic_project_service.services.base_vector_store import BasePgVectorStore

_KB_ID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"

# Deliberately not "on": a restore that hardcodes the default would pass
# against a fixture whose prior value *was* the default.
ENABLE_SORT_WAS = "off"


class _FakeStore(BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


def _capture(*, partial_index: bool = False, **kwargs) -> list[tuple[str, dict]]:
    """Every statement ``vector_search`` executes, with the parameters it binds.

    ``partial_index`` is the answer the store's catalog probe gets back: whether
    this knowledge base has a valid partial HNSW index. It decides a branch, so
    the fake has to be able to answer both ways. ``ENABLE_SORT_WAS`` is the value
    the probe reports ``enable_sort`` currently has, so a spec can tell a restore
    from a hardcoded ``on``.
    """
    session = MagicMock()
    captured: list[tuple[str, dict]] = []

    def spy_execute(text_obj, params=None):
        sql = text_obj.text if hasattr(text_obj, "text") else str(text_obj)
        captured.append((sql, dict(params or {})))
        if "to_regclass" in sql:
            return iter([(ENABLE_SORT_WAS, partial_index)])
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


# ---------------------------------------------------------------------------
# The sort, which the planner prefers to the index at production widths
#
# At 1536 dimensions a vector is stored out of line, and PostgreSQL prices
# detoasting at nothing -- so an exact scan plus a sort costs less on paper than
# the ordered index scan, and the partial index is never chosen. Measured through
# the real store: 0 of 12 executions on the index at 50.5 ms, against 12 of 12 at
# 5.2 ms once the sort is priced out. These specs pin the three things that
# measurement depends on: that the penalty is applied, that it is applied only
# when there is an index to fall on, and that it is taken back off.
# ---------------------------------------------------------------------------


def _enable_sort(statements: list[tuple[str, dict]]) -> list[int]:
    """Indices of the statements that *change* ``enable_sort``.

    Not every statement mentioning it: the probe reads the current value, and a
    reader is not a writer.
    """
    return _settings(statements, "set_config('enable_sort'")


def test_a_search_asks_whether_this_knowledge_base_has_a_valid_partial_index():
    """The catalog probe, by name and in the store's own schema.

    Ungated, the penalty drives a knowledge base with no index onto the shared
    per-dimension index, which post-filters: measured 9.4 ms and recall 1.00
    became 31.3 ms and recall 0.04 at 5% of the table. So the probe is the fix,
    not an optimisation on it.
    """
    statements = _capture()
    probes = [pair for pair in statements if "to_regclass" in pair[0]]
    assert probes, f"nothing asked whether the partial index exists: {statements}"
    sql, params = probes[0]
    assert "indisvalid" in sql, (
        "an index that is INVALID, or still being built, is in the catalog and "
        f"cannot answer a query; the probe must exclude it:\n{sql}"
    )
    index = params["index"]
    schema = _FakeStore(db_session=MagicMock(), knowledge_base_id=_KB_ID).schema
    assert index.startswith(f'"{schema}".'), (
        f"the probe must look in the schema the search reads: {index}"
    )
    assert index.endswith("_1536"), (
        f"one index per dimension, so the probe has to name this search's: {index}"
    )
    assert "hnsw_kb_" in index and "3f2504e04f8911d39a0c0305e82c3301" in index, (
        f"the probe must name this knowledge base's own index: {index}"
    )


def test_the_sort_is_priced_out_when_the_knowledge_base_has_an_index():
    statements = _capture(partial_index=True)
    forced = _enable_sort(statements)
    assert forced, (
        "with a valid partial index the exact sort must be priced out, or the "
        f"planner keeps choosing it: {statements}"
    )
    sql = statements[forced[0]][0]
    normalized = "".join(sql.split()).lower()
    assert "set_config('enable_sort','off',true)" in normalized, sql


def test_the_sort_is_left_alone_when_there_is_no_index_to_fall_on():
    """A knowledge base below the build threshold, or one whose index is INVALID.

    Both answer the probe the same way, and both must come out with the plan they
    have today -- an exact scan, which at those sizes is also the exact answer.
    """
    statements = _capture(partial_index=False)
    assert not _enable_sort(statements), (
        f"nothing may touch enable_sort without an index to use: {statements}"
    )


def test_pricing_the_sort_out_is_transaction_scoped():
    """The third argument to ``set_config`` is what keeps it out of the pool."""
    statements = _capture(partial_index=True)
    sql = statements[_enable_sort(statements)[0]][0]
    assert "true" in "".join(sql.split()).lower(), (
        f"a session-level setting would follow the connection into the pool:\n{sql}"
    )


def test_the_sort_is_priced_out_before_the_search_and_restored_after():
    """``hybrid_search`` runs its keyword leg on this same transaction, and a
    keyword ranking is a sort. So the penalty has to be off again by the time
    ``vector_search`` returns."""
    statements = _capture(partial_index=True)
    touched = _enable_sort(statements)
    search_at = next(i for i, (sql, _) in enumerate(statements) if "ORDER BY" in sql)
    assert len(touched) == 2, f"expected one set and one restore: {statements}"
    assert touched[0] < search_at < touched[1], (
        f"enable_sort must be set before the search and put back after it: {statements}"
    )


def test_the_restore_puts_back_the_value_the_probe_read():
    """Not a hardcoded ``on``: whatever the transaction had before."""
    statements = _capture(partial_index=True)
    restore_at = _enable_sort(statements)[1]
    sql, params = statements[restore_at]
    assert params.get("prior") == ENABLE_SORT_WAS, (
        f"the restore must bind the value the probe read, not a guess: {sql} {params}"
    )


def test_a_two_key_filtered_search_prices_the_sort_out_as_well():
    """The defect that made this fix necessary rather than deferrable.

    ``jsonb @>`` has no statistics, so two keys estimate near zero rows and the
    sort looks free even to a custom plan that knows the filter's value. Measured:
    0 of 12 executions on the partial index, repaired to 12 of 12. Asking for a
    custom plan is not enough on its own, so both settings have to be here.
    """
    statements = _capture(partial_index=True, filter_metadata={"tier": "gold", "kb": "a"})
    assert _enable_sort(statements), statements
    assert _settings(statements, "plan_cache_mode"), statements


# ---------------------------------------------------------------------------
# The re-run, which is what makes the setting safe on a restricted search
# ---------------------------------------------------------------------------


def test_a_short_answer_is_asked_again_with_the_sort_available():
    """An ordered index scan can starve a selective restriction; an exact scan cannot.

    Measured at 384 dimensions: a ``top_k`` of 20 restricted to 12 named items of
    a 12,000-row knowledge base came back with 11 of them from the forced index
    scan, and all 12 from the exact scan the planner picks on its own. So a short
    answer is re-run -- and the re-run asks for a custom plan, or it would be
    handed the plan that came up short.

    The fake session returns no rows, which is a short answer for any ``top_k``.
    """
    statements = _capture(partial_index=True)
    searches = [i for i, (sql, _) in enumerate(statements) if "ORDER BY" in sql]
    assert len(searches) == 2, (
        f"a short answer from the forced index scan must be asked again: {statements}"
    )
    restored = _enable_sort(statements)[1]
    assert restored < searches[1], (
        f"the re-run has to happen with the sort available again: {statements}"
    )
    replanned = [i for i in _settings(statements, "plan_cache_mode") if i < searches[1]]
    assert replanned and replanned[-1] > searches[0], (
        f"the re-run must ask for a custom plan, between the two searches: {statements}"
    )


def test_a_full_answer_is_not_asked_again():
    """The re-run is for the short case only; an ordinary search pays nothing."""
    session = MagicMock()
    captured: list[tuple[str, dict]] = []
    row = ("11111111-1111-4111-8111-111111111111", "text", 0.5, None, {})

    def spy_execute(text_obj, params=None):
        sql = text_obj.text if hasattr(text_obj, "text") else str(text_obj)
        captured.append((sql, dict(params or {})))
        if "to_regclass" in sql:
            return iter([(ENABLE_SORT_WAS, True)])
        if "ORDER BY" in sql:
            return iter([row] * 3)
        return iter([])

    session.execute = spy_execute
    store = _FakeStore(db_session=session, knowledge_base_id=_KB_ID)
    asyncio.run(store.vector_search(embedding=[0.0] * 1536, top_k=3))
    searches = [pair for pair in captured if "ORDER BY" in pair[0]]
    assert len(searches) == 1, (
        f"top_k rows came back, so nothing is short and nothing is re-run: {captured}"
    )
    assert not _settings(captured, "plan_cache_mode"), captured
