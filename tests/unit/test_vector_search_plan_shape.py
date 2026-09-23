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

import pytest

from agentic_project_service.services.base_vector_store import (
    PER_KB_HNSW_EF_SEARCH,
    BasePgVectorStore,
)

_KB_ID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"

# Deliberately not the defaults ("on" and "40"): a restore that hardcodes what
# the setting usually is would pass against a fixture whose prior value *was* that.
ENABLE_SORT_WAS = "off"
EF_SEARCH_WAS = "55"
PLAN_CACHE_MODE_WAS = "force_generic_plan"


class _FakeStore(BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


class _FakeDocumentStore(BasePgVectorStore):
    """A store on one of the other item tables ``ai.embeddings`` holds.

    Same inherited ``vector_search``, different ``TABLE`` -- which is the whole of
    what tells the two apart, and what the partial index's predicate does not
    know about.
    """

    TABLE = "full_documents"
    TEXT_COL = "summary"
    SEARCH_TEXT_COL = "summary"


def _capture(
    *, partial_index: bool = False, store: type[BasePgVectorStore] = _FakeStore, **kwargs
) -> list[tuple[str, dict]]:
    """Every statement ``vector_search`` executes, with the parameters it binds.

    ``partial_index`` is the answer the store's catalog probe gets back: whether
    this knowledge base has a valid partial HNSW index. It decides a branch, so
    the fake has to be able to answer both ways. ``ENABLE_SORT_WAS`` and
    ``EF_SEARCH_WAS`` are the values the probe reports those settings currently
    have, so a spec can tell a restore from a hardcoded default.
    """
    session = MagicMock()
    captured: list[tuple[str, dict]] = []

    def spy_execute(text_obj, params=None):
        sql = text_obj.text if hasattr(text_obj, "text") else str(text_obj)
        captured.append((sql, dict(params or {})))
        if "to_regclass" in sql:
            return iter([(ENABLE_SORT_WAS, EF_SEARCH_WAS, partial_index)])
        if "current_setting('plan_cache_mode')" in sql:
            return _scalar(PLAN_CACHE_MODE_WAS)
        return iter([])

    session.execute = spy_execute
    store_under_test = store(db_session=session, knowledge_base_id=_KB_ID)
    kwargs.setdefault("embedding", [0.0] * 1536)
    kwargs.setdefault("top_k", 10)
    asyncio.run(store_under_test.vector_search(**kwargs))
    assert captured, "vector_search did not execute any SQL"
    return captured


def _scalar(value):
    """A result object whose ``.scalar()`` answers, for the one read that uses it."""
    result = MagicMock()
    result.scalar.return_value = value
    result.__iter__ = lambda self: iter([(value,)])
    return result


def _writes(statements: list[tuple[str, dict]], guc: str) -> list[int]:
    """Indices of the statements that *change* ``guc``.

    Not every statement mentioning it: the probe and the plan-cache save read the
    current value, and a reader is not a writer. Without this distinction a spec
    that means "this setting was applied" passes on the read alone.
    """
    return _settings(statements, f"set_config('{guc}'")


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
    forced = _writes(statements, "plan_cache_mode")
    assert forced, (
        "a filtered search must ask for a custom plan, or it loses the partial "
        f"index once the statement is prepared; statements: {statements}"
    )
    normalized = "".join(statements[forced[0]][0].split()).lower()
    assert "set_config('plan_cache_mode','force_custom_plan',true)" in normalized, statements[
        forced[0]
    ][0]


def test_the_custom_plan_request_precedes_the_search():
    """``SET LOCAL`` only reaches a statement that runs after it, same transaction."""
    statements = _capture(filter_metadata={"tag": "a"})
    first_set = _writes(statements, "plan_cache_mode")[0]
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
    sql = statements[_writes(statements, "plan_cache_mode")[0]][0]
    assert "true" in "".join(sql.split()).lower(), (
        f"the setting must not outlive the transaction:\n{sql}"
    )


def test_the_custom_plan_request_follows_the_argument_not_the_sql_text():
    """Pinned against how the filter is *compiled*.

    The clause the filter becomes is being rewritten to bind the whole filter as
    one jsonb instead of one parameter per key. That changes the SQL and changes
    nothing about why this setting is needed, so the decision keys off the
    argument: any non-empty filter, whatever it compiles to.
    """
    for filter_metadata in ({"tag": "a"}, {"a": 1, "b": 2}, {"nested": {"x": [1, 2]}}):
        statements = _capture(filter_metadata=filter_metadata)
        assert _writes(statements, "plan_cache_mode"), (
            f"no custom plan requested for {filter_metadata}: {statements}"
        )


def test_a_filter_combined_with_other_predicates_still_asks_for_a_custom_plan():
    statements = _capture(
        filter_metadata={"tag": "a"},
        item_ids={"3f2504e0-4f89-11d3-9a0c-0305e82c3302"},
        source_ids=["3f2504e0-4f89-11d3-9a0c-0305e82c3303"],
    )
    assert _writes(statements, "plan_cache_mode"), statements


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
    """Indices of the statements that *change* ``enable_sort``."""
    return _writes(statements, "enable_sort")


def test_a_search_asks_whether_this_knowledge_base_has_a_valid_partial_index():
    """The catalog probe, by name and in the store's own schema.

    Ungated, the penalty drives a knowledge base with no index onto the shared
    per-dimension index, which post-filters and gets *slower* as the knowledge
    base gets smaller: measured 9.4 ms against an exact scan's 1.00-recall plan at
    5% of the table, and 39.8 ms at 1%, where the real-embedding run's worst-case
    recall is also the only one below half. So the probe is the fix, not an
    optimisation on it.
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


def test_a_filtered_search_keeps_the_sort_and_still_asks_for_a_custom_plan():
    """A metadata filter gets the custom plan and *not* the sort penalty.

    The two settings answer different questions and this is where they part. The
    custom plan is what gets the filter's value to the planner, and it helps every
    filtered search. Pricing the sort out helps only the unfiltered one: measured
    at 1536 dimensions through ``vector_search``, a one-source ``source_ids``
    search went 22.7 -> 28.5 ms at recall 1.00 -> 0.49, a 200-item ``item_ids``
    search 11.0 -> 37.8 ms at 1.00 -> 0.80, and a filter matching no row
    3.0 -> 38.3 ms. Slower and less accurate, on a search whose restriction the
    caller wrote down and expects to be honoured.
    """
    statements = _capture(partial_index=True, filter_metadata={"tier": "gold", "kb": "a"})
    assert _writes(statements, "plan_cache_mode"), statements
    assert not _enable_sort(statements), (
        "a filtered search must keep the plan the planner chooses for it; pricing "
        f"the sort out makes it slower and less complete: {statements}"
    )


@pytest.mark.parametrize(
    "restriction",
    [
        {"item_ids": {"3f2504e0-4f89-11d3-9a0c-0305e82c3302"}},
        {"item_ids": set()},
        {"source_ids": ["3f2504e0-4f89-11d3-9a0c-0305e82c3303"]},
        {"source_ids": []},
        {"filter_metadata": {"tier": "gold"}},
    ],
    ids=["item_ids", "empty-item_ids", "source_ids", "empty-source_ids", "filter_metadata"],
)
def test_a_restricted_search_does_not_enter_the_gate_at_all(restriction):
    """Every restriction the search accepts, including the empty ones.

    The gate's probe answers "does this knowledge base have a valid partial
    index", which is not "is this search better off on it". An empty id set is the
    most starved restriction there is -- it matches nothing -- so keying on
    ``is not None`` rather than truthiness is load-bearing, and pinned here.
    """
    statements = _capture(partial_index=True, **restriction)
    assert not _enable_sort(statements), statements
    assert not _writes(statements, "hnsw.ef_search"), statements
    assert len([s for s, _ in statements if "ORDER BY" in s]) == 1, (
        f"a restricted search must run once, on the planner's own plan: {statements}"
    )


def test_the_gate_is_only_for_the_item_table_it_was_measured_on():
    """``ai.embeddings`` is polymorphic and four stores inherit this search.

    The partial index's predicate and the probe name ``(knowledge_base_id, dims)``
    only, so a knowledge base whose embeddings are mostly chunks but whose search
    routes to a document-level store passes the probe and is driven into an HNSW
    walk of rows that cannot join: measured 2.2 -> 25.9 ms and recall 1.00 -> 0.33
    on a 40-row table. The other stores keep the planner's own choice.
    """
    statements = _capture(partial_index=True, store=_FakeDocumentStore)
    assert not _enable_sort(statements), (
        "only the chunks store may be steered onto the partial index; the probe "
        f"cannot tell which item table the embeddings belong to: {statements}"
    )
    assert not _writes(statements, "hnsw.ef_search"), statements


def test_the_chunks_store_is_the_one_that_does_enter_it():
    """The other half of the spec above: the restriction must not be vacuous.

    Without this, making ``vector_search`` skip the gate for *every* store would
    leave the spec above green.
    """
    statements = _capture(partial_index=True, store=_FakeStore)
    assert _enable_sort(statements), statements


# ---------------------------------------------------------------------------
# The re-run, which is what makes the setting safe on a restricted search
# ---------------------------------------------------------------------------


def test_a_short_answer_is_asked_again_with_the_sort_available():
    """An ordered index scan can starve the ``LIMIT``; an exact scan cannot.

    Measured at 384 dimensions: a ``top_k`` of 20 against 12 matching rows of a
    12,000-row knowledge base came back with 11 of them from the forced index
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
    replanned = [i for i in _writes(statements, "plan_cache_mode") if i < searches[1]]
    assert replanned and replanned[-1] > searches[0], (
        f"the re-run must ask for a custom plan, between the two searches: {statements}"
    )


def test_the_re_runs_custom_plan_is_put_back_too():
    """The re-run's replan must not outlive the search either.

    The same argument as the filtered case below it: in the single-knowledge-base
    fast path this session is the request's, so a ``plan_cache_mode`` left on
    makes the hybrid keyword leg, the metadata reads and the billing writes all
    replan for the rest of the transaction.
    """
    statements = _capture(partial_index=True)
    touched = _writes(statements, "plan_cache_mode")
    assert len(touched) == 2, f"expected one set and one restore: {statements}"
    assert statements[touched[1]][1].get("prior") == PLAN_CACHE_MODE_WAS, (
        f"the restore must bind the value that was read, not a guess: {statements[touched[1]]}"
    )


def test_a_shorter_re_run_does_not_replace_a_longer_first_answer():
    """Both plans read the same rows under the same LIMIT, so normally the re-run
    is at least as complete. A second plan that comes back *shorter* is evidence
    of nothing, and letting it win would turn this net into a way to lose rows."""
    session = MagicMock()
    row = ("11111111-1111-4111-8111-111111111111", "text", 0.5, None, {})
    answers = [[row, row], []]

    def spy_execute(text_obj, params=None):
        sql = text_obj.text if hasattr(text_obj, "text") else str(text_obj)
        if "to_regclass" in sql:
            return iter([(ENABLE_SORT_WAS, EF_SEARCH_WAS, True)])
        if "current_setting('plan_cache_mode')" in sql:
            return _scalar(PLAN_CACHE_MODE_WAS)
        if "ORDER BY" in sql:
            return iter(answers.pop(0))
        return iter([])

    session.execute = spy_execute
    store = _FakeStore(db_session=session, knowledge_base_id=_KB_ID)
    items = asyncio.run(store.vector_search(embedding=[0.0] * 1536, top_k=10))
    assert not answers, "both searches must have run for this spec to mean anything"
    assert len(items) == 2, f"the longer of the two answers must be kept, got {items}"


# ---------------------------------------------------------------------------
# hnsw.ef_search, which decides how accurate the index scan is once it happens
# ---------------------------------------------------------------------------


def test_ef_search_is_raised_for_a_search_on_this_knowledge_bases_own_index():
    """Recall degrades with the *absolute* size of the index, not the selectivity.

    Measured on real embeddings: 0.997 at 400 rows, 0.915 at 12,000, and the build
    threshold is 10,000 -- a knowledge base several times that projects to about
    0.85 at pgvector's default 40. 120 measured 0.973 at 12,000 rows for 2.96 ms,
    still 12x faster than the exact scan it replaces.
    """
    statements = _capture(partial_index=True)
    raised = _writes(statements, "hnsw.ef_search")
    assert raised, f"nothing raised hnsw.ef_search on an indexed search: {statements}"
    assert statements[raised[0]][1].get("ef") == str(PER_KB_HNSW_EF_SEARCH), statements[raised[0]]
    normalized = "".join(statements[raised[0]][0].split()).lower()
    assert "true" in normalized, (
        f"a session-level ef_search would follow the connection into the pool: {normalized}"
    )


def test_the_raised_ef_search_stays_inside_the_supported_band():
    """80-400, and the upper bound is a planner cliff rather than taste.

    pgvector's HNSW cost estimate scales with this setting, and past roughly
    600-800 a knowledge base's own partial index prices *above* the shared
    per-dimension index and the planner flips to the shared one -- reproduced at
    12,000 rows, where 600 kept the partial index at 6.9 ms and 800 took the
    shared index at 18.8 ms with lower recall. A value outside the band inverts
    the whole mechanism, silently.
    """
    assert 80 <= PER_KB_HNSW_EF_SEARCH <= 400, PER_KB_HNSW_EF_SEARCH


def test_ef_search_is_left_alone_without_an_index_to_be_accurate_on():
    """It decides accuracy once the planner is on an index, so with no index of
    this knowledge base's own there is nothing for it to decide -- and raising it
    would slow the shared index's post-filtered scan for no recall."""
    statements = _capture(partial_index=False)
    assert not _writes(statements, "hnsw.ef_search"), statements


def test_ef_search_is_restored_to_the_value_the_probe_read():
    """Not pgvector's default 40: whatever this transaction already had.

    ``hybrid_search`` and the reranker read on this same session afterwards.
    """
    statements = _capture(partial_index=True)
    touched = _writes(statements, "hnsw.ef_search")
    assert len(touched) == 2, f"expected one set and one restore: {statements}"
    assert statements[touched[1]][1].get("prior") == EF_SEARCH_WAS, statements[touched[1]]
    search_at = next(i for i, (sql, _) in enumerate(statements) if "ORDER BY" in sql)
    assert touched[0] < search_at < touched[1], statements


def test_ef_search_is_read_in_a_way_that_survives_a_database_without_pgvector():
    """It is pgvector's GUC, not PostgreSQL's.

    A plain ``current_setting`` on an unknown name raises, which would fail the
    whole probe -- and with it the gate -- on any database where the extension is
    not loaded. ``missing_ok`` makes that answer NULL instead.
    """
    probe = BasePgVectorStore._PARTIAL_INDEX_PROBE
    normalized = "".join(probe.split()).lower()
    assert "current_setting('hnsw.ef_search',true)" in normalized, probe


def test_a_database_without_pgvector_sets_no_ef_search():
    """The NULL above must not become a ``set_config`` of the string "None"."""
    session = MagicMock()
    captured: list[tuple[str, dict]] = []

    def spy_execute(text_obj, params=None):
        sql = text_obj.text if hasattr(text_obj, "text") else str(text_obj)
        captured.append((sql, dict(params or {})))
        if "to_regclass" in sql:
            return iter([(ENABLE_SORT_WAS, None, True)])
        if "current_setting('plan_cache_mode')" in sql:
            return _scalar(PLAN_CACHE_MODE_WAS)
        return iter([])

    session.execute = spy_execute
    store = _FakeStore(db_session=session, knowledge_base_id=_KB_ID)
    asyncio.run(store.vector_search(embedding=[0.0] * 1536, top_k=10))
    assert not _writes(captured, "hnsw.ef_search"), captured
    assert _enable_sort(captured), (
        f"the rest of the gate still applies without pgvector's GUC: {captured}"
    )


# ---------------------------------------------------------------------------
# The probe's own failure, which must not take the caller's transaction with it
# ---------------------------------------------------------------------------


def test_the_probe_runs_in_a_savepoint():
    """A probe that errors must not leave the caller on an aborted transaction.

    The handler logs "may fall back to an exact scan" and swallows. Without a
    savepoint that sentence is the opposite of what happens: the search that
    follows fails with ``InFailedSqlTransaction`` and the caller gets an error
    about a statement it never wrote.
    """
    session = MagicMock()
    nested = MagicMock()
    session.begin_nested.return_value = nested
    captured: list[str] = []

    def spy_execute(text_obj, params=None):
        sql = text_obj.text if hasattr(text_obj, "text") else str(text_obj)
        captured.append(sql)
        if "to_regclass" in sql:
            raise RuntimeError("catalog read cancelled")
        return iter([])

    session.execute = spy_execute
    store = _FakeStore(db_session=session, knowledge_base_id=_KB_ID)
    items = asyncio.run(store.vector_search(embedding=[0.0] * 1536, top_k=10))
    assert items == []
    assert session.begin_nested.called, (
        "the probe must run in a savepoint, or its failure aborts the caller's "
        f"transaction: {captured}"
    )
    assert any("ORDER BY" in sql for sql in captured), (
        f"the search must still run after a failed probe: {captured}"
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


# ---------------------------------------------------------------------------
# The diversity-floor path, which deliberately keeps the planner's own plan
# ---------------------------------------------------------------------------


def test_the_per_source_search_is_left_on_the_planners_own_plan():
    """Not an inconsistency with ``vector_search``, and measured rather than argued.

    ``vector_search_per_source`` carries the same knowledge base literal and the
    same iterative-scan mode, so the gate's absence reads like an omission. It is
    not: the query has no ``LIMIT`` on the distance order -- it scores the whole
    knowledge base by design -- so there is no ordered-index-scan-against-sort
    race to win. Measured at 1536 dimensions with the partial index built and
    valid: no HNSW index in any configuration, six sort nodes no setting can
    remove, and ``enable_sort = off`` made it 4.8x slower (50.5 -> 244.7 ms).

    A ``LIMIT`` on the distance order would make this ``vector_search``'s shape,
    and this spec is where that change gets noticed.
    """
    session = MagicMock()
    captured: list[tuple[str, dict]] = []

    def spy_execute(text_obj, params=None):
        sql = text_obj.text if hasattr(text_obj, "text") else str(text_obj)
        captured.append((sql, dict(params or {})))
        if "to_regclass" in sql:
            return iter([(ENABLE_SORT_WAS, EF_SEARCH_WAS, True)])
        if "current_setting('plan_cache_mode')" in sql:
            return _scalar(PLAN_CACHE_MODE_WAS)
        return iter([])

    session.execute = spy_execute
    store = _FakeStore(db_session=session, knowledge_base_id=_KB_ID)
    asyncio.run(
        store.vector_search_per_source(embedding=[0.0] * 1536, per_source_k=3, source_cap=5)
    )
    scored = [sql for sql, _ in captured if "ROW_NUMBER" in sql]
    assert scored, f"the per-source search did not run: {captured}"
    assert not _enable_sort(captured), (
        f"pricing the sort out of a query built on six sorts is a regression: {captured}"
    )
    assert not _writes(captured, "hnsw.ef_search"), captured
    assert not _writes(captured, "plan_cache_mode"), captured
    assert not [sql for sql, _ in captured if "to_regclass" in sql], (
        f"no gate here means no catalog round trip to pay for either: {captured}"
    )
