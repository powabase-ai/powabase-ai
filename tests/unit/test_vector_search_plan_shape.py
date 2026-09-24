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
import re
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text

from agentic_project_service.services.base_vector_store import (
    PER_KB_HNSW_EF_SEARCH,
    BasePgVectorStore,
    item_table_sql_literal,
)

_KB_ID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
_SOURCE_ID = "3f2504e0-4f89-11d3-9a0c-0305e82c3303"

# Deliberately not the defaults ("on" and "40"): a restore that hardcodes what
# the setting usually is would pass against a fixture whose prior value *was* that.
ENABLE_SORT_WAS = "off"
EF_SEARCH_WAS = "55"
ENABLE_INDEXSCAN_WAS = "off"


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
    *,
    partial_index: bool = False,
    store: type[BasePgVectorStore] = _FakeStore,
    ef_search_was: str | None = EF_SEARCH_WAS,
    **kwargs,
) -> list[tuple[str, dict]]:
    """Every statement ``vector_search`` executes, with the parameters it binds.

    ``partial_index`` is the answer the store's catalog probe gets back: whether
    this knowledge base has a valid partial HNSW index. It decides a branch, so
    the fake has to be able to answer both ways. ``ENABLE_SORT_WAS`` and
    ``EF_SEARCH_WAS`` are the values the probe reports those settings currently
    have, so a spec can tell a restore from a hardcoded default.

    ``ef_search_was`` is ``None`` for the case a real fresh connection is always
    in: pgvector registers ``hnsw.ef_search`` on the first *use of the vector
    type*, so until then the probe's ``current_setting(..., true)`` answers NULL.
    Hardcoding a value here left that branch -- the only one a pooled
    connection's first search takes -- unexercised.
    """
    session = MagicMock()
    captured: list[tuple[str, dict]] = []

    def spy_execute(text_obj, params=None):
        sql = text_obj.text if hasattr(text_obj, "text") else str(text_obj)
        captured.append((sql, dict(params or {})))
        if "to_regclass" in sql:
            return iter([(ENABLE_SORT_WAS, ef_search_was, partial_index)])
        if "current_setting('enable_indexscan')" in sql:
            return _scalar(ENABLE_INDEXSCAN_WAS)
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


@pytest.mark.parametrize(
    ("store", "item_table"),
    [(_FakeStore, "chunks"), (_FakeDocumentStore, "full_documents")],
    ids=["chunks", "documents"],
)
def test_the_item_table_reaches_sql_as_a_literal_too(store, item_table):
    """``ai.embeddings`` is polymorphic, so the index is one population by predicate.

    A knowledge base crosses the build threshold on the *sum* over its item tables.
    An index that mixes populations is then walked for entries that cannot join:
    measured on the same 1,000 chunk rows, chunks alone against chunks plus 9,000
    document rows, recall 0.858 -> 0.383; and on 6,000 chunk rows with and without
    6,000 graph-node rows, 0.925 -> 0.812 with the worst query at 0.700 -> 0.300.

    The index predicate names the item table, so the query has to name it as a
    *literal* or a generic plan cannot prove the predicate and the index is lost --
    confirmed by EXPLAIN ANALYZE, which drops to the shared per-dimension index
    (12.2 ms against 1.45 ms) as soon as the clause is bound or absent.

    The store's own table, not a hardcoded ``chunks``: an embedding's ``item_table``
    is the table its item lives in, so this is the semantically correct restriction
    for every store that inherits the search, and the one that keeps a document
    store off an index built for chunks.
    """
    sql, params = _search(_capture(store=store))
    normalized = "".join(sql.split())
    assert f"e.item_table='{item_table}'" in normalized, (
        "the item table must be a literal on the embeddings side, or the index "
        f"predicate cannot be proved:\n{sql}"
    )
    assert "item_table" not in params, f"a bound item table proves nothing to the planner: {params}"


def test_an_item_table_that_could_break_the_quoting_is_rejected():
    """The one gate between a store's ``TABLE`` and a SQL string literal.

    It is a class attribute rather than caller data, and the same attribute is
    already interpolated as an identifier in the ``FROM`` clause -- but the check
    belongs where the quoting happens, not in a comment about why it is safe.
    """
    assert item_table_sql_literal("chunks") == "'chunks'"
    assert item_table_sql_literal("doc2json_documents") == "'doc2json_documents'"
    for bad in ["chunks'; DROP TABLE x --", "Chunks", "ai.chunks", "", "chunks chunks", None, 7]:
        with pytest.raises(ValueError):
            item_table_sql_literal(bad)


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
    """Not a hardcoded ``on``: whatever the transaction had before.

    The SQL text is asserted as well as the parameters, and that is not
    belt-and-braces: a restore that stops using the bind and hardcodes ``'on'``
    still *passes* a parameter assertion, because the bind stays in the dict the
    caller built. Two mutations hid there -- the hardcoded value, and a restore
    made session-scoped (``true`` -> ``false``), which leaks the setting into the
    pool. Both are visible only in the statement.
    """
    statements = _capture(partial_index=True)
    restore_at = _enable_sort(statements)[1]
    sql, params = statements[restore_at]
    normalized = "".join(sql.split()).lower()
    assert "set_config('enable_sort',:prior,true)" in normalized, (
        f"the restore must bind the value the probe read and stay transaction-local:\n{sql}"
    )
    assert params.get("prior") == ENABLE_SORT_WAS, (
        f"the restore must bind the value the probe read, not a guess: {sql} {params}"
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
@pytest.mark.parametrize("store", [_FakeStore, _FakeDocumentStore], ids=["chunks", "documents"])
def test_a_restricted_search_has_the_index_priced_out_not_merely_unforced(restriction, store):
    """Every restriction the search accepts, including the empty ones.

    Over every store that inherits this search, not just the chunks one. Making
    exactness conditional on the item table is a *silent wrong answer* for the
    document-level stores: a caller who names ``item_ids`` gets a full page from an
    approximate scan, and by this mechanism's own argument a full page has no
    signal in it to notice. The gate on the item table belongs on the half that
    steers *towards* the index, never on the half that makes an answer exact.

    Two propositions in one spec, because either alone passes a broken
    implementation: the sort penalty must not be applied, *and* the index must be
    priced out. Not forcing is not enough -- on a table shape where the planner
    takes the index for a restricted search unaided it returns a full page of
    ``top_k`` rows of which 12 to 17 of 20 were not the nearest matching ones, and
    a full page has no signal in it. Which shapes those are is measured in
    ``_insisting_on_an_exact_search``'s docstring and is a property of the table
    rather than of the vector width.

    An empty id set is the most starved restriction there is -- it matches nothing
    -- so keying on ``is not None`` rather than truthiness is load-bearing, and
    pinned here.
    """
    statements = _capture(partial_index=True, store=store, **restriction)
    priced_out = _writes(statements, "enable_indexscan")
    assert priced_out, (
        "a restricted search must have the approximate index priced out, not just "
        f"left to the planner: {statements}"
    )
    normalized = "".join(statements[priced_out[0]][0].split()).lower()
    assert "set_config('enable_indexscan','off',true)" in normalized, statements[priced_out[0]]
    assert not _enable_sort(statements), statements
    assert not _writes(statements, "hnsw.ef_search"), statements
    assert len([s for s, _ in statements if "ORDER BY" in s]) == 1, (
        f"a restricted search runs once; there is no re-run to repair it: {statements}"
    )


@pytest.mark.parametrize(
    "restriction",
    [
        {"item_ids": {"3f2504e0-4f89-11d3-9a0c-0305e82c3302"}},
        {"source_ids": ["3f2504e0-4f89-11d3-9a0c-0305e82c3303"]},
        {"filter_metadata": {"tier": "gold"}},
    ],
    ids=["item_ids", "source_ids", "filter_metadata"],
)
@pytest.mark.parametrize("store", [_FakeStore, _FakeDocumentStore], ids=["chunks", "documents"])
def test_the_exact_search_puts_enable_indexscan_back(restriction, store):
    """``hybrid_search`` runs its keyword leg on this same session afterwards, and
    a keyword ranking wants its index scans. Not a hardcoded ``on`` either:
    whatever the transaction had -- and still transaction-local, or the value
    follows the connection back into the pool.

    Asserted on the statement and not only on the bound parameters: a restore that
    stops using the bind leaves the parameter dict untouched, so a params-only
    assertion passes a hardcoded value and a session-scoped restore alike.
    """
    statements = _capture(partial_index=True, store=store, **restriction)
    touched = _writes(statements, "enable_indexscan")
    assert len(touched) == 2, f"expected one set and one restore: {statements}"
    sql, params = statements[touched[1]]
    normalized = "".join(sql.split()).lower()
    assert "set_config('enable_indexscan',:prior,true)" in normalized, (
        f"the restore must bind the value that was read and stay transaction-local:\n{sql}"
    )
    assert params.get("prior") == ENABLE_INDEXSCAN_WAS, statements[touched[1]]
    search_at = next(i for i, (sql, _) in enumerate(statements) if "ORDER BY" in sql)
    assert touched[0] < search_at < touched[1], statements


def test_a_restricted_search_does_not_pay_for_the_catalog_probe():
    """It does not matter whether the knowledge base has an index: the answer has
    to be exact either way. So there is nothing to ask the catalog."""
    statements = _capture(partial_index=True, source_ids=["3f2504e0-4f89-11d3-9a0c-0305e82c3303"])
    assert not [sql for sql, _ in statements if "to_regclass" in sql], statements


def test_an_unrestricted_search_is_not_made_exact():
    """The other half of the symmetry, and it must not be vacuous.

    Pricing the index out of *every* search would satisfy every spec above while
    throwing away the feature.
    """
    statements = _capture(partial_index=True)
    assert not _writes(statements, "enable_indexscan"), (
        f"an unrestricted search is the one that should use the index: {statements}"
    )
    assert _enable_sort(statements), statements


@pytest.mark.parametrize("empty", [None, {}], ids=["None", "empty-object"])
def test_an_empty_metadata_filter_is_not_a_restriction(empty):
    """``filter_metadata={}`` adds no clause, so it cannot starve anything.

    It must therefore be treated as the unrestricted search it is, and keep the
    index -- the decision keys on what narrows the search, not on which arguments
    were passed.
    """
    statements = _capture(partial_index=True, filter_metadata=empty)
    assert not _writes(statements, "enable_indexscan"), statements
    assert _enable_sort(statements), statements


def test_the_gate_is_only_for_the_item_table_it_was_measured_on():
    """``ai.embeddings`` is polymorphic and four stores inherit this search.

    The index is *named* after ``(knowledge_base_id, dims)`` and the probe can only
    look for a name, so a knowledge base whose search routes to a document-level
    store passes the probe just the same -- and would be driven into an HNSW walk
    of an index whose predicate restricts it to another store's rows: measured
    2.2 -> 25.9 ms and recall 1.00 -> 0.33 on a 40-row table.

    "The other stores keep the planner's own choice" holds for the *unrestricted*
    search only, which is the one this spec drives. A restricted search from those
    stores is priced off the index like any other, and that is pinned separately.
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
    sql, params = statements[touched[1]]
    normalized = "".join(sql.split()).lower()
    assert "set_config('hnsw.ef_search',:prior,true)" in normalized, (
        f"the restore must bind the value the probe read and stay transaction-local:\n{sql}"
    )
    assert params.get("prior") == EF_SEARCH_WAS, statements[touched[1]]
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


def test_ef_search_is_raised_even_when_the_probe_read_no_value_at_all():
    """The branch every pooled connection's *first* search takes.

    pgvector registers its GUCs in ``_PG_init``, which runs on the first use of
    the vector type -- not at ``CREATE EXTENSION``, not at connection start, and
    ``SET LOCAL hnsw.iterative_scan`` does not trigger it either (a dotted name is
    accepted as a placeholder). The probe runs before any vector operation, so on
    a fresh connection it reads NULL. Demonstrated on a live server: ``<NULL>``,
    then a single distance operation, then ``40``.

    Gating the raise on that read therefore left the first search of every
    connection at pgvector's default 40 rather than the value this mechanism
    measured -- recall 0.915 against 0.973 at 12,000 rows, and about 0.85 several
    times above the build threshold -- once per connection per pool lifetime, on
    the search most likely to be cold. So it is set unconditionally, which is safe:
    a ``set_config`` on an unloaded pgvector GUC creates a placeholder whose value
    survives ``_PG_init`` and still dies with the transaction.
    """
    statements = _capture(partial_index=True, ef_search_was=None)
    raised = _writes(statements, "hnsw.ef_search")
    assert raised, (
        "the probe reads NULL on a fresh connection, which is the common case, not "
        f"the exotic one -- ef_search must still be raised: {statements}"
    )
    sql, params = statements[raised[0]]
    assert params.get("ef") == str(PER_KB_HNSW_EF_SEARCH), statements[raised[0]]
    assert "None" not in str(params.values()), (
        f'the NULL must not become a set_config of the string "None": {params}'
    )
    normalized = "".join(sql.split()).lower()
    assert "set_config('hnsw.ef_search',:ef,true)" in normalized, sql
    search_at = next(i for i, (s, _) in enumerate(statements) if "ORDER BY" in s)
    assert raised[0] < search_at, (
        f"the setting has to be in place before the statement is planned: {statements}"
    )
    assert len(raised) == 1, (
        "with nothing read there is nothing to put back, and a transaction-local "
        f"placeholder dies with the transaction: {statements}"
    )
    assert _enable_sort(statements), (
        f"the rest of the gate applies on this branch too: {statements}"
    )


# ---------------------------------------------------------------------------
# A statement that fails must not take the caller's transaction with it
# ---------------------------------------------------------------------------


class _InjectedServerError(Exception):
    """What the server raises at the one statement a spec picked out."""


class _AbortedTransaction(Exception):
    """``InFailedSqlTransaction``: every statement after an unprotected failure."""


class _Savepoint:
    """What ``Session.begin_nested`` hands back: a savepoint as a context manager.

    On the way out with an exception it rolls back to the savepoint, which is the
    whole of what this models -- and then re-raises, the way SQLAlchemy's does.
    """

    def __init__(self, session: "_SavepointModellingSession"):
        self._session = session

    def __enter__(self):
        self._session._enter_savepoint()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._session._leave_savepoint(failed=exc_type is not None)
        return False


class _SavepointModellingSession:
    """A session that fails one *statement* and then behaves the way PostgreSQL does.

    Two things this has that a ``MagicMock`` does not, and both are why the
    assertion this replaces could not fail:

    * it picks its victim by reading the **statement**, not a bound parameter or a
      call count -- the shape that caught the mutation the ``item_table`` work was
      most afraid of (``_PopulationConn._restricted_to`` in ``tests/pg_search``),
      because a statement is what a dedent moves and a bind is not;
    * it models the consequence. A failure with no savepoint around it leaves the
      transaction **aborted**, so every later statement -- the search included --
      raises. A failure inside one is undone by the rollback the savepoint does on
      its way out, and the transaction is usable again. That is the difference
      between the two versions of this code, and a mock has no way to express it.

    It is a model, and it is one on purpose: the pinned proposition is "this
    statement is inside a savepoint", which is a property of the module and not of
    any server. The model's fidelity was checked against a real server rather than
    reasoned about -- a released savepoint keeps a transaction-local
    ``set_config`` in force, a rolled-back one undoes it and leaves the
    transaction usable, and without one the next statement is refused -- and the
    same defects were reproduced end to end by injecting a real server-side error
    at each statement in turn (a shadow ``set_config`` ahead of ``pg_catalog`` on
    the ``search_path``). The live tier is where that injection belongs; this is
    where a dedent gets caught in seconds.
    """

    def __init__(
        self,
        *,
        fail_at: str,
        occurrence: int = 1,
        partial_index: bool = True,
        ef_search_was: str | None = EF_SEARCH_WAS,
    ):
        self.fail_at = fail_at
        # Which execution of the matching statement fails. The sets and the
        # restores are the same statement text, so nothing but a count can tell a
        # spec about the restore apart from a spec about the set.
        self.occurrence = occurrence
        self._seen = 0
        self.partial_index = partial_index
        self.ef_search_was = ef_search_was
        self.statements: list[tuple[str, dict]] = []
        self.settings: dict[str, str | None] = {}
        # What was in force when the search itself was planned, which is the only
        # moment any of these settings matters. Read after the fact, a restore that
        # puts a setting back to the value it was set *to* is indistinguishable
        # from the set surviving -- and that is exactly the difference a shared
        # savepoint erases.
        self.settings_at_the_search: dict[str, str | None] = {}
        self.failed_at: list[str] = []
        self._levels: list[list[str]] = []
        self._aborted = False

    # -- the savepoint half -------------------------------------------------
    def begin_nested(self) -> _Savepoint:
        return _Savepoint(self)

    def _enter_savepoint(self) -> None:
        # ``SAVEPOINT`` is itself refused inside an aborted transaction, and the
        # transaction stays aborted -- confirmed on a live server, where it answers
        # "current transaction is aborted, commands ignored until end of
        # transaction block" like any other statement. Without this the model lets
        # a later savepoint *clear* an abort an earlier bare statement caused, and
        # a dedent of the restore that runs first then goes unnoticed.
        if self._aborted:
            raise _AbortedTransaction("current transaction is aborted; SAVEPOINT refused")
        self._levels.append([])

    def _leave_savepoint(self, *, failed: bool) -> None:
        written = self._levels.pop()
        if not failed:
            return
        # ROLLBACK TO SAVEPOINT: the subtransaction's writes are undone and the
        # transaction is usable again.
        for guc in written:
            self.settings.pop(guc, None)
        self._aborted = False

    # -- the statement half -------------------------------------------------
    def execute(self, clause, params=None):
        sql = clause.text if hasattr(clause, "text") else str(clause)
        self.statements.append((sql, dict(params or {})))
        flat = "".join(sql.split())
        if self._aborted:
            raise _AbortedTransaction(
                "current transaction is aborted, commands ignored until end of "
                f"transaction block; refused: {flat[:60]}"
            )
        if self.fail_at and self.fail_at in flat:
            self._seen += 1
            if self._seen == self.occurrence:
                self.failed_at.append(flat)
                self._aborted = True
                raise _InjectedServerError(f"injected server-side failure at {flat[:60]}")
        self._record_a_setting(flat, params or {})
        if "ORDER BY" in sql:
            self.settings_at_the_search = dict(self.settings)
        if "to_regclass" in sql:
            return iter([(ENABLE_SORT_WAS, self.ef_search_was, self.partial_index)])
        if "current_setting('enable_indexscan')" in sql:
            return _scalar(ENABLE_INDEXSCAN_WAS)
        return iter([])

    def _record_a_setting(self, flat: str, params: dict) -> None:
        match = re.search(r"set_config\('([^']+)',(:?\w+|'[^']*')", flat)
        if not match:
            return
        guc, raw = match.group(1), match.group(2)
        self.settings[guc] = params.get(raw[1:]) if raw.startswith(":") else raw.strip("'")
        if self._levels:
            self._levels[-1].append(guc)


def _run_a_search(session, **kwargs):
    """``vector_search`` against a modelling session; the statements it got through."""
    store = _FakeStore(db_session=session, knowledge_base_id=_KB_ID)
    kwargs.setdefault("embedding", [0.0] * 1536)
    kwargs.setdefault("top_k", 10)
    items = asyncio.run(store.vector_search(**kwargs))
    assert items == [], items
    return session.statements


def test_the_probe_runs_in_a_savepoint():
    """A probe that errors must not leave the caller on an aborted transaction.

    The handler logs "may fall back to an exact scan" and swallows. Without a
    savepoint that sentence is the opposite of what happens: the search that
    follows fails with ``InFailedSqlTransaction`` and the caller gets an error
    about a statement it never wrote.
    """
    session = _SavepointModellingSession(fail_at="to_regclass")
    statements = _run_a_search(session)
    assert session.failed_at, f"the injection missed the probe: {statements}"
    assert any("ORDER BY" in sql for sql, _ in statements), (
        f"the search must still run after a failed probe: {statements}"
    )
    assert not _writes(statements, "enable_sort"), (
        f"with no answer from the probe there is no index to steer at: {statements}"
    )


@pytest.mark.parametrize(
    ("fail_at", "still_applied"),
    [
        ("set_config('enable_sort'", None),
        ("set_config('hnsw.ef_search'", "enable_sort"),
    ],
)
def test_each_setting_the_partial_index_gate_makes_runs_in_a_savepoint(fail_at, still_applied):
    """Not only the probe: both of this block's own settings, which is where it bit.

    The probe was savepointed and the two ``set_config`` calls beside it were not,
    so a failure at either aborted the caller's transaction and the search raised
    ``InternalError`` -- while the handler logged that the search "may miss the
    knowledge base's partial HNSW index" or "will run at pgvector's default
    recall". Both sentences were false, and the block round 3 held up as the
    correct one was the block they were in.

    ``still_applied`` is the second half, and it is why the two statements take a
    savepoint each rather than sharing one: a rolled-back ``ef_search`` must leave
    ``enable_sort = off`` alone, or the handler's "runs at pgvector's default
    recall" turns into a search that is not on the index at all.
    """
    session = _SavepointModellingSession(fail_at=fail_at)
    statements = _run_a_search(session)
    assert session.failed_at, f"the injection missed its statement: {statements}"
    assert any("ORDER BY" in sql for sql, _ in statements), (
        f"a failed session setting must not make the search itself raise: {statements}"
    )
    if still_applied:
        assert session.settings_at_the_search.get(still_applied) == "off", (
            "one savepoint per statement: rolling this one back must not undo the "
            "setting that was already released, which has to still be in force when "
            f"the search is planned: {session.settings_at_the_search}"
        )


@pytest.mark.parametrize(
    "fail_at",
    ["current_setting('enable_indexscan')", "set_config('enable_indexscan'"],
)
def test_the_exact_searchs_own_setting_runs_in_a_savepoint(fail_at):
    """The twin of the specs above, on the other half of the symmetry.

    ``_insisting_on_an_exact_search`` reads a setting and writes one, and either
    statement can be cancelled like any other. Without a savepoint that failure
    leaves the caller's transaction aborted, so the search that follows raises
    ``InFailedSqlTransaction`` while the handler logs that it had merely degraded
    -- proved live both ways with a server-side failure injected at each
    statement, before and after.

    Parametrized over both statements because the version of this spec that only
    injected at the *read* was green with the write dedented back out of the
    savepoint: the failing statement has to be the one the spec picks out.
    """
    session = _SavepointModellingSession(fail_at=fail_at)
    statements = _run_a_search(session, source_ids=["3f2504e0-4f89-11d3-9a0c-0305e82c3303"])
    assert session.failed_at, f"the injection missed its statement: {statements}"
    assert any("ORDER BY" in sql for sql, _ in statements), (
        f"the search must still run after the setting could not be applied: {statements}"
    )


@pytest.mark.parametrize(
    ("fail_at", "restriction"),
    [
        ("set_config('enable_indexscan'", {"source_ids": [_SOURCE_ID]}),
        ("set_config('enable_sort'", {}),
        ("set_config('hnsw.ef_search'", {}),
    ],
)
def test_a_restore_that_fails_does_not_hand_back_a_broken_transaction(fail_at, restriction):
    """The restores are the other three of the eight statements, and they matter more.

    A restore runs after the rows are off the cursor, so a bare failure there gives
    the caller a search that *answered* and a transaction that no longer works --
    and the error surfaces on whatever the request does next, which is a failure
    attributed to the wrong statement. Measured with a server-side failure injected
    at each restore on its own: the search returned its rows and the caller's
    transaction was aborted.

    ``occurrence=2`` because a restore is the same statement text as the set it puts
    back, so nothing but a count can tell a spec about one from a spec about the
    other.

    This costs nothing in the case a restore usually fails in -- a transaction the
    search itself aborted -- because there the savepoint cannot be taken either and
    the handler logs what it logged before.
    """
    session = _SavepointModellingSession(fail_at=fail_at, occurrence=2)
    statements = _run_a_search(session, **restriction)
    assert session.failed_at, f"the injection missed the restore: {statements}"
    session.execute(text("SELECT 1"))


def test_a_setting_that_could_not_be_applied_is_not_restored_either():
    """Nothing was changed, so there is nothing to put back.

    A restore on this path would be a second statement inside a transaction the
    savepoint has just rolled back to -- and it would bind a value read from a
    statement that raised.
    """
    session = MagicMock()
    captured: list[tuple[str, dict]] = []

    def spy_execute(text_obj, params=None):
        sql = text_obj.text if hasattr(text_obj, "text") else str(text_obj)
        captured.append((sql, dict(params or {})))
        if "set_config('enable_indexscan'" in sql:
            raise RuntimeError("cancelled")
        if "current_setting('enable_indexscan')" in sql:
            return _scalar(ENABLE_INDEXSCAN_WAS)
        return iter([])

    session.execute = spy_execute
    store = _FakeStore(db_session=session, knowledge_base_id=_KB_ID)
    asyncio.run(
        store.vector_search(
            embedding=[0.0] * 1536,
            top_k=10,
            item_ids=["3f2504e0-4f89-11d3-9a0c-0305e82c3302"],
        )
    )
    assert len(_writes(captured, "enable_indexscan")) == 1, (
        f"only the attempt that failed; no restore of a setting never made: {captured}"
    )
    assert any("ORDER BY" in sql for sql, _ in captured), captured


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
    and this spec is where that change gets noticed. It would not be enough: the
    query does not select ``e.item_table``, and the partial index's predicate names
    it, so a statement that cannot prove that clause matches no partial index
    however it is planned. The ``LIMIT``, the literal and the gate are one change.
    """
    session = MagicMock()
    captured: list[tuple[str, dict]] = []

    def spy_execute(text_obj, params=None):
        sql = text_obj.text if hasattr(text_obj, "text") else str(text_obj)
        captured.append((sql, dict(params or {})))
        if "to_regclass" in sql:
            return iter([(ENABLE_SORT_WAS, EF_SEARCH_WAS, True)])
        if "current_setting('enable_indexscan')" in sql:
            return _scalar(ENABLE_INDEXSCAN_WAS)
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
