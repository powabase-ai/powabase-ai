"""A knowledge base too small for an index of its own is searched exactly.

Below ``VECTOR_PER_KB_INDEX_MIN_ROWS`` a knowledge base has no partial HNSW index,
and its unrestricted search used to be left to the planner -- which, on a table
where other knowledge bases hold most of the rows, walks the *shared* index over
every knowledge base and filters afterwards. Measured on a production project with
a 4.97M-row embeddings table: 300-600 ms cold for a knowledge base of a few
thousand rows, and **0 of 20** requested rows for a 24,118-row one whose
neighbours lay in other knowledge bases. An exact search reads only the knowledge
base's own rows, so its cost does not depend on anyone else's data.

These specs pin the decision and the statements it produces. Whether the planner
really answers them with a lookup on the knowledge-base btree and a sort -- and
never the HNSW index or a whole-table ``Seq Scan`` -- is what
``tests/pg_search/test_exact_search_below_floor_live.py`` shows on a real server.
"""

from __future__ import annotations

import asyncio
import logging
import re

import pytest

from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services import pg_vector_index as pvi
from agentic_project_service.services.base_vector_store import (
    BasePgVectorStore,
    takes_the_exact_path,
)
from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY

_KB_ID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
_DIMS = 1536
SETTING = "VECTOR_EXACT_SEARCH_MAX_ROWS"
# Deliberately not the default: a store that hardcoded 5000 instead of reading
# the setting would pass every spec that used the default.
_CAP = 4_321
# What the fake reports for the settings the store reads back and restores. Not
# the server defaults, so a restore that hardcodes "on" is caught.
_SEQSCAN_WAS = "maybe"
_INDEXSCAN_WAS = "perhaps"


class _ChunkStore(BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


class _DocumentStore(BasePgVectorStore):
    TABLE = "full_documents"
    TEXT_COL = "full_text_path"
    SEARCH_TEXT_COL = "summary"


class _GraphNodeStore(BasePgVectorStore):
    TABLE = "graph_index_nodes"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


class _Doc2JsonStore(BasePgVectorStore):
    TABLE = "doc2json_documents"
    TEXT_COL = "summary"
    SEARCH_TEXT_COL = "summary"


class _Result:
    """Just enough of a SQLAlchemy result for the calls the store makes."""

    def __init__(self, rows):
        self._rows = list(rows)

    def __iter__(self):
        return iter(self._rows)

    def all(self):
        return list(self._rows)

    def fetchall(self):
        return list(self._rows)

    def scalar(self):
        return self._rows[0][0] if self._rows else None


class _Savepoint:
    def __init__(self, session: _Session):
        self._session = session

    def __enter__(self):
        if self._session.aborted:
            raise RuntimeError("current transaction is aborted; SAVEPOINT refused")
        self._session.levels.append(dict(self._session.settings))
        return self

    def __exit__(self, exc_type, exc, tb):
        before = self._session.levels.pop()
        if exc_type is not None:
            # ROLLBACK TO SAVEPOINT: this level's settings are undone and the
            # transaction is usable again.
            self._session.settings = before
            self._session.aborted = False
        return False


class _Session:
    """A session that answers the store's statements and models PostgreSQL's abort.

    ``own_index`` is what the catalog says about this knowledge base's partial
    index at this dimension, ``kb_rows`` how many embeddings it holds for the
    store's item table -- the count answers ``min(kb_rows, LIMIT)`` the way the real
    one does. ``fail_at`` names a fragment of one statement to fail server-side;
    outside a savepoint that leaves the transaction aborted and every later
    statement refused, which is the consequence a ``MagicMock`` cannot express.
    """

    def __init__(self, *, own_index: bool = False, kb_rows: int = 300, fail_at: str | None = None):
        self.own_index = own_index
        self.kb_rows = kb_rows
        self.fail_at = fail_at
        self.statements: list[tuple[str, dict]] = []
        self.settings: dict[str, str] = {}
        self.settings_at_the_search: dict[str, str] | None = None
        self.levels: list[dict[str, str]] = []
        self.aborted = False

    def begin_nested(self):
        return _Savepoint(self)

    def execute(self, clause, params=None):
        sql = clause.text if hasattr(clause, "text") else str(clause)
        params = dict(params or {})
        self.statements.append((sql, params))
        flat = "".join(sql.split())
        if self.aborted:
            raise RuntimeError(f"current transaction is aborted; refused: {flat[:60]}")
        if self.fail_at and self.fail_at in flat:
            self.aborted = True
            raise RuntimeError(f"injected server-side failure at {flat[:60]}")
        match = re.search(r"set_config\('([^']+)',(:?\w+|'[^']*')", flat)
        if match:
            guc, raw = match.group(1), match.group(2)
            self.settings[guc] = params[raw[1:]] if raw.startswith(":") else raw.strip("'")
            return _Result([(self.settings[guc],)])
        if "FROMpg_class" in flat:
            name = pvi.per_kb_index_name(_KB_ID, _DIMS)
            return _Result([(name, True, None)] if self.own_index else [])
        if "count(*)" in flat:
            limit = int(re.search(r"LIMIT(\d+)", flat).group(1))
            return _Result([(min(self.kb_rows, limit),)])
        if "current_setting('enable_seqscan')" in flat:
            return _Result([(_SEQSCAN_WAS,)])
        if "current_setting('enable_indexscan')" in flat:
            return _Result([(_INDEXSCAN_WAS,)])
        if "to_regclass" in flat:
            return _Result([("on", "40", self.own_index)])
        if "ORDER BY" in sql:
            self.settings_at_the_search = dict(self.settings)
        return _Result([])


@pytest.fixture(autouse=True)
def _setting(monkeypatch):
    values = {SETTING: _CAP}
    monkeypatch.setattr(bvs, "get_setting", lambda key: values[key])
    return values


def _run(store=_ChunkStore, **kwargs) -> _Session:
    session_kwargs = {
        k: kwargs.pop(k) for k in ("own_index", "kb_rows", "fail_at") if k in kwargs
    }
    session = _Session(**session_kwargs)
    kwargs.setdefault("embedding", [0.0] * _DIMS)
    kwargs.setdefault("top_k", 10)
    items = asyncio.run(store(db_session=session, knowledge_base_id=_KB_ID).vector_search(**kwargs))
    assert items == []
    return session


def _search_sql(session: _Session) -> str:
    searches = [sql for sql, _ in session.statements if "ORDER BY" in sql]
    assert len(searches) == 1, f"expected exactly one search: {session.statements}"
    return searches[0]


def _flat(sql: str) -> str:
    return " ".join(sql.split())


def _is_exact_shape(sql: str) -> bool:
    return "OFFSET 0" in _flat(sql)


def _counts(session: _Session) -> list[str]:
    return [sql for sql, _ in session.statements if "count(*)" in sql]


def _catalog_probes(session: _Session) -> list[str]:
    return [sql for sql, _ in session.statements if "pg_class" in sql]


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_rows", "has_its_own_index", "rows", "exact"),
    [
        (5000, True, 300, False),  # its own index answers; the HNSW path is kept
        (5000, False, 300, True),  # small and unindexed: exact
        (5000, False, 5000, True),  # the cap is inclusive
        (5000, False, 5001, False),  # over the cap: today's behaviour
        (5000, False, None, False),  # the count could not be read: today's behaviour
        (0, False, 0, False),  # 0 turns the feature off
        (0, False, 300, False),
    ],
)
def test_the_decision(max_rows, has_its_own_index, rows, exact):
    assert (
        takes_the_exact_path(max_rows=max_rows, has_its_own_index=has_its_own_index, rows=rows)
        is exact
    )


# ---------------------------------------------------------------------------
# The branches, through ``vector_search``
# ---------------------------------------------------------------------------


def test_a_small_unindexed_knowledge_base_is_searched_exactly():
    session = _run(own_index=False, kb_rows=300)
    sql = _search_sql(session)
    assert _is_exact_shape(sql), f"expected the fenced exact shape:\n{sql}"
    # Nothing steers it at an index: no sort penalty, no raised ef_search.
    assert "enable_sort" not in session.settings_at_the_search, session.statements
    assert "hnsw.ef_search" not in session.settings_at_the_search, session.statements


def test_a_knowledge_base_with_its_own_index_keeps_the_hnsw_path():
    """Indexed wins, whatever its size -- even under the cap, where the two
    settings are inconsistent (an index built at the build threshold and kept down
    to the drop threshold). It also never pays for the count."""
    session = _run(own_index=True, kb_rows=300)
    sql = _search_sql(session)
    assert not _is_exact_shape(sql), sql
    assert session.settings_at_the_search.get("enable_sort") == "off", session.statements
    assert not _counts(session), "an indexed knowledge base must not pay for the count"


def test_an_unindexed_knowledge_base_over_the_cap_keeps_todays_behaviour():
    session = _run(own_index=False, kb_rows=_CAP + 1)
    sql = _search_sql(session)
    assert not _is_exact_shape(sql), sql
    # The whole-table scan penalty the decision ran under is put back before the
    # search the planner is left to plan on its own.
    assert session.settings_at_the_search.get("enable_seqscan") == _SEQSCAN_WAS, (
        f"enable_seqscan leaked into today's plan: {session.statements}"
    )
    assert "enable_sort" not in session.settings_at_the_search


def test_the_setting_at_zero_turns_it_off(_setting):
    _setting[SETTING] = 0
    session = _run(own_index=False, kb_rows=1)
    assert not _is_exact_shape(_search_sql(session))
    assert not _counts(session), "off must cost nothing"
    assert not _catalog_probes(session), "off must cost nothing"
    assert "enable_seqscan" not in "".join(sql for sql, _ in session.statements)


def test_the_cap_is_the_setting_and_the_count_stops_one_past_it():
    session = _run(own_index=False, kb_rows=10**7)
    (count,) = _counts(session)
    assert f"LIMIT {_CAP + 1}" in _flat(count), (
        f"the count must stop at the cap + 1, or it walks a big knowledge base:\n{count}"
    )


def test_the_count_names_the_population_with_literals():
    """The same three literals the search carries, and for the same reason: a
    generic plan on a bound knowledge base id has no estimate but 1/n_distinct."""
    session = _run(own_index=False, kb_rows=300)
    (count,) = _counts(session)
    flat = _flat(count)
    assert f"knowledge_base_id = '{_KB_ID}'" in flat, flat
    assert "item_table = 'chunks'" in flat, flat
    assert f"dims = {_DIMS}" in flat, flat
    assert ":" not in flat.replace("::", ""), f"nothing in the count may be bound:\n{flat}"


def test_the_exact_search_carries_every_value_as_a_literal():
    session = _run(own_index=False, kb_rows=300, top_k=7)
    flat = _flat(_search_sql(session))
    assert flat.count(f"knowledge_base_id = '{_KB_ID}'") == 2, flat
    assert "item_table = 'chunks'" in flat, flat
    assert f"dims = {_DIMS}" in flat, flat
    assert flat.rstrip().endswith("LIMIT 7"), flat
    assert f"<=> CAST(:embedding AS vector({_DIMS}))" in flat, flat


def test_the_fence_is_around_the_embeddings_and_the_order_is_outside_it():
    """``OFFSET 0`` keeps the subquery from being pulled up, so the ``ORDER BY``
    that an HNSW scan would serve is never on the embeddings relation itself.
    That is what makes the shape exact *by construction*: no setting has to
    succeed for the index to be out of reach."""
    flat = _flat(_search_sql(_run(own_index=False, kb_rows=300)))
    fence = re.search(r"FROM \( ?(SELECT .*? OFFSET 0) ?\) e", flat)
    assert fence, flat
    assert "ORDER BY" not in fence.group(1), flat
    assert "LIMIT" not in fence.group(1), flat
    outer = flat[fence.end() :]
    assert "ORDER BY (e.embedding::vector(1536)) <=>" in outer, flat


def test_the_whole_table_scan_is_priced_out_for_the_count_and_the_search_and_put_back():
    session = _run(own_index=False, kb_rows=300)
    statements = [sql for sql, _ in session.statements]
    sets = [i for i, sql in enumerate(statements) if "set_config('enable_seqscan'" in sql]
    count_at = next(i for i, sql in enumerate(statements) if "count(*)" in sql)
    search_at = next(i for i, sql in enumerate(statements) if "ORDER BY" in sql)
    assert len(sets) == 2, statements
    assert sets[0] < count_at < search_at < sets[1], statements
    assert session.settings_at_the_search["enable_seqscan"] == "off"
    # Only the sequential scan: pricing out index scans *and* bitmap scans is the
    # combination that leaves a parallel scan of the whole heap as the only plan.
    assert "enable_bitmapscan" not in "".join(statements), statements
    assert "enable_indexscan" not in session.settings_at_the_search
    # Put back to what was read, not to a hardcoded default.
    assert session.settings["enable_seqscan"] == _SEQSCAN_WAS


def test_the_setting_is_transaction_scoped():
    session = _run(own_index=False, kb_rows=300)
    sets = [
        "".join(sql.split()) for sql, _ in session.statements if "set_config('enable_seqscan'" in sql
    ]
    assert sets, session.statements
    assert all(s.endswith(",true)") for s in sets), (
        f"without the third argument the penalty would ride the connection into the pool: {sets}"
    )


@pytest.mark.parametrize("store", [_DocumentStore, _GraphNodeStore, _Doc2JsonStore])
def test_the_other_item_tables_are_always_unindexed_and_skip_the_catalog(store):
    """The per-KB index covers ``item_table = 'chunks'`` only, so for these stores
    the probe could only answer "no" -- or, worse, "yes" about an index that holds
    none of their rows."""
    session = _run(store=store, own_index=True, kb_rows=40)
    assert not _catalog_probes(session), session.statements
    (count,) = _counts(session)
    assert f"item_table = '{store.TABLE}'" in _flat(count), count
    sql = _search_sql(session)
    assert _is_exact_shape(sql), sql
    assert f"c.{store.TEXT_COL}" in sql, sql
    assert f'"ai".{store.TABLE} c' in sql, sql


@pytest.mark.parametrize("store", [_DocumentStore, _GraphNodeStore, _Doc2JsonStore])
def test_the_other_item_tables_over_the_cap_keep_the_planners_plan(store):
    session = _run(store=store, kb_rows=_CAP + 1)
    assert not _is_exact_shape(_search_sql(session))
    assert "enable_sort" not in session.settings_at_the_search


@pytest.mark.parametrize(
    "restriction",
    [
        {"source_ids": ["3f2504e0-4f89-11d3-9a0c-0305e82c3303"]},
        {"item_ids": {"3f2504e0-4f89-11d3-9a0c-0305e82c3304"}},
        {"filter_metadata": {"tier": "gold"}},
    ],
    ids=["source_ids", "item_ids", "filter_metadata"],
)
def test_a_restricted_search_keeps_its_own_exact_path_and_pays_for_nothing_new(restriction):
    """Already exact through ``_insisting_on_an_exact_search``; composing means not
    paying for a second decision whose answer cannot change anything."""
    session = _run(own_index=False, kb_rows=300, **restriction)
    assert not _counts(session), session.statements
    assert not _catalog_probes(session), session.statements
    assert not _is_exact_shape(_search_sql(session))
    assert session.settings_at_the_search.get("enable_indexscan") == "off"


# ---------------------------------------------------------------------------
# A failure degrades the plan, never the answer or the transaction
# ---------------------------------------------------------------------------


def test_a_catalog_probe_that_fails_keeps_todays_path_on_a_usable_transaction():
    session = _run(own_index=False, kb_rows=300, fail_at="FROMpg_class")
    assert not _is_exact_shape(_search_sql(session))
    assert not _counts(session), session.statements


def test_a_count_that_fails_keeps_todays_path_on_a_usable_transaction():
    session = _run(own_index=False, kb_rows=300, fail_at="count(*)")
    assert not _is_exact_shape(_search_sql(session))
    assert session.settings_at_the_search.get("enable_seqscan") == _SEQSCAN_WAS


def test_a_scan_penalty_that_cannot_be_set_still_searches_exactly():
    """The fence is what makes the search exact; the penalty only bounds its cost.
    So losing the penalty loses the bound on a table where the knowledge base is a
    large share, and nothing else."""
    session = _run(own_index=False, kb_rows=300, fail_at="set_config('enable_seqscan','off'")
    assert _is_exact_shape(_search_sql(session))
    assert "enable_seqscan" not in session.settings_at_the_search


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_one_debug_line_per_exact_search_and_nothing_louder(caplog):
    caplog.set_level(logging.DEBUG, logger=bvs.logger.name)
    _run(own_index=False, kb_rows=300)
    exact = [r for r in caplog.records if "exact" in r.getMessage().lower()]
    assert len(exact) == 1, [r.getMessage() for r in caplog.records]
    assert exact[0].levelno == logging.DEBUG
    assert not [r for r in caplog.records if r.levelno >= logging.INFO], [
        r.getMessage() for r in caplog.records
    ]


def test_todays_path_logs_nothing_new(caplog):
    caplog.set_level(logging.DEBUG, logger=bvs.logger.name)
    _run(own_index=False, kb_rows=_CAP + 1)
    assert not [r for r in caplog.records if "exact" in r.getMessage().lower()]


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------


def test_the_setting_is_registered_with_the_decided_default_and_range():
    defn = SETTINGS_REGISTRY[SETTING]
    assert defn.type == "int"
    assert defn.default == 5_000
    assert defn.min == 0
    assert defn.max == 50_000
    assert defn.advanced is True
    assert defn.category == SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"].category


def test_the_setting_describes_how_it_meets_the_per_kb_thresholds():
    description = SETTINGS_REGISTRY[SETTING].description
    assert "VECTOR_PER_KB_INDEX_MIN_ROWS" in description
    assert "VECTOR_PER_KB_INDEX_DROP_ROWS" in description
    assert "0" in description


def test_the_cap_never_exceeds_the_default_build_threshold():
    """Above the build threshold the per-KB index is the designed answer; an exact
    search there costs hundreds of milliseconds (360-450 ms warm at 73,288 rows)."""
    assert SETTINGS_REGISTRY[SETTING].max <= SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"].default


@pytest.mark.parametrize(("stored", "used"), [(90_000, 50_000), (-3, 0), (1234, 1234)])
def test_a_stored_value_is_clamped_to_the_registry_range(_setting, stored, used):
    _setting[SETTING] = stored
    assert bvs.exact_search_max_rows() == used
