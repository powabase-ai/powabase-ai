"""Unit specs for the pg_search-backed per-KB BM25 index helpers.

Everything here is pure or runs against a spy session: index naming, the
ts_language -> pg_search stemmer mapping, the emitted DDL, tokenizer-change
detection, query normalisation and the index state machine.
"""

from __future__ import annotations

import re
import uuid
from unittest.mock import MagicMock

import pytest

from agentic_project_service.services import pg_bm25_index as pgb

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
KB2 = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"


# ---------------------------------------------------------------------------
# Index naming
# ---------------------------------------------------------------------------


def test_index_name_is_deterministic_and_kb_scoped():
    assert pgb.bm25_index_name(KB, "chunks") == pgb.bm25_index_name(KB, "chunks")
    assert pgb.bm25_index_name(KB, "chunks") != pgb.bm25_index_name(KB2, "chunks")
    assert pgb.bm25_index_name(KB, "chunks") != pgb.bm25_index_name(KB, "full_documents")


def test_index_name_has_no_dashes_and_starts_with_bm25():
    name = pgb.bm25_index_name(KB, "chunks")
    assert name == "bm25_chunks_" + uuid.UUID(KB).hex
    assert "-" not in name


def test_index_name_fits_the_postgres_identifier_limit_for_every_table():
    for item_table in pgb.BM25_ITEM_TABLES:
        name = pgb.bm25_index_name(KB, item_table)
        assert len(name.encode("utf-8")) <= 63, (item_table, name)


def test_index_name_rejects_a_non_uuid_kb_id():
    with pytest.raises(ValueError):
        pgb.bm25_index_name("kb'; DROP TABLE ai.chunks; --", "chunks")


def test_index_name_rejects_an_unknown_item_table():
    with pytest.raises(ValueError):
        pgb.bm25_index_name(KB, "not_an_item_table")


# ---------------------------------------------------------------------------
# Strategy -> item table
# ---------------------------------------------------------------------------


def test_strategy_map_sends_full_document_to_full_documents():
    from agentic_project_service.services.sparse_retrieval import STRATEGY_TO_BM25_ITEM_TABLE

    assert STRATEGY_TO_BM25_ITEM_TABLE["full_document"] == "full_documents"


def test_page_index_has_no_bm25_item_table():
    """page_index is tree_search-only, so it never gets a keyword index."""
    from agentic_project_service.services.sparse_retrieval import STRATEGY_TO_BM25_ITEM_TABLE

    assert "page_index" not in STRATEGY_TO_BM25_ITEM_TABLE
    assert pgb.pg_bm25_item_table("page_index") is None


@pytest.mark.parametrize(
    ("strategy", "item_table"),
    [
        ("chunk_embed", "chunks"),
        ("full_document", "full_documents"),
        ("graph_index", "graph_index_nodes"),
        ("doc2json", "doc2json_documents"),
    ],
)
def test_pg_bm25_item_table_per_strategy(strategy, item_table):
    assert pgb.pg_bm25_item_table(strategy) == item_table


# ---------------------------------------------------------------------------
# ts_language -> stemmer
# ---------------------------------------------------------------------------


def test_german_maps_to_the_german_stemmer():
    assert pgb.pg_search_stemmer("german") == "german"


@pytest.mark.parametrize("language", ["simple", "armenian", "basque", "catalan", "hindi"])
def test_languages_pg_search_cannot_stem_fall_back_to_no_stemmer(language):
    assert pgb.pg_search_stemmer(language) is None


def test_unknown_or_missing_language_falls_back_to_no_stemmer():
    assert pgb.pg_search_stemmer(None) is None
    assert pgb.pg_search_stemmer("klingon") is None
    assert pgb.pg_search_stemmer("german'); DROP TABLE ai.chunks; --") is None


def test_language_matching_is_case_insensitive():
    assert pgb.pg_search_stemmer("German") == "german"


def test_every_supported_stemmer_is_a_valid_ts_language():
    from agentic_project_service.services.base_vector_store import VALID_TS_LANGUAGES

    assert pgb.PG_SEARCH_STEMMERS <= VALID_TS_LANGUAGES


# ---------------------------------------------------------------------------
# Tokenizer cast + text expression
# ---------------------------------------------------------------------------


def test_tokenizer_cast_carries_the_stemmer():
    assert pgb.bm25_tokenizer_cast("chunks", "german") == "::pdb.simple('stemmer=german')"


def test_tokenizer_cast_without_a_stemmer_is_the_bare_tokenizer():
    assert pgb.bm25_tokenizer_cast("chunks", "hindi") == "::pdb.simple"


def test_expression_indexes_get_an_alias_argument():
    """pg_search refuses an indexed expression that has no alias= argument."""
    cast = pgb.bm25_tokenizer_cast("graph_index_nodes", "german")
    assert "alias=" in cast
    assert cast == "::pdb.simple('alias=bm25_text', 'stemmer=german')"


def test_expression_index_alias_survives_the_no_stemmer_case():
    assert (
        pgb.bm25_tokenizer_cast("graph_index_nodes", "hindi") == "::pdb.simple('alias=bm25_text')"
    )


@pytest.mark.parametrize(
    ("item_table", "expr"),
    [
        ("chunks", "text"),
        ("full_documents", "summary"),
        ("doc2json_documents", "summary"),
        ("graph_index_nodes", "(COALESCE(title, '') || ' ' || COALESCE(text, ''))"),
    ],
)
def test_text_expression_per_table(item_table, expr):
    assert pgb.bm25_text_expression(item_table) == expr


def test_text_expression_can_be_alias_qualified_for_a_query():
    assert pgb.bm25_text_expression("chunks", alias="c") == "c.text"
    assert pgb.bm25_text_expression("graph_index_nodes", alias="c") == (
        "(COALESCE(c.title, '') || ' ' || COALESCE(c.text, ''))"
    )


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------


def test_create_ddl_shape():
    ddl = pgb.bm25_index_ddl(KB, "chunks", "german")
    assert ddl == (
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
        f'bm25_chunks_{uuid.UUID(KB).hex} ON "ai".chunks_kb_{uuid.UUID(KB).hex} '
        "USING bm25 (id, (text::pdb.simple('stemmer=german')), source_id, meta) "
        "WITH (key_field = 'id')"
    )


def test_create_ddl_scopes_the_kb_by_relation_not_by_predicate():
    """One index per relation, so the knowledge base *is* the relation.

    The index carries no ``WHERE`` clause and no bind parameter: the partition
    bound is what restricts it to this KB's rows.
    """
    ddl = pgb.bm25_index_ddl(KB, "chunks", "german")
    assert "WHERE" not in ddl
    # No bind parameter -- the `::` in the tokenizer cast is the only colon.
    assert not re.search(r"(?<!:):[a-z_]", ddl)
    assert uuid.UUID(KB).hex in ddl


def test_create_ddl_is_concurrent_and_idempotent():
    ddl = pgb.bm25_index_ddl(KB, "full_documents", "english")
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" in ddl
    assert "summary::pdb.simple('stemmer=english')" in ddl


def test_drop_ddl_shape():
    assert pgb.bm25_drop_ddl(KB, "chunks") == (
        f'DROP INDEX CONCURRENTLY IF EXISTS "ai".bm25_chunks_{uuid.UUID(KB).hex}'
    )


def test_ddl_rejects_an_injected_kb_id():
    with pytest.raises(ValueError):
        pgb.bm25_index_ddl("' OR 1=1 --", "chunks", "german")


# ---------------------------------------------------------------------------
# Tokenizer-change detection (ts_language change => drop + recreate)
# ---------------------------------------------------------------------------


_GERMAN_DEF = (
    "CREATE INDEX bm25_chunks_x ON ai.chunks USING bm25 "
    "(id, ((text)::pdb.simple('stemmer=german')), source_id, meta) WITH (key_field=id)"
)
_PLAIN_DEF = (
    "CREATE INDEX bm25_chunks_x ON ai.chunks USING bm25 "
    "(id, ((text)::pdb.simple), source_id, meta) WITH (key_field=id)"
)


def test_indexdef_matches_the_same_tokenizer():
    assert pgb.indexdef_matches_tokenizer(_GERMAN_DEF, "::pdb.simple('stemmer=german')")


def test_indexdef_does_not_match_a_changed_stemmer():
    assert not pgb.indexdef_matches_tokenizer(_GERMAN_DEF, "::pdb.simple('stemmer=english')")


def test_bare_tokenizer_is_not_confused_with_a_stemmed_one():
    assert pgb.indexdef_matches_tokenizer(_PLAIN_DEF, "::pdb.simple")
    assert not pgb.indexdef_matches_tokenizer(_PLAIN_DEF, "::pdb.simple('stemmer=german')")
    assert not pgb.indexdef_matches_tokenizer(_GERMAN_DEF, "::pdb.simple")


# ---------------------------------------------------------------------------
# Query normalisation — user text reaches pg_search's match operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "Wanderung",
        "it's a quote",
        'a "quoted" phrase',
        "field:value",
        "back\\slash",
        "plus+minus-tilde~caret^",
        "(parens) [brackets] {braces}",
        "AND OR NOT",
        "/slashes/",
        "a" * 50_000,
        "",
        "   ",
        "😀 ünïcode",
    ],
)
def test_normalisation_never_raises_and_returns_a_string(raw):
    assert isinstance(pgb.normalize_bm25_query(raw), str)


def test_normalisation_strips_nul_bytes():
    """psycopg refuses a text parameter containing NUL, so it must never reach it."""
    out = pgb.normalize_bm25_query("Wander\x00ung")
    assert "\x00" not in out
    assert "Wanderung" in out.replace(" ", "")


def test_normalisation_collapses_control_characters_to_whitespace():
    assert pgb.normalize_bm25_query("Wanderung\n\tBerges") == "Wanderung Berges"


def test_normalisation_bounds_the_query_length():
    out = pgb.normalize_bm25_query("wort " * 100_000)
    assert len(out) <= pgb.MAX_BM25_QUERY_CHARS


def test_normalisation_keeps_operator_characters_verbatim():
    """`|||` tokenizes its right-hand side; it never parses query syntax, so
    nothing needs escaping and a term must not be silently mangled."""
    assert pgb.normalize_bm25_query("a:b c+d e-f g~h i^j") == "a:b c+d e-f g~h i^j"
    assert pgb.normalize_bm25_query("it's") == "it's"


def test_normalisation_of_blank_input_is_empty():
    assert pgb.normalize_bm25_query("   \n ") == ""
    assert pgb.normalize_bm25_query(None) == ""


# ---------------------------------------------------------------------------
# Extension availability + index state, against a spy session
# ---------------------------------------------------------------------------


def _session(rows: list):
    """Session whose execute() returns each queued result in turn."""
    session = MagicMock()
    results = []
    for row in rows:
        r = MagicMock()
        r.first.return_value = row
        r.fetchone.return_value = row
        r.scalar.return_value = row[0] if row else None
        results.append(r)
    session.execute.side_effect = results
    return session


@pytest.fixture(autouse=True)
def _clear_caches():
    pgb.reset_pg_bm25_caches()
    yield
    pgb.reset_pg_bm25_caches()


def test_extension_detected_when_pg_extension_has_a_row():
    assert pgb.pg_search_installed(_session([(1,)])) is True


def test_extension_absent_when_no_row():
    assert pgb.pg_search_installed(_session([None])) is False


def test_extension_detection_never_raises_on_a_broken_session():
    session = MagicMock()
    session.execute.side_effect = RuntimeError("no connection")
    assert pgb.pg_search_installed(session) is False


def test_extension_detection_is_cached():
    session = _session([(1,), (1,)])
    assert pgb.pg_search_installed(session) is True
    assert pgb.pg_search_installed(session) is True
    assert session.execute.call_count == 1


def test_index_state_absent_when_the_index_is_missing():
    assert pgb.bm25_index_state(_session([None]), KB, "chunks") == "absent"


def test_index_state_building_while_indisvalid_is_false():
    assert pgb.bm25_index_state(_session([(False,)]), KB, "chunks") == "building"


def test_index_state_ready_when_valid():
    assert pgb.bm25_index_state(_session([(True,)]), KB, "chunks") == "ready"


def test_index_state_absent_on_a_broken_session():
    session = MagicMock()
    session.execute.side_effect = RuntimeError("boom")
    assert pgb.bm25_index_state(session, KB, "chunks") == "absent"


def test_index_ready_is_cached_per_kb_and_table():
    session = _session([(True,), (True,)])
    assert pgb.bm25_index_ready(session, KB, "chunks") is True
    assert pgb.bm25_index_ready(session, KB, "chunks") is True
    assert session.execute.call_count == 1
    # a different table is a different cache entry
    session2 = _session([(False,)])
    assert pgb.bm25_index_ready(session2, KB, "full_documents") is False


def test_cache_invalidation_forces_a_fresh_read():
    session = _session([(True,), (True,)])
    assert pgb.bm25_index_ready(session, KB, "chunks") is True
    pgb.invalidate_bm25_index_cache(KB)
    assert pgb.bm25_index_ready(session, KB, "chunks") is True
    assert session.execute.call_count == 2
