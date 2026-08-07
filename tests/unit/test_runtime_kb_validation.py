"""Validation for the per-request runtime_knowledge_bases field."""

from unittest.mock import MagicMock

from agentic_project_service.routes._runtime_kb import (
    RUNTIME_KB_MAX_ENTRIES,
    validate_runtime_knowledge_bases,
)


def _db_returning_ids(ids):
    """A db_session whose execute() yields rows of existing KB ids."""
    db = MagicMock()
    db.execute.return_value = [(i,) for i in ids]
    return db


def test_absent_field_is_valid_and_empty():
    configs, err = validate_runtime_knowledge_bases({}, MagicMock())
    assert configs == [] and err is None


def test_non_list_rejected():
    _, err = validate_runtime_knowledge_bases({"runtime_knowledge_bases": "kb-1"}, MagicMock())
    assert err is not None and "list" in err


def test_entry_without_id_rejected():
    data = {"runtime_knowledge_bases": [{"top_k": 3}]}
    _, err = validate_runtime_knowledge_bases(data, MagicMock())
    assert err is not None and "id" in err


def test_cap_enforced():
    data = {
        "runtime_knowledge_bases": [{"id": f"kb-{i}"} for i in range(RUNTIME_KB_MAX_ENTRIES + 1)]
    }
    _, err = validate_runtime_knowledge_bases(data, MagicMock())
    assert err is not None and str(RUNTIME_KB_MAX_ENTRIES) in err


_KB_KNOWN = "11111111-1111-1111-1111-111111111111"
_KB_GHOST = "22222222-2222-2222-2222-222222222222"
_KB_1 = "33333333-3333-3333-3333-333333333333"
_KB_2 = "44444444-4444-4444-4444-444444444444"


def test_unknown_kb_id_rejected():
    data = {"runtime_knowledge_bases": [{"id": _KB_KNOWN}, {"id": _KB_GHOST}]}
    _, err = validate_runtime_knowledge_bases(data, _db_returning_ids([_KB_KNOWN]))
    assert err is not None and _KB_GHOST in err


_SRC_1 = "55555555-5555-5555-5555-555555555555"


def _db_returning_ids_then(ids_lists):
    """A db_session whose execute() yields successive id-row lists per call
    (KB existence query, then source existence query)."""
    db = MagicMock()
    db.execute.side_effect = [[(i,) for i in ids] for ids in ids_lists]
    return db


def test_valid_entries_pass_through_unmodified():
    entries = [{"id": _KB_1, "top_k": 3, "source_ids": [_SRC_1]}, {"id": _KB_2}]
    configs, err = validate_runtime_knowledge_bases(
        {"runtime_knowledge_bases": entries},
        _db_returning_ids_then([[_KB_1, _KB_2], [_SRC_1]]),
    )
    assert err is None and configs == entries


def test_malformed_uuid_rejected_without_db():
    db = MagicMock()
    configs, err = validate_runtime_knowledge_bases(
        {"runtime_knowledge_bases": [{"id": "not-a-uuid"}]}, db
    )
    assert configs == []
    assert err is not None and "not-a-uuid" in err
    db.execute.assert_not_called()

    db2 = MagicMock()
    configs2, err2 = validate_runtime_knowledge_bases(
        {"runtime_knowledge_bases": [{"id": 123}]}, db2
    )
    assert configs2 == []
    assert err2 is not None and "123" in err2
    db2.execute.assert_not_called()


_KB_NORM = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def test_id_normalized_to_canonical_form():
    """Uppercase / braced / undashed ids all parse as valid UUIDs but fail a
    plain set-comparison against psycopg's canonical lowercase-dashed form,
    and later miss the tool builder's kb_map lookup — silently dropping the
    KB. The validator must normalize before both the DB check and the
    returned config."""
    variants = [
        _KB_NORM.upper(),
        "{" + _KB_NORM + "}",
        _KB_NORM.replace("-", ""),
    ]
    for variant in variants:
        db = _db_returning_ids([_KB_NORM])
        configs, err = validate_runtime_knowledge_bases(
            {"runtime_knowledge_bases": [{"id": variant}]}, db
        )
        assert err is None, f"variant {variant!r} should validate: {err}"
        assert configs == [{"id": _KB_NORM}], f"variant {variant!r} not normalized: {configs}"


def test_duplicate_ids_after_normalization_rejected():
    data = {"runtime_knowledge_bases": [{"id": _KB_1}, {"id": _KB_1.upper()}]}
    _, err = validate_runtime_knowledge_bases(data, MagicMock())
    assert err is not None and "duplicate" in err and _KB_1 in err


def test_top_k_rejected_out_of_range_or_wrong_type():
    for bad in (0, 101, "5", True):
        data = {"runtime_knowledge_bases": [{"id": _KB_1, "top_k": bad}]}
        _, err = validate_runtime_knowledge_bases(data, MagicMock())
        assert err is not None and "top_k" in err, f"top_k={bad!r} should be rejected"


def test_retrieval_method_rejected_unknown():
    data = {"runtime_knowledge_bases": [{"id": _KB_1, "retrieval_method": "semantic"}]}
    _, err = validate_runtime_knowledge_bases(data, MagicMock())
    assert err is not None and "retrieval_method" in err


def test_similarity_threshold_rejected_out_of_range_or_wrong_type():
    for bad in (1.5, -0.1, True):
        data = {"runtime_knowledge_bases": [{"id": _KB_1, "similarity_threshold": bad}]}
        _, err = validate_runtime_knowledge_bases(data, MagicMock())
        assert err is not None and "similarity_threshold" in err, (
            f"threshold={bad!r} should be rejected"
        )


def test_filter_metadata_rejected_when_not_object():
    data = {"runtime_knowledge_bases": [{"id": _KB_1, "filter_metadata": "x"}]}
    _, err = validate_runtime_knowledge_bases(data, MagicMock())
    assert err is not None and "filter_metadata" in err


def test_source_ids_rejected_when_not_a_list():
    data = {"runtime_knowledge_bases": [{"id": _KB_1, "source_ids": "abc"}]}
    _, err = validate_runtime_knowledge_bases(data, MagicMock())
    assert err is not None and "source_ids" in err


def test_source_ids_rejected_when_unknown_in_db():
    """A stale source id must 400 up front — otherwise it silently filters
    retrieval to zero chunks downstream."""
    db = _db_returning_ids_then([[_KB_1], []])
    data = {"runtime_knowledge_bases": [{"id": _KB_1, "source_ids": [_SRC_1]}]}
    _, err = validate_runtime_knowledge_bases(data, db)
    assert err is not None and "unknown source id" in err and _SRC_1 in err


def test_valid_knobs_accepted():
    entry = {
        "id": _KB_1,
        "top_k": 5,
        "retrieval_method": "hybrid",
        "similarity_threshold": 0.5,
        "filter_metadata": {"key": "value"},
    }
    configs, err = validate_runtime_knowledge_bases(
        {"runtime_knowledge_bases": [entry]}, _db_returning_ids([_KB_1])
    )
    assert err is None
    assert configs == [entry]


def test_exactly_max_entries_passes():
    """Boundary: exactly RUNTIME_KB_MAX_ENTRIES valid entries must pass."""
    ids = [f"33333333-3333-3333-3333-{i:012d}" for i in range(RUNTIME_KB_MAX_ENTRIES)]
    data = {"runtime_knowledge_bases": [{"id": i} for i in ids]}
    configs, err = validate_runtime_knowledge_bases(data, _db_returning_ids(ids))
    assert err is None
    assert len(configs) == RUNTIME_KB_MAX_ENTRIES
