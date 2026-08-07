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


def test_valid_entries_pass_through_unmodified():
    entries = [{"id": _KB_1, "top_k": 3, "source_ids": ["s1"]}, {"id": _KB_2}]
    configs, err = validate_runtime_knowledge_bases(
        {"runtime_knowledge_bases": entries}, _db_returning_ids([_KB_1, _KB_2])
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
