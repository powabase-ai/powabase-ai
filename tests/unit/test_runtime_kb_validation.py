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
    data = {"runtime_knowledge_bases": [{"id": f"kb-{i}"} for i in range(RUNTIME_KB_MAX_ENTRIES + 1)]}
    _, err = validate_runtime_knowledge_bases(data, MagicMock())
    assert err is not None and str(RUNTIME_KB_MAX_ENTRIES) in err


def test_unknown_kb_id_rejected():
    data = {"runtime_knowledge_bases": [{"id": "kb-known"}, {"id": "kb-ghost"}]}
    _, err = validate_runtime_knowledge_bases(data, _db_returning_ids(["kb-known"]))
    assert err is not None and "kb-ghost" in err


def test_valid_entries_pass_through_unmodified():
    entries = [{"id": "kb-1", "top_k": 3, "source_ids": ["s1"]}, {"id": "kb-2"}]
    configs, err = validate_runtime_knowledge_bases(
        {"runtime_knowledge_bases": entries}, _db_returning_ids(["kb-1", "kb-2"])
    )
    assert err is None and configs == entries
