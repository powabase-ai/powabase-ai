"""runtime_knowledge_bases[].agent_ids: which orchestration sub-agents see an entry."""

from unittest.mock import MagicMock

import pytest

from agentic_project_service.routes._runtime_kb import validate_runtime_knowledge_bases

KB = "33333333-3333-3333-3333-333333333333"
A1 = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
A2 = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def _db(ids):
    db = MagicMock()
    db.execute.return_value = [(i,) for i in ids]
    return db


def _body(entry):
    return {"runtime_knowledge_bases": [entry]}


def test_rejected_on_agent_runs():
    _, err = validate_runtime_knowledge_bases(_body({"id": KB, "agent_ids": [A1]}), _db([KB]))
    assert err == "'agent_ids' is only accepted on orchestration runs"


def test_normalized_and_kept():
    configs, err = validate_runtime_knowledge_bases(
        _body({"id": KB, "agent_ids": [A1.upper()]}), _db([KB]), allowed_agent_ids={A1, A2}
    )
    assert err is None
    assert configs == [{"id": KB, "agent_ids": [A1]}]


def test_the_callers_entry_is_not_mutated():
    entry = {"id": KB, "agent_ids": [A1.upper()]}
    validate_runtime_knowledge_bases(_body(entry), _db([KB]), allowed_agent_ids={A1})
    assert entry["agent_ids"] == [A1.upper()]


def test_an_agent_outside_the_orchestration_is_rejected():
    _, err = validate_runtime_knowledge_bases(
        _body({"id": KB, "agent_ids": [A1]}), _db([KB]), allowed_agent_ids={A2}
    )
    assert err == f"agent_ids not in this orchestration: {A1}"


def test_an_orchestration_without_agents_rejects_every_agent_id():
    _, err = validate_runtime_knowledge_bases(
        _body({"id": KB, "agent_ids": [A1]}), _db([KB]), allowed_agent_ids=set()
    )
    assert err == f"agent_ids not in this orchestration: {A1}"


@pytest.mark.parametrize("bad", [[], "x", ["not-a-uuid"], [None], {"a": 1}])
def test_malformed_agent_ids_are_rejected(bad):
    _, err = validate_runtime_knowledge_bases(
        _body({"id": KB, "agent_ids": bad}), _db([KB]), allowed_agent_ids={A1}
    )
    assert err is not None


def test_entries_without_agent_ids_are_unchanged():
    configs, err = validate_runtime_knowledge_bases(
        _body({"id": KB}), _db([KB]), allowed_agent_ids=set()
    )
    assert err is None and configs == [{"id": KB}]


def test_a_null_agent_ids_is_absent_and_dropped():
    configs, err = validate_runtime_knowledge_bases(_body({"id": KB, "agent_ids": None}), _db([KB]))
    assert err is None and configs == [{"id": KB}]
