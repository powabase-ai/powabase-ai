"""Every registered citation unit round-trips through ai.message_citations.

Runs on a real Postgres. Readers keep their historical contract (cited units
only) and add the new fields. GET /api/agents/runs/<run_id> exposes every unit.
"""

import uuid

from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.models.tenant import AgentRunStatus
from agentic_project_service.services.citations import persist_citations
from agentic_project_service.services.session import (
    fetch_citations_for_runs,
    list_runs_for_session,
    persist_agent_run,
)

ITEM = "11111111-1111-1111-1111-111111111111"
KB = "22222222-2222-2222-2222-222222222222"


def _run(session_uuid: str | None = None) -> tuple[str, str]:
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    run_uuid = persist_agent_run(
        db_session=db.session,
        run_id=run_id,
        status=AgentRunStatus.COMPLETED,
        input_messages=[{"role": "user", "content": "hi"}],
        output_messages=[{"role": "assistant", "content": "hello"}],
        content="hello",
        db_session_uuid=session_uuid,
    )
    db.session.commit()
    return run_id, run_uuid


def _units(source_id: str) -> list[dict]:
    return [
        {
            "key": "1",
            "kind": "kb_chunk",
            "tool_name": "knowledge_search",
            "call_id": "call_a",
            "title": "test.pdf",
            "url": None,
            "item_id": ITEM,
            "source_id": source_id,
            "source_name": "test.pdf",
            "knowledge_base_id": KB,
            "text_excerpt": "alpha",
            "meta": {"pages": [1]},
            "cited": True,
        },
        {
            "key": "2",
            "kind": "tool_item",
            "tool_name": "mcp__docs__search_items",
            "call_id": "call_b",
            "title": "Item A",
            "url": "https://example.org/a",
            "item_id": None,
            "source_id": None,
            "source_name": "",
            "knowledge_base_id": None,
            "text_excerpt": '{"name": "Item A"}',
            "meta": {"name": "Item A"},
            "cited": False,
        },
        {
            "key": "3",
            "kind": "tool_call",
            "tool_name": "mcp__docs__lookup",
            "call_id": "call_c",
            "title": "mcp__docs__lookup {}",
            "url": None,
            "item_id": None,
            "source_id": None,
            "source_name": "",
            "knowledge_base_id": None,
            "text_excerpt": "Plain answer text.",
            "meta": {"raw": "Plain answer text.", "arguments": {}},
            "cited": True,
        },
    ]


def test_readers_return_cited_units_by_default_and_all_on_request(app, test_source):
    with app.app_context():
        run_id, run_uuid = _run()
        persist_citations(db.session, run_id, _units(test_source["id"]))
        db.session.commit()

        cited = fetch_citations_for_runs(db.session, [run_uuid])[run_uuid]
        everything = fetch_citations_for_runs(db.session, [run_uuid], include_uncited=True)[
            run_uuid
        ]

    assert [c["key"] for c in cited] == ["1", "3"]
    assert [c["key"] for c in everything] == ["1", "2", "3"]
    first, second, third = everything
    assert first == {
        "key": "1",
        "item_id": ITEM,
        "source_id": test_source["id"],
        "text_excerpt": "alpha",
        "meta": {"pages": [1]},
        "source_name": "test.pdf",
        "kind": "kb_chunk",
        "tool_name": "knowledge_search",
        "call_id": "call_a",
        "title": "test.pdf",
        "url": None,
        "knowledge_base_id": KB,
        "cited": True,
    }
    assert (second["kind"], second["url"], second["cited"]) == (
        "tool_item",
        "https://example.org/a",
        False,
    )
    assert third["meta"] == {"raw": "Plain answer text.", "arguments": {}}


def test_a_legacy_citation_dict_is_a_cited_kb_chunk(app):
    with app.app_context():
        run_id, run_uuid = _run()
        persist_citations(
            db.session,
            run_id,
            [{"key": "1", "item_id": ITEM, "source_id": None, "text_excerpt": "x", "meta": {}}],
        )
        db.session.commit()
        [row] = fetch_citations_for_runs(db.session, [run_uuid])[run_uuid]
    assert (row["kind"], row["cited"]) == ("kb_chunk", True)


def test_non_uuid_ids_and_missing_sources_store_null(app):
    """Review focus 5: one bad id must not abort the run's whole insert."""
    unit = {
        "key": "1",
        "kind": "kb_chunk",
        "item_id": "toc-a",
        "source_id": str(uuid.uuid4()),  # no such source
        "knowledge_base_id": "kb-1",
        "text_excerpt": "x",
        "meta": {},
        "cited": True,
    }
    other = {**unit, "key": "2", "source_id": "not-a-uuid"}
    with app.app_context():
        run_id, run_uuid = _run()
        persist_citations(db.session, run_id, [unit, other])
        db.session.commit()
        rows = fetch_citations_for_runs(db.session, [run_uuid])[run_uuid]
    assert [(r["item_id"], r["source_id"], r["knowledge_base_id"]) for r in rows] == [
        (None, None, None),
        (None, None, None),
    ]


def test_session_runs_list_only_cited_units(app, test_source):
    session_uuid = str(uuid.uuid4())
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    with app.app_context():
        db.session.execute(
            text(
                'INSERT INTO "ai".agent_sessions (id, session_id, agent_id) '
                "VALUES (:id, :sid, NULL)"
            ),
            {"id": session_uuid, "sid": session_id},
        )
        db.session.commit()
        run_id, _ = _run(session_uuid)
        persist_citations(db.session, run_id, _units(test_source["id"]))
        db.session.commit()
        [run] = list_runs_for_session(db.session, session_id)
    assert [c["key"] for c in run["citations"]] == ["1", "3"]


def test_get_run_exposes_every_unit(app, client, mock_auth, auth_headers, test_source):
    with app.app_context():
        run_id, _ = _run()
        persist_citations(db.session, run_id, _units(test_source["id"]))
        db.session.commit()

    resp = client.get(f"/api/agents/runs/{run_id}", headers=auth_headers)
    assert resp.status_code == 200
    units = resp.get_json()["citation_units"]
    assert [(u["key"], u["kind"], u["cited"]) for u in units] == [
        ("1", "kb_chunk", True),
        ("2", "tool_item", False),
        ("3", "tool_call", True),
    ]
