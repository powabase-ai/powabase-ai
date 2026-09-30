"""Text Postgres cannot store is cleaned where it is written, not rejected.

Postgres ``text`` has no NUL character and ``jsonb`` refuses the ``\\u0000``
escape and unpaired surrogates, so one such character in an extracted
document, a user's message or a model's reply used to fail the whole write:
indexing a source failed, and a run could not be recorded. Runs against a
real Postgres, because only the server says what it refuses.
"""

import asyncio
import uuid

from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.models.tenant import AgentRunStatus
from agentic_project_service.services.session import persist_agent_run


def _run_row(run_id):
    return db.session.execute(
        text(
            'SELECT input_messages, output_messages, content, error FROM "ai".agent_runs '
            "WHERE run_id = :rid"
        ),
        {"rid": run_id},
    ).one()


def _tool_result(run_id):
    return db.session.execute(
        text(
            'SELECT e.result, e.result_preview FROM "ai".tool_call_events e '
            'JOIN "ai".agent_runs r ON r.id = e.agent_run_id WHERE r.run_id = :rid'
        ),
        {"rid": run_id},
    ).one()


def _persist(**kwargs):
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    persist_agent_run(
        db_session=db.session,
        run_id=run_id,
        status=AgentRunStatus.COMPLETED,
        **kwargs,
    )
    db.session.commit()
    return run_id


def test_a_run_whose_message_holds_nul_is_recorded_without_it(app):
    with app.app_context():
        run_id = _persist(
            input_messages=[{"role": "user", "content": "before\x00after"}],
            output_messages=[{"role": "assistant", "content": "re\x00ply"}],
            content="re\x00ply",
            error="warn\x00ing",
            tool_calls=[{"tool_name": "search", "arguments": {"q": "a\x00b"}, "result": "hit\x00"}],
        )
        row = _run_row(run_id)
        result, preview = _tool_result(run_id)

    assert row.input_messages == [{"role": "user", "content": "beforeafter"}]
    assert row.output_messages == [{"role": "assistant", "content": "reply"}]
    assert row.content == "reply"
    assert row.error == "warning"
    assert result == "hit"
    assert preview == "hit"


def test_an_unpaired_surrogate_becomes_a_replacement_character(app):
    """A JSON request body may carry an escaped lone surrogate, which Python
    decodes into a str that cannot be encoded as UTF-8 at all."""
    with app.app_context():
        run_id = _persist(
            input_messages=[{"role": "user", "content": "x\ud800y"}],
            content="x\udc00y",
        )
        row = _run_row(run_id)

    assert row.input_messages == [{"role": "user", "content": "x\ufffdy"}]
    assert row.content == "x\ufffdy"


def test_text_that_merely_spells_an_escape_is_kept(app):
    """A message *about* ``\\u0000`` -- a backslash and five characters -- is
    ordinary text and must reach the database unchanged, and so must a
    character outside the Basic Multilingual Plane."""
    literal = "write \\u0000 for NUL \U0001f600"
    with app.app_context():
        run_id = _persist(input_messages=[{"role": "user", "content": literal}], content=literal)
        row = _run_row(run_id)

    assert row.input_messages == [{"role": "user", "content": literal}]
    assert row.content == literal


def test_a_chunk_holding_nul_is_indexed_without_it(app, test_knowledge_base):
    from agentic_project_service.services.knowledge_store import PgVectorKnowledgeStore

    kb_id = test_knowledge_base["id"]
    src_id, is_id = str(uuid.uuid4()), str(uuid.uuid4())
    with app.app_context():
        db.session.execute(
            text(
                "INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status) "
                "VALUES (:id, 's', 'text/plain', 'sources/s.txt', 'extracted')"
            ),
            {"id": src_id},
        )
        db.session.execute(
            text(
                "INSERT INTO ai.indexed_sources (id, knowledge_base_id, source_id, index_status) "
                "VALUES (:id, :kb, :src, 'indexing')"
            ),
            {"id": is_id, "kb": kb_id, "src": src_id},
        )
        db.session.commit()

        store = PgVectorKnowledgeStore(db.session, kb_id)
        asyncio.run(
            store.store_chunks(
                is_id,
                [
                    {
                        "text": "col\x00umn",
                        "chunk_index": 0,
                        "source_id": src_id,
                        "meta": {"heading": "Intro\x00"},
                    }
                ],
            )
        )
        db.session.commit()
        row = db.session.execute(
            text("SELECT text, meta FROM ai.chunks WHERE indexed_source_id = :id"),
            {"id": is_id},
        ).one()

    assert row.text == "column"
    assert row.meta == {"heading": "Intro"}


def test_a_jsonb_value_written_through_the_orm_is_cleaned(app):
    from agentic_project_service.models.tenant import Source

    with app.app_context():
        source = Source(
            id=uuid.uuid4(),
            name="n\x00ame",
            file_type="text/plain",
            storage_path="sources/o.txt",
            extraction_status="pending",
            auto_metadata={"title": "Ti\x00tle", "parts": ["a\x00"]},
        )
        db.session.add(source)
        db.session.commit()
        row = db.session.execute(
            text("SELECT name, auto_metadata FROM ai.sources WHERE id = :id"),
            {"id": str(source.id)},
        ).one()

    assert row.name == "name"
    assert row.auto_metadata == {"title": "Title", "parts": ["a"]}


def test_a_multi_row_orm_flush_is_cleaned(app):
    """Several objects of one model in one flush go out as a single
    multi-row INSERT ... RETURNING (SQLAlchemy's insertmanyvalues)."""
    from agentic_project_service.models.tenant import Source

    with app.app_context():
        sources = [
            Source(
                name=f"many\x00{i}",
                file_type="text/plain",
                storage_path=f"sources/many{i}.txt",
                auto_metadata={"n\x00": i},
            )
            for i in range(3)
        ]
        db.session.add_all(sources)
        db.session.commit()
        rows = db.session.execute(
            text("SELECT name, auto_metadata FROM ai.sources WHERE id = ANY(:ids) ORDER BY name"),
            {"ids": [s.id for s in sources]},
        ).all()

    assert [(r.name, r.auto_metadata) for r in rows] == [(f"many{i}", {"n": i}) for i in range(3)]


def test_a_multi_row_insert_returning_is_cleaned(app):
    from sqlalchemy import insert

    from agentic_project_service.models.tenant import Source

    table = Source.__table__
    with app.app_context():
        ids = (
            db.session.execute(
                insert(table).returning(table.c.id),
                [
                    {
                        "name": f"ret\x00{i}",
                        "file_type": "text/plain",
                        "storage_path": f"sources/r{i}",
                    }
                    for i in range(3)
                ],
            )
            .scalars()
            .all()
        )
        db.session.commit()
        names = (
            db.session.execute(
                text("SELECT name FROM ai.sources WHERE id = ANY(:ids) ORDER BY name"),
                {"ids": list(ids)},
            )
            .scalars()
            .all()
        )

    assert len(ids) == 3
    assert names == ["ret0", "ret1", "ret2"]
