"""Tests for the docs-RAG refresh service (services/docs_refresh.py).

Covers the pure logic (hashing, title extraction, markdown discovery, dedup),
the idempotent KB bootstrap (against the test DB), the unchanged-doc short
circuit, and the refresh orchestration with git/http/ingest mocked.
"""

import uuid
from unittest.mock import MagicMock

from sqlalchemy import text

from agentic_project_service.db import db
from agentic_project_service.services import docs_refresh as dr


# --- pure helpers ----------------------------------------------------------


def test_sha256_text_stable_and_sensitive():
    assert dr.sha256_text("hello") == dr.sha256_text("hello")
    assert dr.sha256_text("hello") != dr.sha256_text("hello!")


def test_doc_title_prefers_h1():
    assert dr.doc_title("# Auth connection\n\nbody", "fallback") == "Auth connection"


def test_doc_title_frontmatter():
    md = "---\ntitle: My Guide\n---\n\ncontent"
    assert dr.doc_title(md, "fallback") == "My Guide"


def test_doc_title_fallback():
    assert dr.doc_title("just text, no heading", "rel/path.md") == "rel/path.md"


def test_should_reindex():
    assert dr.should_reindex(None, "abc") is True
    assert dr.should_reindex("old", "new") is True
    assert dr.should_reindex("same", "same") is False


def test_discover_markdown_docs(tmp_path):
    (tmp_path / "guides").mkdir()
    (tmp_path / "guides" / "auth.md").write_text("# Auth\nhow to connect", encoding="utf-8")
    (tmp_path / "intro.mdx").write_text("# Intro\nwelcome", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")  # skipped
    (tmp_path / "notes.txt").write_text("not markdown", encoding="utf-8")  # skipped

    docs = dr.discover_markdown_docs(tmp_path, key_prefix="docs")
    keys = [d.key for d in docs]
    assert keys == ["docs:guides/auth.md", "docs:intro.mdx"]  # sorted, empties/non-md skipped
    auth = docs[0]
    assert auth.title == "Auth"
    assert auth.content_hash == dr.sha256_text("# Auth\nhow to connect")


def test_doc_url_from_docs_key():
    rec = dr.DocRecord(key="docs:guides/auth-connection.md", title="t", content="c", content_hash="h")
    assert dr._doc_url(rec) == "https://docs.powabase.ai/guides/auth-connection"
    other = dr.DocRecord(key="agent-skills:x.md", title="t", content="c", content_hash="h")
    assert dr._doc_url(other) is None


# --- bootstrap (DB) --------------------------------------------------------


def test_bootstrap_docs_kb_is_idempotent(client, app):
    with app.app_context():
        kb1 = dr.bootstrap_docs_kb(db.session)
        kb2 = dr.bootstrap_docs_kb(db.session)
        assert kb1 == kb2

        row = db.session.execute(
            text(
                'SELECT name, indexing_config FROM ai.knowledge_bases WHERE id = :id'
            ),
            {"id": kb1},
        ).fetchone()
        assert row[0] == dr.DOCS_KB_NAME
        assert row[1]["strategy"] == "full_document"

        # only one docs KB exists
        count = db.session.execute(
            text("SELECT count(*) FROM ai.knowledge_bases WHERE name = :n"),
            {"n": dr.DOCS_KB_NAME},
        ).scalar()
        assert count == 1


def test_bootstrap_docs_kb_uses_dense_retrieval(client, app):
    """The docs KB is created with dense (vector_search) retrieval, not the
    full_document strategy default (hybrid) — dense suits doc Q&A and sidesteps
    the BM25s sparse-index bug on incrementally-indexed KBs."""
    with app.app_context():
        kb_id = dr.bootstrap_docs_kb(db.session)
        row = db.session.execute(
            text("SELECT retrieval_config FROM ai.knowledge_bases WHERE id = :id"),
            {"id": kb_id},
        ).fetchone()
        assert row[0]["method"] == "vector_search"


# --- ingestion dedup -------------------------------------------------------


def test_ingest_extracted_and_indexed_short_circuits(client, app):
    """A doc that's extracted, hash-matching, AND already indexed into THIS KB is
    skipped (no upload)."""
    with app.app_context():
        source_id = str(uuid.uuid4())
        kb_id = dr.bootstrap_docs_kb(db.session)  # a real KB (satisfies the FK)
        name = "docs:guides/auth.md"
        content_hash = dr.sha256_text("# Auth\nbody")
        db.session.execute(
            text(
                "INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status, content_hash) "
                "VALUES (:id, :n, 'md', 'sources/x', 'extracted', :h)"
            ),
            {"id": source_id, "n": name, "h": content_hash},
        )
        db.session.execute(
            text(
                "INSERT INTO ai.indexed_sources (id, knowledge_base_id, source_id, index_status) "
                "VALUES (:iid, :kb, :sid, 'indexed')"
            ),
            {"iid": str(uuid.uuid4()), "kb": kb_id, "sid": source_id},
        )
        db.session.commit()

        storage = MagicMock()
        rec = dr.DocRecord(key=name, title="Auth", content="# Auth\nbody", content_hash=content_hash)
        result = dr._ingest_markdown_source(db.session, storage, kb_id, rec)

        assert result == "unchanged"
        storage.upload.assert_not_called()


def test_ingest_extracted_but_index_failed_redispatches(client, app, mocker):
    """A doc whose indexed_sources row exists but is NOT 'indexed' (e.g. 'failed'
    after a post-extraction embed error) is re-dispatched, not skipped. Regression
    for the strand-forever bug: the skip guard tested mere row existence, so a
    'failed' row matched forever and the doc was never re-indexed."""
    with app.app_context():
        source_id = str(uuid.uuid4())
        kb_id = dr.bootstrap_docs_kb(db.session)
        name = "docs:guides/auth.md"
        content_hash = dr.sha256_text("# Auth\nbody")
        db.session.execute(
            text(
                "INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status, content_hash) "
                "VALUES (:id, :n, 'md', 'sources/x', 'extracted', :h)"
            ),
            {"id": source_id, "n": name, "h": content_hash},
        )
        db.session.execute(
            text(
                "INSERT INTO ai.indexed_sources (id, knowledge_base_id, source_id, index_status) "
                "VALUES (:iid, :kb, :sid, 'failed')"
            ),
            {"iid": str(uuid.uuid4()), "kb": kb_id, "sid": source_id},
        )
        db.session.commit()

        mocker.patch("agentic_project_service.tasks.extraction.update_source_extraction_result")
        index_mock = mocker.patch(
            "agentic_project_service.routes.knowledge_bases.index_source_into_kb",
            return_value={"status": "success"},
        )
        storage = MagicMock()
        rec = dr.DocRecord(key=name, title="Auth", content="# Auth\nbody", content_hash=content_hash)
        result = dr._ingest_markdown_source(db.session, storage, kb_id, rec)

        assert result == "dispatched"
        index_mock.assert_called_once()
        storage.upload.assert_called_once()


def test_ingest_extracted_but_unindexed_redispatches(client, app, mocker):
    """A doc that's extracted + hash-matching but NOT indexed into the KB is
    re-dispatched, not skipped. Regression for the strand-forever bug: a
    post-extraction index failure (transient error, or the pre-fix billing 402)
    left the doc at 'extracted', and every later refresh skipped it as
    'unchanged' — so it was never indexed."""
    with app.app_context():
        source_id = str(uuid.uuid4())
        kb_id = str(uuid.uuid4())
        name = "docs:guides/auth.md"
        content_hash = dr.sha256_text("# Auth\nbody")
        db.session.execute(
            text(
                "INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status, content_hash) "
                "VALUES (:id, :n, 'md', 'sources/x', 'extracted', :h)"
            ),
            {"id": source_id, "n": name, "h": content_hash},
        )
        db.session.commit()  # deliberately NO indexed_sources row

        # Stub the post-extraction write + the async index dispatch (imported
        # lazily inside _ingest_markdown_source, so patch them at their source).
        mocker.patch("agentic_project_service.tasks.extraction.update_source_extraction_result")
        index_mock = mocker.patch(
            "agentic_project_service.routes.knowledge_bases.index_source_into_kb",
            return_value={"status": "success"},
        )
        storage = MagicMock()
        rec = dr.DocRecord(key=name, title="Auth", content="# Auth\nbody", content_hash=content_hash)
        result = dr._ingest_markdown_source(db.session, storage, kb_id, rec)

        assert result == "dispatched"
        index_mock.assert_called_once()
        storage.upload.assert_called_once()


# --- orchestration ---------------------------------------------------------


def test_refresh_docs_kb_orchestration(client, app, mocker):
    """Orchestrator wires llms-full + repos through ingestion and tallies results."""
    with app.app_context():
        mocker.patch.object(dr, "get_storage", return_value=MagicMock())
        mocker.patch.object(
            dr,
            "fetch_llms_full",
            return_value=dr.DocRecord(
                key="llms-full:llms-full.txt", title="t", content="c", content_hash="h"
            ),
        )
        # Skip real git clones; the repo loop contributes no docs.
        mocker.patch.object(dr, "_git_clone", return_value=False)
        ingest = mocker.patch.object(dr, "_ingest_markdown_source", return_value="dispatched")

        result = dr.refresh_docs_kb()

        assert result["docs"] == 1
        assert result["dispatched"] == 1
        assert result["unchanged"] == 0
        assert result["error"] == 0
        assert result["kb_id"]
        ingest.assert_called_once()


def test_refresh_dedups_identical_content(client, app, mocker):
    """Two docs with identical content are ingested once (UNIQUE(content_hash))."""
    with app.app_context():
        mocker.patch.object(dr, "get_storage", return_value=MagicMock())
        dup = dr.DocRecord(key="a", title="t", content="same", content_hash=dr.sha256_text("same"))
        dup2 = dr.DocRecord(key="b", title="t", content="same", content_hash=dr.sha256_text("same"))
        mocker.patch.object(dr, "fetch_llms_full", return_value=dup)
        mocker.patch.object(dr, "_git_clone", return_value=True)
        mocker.patch.object(dr, "discover_markdown_docs", return_value=[dup2])
        ingest = mocker.patch.object(dr, "_ingest_markdown_source", return_value="dispatched")

        result = dr.refresh_docs_kb()
        assert result["dispatched"] == 1
        assert ingest.call_count == 1  # deduped by content_hash


def test_refresh_reports_source_reachability(client, app, mocker):
    """The result tallies upstream-source reachability, and a total gather
    failure (llms-full down + all git clones fail) is logged as an ERROR rather
    than an INFO "complete" that looks like a healthy no-op."""
    with app.app_context():
        mocker.patch.object(dr, "get_storage", return_value=MagicMock())
        mocker.patch.object(dr, "fetch_llms_full", return_value=None)  # source down
        mocker.patch.object(dr, "_git_clone", return_value=False)  # all repos unreachable
        error_log = mocker.patch.object(dr.logger, "error")

        result = dr.refresh_docs_kb()

        assert result["docs"] == 0
        assert result["sources_ok"] == 0
        assert result["sources_total"] == 1 + len(dr.DOCS_REPOS)
        error_log.assert_called_once()  # loud, not a silent INFO no-op


def test_refresh_prunes_doc_removed_upstream(client, app, mocker):
    """A doc that's no longer produced by any upstream source is pruned from
    ai.sources — it must not stay live/citable in the KB forever. The "docs"
    prefix still gathers a sibling doc this run (only removed.md is actually
    gone), so pruning that prefix is safe (contrast with
    test_refresh_empty_gather_does_not_prune below, where the WHOLE prefix
    gathers zero docs and must NOT be pruned)."""
    with app.app_context():
        kb_id = dr.bootstrap_docs_kb(db.session)
        stale_id = str(uuid.uuid4())
        db.session.execute(
            text(
                "INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status, content_hash) "
                "VALUES (:id, :n, 'md', 'sources/x', 'extracted', :h)"
            ),
            {"id": stale_id, "n": "docs:removed.md", "h": dr.sha256_text("gone")},
        )
        db.session.commit()

        mocker.patch.object(dr, "get_storage", return_value=MagicMock())
        mocker.patch.object(dr, "fetch_llms_full", return_value=None)
        mocker.patch.object(dr, "_git_clone", return_value=True)
        surviving = dr.DocRecord(
            key="docs:present.md", title="t", content="c", content_hash=dr.sha256_text("c")
        )
        mocker.patch.object(
            dr,
            "discover_markdown_docs",
            side_effect=lambda root, key_prefix: [surviving] if key_prefix == "docs" else [],
        )
        mocker.patch.object(dr, "_ingest_markdown_source", return_value="dispatched")

        result = dr.refresh_docs_kb()

        assert result["pruned"] == 1
        row = db.session.execute(
            text("SELECT 1 FROM ai.sources WHERE id = :id"), {"id": stale_id}
        ).fetchone()
        assert row is None
        # sanity: kb bootstrap from setup is untouched
        assert dr.bootstrap_docs_kb(db.session) == kb_id


def test_refresh_empty_gather_does_not_prune(client, app, mocker):
    """Anti-wipe regression: a clone that SUCCEEDS but yields zero markdown docs
    for a prefix (upstream force-push-to-empty, default-branch rename, etc.)
    must NOT be treated as "everything under that prefix is gone". Pruning on a
    "reachable" but empty gather would wipe the whole prefix from the docs KB.
    A seeded docs:* row for that prefix must survive, and pruned must be 0."""
    with app.app_context():
        seeded_id = str(uuid.uuid4())
        db.session.execute(
            text(
                "INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status, content_hash) "
                "VALUES (:id, :n, 'md', 'sources/x', 'extracted', :h)"
            ),
            {"id": seeded_id, "n": "docs:guides/still-here.md", "h": dr.sha256_text("still here")},
        )
        db.session.commit()

        mocker.patch.object(dr, "get_storage", return_value=MagicMock())
        mocker.patch.object(dr, "fetch_llms_full", return_value=None)
        # Clone "succeeds" (repo reachable) for every repo, but the checkout
        # yields no markdown files at all — the empty-gather-after-success case.
        mocker.patch.object(dr, "_git_clone", return_value=True)
        mocker.patch.object(dr, "discover_markdown_docs", return_value=[])
        ingest = mocker.patch.object(dr, "_ingest_markdown_source", return_value="dispatched")

        result = dr.refresh_docs_kb()

        assert result["pruned"] == 0
        ingest.assert_not_called()
        row = db.session.execute(
            text("SELECT 1 FROM ai.sources WHERE id = :id"), {"id": seeded_id}
        ).fetchone()
        assert row is not None  # seeded row survives the empty-gather run


def test_refresh_rename_with_same_content_frees_hash(client, app, mocker):
    """Renaming a doc upstream (a.md -> b.md) with IDENTICAL content must not
    permanently strand it: pruning the stale a.md row before ingestion frees
    the UNIQUE(content_hash) slot so b.md indexes cleanly with no error."""
    with app.app_context():
        content_hash = dr.sha256_text("same content")
        old_id = str(uuid.uuid4())
        db.session.execute(
            text(
                "INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status, content_hash) "
                "VALUES (:id, :n, 'md', 'sources/x', 'extracted', :h)"
            ),
            {"id": old_id, "n": "docs:a.md", "h": content_hash},
        )
        db.session.commit()

        mocker.patch.object(dr, "get_storage", return_value=MagicMock())
        mocker.patch.object(dr, "fetch_llms_full", return_value=None)
        mocker.patch.object(dr, "_git_clone", return_value=True)
        renamed = dr.DocRecord(
            key="docs:b.md", title="t", content="same content", content_hash=content_hash
        )
        mocker.patch.object(dr, "discover_markdown_docs", return_value=[renamed])
        mocker.patch("agentic_project_service.tasks.extraction.update_source_extraction_result")
        index_mock = mocker.patch(
            "agentic_project_service.routes.knowledge_bases.index_source_into_kb",
            return_value={"status": "success"},
        )

        result = dr.refresh_docs_kb()

        assert result["error"] == 0  # no IntegrityError from the freed content_hash
        assert result["dispatched"] == 1
        index_mock.assert_called_once()

        old_row = db.session.execute(
            text("SELECT 1 FROM ai.sources WHERE name = 'docs:a.md'")
        ).fetchone()
        assert old_row is None
        new_row = db.session.execute(
            text("SELECT content_hash FROM ai.sources WHERE name = 'docs:b.md'")
        ).fetchone()
        assert new_row is not None
        assert new_row[0] == content_hash


def test_refresh_still_skips_unchanged_doc_after_pruning(client, app, mocker):
    """Pruning must not disturb the existing content_hash skip-unchanged
    optimization: a doc that's unchanged and already indexed is still skipped
    (no re-upload, no re-dispatch)."""
    with app.app_context():
        kb_id = dr.bootstrap_docs_kb(db.session)
        source_id = str(uuid.uuid4())
        content_hash = dr.sha256_text("# Auth\nbody")
        db.session.execute(
            text(
                "INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status, content_hash) "
                "VALUES (:id, :n, 'md', 'sources/x', 'extracted', :h)"
            ),
            {"id": source_id, "n": "docs:guides/auth.md", "h": content_hash},
        )
        db.session.execute(
            text(
                "INSERT INTO ai.indexed_sources (id, knowledge_base_id, source_id, index_status) "
                "VALUES (:iid, :kb, :sid, 'indexed')"
            ),
            {"iid": str(uuid.uuid4()), "kb": kb_id, "sid": source_id},
        )
        db.session.commit()

        storage = MagicMock()
        mocker.patch.object(dr, "get_storage", return_value=storage)
        mocker.patch.object(dr, "fetch_llms_full", return_value=None)
        mocker.patch.object(dr, "_git_clone", return_value=True)
        unchanged = dr.DocRecord(
            key="docs:guides/auth.md",
            title="Auth",
            content="# Auth\nbody",
            content_hash=content_hash,
        )
        mocker.patch.object(dr, "discover_markdown_docs", return_value=[unchanged])

        result = dr.refresh_docs_kb()

        assert result["unchanged"] == 1
        assert result["dispatched"] == 0
        assert result["pruned"] == 0
        storage.upload.assert_not_called()
        row = db.session.execute(
            text("SELECT 1 FROM ai.sources WHERE id = :id"), {"id": source_id}
        ).fetchone()
        assert row is not None


def test_refresh_logs_per_doc_index_error(client, app, mocker):
    """A doc whose index dispatch returns "error:<msg>" is counted AND its message
    is logged with the doc key (not an anonymous error count)."""
    with app.app_context():
        mocker.patch.object(dr, "get_storage", return_value=MagicMock())
        mocker.patch.object(
            dr,
            "fetch_llms_full",
            return_value=dr.DocRecord(key="k", title="t", content="c", content_hash="h"),
        )
        mocker.patch.object(dr, "_git_clone", return_value=False)
        mocker.patch.object(dr, "_ingest_markdown_source", return_value="error:boom")
        warn_log = mocker.patch.object(dr.logger, "warning")

        result = dr.refresh_docs_kb()

        assert result["error"] == 1
        assert any("boom" in str(c.args) for c in warn_log.call_args_list)
