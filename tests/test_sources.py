"""Tests for source CRUD routes."""

import hashlib
import io
import json
import uuid


class TestListSources:
    def test_list(self, client, mock_auth, auth_headers, test_source):
        resp = client.get("/api/sources", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] >= 1
        assert data["limit"] == 50
        assert data["offset"] == 0
        assert any(s["id"] == test_source["id"] for s in data["sources"])

    def test_list_empty(self, client, mock_auth, auth_headers):
        resp = client.get("/api/sources", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["total"] == 0

    def test_list_filter_by_status(self, client, mock_auth, auth_headers, test_source):
        resp = client.get(
            "/api/sources?status=extracted",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] >= 1
        assert all(s["extraction_status"] == "extracted" for s in data["sources"])

    def test_list_pagination(self, client, mock_auth, auth_headers, test_source):
        resp = client.get(
            "/api/sources?limit=1&offset=0",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert len(resp.get_json()["sources"]) <= 1

    def test_list_search_by_name(self, client, mock_auth, auth_headers, test_source):
        """`q` does a case-insensitive substring match on name (Studio's
        sources list search box — mirrors list_agents/list_knowledge_bases)."""
        resp = client.get("/api/sources?q=TEST", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert any(s["id"] == test_source["id"] for s in data["sources"])

        resp = client.get("/api/sources?q=no-such-source", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["sources"] == []

    def test_list_sort_by_name(self, client, mock_auth, auth_headers, app):
        from agentic_project_service.db import db
        from sqlalchemy import text

        with app.app_context():
            for name in ("bravo.pdf", "alpha.pdf", "charlie.pdf"):
                db.session.execute(
                    text(
                        'INSERT INTO "ai".sources '
                        "(id, name, file_type, storage_path, extraction_status) "
                        "VALUES (gen_random_uuid(), :name, 'application/pdf', "
                        ":storage_path, 'extracted')"
                    ),
                    {"name": name, "storage_path": f"sources/{name}"},
                )
            db.session.commit()

        resp = client.get("/api/sources?sort=name&order=asc", headers=auth_headers)
        assert resp.status_code == 200
        names = [s["name"] for s in resp.get_json()["sources"]]
        assert names == sorted(names)

    def test_list_invalid_sort_rejected(self, client, mock_auth, auth_headers):
        resp = client.get("/api/sources?sort=storage_path", headers=auth_headers)
        assert resp.status_code == 400


class TestGetSource:
    def test_get(self, client, mock_auth, auth_headers, test_source):
        resp = client.get(
            f"/api/sources/{test_source['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] == test_source["id"]
        assert data["name"] == "test.pdf"
        assert data["extraction_status"] == "extracted"

    def test_get_not_found(self, client, mock_auth, auth_headers):
        resp = client.get(
            f"/api/sources/{uuid.uuid4()}",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestUploadSource:
    def test_upload(self, client, mock_auth, auth_headers, mocker):
        # Mock storage and extraction task
        mock_storage = mocker.MagicMock()
        mock_storage.upload.return_value = "sources/test-upload.pdf"
        mocker.patch(
            "agentic_project_service.routes.sources.get_storage",
            return_value=mock_storage,
        )
        mock_task = mocker.MagicMock()
        mock_task.id = "task-123"
        mocker.patch(
            "agentic_project_service.routes.sources.extract_source.delay",
            return_value=mock_task,
        )

        data = {
            "file": (io.BytesIO(b"fake pdf content"), "test.pdf"),
            "name": "Test Upload",
        }
        resp = client.post(
            "/api/sources/upload",
            data=data,
            headers={"Authorization": "Bearer fake-token"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201
        result = resp.get_json()
        assert result["name"] == "Test Upload"
        assert result["extraction_status"] == "pending"
        assert result["task_id"] == "task-123"

    def test_upload_no_file(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/sources/upload",
            headers={"Authorization": "Bearer fake-token"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_upload_unsupported_extension(self, client, mock_auth, auth_headers):
        data = {
            "file": (io.BytesIO(b"content"), "test.exe"),
        }
        resp = client.post(
            "/api/sources/upload",
            data=data,
            headers={"Authorization": "Bearer fake-token"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.get_json()["error"]

    def test_upload_with_metadata(self, client, mock_auth, auth_headers, mocker):
        mock_storage = mocker.MagicMock()
        mock_storage.upload.return_value = "sources/meta.txt"
        mocker.patch(
            "agentic_project_service.routes.sources.get_storage",
            return_value=mock_storage,
        )
        mock_task = mocker.MagicMock()
        mock_task.id = "task-456"
        mocker.patch(
            "agentic_project_service.routes.sources.extract_source.delay",
            return_value=mock_task,
        )

        data = {
            "file": (io.BytesIO(b"hello"), "test.txt"),
            "metadata": json.dumps({"author": "test"}),
        }
        resp = client.post(
            "/api/sources/upload",
            data=data,
            headers={"Authorization": "Bearer fake-token"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201


class TestUpdateSource:
    def test_update_name(self, client, mock_auth, auth_headers, test_source):
        resp = client.patch(
            f"/api/sources/{test_source['id']}",
            json={"name": "renamed.pdf"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "renamed.pdf"

    def test_update_metadata(self, client, mock_auth, auth_headers, test_source):
        resp = client.patch(
            f"/api/sources/{test_source['id']}",
            json={"metadata": {"tag": "important"}},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["metadata"]["tag"] == "important"

    def test_update_no_data(self, client, mock_auth, auth_headers, test_source):
        resp = client.patch(
            f"/api/sources/{test_source['id']}",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400


class TestDeleteSource:
    def test_delete(self, client, mock_auth, auth_headers, test_source, mocker):
        mock_storage = mocker.MagicMock()
        mocker.patch(
            "agentic_project_service.routes.sources.get_storage",
            return_value=mock_storage,
        )

        resp = client.delete(
            f"/api/sources/{test_source['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["message"] == "Source deleted"

    def test_delete_not_found(self, client, mock_auth, auth_headers, mocker):
        mock_storage = mocker.MagicMock()
        mocker.patch(
            "agentic_project_service.routes.sources.get_storage",
            return_value=mock_storage,
        )

        resp = client.delete(
            f"/api/sources/{uuid.uuid4()}",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestDeleteSourceWarning:
    def test_delete_indexed_source_returns_warning(
        self,
        client,
        mock_auth,
        auth_headers,
        test_source,
        test_indexed_source,
        mocker,
    ):
        mock_storage = mocker.MagicMock()
        mocker.patch(
            "agentic_project_service.routes.sources.get_storage",
            return_value=mock_storage,
        )

        resp = client.delete(
            f"/api/sources/{test_source['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "warning" in data
        assert "Test KB" in data["warning"]

    def test_delete_unindexed_source_no_warning(
        self, client, mock_auth, auth_headers, test_source, mocker
    ):
        mock_storage = mocker.MagicMock()
        mocker.patch(
            "agentic_project_service.routes.sources.get_storage",
            return_value=mock_storage,
        )

        resp = client.delete(
            f"/api/sources/{test_source['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "warning" not in data


class TestCancelExtraction:
    def test_cancel_pending(self, client, mock_auth, auth_headers, app, mocker):
        from agentic_project_service.db import db
        from sqlalchemy import text

        source_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".sources
                        (id, name, file_type, storage_path, extraction_status, celery_task_id)
                    VALUES (:id, 'pending.pdf', 'application/pdf', 'sources/p.pdf', 'pending', 'task-999')
                    """
                ),
                {"id": source_id},
            )
            db.session.commit()

        mocker.patch(
            "agentic_project_service.routes.sources.celery_app.control.revoke",
        )

        resp = client.post(
            f"/api/sources/{source_id}/cancel",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["message"] == "Extraction cancelled"

    def test_cancel_already_extracted(self, client, mock_auth, auth_headers, test_source):
        resp = client.post(
            f"/api/sources/{test_source['id']}/cancel",
            headers=auth_headers,
        )
        assert resp.status_code == 409


class TestAuthRequired:
    def test_no_token(self, client):
        resp = client.get("/api/sources")
        assert resp.status_code == 401


class TestUploadDeduplication:
    """Content-hash duplicate detection on POST /api/sources/upload."""

    def test_duplicate_upload_returns_409_with_existing_source(
        self, client, mock_auth, auth_headers, mocker
    ):
        mock_storage = mocker.MagicMock()
        mock_storage.upload.return_value = "sources/dup-test.pdf"
        mocker.patch(
            "agentic_project_service.routes.sources.get_storage",
            return_value=mock_storage,
        )
        mock_task = mocker.MagicMock()
        mock_task.id = "task-dup-1"
        mocker.patch(
            "agentic_project_service.routes.sources.extract_source.delay",
            return_value=mock_task,
        )

        file_bytes = b"hello world this is a test pdf"
        expected_hash = hashlib.sha256(file_bytes).hexdigest()

        # First upload: succeeds, returns 201
        r1 = client.post(
            "/api/sources/upload",
            data={"file": (io.BytesIO(file_bytes), "first.pdf")},
            headers={"Authorization": "Bearer fake-token"},
            content_type="multipart/form-data",
        )
        assert r1.status_code == 201, r1.get_json()
        first_id = r1.get_json()["id"]

        # Second upload of identical bytes: 409 with the first row in body
        r2 = client.post(
            "/api/sources/upload",
            data={"file": (io.BytesIO(file_bytes), "second-different-name.pdf")},
            headers={"Authorization": "Bearer fake-token"},
            content_type="multipart/form-data",
        )
        assert r2.status_code == 409, r2.get_json()
        body = r2.get_json()
        assert body["error"] == "duplicate_source"
        assert body["duplicate"]["id"] == first_id
        # content_hash must NOT be in the duplicate body
        assert "content_hash" not in body["duplicate"]
        # First-row hash should be persisted (internal sanity check via direct DB read)
        from agentic_project_service.db import db
        from sqlalchemy import text as sa_text

        row = db.session.execute(
            sa_text("SELECT content_hash FROM ai.sources WHERE id = :id"),
            {"id": first_id},
        ).fetchone()
        assert row[0] == expected_hash

    def test_duplicate_upload_does_not_call_storage_upload_twice(
        self, client, mock_auth, auth_headers, mocker
    ):
        mock_storage = mocker.MagicMock()
        mock_storage.upload.return_value = "sources/dup-test-2.pdf"
        mocker.patch(
            "agentic_project_service.routes.sources.get_storage",
            return_value=mock_storage,
        )
        mock_task = mocker.MagicMock()
        mock_task.id = "task-dup-2"
        mocker.patch(
            "agentic_project_service.routes.sources.extract_source.delay",
            return_value=mock_task,
        )

        file_bytes = b"another fixture, identical to itself"

        r1 = client.post(
            "/api/sources/upload",
            data={"file": (io.BytesIO(file_bytes), "x.pdf")},
            headers={"Authorization": "Bearer fake-token"},
            content_type="multipart/form-data",
        )
        assert r1.status_code == 201
        assert mock_storage.upload.call_count == 1

        r2 = client.post(
            "/api/sources/upload",
            data={"file": (io.BytesIO(file_bytes), "x.pdf")},
            headers={"Authorization": "Bearer fake-token"},
            content_type="multipart/form-data",
        )
        assert r2.status_code == 409
        # Storage.upload must NOT be called a second time — duplicate is caught
        # before touching storage.
        assert mock_storage.upload.call_count == 1


class TestImportFromStorageDeduplication:
    """Content-hash duplicate detection on POST /api/sources/import-from-storage."""

    def test_import_of_duplicate_bytes_returns_409(self, client, mock_auth, auth_headers, mocker):
        # Single storage mock used for both calls (upload path and import path).
        mock_storage = mocker.MagicMock()
        mock_storage.upload.return_value = "sources/from-storage.pdf"
        mock_storage.download.return_value = b"shared bytes for both paths"
        mocker.patch(
            "agentic_project_service.routes.sources.get_storage",
            return_value=mock_storage,
        )
        mock_task = mocker.MagicMock()
        mock_task.id = "task-import-dup"
        mocker.patch(
            "agentic_project_service.routes.sources.extract_source.delay",
            return_value=mock_task,
        )

        # First, upload the file via /upload
        r1 = client.post(
            "/api/sources/upload",
            data={"file": (io.BytesIO(b"shared bytes for both paths"), "u.pdf")},
            headers={"Authorization": "Bearer fake-token"},
            content_type="multipart/form-data",
        )
        assert r1.status_code == 201, r1.get_json()
        first_id = r1.get_json()["id"]

        # Then, import the "same" bytes via /import-from-storage
        r2 = client.post(
            "/api/sources/import-from-storage",
            json={"bucket": "user-bucket", "path": "folder/dup.pdf"},
            headers={"Authorization": "Bearer fake-token", "Content-Type": "application/json"},
        )
        assert r2.status_code == 409, r2.get_json()
        body = r2.get_json()
        assert body["error"] == "duplicate_source"
        assert body["duplicate"]["id"] == first_id
        assert "content_hash" not in body["duplicate"]


class TestImportUrlNotAffectedByContentDedup:
    """/import-url must not check content_hash. URLs are deduped by normalized URL."""

    def test_url_imports_dont_set_content_hash(
        self, client, mock_auth, auth_headers, mocker, monkeypatch
    ):
        # FIRECRAWL_API_KEY is platform-paid env-injected (not per-project setting).
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-api-key")
        mock_task = mocker.MagicMock()
        mock_task.id = "task-url"
        mocker.patch(
            "agentic_project_service.routes.sources.extract_url_source.delay",
            return_value=mock_task,
        )
        mocker.patch(
            "agentic_project_service.routes.sources._validate_url",
            return_value=True,
        )
        mocker.patch(
            "agentic_project_service.routes.sources.get_setting",
            side_effect=lambda key: {
                "URL_IMPORT_MAX_PAGES": 100,
                "FIRECRAWL_API_BASE": "https://api.firecrawl.dev",
            }.get(key),
        )
        mocker.patch(
            "agentic_project_service.routes.sources.get_all_user_provider_keys",
            return_value={},
        )

        r = client.post(
            "/api/sources/import-url",
            json={"mode": "urls", "urls": ["https://example.com/a"]},
            headers={"Authorization": "Bearer fake-token", "Content-Type": "application/json"},
        )
        assert r.status_code == 201, r.get_json()
        created = r.get_json()["sources"]
        assert len(created) == 1

        # Verify the row has NULL content_hash — URL imports don't populate it.
        from agentic_project_service.db import db
        from sqlalchemy import text as sa_text

        row = db.session.execute(
            sa_text("SELECT content_hash FROM ai.sources WHERE id = :id"),
            {"id": created[0]["id"]},
        ).fetchone()
        assert row[0] is None


class TestImportUrlReturns503OnMissingFirecrawlKey:
    """FIRECRAWL_API_KEY is platform-paid env-injected. When it's missing
    from the pod env (AWS SM seed gap, ESO sync lag), POST /import-url must
    return 503 — not 400. 400 implies the tenant did something wrong; under
    the platform-paid model the tenant cannot fix this, so the status code
    must signal operator-side unavailability.

    Counterfactual: revert sources.py:640 from 503 back to 400 → this test
    fails.
    """

    def test_503_status_on_missing_env(self, client, mock_auth, auth_headers, mocker, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        # _validate_url + URL_IMPORT_MAX_PAGES setting still need to be
        # mocked so the pre-validation code path doesn't error before the
        # env check fires.
        mocker.patch(
            "agentic_project_service.routes.sources._validate_url",
            return_value=True,
        )
        mocker.patch(
            "agentic_project_service.routes.sources.get_setting",
            side_effect=lambda key: {"URL_IMPORT_MAX_PAGES": 100}.get(key),
        )

        resp = client.post(
            "/api/sources/import-url",
            json={"mode": "urls", "urls": ["https://example.com/a"]},
            headers={"Authorization": "Bearer fake-token", "Content-Type": "application/json"},
        )

        assert resp.status_code == 503, resp.get_json()
        # The error message must NOT direct tenants at Studio Settings.
        body = resp.get_json()
        assert "error" in body
        assert "Settings > Tools" not in body["error"]


class TestUploadDedupRaceRecovery:
    """The DB UNIQUE constraint is the authoritative race-safety net. If the
    pre-check SELECT misses (concurrent INSERT wins between our SELECT and
    our INSERT), the IntegrityError handler must delete the just-uploaded
    storage object and return 409.
    """

    def test_integrity_error_cleans_up_storage_and_returns_409(
        self, client, mock_auth, auth_headers, mocker
    ):
        mock_storage = mocker.MagicMock()
        mock_storage.upload.return_value = "sources/race.pdf"
        mocker.patch(
            "agentic_project_service.routes.sources.get_storage",
            return_value=mock_storage,
        )
        mock_task = mocker.MagicMock()
        mock_task.id = "task-race"
        mocker.patch(
            "agentic_project_service.routes.sources.extract_source.delay",
            return_value=mock_task,
        )

        file_bytes = b"racing bytes"

        # First upload: real, lands in DB normally.
        r1 = client.post(
            "/api/sources/upload",
            data={"file": (io.BytesIO(file_bytes), "first.pdf")},
            headers={"Authorization": "Bearer fake-token"},
            content_type="multipart/form-data",
        )
        assert r1.status_code == 201, r1.get_json()

        # Second upload: simulate the race by making the pre-check SELECT
        # return None (as if the first INSERT hadn't committed yet), so the
        # route proceeds to upload + INSERT, where the real UNIQUE index
        # fires IntegrityError.
        from agentic_project_service.db import db

        original_execute = db.session.execute
        select_seen = {"count": 0}

        def fake_execute(stmt, params=None):
            sql = str(stmt)
            # Spoof only the pre-check on the second upload so the route
            # proceeds to upload + INSERT and the real UNIQUE index fires.
            if "content_hash = :h" in sql and "LIMIT 1" in sql:
                select_seen["count"] += 1
                if select_seen["count"] == 1:
                    return mocker.MagicMock(fetchone=lambda: None)
            return original_execute(stmt, params)

        mocker.patch.object(db.session, "execute", side_effect=fake_execute)

        r2 = client.post(
            "/api/sources/upload",
            data={"file": (io.BytesIO(file_bytes), "second.pdf")},
            headers={"Authorization": "Bearer fake-token"},
            content_type="multipart/form-data",
        )
        assert r2.status_code == 409, r2.get_json()
        assert r2.get_json()["error"] == "duplicate_source"
        mock_storage.delete.assert_called_once_with("sources", mocker.ANY)
        _, paths_arg = mock_storage.delete.call_args.args
        assert len(paths_arg) == 1


class TestUploadNonDedupIntegrityErrorCleanup:
    """If storage.upload() succeeds but the INSERT fails for a non-dedup
    reason (e.g., a future NOT NULL or CHECK violation), the storage object
    must NOT leak — the outer except Exception path runs _safe_cleanup_storage.
    """

    def test_non_dedup_integrity_error_returns_500_and_deletes_storage_object(
        self, client, mock_auth, auth_headers, mocker
    ):
        from sqlalchemy.exc import IntegrityError

        mock_storage = mocker.MagicMock()
        mock_storage.upload.return_value = "sources/non-dedup.pdf"
        mocker.patch(
            "agentic_project_service.routes.sources.get_storage",
            return_value=mock_storage,
        )

        # Spoof the INSERT to raise an IntegrityError whose constraint_name
        # is NOT the dedup unique index, so the route's constraint check
        # re-raises and the outer 500 handler runs the storage cleanup.
        from agentic_project_service.db import db

        original_execute = db.session.execute

        def fake_execute(stmt, params=None):
            sql = str(stmt)
            if "INSERT INTO" in sql and "sources" in sql:

                class _FakeDiag:
                    constraint_name = "sources_some_other_constraint"

                class _FakeOrig(Exception):
                    diag = _FakeDiag()

                    def __str__(self):
                        return "null value in column violates not-null constraint"

                orig = _FakeOrig()
                raise IntegrityError(statement=sql, params=params, orig=orig)
            return original_execute(stmt, params)

        mocker.patch.object(db.session, "execute", side_effect=fake_execute)

        r = client.post(
            "/api/sources/upload",
            data={"file": (io.BytesIO(b"non-dedup payload"), "non-dedup.pdf")},
            headers={"Authorization": "Bearer fake-token"},
            content_type="multipart/form-data",
        )
        # The route should 500 (non-dedup integrity violation propagates).
        assert r.status_code == 500, r.get_json()

        # Body is the redacted generic error, NOT the raw psycopg text.
        # Constraint names, schema names, parameter binds, etc. would all
        # leak to the client if str(exc) were serialized.
        body = r.get_json()
        assert body == {"error": "Internal error"}
        assert "null value" not in str(body)
        assert "constraint" not in str(body).lower()

        # And — the orphan-prevention safety net must have deleted the
        # just-uploaded storage object exactly once with a list-shaped path.
        mock_storage.delete.assert_called_once()
        bucket_arg, paths_arg = mock_storage.delete.call_args.args
        assert bucket_arg == "sources"
        assert isinstance(paths_arg, list)
        assert len(paths_arg) == 1


class TestImportFromStorageNonDedupIntegrityErrorCleanup:
    """Mirror of TestUploadNonDedupIntegrityErrorCleanup for the
    /import-from-storage route. Same redaction + cleanup contract: the
    body must be the generic Internal error, NOT the raw psycopg text,
    and the just-uploaded storage object must be deleted.
    """

    def test_non_dedup_integrity_error_returns_500_and_deletes_storage_object(
        self, client, mock_auth, mocker
    ):
        from sqlalchemy.exc import IntegrityError

        mock_storage = mocker.MagicMock()
        mock_storage.upload.return_value = "sources/import-non-dedup.pdf"
        mock_storage.download.return_value = b"non-dedup import payload"
        mocker.patch(
            "agentic_project_service.routes.sources.get_storage",
            return_value=mock_storage,
        )

        from agentic_project_service.db import db

        original_execute = db.session.execute

        def fake_execute(stmt, params=None):
            sql = str(stmt)
            if "INSERT INTO" in sql and "sources" in sql:

                class _FakeDiag:
                    constraint_name = "sources_some_other_constraint"

                class _FakeOrig(Exception):
                    diag = _FakeDiag()

                    def __str__(self):
                        return "null value in column violates not-null constraint"

                raise IntegrityError(statement=sql, params=params, orig=_FakeOrig())
            return original_execute(stmt, params)

        mocker.patch.object(db.session, "execute", side_effect=fake_execute)

        r = client.post(
            "/api/sources/import-from-storage",
            json={"bucket": "user-bucket", "path": "folder/dup.pdf"},
            headers={"Authorization": "Bearer fake-token", "Content-Type": "application/json"},
        )
        assert r.status_code == 500, r.get_json()

        # Body is the redacted generic error — same contract as /upload.
        body = r.get_json()
        assert body == {"error": "Internal error"}
        assert "null value" not in str(body)
        assert "constraint" not in str(body).lower()

        # Storage cleanup must have run.
        mock_storage.delete.assert_called_once()
        bucket_arg, paths_arg = mock_storage.delete.call_args.args
        assert bucket_arg == "sources"
        assert isinstance(paths_arg, list)
        assert len(paths_arg) == 1
