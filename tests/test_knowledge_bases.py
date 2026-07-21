"""Tests for knowledge base CRUD routes."""

import json
import uuid

from sqlalchemy import text


class TestCreateKnowledgeBase:
    def test_create(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/knowledge-bases",
            json={"name": "My KB"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["name"] == "My KB"
        assert "id" in data
        assert data["indexing_config"]["strategy"] == "chunk_embed"

    def test_create_with_description(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/knowledge-bases",
            json={"name": "Docs KB", "description": "Documentation"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.get_json()["description"] == "Documentation"

    def test_create_with_custom_config(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/knowledge-bases",
            json={
                "name": "Custom KB",
                "indexing_config": {"strategy": "chunk_embed", "chunk_size": 1000},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["indexing_config"]["chunk_size"] == 1000

    def test_create_missing_name(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/knowledge-bases",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400


class TestListKnowledgeBases:
    def test_list(self, client, mock_auth, auth_headers, test_knowledge_base):
        resp = client.get("/api/knowledge-bases", headers=auth_headers)
        assert resp.status_code == 200
        kbs = resp.get_json()["knowledge_bases"]
        assert any(kb["id"] == test_knowledge_base["id"] for kb in kbs)

    def test_list_empty(self, client, mock_auth, auth_headers):
        resp = client.get("/api/knowledge-bases", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["knowledge_bases"] == []

    def test_list_includes_counts(self, client, mock_auth, auth_headers, test_knowledge_base):
        resp = client.get("/api/knowledge-bases", headers=auth_headers)
        kb = resp.get_json()["knowledge_bases"][0]
        # Response shape changed in 2026-05-21 KB-list pagination work:
        # top-level `source_count` (total only) was replaced with `source_counts`
        # (status breakdown). `chunk_count` stayed at top level.
        assert "source_counts" in kb
        assert "total" in kb["source_counts"]
        assert "chunk_count" in kb


class TestGetKnowledgeBase:
    def test_get_returns_metadata(self, client, mock_auth, auth_headers, test_knowledge_base):
        resp = client.get(
            f"/api/knowledge-bases/{test_knowledge_base['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] == test_knowledge_base["id"]
        assert data["name"] == test_knowledge_base["name"]

    def test_get_does_not_return_indexed_sources_array(
        self, client, mock_auth, auth_headers, test_knowledge_base
    ):
        resp = client.get(
            f"/api/knowledge-bases/{test_knowledge_base['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        # Old shape returned a full array; new shape must not.
        assert "indexed_sources" not in data

    def test_get_returns_source_counts_with_zero_when_empty(
        self, client, mock_auth, auth_headers, test_knowledge_base
    ):
        resp = client.get(
            f"/api/knowledge-bases/{test_knowledge_base['id']}",
            headers=auth_headers,
        )
        data = resp.get_json()
        assert "source_counts" in data
        counts = data["source_counts"]
        for k in ("indexed", "failed", "pending", "indexing", "cancelled", "total"):
            assert k in counts
            assert counts[k] == 0

    def test_get_returns_source_counts_aggregated_by_status(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        from agentic_project_service.db import db

        kb_id = test_knowledge_base["id"]
        # Insert two sources and matching indexed_source rows with different statuses.
        with app.app_context():
            for status in ("indexed", "indexed", "failed", "pending"):
                src_id = str(uuid.uuid4())
                db.session.execute(
                    text(
                        """
                        INSERT INTO "ai".sources (id, name, file_type, storage_path, extraction_status)
                        VALUES (:id, :name, 'application/pdf', 'sources/x.pdf', 'extracted')
                        """
                    ),
                    {"id": src_id, "name": f"src-{status}.pdf"},
                )
                db.session.execute(
                    text(
                        """
                        INSERT INTO "ai".indexed_sources (id, source_id, knowledge_base_id, index_status)
                        VALUES (:id, :src_id, :kb_id, :status)
                        """
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "src_id": src_id,
                        "kb_id": kb_id,
                        "status": status,
                    },
                )
            db.session.commit()

        resp = client.get(f"/api/knowledge-bases/{kb_id}", headers=auth_headers)
        counts = resp.get_json()["source_counts"]
        assert counts["indexed"] == 2
        assert counts["failed"] == 1
        assert counts["pending"] == 1
        assert counts["indexing"] == 0
        assert counts["total"] == 4

    def test_get_not_found(self, client, mock_auth, auth_headers):
        resp = client.get(
            f"/api/knowledge-bases/{uuid.uuid4()}",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestUpdateKnowledgeBase:
    def test_update_name(self, client, mock_auth, auth_headers, test_knowledge_base):
        resp = client.patch(
            f"/api/knowledge-bases/{test_knowledge_base['id']}",
            json={"name": "Renamed KB"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "Renamed KB"

    def test_update_description(self, client, mock_auth, auth_headers, test_knowledge_base):
        resp = client.patch(
            f"/api/knowledge-bases/{test_knowledge_base['id']}",
            json={"description": "Updated desc"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["description"] == "Updated desc"

    def test_update_no_data(self, client, mock_auth, auth_headers, test_knowledge_base):
        resp = client.patch(
            f"/api/knowledge-bases/{test_knowledge_base['id']}",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400


class TestDeleteKnowledgeBase:
    def test_delete(self, client, mock_auth, auth_headers, test_knowledge_base, mocker):
        # Mock the enrichment cleanup (lazy import inside delete_knowledge_base)
        mocker.patch(
            "agentic_project_service.services.metadata_enricher.MetadataEnricher",
        )
        resp = client.delete(
            f"/api/knowledge-bases/{test_knowledge_base['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["message"] == "Knowledge base deleted"


class TestDeleteKBWarning:
    def test_delete_kb_with_agent_returns_warning(
        self, client, mock_auth, auth_headers, test_knowledge_base, app, mocker
    ):
        mocker.patch(
            "agentic_project_service.services.metadata_enricher.MetadataEnricher",
        )
        # Create an agent that references this KB in its settings
        kb_id = test_knowledge_base["id"]
        agent_id = str(uuid.uuid4())
        from agentic_project_service.db import db

        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".agents (id, name, model, settings)
                    VALUES (:id, 'KB Agent', 'gpt-4o-mini', :settings)
                    """
                ),
                {
                    "id": agent_id,
                    "settings": json.dumps({"knowledge_base_id": kb_id}),
                },
            )
            db.session.commit()

        resp = client.delete(
            f"/api/knowledge-bases/{kb_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "warning" in data
        assert "KB Agent" in data["warning"]

    def test_delete_kb_no_agent_no_warning(
        self, client, mock_auth, auth_headers, test_knowledge_base, mocker
    ):
        mocker.patch(
            "agentic_project_service.services.metadata_enricher.MetadataEnricher",
        )
        resp = client.delete(
            f"/api/knowledge-bases/{test_knowledge_base['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "warning" not in data


class TestAddSourceToKB:
    def test_add_source(
        self, client, mock_auth, auth_headers, test_knowledge_base, test_source, mocker
    ):
        mock_task = mocker.MagicMock()
        mock_task.id = "idx-task-1"
        mocker.patch(
            "agentic_project_service.routes.knowledge_bases.index_source.delay",
            return_value=mock_task,
        )

        resp = client.post(
            f"/api/knowledge-bases/{test_knowledge_base['id']}/sources",
            json={"source_id": test_source["id"]},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["source_id"] == test_source["id"]
        assert data["index_status"] == "pending"
        assert data["task_id"] == "idx-task-1"

    def test_add_source_missing_id(self, client, mock_auth, auth_headers, test_knowledge_base):
        resp = client.post(
            f"/api/knowledge-bases/{test_knowledge_base['id']}/sources",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_add_source_not_extracted(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        """Can't index a source that hasn't been extracted yet."""
        from agentic_project_service.db import db
        from sqlalchemy import text

        source_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".sources
                        (id, name, file_type, storage_path, extraction_status)
                    VALUES (:id, 'pending.pdf', 'application/pdf', 'sources/p.pdf', 'pending')
                    """
                ),
                {"id": source_id},
            )
            db.session.commit()

        resp = client.post(
            f"/api/knowledge-bases/{test_knowledge_base['id']}/sources",
            json={"source_id": source_id},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "extracted" in resp.get_json()["error"]


class TestRemoveSourceFromKB:
    def _seed_source_and_indexed(self, app, kb_id, status="indexed", celery_task_id=None):
        """Insert a (source, indexed_source) pair. Returns (source_id, indexed_source_id)."""
        from agentic_project_service.db import db

        source_id = str(uuid.uuid4())
        indexed_source_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".sources (id, name, file_type, storage_path, extraction_status)
                    VALUES (:id, :name, 'application/pdf', 'sources/x.pdf', 'extracted')
                    """
                ),
                {"id": source_id, "name": "test-source.pdf"},
            )
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".indexed_sources
                        (id, source_id, knowledge_base_id, index_status, celery_task_id)
                    VALUES (:id, :src_id, :kb_id, :status, :task_id)
                    """
                ),
                {
                    "id": indexed_source_id,
                    "src_id": source_id,
                    "kb_id": kb_id,
                    "status": status,
                    "task_id": celery_task_id,
                },
            )
            db.session.commit()
        return source_id, indexed_source_id

    def test_remove_source_succeeds(
        self, app, client, mock_auth, auth_headers, test_knowledge_base
    ):
        kb_id = test_knowledge_base["id"]
        source_id, indexed_source_id = self._seed_source_and_indexed(app, kb_id, status="indexed")

        resp = client.delete(
            f"/api/knowledge-bases/{kb_id}/sources/{indexed_source_id}",
            headers=auth_headers,
        )

        assert resp.status_code == 200, resp.get_data(as_text=True)
        data = resp.get_json()
        assert data["message"] == "Source removed from knowledge base"
        assert data["deleted_indexed_source_id"] == indexed_source_id
        assert data["kb_id"] == kb_id

        # indexed_source row is gone
        from agentic_project_service.db import db

        with app.app_context():
            row = db.session.execute(
                text('SELECT id FROM "ai".indexed_sources WHERE id = :id'),
                {"id": indexed_source_id},
            ).fetchone()
            assert row is None

            # underlying source row remains
            src_row = db.session.execute(
                text('SELECT id FROM "ai".sources WHERE id = :id'),
                {"id": source_id},
            ).fetchone()
            assert src_row is not None

    def test_remove_source_cascades_to_chunks(
        self, app, client, mock_auth, auth_headers, test_knowledge_base
    ):
        kb_id = test_knowledge_base["id"]
        source_id, indexed_source_id = self._seed_source_and_indexed(app, kb_id, status="indexed")

        from agentic_project_service.db import db

        chunk_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".chunks (id, indexed_source_id, knowledge_base_id, source_id, text, chunk_index)
                    VALUES (:id, :is_id, :kb_id, :src_id, 'test chunk content', 0)
                    """
                ),
                {"id": chunk_id, "is_id": indexed_source_id, "kb_id": kb_id, "src_id": source_id},
            )
            db.session.commit()

        resp = client.delete(
            f"/api/knowledge-bases/{kb_id}/sources/{indexed_source_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.get_data(as_text=True)

        with app.app_context():
            chunk_row = db.session.execute(
                text('SELECT id FROM "ai".chunks WHERE id = :id'),
                {"id": chunk_id},
            ).fetchone()
            assert chunk_row is None, "chunks row should have been cascaded away"

    def test_remove_source_revokes_celery_when_indexing(
        self, app, client, mocker, mock_auth, auth_headers, test_knowledge_base
    ):
        kb_id = test_knowledge_base["id"]
        source_id, indexed_source_id = self._seed_source_and_indexed(
            app, kb_id, status="indexing", celery_task_id="task-xyz"
        )
        revoke_mock = mocker.patch(
            "agentic_project_service.routes.knowledge_bases.celery_app.control.revoke"
        )

        resp = client.delete(
            f"/api/knowledge-bases/{kb_id}/sources/{indexed_source_id}",
            headers=auth_headers,
        )

        assert resp.status_code == 200
        revoke_mock.assert_called_once_with("task-xyz", terminate=True)

    def test_remove_source_revokes_celery_when_pending(
        self, app, client, mocker, mock_auth, auth_headers, test_knowledge_base
    ):
        kb_id = test_knowledge_base["id"]
        source_id, indexed_source_id = self._seed_source_and_indexed(
            app, kb_id, status="pending", celery_task_id="task-abc"
        )
        revoke_mock = mocker.patch(
            "agentic_project_service.routes.knowledge_bases.celery_app.control.revoke"
        )

        resp = client.delete(
            f"/api/knowledge-bases/{kb_id}/sources/{indexed_source_id}",
            headers=auth_headers,
        )

        assert resp.status_code == 200
        revoke_mock.assert_called_once_with("task-abc", terminate=True)

    def test_remove_source_when_revoke_raises_still_deletes(
        self, app, client, mocker, mock_auth, auth_headers, test_knowledge_base
    ):
        kb_id = test_knowledge_base["id"]
        source_id, indexed_source_id = self._seed_source_and_indexed(
            app, kb_id, status="indexing", celery_task_id="task-bad"
        )
        mocker.patch(
            "agentic_project_service.routes.knowledge_bases.celery_app.control.revoke",
            side_effect=RuntimeError("broker down"),
        )

        resp = client.delete(
            f"/api/knowledge-bases/{kb_id}/sources/{indexed_source_id}",
            headers=auth_headers,
        )

        assert resp.status_code == 200
        from agentic_project_service.db import db

        with app.app_context():
            row = db.session.execute(
                text('SELECT id FROM "ai".indexed_sources WHERE id = :id'),
                {"id": indexed_source_id},
            ).fetchone()
            assert row is None, "row should be deleted even when revoke fails"

    def test_remove_source_404_when_not_found(
        self, app, client, mock_auth, auth_headers, test_knowledge_base
    ):
        kb_id = test_knowledge_base["id"]
        bogus_id = str(uuid.uuid4())

        resp = client.delete(
            f"/api/knowledge-bases/{kb_id}/sources/{bogus_id}",
            headers=auth_headers,
        )

        assert resp.status_code == 404
        assert resp.get_json()["error"] == "Indexed source not found"

    def test_remove_source_404_when_belongs_to_other_kb(
        self, app, client, mock_auth, auth_headers, test_knowledge_base
    ):
        kb_a_id = test_knowledge_base["id"]
        # create a second KB
        resp = client.post(
            "/api/knowledge-bases",
            json={"name": "Other KB"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        kb_b_id = resp.get_json()["id"]

        # indexed_source belongs to KB-A
        _src, indexed_source_id = self._seed_source_and_indexed(app, kb_a_id, status="indexed")

        # request DELETE via KB-B's id — should be 404
        resp = client.delete(
            f"/api/knowledge-bases/{kb_b_id}/sources/{indexed_source_id}",
            headers=auth_headers,
        )

        assert resp.status_code == 404

        # KB-A's indexed_source row should remain
        from agentic_project_service.db import db

        with app.app_context():
            row = db.session.execute(
                text('SELECT id FROM "ai".indexed_sources WHERE id = :id'),
                {"id": indexed_source_id},
            ).fetchone()
            assert row is not None, "row under KB-A must not be deleted"

    def test_remove_source_400_on_bad_uuid_kb(self, client, mock_auth, auth_headers):
        resp = client.delete(
            f"/api/knowledge-bases/not-a-uuid/sources/{uuid.uuid4()}",
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_remove_source_400_on_bad_uuid_indexed_source(
        self, client, mock_auth, auth_headers, test_knowledge_base
    ):
        kb_id = test_knowledge_base["id"]
        resp = client.delete(
            f"/api/knowledge-bases/{kb_id}/sources/not-a-uuid",
            headers=auth_headers,
        )
        assert resp.status_code == 400


class TestAuthRequired:
    def test_no_token(self, client):
        resp = client.get("/api/knowledge-bases")
        assert resp.status_code == 401


class TestKBDriftIndicator:
    def _set_kb_config(self, app, kb_id, config: dict):
        from agentic_project_service.db import db

        with app.app_context():
            db.session.execute(
                text(
                    """UPDATE "ai".knowledge_bases SET indexing_config = CAST(:c AS jsonb) WHERE id = :id"""
                ),
                {"c": json.dumps(config), "id": kb_id},
            )
            db.session.commit()

    def _insert_indexed_row(self, app, kb_id, status, snapshot: dict | None):
        from agentic_project_service.db import db

        src_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".sources (id, name, file_type, storage_path, extraction_status)
                    VALUES (:id, 'x.pdf', 'application/pdf', 'sources/x.pdf', 'extracted')
                    """
                ),
                {"id": src_id},
            )
            db.session.execute(
                text(
                    """
                    INSERT INTO "ai".indexed_sources (id, source_id, knowledge_base_id, index_status, indexing_config_snapshot)
                    VALUES (:id, :src, :kb, :st, CAST(:snap AS jsonb))
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "src": src_id,
                    "kb": kb_id,
                    "st": status,
                    "snap": json.dumps(snapshot) if snapshot is not None else None,
                },
            )
            db.session.commit()

    def test_drift_none_when_no_indexed_rows(
        self, client, mock_auth, auth_headers, test_knowledge_base
    ):
        resp = client.get(f"/api/knowledge-bases/{test_knowledge_base['id']}", headers=auth_headers)
        assert resp.get_json()["drift"] == "none"

    def test_drift_none_when_snapshot_matches(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        kb_id = test_knowledge_base["id"]
        cfg = {"strategy": "chunk_embed", "chunk_size": 1000, "overlap": 200}
        self._set_kb_config(app, kb_id, cfg)
        self._insert_indexed_row(app, kb_id, "indexed", cfg)

        resp = client.get(f"/api/knowledge-bases/{kb_id}", headers=auth_headers)
        assert resp.get_json()["drift"] == "none"

    def test_drift_full_for_chunk_embed_when_snapshot_differs(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        kb_id = test_knowledge_base["id"]
        self._set_kb_config(app, kb_id, {"strategy": "chunk_embed", "chunk_size": 1500})
        self._insert_indexed_row(
            app, kb_id, "indexed", {"strategy": "chunk_embed", "chunk_size": 1000}
        )

        resp = client.get(f"/api/knowledge-bases/{kb_id}", headers=auth_headers)
        assert resp.get_json()["drift"] == "full"

    def test_drift_ignores_pending_and_indexing_rows(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        kb_id = test_knowledge_base["id"]
        self._set_kb_config(app, kb_id, {"strategy": "chunk_embed", "chunk_size": 1500})
        # Drifty snapshot but row is still in flight — should not count.
        self._insert_indexed_row(
            app, kb_id, "pending", {"strategy": "chunk_embed", "chunk_size": 1000}
        )
        self._insert_indexed_row(
            app, kb_id, "indexing", {"strategy": "chunk_embed", "chunk_size": 1000}
        )

        resp = client.get(f"/api/knowledge-bases/{kb_id}", headers=auth_headers)
        assert resp.get_json()["drift"] == "none"

    def test_drift_enrichment_only_when_only_enrichment_model_differs(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        kb_id = test_knowledge_base["id"]
        cur = {
            "strategy": "graph_index",
            "chunk_size": 1000,
            "enrichment_model": "gpt-5-mini",
            "embedding_model": "text-embedding-3-small",
        }
        snap = {**cur, "enrichment_model": "gpt-5-nano"}
        self._set_kb_config(app, kb_id, cur)
        self._insert_indexed_row(app, kb_id, "indexed", snap)

        resp = client.get(f"/api/knowledge-bases/{kb_id}", headers=auth_headers)
        assert resp.get_json()["drift"] == "enrichment_only"

    def test_drift_full_for_graph_index_when_non_enrichment_field_differs(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        kb_id = test_knowledge_base["id"]
        cur = {
            "strategy": "graph_index",
            "chunk_size": 1500,
            "enrichment_model": "gpt-5-mini",
            "embedding_model": "text-embedding-3-small",
        }
        snap = {**cur, "chunk_size": 1000}  # non-enrichment field differs
        self._set_kb_config(app, kb_id, cur)
        self._insert_indexed_row(app, kb_id, "indexed", snap)

        resp = client.get(f"/api/knowledge-bases/{kb_id}", headers=auth_headers)
        assert resp.get_json()["drift"] == "full"


class TestListIndexedSources:
    @staticmethod
    def _seed(app, kb_id: str, rows: list[dict]):
        """Seed indexed_source rows with paired sources. Each row: {name, status, created_at?}."""
        from agentic_project_service.db import db

        with app.app_context():
            for r in rows:
                src_id = str(uuid.uuid4())
                db.session.execute(
                    text(
                        """
                        INSERT INTO "ai".sources (id, name, file_type, storage_path, extraction_status, created_at)
                        VALUES (:id, :name, 'application/pdf', 'sources/x.pdf', 'extracted',
                                COALESCE(CAST(:created_at AS timestamptz), NOW()))
                        """
                    ),
                    {"id": src_id, "name": r["name"], "created_at": r.get("created_at")},
                )
                db.session.execute(
                    text(
                        """
                        INSERT INTO "ai".indexed_sources (id, source_id, knowledge_base_id, index_status)
                        VALUES (:id, :src, :kb, :st)
                        """
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "src": src_id,
                        "kb": kb_id,
                        "st": r["status"],
                    },
                )
            db.session.commit()

    def test_basic_shape_returns_items_total_limit_offset(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        kb_id = test_knowledge_base["id"]
        self._seed(app, kb_id, [{"name": "a.pdf", "status": "indexed"}])

        resp = client.get(f"/api/knowledge-bases/{kb_id}/sources", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "items" in data and isinstance(data["items"], list)
        assert "total" in data and isinstance(data["total"], int)
        assert data["limit"] == 50
        assert data["offset"] == 0
        assert data["total"] == 1
        assert len(data["items"]) == 1
        item = data["items"][0]
        for k in (
            "id",
            "source_id",
            "index_status",
            "indexed_at",
            "stats",
            "error_message",
            "source_name",
            "file_type",
            "source_created_at",
        ):
            assert k in item
        # Per-row indexing_config_snapshot should NOT be returned (drift is server-side).
        assert "indexing_config_snapshot" not in item

    def test_kb_not_found_returns_404(self, client, mock_auth, auth_headers):
        resp = client.get(
            f"/api/knowledge-bases/{uuid.uuid4()}/sources",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_invalid_kb_id_returns_404(self, client, mock_auth, auth_headers):
        # _require_uuid currently returns 404 for malformed UUIDs (used across
        # all KB routes for consistency). REST-conventional 400 would be a
        # cross-cutting fix in a follow-up.
        resp = client.get("/api/knowledge-bases/not-a-uuid/sources", headers=auth_headers)
        assert resp.status_code == 404

    def test_filter_by_status(self, client, mock_auth, auth_headers, test_knowledge_base, app):
        kb_id = test_knowledge_base["id"]
        self._seed(
            app,
            kb_id,
            [
                {"name": "a.pdf", "status": "indexed"},
                {"name": "b.pdf", "status": "indexed"},
                {"name": "c.pdf", "status": "failed"},
            ],
        )
        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/sources?status=failed",
            headers=auth_headers,
        )
        data = resp.get_json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["source_name"] == "c.pdf"
        assert data["items"][0]["index_status"] == "failed"

    def test_search_by_name_case_insensitive(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        kb_id = test_knowledge_base["id"]
        self._seed(
            app,
            kb_id,
            [
                {"name": "Annual Report 2025.pdf", "status": "indexed"},
                {"name": "tax-form-q3.pdf", "status": "indexed"},
                {"name": "annual-summary.pdf", "status": "indexed"},
            ],
        )
        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/sources?q=annual",
            headers=auth_headers,
        )
        data = resp.get_json()
        assert data["total"] == 2
        names = [i["source_name"] for i in data["items"]]
        assert "Annual Report 2025.pdf" in names
        assert "annual-summary.pdf" in names

    def test_search_combines_with_status_filter(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        kb_id = test_knowledge_base["id"]
        self._seed(
            app,
            kb_id,
            [
                {"name": "report-good.pdf", "status": "indexed"},
                {"name": "report-bad.pdf", "status": "failed"},
                {"name": "other-bad.pdf", "status": "failed"},
            ],
        )
        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/sources?q=report&status=failed",
            headers=auth_headers,
        )
        data = resp.get_json()
        assert data["total"] == 1
        assert data["items"][0]["source_name"] == "report-bad.pdf"

    def test_default_sort_pins_failed_first_then_created_at_desc(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        kb_id = test_knowledge_base["id"]
        self._seed(
            app,
            kb_id,
            [
                {
                    "name": "old-indexed.pdf",
                    "status": "indexed",
                    "created_at": "2026-01-01T00:00:00Z",
                },
                {
                    "name": "new-indexed.pdf",
                    "status": "indexed",
                    "created_at": "2026-04-01T00:00:00Z",
                },
                {
                    "name": "old-failed.pdf",
                    "status": "failed",
                    "created_at": "2025-06-01T00:00:00Z",
                },
                {
                    "name": "new-failed.pdf",
                    "status": "failed",
                    "created_at": "2026-03-01T00:00:00Z",
                },
            ],
        )
        resp = client.get(f"/api/knowledge-bases/{kb_id}/sources", headers=auth_headers)
        items = resp.get_json()["items"]
        # Failed first (newest of the failed at top), then indexed (newest first).
        assert [i["source_name"] for i in items] == [
            "new-failed.pdf",
            "old-failed.pdf",
            "new-indexed.pdf",
            "old-indexed.pdf",
        ]

    def test_explicit_sort_by_name_does_not_pin_failed(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        kb_id = test_knowledge_base["id"]
        self._seed(
            app,
            kb_id,
            [
                {"name": "alpha.pdf", "status": "indexed"},
                {"name": "bravo.pdf", "status": "failed"},
                {"name": "charlie.pdf", "status": "indexed"},
            ],
        )
        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/sources?sort=name&order=asc",
            headers=auth_headers,
        )
        items = resp.get_json()["items"]
        assert [i["source_name"] for i in items] == ["alpha.pdf", "bravo.pdf", "charlie.pdf"]

    def test_explicit_sort_by_created_at_asc(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        kb_id = test_knowledge_base["id"]
        self._seed(
            app,
            kb_id,
            [
                {"name": "older.pdf", "status": "indexed", "created_at": "2026-01-01T00:00:00Z"},
                {"name": "newer.pdf", "status": "indexed", "created_at": "2026-04-01T00:00:00Z"},
            ],
        )
        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/sources?sort=created_at&order=asc",
            headers=auth_headers,
        )
        items = resp.get_json()["items"]
        assert [i["source_name"] for i in items] == ["older.pdf", "newer.pdf"]

    def test_pagination_limit_offset(
        self, client, mock_auth, auth_headers, test_knowledge_base, app
    ):
        kb_id = test_knowledge_base["id"]
        self._seed(
            app,
            kb_id,
            [
                {
                    "name": f"src-{i:02d}.pdf",
                    "status": "indexed",
                    "created_at": f"2026-01-{(i % 28) + 1:02d}T00:00:00Z",
                }
                for i in range(5)
            ],
        )
        # Page 1 of 2 with limit=2
        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/sources?sort=name&order=asc&limit=2&offset=0",
            headers=auth_headers,
        )
        data = resp.get_json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        # Page 2
        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/sources?sort=name&order=asc&limit=2&offset=2",
            headers=auth_headers,
        )
        data = resp.get_json()
        assert len(data["items"]) == 2
        # No overlap between pages
        page1_names = {"src-00.pdf", "src-01.pdf"}
        page2_names = {i["source_name"] for i in data["items"]}
        assert page1_names.isdisjoint(page2_names)

    def test_limit_capped_at_200(self, client, mock_auth, auth_headers, test_knowledge_base):
        resp = client.get(
            f"/api/knowledge-bases/{test_knowledge_base['id']}/sources?limit=99999",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["limit"] == 200


class TestListKnowledgeBasesPagination:
    def _seed_kbs(self, app, count: int, name_prefix: str = "KB"):
        """Helper: insert `count` KBs into ai.knowledge_bases."""
        with app.app_context():
            from agentic_project_service.db import db

            for i in range(count):
                db.session.execute(
                    text("""
                        INSERT INTO ai.knowledge_bases (id, name, description, indexing_config, retrieval_config)
                        VALUES (gen_random_uuid(), :name, :desc, '{}'::jsonb, '{}'::jsonb)
                    """),
                    {"name": f"{name_prefix} {i:03d}", "desc": f"KB number {i}"},
                )
            db.session.commit()

    def test_default_returns_first_page_50(self, client, mock_auth, auth_headers, app):
        self._seed_kbs(app, 60)
        resp = client.get("/api/knowledge-bases", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "knowledge_bases" in data
        assert len(data["knowledge_bases"]) == 50
        assert data["total"] == 60
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_limit_and_offset(self, client, mock_auth, auth_headers, app):
        self._seed_kbs(app, 30)
        resp = client.get("/api/knowledge-bases?limit=10&offset=20", headers=auth_headers)
        data = resp.get_json()
        assert len(data["knowledge_bases"]) == 10
        assert data["total"] == 30
        assert data["limit"] == 10
        assert data["offset"] == 20

    def test_search_by_name(self, client, mock_auth, auth_headers, app):
        self._seed_kbs(app, 10, name_prefix="Alpha")
        self._seed_kbs(app, 10, name_prefix="Beta")
        resp = client.get("/api/knowledge-bases?q=Alpha", headers=auth_headers)
        data = resp.get_json()
        assert data["total"] == 10
        assert all("Alpha" in kb["name"] for kb in data["knowledge_bases"])

    def test_search_escapes_wildcards(self, client, mock_auth, auth_headers, app):
        with app.app_context():
            from agentic_project_service.db import db

            db.session.execute(
                text(
                    "INSERT INTO ai.knowledge_bases (id, name, indexing_config, retrieval_config) "
                    "VALUES (gen_random_uuid(), '50%_off', '{}'::jsonb, '{}'::jsonb)"
                )
            )
            db.session.execute(
                text(
                    "INSERT INTO ai.knowledge_bases (id, name, indexing_config, retrieval_config) "
                    "VALUES (gen_random_uuid(), 'foobar', '{}'::jsonb, '{}'::jsonb)"
                )
            )
            db.session.commit()
        # Searching for literal "%" via URL-encoded form should match only the
        # percent KB, not match "foobar" via wildcard.
        resp = client.get("/api/knowledge-bases?q=%25", headers=auth_headers)
        data = resp.get_json()
        names = {kb["name"] for kb in data["knowledge_bases"]}
        assert "50%_off" in names
        assert "foobar" not in names

    def test_sort_by_name(self, client, mock_auth, auth_headers, app):
        self._seed_kbs(app, 5)
        resp = client.get("/api/knowledge-bases?sort=name&order=asc", headers=auth_headers)
        data = resp.get_json()
        names = [kb["name"] for kb in data["knowledge_bases"]]
        assert names == sorted(names)

    def test_sort_by_created_at_desc_default(self, client, mock_auth, auth_headers, app):
        self._seed_kbs(app, 3)
        resp = client.get("/api/knowledge-bases", headers=auth_headers)
        data = resp.get_json()
        timestamps = [kb["created_at"] for kb in data["knowledge_bases"]]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_sort_invalid_returns_400(self, client, mock_auth, auth_headers):
        resp = client.get("/api/knowledge-bases?sort=bogus", headers=auth_headers)
        assert resp.status_code == 400
        assert "bogus" in resp.get_json()["error"]

    def test_limit_clamped(self, client, mock_auth, auth_headers, app):
        self._seed_kbs(app, 200)
        resp = client.get("/api/knowledge-bases?limit=999", headers=auth_headers)
        data = resp.get_json()
        assert data["limit"] == 100
        assert len(data["knowledge_bases"]) == 100


class TestListKnowledgeBasesAggregates:
    def test_response_includes_source_counts(self, client, mock_auth, auth_headers, app):
        with app.app_context():
            from agentic_project_service.db import db

            result = db.session.execute(
                text("""
                    INSERT INTO ai.knowledge_bases (id, name, indexing_config, retrieval_config)
                    VALUES (gen_random_uuid(), 'KB with sources', '{}'::jsonb, '{}'::jsonb)
                    RETURNING id
                """)
            )
            kb_id = result.scalar()
            # indexed_sources.source_id has FK to sources.id, AND
            # UNIQUE(knowledge_base_id, source_id) — each indexed_source row
            # needs its own distinct source row.
            for status in ("indexed", "indexed", "failed", "pending"):
                source_id = db.session.execute(
                    text("""
                        INSERT INTO ai.sources (id, name, file_type, storage_path, extraction_status)
                        VALUES (gen_random_uuid(), 'tester.pdf', 'application/pdf', 'sources/x.pdf', 'extracted')
                        RETURNING id
                    """)
                ).scalar()
                db.session.execute(
                    text("""
                        INSERT INTO ai.indexed_sources
                            (id, knowledge_base_id, source_id, index_status, indexing_config_snapshot)
                        VALUES (gen_random_uuid(), :kb_id, :source_id, :status, '{}'::jsonb)
                    """),
                    {"kb_id": kb_id, "source_id": source_id, "status": status},
                )
            db.session.commit()

        resp = client.get("/api/knowledge-bases", headers=auth_headers)
        data = resp.get_json()
        kb = next(k for k in data["knowledge_bases"] if k["name"] == "KB with sources")
        assert kb["source_counts"] == {
            "pending": 1,
            "indexing": 0,
            "indexed": 2,
            "failed": 1,
            "cancelled": 0,
            "total": 4,
        }

    def test_empty_kb_has_zero_source_counts(self, client, mock_auth, auth_headers, app):
        with app.app_context():
            from agentic_project_service.db import db

            db.session.execute(
                text(
                    "INSERT INTO ai.knowledge_bases (id, name, indexing_config, retrieval_config) "
                    "VALUES (gen_random_uuid(), 'Empty KB', '{}'::jsonb, '{}'::jsonb)"
                )
            )
            db.session.commit()
        resp = client.get("/api/knowledge-bases", headers=auth_headers)
        kb = next(k for k in resp.get_json()["knowledge_bases"] if k["name"] == "Empty KB")
        assert kb["source_counts"]["total"] == 0
        assert kb["chunk_count"] == 0

    def test_response_includes_enrichment_status_none_by_default(
        self, client, mock_auth, auth_headers, app
    ):
        with app.app_context():
            from agentic_project_service.db import db

            db.session.execute(
                text(
                    "INSERT INTO ai.knowledge_bases (id, name, indexing_config, retrieval_config) "
                    "VALUES (gen_random_uuid(), 'No enrichment KB', '{}'::jsonb, '{}'::jsonb)"
                )
            )
            db.session.commit()
        resp = client.get("/api/knowledge-bases", headers=auth_headers)
        kb = next(k for k in resp.get_json()["knowledge_bases"] if k["name"] == "No enrichment KB")
        assert kb["enrichment_status"] == "none"
        assert kb["enrichment_progress"] is None

    def test_enrichment_status_db_to_api_mapping(self, client, mock_auth, auth_headers, app):
        """DB stores idle/enriching/completed/completed_with_errors/failed.
        API spec promises none/enriching/enriched/failed. This verifies the
        mapping in routes/knowledge_bases.py:_ENRICHMENT_STATUS_MAP.
        """
        from agentic_project_service.db import db

        cases = [
            ("idle", "none"),
            ("enriching", "enriching"),
            ("completed", "enriched"),
            ("completed_with_errors", "enriched"),
            ("failed", "failed"),
        ]
        with app.app_context():
            for db_status, api_status in cases:
                kb_id = db.session.execute(
                    text(
                        "INSERT INTO ai.knowledge_bases (id, name, indexing_config, retrieval_config) "
                        "VALUES (gen_random_uuid(), :name, '{}'::jsonb, '{}'::jsonb) RETURNING id"
                    ),
                    {"name": f"KB-enrich-{db_status}"},
                ).scalar()
                db.session.execute(
                    text(
                        "INSERT INTO ai.enrichment_configs "
                        "(id, knowledge_base_id, fields, status, enriched_count, total_count, metadata_table_name) "
                        "VALUES (gen_random_uuid(), :kb_id, '[]'::jsonb, :status, 0, 0, :tbl)"
                    ),
                    {
                        "kb_id": kb_id,
                        "status": db_status,
                        "tbl": f"meta_{str(kb_id).replace('-', '_')}",
                    },
                )
            db.session.commit()

        resp = client.get("/api/knowledge-bases?q=KB-enrich-&limit=10", headers=auth_headers)
        data = resp.get_json()
        by_name = {kb["name"]: kb for kb in data["knowledge_bases"]}
        for db_status, api_status in cases:
            assert (
                by_name[f"KB-enrich-{db_status}"]["enrichment_status"] == api_status
            ), f"DB status {db_status!r} should map to API status {api_status!r}"

    def test_sort_stable_with_dup_created_at(self, client, mock_auth, auth_headers, app):
        """Two KBs with identical created_at should appear in a deterministic
        order across requests. Stable secondary sort is `kb.id ASC`.
        """
        from agentic_project_service.db import db

        with app.app_context():
            # Insert two KBs with explicit identical created_at.
            ts = "2026-05-21T12:00:00+00:00"
            db.session.execute(
                text(
                    "INSERT INTO ai.knowledge_bases "
                    "(id, name, indexing_config, retrieval_config, created_at) "
                    "VALUES (gen_random_uuid(), :n, '{}'::jsonb, '{}'::jsonb, :ts)"
                ),
                {"n": "DupTimeKB-A", "ts": ts},
            )
            db.session.execute(
                text(
                    "INSERT INTO ai.knowledge_bases "
                    "(id, name, indexing_config, retrieval_config, created_at) "
                    "VALUES (gen_random_uuid(), :n, '{}'::jsonb, '{}'::jsonb, :ts)"
                ),
                {"n": "DupTimeKB-B", "ts": ts},
            )
            db.session.commit()

        first = client.get(
            "/api/knowledge-bases?q=DupTimeKB&limit=10", headers=auth_headers
        ).get_json()
        second = client.get(
            "/api/knowledge-bases?q=DupTimeKB&limit=10", headers=auth_headers
        ).get_json()
        first_ids = [k["id"] for k in first["knowledge_bases"]]
        second_ids = [k["id"] for k in second["knowledge_bases"]]
        assert (
            first_ids == second_ids
        ), "Stable sort by kb.id must produce same order across requests"
        assert len(first_ids) == 2


class TestAvailableSources:
    """GET /<kb_id>/available-sources — wraps ai.list_sources_excluding_kb."""

    def test_excludes_already_indexed_sources(
        self, client, mock_auth, auth_headers, test_knowledge_base, test_source, app
    ):
        from agentic_project_service.db import db

        kb_id = test_knowledge_base["id"]
        # test_source is extracted but NOT indexed into this KB -> eligible.
        other_source_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text(
                    'INSERT INTO "ai".sources (id, name, file_type, storage_path, extraction_status) '
                    "VALUES (:id, 'other.pdf', 'application/pdf', 'sources/other.pdf', 'extracted')"
                ),
                {"id": other_source_id},
            )
            # Index other_source_id into the KB -> must be excluded from the response.
            db.session.execute(
                text(
                    'INSERT INTO "ai".indexed_sources (id, source_id, knowledge_base_id, index_status) '
                    "VALUES (gen_random_uuid(), :sid, :kb, 'indexed')"
                ),
                {"sid": other_source_id, "kb": kb_id},
            )
            db.session.commit()

        resp = client.get(f"/api/knowledge-bases/{kb_id}/available-sources", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        ids = [s["id"] for s in data["sources"]]
        assert test_source["id"] in ids
        assert other_source_id not in ids
        assert data["total"] == 1

    def test_search_and_empty_result(self, client, mock_auth, auth_headers, test_knowledge_base):
        kb_id = test_knowledge_base["id"]
        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/available-sources?q=no-such-source",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["sources"] == []
        assert data["total"] == 0


class TestKBInspectorContentViewers:
    """GET .../indexed-sources/<id>/{chunks,page-index-*,graph-index-*,full-document,doc2json-document}."""

    def test_indexed_source_not_in_kb_returns_404(
        self, client, mock_auth, auth_headers, test_knowledge_base, test_indexed_source, app
    ):
        from agentic_project_service.db import db

        # A second, unrelated KB — test_indexed_source belongs to a different KB.
        other_kb_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text(
                    "INSERT INTO ai.knowledge_bases (id, name, indexing_config, retrieval_config) "
                    "VALUES (:id, 'other-kb', '{}'::jsonb, '{}'::jsonb)"
                ),
                {"id": other_kb_id},
            )
            db.session.commit()

        for suffix in (
            "chunks",
            "page-index-nodes",
            "page-index-toc",
            "graph-index-nodes",
            "graph-index-toc",
            "full-document",
            "doc2json-document",
        ):
            resp = client.get(
                f"/api/knowledge-bases/{other_kb_id}/indexed-sources/"
                f"{test_indexed_source['id']}/{suffix}",
                headers=auth_headers,
            )
            assert resp.status_code == 404, f"{suffix} should 404 for a mismatched kb_id"

    def test_chunks_pagination_and_ordering(
        self, client, mock_auth, auth_headers, test_knowledge_base, test_indexed_source, app
    ):
        from agentic_project_service.db import db

        kb_id = test_knowledge_base["id"]
        isid = test_indexed_source["id"]
        src_id = test_indexed_source["source_id"]
        with app.app_context():
            for i in range(3):
                db.session.execute(
                    text(
                        'INSERT INTO "ai".chunks '
                        "(id, indexed_source_id, knowledge_base_id, source_id, text, chunk_index) "
                        "VALUES (gen_random_uuid(), :isid, :kb, :src, :txt, :idx)"
                    ),
                    {"isid": isid, "kb": kb_id, "src": src_id, "txt": f"chunk {i}", "idx": i},
                )
            db.session.commit()

        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/indexed-sources/{isid}/chunks?limit=2",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 3
        assert data["limit"] == 2
        assert len(data["chunks"]) == 2
        assert [c["chunk_index"] for c in data["chunks"]] == [0, 1]
        assert data["chunks"][0]["text"] == "chunk 0"

        resp2 = client.get(
            f"/api/knowledge-bases/{kb_id}/indexed-sources/{isid}/chunks?limit=2&offset=2",
            headers=auth_headers,
        )
        assert [c["chunk_index"] for c in resp2.get_json()["chunks"]] == [2]

    def test_page_index_nodes_and_toc(
        self, client, mock_auth, auth_headers, test_knowledge_base, test_indexed_source, app
    ):
        from agentic_project_service.db import db

        kb_id = test_knowledge_base["id"]
        isid = test_indexed_source["id"]
        src_id = test_indexed_source["source_id"]
        toc_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text(
                    'INSERT INTO "ai".page_index_toc '
                    "(id, indexed_source_id, knowledge_base_id, source_id, doc_name, "
                    "doc_description, structure) "
                    "VALUES (:id, :isid, :kb, :src, 'My Doc', 'desc', :structure)"
                ),
                {
                    "id": toc_id,
                    "isid": isid,
                    "kb": kb_id,
                    "src": src_id,
                    "structure": json.dumps([{"node_id": "n1", "title": "Node 1", "level": 0}]),
                },
            )
            db.session.execute(
                text(
                    'INSERT INTO "ai".page_index_nodes '
                    "(id, toc_id, indexed_source_id, knowledge_base_id, source_id, "
                    "node_id, title, depth, text) "
                    "VALUES (gen_random_uuid(), :toc, :isid, :kb, :src, 'n1', 'Node 1', 0, 'body text')"
                ),
                {"toc": toc_id, "isid": isid, "kb": kb_id, "src": src_id},
            )
            db.session.commit()

        nodes_resp = client.get(
            f"/api/knowledge-bases/{kb_id}/indexed-sources/{isid}/page-index-nodes",
            headers=auth_headers,
        )
        assert nodes_resp.status_code == 200
        nodes = nodes_resp.get_json()["nodes"]
        assert len(nodes) == 1
        assert nodes[0]["node_id"] == "n1"
        assert nodes[0]["title"] == "Node 1"
        assert nodes[0]["depth"] == 0
        assert nodes[0]["text"] == "body text"

        toc_resp = client.get(
            f"/api/knowledge-bases/{kb_id}/indexed-sources/{isid}/page-index-toc",
            headers=auth_headers,
        )
        assert toc_resp.status_code == 200
        toc = toc_resp.get_json()["toc"]
        assert toc["doc_name"] == "My Doc"
        assert toc["doc_description"] == "desc"
        assert toc["structure"] == [{"node_id": "n1", "title": "Node 1", "level": 0}]

    def test_page_index_toc_null_when_absent(
        self, client, mock_auth, auth_headers, test_knowledge_base, test_indexed_source
    ):
        kb_id = test_knowledge_base["id"]
        isid = test_indexed_source["id"]
        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/indexed-sources/{isid}/page-index-toc",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["toc"] is None

    def test_graph_index_nodes_and_toc(
        self, client, mock_auth, auth_headers, test_knowledge_base, test_indexed_source, app
    ):
        from agentic_project_service.db import db

        kb_id = test_knowledge_base["id"]
        isid = test_indexed_source["id"]
        src_id = test_indexed_source["source_id"]
        toc_id = str(uuid.uuid4())
        with app.app_context():
            db.session.execute(
                text(
                    'INSERT INTO "ai".graph_index_toc '
                    "(id, indexed_source_id, knowledge_base_id, source_id, doc_name, structure) "
                    "VALUES (:id, :isid, :kb, :src, 'Graph Doc', :structure)"
                ),
                {"id": toc_id, "isid": isid, "kb": kb_id, "src": src_id, "structure": "[]"},
            )
            db.session.execute(
                text(
                    'INSERT INTO "ai".graph_index_nodes '
                    "(id, toc_id, indexed_source_id, knowledge_base_id, source_id, "
                    "node_id, title, depth, text) "
                    "VALUES (gen_random_uuid(), :toc, :isid, :kb, :src, 'g1', 'Graph Node', 1, 'graph body')"
                ),
                {"toc": toc_id, "isid": isid, "kb": kb_id, "src": src_id},
            )
            db.session.commit()

        nodes_resp = client.get(
            f"/api/knowledge-bases/{kb_id}/indexed-sources/{isid}/graph-index-nodes",
            headers=auth_headers,
        )
        nodes = nodes_resp.get_json()["nodes"]
        assert len(nodes) == 1
        assert nodes[0]["node_id"] == "g1"
        assert nodes[0]["depth"] == 1

        toc_resp = client.get(
            f"/api/knowledge-bases/{kb_id}/indexed-sources/{isid}/graph-index-toc",
            headers=auth_headers,
        )
        assert toc_resp.get_json()["toc"]["doc_name"] == "Graph Doc"

    def test_full_document(
        self, client, mock_auth, auth_headers, test_knowledge_base, test_indexed_source, app
    ):
        from agentic_project_service.db import db

        kb_id = test_knowledge_base["id"]
        isid = test_indexed_source["id"]
        src_id = test_indexed_source["source_id"]
        with app.app_context():
            db.session.execute(
                text(
                    'INSERT INTO "ai".full_documents '
                    "(id, indexed_source_id, knowledge_base_id, source_id, summary, "
                    "full_text_path, summary_model, summary_tokens, full_text_tokens) "
                    "VALUES (gen_random_uuid(), :isid, :kb, :src, 'the summary', "
                    "'full/text.txt', 'gpt-5-mini', 42, 1000)"
                ),
                {"isid": isid, "kb": kb_id, "src": src_id},
            )
            db.session.commit()

        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/indexed-sources/{isid}/full-document",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        doc = resp.get_json()["document"]
        assert doc["summary"] == "the summary"
        assert doc["summary_model"] == "gpt-5-mini"
        assert doc["summary_tokens"] == 42
        assert doc["full_text_tokens"] == 1000
        # full_text_path is intentionally NOT exposed (internal storage detail,
        # not selected by the original .select() call either).
        assert "full_text_path" not in doc

    def test_full_document_null_when_absent(
        self, client, mock_auth, auth_headers, test_knowledge_base, test_indexed_source
    ):
        kb_id = test_knowledge_base["id"]
        isid = test_indexed_source["id"]
        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/indexed-sources/{isid}/full-document",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["document"] is None

    def test_doc2json_document_with_source_derivatives(
        self, client, mock_auth, auth_headers, test_knowledge_base, test_indexed_source, app
    ):
        from agentic_project_service.db import db

        kb_id = test_knowledge_base["id"]
        isid = test_indexed_source["id"]
        src_id = test_indexed_source["source_id"]
        with app.app_context():
            db.session.execute(
                text('UPDATE "ai".sources SET derivatives = :d WHERE id = :sid'),
                {"sid": src_id, "d": json.dumps({"image": ["p0.png", "p1.png"]})},
            )
            db.session.execute(
                text(
                    'INSERT INTO "ai".doc2json_documents '
                    "(id, indexed_source_id, knowledge_base_id, source_id, summary, "
                    "extracted_json, json_schema, extraction_model, window_count) "
                    "VALUES (gen_random_uuid(), :isid, :kb, :src, 'doc summary', "
                    ":extracted, '{}'::jsonb, 'gpt-5-mini', 3)"
                ),
                {
                    "isid": isid,
                    "kb": kb_id,
                    "src": src_id,
                    "extracted": json.dumps({"field": "value"}),
                },
            )
            db.session.commit()

        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/indexed-sources/{isid}/doc2json-document",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["document"]["summary"] == "doc summary"
        assert data["document"]["extracted_json"] == {"field": "value"}
        assert data["document"]["extraction_model"] == "gpt-5-mini"
        assert data["document"]["window_count"] == 3
        assert data["source_derivatives"] == {"image": ["p0.png", "p1.png"]}

    def test_doc2json_document_null_when_absent(
        self, client, mock_auth, auth_headers, test_knowledge_base, test_indexed_source
    ):
        kb_id = test_knowledge_base["id"]
        isid = test_indexed_source["id"]
        resp = client.get(
            f"/api/knowledge-bases/{kb_id}/indexed-sources/{isid}/doc2json-document",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["document"] is None
        assert data["source_derivatives"] is None
