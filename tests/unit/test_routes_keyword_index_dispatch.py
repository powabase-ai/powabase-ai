"""Which keyword-index task each knowledge-base route dispatches.

A knowledge base's keyword leg is served by exactly one index: its pg_search
BM25 index when the extension is installed and the strategy's item table is
partitioned, otherwise the bm25s file index. Every route that starts an index
build has to pick the same one, and must never start both -- the file-index
build re-tokenises the whole knowledge base, which is the work the pg_search
index exists to remove.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from agentic_project_service.routes import knowledge_bases as kb_route

R = "agentic_project_service.routes.knowledge_bases"
# The rule itself lives in the service; the route helper delegates to it.
S = "agentic_project_service.services.pg_bm25_index"

_AUTH = patch(
    "agentic_project_service.auth.decode_jwt",
    return_value={"sub": "user-1", "role": "service_role"},
)


def _client():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(kb_route.knowledge_bases_bp)
    return app.test_client()


def _headers():
    return {"Authorization": "Bearer fake"}


@pytest.fixture
def tasks():
    with (
        patch(f"{R}.ensure_pg_bm25_index") as ensure,
        patch(f"{R}.build_bm25_for_kb") as build,
        patch(f"{R}.drop_pg_bm25_index") as drop,
    ):
        ensure.delay.return_value = MagicMock(id="task-ensure")
        build.delay.return_value = MagicMock(id="task-build")
        yield {"ensure": ensure, "build": build, "drop": drop}


# ---------------------------------------------------------------------------
# The single predicate
# ---------------------------------------------------------------------------


class TestKeywordIndexBackend:
    @pytest.fixture(autouse=True)
    def _db(self):
        with patch(f"{R}.db"):
            yield

    def test_pg_search_when_installed_mapped_and_partitioned(self):
        with (
            patch(f"{S}.pg_search_installed", return_value=True),
            patch(f"{S}._item_table_is_partitioned", return_value=True) as partitioned,
        ):
            assert kb_route._keyword_index_backend("chunk_embed") == "pg_search"
        assert partitioned.call_args.args[1] == "chunks"

    def test_bm25s_without_the_extension(self):
        with (
            patch(f"{S}.pg_search_installed", return_value=False),
            patch(f"{S}._item_table_is_partitioned") as partitioned,
        ):
            assert kb_route._keyword_index_backend("chunk_embed") == "bm25s"
        partitioned.assert_not_called()

    def test_bm25s_when_the_item_table_is_not_partitioned_yet(self):
        with (
            patch(f"{S}.pg_search_installed", return_value=True),
            patch(f"{S}._item_table_is_partitioned", return_value=False),
        ):
            assert kb_route._keyword_index_backend("full_document") == "bm25s"

    def test_bm25s_when_the_item_table_is_never_partitioned(self):
        with (
            patch(f"{S}.pg_search_installed", return_value=True),
            patch(f"{S}.PARTITIONED_ITEM_TABLES", frozenset({"full_documents"})),
            patch(f"{S}._item_table_is_partitioned", return_value=True) as partitioned,
        ):
            assert kb_route._keyword_index_backend("chunk_embed") == "bm25s"
        partitioned.assert_not_called()

    @pytest.mark.parametrize("strategy", ["page_index", "doc2json", "no_such_strategy"])
    def test_none_for_a_strategy_without_an_item_table(self, strategy):
        with patch(f"{S}.pg_search_installed", return_value=True):
            assert kb_route._keyword_index_backend(strategy) is None

    def test_a_missing_strategy_means_chunk_embed(self):
        with (
            patch(f"{S}.pg_search_installed", return_value=False),
        ):
            assert kb_route._keyword_index_backend(None) == "bm25s"

    def test_an_unreadable_catalog_falls_back_to_the_file_index(self):
        with (
            patch(f"{S}.pg_search_installed", return_value=True),
            patch(f"{S}._item_table_is_partitioned", side_effect=RuntimeError("connection lost")),
        ):
            assert kb_route._keyword_index_backend("chunk_embed") == "bm25s"


# ---------------------------------------------------------------------------
# POST /knowledge-bases
# ---------------------------------------------------------------------------


class TestCreate:
    def _post(self, body):
        with _AUTH, patch(f"{R}.db"):
            return _client().post("/api/knowledge-bases", json=body, headers=_headers())

    def test_hybrid_kb_on_pg_search_dispatches_only_the_pg_index(self, tasks):
        with patch(f"{R}._keyword_index_backend", return_value="pg_search"):
            resp = self._post(
                {
                    "name": "KB",
                    "indexing_config": {"strategy": "chunk_embed"},
                    "retrieval_config": {"method": "hybrid"},
                }
            )
        assert resp.status_code == 201
        tasks["ensure"].delay.assert_called_once_with(resp.get_json()["id"])
        tasks["build"].delay.assert_not_called()

    def test_hybrid_kb_on_the_file_index_dispatches_no_pg_index(self, tasks):
        """A new KB has no items, so there is no file index to build yet either:
        per-source indexing grows it."""
        with patch(f"{R}._keyword_index_backend", return_value="bm25s"):
            resp = self._post(
                {
                    "name": "KB",
                    "indexing_config": {"strategy": "chunk_embed"},
                    "retrieval_config": {"method": "full_text"},
                }
            )
        assert resp.status_code == 201
        tasks["ensure"].delay.assert_not_called()
        tasks["build"].delay.assert_not_called()

    def test_vector_kb_dispatches_nothing(self, tasks):
        with patch(f"{R}._keyword_index_backend", return_value="pg_search"):
            resp = self._post({"name": "KB", "retrieval_config": {"method": "vector_search"}})
        assert resp.status_code == 201
        tasks["ensure"].delay.assert_not_called()
        tasks["build"].delay.assert_not_called()


# ---------------------------------------------------------------------------
# PATCH /knowledge-bases/<id>
# ---------------------------------------------------------------------------


class TestUpdate:
    def _patch(self, old_config, new_config, backend, *, auto_indexing=True):
        kb_id = str(uuid.uuid4())
        with (
            _AUTH,
            patch(f"{R}.db"),
            patch(f"{R}._read_existing_retrieval_config", return_value=old_config),
            patch(f"{R}._read_kb_strategy", return_value="chunk_embed"),
            patch(f"{R}._keyword_index_backend", return_value=backend),
            patch(f"{R}._pg_search_available", return_value=backend == "pg_search"),
            patch(f"{R}.get_setting", return_value=auto_indexing),
            patch(f"{R}.get_knowledge_base", return_value=({"id": kb_id}, 200)),
        ):
            resp = _client().patch(
                f"/api/knowledge-bases/{kb_id}",
                json={"retrieval_config": new_config},
                headers=_headers(),
            )
        assert resp.status_code == 200
        return kb_id

    @pytest.mark.parametrize("method", ["hybrid", "full_text"])
    def test_to_keyword_on_pg_search_dispatches_only_the_pg_index(self, tasks, method):
        kb_id = self._patch({"method": "vector_search"}, {"method": method}, "pg_search")
        tasks["ensure"].delay.assert_called_once_with(kb_id)
        tasks["build"].delay.assert_not_called()
        tasks["drop"].delay.assert_not_called()

    @pytest.mark.parametrize("method", ["hybrid", "full_text"])
    def test_to_keyword_on_the_file_index_dispatches_only_the_file_build(self, tasks, method):
        kb_id = self._patch({"method": "vector_search"}, {"method": method}, "bm25s")
        tasks["build"].delay.assert_called_once_with(kb_id)
        tasks["ensure"].delay.assert_not_called()
        tasks["drop"].delay.assert_not_called()

    def test_file_build_still_respects_auto_indexing_off(self, tasks):
        self._patch({"method": "vector_search"}, {"method": "hybrid"}, "bm25s", auto_indexing=False)
        tasks["build"].delay.assert_not_called()
        tasks["ensure"].delay.assert_not_called()

    def test_ts_language_change_on_pg_search_rebuilds_only_the_pg_index(self, tasks):
        kb_id = self._patch(
            {"method": "hybrid", "ts_language": "english"},
            {"method": "hybrid", "ts_language": "german"},
            "pg_search",
        )
        tasks["ensure"].delay.assert_called_once_with(kb_id)
        tasks["build"].delay.assert_not_called()

    def test_keyword_to_keyword_on_the_file_index_dispatches_nothing(self, tasks):
        self._patch({"method": "hybrid"}, {"method": "full_text"}, "bm25s")
        tasks["build"].delay.assert_not_called()
        tasks["ensure"].delay.assert_not_called()

    @pytest.mark.parametrize("method", ["vector_search", "tree_search"])
    def test_off_keyword_on_pg_search_drops_the_index_and_keeps_the_partitions(self, tasks, method):
        kb_id = self._patch({"method": "hybrid"}, {"method": method}, "pg_search")
        tasks["drop"].delay.assert_called_once_with(kb_id, drop_partitions=False)
        tasks["ensure"].delay.assert_not_called()
        tasks["build"].delay.assert_not_called()

    def test_off_keyword_without_pg_search_dispatches_no_drop(self, tasks):
        self._patch({"method": "full_text"}, {"method": "vector_search"}, "bm25s")
        tasks["drop"].delay.assert_not_called()

    def test_vector_to_tree_dispatches_nothing(self, tasks):
        self._patch({"method": "vector_search"}, {"method": "tree_search"}, "pg_search")
        tasks["drop"].delay.assert_not_called()
        tasks["ensure"].delay.assert_not_called()
        tasks["build"].delay.assert_not_called()

    def test_a_broker_failure_on_the_drop_does_not_fail_the_patch(self, tasks):
        tasks["drop"].delay.side_effect = Exception("broker unreachable")
        self._patch({"method": "hybrid"}, {"method": "vector_search"}, "pg_search")


class TestUpdateStrategy:
    """A PATCH that changes ``indexing_config.strategy`` moves the keyword text
    to another item table, so the KB needs its index there -- and the one on
    the old table is dead weight."""

    def _patch(self, body, *, old_strategy, new_backend, method="hybrid", pg_available=True):
        kb_id = str(uuid.uuid4())
        with (
            _AUTH,
            patch(f"{R}.db"),
            patch(f"{R}._read_existing_retrieval_config", return_value={"method": method}),
            patch(f"{R}._read_kb_strategy", return_value=old_strategy),
            patch(f"{R}._keyword_index_backend", return_value=new_backend) as backend,
            patch(f"{R}._pg_search_available", return_value=pg_available),
            patch(f"{R}.get_setting", return_value=True),
            patch(f"{R}.get_knowledge_base", return_value=({"id": kb_id}, 200)),
        ):
            resp = _client().patch(f"/api/knowledge-bases/{kb_id}", json=body, headers=_headers())
        assert resp.status_code == 200
        return kb_id, backend

    @pytest.mark.parametrize("method", ["hybrid", "full_text"])
    def test_strategy_only_change_on_pg_search_dispatches_the_ensure(self, tasks, method):
        kb_id, backend = self._patch(
            {"indexing_config": {"strategy": "full_document"}},
            old_strategy="chunk_embed",
            new_backend="pg_search",
            method=method,
        )
        tasks["ensure"].delay.assert_called_once_with(kb_id)
        tasks["drop"].delay.assert_not_called()
        tasks["build"].delay.assert_not_called()
        assert backend.call_args.args[0] == "full_document"

    def test_strategy_change_to_one_without_an_item_table_drops_the_index(self, tasks):
        kb_id, _ = self._patch(
            {"indexing_config": {"strategy": "page_index"}},
            old_strategy="chunk_embed",
            new_backend=None,
        )
        tasks["drop"].delay.assert_called_once_with(kb_id, drop_partitions=False)
        tasks["ensure"].delay.assert_not_called()

    def test_unchanged_strategy_dispatches_nothing(self, tasks):
        self._patch(
            {"indexing_config": {"strategy": "chunk_embed", "chunk_size": 900}},
            old_strategy="chunk_embed",
            new_backend="pg_search",
        )
        tasks["ensure"].delay.assert_not_called()
        tasks["drop"].delay.assert_not_called()

    def test_strategy_change_on_a_vector_kb_dispatches_nothing(self, tasks):
        self._patch(
            {"indexing_config": {"strategy": "full_document"}},
            old_strategy="chunk_embed",
            new_backend="pg_search",
            method="vector_search",
        )
        tasks["ensure"].delay.assert_not_called()
        tasks["drop"].delay.assert_not_called()

    def test_strategy_change_without_pg_search_dispatches_nothing(self, tasks):
        self._patch(
            {"indexing_config": {"strategy": "full_document"}},
            old_strategy="chunk_embed",
            new_backend="bm25s",
            pg_available=False,
        )
        tasks["ensure"].delay.assert_not_called()
        tasks["drop"].delay.assert_not_called()

    def test_strategy_and_retrieval_change_together_dispatch_one_ensure(self, tasks):
        kb_id, _ = self._patch(
            {
                "indexing_config": {"strategy": "full_document"},
                "retrieval_config": {"method": "hybrid"},
            },
            old_strategy="chunk_embed",
            new_backend="pg_search",
        )
        tasks["ensure"].delay.assert_called_once_with(kb_id)
        tasks["drop"].delay.assert_not_called()


# ---------------------------------------------------------------------------
# POST /knowledge-bases/<id>/build-bm25
# ---------------------------------------------------------------------------


class TestBuildEndpoint:
    def _post(self, strategy="chunk_embed"):
        kb_id = str(uuid.uuid4())
        kb = {
            "id": kb_id,
            "retrieval_config": {"method": "hybrid"},
            "indexing_config": {"strategy": strategy},
        }
        with _AUTH, patch(f"{R}._fetch_kb_or_404", return_value=kb), patch(f"{R}.db"):
            resp = _client().post(f"/api/knowledge-bases/{kb_id}/build-bm25", headers=_headers())
        return kb_id, resp

    def test_pg_search_path_dispatches_only_the_pg_index(self, tasks):
        with (
            patch(f"{S}.pg_search_installed", return_value=True),
            patch(f"{S}._item_table_is_partitioned", return_value=True),
        ):
            kb_id, resp = self._post()
        assert resp.status_code == 202
        assert resp.get_json() == {"task_id": "task-ensure", "knowledge_base_id": kb_id}
        tasks["ensure"].delay.assert_called_once_with(kb_id)
        tasks["build"].delay.assert_not_called()

    def test_without_the_extension_dispatches_only_the_file_build(self, tasks):
        with patch(f"{S}.pg_search_installed", return_value=False):
            kb_id, resp = self._post()
        assert resp.status_code == 202
        tasks["build"].delay.assert_called_once_with(kb_id)
        tasks["ensure"].delay.assert_not_called()

    def test_an_unpartitioned_item_table_builds_the_file_index_it_actually_reads(self, tasks):
        """The pg task would skip with table_not_partitioned and leave the status
        absent forever; the search path reads the file index here, so build it."""
        with (
            patch(f"{S}.pg_search_installed", return_value=True),
            patch(f"{S}._item_table_is_partitioned", return_value=False),
        ):
            kb_id, resp = self._post()
        assert resp.status_code == 202
        assert resp.get_json()["task_id"] == "task-build"
        tasks["build"].delay.assert_called_once_with(kb_id)
        tasks["ensure"].delay.assert_not_called()

    def test_a_never_partitioned_item_table_builds_the_file_index(self, tasks):
        with (
            patch(f"{S}.pg_search_installed", return_value=True),
            patch(f"{S}.PARTITIONED_ITEM_TABLES", frozenset()),
            patch(f"{S}._item_table_is_partitioned", return_value=True),
        ):
            kb_id, resp = self._post()
        assert resp.status_code == 202
        tasks["build"].delay.assert_called_once_with(kb_id)
        tasks["ensure"].delay.assert_not_called()

    def test_an_unmapped_strategy_is_a_400_on_pg_search_too(self, tasks):
        with (
            patch(f"{S}.pg_search_installed", return_value=True),
            patch(f"{S}._item_table_is_partitioned", return_value=True),
        ):
            _, resp = self._post(strategy="page_index")
        assert resp.status_code == 400
        assert "page_index" in resp.get_json()["error"]
        tasks["build"].delay.assert_not_called()
        tasks["ensure"].delay.assert_not_called()


# ---------------------------------------------------------------------------
# bm25_status
# ---------------------------------------------------------------------------


class TestStatus:
    KB = {
        "id": "11111111-1111-1111-1111-111111111111",
        "retrieval_config": {"method": "hybrid"},
        "indexing_config": {"strategy": "chunk_embed"},
    }

    def test_pg_search_backend_reports_the_pg_index(self):
        with (
            patch(f"{R}._keyword_index_backend", return_value="pg_search"),
            patch(f"{R}.pg_bm25_status", return_value="building"),
        ):
            assert kb_route._compute_bm25_status(self.KB) == "building"

    def test_a_kb_still_on_its_file_index_reports_that_index_not_absent(self):
        """A KB from before pg_search has no partition of its own, and its
        keyword leg reads its bm25s file index until an operator builds its
        pg_search index. Its status is that file index's, not "absent"."""
        with (
            patch(f"{R}._keyword_index_backend", return_value="pg_search"),
            patch(f"{R}.pg_bm25_status", return_value="absent"),
            patch(f"{R}.get_setting", return_value=False),
            patch(f"{R}.SparseIndexStore") as store_cls,
            patch(f"{R}._count_items_for_kb_bm25", return_value=3),
        ):
            store_cls.return_value.index_exists.return_value = True
            store_cls.return_value.read_metadata.return_value = {"item_count": 3}
            assert kb_route._compute_bm25_status(self.KB) == "ready"

    def test_a_kb_with_no_index_at_all_is_absent(self):
        with (
            patch(f"{R}._keyword_index_backend", return_value="pg_search"),
            patch(f"{R}.pg_bm25_status", return_value="absent"),
            patch(f"{R}.get_setting", return_value=True),
            patch(f"{R}.SparseIndexStore") as store_cls,
        ):
            store_cls.return_value.index_exists.return_value = False
            assert kb_route._compute_bm25_status(self.KB) == "absent"

    def test_file_index_backend_ignores_the_pg_answer(self):
        """With the extension installed but the table unpartitioned, the pg
        status is a permanent "absent" that says nothing about the file index
        the keyword leg actually uses."""
        with (
            patch(f"{R}._keyword_index_backend", return_value="bm25s"),
            patch(f"{R}.pg_bm25_status", return_value="absent") as pg_status,
            patch(f"{R}.get_setting", return_value=False),
            patch(f"{R}.SparseIndexStore") as store_cls,
            patch(f"{R}._count_items_for_kb_bm25", return_value=3),
        ):
            store_cls.return_value.index_exists.return_value = True
            store_cls.return_value.read_metadata.return_value = {"item_count": 3}
            assert kb_route._compute_bm25_status(self.KB) == "ready"
        pg_status.assert_not_called()
