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
from datetime import datetime, timezone
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


class _gate:
    """Patch the two catalog reads that decide whether a PATCH may dispatch the
    ensure."""

    def __init__(self, partition, rows_in_default):
        self._patches = {
            "partition_exists": patch(f"{S}.partition_exists", return_value=partition),
            "kb_has_rows_in_default": patch(
                f"{S}.kb_has_rows_in_default", return_value=rows_in_default
            ),
        }

    def __enter__(self):
        return {name: p.__enter__() for name, p in self._patches.items()}

    def __exit__(self, *exc):
        for p in reversed(list(self._patches.values())):
            p.__exit__(*exc)
        return False


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
    def _patch(
        self,
        old_config,
        new_config,
        backend,
        *,
        auto_indexing=True,
        partition=True,
        rows_in_default=False,
    ):
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
            _gate(partition, rows_in_default) as gate,
        ):
            resp = _client().patch(
                f"/api/knowledge-bases/{kb_id}",
                json={"retrieval_config": new_config},
                headers=_headers(),
            )
        assert resp.status_code == 200
        self.body = resp.get_json()
        self.gate = gate
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

    @pytest.mark.parametrize(
        "old, new",
        [
            ({"method": "hybrid"}, {"method": "full_text"}),
            ({"method": "full_text"}, {"method": "hybrid"}),
            (
                {"method": "hybrid", "ts_language": "german"},
                {"method": "hybrid", "ts_language": "german"},
            ),
            ({"method": "hybrid"}, {"method": "hybrid", "top_k": 7}),
        ],
    )
    def test_keyword_to_keyword_on_pg_search_dispatches_nothing(self, tasks, old, new):
        """Nothing about the index changes, so nothing may be dispatched -- even
        for a KB whose partition exists and whose ensure would be cheap."""
        self._patch(old, new, "pg_search", partition=True)
        tasks["ensure"].delay.assert_not_called()
        tasks["build"].delay.assert_not_called()
        tasks["drop"].delay.assert_not_called()
        assert "bm25_note" not in self.body


class TestUpdateDoesNotStartAMove:
    """A PATCH may reconcile a KB's pg_search index only when that cannot move
    rows out of DEFAULT: its partition already exists, or it has no rows in
    DEFAULT to move. Anything else blocks writes to the whole item table for
    the length of the move, which only an operator may start
    (``POST /build-bm25``)."""

    _patch = TestUpdate._patch

    @pytest.mark.parametrize(
        "old, new",
        [
            ({"method": "vector_search"}, {"method": "hybrid"}),
            (
                {"method": "hybrid", "ts_language": "english"},
                {"method": "hybrid", "ts_language": "german"},
            ),
        ],
    )
    def test_an_existing_partition_dispatches_the_ensure(self, tasks, old, new):
        kb_id = self._patch(old, new, "pg_search", partition=True, rows_in_default=None)
        tasks["ensure"].delay.assert_called_once_with(kb_id)
        assert "bm25_note" not in self.body
        assert self.gate["partition_exists"].call_args.args[1:] == (kb_id, "chunks")

    def test_no_rows_in_default_dispatches_the_ensure(self, tasks):
        kb_id = self._patch(
            {"method": "vector_search"},
            {"method": "hybrid"},
            "pg_search",
            partition=False,
            rows_in_default=False,
        )
        tasks["ensure"].delay.assert_called_once_with(kb_id)
        assert "bm25_note" not in self.body
        assert self.gate["kb_has_rows_in_default"].call_args.args[1:] == (kb_id, "chunks")

    @pytest.mark.parametrize("rows_in_default", [True, None])
    @pytest.mark.parametrize(
        "old, new",
        [
            ({"method": "vector_search"}, {"method": "hybrid"}),
            ({"method": "vector_search"}, {"method": "full_text"}),
            (
                {"method": "hybrid", "ts_language": "english"},
                {"method": "hybrid", "ts_language": "german"},
            ),
        ],
    )
    def test_rows_in_default_dispatch_nothing_and_point_at_build_bm25(
        self, tasks, old, new, rows_in_default
    ):
        """``None`` is "cannot tell", which must not start a move either."""
        kb_id = self._patch(old, new, "pg_search", partition=False, rows_in_default=rows_in_default)
        tasks["ensure"].delay.assert_not_called()
        tasks["build"].delay.assert_not_called()
        tasks["drop"].delay.assert_not_called()
        note = self.body["bm25_note"]
        assert f"POST /api/knowledge-bases/{kb_id}/build-bm25" in note
        assert "blocks writes" in note
        assert "chunks" in note
        # Not "keeps its existing path": a KB with no file index has none.
        assert "bm25s file index if it has one" in note
        assert "bounded full-text fallback" in note

    @pytest.mark.parametrize(
        "partition, rows_in_default, dispatched",
        [(True, None, True), (False, False, True), (False, True, False)],
    )
    def test_the_gate_ends_its_transaction_before_dispatching(
        self, tasks, partition, rows_in_default, dispatched
    ):
        """The DEFAULT probe holds ACCESS SHARE on DEFAULT until its transaction
        ends, and a move waits for that lock: the PATCH must end it before it
        dispatches the ensure (which may start that move) and before it builds
        the response."""
        calls = []
        kb_id = str(uuid.uuid4())
        with (
            _AUTH,
            patch(f"{R}.db") as db,
            patch(f"{R}._read_existing_retrieval_config", return_value={"method": "vector_search"}),
            patch(f"{R}._read_kb_strategy", return_value="chunk_embed"),
            patch(f"{R}._keyword_index_backend", return_value="pg_search"),
            patch(f"{R}._pg_search_available", return_value=True),
            patch(f"{R}.get_setting", return_value=True),
            patch(
                f"{R}.get_knowledge_base",
                side_effect=lambda _id: calls.append("respond") or ({"id": _id}, 200),
            ),
            patch(
                f"{S}.partition_exists",
                side_effect=lambda *a: calls.append("probe partition") or partition,
            ),
            patch(
                f"{S}.kb_has_rows_in_default",
                side_effect=lambda *a: calls.append("probe DEFAULT") or rows_in_default,
            ),
        ):
            db.session.rollback.side_effect = lambda: calls.append("end transaction")
            tasks["ensure"].delay.side_effect = lambda *_: calls.append("dispatch")
            resp = _client().patch(
                f"/api/knowledge-bases/{kb_id}",
                json={"retrieval_config": {"method": "hybrid"}},
                headers=_headers(),
            )
        assert resp.status_code == 200
        probes = ["probe partition"] if partition else ["probe partition", "probe DEFAULT"]
        expected = [*probes, "end transaction", *(["dispatch"] if dispatched else []), "respond"]
        assert calls == expected

    def test_an_unreadable_gate_dispatches_nothing(self, tasks):
        with patch(f"{S}.partition_exists", side_effect=RuntimeError("connection lost")):
            kb_id = str(uuid.uuid4())
            with (
                _AUTH,
                patch(f"{R}.db"),
                patch(f"{R}._read_existing_retrieval_config", return_value={}),
                patch(f"{R}._read_kb_strategy", return_value="chunk_embed"),
                patch(f"{R}._keyword_index_backend", return_value="pg_search"),
                patch(f"{R}._pg_search_available", return_value=True),
                patch(f"{R}.get_setting", return_value=True),
                patch(f"{R}.get_knowledge_base", return_value=({"id": kb_id}, 200)),
            ):
                resp = _client().patch(
                    f"/api/knowledge-bases/{kb_id}",
                    json={"retrieval_config": {"method": "hybrid"}},
                    headers=_headers(),
                )
        assert resp.status_code == 200
        tasks["ensure"].delay.assert_not_called()
        assert "build-bm25" in resp.get_json()["bm25_note"]

    def test_a_failed_get_is_returned_without_a_note(self, tasks):
        kb_id = str(uuid.uuid4())
        with (
            _AUTH,
            patch(f"{R}.db"),
            patch(f"{R}._read_existing_retrieval_config", return_value={}),
            patch(f"{R}._read_kb_strategy", return_value="chunk_embed"),
            patch(f"{R}._keyword_index_backend", return_value="pg_search"),
            patch(f"{R}._pg_search_available", return_value=True),
            patch(f"{R}.get_knowledge_base", return_value=({"error": "gone"}, 404)),
            _gate(False, True),
        ):
            resp = _client().patch(
                f"/api/knowledge-bases/{kb_id}",
                json={"retrieval_config": {"method": "hybrid"}},
                headers=_headers(),
            )
        assert resp.status_code == 404
        assert resp.get_json() == {"error": "gone"}


class TestUpdateStrategy:
    """A PATCH that changes ``indexing_config.strategy`` moves the keyword text
    to another item table, so the KB needs its index there -- and the one on
    the old table is dead weight."""

    def _patch(
        self,
        body,
        *,
        old_strategy,
        new_backend,
        method="hybrid",
        pg_available=True,
        partition=True,
        rows_in_default=False,
    ):
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
            _gate(partition, rows_in_default) as gate,
        ):
            resp = _client().patch(f"/api/knowledge-bases/{kb_id}", json=body, headers=_headers())
        assert resp.status_code == 200
        self.body = resp.get_json()
        self.gate = gate
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

    @pytest.mark.parametrize(
        "body",
        [
            {"indexing_config": {"strategy": "full_document"}},
            {
                "indexing_config": {"strategy": "full_document"},
                "retrieval_config": {"method": "hybrid"},
            },
        ],
    )
    def test_rows_in_the_new_tables_default_start_no_move(self, tasks, body):
        """The index on the table the KB left is dead weight either way, so it
        is dropped; the new one waits for POST /build-bm25."""
        kb_id, _ = self._patch(
            body,
            old_strategy="chunk_embed",
            new_backend="pg_search",
            partition=False,
            rows_in_default=True,
        )
        tasks["ensure"].delay.assert_not_called()
        tasks["drop"].delay.assert_called_once_with(kb_id, drop_partitions=False)
        assert "build-bm25" in self.body["bm25_note"]
        assert "full_documents" in self.body["bm25_note"]
        assert self.gate["partition_exists"].call_args.args[1:] == (kb_id, "full_documents")

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
        body = resp.get_json()
        assert (body["task_id"], body["knowledge_base_id"]) == ("task-ensure", kb_id)
        assert set(body) == {"task_id", "knowledge_base_id", "note"}
        tasks["ensure"].delay.assert_called_once_with(kb_id, allow_row_move=True)
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

    @pytest.mark.parametrize("installed", [True, False])
    def test_the_202_states_the_write_block_and_the_restart(self, tasks, installed):
        """Both obligations an operator has before calling this: the first build
        of an existing KB blocks writes to its whole item table, and the
        extension is only enabled once the service restarts after the Postgres
        image swap (otherwise this silently builds the file index)."""
        with (
            patch(f"{S}.pg_search_installed", return_value=installed),
            patch(f"{S}._item_table_is_partitioned", return_value=True),
        ):
            _, resp = self._post()
        assert resp.status_code == 202
        note = resp.get_json()["note"]
        assert "blocks writes" in note
        assert "chunks" in note
        assert "restart the project service" in note

    def test_the_docstring_states_the_write_block_and_the_restart(self):
        doc = " ".join(kb_route.build_bm25_endpoint.__doc__.split())
        assert "blocks writes" in doc
        assert "restart" in doc

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


class TestStatusReportsThePersistedBuildOutcome:
    """On a pg_search item table, a KB whose own index is not serving yet
    reports the last recorded outcome of its move/build -- a retrying or
    failed move must not look like a KB nobody scheduled."""

    KB = TestStatus.KB

    @staticmethod
    def _outcome(status, reason=None, item_table="chunks", attempts=None):
        return {
            "status": status,
            "reason": reason,
            "item_table": item_table,
            "attempts": attempts,
            "updated_at": None,
        }

    def _detail(self, *, pg_state, outcome, backend="pg_search", auto_indexing=True, file=True):
        with (
            patch(f"{R}.db"),
            patch(f"{R}._keyword_index_backend", return_value=backend),
            patch(f"{R}.pg_bm25_status", return_value=pg_state),
            patch(f"{R}.read_bm25_build_outcome", return_value=outcome) as read,
            patch(f"{R}.get_setting", return_value=auto_indexing),
            patch(f"{R}.SparseIndexStore") as store_cls,
            patch(f"{R}._count_items_for_kb_bm25", return_value=3),
        ):
            store_cls.return_value.index_exists.return_value = file
            store_cls.return_value.read_metadata.return_value = {"item_count": 3}
            return kb_route._bm25_status_detail(self.KB), read

    @pytest.mark.parametrize("status", ["queued", "moving", "building", "retrying", "failed"])
    def test_a_file_served_kb_reports_its_pending_or_failed_move(self, status):
        (got, reason), _ = self._detail(
            pg_state="absent", outcome=self._outcome(status, reason="lock_not_available")
        )
        assert got == status
        assert reason == "lock_not_available"

    def test_the_status_is_reported_with_auto_indexing_on(self):
        """Auto-indexing on used to omit the field for a file-served KB."""
        (got, _), _ = self._detail(
            pg_state="absent", outcome=self._outcome("failed", "gave up"), auto_indexing=True
        )
        assert got == "failed"

    def test_an_invalid_index_with_a_failed_build_reports_failed(self):
        (got, reason), _ = self._detail(
            pg_state="building", outcome=self._outcome("failed", "index build failed")
        )
        assert (got, reason) == ("failed", "index build failed")

    def test_a_served_kb_ignores_the_outcome(self):
        (got, reason), read = self._detail(
            pg_state="ready", outcome=self._outcome("failed", "old failure")
        )
        assert (got, reason) == ("ready", None)
        read.assert_not_called()

    def test_an_outcome_for_another_item_table_is_ignored(self):
        (got, reason), _ = self._detail(
            pg_state="absent",
            outcome=self._outcome("failed", "old table", item_table="full_documents"),
            auto_indexing=False,
        )
        assert (got, reason) == ("ready", None)

    def test_a_ready_outcome_without_a_ready_index_is_not_reported(self):
        """The index was dropped since (a strategy or ts_language change); the
        recorded "ready" is history, and the file index is what search reads."""
        (got, reason), _ = self._detail(
            pg_state="absent", outcome=self._outcome("ready"), auto_indexing=False
        )
        assert (got, reason) == ("ready", None)

    def test_no_outcome_keeps_reporting_the_file_index(self):
        (got, reason), _ = self._detail(pg_state="absent", outcome=None, auto_indexing=False)
        assert (got, reason) == ("ready", None)

    def test_the_file_index_backend_never_reads_the_outcome(self):
        (got, _), read = self._detail(
            pg_state="absent",
            outcome=self._outcome("failed", "x"),
            backend="bm25s",
            auto_indexing=False,
        )
        assert got == "ready"
        read.assert_not_called()

    def test_an_extension_absent_record_never_reaches_a_file_index_backend(self):
        """The worker records a skipped build as ``failed`` with ``not built:
        extension_absent``. On the bm25s backend that record must not become
        the KB's status: the file index is what search reads."""
        (got, reason), read = self._detail(
            pg_state="absent",
            outcome=self._outcome("failed", "not built: extension_absent"),
            backend="bm25s",
            auto_indexing=False,
        )
        assert (got, reason) == ("ready", None)
        read.assert_not_called()

    @pytest.mark.parametrize(
        "reason",
        [
            "not built: extension_absent",
            "not built: table_not_partitioned",
            "not built: retrieval_method",
            "not built: strategy",
        ],
    )
    def test_a_skip_whose_precondition_now_holds_is_not_reported(self, reason):
        """Once pg_search serves the KB's table, a record saying the extension
        was absent (or the table unpartitioned, or the method not keyword) is
        from before that changed, not the state of a build still to run."""
        (got, got_reason), _ = self._detail(
            pg_state="absent", outcome=self._outcome("failed", reason), auto_indexing=False
        )
        assert (got, got_reason) == ("ready", None)

    def test_a_skip_for_a_missing_default_partition_is_reported(self):
        (got, reason), _ = self._detail(
            pg_state="absent",
            outcome=self._outcome("failed", "not built: default_partition_absent"),
        )
        assert (got, reason) == ("failed", "not built: default_partition_absent")

    def test_an_unreadable_outcome_falls_back_to_the_index_state(self):
        """Outside an app context even ``db.session`` raises; the status must
        still come back."""
        with (
            patch(f"{R}._keyword_index_backend", return_value="pg_search"),
            patch(f"{R}.pg_bm25_status", return_value="building"),
            patch(f"{R}.read_bm25_build_outcome", side_effect=RuntimeError("no app context")),
        ):
            assert kb_route._bm25_status_detail(self.KB) == ("building", None)

    def test_compute_bm25_status_returns_only_the_status(self):
        with patch(f"{R}._bm25_status_detail", return_value=("failed", "why")):
            assert kb_route._compute_bm25_status(self.KB) == "failed"

    def _detail_recorded(self, outcome):
        recorded: dict = {}
        with (
            patch(f"{R}.db"),
            patch(f"{R}._keyword_index_backend", return_value="pg_search"),
            patch(f"{R}.pg_bm25_status", return_value="absent"),
            patch(f"{R}.read_bm25_build_outcome", return_value=outcome),
        ):
            return kb_route._bm25_status_detail(self.KB, recorded), recorded

    @pytest.mark.parametrize("status", ["moving", "building"])
    def test_an_in_progress_outcome_silent_for_too_long_is_reported_stale(self, status):
        """A worker killed mid-move, or a run that found the index lock held,
        leaves its last status behind for ever."""
        from datetime import datetime, timedelta, timezone

        long_ago = datetime.now(timezone.utc) - timedelta(
            seconds=kb_route.BM25_BUILD_OUTCOME_STALE_SECONDS + 60
        )
        outcome = self._outcome(status)
        outcome["updated_at"] = long_ago
        (got, reason), recorded = self._detail_recorded(outcome)
        assert got == "stale"
        assert repr(status) in reason and "POST /build-bm25" in reason
        assert recorded["updated_at"] == long_ago

    @pytest.mark.parametrize("status", ["moving", "failed", "retrying"])
    def test_a_recent_or_terminal_outcome_is_reported_as_recorded(self, status):
        from datetime import datetime, timedelta, timezone

        outcome = self._outcome(status, reason="why")
        outcome["updated_at"] = datetime.now(timezone.utc) - timedelta(
            seconds=kb_route.BM25_BUILD_OUTCOME_STALE_SECONDS + 60 if status != "moving" else 5
        )
        (got, reason), _ = self._detail_recorded(outcome)
        assert (got, reason) == (status, "why")


class TestStatusField:
    def _get(self, detail):
        kb_id = "11111111-1111-1111-1111-111111111111"
        kb = {
            "id": kb_id,
            "name": "kb",
            "description": None,
            "indexing_config": {"strategy": "chunk_embed"},
            "retrieval_config": {"method": "hybrid"},
            "created_at": None,
            "updated_at": None,
        }
        with (
            _AUTH,
            patch(f"{R}.db") as db,
            patch(f"{R}._fetch_kb_or_404", return_value=kb),
            patch(f"{R}._compute_drift", return_value="none"),
            patch(f"{R}._bm25_status_detail", return_value=detail),
        ):
            db.session.execute.return_value = iter([])
            return _client().get(f"/api/knowledge-bases/{kb_id}", headers=_headers()).get_json()

    def test_the_reason_is_in_the_response(self):
        body = self._get(("retrying", "lock_not_available (attempt 2)"))
        assert body["bm25_status"] == "retrying"
        assert body["bm25_status_reason"] == "lock_not_available (attempt 2)"

    def test_no_reason_no_field(self):
        body = self._get(("ready", None))
        assert body["bm25_status"] == "ready"
        assert "bm25_status_reason" not in body
        assert "bm25_status_updated_at" not in body

    def test_a_recorded_status_carries_when_it_was_recorded(self):
        def detail(kb, recorded=None):
            recorded["updated_at"] = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)
            return "moving", None

        kb_id = "11111111-1111-1111-1111-111111111111"
        kb = {
            "id": kb_id,
            "name": "kb",
            "description": None,
            "indexing_config": {"strategy": "chunk_embed"},
            "retrieval_config": {"method": "hybrid"},
            "created_at": None,
            "updated_at": None,
        }
        with (
            _AUTH,
            patch(f"{R}.db") as db,
            patch(f"{R}._fetch_kb_or_404", return_value=kb),
            patch(f"{R}._compute_drift", return_value="none"),
            patch(f"{R}._bm25_status_detail", side_effect=detail),
        ):
            db.session.execute.return_value = iter([])
            body = _client().get(f"/api/knowledge-bases/{kb_id}", headers=_headers()).get_json()
        assert body["bm25_status"] == "moving"
        assert body["bm25_status_updated_at"] == "2026-09-16T10:00:00+00:00"
