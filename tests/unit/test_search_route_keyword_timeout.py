"""POST /knowledge-bases/<id>/search maps a keyword-search timeout to 503, and
reports a hybrid search that silently lost its keyword leg."""

import json
import logging
import uuid
from unittest.mock import patch

from agentic.knowledge.models import RetrievedItem

from agentic_project_service.routes import knowledge_bases as kb_route
from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services.base_vector_store import KeywordSearchTimeout


def _app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(kb_route.knowledge_bases_bp)
    return app


def _kb(strategy: str | None, stored_method: str) -> dict:
    indexing_config = {} if strategy is None else {"strategy": strategy}
    return {
        "id": "kb",
        "indexing_config": indexing_config,
        "retrieval_config": {"method": stored_method},
    }


def _timeout_503(
    kb: dict | tuple | None = None,
    request_method: str = "full_text",
    *,
    fetch_error: Exception | None = None,
    auto_indexing: bool = True,
):
    """Drive the 503 path with a given KB row and return the parsed body."""
    kb_id = str(uuid.uuid4())
    fetch = (
        patch(
            "agentic_project_service.routes.knowledge_bases._fetch_kb_or_404",
            side_effect=fetch_error,
        )
        if fetch_error is not None
        else patch(
            "agentic_project_service.routes.knowledge_bases._fetch_kb_or_404", return_value=kb
        )
    )
    with (
        patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"}),
        patch("agentic_project_service.routes.knowledge_bases.db"),
        patch(
            "agentic_project_service.routes.knowledge_bases.get_setting",
            side_effect=lambda key: auto_indexing if key == "BM25_AUTO_INDEXING" else None,
        ),
        fetch,
        patch(
            "agentic_project_service.services.knowledge_search.search_knowledge_base",
            side_effect=KeywordSearchTimeout(kb_id, 10000),
        ),
        _app().test_client() as c,
    ):
        resp = c.post(
            f"/api/knowledge-bases/{kb_id}/search",
            headers={"Authorization": "Bearer fake.jwt.token"},
            json={"query": "weather", "retrieval_method": request_method},
        )
    assert resp.status_code == 503
    assert resp.is_json, resp.data[:200]
    body = resp.get_json()
    assert body["code"] == "keyword_search_timeout"
    assert body["timeout_ms"] == 10000
    # Every branch must keep saying that retrying does not self-heal, and never
    # leak a bare None where a strategy or method name belongs. The wording is
    # about the retry, not about builds in general: with auto-indexing on, the
    # stored-method remedy does start a build, and a blanket "no build starts on
    # its own" would contradict it in the same sentence.
    assert "Retrying the same search does not start a build" in body["error"]
    assert "No build starts on its own" not in body["error"]
    assert "queued" not in body["error"]
    assert "None" not in body["error"]
    return body


def _is_generic(error: str) -> bool:
    return "build-bm25" in error and "vector_search" in error and "strategy" not in error


def test_keyword_timeout_is_503():
    """The buildable case: mapped strategy, stored method already hybrid."""
    body = _timeout_503(_kb("chunk_embed", "hybrid"))
    assert _is_generic(body["error"])


def test_unmapped_strategy_is_never_told_to_build():
    """doc2json has no BM25 item table, so a build cannot help it.

    POST /build-bm25 now refuses such a KB with a 400 (before, it answered 202
    and the task then failed once with ValueError). doc2json is the one
    unmapped strategy that can reach this 503: page_index is rejected for
    full_text/hybrid by validate_retriever before any query runs.
    """
    body = _timeout_503(_kb("doc2json", "hybrid"))
    assert "build-bm25" not in body["error"]
    assert "vector_search" in body["error"]
    assert "doc2json" in body["error"]


def test_missing_strategy_key_is_chunk_embed_not_unmapped():
    """A config without "strategy" is searched as chunk_embed, so remedy it as one.

    Every other strategy read on this path defaults to chunk_embed; reading it
    as "no strategy" told a buildable KB that a build could not help it.

    Uses the stored-method branch on purpose: only a KB resolved as a real,
    mapped strategy reaches it, so this cannot pass by falling through to the
    generic wording the way an unresolved strategy would.
    """
    body = _timeout_503(_kb(None, "vector_search"), auto_indexing=False)
    assert "stored retrieval method" in body["error"]
    assert "build-bm25" in body["error"]


def test_json_string_indexing_config_still_yields_a_json_503():
    """Legacy rows can hold a JSON string where an object belongs.

    Reading .get() off one used to raise inside the except KeywordSearchTimeout
    handler, which the route's generic handler cannot catch, so the caller got a
    Flask HTML 500 and lost code, timeout_ms and the remedy.
    """
    kb = _kb("doc2json", "hybrid")
    kb["indexing_config"] = json.dumps(kb["indexing_config"])
    body = _timeout_503(kb)
    # Parsed, not discarded: a doc2json KB must still not be told to build.
    assert "build-bm25" not in body["error"]
    assert "doc2json" in body["error"]


def test_json_string_retrieval_config_still_yields_a_json_503():
    kb = _kb("chunk_embed", "vector_search")
    kb["retrieval_config"] = json.dumps(kb["retrieval_config"])
    body = _timeout_503(kb, auto_indexing=False)
    # Parsed: the stored vector_search method is still seen.
    assert "stored retrieval method" in body["error"]


def test_unparseable_string_config_falls_back_to_the_generic_remedy():
    kb = _kb("doc2json", "hybrid")
    kb["indexing_config"] = "not json at all"
    body = _timeout_503(kb)
    assert _is_generic(body["error"])


def test_stored_method_branch_with_auto_indexing_on_does_not_ask_for_a_second_build():
    """With auto-indexing on, the PATCH to hybrid/full_text dispatches the build.

    Telling the caller to POST /build-bm25 as well would start a second
    concurrent rebuild of the same index files.
    """
    body = _timeout_503(
        _kb("chunk_embed", "vector_search"), request_method="full_text", auto_indexing=True
    )
    assert "stored retrieval method" in body["error"]
    assert "hybrid or full_text" in body["error"]
    assert "automatically" in body["error"]
    assert "build-bm25" not in body["error"]


def test_stored_method_branch_with_auto_indexing_off_names_the_build():
    """With auto-indexing off nothing dispatches, so the build must be requested.

    That endpoint 400s unless the KB's STORED method is hybrid or full_text, so
    the remedy names that step first.
    """
    body = _timeout_503(
        _kb("chunk_embed", "vector_search"), request_method="full_text", auto_indexing=False
    )
    assert "stored retrieval method" in body["error"]
    assert "hybrid or full_text" in body["error"]
    assert "build-bm25" in body["error"]
    assert "automatically" not in body["error"]


def test_unresolvable_kb_falls_back_to_the_generic_remedy():
    body = _timeout_503((None, 404))
    assert _is_generic(body["error"])


def test_failed_kb_lookup_falls_back_to_the_generic_remedy_and_logs_the_cause(caplog):
    """The lookup runs after a cancelled statement; if it fails, say why.

    Silently swallowing it would hide, for instance, an aborted transaction left
    behind by a savepoint rollback that stopped working.
    """
    with caplog.at_level(logging.WARNING, logger=kb_route.logger.name):
        body = _timeout_503(fetch_error=RuntimeError("lookup exploded"))
    assert _is_generic(body["error"])
    records = [r for r in caplog.records if "keyword-timeout remedy" in r.getMessage()]
    assert len(records) == 1
    assert "lookup exploded" in records[0].getMessage()
    assert records[0].exc_info is not None


def _item() -> RetrievedItem:
    return RetrievedItem(
        item_id="v1",
        text="a vector hit",
        score=0.9,
        source_id=None,
        knowledge_base_id="kb",
        meta={},
    )


def _post(client, kb_id: str, method: str = "hybrid"):
    return client.post(
        f"/api/knowledge-bases/{kb_id}/search",
        headers={"Authorization": "Bearer fake.jwt.token"},
        json={"query": "weather", "retrieval_method": method},
    )


@patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"})
@patch("agentic_project_service.routes.knowledge_bases.db")
@patch("agentic_project_service.services.knowledge_search.search_knowledge_base")
def test_degraded_hybrid_search_reports_the_dropped_leg(mock_search, _db, _jwt):
    """A hybrid answer built from vector results alone must say so.

    Otherwise it is indistinguishable from a healthy hybrid answer: the items
    still carry retrieval_method="hybrid" and nothing else changes.
    """
    kb_id = str(uuid.uuid4())

    def degrade(**kwargs):
        bvs.record_retrieval_degradation(bvs.KEYWORD_SEARCH_TIMEOUT)
        return [_item()]

    mock_search.side_effect = degrade

    with _app().test_client() as c:
        resp = _post(c, kb_id)

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["degraded"] == ["keyword_search_timeout"]
    # The resolved method is unchanged: the search really did run as hybrid.
    assert body["retrieval_method"] == "hybrid"


@patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"})
@patch("agentic_project_service.routes.knowledge_bases.db")
@patch("agentic_project_service.services.knowledge_search.search_knowledge_base")
def test_healthy_search_omits_the_degraded_field(mock_search, _db, _jwt):
    kb_id = str(uuid.uuid4())
    mock_search.return_value = [_item()]

    with _app().test_client() as c:
        resp = _post(c, kb_id)

    assert resp.status_code == 200
    assert "degraded" not in resp.get_json()


@patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"})
@patch("agentic_project_service.routes.knowledge_bases.db")
@patch("agentic_project_service.services.knowledge_search.search_knowledge_base")
def test_degradation_does_not_leak_into_the_next_request(mock_search, _db, _jwt):
    """flask.g is app-context-scoped, not request-scoped.

    Under an outer app context -- which a test client inherits, and which the
    service can hold across several searches -- g survives from one request to
    the next, so the route has to clear the record before it dispatches. Run
    inside `with app.app_context()` precisely so this is not vacuous.
    """
    kb_id = str(uuid.uuid4())
    calls = {"n": 0}

    def maybe_degrade(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            bvs.record_retrieval_degradation(bvs.KEYWORD_SEARCH_TIMEOUT)
        return [_item()]

    mock_search.side_effect = maybe_degrade

    app = _app()
    with app.app_context(), app.test_client() as c:
        first = _post(c, kb_id)
        second = _post(c, kb_id)

    assert first.get_json()["degraded"] == ["keyword_search_timeout"]
    assert "degraded" not in second.get_json()


def test_recording_a_degradation_outside_a_request_is_a_no_op():
    """Celery tasks and bare threads have no request context to write to."""
    bvs.record_retrieval_degradation(bvs.KEYWORD_SEARCH_TIMEOUT)
    assert bvs.get_retrieval_degradations() == []


def test_reset_clears_a_recorded_degradation():
    app = _app()
    with app.test_request_context("/"):
        bvs.record_retrieval_degradation(bvs.KEYWORD_SEARCH_TIMEOUT)
        assert bvs.get_retrieval_degradations() == ["keyword_search_timeout"]
        bvs.reset_retrieval_degradations()
        assert bvs.get_retrieval_degradations() == []


def test_reset_outside_a_request_is_a_no_op():
    bvs.reset_retrieval_degradations()


def test_reasons_are_deduplicated_and_ordered_on_read():
    """Worker threads share one g, so writes race; dedupe where it is safe.

    context_handler runs retrieval in a ThreadPoolExecutor over a copied
    context, so several threads append to the same list. Deduplicating on read
    means a lost update or a duplicated append cannot change the output.
    """
    app = _app()
    with app.test_request_context("/"):
        for reason in ("zzz_other", bvs.KEYWORD_SEARCH_TIMEOUT, "zzz_other"):
            bvs.record_retrieval_degradation(reason)
        assert bvs.get_retrieval_degradations() == ["keyword_search_timeout", "zzz_other"]
