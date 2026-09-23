"""The row limit is interpolated into the SQL now, so it is validated — pin both ends.

``vector_search`` interpolates ``top_k`` into ``LIMIT`` (a bound limit leaves a
prepared statement's generic plan unable to match the partial index predicate),
so it raises ``ValueError`` before any SQL for a limit it will not interpolate,
and for a knowledge base id that is not a UUID. Two things that follows from
were claimed and untested: that the search route turns that into a 400, and that
the ceiling a caller actually faces is half of ``MAX_TOP_K``, because hybrid
search doubles the limit before the check.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_project_service.routes import knowledge_bases as kb_route
from agentic_project_service.services import base_vector_store as bvs
from agentic_project_service.services.base_vector_store import MAX_TOP_K, BasePgVectorStore

KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


class _FakeStore(BasePgVectorStore):
    TABLE = "chunks"
    TEXT_COL = "text"
    SEARCH_TEXT_COL = "text"


def _spy_session() -> tuple[MagicMock, list[str]]:
    session = MagicMock()
    executed: list[str] = []

    def execute(statement, params=None):
        executed.append(statement.text if hasattr(statement, "text") else str(statement))
        return iter([])

    session.execute = execute
    return session, executed


# ---------------------------------------------------------------------------
# The store's own guard, and that it fires before any SQL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top_k", [MAX_TOP_K + 1, -1, "many", None])
def test_a_limit_that_cannot_be_interpolated_raises_before_any_sql(top_k):
    session, executed = _spy_session()
    store = _FakeStore(db_session=session, knowledge_base_id=KB)
    with pytest.raises(ValueError, match="top_k"):
        asyncio.run(store.vector_search(embedding=[0.0] * 1536, top_k=top_k))
    assert executed == [], f"the guard has to precede the SQL, executed: {executed}"


def test_a_knowledge_base_id_that_is_not_a_uuid_raises_before_any_sql():
    session, executed = _spy_session()
    store = _FakeStore(db_session=session, knowledge_base_id="kb-test")
    with pytest.raises(ValueError, match="knowledge_base_id"):
        asyncio.run(store.vector_search(embedding=[0.0] * 1536, top_k=10))
    assert executed == [], f"the guard has to precede the SQL, executed: {executed}"


# ---------------------------------------------------------------------------
# The boundary a caller actually faces
# ---------------------------------------------------------------------------


def _hybrid(top_k: int) -> list:
    session, _ = _spy_session()
    store = _FakeStore(db_session=session, knowledge_base_id=KB)
    # The keyword leg is a different mechanism; this is about what the vector
    # leg is handed.
    store.keyword_search_for_hybrid = AsyncMock(return_value=[])
    return asyncio.run(store.hybrid_search("weather", [0.0] * 1536, top_k=top_k))


def test_hybrid_search_halves_the_effective_ceiling():
    """It fetches ``top_k * 2`` candidates per leg before fusing them.

    So the largest ``top_k`` a hybrid caller can ask for is MAX_TOP_K // 2, and
    the number in the error names the doubled value -- which is exactly the
    surprise worth pinning, because nothing else says the documented limit is
    not the limit.
    """
    ceiling = MAX_TOP_K // 2
    assert _hybrid(ceiling) == []

    with pytest.raises(ValueError) as exc:
        _hybrid(ceiling + 1)
    assert str(MAX_TOP_K + 2) in str(exc.value), str(exc.value)


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def _app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(kb_route.knowledge_bases_bp)
    return app


def _search(kb_id: str, body: dict, *, top_k_guard: bool = True):
    """POST the search route with the store's real ``top_k`` guard in the way.

    The guard is applied where the store applies it rather than raised by hand,
    so this fails both if the route stops mapping ValueError to 400 and if the
    store stops rejecting the value.
    """

    def fake_search(**kwargs):
        if top_k_guard:
            bvs.validated_top_k(kwargs["top_k"])
        return []

    with (
        patch("agentic_project_service.auth.decode_jwt", return_value={"role": "service_role"}),
        patch.object(kb_route, "db"),
        patch(
            "agentic_project_service.services.knowledge_search.search_knowledge_base",
            side_effect=fake_search,
        ),
        _app().test_client() as client,
    ):
        return client.post(
            f"/api/knowledge-bases/{kb_id}/search",
            headers={"Authorization": "Bearer fake.jwt.token"},
            json=body,
        )


def test_a_top_k_the_store_refuses_is_a_400_naming_the_limit():
    resp = _search(KB, {"query": "weather", "top_k": MAX_TOP_K + 1})
    assert resp.status_code == 400, resp.data[:300]
    error = resp.get_json()["error"]
    assert "top_k" in error and str(MAX_TOP_K) in error, error


def test_an_accepted_top_k_still_searches():
    """Otherwise the 400 above would pass against a route that 400s everything."""
    resp = _search(KB, {"query": "weather", "top_k": 10})
    assert resp.status_code == 200, resp.data[:300]
    assert resp.get_json()["total_results"] == 0


def test_a_non_uuid_knowledge_base_id_is_refused_by_the_route_itself():
    """It never reaches the store, so the status is the route's 404, not a 400.

    Worth pinning either way: the store's new UUID guard would otherwise be the
    first thing a caller hits with a bad id, and it raises ValueError -- which
    this route maps to 400. Two different answers for one bad input, decided by
    which guard runs first.
    """
    resp = _search("kb-test", {"query": "weather"})
    assert resp.status_code == 404, resp.data[:300]
    assert "knowledge base id" in resp.get_json()["error"].lower()


def test_the_route_keeps_reporting_a_uuid_it_normalises_the_same_way():
    """A braced UUID is accepted by both guards and must not 404 or 400."""
    resp = _search("{" + KB + "}", {"query": "weather", "top_k": 5})
    assert resp.status_code == 200, resp.data[:300]
    # And the store's literal for it is the canonical form, not the braced text.
    assert bvs.kb_sql_literal("{" + KB + "}") == f"'{uuid.UUID(KB)}'"
