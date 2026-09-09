"""``retrieval_config`` and ``indexing_config`` cannot share a coercion.

Both columns are unvalidated JSONB, and both can hold a JSON string written
before the route rejected one. Reading either raw and calling ``.get()`` on
it raises — but the two failures are not equivalent, so the repair for them
is not either.

An empty ``retrieval_config`` means "use the defaults", which is a correct
answer. An empty ``indexing_config`` means "this is a chunk_embed knowledge
base", which routes the search at a different table: a graph_index KB whose
data lives in ``graph_index_nodes`` runs against ``chunks``, finds nothing,
and reports success — HTTP 200, ``errors: []``, zero results, and
``indexing_strategy: chunk_embed`` in the response. That is a wrong fact
rather than an omitted one, and the operator's only trace is a log line with
no request id. Coercing it hides a failure the pre-coercion ``AttributeError``
at least surfaced in ``errors[]``.
"""

from __future__ import annotations

import pytest

from agentic_project_service.services import knowledge_search


class _KbRowSession:
    """Answers the KB-config SELECT, and the item COUNT(*) that follows it."""

    def __init__(self, row):
        self._row = row

    def execute(self, *args, **kwargs):
        row = self._row

        class _Result:
            def fetchone(self):
                return row

            def scalar(self):
                return 1

        return _Result()


def _search(indexing_config, retrieval_config):
    return knowledge_search.search_knowledge_base(
        db_session=_KbRowSession(("kb-1", "KB", indexing_config, retrieval_config)),
        knowledge_base_id="kb-1",
        query="q",
    )


def test_a_malformed_indexing_config_raises_instead_of_rerouting():
    with pytest.raises(ValueError) as excinfo:
        _search("graph_index", {"method": "hybrid"})

    message = str(excinfo.value)
    assert "indexing_config" in message
    assert "kb-1" in message
    assert "str" in message, "the message has to name the shape that was stored"


def test_a_malformed_retrieval_config_still_degrades_to_defaults(monkeypatch):
    """The other half of the coercion has to keep working — defaults are a
    correct answer for a retrieval_config that cannot be read."""
    monkeypatch.setattr(knowledge_search, "_execute_retrieval_pipeline", lambda **kw: [])

    assert _search({"strategy": "chunk_embed"}, "hybrid") == []


def test_a_null_indexing_config_still_means_defaults(monkeypatch):
    """NULL is the ordinary pre-registry shape, not a malformed one."""
    monkeypatch.setattr(knowledge_search, "_execute_retrieval_pipeline", lambda **kw: [])

    assert _search(None, {"method": "hybrid"}) == []
