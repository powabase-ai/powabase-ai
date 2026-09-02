"""A knowledge base whose config columns hold the wrong shape must not take
down retrieval for every other knowledge base in the request.

``retrieval_config`` is unvalidated JSONB. The route now rejects a non-object
on write, but rows written before that are already out there, and a JSON
string is perfectly valid JSONB. Reading one raw and then calling ``.get()``
on it raises ``AttributeError`` — and in ``execute_retrieval`` those reads
happen outside the per-KB ``try``, so one bad row fails the whole multi-KB
retrieval rather than degrading a single knowledge base.
"""

from __future__ import annotations

import logging

from agentic_project_service.services.context_handler import _load_kb_configs


def _row(kb_id, name="KB", indexing_config=None, retrieval_config=None):
    """A row shaped like the SELECT in execute_retrieval."""
    return (kb_id, name, indexing_config, retrieval_config)


class TestLoadKbConfigs:
    def test_reads_well_formed_configs(self):
        display, retrieval = _load_kb_configs(
            [
                _row(
                    "kb-1",
                    name="Contracts",
                    indexing_config={"strategy": "graph_index", "chunk_size": 800, "overlap": 40},
                    retrieval_config={"method": "hybrid", "top_k": 7},
                )
            ],
            include_chunking=True,
        )

        assert display["kb-1"] == {
            "kb_name": "Contracts",
            "indexing_strategy": "graph_index",
            "chunk_size": 800,
            "overlap": 40,
        }
        assert retrieval["kb-1"] == {"method": "hybrid", "top_k": 7}

    def test_omits_chunking_fields_when_not_requested(self):
        display, _ = _load_kb_configs(
            [_row("kb-1", indexing_config={"strategy": "chunk_embed", "chunk_size": 800})],
            include_chunking=False,
        )

        assert display["kb-1"] == {"kb_name": "KB", "indexing_strategy": "chunk_embed"}

    def test_nulls_read_as_empty_configs(self):
        display, retrieval = _load_kb_configs([_row("kb-1", name=None)], include_chunking=False)

        assert display["kb-1"]["kb_name"] == ""
        assert display["kb-1"]["indexing_strategy"] == "chunk_embed"
        assert retrieval["kb-1"] == {}

    def test_a_string_retrieval_config_degrades_to_empty(self, caplog):
        with caplog.at_level(logging.WARNING):
            _, retrieval = _load_kb_configs(
                [_row("kb-1", retrieval_config="hybrid")], include_chunking=False
            )

        assert retrieval["kb-1"] == {}
        assert "kb-1" in caplog.text

    def test_a_string_indexing_config_degrades_to_empty(self):
        display, _ = _load_kb_configs(
            [_row("kb-1", indexing_config="graph_index")], include_chunking=True
        )

        assert display["kb-1"]["indexing_strategy"] == "chunk_embed"

    def test_one_poisoned_row_does_not_hide_the_others(self):
        """The failure mode this exists to prevent: every other knowledge base
        in the same request disappearing because of one bad row."""
        display, retrieval = _load_kb_configs(
            [
                _row("kb-good", retrieval_config={"method": "hybrid"}),
                _row("kb-bad", retrieval_config="not-an-object"),
            ],
            include_chunking=False,
        )

        assert set(retrieval) == {"kb-good", "kb-bad"}
        assert retrieval["kb-good"] == {"method": "hybrid"}
        assert set(display) == {"kb-good", "kb-bad"}

    def test_loaded_configs_are_safe_to_call_get_on(self):
        """What the callers actually do with these, at eight sites, several of
        them outside any try block."""
        _, retrieval = _load_kb_configs(
            [_row("kb-1", retrieval_config="hybrid")], include_chunking=False
        )

        assert retrieval.get("kb-1", {}).get("method", "vector_search") == "vector_search"
        assert retrieval.get("kb-1", {}).get("context_mode", "text") == "text"
