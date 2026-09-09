"""What the create/update routes accept into the two JSONB config columns.

The columns are unvalidated JSONB, so whatever the route writes is what every
later read has to survive. Two shapes are worth rejecting at the boundary:

*A non-object.* ``retrieval_config`` has been guarded since the round-4 fix;
``indexing_config`` was not, so ``PATCH {"indexing_config": "graph_index"}``
returned 200 and wrote ``'"graph_index"'`` — the row that then makes a
graph_index knowledge base search the wrong table and report success.

*A graph_expansion outside its bounds.* The read path clamps and warns, so a
``max_referenced_nodes`` of 1000000 was accepted, read back by Studio as
1000000, and silently applied as 100. A ceiling the API accepts and then
ignores is not a contract.

These run against the pure validator rather than the routes, so they run in
CI — ``tests/test_knowledge_bases.py`` needs Postgres and no CI job runs it.
"""

from __future__ import annotations

import pytest

from agentic_project_service.routes.knowledge_bases import _config_shape_error
from agentic_project_service.strategies.graph_defaults import (
    GRAPH_MAX_CHILDREN_CEILING,
    GRAPH_MAX_REFERENCED_CEILING,
)


class TestConfigShape:
    def test_well_formed_configs_pass(self):
        assert (
            _config_shape_error(
                {
                    "indexing_config": {"strategy": "graph_index"},
                    "retrieval_config": {"method": "hybrid", "top_k": 5},
                }
            )
            is None
        )

    def test_absent_configs_pass(self):
        """PATCH is partial — a payload that touches neither column is fine."""
        assert _config_shape_error({"name": "Contracts"}) is None

    @pytest.mark.parametrize("value", ["graph_index", 5, True, ["graph_index"]])
    def test_a_non_object_indexing_config_is_rejected(self, value):
        error = _config_shape_error({"indexing_config": value})

        assert error is not None
        assert "indexing_config" in error

    @pytest.mark.parametrize("value", ["hybrid", 5, ["hybrid"]])
    def test_a_non_object_retrieval_config_is_rejected(self, value):
        error = _config_shape_error({"retrieval_config": value})

        assert error is not None
        assert "retrieval_config" in error


class TestGraphExpansionBounds:
    def _error(self, graph_expansion):
        return _config_shape_error({"retrieval_config": {"graph_expansion": graph_expansion}})

    def test_the_shipped_defaults_pass(self):
        from agentic_project_service.strategies.registry import STRATEGY_REGISTRY

        cfg = STRATEGY_REGISTRY["graph_index"]["default_retrieval_config"]["graph_expansion"]

        assert self._error(cfg) is None

    def test_an_absent_block_passes(self):
        assert _config_shape_error({"retrieval_config": {"method": "hybrid"}}) is None

    def test_a_non_object_block_is_rejected(self):
        error = self._error("off")

        assert error is not None
        assert "graph_expansion" in error

    def test_a_reference_cap_above_the_ceiling_is_rejected(self):
        error = self._error({"max_referenced_nodes": GRAPH_MAX_REFERENCED_CEILING + 1})

        assert error is not None
        assert "max_referenced_nodes" in error
        assert str(GRAPH_MAX_REFERENCED_CEILING) in error

    def test_a_child_cap_above_the_ceiling_is_rejected(self):
        error = self._error({"max_children_per_parent": GRAPH_MAX_CHILDREN_CEILING + 1})

        assert error is not None
        assert "max_children_per_parent" in error
        assert str(GRAPH_MAX_CHILDREN_CEILING) in error

    def test_the_ceilings_themselves_are_accepted(self):
        assert (
            self._error(
                {
                    "max_referenced_nodes": GRAPH_MAX_REFERENCED_CEILING,
                    "max_children_per_parent": GRAPH_MAX_CHILDREN_CEILING,
                }
            )
            is None
        )

    def test_a_negative_cap_is_rejected(self):
        error = self._error({"max_referenced_nodes": -1})

        assert error is not None
        assert "max_referenced_nodes" in error

    @pytest.mark.parametrize("value", ["10", 10.5, True, None])
    def test_a_non_integer_cap_is_rejected(self, value):
        """``True`` is an int in Python and would clamp to 1 — a cap of one
        reference is not what an operator writing ``true`` meant."""
        error = self._error({"max_referenced_nodes": value})

        assert error is not None
        assert "max_referenced_nodes" in error

    @pytest.mark.parametrize("value", ["false", 0, 1, None])
    def test_a_non_boolean_switch_is_rejected(self, value):
        """``"false"`` is truthy — the string that enables the flood the
        default exists to prevent."""
        error = self._error({"include_children": value})

        assert error is not None
        assert "include_children" in error

    def test_both_switches_accept_booleans(self):
        assert self._error({"include_children": True, "include_doc_toc": False}) is None
