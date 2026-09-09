"""What the create/update routes accept into the two JSONB config columns.

The columns are unvalidated JSONB, so whatever the route writes is what every
later read has to survive. Two shapes are worth rejecting at the boundary:

*A non-object.* ``retrieval_config`` was guarded first; ``indexing_config``
was not, so ``PATCH {"indexing_config": "graph_index"}`` returned 200 and
wrote ``'"graph_index"'`` — the row that then makes a graph_index knowledge
base search the wrong table and report a completed retrieval with no
results and no error.

*A graph_expansion outside its bounds.* The read path clamps and warns, so a
``max_referenced_nodes`` of 1000000 was accepted, read back by Studio as
1000000, and silently applied as 100. A ceiling the API accepts and then
ignores is not a contract.

Most of these run against the pure validator, so they run in CI —
``tests/test_knowledge_bases.py`` needs Postgres and no CI job runs it. The
last class drives the two routes through a Flask test client, because a
validator nothing calls is dead weight that no unit test of the validator
can notice: both call sites can be deleted with every other test here
still green.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agentic_project_service.routes import knowledge_bases as kb_routes
from agentic_project_service.routes.knowledge_bases import _config_shape_error
from agentic_project_service.strategies.graph_defaults import (
    GRAPH_MAX_CHILDREN_CEILING,
    GRAPH_MAX_REFERENCED_CEILING,
)


class TestConfigShape:
    def test_a_non_object_body_is_rejected(self):
        """``request.get_json()`` returns whatever was sent, and a truthy
        non-dict — ``[1, 2]`` — reaches here on both routes: every read below
        would raise on one, since ``field in data`` is a harmless membership
        test that lets it through as far as ``.get``."""
        error = _config_shape_error([1, 2])

        assert error is not None
        assert "object" in error

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
        """``True`` is an int in Python, so it needs its own check to be
        refused here. The read path already declines to interpret it and
        falls back to the default — so this is not preventing a failure, it
        is refusing to store a value whose stored form and effect disagree."""
        error = self._error({"max_referenced_nodes": value})

        assert error is not None
        assert "max_referenced_nodes" in error

    @pytest.mark.parametrize("value", ["false", 0, 1, None])
    def test_a_non_boolean_switch_is_rejected(self, value):
        """``"false"`` is a string, and the read path returns the default for
        one rather than coercing — so this refuses the value instead of
        storing something whose stored form and effect disagree."""
        error = self._error({"include_children": value})

        assert error is not None
        assert "include_children" in error

    def test_both_switches_accept_booleans(self):
        assert self._error({"include_children": True, "include_doc_toc": False}) is None


class TestTheRoutesCallIt:
    """Wiring, not logic. Every test above passes with both call sites
    deleted; these are the ones that don't. The validator runs before any
    DB access on both routes, so a test client with auth patched is enough.
    """

    def _client(self):
        from flask import Flask

        app = Flask(__name__)
        app.register_blueprint(kb_routes.knowledge_bases_bp)
        return app.test_client()

    def _post(self, body):
        with patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ):
            return self._client().post(
                "/api/knowledge-bases",
                json=body,
                headers={"Authorization": "Bearer fake"},
            )

    def _patch(self, body):
        with patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "authenticated"},
        ):
            return self._client().patch(
                "/api/knowledge-bases/11111111-1111-1111-1111-111111111111",
                json=body,
                headers={"Authorization": "Bearer fake"},
            )

    def test_create_rejects_a_non_object_indexing_config(self):
        """It used to 500 here — ``.get("strategy")`` ran three lines before
        the retrieval_config guard."""
        response = self._post({"name": "KB", "indexing_config": "graph_index"})

        assert response.status_code == 400
        assert "indexing_config" in response.get_json()["error"]

    def test_update_rejects_a_non_object_indexing_config(self):
        """It used to return 200 and write ``'"graph_index"'``."""
        response = self._patch({"indexing_config": "graph_index"})

        assert response.status_code == 400
        assert "indexing_config" in response.get_json()["error"]

    def test_create_rejects_a_cap_above_the_ceiling(self):
        response = self._post(
            {
                "name": "KB",
                "retrieval_config": {"graph_expansion": {"max_referenced_nodes": 1000000}},
            }
        )

        assert response.status_code == 400
        assert "max_referenced_nodes" in response.get_json()["error"]

    def test_update_rejects_a_cap_above_the_ceiling(self):
        response = self._patch(
            {"retrieval_config": {"graph_expansion": {"max_referenced_nodes": 1000000}}}
        )

        assert response.status_code == 400
        assert "max_referenced_nodes" in response.get_json()["error"]

    @pytest.mark.parametrize("body", [[1, 2], "a string", 123])
    def test_update_with_a_non_object_body_is_a_400_not_a_500(self, body):
        """PATCH's emptiness check is ``if not data``, which a truthy non-dict
        passes; the validator's own reads are then what it reaches."""
        assert self._patch(body).status_code == 400

    @pytest.mark.parametrize("body", [[1, 2], "a string", 123])
    def test_create_with_a_non_object_body_is_a_400_not_a_500(self, body):
        """POST's is ``if not data or not data.get("name")``, so the body dies
        one line *before* the validator unless the guard runs first. Every
        other case here is paired across the two routes; this one was not,
        and the unpaired route was the one that still 500'd."""
        assert self._post(body).status_code == 400

    def test_a_malformed_stored_config_does_not_break_the_request_that_repairs_it(self):
        """``_read_existing_retrieval_config`` runs before the UPDATE, to spot
        a method transition. Reading a stored non-object raw would 500 the one
        request that can fix it — leaving no way to repair the row through the
        API at all, which is the failure its docstring names."""
        with patch.object(kb_routes.db, "session") as session:
            session.execute.return_value.fetchone.return_value = ("hybrid",)

            assert kb_routes._read_existing_retrieval_config("kb-1") == {}

    def test_create_with_an_empty_object_still_asks_for_a_name(self):
        """An empty body is not a malformed one — moving the guard earlier
        must not take over the answer the route already gives."""
        response = self._post({})

        assert response.status_code == 400
        assert "Name" in response.get_json()["error"]
