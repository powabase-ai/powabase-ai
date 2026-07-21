from unittest.mock import patch
import pytest
from werkzeug.exceptions import ServiceUnavailable


def test_graph_precheck_scales_with_node_count(recording_billing):
    """run_graph_reenrichment precheck must request node_count × per-node credits,
    not the flat 1000."""
    import agentic_project_service.routes.knowledge_bases as kb

    with patch.object(kb, "_count_graph_nodes_for_kb", return_value=6000):
        kb._graph_check_balance("kb-1", indexed_source_id=None)

    assert recording_billing.balance_checks == [6000 * kb._INDEXING_MAX_UNIT_CREDITS]


def test_graph_precheck_fails_closed_on_lookup_error():
    import agentic_project_service.routes.knowledge_bases as kb

    with (
        patch.object(kb, "_count_graph_nodes_for_kb", side_effect=RuntimeError("DB timeout")),
        pytest.raises(ServiceUnavailable),
    ):
        kb._graph_check_balance("kb-1", indexed_source_id=None)
