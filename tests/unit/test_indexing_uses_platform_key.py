"""Guardrail tests: platform-billed indexing strategies must not consult the
provider-keys resolver. They use the platform env key, matching the other
platform-billed paths (metadata_enrichment, query_enrichment, page_index, doc2json).
"""


def test_run_indexing_does_not_call_resolver():
    """run_indexing (chunk_embed) must use env key, not user's BYOK."""
    from agentic_project_service.tasks import indexing as indexing_mod
    import inspect

    src = inspect.getsource(indexing_mod.run_indexing)
    assert "resolve_api_key_for_model" not in src


def test_run_full_document_indexing_does_not_call_resolver():
    from agentic_project_service.tasks import indexing as indexing_mod
    import inspect

    src = inspect.getsource(indexing_mod.run_full_document_indexing)
    assert "resolve_api_key_for_model" not in src


def test_run_graph_index_indexing_does_not_call_resolver():
    from agentic_project_service.tasks import indexing as indexing_mod
    import inspect

    src = inspect.getsource(indexing_mod.run_graph_index_indexing)
    assert "resolve_api_key_for_model" not in src


def test_reenrich_graph_references_does_not_call_resolver():
    """reenrich_graph_references is a 4th platform-billed indexing path
    (action='indexing_graphindex'). Must use env key, not user's BYOK.
    Caught during plan calibration review (F1)."""
    from agentic_project_service.tasks import indexing as indexing_mod
    import inspect

    src = inspect.getsource(indexing_mod.reenrich_graph_references)
    assert "resolve_api_key_for_model" not in src
