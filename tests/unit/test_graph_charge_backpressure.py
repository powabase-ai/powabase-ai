import asyncio as _real_asyncio
from unittest.mock import MagicMock

# Note: the make_per_batch_billing_callback unit tests that used to live here
# (abort-on-402, continue-on-success, idempotency-key fingerprinting) exercised
# the private services.billing_cloud.credits_client adapter, which this OSS
# build excludes. Removed along with the rest of the billing_cloud test suite.
# The tests below drive the same per-batch backpressure contract through the
# OSS-shipped billing_port facade instead.


def test_reenrich_uses_single_event_loop(monkeypatch):
    """reenrich_graph_references must not call asyncio.run per ToC — that churns
    the global LiteLLM LoggingWorker queue across loops (the #445 OOM vector).

    Arranges 3 ToCs with 2 nodes each, stubs every heavy collaborator, and
    counts asyncio.run calls. The refactored implementation must call it
    exactly once (one _run_all_tocs() coroutine) regardless of ToC count.
    """
    import agentic_project_service.tasks.indexing as idx

    # --- count asyncio.run calls via a wrapping shim ---
    # Capture the real asyncio.run BEFORE patching so the shim doesn't recurse.
    _original_run = _real_asyncio.run
    run_calls = {"n": 0}

    def _counting_run(coro):
        run_calls["n"] += 1
        return _original_run(coro)

    monkeypatch.setattr(idx.asyncio, "run", _counting_run)

    # --- stub get_knowledge_base ---
    monkeypatch.setattr(
        idx,
        "get_knowledge_base",
        lambda _kb_id: {
            "id": "kb-1",
            "name": "Test KB",
            "description": "",
            "indexing_config": {"strategy": "graph_index"},
            "retrieval_config": {},
        },
    )

    # --- stub GraphIndexStore ---
    fake_nodes = [
        {
            "node_id": "n1",
            "title": "Node 1",
            "text": "body",
            "meta": {},
            "indexed_source_id": "is-1",
        },
        {
            "node_id": "n2",
            "title": "Node 2",
            "text": "body",
            "meta": {},
            "indexed_source_id": "is-1",
        },
    ]
    fake_tocs = [{"id": f"toc-{i}"} for i in range(3)]

    class FakeGraphIndexStore:
        def __init__(self, *a, **kw):
            pass

        def get_tocs(self):
            return fake_tocs

        def get_all_nodes_for_toc(self, toc_id):
            return fake_nodes

        def update_node_embedding(self, *a, **kw):
            pass

    monkeypatch.setattr(idx, "GraphIndexStore", FakeGraphIndexStore)

    # --- stub enrich_referenced_nodes (async) ---
    async def _fake_enrich(**kwargs):
        return ({}, [])

    monkeypatch.setattr(
        idx,
        "enrich_referenced_nodes" if hasattr(idx, "enrich_referenced_nodes") else "_noop",
        _fake_enrich,
        raising=False,
    )
    # Also patch inside the services module that reenrich_graph_references imports at call time
    import agentic_project_service.services.graph_enricher as ge_mod

    monkeypatch.setattr(ge_mod, "enrich_referenced_nodes", _fake_enrich)

    # Patch the lazy import inside the function body
    import sys
    import types

    fake_ge_module = types.ModuleType("agentic_project_service.services.graph_enricher")
    fake_ge_module.enrich_referenced_nodes = _fake_enrich
    monkeypatch.setitem(
        sys.modules,
        "agentic_project_service.services.graph_enricher",
        ge_mod,  # keep the real module but with patched function above
    )

    # --- stub LiteLLMEmbedder ---
    class FakeEmbedder:
        def __init__(self, *a, **kw):
            pass

        async def aembed_batch(self, texts):
            return [[0.1, 0.2, 0.3]] * len(texts)

    import agentic.knowledge.embedder as emb_mod

    monkeypatch.setattr(emb_mod, "LiteLLMEmbedder", FakeEmbedder)
    # Patch the name as imported inside reenrich_graph_references (lazy import)
    monkeypatch.setattr(
        idx,
        "LiteLLMEmbedder" if hasattr(idx, "LiteLLMEmbedder") else "_noop",
        FakeEmbedder,
        raising=False,
    )

    # --- stub db.session ---
    fake_session = MagicMock()
    monkeypatch.setattr(idx.db, "session", fake_session, raising=False)

    # --- stub post-loop helpers ---
    monkeypatch.setattr(idx, "update_indexed_source_config_snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(idx, "update_indexed_source_status", lambda *a, **kw: None)
    monkeypatch.setattr(
        idx,
        "ensure_embedding_index" if hasattr(idx, "ensure_embedding_index") else "_noop",
        lambda *a, **kw: None,
        raising=False,
    )

    # Also patch ensure_embedding_index inside base_vector_store (lazy-imported in the loop)
    import agentic_project_service.services.base_vector_store as bvs_mod

    monkeypatch.setattr(bvs_mod, "ensure_embedding_index", lambda *a, **kw: None)

    # Per-batch charging now flows through the billing port; the default no-op
    # adapter returns None (no callback), which is fine — this test only counts
    # asyncio.run calls, and _fake_enrich never invokes on_batch_complete.

    # --- invoke via .run() which bypasses Celery delivery wiring ---
    result = idx.reenrich_graph_references.run(
        knowledge_base_id="kb-1",
        retry_failed=False,
        indexed_source_id=None,
        provider_keys=None,
        billing_idempotency_key=None,
        billing_org_id=None,
        billing_project_id=None,
    )

    assert result["status"] == "success", f"task returned error: {result}"
    assert run_calls["n"] == 1, (
        f"expected exactly one asyncio.run for the whole job (single event loop), "
        f"got {run_calls['n']} — each extra call is a new event loop that churns "
        f"the LiteLLM LoggingWorker queue (the #445 OOM vector)"
    )


def test_reenrich_abort_on_402_stops_toc_loop(monkeypatch):
    """When billing returns 402 on the first batch, the ToC loop must stop after
    the first ToC: ToC 2 and 3 must never be enriched, and Stage 3 re-embed
    must be skipped even for the aborted ToC.

    Counterfactual: without the `if aborted["flag"]: break` guard, all 3 ToCs
    would be enriched and embed would run for each, so the assertions below
    would fail.
    """
    import agentic_project_service.tasks.indexing as idx

    # --- stub get_knowledge_base ---
    monkeypatch.setattr(
        idx,
        "get_knowledge_base",
        lambda _kb_id: {
            "id": "kb-abort",
            "name": "Abort KB",
            "description": "",
            "indexing_config": {"strategy": "graph_index"},
            "retrieval_config": {},
        },
    )

    # --- stub GraphIndexStore: 3 ToCs, 2 nodes each ---
    fake_nodes = [
        {
            "node_id": "n1",
            "title": "Node 1",
            "text": "body",
            "meta": {},
            "indexed_source_id": "is-1",
        },
        {
            "node_id": "n2",
            "title": "Node 2",
            "text": "body",
            "meta": {},
            "indexed_source_id": "is-1",
        },
    ]
    fake_tocs = [{"id": f"toc-{i}"} for i in range(3)]
    tocs_enriched = []

    class FakeGraphIndexStore:
        def __init__(self, *a, **kw):
            pass

        def get_tocs(self):
            return fake_tocs

        def get_all_nodes_for_toc(self, toc_id):
            return fake_nodes

        def update_node_embedding(self, *a, **kw):
            pass

    monkeypatch.setattr(idx, "GraphIndexStore", FakeGraphIndexStore)

    # --- stub enrich_referenced_nodes: records which ToC was enriched ---
    async def _fake_enrich(**kwargs):
        tocs_enriched.append(kwargs["toc_id"])
        # Invoke on_batch_complete so graph_batch_cb fires (simulates a real batch
        # of 1 successful item with a non-empty id list).
        cb = kwargs.get("on_batch_complete")
        if cb is not None:
            cb(1, ["node-id-0"])
        return ({"n1": ["n2"]}, [])

    import agentic_project_service.services.graph_enricher as ge_mod

    monkeypatch.setattr(ge_mod, "enrich_referenced_nodes", _fake_enrich)
    import sys

    monkeypatch.setitem(sys.modules, "agentic_project_service.services.graph_enricher", ge_mod)

    # --- stub LiteLLMEmbedder: spy on aembed_batch calls ---
    embed_calls = {"n": 0}

    class FakeEmbedder:
        def __init__(self, *a, **kw):
            pass

        async def aembed_batch(self, texts):
            embed_calls["n"] += 1
            return [[0.1, 0.2, 0.3]] * len(texts)

    import agentic.knowledge.embedder as emb_mod

    monkeypatch.setattr(emb_mod, "LiteLLMEmbedder", FakeEmbedder)

    # --- stub db.session ---
    fake_session = MagicMock()
    monkeypatch.setattr(idx.db, "session", fake_session, raising=False)

    # --- stub post-loop helpers ---
    monkeypatch.setattr(idx, "update_indexed_source_config_snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(idx, "update_indexed_source_status", lambda *a, **kw: None)

    import agentic_project_service.services.base_vector_store as bvs_mod

    monkeypatch.setattr(bvs_mod, "ensure_embedding_index", lambda *a, **kw: None)

    # --- per-batch callback aborts on the first batch (via the billing port) ---
    # RecordingBillingAdapter(per_batch_abort_after=0) reproduces the cloud
    # callback's insufficient-credits backpressure: the graph per-batch callback
    # returns 'abort' on batch 0. The autouse _billing_adapter_isolation fixture
    # restores the default adapter afterward.
    from agentic_project_service.services import billing_port
    from tests.support.billing import RecordingBillingAdapter

    billing_port.set_billing_adapter(RecordingBillingAdapter(per_batch_abort_after=0))
    result = idx.reenrich_graph_references.run(
        knowledge_base_id="kb-abort",
        retry_failed=False,
        indexed_source_id=None,
        provider_keys=None,
    )

    assert result["status"] == "success", f"task returned unexpected error: {result}"

    # Only the first ToC must have been enriched; ToCs 1 and 2 must not run.
    assert tocs_enriched == [
        "toc-0"
    ], f"expected only toc-0 to be enriched before abort, got {tocs_enriched}"

    # Stage 3 re-embed must have been skipped for the aborted ToC.
    assert embed_calls["n"] == 0, (
        f"expected aembed_batch to never be called on abort (embed skipped), "
        f"got {embed_calls['n']} call(s)"
    )
