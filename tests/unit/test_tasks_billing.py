"""Tests for billing wiring in Celery background tasks.

Covers tasks/extraction.py, tasks/url_extraction.py, tasks/indexing.py, and
tasks/enrichment.py — the Celery tasks that charge on completion.

Strategy:
  * Patch heavy collaborators (DB queries, storage, asyncio.run wrappers)
  * Call the underlying task function via .run(...) which skips Celery's
    delivery wiring and exercises the function body directly.
  * All four task modules now bill through the billing port
    (services/billing_port.py): install a RecordingBillingAdapter and assert
    on ``rec.charges`` (main charges) / ``rec.per_batch_calls`` (per-batch
    wiring). No task module calls post_charge or make_per_batch_billing_callback
    directly anymore.

Anti-discipline check: idempotency never comes from ``self.request.id`` (which
changes on retry → double-charge). Extraction recomputes ``idempotency_parts``
from its OWN stable args; indexing threads them from the caller (the two caller
families use DIFFERENT key schemes — see the reindex-vs-single-index split
test). The recorded-charge assertions pin those parts.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agentic_project_service.services import billing_port
from tests.support.billing import RecordingBillingAdapter


TASKS_DIR = Path(__file__).resolve().parents[2] / "src" / "agentic_project_service" / "tasks"


# ---------------------------------------------------------------------------
# Signature checks — tasks keep vestigial billing_* params (deploy-compat) and
# accept the threaded key inputs where the caller-scheme requires them.
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text()


def test_extraction_task_accepts_billing_kwargs():
    """extract_source keeps the three ``billing_*`` params — now vestigial but
    retained for deploy-compat (in-flight tasks enqueued before the port
    migration pass them; removing them would TypeError on a cross-deploy
    retry) — and accepts the new ``reextract_seed`` per-call key tail."""
    src = _read(TASKS_DIR / "extraction.py")
    tree = ast.parse(src)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "extract_source"
    )
    args = {a.arg for a in fn.args.args}
    kwonly = {a.arg for a in fn.args.kwonlyargs}
    all_args = args | kwonly
    assert "billing_idempotency_key" in all_args
    assert "billing_org_id" in all_args
    assert "billing_project_id" in all_args
    assert "reextract_seed" in all_args


def test_indexing_task_accepts_billing_kwargs():
    """index_source keeps the three ``billing_*`` params (vestigial, deploy-compat)
    AND accepts the new threaded key inputs (idempotency_action/idempotency_parts)
    that the caller families use to reproduce their distinct pre-port keys."""
    src = _read(TASKS_DIR / "indexing.py")
    tree = ast.parse(src)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "index_source"
    )
    args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    assert "billing_idempotency_key" in args
    assert "billing_org_id" in args
    assert "billing_project_id" in args
    assert "idempotency_action" in args
    assert "idempotency_parts" in args


def test_enrichment_task_accepts_billing_kwargs():
    """enrich_knowledge_base signature must accept billing kwargs."""
    src = _read(TASKS_DIR / "enrichment.py")
    tree = ast.parse(src)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "enrich_knowledge_base"
    )
    args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    assert "billing_idempotency_key" in args
    assert "billing_org_id" in args
    assert "billing_project_id" in args


def test_url_extraction_task_accepts_billing_kwargs():
    """extract_url_source keeps the three ``billing_*`` params (vestigial,
    deploy-compat) and accepts the new ``reextract_seed`` per-call key tail."""
    src = _read(TASKS_DIR / "url_extraction.py")
    tree = ast.parse(src)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "extract_url_source"
    )
    args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    assert "billing_idempotency_key" in args
    assert "billing_org_id" in args
    assert "billing_project_id" in args
    assert "reextract_seed" in args


def test_reindex_kb_task_accepts_billing_kwargs():
    """reindex_knowledge_base forwards billing context to child tasks."""
    src = _read(TASKS_DIR / "indexing.py")
    tree = ast.parse(src)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "reindex_knowledge_base"
    )
    args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    # Reindex doesn't post_charge itself; child tasks do. So it only needs
    # org/project to forward.
    assert "billing_org_id" in args
    assert "billing_project_id" in args


def test_reenrich_graph_task_accepts_billing_kwargs():
    """reenrich_graph_references signature must accept billing kwargs."""
    src = _read(TASKS_DIR / "indexing.py")
    tree = ast.parse(src)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "reenrich_graph_references"
    )
    args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    assert "billing_idempotency_key" in args
    assert "billing_org_id" in args
    assert "billing_project_id" in args


# ---------------------------------------------------------------------------
# Indexing strategy -> billing action mapping.
# ---------------------------------------------------------------------------


def test_resolve_indexing_action_for_each_strategy():
    """Each strategy maps to its catalog action; unknown falls back."""
    from agentic_project_service.tasks.indexing import _resolve_indexing_action

    assert _resolve_indexing_action("chunk_embed") == "indexing_chunkembed"
    assert _resolve_indexing_action("page_index") == "indexing_pageindex"
    assert _resolve_indexing_action("graph_index") == "indexing_graphindex"
    assert _resolve_indexing_action("doc2json") == "indexing_doc2json"
    # full_document maps to chunkembed (no distinct catalog entry).
    assert _resolve_indexing_action("full_document") == "indexing_chunkembed"
    # Unknown strategy falls back to chunkembed (safe default).
    assert _resolve_indexing_action("brand_new_strategy") == "indexing_chunkembed"


def test_quantity_from_stats_prefers_explicit_token_counts():
    """When stats has total_tokens / input_tokens / full_text_tokens, use it."""
    from agentic_project_service.tasks.indexing import _quantity_from_stats

    # 2500 tokens → ceil(2500 / 1000) = 3 units
    assert _quantity_from_stats({"total_tokens": 2500}) == 3
    assert _quantity_from_stats({"input_tokens": 1500}) == 2
    assert _quantity_from_stats({"full_text_tokens": 999}) == 1


def test_quantity_from_stats_falls_back_to_chars():
    """No explicit token count → total_chars / 4 / 1000."""
    from agentic_project_service.tasks.indexing import _quantity_from_stats

    # 20000 chars / 4 = 5000 tokens → 5 units
    assert _quantity_from_stats({"total_chars": 20000}) == 5


def test_quantity_from_stats_minimum_is_one():
    """A successful index never bills 0 (returns at least 1 unit)."""
    from agentic_project_service.tasks.indexing import _quantity_from_stats

    assert _quantity_from_stats({}) == 1
    assert _quantity_from_stats({"total_chars": 0, "total_tokens": 0}) == 1
    assert _quantity_from_stats({"total_chars": 10}) == 1


# ---------------------------------------------------------------------------
# OCR billing — extraction task posts ocr_pages only for OCR methods.
# ---------------------------------------------------------------------------


def test_ocr_extraction_methods_set_matches_pdf_extractor_methods():
    """The OCR allowlist must include the three cloud OCR methods. Sanity-
    check: any new OCR backend in agentic/ingest/extractor/pdf.py must be
    added here too (else its pages don't bill)."""
    from agentic_project_service.tasks.extraction import _OCR_EXTRACTION_METHODS

    assert "mistral_ocr" in _OCR_EXTRACTION_METHODS
    assert "paddleocr_vl" in _OCR_EXTRACTION_METHODS
    assert "lighton_ocr" in _OCR_EXTRACTION_METHODS
    # Non-OCR methods explicitly excluded.
    assert "fitz" not in _OCR_EXTRACTION_METHODS
    assert "pdfplumber" not in _OCR_EXTRACTION_METHODS
    assert "opendataloader" not in _OCR_EXTRACTION_METHODS


# ---------------------------------------------------------------------------
# Behavioral tests: task body actually calls post_charge with expected args.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_session(monkeypatch):
    """Stub the global db.session so task body can run without a real DB."""
    from agentic_project_service.tasks import extraction as ext_mod
    from agentic_project_service.tasks import indexing as idx_mod
    from agentic_project_service.tasks import url_extraction as url_mod
    from agentic_project_service.tasks import enrichment as enr_mod

    fake_session = MagicMock()
    monkeypatch.setattr(ext_mod.db, "session", fake_session, raising=False)
    monkeypatch.setattr(idx_mod.db, "session", fake_session, raising=False)
    monkeypatch.setattr(url_mod.db, "session", fake_session, raising=False)
    monkeypatch.setattr(enr_mod.db, "session", fake_session, raising=False)
    return fake_session


def _ext_source() -> dict:
    return {
        "id": "src-1",
        "name": "x.pdf",
        "file_type": "application/pdf",
        "storage_path": "sources/x.pdf",
        "extraction_status": "pending",
        "derivatives": {},
        "metadata": {},
        "auto_metadata": {},
    }


def _stub_extraction(monkeypatch, ext_mod, mock_db_session, fake_auto_metadata):
    """Stub the heavy collaborators so extract_source.run() reaches the charge."""
    monkeypatch.setattr(ext_mod, "get_source", lambda _: _ext_source())
    monkeypatch.setattr(ext_mod, "update_source_status", lambda *a, **kw: None)
    monkeypatch.setattr(ext_mod, "update_source_extraction_result", lambda *a, **kw: None)
    monkeypatch.setattr(ext_mod, "get_storage", lambda: MagicMock())
    monkeypatch.setattr(
        ext_mod, "asyncio", SimpleNamespace(run=lambda _coro: ({}, fake_auto_metadata))
    )
    # Avoid the second DB scalar() check for cancellation.
    mock_db_session.execute.return_value.scalar.return_value = "extracting"


def test_extraction_charges_ocr_pages_when_method_is_ocr(
    monkeypatch, mock_db_session, recording_billing
):
    """extract_source charges ocr_pages with the actual page_count; the KEY
    (idempotency_action + parts) reproduces the old stable enqueue-time key."""
    from agentic_project_service.tasks import extraction as ext_mod

    _stub_extraction(
        monkeypatch,
        ext_mod,
        mock_db_session,
        {"extraction_method": "mistral_ocr", "page_count": 7, "char_count": 12000},
    )

    ext_mod.extract_source.run(source_id="src-1", bucket_id="sources", extraction_model="mistral")

    assert len(recording_billing.charges) == 1
    charge = recording_billing.charges[0]
    assert charge["action"] == "ocr_pages"  # billed category (actual method)
    assert charge["quantity"] == 7
    assert charge["ref_type"] == "extraction"
    assert charge["ref_id"] == "src-1"
    # requested == actual here, so the key action matches the billed action.
    assert charge["idempotency_action"] == "ocr_pages"
    assert charge["idempotency_parts"] == ("src-1",)


def test_extraction_charges_advanced_ocr_when_method_is_llamaparse(
    monkeypatch, mock_db_session, recording_billing
):
    """Happy path (no fallback): LlamaParse requested AND actual → advanced_ocr
    on BOTH the billed action and the key action."""
    from agentic_project_service.tasks import extraction as ext_mod

    _stub_extraction(
        monkeypatch,
        ext_mod,
        mock_db_session,
        {"extraction_method": "llamaparse_ocr", "page_count": 4, "char_count": 9000},
    )

    ext_mod.extract_source.run(
        source_id="src-1", bucket_id="sources", extraction_model="llamaparse"
    )

    assert len(recording_billing.charges) == 1
    charge = recording_billing.charges[0]
    assert charge["action"] == "advanced_ocr"  # billed
    assert charge["idempotency_action"] == "advanced_ocr"  # key (requested == actual)
    assert charge["idempotency_parts"] == ("src-1",)
    assert charge["quantity"] == 4
    assert charge["metadata"] == {"extraction_method": "llamaparse_ocr"}


def test_extraction_fallback_bills_actual_but_keys_on_requested(
    monkeypatch, mock_db_session, recording_billing
):
    """LOAD-BEARING: LlamaParse requested (extraction_model='llamaparse') but
    the engine fell back to Mistral (method='mistral_ocr'). The charge must be
    BILLED as ocr_pages (the actual, cheaper method) while the idempotency KEY
    stays on advanced_ocr (the requested action) so the key is retry-stable
    across attempts whose actual method varies. This reproduces the pre-port
    behavior where the route stashed a key built from the *requested* action
    and the worker charged the *actual* method's action."""
    from agentic_project_service.tasks import extraction as ext_mod

    _stub_extraction(
        monkeypatch,
        ext_mod,
        mock_db_session,
        {"extraction_method": "mistral_ocr", "page_count": 6, "char_count": 11000},
    )

    ext_mod.extract_source.run(
        source_id="src-1", bucket_id="sources", extraction_model="llamaparse"
    )

    assert len(recording_billing.charges) == 1
    charge = recording_billing.charges[0]
    assert charge["action"] == "ocr_pages"  # BILLED = actual (fallback) method
    assert charge["idempotency_action"] == "advanced_ocr"  # KEY = requested method
    assert charge["idempotency_parts"] == ("src-1",)
    assert charge["quantity"] == 6


def test_extraction_skips_charge_when_method_is_local(
    monkeypatch, mock_db_session, recording_billing
):
    """Non-OCR extraction (fitz/pdfplumber/etc.) → no charge at all."""
    from agentic_project_service.tasks import extraction as ext_mod

    _stub_extraction(
        monkeypatch,
        ext_mod,
        mock_db_session,
        {"extraction_method": "fitz", "page_count": 7, "char_count": 12000},
    )

    ext_mod.extract_source.run(source_id="src-1", bucket_id="sources", extraction_model="auto")

    assert recording_billing.charges == []


def test_extraction_idempotency_parts_stable_across_retries(
    monkeypatch, mock_db_session, recording_billing
):
    """Two invocations (a retry) recompute the SAME idempotency_parts from the
    task's own args → the cloud adapter builds the same key → dedup. The port's
    key derivation never touches self.request.id."""
    from agentic_project_service.tasks import extraction as ext_mod

    _stub_extraction(
        monkeypatch,
        ext_mod,
        mock_db_session,
        {"extraction_method": "mistral_ocr", "page_count": 5},
    )

    for _ in range(2):
        ext_mod.extract_source.run(
            source_id="src-1", bucket_id="sources", extraction_model="mistral"
        )

    parts = [c["idempotency_parts"] for c in recording_billing.charges]
    assert parts == [("src-1",), ("src-1",)]


def test_extraction_reextract_seed_threads_into_idempotency_parts(
    monkeypatch, mock_db_session, recording_billing
):
    """A reextract passes a per-call ``reextract_seed`` so its key is distinct
    from the original upload's (and from other reextracts) — preserving the
    4-part per-call key the route used pre-port. Upload path (no seed) →
    (source_id,); reextract → (source_id, seed)."""
    from agentic_project_service.tasks import extraction as ext_mod

    _stub_extraction(
        monkeypatch,
        ext_mod,
        mock_db_session,
        {"extraction_method": "mistral_ocr", "page_count": 5},
    )

    ext_mod.extract_source.run(
        source_id="src-1",
        bucket_id="sources",
        extraction_model="mistral",
        reextract_seed="seed-xyz",
    )

    assert len(recording_billing.charges) == 1
    assert recording_billing.charges[0]["idempotency_parts"] == ("src-1", "seed-xyz")


# ---------------------------------------------------------------------------
# Indexing behavioral tests: index_source charges the strategy-resolved action
# through the billing port and reproduces the caller's idempotency KEY inputs.
# The reindex-vs-single-index KEY split is load-bearing (see the split test).
# ---------------------------------------------------------------------------


def _stub_index_source(monkeypatch, idx_mod, mock_db_session, *, strategy, stats):
    """Stub index_source's collaborators so ``.run()`` (indexed_source_id=None
    path) reaches the billing charge with a known strategy + stats, without
    running any real extraction/indexing/DB work."""
    monkeypatch.setattr(
        idx_mod,
        "get_knowledge_base",
        lambda _kb: {
            "id": "kb-1",
            "indexing_config": {"strategy": strategy},
            "retrieval_config": {},
        },
    )
    monkeypatch.setattr(
        idx_mod,
        "get_source",
        lambda _s: {
            "id": "src-1",
            "name": "x.pdf",
            "file_type": "application/pdf",
            "storage_path": "sources/x.pdf",
            "extraction_status": "extracted",
            "derivatives": {},
            "metadata": {},
            "auto_metadata": {},
        },
    )
    monkeypatch.setattr(idx_mod, "update_indexed_source_status", lambda *a, **kw: None)
    monkeypatch.setattr(idx_mod, "update_indexed_source_result", lambda *a, **kw: None)
    monkeypatch.setattr(idx_mod, "get_storage", lambda: MagicMock())
    monkeypatch.setattr(idx_mod, "get_text_derivative_content", lambda *a, **kw: "text content")
    monkeypatch.setattr(idx_mod, "get_page_texts_from_derivative", lambda *a, **kw: None)
    monkeypatch.setattr(idx_mod, "init_accumulator", lambda: SimpleNamespace(to_dict=lambda: {}))
    # Strategy dispatch → return the given stats without running async work.
    monkeypatch.setattr(idx_mod, "asyncio", SimpleNamespace(run=lambda _coro: stats))
    # 1st fetchone: the get-or-create indexed_source row (indexed_source_id=None
    # path). 2nd fetchone: the enrichment-config lookup (None → skip auto-enrich).
    mock_db_session.execute.return_value.fetchone.side_effect = [("ix-db",), None]


def test_index_source_single_index_bills_and_keys_on_resolved_action(
    monkeypatch, mock_db_session, recording_billing
):
    """Single-index route scheme: the caller threads idempotency_action ==
    the resolved indexing_<strategy>, so the KEY action equals the BILLED
    action (no split). ref_type/quantity/metadata carry over from the old
    charge."""
    from agentic_project_service.tasks import indexing as idx_mod

    _stub_index_source(
        monkeypatch,
        idx_mod,
        mock_db_session,
        strategy="chunk_embed",
        stats={"total_tokens": 3000, "artifact_count": 5},
    )

    idx_mod.index_source.run(
        "kb-1",
        "src-1",
        indexed_source_id=None,
        idempotency_action="indexing_chunkembed",
        idempotency_parts=["ix-1"],
    )

    assert len(recording_billing.charges) == 1
    charge = recording_billing.charges[0]
    assert charge["action"] == "indexing_chunkembed"  # billed = resolved strategy
    assert charge["idempotency_action"] == "indexing_chunkembed"  # key == billed (no split)
    assert charge["idempotency_parts"] == ("ix-1",)
    assert charge["ref_type"] == "indexing"
    assert charge["quantity"] == 3  # ceil(3000 / 1000)
    assert charge["metadata"] == {"strategy": "chunk_embed", "source_id": "src-1"}


def test_index_source_reindex_child_bills_resolved_but_keys_on_literal_indexing(
    monkeypatch, mock_db_session, recording_billing
):
    """LOAD-BEARING SPLIT: the reindex fan-out / batch / watchdog paths thread
    the literal "indexing" namespace as the KEY action (with (idx_source_id,
    source_id) parts), while the charge is still BILLED as the resolved
    indexing_<strategy>. This reproduces the pre-port behavior where the fan-out
    stashed a key on ("indexing", ...) but the worker charged the strategy's
    action. Collapsing the two would change one path's ledger keys."""
    from agentic_project_service.tasks import indexing as idx_mod

    _stub_index_source(
        monkeypatch,
        idx_mod,
        mock_db_session,
        strategy="chunk_embed",
        stats={"total_tokens": 3000, "artifact_count": 5},
    )

    idx_mod.index_source.run(
        "kb-1",
        "src-1",
        indexed_source_id=None,
        idempotency_action="indexing",
        idempotency_parts=["ix-1", "src-1"],
    )

    assert len(recording_billing.charges) == 1
    charge = recording_billing.charges[0]
    assert charge["action"] == "indexing_chunkembed"  # BILLED = resolved strategy
    assert charge["idempotency_action"] == "indexing"  # KEY = literal namespace (split!)
    assert charge["idempotency_parts"] == ("ix-1", "src-1")  # 2 parts, not 1


def test_reindex_kb_threads_literal_indexing_key_to_children(monkeypatch, mock_db_session):
    """reindex_knowledge_base fans out to index_source, threading the child KEY
    inputs on the literal "indexing" namespace + (idx_source_id, source_id) —
    the reindex side of the split. Identity is the adapter's job; no billing_*
    args are forwarded."""
    from agentic_project_service.tasks import indexing as idx_mod

    # One indexed_source row: (id, source_id).
    select_result = MagicMock()
    select_result.__iter__ = lambda self: iter([("ix-1", "src-1")])
    mock_db_session.execute.return_value = select_result

    captured = []
    monkeypatch.setattr(idx_mod.index_source, "delay", lambda *a, **kw: captured.append(kw))

    idx_mod.reindex_knowledge_base.run("kb-1")

    assert len(captured) == 1
    kw = captured[0]
    assert kw["indexed_source_id"] == "ix-1"
    assert kw["source_id"] == "src-1"
    assert kw["idempotency_action"] == "indexing"
    assert kw["idempotency_parts"] == ["ix-1", "src-1"]
    # No verbatim key / identity forwarded — the port owns identity.
    assert "billing_idempotency_key" not in kw
    assert "billing_org_id" not in kw


def test_index_source_charge_insufficient_credits_does_not_fail_op(monkeypatch, mock_db_session):
    """A charge returning insufficient_credits after the index artifacts are
    committed must NOT flip the op to error. Per spec line 54 a post-success
    402 is bounded over-serve, not an op failure (replaces the old
    post_charge-result-not-captured AST guard with a behavioral one)."""
    from agentic_project_service.tasks import indexing as idx_mod

    _stub_index_source(
        monkeypatch,
        idx_mod,
        mock_db_session,
        strategy="chunk_embed",
        stats={"total_tokens": 3000, "artifact_count": 5},
    )
    failed_status: list = []
    monkeypatch.setattr(
        idx_mod,
        "_mark_indexed_source_failed",
        lambda *a, **kw: failed_status.append((a, kw)),
    )

    billing_port.set_billing_adapter(RecordingBillingAdapter(insufficient=True))
    result = idx_mod.index_source.run(
        "kb-1",
        "src-1",
        indexed_source_id=None,
        idempotency_action="indexing_chunkembed",
        idempotency_parts=["ix-1"],
    )

    assert result["status"] == "success"
    assert failed_status == [], f"index op must not fail on post-success 402; got {failed_status}"


def test_enrichment_wires_per_batch_metadata_enrichment(
    monkeypatch, mock_db_session, recording_billing
):
    """enrich_knowledge_base wires the per-batch callback through the billing
    port with action='metadata_enrichment' and enabled defaulting True (the
    adapter ctx-gates when billing is off). This replaces the AST guard that
    pinned the old make_per_batch_billing_callback import."""
    from agentic_project_service.tasks import enrichment as enr_mod

    monkeypatch.setattr(
        enr_mod,
        "_get_enrichment_config",
        lambda _kb: {
            "id": "cfg-1",
            "fields": [{"name": "f"}],
            "llm_model": None,
            "metadata_table_name": "t",
            "max_tokens": 100,
            "use_multimodal": False,
        },
    )
    monkeypatch.setattr(enr_mod, "_get_kb_strategy", lambda _kb: "chunk_embed")
    monkeypatch.setattr(enr_mod, "_update_enrichment_status", lambda *a, **kw: None)
    # Atomic-claim UPDATE returns a claimed row.
    mock_db_session.execute.return_value.fetchone.return_value = ("cfg-1",)

    captured_cb = {}

    class FakeEnricher:
        def __init__(self, *a, **kw):
            pass

        def run_enrichment(self, **kw):
            captured_cb["on_batch_complete"] = kw.get("on_batch_complete")

            async def _coro():
                return {"errors": []}

            return _coro()

        def count_by_status(self, _t):
            return (1, 0)

        def count_total_items(self, _s):
            return 1

    monkeypatch.setattr(enr_mod, "MetadataEnricher", FakeEnricher)

    enr_mod.enrich_knowledge_base.run("kb-1")

    # The port recorded the per-batch wiring: metadata_enrichment, enabled.
    assert recording_billing.per_batch_calls == [
        {"config_id": "cfg-1", "action": "metadata_enrichment", "enabled": True}
    ]
    # And the enricher received the callback the port returned (non-None here
    # because recording_billing is "configured").
    assert captured_cb["on_batch_complete"] is not None


def _stub_url_extraction(monkeypatch, url_mod, mock_db_session):
    """Stub firecrawl + httpx + storage so extract_url_source.run() succeeds."""
    monkeypatch.setattr(
        url_mod,
        "get_source",
        lambda _: {
            "id": "src-1",
            "name": "x",
            "file_type": "text/html",
            "storage_path": "x",
            "extraction_status": "pending",
            "derivatives": {},
            "metadata": {},
            "auto_metadata": {},
        },
    )
    monkeypatch.setattr(url_mod, "update_source_status", lambda *a, **kw: None)
    monkeypatch.setattr(url_mod, "update_source_extraction_result", lambda *a, **kw: None)
    # FIRECRAWL_API_KEY is platform-paid env-injected (read from os.environ);
    # FIRECRAWL_API_BASE remains a tenant-configurable setting.
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")
    monkeypatch.setattr(
        url_mod, "get_setting", lambda k: "fake-key" if k.startswith("FIRECRAWL") else 1
    )

    class FakeClient:
        def __init__(self, *_a, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def post(self, *_a, **_kw):
            class R:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"data": {"markdown": "hi", "html": "<p>hi</p>", "metadata": {}}}

            return R()

        def get(self, *_a, **_kw):
            class R:
                status_code = 200
                content = b""
                headers = {}

                def raise_for_status(self):
                    pass

            return R()

    monkeypatch.setattr(url_mod.httpx, "Client", FakeClient)
    monkeypatch.setattr(
        url_mod,
        "get_storage",
        lambda: SimpleNamespace(
            ensure_bucket=lambda *_a: None,
            upload=lambda **_kw: "sources/x",
        ),
    )
    monkeypatch.setattr(url_mod, "get_derivative_storage_path", lambda *a, **kw: "x")
    monkeypatch.setattr(url_mod, "get_source_storage_path", lambda *a, **kw: "x")
    mock_db_session.execute.return_value.scalar.return_value = "extracting"


def test_url_extraction_charges_web_scrape(monkeypatch, mock_db_session, recording_billing):
    """extract_url_source charges web_scrape, quantity=1 per URL; the KEY
    (web_scrape + source_id) reproduces the old stable enqueue-time key."""
    from agentic_project_service.tasks import url_extraction as url_mod

    _stub_url_extraction(monkeypatch, url_mod, mock_db_session)

    url_mod.extract_url_source.run(
        source_id="src-1", bucket_id="sources", url="https://example.com"
    )

    assert len(recording_billing.charges) == 1
    charge = recording_billing.charges[0]
    assert charge["action"] == "web_scrape"
    assert charge["quantity"] == 1
    assert charge["ref_type"] == "extraction"
    assert charge["ref_id"] == "src-1"
    # web_scrape is fixed on both sides → no idempotency_action override needed;
    # the key action defaults to the billed action in the adapter.
    assert charge["idempotency_action"] is None
    assert charge["idempotency_parts"] == ("src-1",)
    assert charge["metadata"] == {"url": "https://example.com"}


def test_url_extraction_reextract_seed_threads_into_idempotency_parts(
    monkeypatch, mock_db_session, recording_billing
):
    """A URL reextract threads its per-call ``reextract_seed`` into the key
    tail, exactly as the PDF path does (source_id, seed)."""
    from agentic_project_service.tasks import url_extraction as url_mod

    _stub_url_extraction(monkeypatch, url_mod, mock_db_session)

    url_mod.extract_url_source.run(
        source_id="src-1",
        bucket_id="sources",
        url="https://example.com",
        reextract_seed="seed-url",
    )

    assert len(recording_billing.charges) == 1
    assert recording_billing.charges[0]["idempotency_parts"] == ("src-1", "seed-url")


def test_url_extraction_retries_on_missing_firecrawl_env(monkeypatch, mock_db_session):
    """When FIRECRAWL_API_KEY is missing from pod env (platform
    misconfiguration), the Celery task must call self.retry so the
    worker re-enqueues the job. Returning a plain status dict would
    have Celery see "success" and never retry → sources permanently
    stuck in `failed` for every URL submitted during the misconfig
    window.

    Patches self.retry to raise the real ``celery.exceptions.Retry``,
    not a sentinel, so this test also pins the ``except Retry: raise``
    propagation guard in url_extraction.py. Without that guard the
    generic ``except Exception`` would catch Retry and return a status
    dict — the test would see no Retry exception and fail.

    Counterfactual A: revert to ``return {"status": "error", ...}`` →
    retry_mock not called, fails.
    Counterfactual B: delete ``except Retry: raise`` → generic handler
    swallows Retry, no exception out, fails.
    """
    from celery.exceptions import Retry

    from agentic_project_service.tasks import url_extraction as url_mod

    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setattr(
        url_mod,
        "get_source",
        lambda _: {
            "id": "src-1",
            "name": "x",
            "file_type": "text/html",
            "storage_path": "x",
            "extraction_status": "pending",
            "derivatives": {},
            "metadata": {},
            "auto_metadata": {},
        },
    )
    monkeypatch.setattr(url_mod, "update_source_status", lambda *a, **kw: None)
    monkeypatch.setattr(url_mod, "update_source_extraction_result", lambda *a, **kw: None)

    retry_mock = MagicMock(side_effect=Retry("platform misconfig retry"))
    monkeypatch.setattr(url_mod.extract_url_source, "retry", retry_mock)

    rec = RecordingBillingAdapter()
    billing_port.set_billing_adapter(rec)
    with pytest.raises(Retry):
        url_mod.extract_url_source.run(
            source_id="src-1",
            bucket_id="sources",
            url="https://example.com",
        )

    retry_mock.assert_called_once()
    kwargs = retry_mock.call_args.kwargs
    assert kwargs["countdown"] == 600
    assert kwargs["max_retries"] == 24
    assert isinstance(kwargs["exc"], RuntimeError)
    # No charge fires on retry — billing must wait for the successful
    # extraction attempt that follows AWS SM seed + ESO sync.
    assert rec.charges == []


# ---------------------------------------------------------------------------
# Routes that schedule the tasks must check balance + forward args.
# ---------------------------------------------------------------------------


def test_estimated_extraction_cost_is_env_independent(monkeypatch):
    """The pre-charge cost estimate is a pure function of the action — it no
    longer reads BillingContext (the balance gate now routes through the port,
    whose adapter no-ops when unconfigured). So it returns the full estimate
    even with no billing env set."""
    from agentic_project_service.routes import sources

    monkeypatch.delenv("BILLING_ORG_ID", raising=False)
    monkeypatch.delenv("PROJECT_ID", raising=False)
    monkeypatch.delenv("BILLING_PROJECT_ID", raising=False)
    cost = sources._estimated_extraction_cost("ocr_pages")
    assert cost == sources._EXTRACTION_ESTIMATED_PAGES * sources._OCR_UNIT_CREDITS


def test_estimated_extraction_cost_ocr_pages():
    """ocr_pages action uses the ocr_pages unit credits constant."""
    from agentic_project_service.routes import sources

    assert (
        sources._estimated_extraction_cost("ocr_pages")
        == sources._EXTRACTION_ESTIMATED_PAGES * sources._OCR_UNIT_CREDITS
    )


def test_estimated_extraction_cost_web_scrape():
    """web_scrape action uses the web_scrape unit credits constant."""
    from agentic_project_service.routes import sources

    assert (
        sources._estimated_extraction_cost("web_scrape")
        == sources._EXTRACTION_ESTIMATED_PAGES * sources._WEB_SCRAPE_UNIT_CREDITS
    )


def test_estimated_extraction_cost_advanced_ocr():
    """advanced_ocr action uses the advanced-OCR unit credits constant."""
    from agentic_project_service.routes import sources

    assert (
        sources._estimated_extraction_cost("advanced_ocr")
        == sources._EXTRACTION_ESTIMATED_PAGES * sources._ADVANCED_OCR_UNIT_CREDITS
    )


def test_extraction_billing_action_maps_llamaparse_to_advanced_ocr():
    """LlamaParse routes to advanced_ocr; everything else to ocr_pages."""
    from agentic_project_service.routes import sources

    assert sources._extraction_billing_action("llamaparse") == "advanced_ocr"
    assert sources._extraction_billing_action("mistral") == "ocr_pages"
    assert sources._extraction_billing_action("auto") == "ocr_pages"
    assert sources._extraction_billing_action(None) == "ocr_pages"


def test_knowledge_bases_billing_kwargs_is_env_independent(monkeypatch):
    """The helper no longer reads BillingContext — identity + key are the
    adapter's job. It returns the threaded key inputs regardless of env, so it
    yields the same kwargs with billing unconfigured as with it configured."""
    from agentic_project_service.routes import knowledge_bases

    monkeypatch.delenv("BILLING_ORG_ID", raising=False)
    monkeypatch.delenv("PROJECT_ID", raising=False)
    monkeypatch.delenv("BILLING_PROJECT_ID", raising=False)
    kwargs = knowledge_bases._maybe_indexing_billing_kwargs(
        action="indexing_chunkembed", ref_id="x"
    )
    assert kwargs == {
        "idempotency_action": "indexing_chunkembed",
        "idempotency_parts": ["x"],
    }


def test_knowledge_bases_billing_kwargs_threads_action_and_ref_id():
    """Single-index route uses the resolved indexing_<strategy> as the key
    action (key action == billed action); the batch-reindex routes pass the
    literal "indexing" (key action != billed action). Both flow through this
    helper as (action, ref_id) -> {idempotency_action, idempotency_parts}."""
    from agentic_project_service.routes import knowledge_bases

    single = knowledge_bases._maybe_indexing_billing_kwargs(
        action="indexing_pageindex", ref_id="ix-1"
    )
    assert single == {
        "idempotency_action": "indexing_pageindex",
        "idempotency_parts": ["ix-1"],
    }
    batch = knowledge_bases._maybe_indexing_billing_kwargs(action="indexing", ref_id="ix-2")
    assert batch == {"idempotency_action": "indexing", "idempotency_parts": ["ix-2"]}


# ---------------------------------------------------------------------------
# Post-success 402 handling — post_charge NEVER raises. A 402 is reported as
# ChargeResult(success=False, failure_mode='insufficient_credits') and the
# task continues with its already-committed success state. Per spec line 54
# v1 accepts bounded over-serve as the cost of correctness.
# ---------------------------------------------------------------------------


def test_extraction_charge_insufficient_credits_does_not_fail_op(monkeypatch, mock_db_session):
    """A charge returning ChargeOutcome(insufficient_credits=True) after the OCR
    pages have been committed must NOT flip the op status to 'failed'. Per spec
    line 54 a post-success 402 is bounded over-serve, not an op failure."""
    from agentic_project_service.tasks import extraction as ext_mod

    _stub_extraction(
        monkeypatch,
        ext_mod,
        mock_db_session,
        {"extraction_method": "mistral_ocr", "page_count": 7, "char_count": 12000},
    )
    status_calls: list[tuple] = []
    monkeypatch.setattr(
        ext_mod, "update_source_status", lambda *a, **kw: status_calls.append((a, kw))
    )

    billing_port.set_billing_adapter(RecordingBillingAdapter(insufficient=True))
    result = ext_mod.extract_source.run(
        source_id="src-1", bucket_id="sources", extraction_model="mistral"
    )

    assert result["status"] == "success"
    assert result["source_id"] == "src-1"
    failed_calls = [c for c in status_calls if (len(c[0]) >= 2 and c[0][1] == "failed")]
    assert (
        failed_calls == []
    ), f"Expected NO 'failed' status updates after post-success 402; got: {failed_calls}"


def test_url_extraction_charge_insufficient_credits_does_not_fail_op(monkeypatch, mock_db_session):
    """extract_url_source must not flip status to 'failed' when the charge
    reports insufficient_credits after a successful URL scrape."""
    from agentic_project_service.tasks import url_extraction as url_mod

    _stub_url_extraction(monkeypatch, url_mod, mock_db_session)
    status_calls: list[tuple] = []
    monkeypatch.setattr(
        url_mod, "update_source_status", lambda *a, **kw: status_calls.append((a, kw))
    )

    billing_port.set_billing_adapter(RecordingBillingAdapter(insufficient=True))
    result = url_mod.extract_url_source.run(
        source_id="src-1", bucket_id="sources", url="https://example.com"
    )

    assert result["status"] == "success"
    failed_calls = [c for c in status_calls if (len(c[0]) >= 2 and c[0][1] == "failed")]
    assert (
        failed_calls == []
    ), f"Expected NO 'failed' status updates after post-success 402; got: {failed_calls}"


@pytest.mark.parametrize(
    "filename",
    ["extraction.py", "url_extraction.py", "indexing.py", "enrichment.py"],
)
def test_tasks_do_not_import_removed_insufficient_credits(filename: str):
    """Task modules no longer reference the (now-removed) InsufficientCredits
    class — the billing charge reports outcome on every path. A stray import
    or reference would be dead code and a hint that try/except boilerplate
    was reintroduced."""
    src = _read(TASKS_DIR / filename)
    assert (
        "InsufficientCredits" not in src
    ), f"{filename}: InsufficientCredits has been removed; remove the reference"
