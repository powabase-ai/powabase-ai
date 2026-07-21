"""Configuration routes — exposes KB defaults as a single source of truth."""

import copy

from flask import Blueprint, jsonify

from agentic.knowledge.model_config import (
    EXTRACTION_DEFAULT_METHOD,
    EXTRACTION_FALLBACK_CHAIN,
    HYBRID_DEFAULT_VECTOR_WEIGHT,
    METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS,
    METADATA_ENRICHMENT_DEFAULT_MODEL,
    QUERY_ENRICHMENT_DEFAULT_MODEL,
    RERANKER_CANDIDATE_COUNT,
    RERANKER_DEFAULT_MODEL,
)

from ..auth import require_auth
from ..strategies import RETRIEVER_LABELS, STRATEGY_REGISTRY

config_bp = Blueprint("config", __name__, url_prefix="/api/config")

RERANKER_OPTIONS = [
    {
        "value": "cohere/rerank-english-v3.0",
        "label": "Cohere Rerank English v3",
        "provider": "Cohere",
    },
    {
        "value": "cohere/rerank-multilingual-v3.0",
        "label": "Cohere Rerank Multilingual v3",
        "provider": "Cohere",
    },
    {
        "value": "jina_ai/jina-reranker-v2-base-multilingual",
        "label": "Jina Reranker v2",
        "provider": "Jina AI",
    },
    {
        "value": "voyage/rerank-2.5",
        "label": "Voyage Rerank 2.5",
        "provider": "Voyage",
    },
    {
        "value": "voyage/rerank-2.5-lite",
        "label": "Voyage Rerank 2.5 Lite",
        "provider": "Voyage",
    },
    {
        "value": "zerank-2",
        "label": "ZeroEntropy zerank-2",
        "provider": "ZeroEntropy",
    },
]


@config_bp.route("/kb-defaults", methods=["GET"])
@require_auth
def kb_defaults():
    """Return KB configuration defaults for the frontend."""
    strategies = {}
    for name, entry in STRATEGY_REGISTRY.items():
        retriever_labels = {r: RETRIEVER_LABELS.get(r, r) for r in entry["compatible_retrievers"]}
        strategies[name] = {
            "label": entry["label"],
            "compatible_retrievers": entry["compatible_retrievers"],
            "retriever_labels": retriever_labels,
            "default_retrieval_method": entry["default_retrieval_method"],
            "supports_reranker": entry["supports_reranker"],
            "default_indexing_config": copy.deepcopy(entry["default_indexing_config"]),
            "default_retrieval_config": copy.deepcopy(entry["default_retrieval_config"]),
        }

    return jsonify(
        {
            "strategies": strategies,
            "reranker": {
                "default_model": RERANKER_DEFAULT_MODEL,
                "candidate_count": RERANKER_CANDIDATE_COUNT,
                "options": RERANKER_OPTIONS,
            },
            "query_enrichment": {
                "model": QUERY_ENRICHMENT_DEFAULT_MODEL,
            },
            "enrichment": {
                "model": METADATA_ENRICHMENT_DEFAULT_MODEL,
                "max_tokens": METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS,
            },
            "hybrid_vector_weight": HYBRID_DEFAULT_VECTOR_WEIGHT,
            "extraction": {
                "default_method": EXTRACTION_DEFAULT_METHOD,
                "fallback_chain": EXTRACTION_FALLBACK_CHAIN,
                "options": [
                    {
                        "value": "auto",
                        "label": "Auto (recommended)",
                        "description": "Uses fallback chain: tries each method in order until one succeeds.",
                    },
                    {
                        "value": "mistral",
                        "label": "Mistral OCR",
                        "description": "Best for scanned PDFs. Requires MISTRAL_API_KEY.",
                    },
                    {
                        "value": "paddleocr",
                        "label": "PaddleOCR",
                        "description": "PaddleOCR-VL API. Requires PADDLEOCR_API_KEY.",
                    },
                    {
                        "value": "lighton",
                        "label": "LightOn OCR",
                        "description": "LightOn OCR API. Requires LIGHTON_API_KEY.",
                    },
                    {
                        "value": "llamaparse",
                        "label": "LlamaParse (Advanced OCR)",
                        "description": "Advanced OCR for complex PDFs. Requires LLAMAPARSE_API_KEY. Billed per page at the advanced-OCR rate.",
                    },
                    {
                        "value": "opendataloader",
                        "label": "OpenDataLoader",
                        "description": "High-accuracy structural extraction.",
                    },
                    {
                        "value": "fitz",
                        "label": "PyMuPDF (fitz)",
                        "description": "Fast, good for text-based PDFs.",
                    },
                    {
                        "value": "pdfplumber",
                        "label": "pdfplumber",
                        "description": "Reliable fallback for edge cases.",
                    },
                ],
            },
        }
    )
