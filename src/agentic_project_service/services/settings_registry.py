"""Settings registry — single source of truth for all configurable settings.

Maps every tunable config variable to metadata (type, default, validation).
Provides `get_setting()` for runtime code to read DB overrides with fallback
to defaults, and `validate_setting()` for the API layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from flask import g, has_app_context
from sqlalchemy import text

from ..db import AI_SCHEMA, db

logger = logging.getLogger(__name__)

# Placeholder returned in API responses for secret settings
SECRET_MASK = "••••••••"

# Model choices shared across multiple settings
_LLM_MODEL_CHOICES = [
    # OpenAI
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o3",
    "o3-mini",
    "o3-pro",
    "o4-mini",
    # Anthropic
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    # Anthropic (dated aliases — kept for backward compatibility)
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5-20251001",
    # Google
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-2.5-pro",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite",
    # OpenRouter (slugs must match LiteLLM's openrouter/ cost-map keys).
    # Every entry must support function calling — this list feeds the agent
    # and copilot model pickers (AGENT_DEFAULT_MODEL, copilot_model), and an
    # agent/copilot fails on its first tool call with a non-tool model. That
    # is why mistral-small-3.1-24b-instruct is excluded here even though it
    # resolves in LiteLLM: litellm.supports_function_calling(...) is False.
    # Enforced by tests/unit/test_llm_model_choices.py.
    "openrouter/qwen/qwen3-235b-a22b-2507",
    "openrouter/mistralai/mistral-large-2512",
]

_EMBEDDING_MODEL_CHOICES = [
    # OpenAI
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
    # Cohere
    "embed-english-v3.0",
    "embed-english-light-v3.0",
    "embed-multilingual-v3.0",
    "embed-multilingual-light-v3.0",
    # Voyage AI
    "voyage/voyage-01",
    "voyage/voyage-lite-01",
    "voyage/voyage-lite-01-instruct",
    # Google Gemini
    "gemini/text-embedding-004",
    # Mistral
    "mistral/mistral-embed",
]

_IMAGE_DELIVERY_CHOICES = ["base64", "url"]

EXTRACTION_METHOD_CHOICES = [
    "auto",
    "mistral",
    "opendataloader",
    "paddleocr",
    "lighton",
    "llamaparse",
    "fitz",
    "pdfplumber",
]


@dataclass
class SettingDef:
    """Metadata for a single configurable setting."""

    key: str
    category: str
    label: str
    type: str  # "int", "float", "str", "bool"
    default: Any
    advanced: bool = False
    description: str = ""
    subcategory: str = ""
    secret: bool = False
    min: float | None = None
    max: float | None = None
    choices: list[str] = field(default_factory=list)


def _build_registry() -> dict[str, SettingDef]:
    """Build the full settings registry. Called once at import time."""
    defs: list[SettingDef] = []

    # =========================================================================
    # Copilot
    # =========================================================================
    cat = "copilot"

    defs += [
        SettingDef(
            key="copilot_model",
            category=cat,
            label="Copilot Model",
            type="str",
            default="claude-opus-4-6",
            choices=_LLM_MODEL_CHOICES,
            subcategory="llm_model",
            description="LLM model used by the workflow copilot agent.",
        ),
        SettingDef(
            key="COPILOT_TEMPERATURE",
            category=cat,
            label="Temperature",
            type="float",
            default=0.7,
            min=0,
            max=2,
            description="Sampling temperature for the copilot agent.",
        ),
        SettingDef(
            key="COPILOT_MAX_STEPS",
            category=cat,
            label="Max Steps",
            type="int",
            default=25,
            min=1,
            max=100,
            description="Maximum ReAct reasoning steps before forced answer.",
        ),
        SettingDef(
            key="COPILOT_REASONING_EFFORT",
            category=cat,
            label="Reasoning Effort",
            type="str",
            default="medium",
            choices=["low", "medium", "high"],
            description=(
                "Reasoning effort forwarded to reasoning-capable models. "
                "Silently dropped for models without reasoning support."
            ),
        ),
        SettingDef(
            key="MAX_BLOCK_NAME_LEN",
            category=cat,
            label="Max Block Name Length",
            type="int",
            default=100,
            min=10,
            max=500,
            advanced=True,
            description="Maximum length for block display names before truncation.",
        ),
        SettingDef(
            key="MAX_CONFIG_VALUE_LEN",
            category=cat,
            label="Max Config Value Length",
            type="int",
            default=2000,
            min=100,
            max=50000,
            advanced=True,
            description="Maximum length for any single config value before truncation.",
        ),
        SettingDef(
            key="MAX_TOTAL_STATE_LEN",
            category=cat,
            label="Max Workflow State Length",
            type="int",
            default=50000,
            min=1000,
            max=500000,
            advanced=True,
            description="Hard cap on total serialized workflow state.",
        ),
        SettingDef(
            key="MAX_CONFIG_DEPTH",
            category=cat,
            label="Max Config Nesting Depth",
            type="int",
            default=10,
            min=1,
            max=50,
            advanced=True,
            description="Maximum nesting depth when recursively truncating config dicts.",
        ),
        SettingDef(
            key="SYSTEM_PROMPT_TRUNCATE",
            category=cat,
            label="System Prompt Truncate Length",
            type="int",
            default=1000,
            min=100,
            max=50000,
            advanced=True,
            description="Truncation length for agent system prompts in copilot context.",
        ),
    ]

    # =========================================================================
    # Agents
    # =========================================================================
    cat = "agents"

    defs += [
        SettingDef(
            key="AGENT_DEFAULT_MODEL",
            category=cat,
            label="Default Agent Model",
            type="str",
            default="gpt-5.4-mini",
            choices=_LLM_MODEL_CHOICES,
            subcategory="llm_model",
            description="Default LLM model for agent execution.",
        ),
        SettingDef(
            key="DEFAULT_MAX_CONTEXT_TOKENS",
            category=cat,
            label="Agent Max Context Tokens",
            type="int",
            default=32000,
            min=1000,
            max=256000,
            description="Maximum tokens for formatted context sent to the agent LLM.",
        ),
        SettingDef(
            key="DELEGATE_MAX_STEPS",
            category=cat,
            label="Delegate Max Steps",
            type="int",
            default=10,
            min=1,
            max=50,
            description="Maximum ReAct steps for a DelegateTool sub-agent.",
        ),
    ]

    # =========================================================================
    # Tools
    # =========================================================================
    cat = "tools"

    defs += [
        SettingDef(
            key="CUSTOM_TOOL_TIMEOUT",
            category=cat,
            label="Custom Tool HTTP Timeout (s)",
            type="int",
            default=30,
            min=5,
            max=300,
            description="Default HTTP timeout for custom tool endpoint calls.",
        ),
        SettingDef(
            key="MCP_TOOL_TIMEOUT",
            category=cat,
            label="MCP Tool Timeout (s)",
            type="int",
            default=30,
            min=5,
            max=300,
            description="Default timeout for MCP tool calls.",
        ),
        SettingDef(
            key="MAX_TOOL_OUTPUT_LENGTH",
            category=cat,
            label="Max Tool Output Length",
            type="int",
            default=10000,
            min=1000,
            max=200000,
            advanced=True,
            description="Max characters for custom tool HTTP response bodies.",
        ),
        SettingDef(
            key="DEFAULT_MAX_RESULT_CHARS",
            category=cat,
            label="Default Max Result Chars",
            type="int",
            default=50000,
            min=1000,
            max=500000,
            advanced=True,
            description="Default max_result_chars for ToolDefinition base class.",
        ),
        # EXA_API_KEY and FIRECRAWL_API_KEY are platform-paid secrets injected
        # via pod env (AWS SM → ExternalSecret → project-secrets). They are NOT
        # tenant-managed — web_search_handler / web_scrape_handler read them
        # directly from os.environ. See agentic-platform/CLAUDE.md, "Operator
        # runbook — Exa/Firecrawl platform keys".
        SettingDef(
            key="FIRECRAWL_API_BASE",
            category=cat,
            label="Firecrawl API Base URL",
            type="str",
            default="https://api.firecrawl.dev/v1",
            advanced=True,
            description="Base URL for Firecrawl API. Change for self-hosted instances.",
        ),
        SettingDef(
            key="VISION_MODEL",
            category=cat,
            label="Vision Analysis Model",
            type="str",
            default="gpt-5-mini",
            choices=_LLM_MODEL_CHOICES,
            subcategory="llm_model",
            description="LLM model used for image analysis in web_scrape (include_images).",
        ),
        SettingDef(
            key="WEB_SCRAPE_MAX_CHARS",
            category=cat,
            label="Web Scrape Max Output Chars",
            type="int",
            default=200000,
            min=10000,
            max=500000,
            description="Maximum characters returned by the web_scrape tool.",
        ),
        SettingDef(
            key="WEB_SCRAPE_MAX_IMAGES",
            category=cat,
            label="Web Scrape Max Images",
            type="int",
            default=10,
            min=1,
            max=50,
            description="Maximum number of images to analyze per web_scrape call when include_images is enabled.",
        ),
        SettingDef(
            key="VISION_TIMEOUT",
            category=cat,
            label="Vision Analysis Timeout (s)",
            type="int",
            default=30,
            min=10,
            max=120,
            description="Timeout in seconds for each image analysis call.",
        ),
        SettingDef(
            key="VISION_MAX_WORKERS",
            category=cat,
            label="Vision Concurrent Workers",
            type="int",
            default=3,
            min=1,
            max=10,
            advanced=True,
            description="Maximum concurrent vision LLM calls during image analysis.",
        ),
    ]

    # =========================================================================
    # Knowledge Indexing
    # =========================================================================
    cat = "knowledge-indexing"

    defs += [
        # --- Main settings ---
        SettingDef(
            key="CHUNK_EMBED_DEFAULT_CHUNK_SIZE",
            category=cat,
            label="Chunk Size (chars)",
            type="int",
            default=2000,
            min=100,
            max=50000,
            description="Default chunk size for the chunk-embed strategy.",
        ),
        SettingDef(
            key="CHUNK_EMBED_DEFAULT_OVERLAP",
            category=cat,
            label="Chunk Overlap (chars)",
            type="int",
            default=50,
            min=0,
            max=5000,
            description="Default overlap between chunks.",
        ),
        SettingDef(
            key="CHUNK_EMBED_EMBEDDING_MODEL",
            category=cat,
            label="Chunk Embed Model",
            type="str",
            default="text-embedding-3-small",
            choices=_EMBEDDING_MODEL_CHOICES,
            description="Embedding model for the chunk-embed strategy.",
        ),
        SettingDef(
            key="PAGEINDEX_INDEXING_MODEL",
            category=cat,
            label="PageIndex Indexing Model",
            type="str",
            default="gpt-5-mini",
            choices=_LLM_MODEL_CHOICES,
            description="LLM model for PageIndex tree building and summary generation.",
        ),
        SettingDef(
            key="GRAPHINDEX_INDEXING_MODEL",
            category=cat,
            label="GraphIndex Indexing Model",
            type="str",
            default="gpt-5-mini",
            choices=_LLM_MODEL_CHOICES,
            description="LLM model for GraphIndex ToC building.",
        ),
        SettingDef(
            key="GRAPHINDEX_ENRICHMENT_MODEL",
            category=cat,
            label="GraphIndex Enrichment Model",
            type="str",
            default="gpt-5-mini",
            choices=_LLM_MODEL_CHOICES,
            description="LLM model for GraphIndex referenced_nodes enrichment.",
        ),
        SettingDef(
            key="GRAPHINDEX_EMBEDDING_MODEL",
            category=cat,
            label="GraphIndex Embedding Model",
            type="str",
            default="text-embedding-3-small",
            choices=_EMBEDDING_MODEL_CHOICES,
            description="Embedding model for GraphIndex node summaries.",
        ),
        SettingDef(
            key="FULLDOC_SUMMARY_MODEL",
            category=cat,
            label="Full Doc Summary Model",
            type="str",
            default="gpt-5-mini",
            choices=_LLM_MODEL_CHOICES,
            description="LLM model for full-document summary generation.",
        ),
        SettingDef(
            key="FULLDOC_EMBEDDING_MODEL",
            category=cat,
            label="Full Doc Embedding Model",
            type="str",
            default="text-embedding-3-small",
            choices=_EMBEDDING_MODEL_CHOICES,
            description="Embedding model for full-document summaries.",
        ),
        SettingDef(
            key="DOC2JSON_EXTRACTION_MODEL",
            category=cat,
            label="Doc2JSON Extraction Model",
            type="str",
            default="gpt-5-mini",
            choices=_LLM_MODEL_CHOICES,
            description="LLM model for Doc2JSON sliding-window extraction.",
        ),
        SettingDef(
            key="DOC2JSON_EMBEDDING_MODEL",
            category=cat,
            label="Doc2JSON Embedding Model",
            type="str",
            default="text-embedding-3-small",
            choices=_EMBEDDING_MODEL_CHOICES,
            description="Embedding model for Doc2JSON summaries.",
        ),
        SettingDef(
            key="EXTRACTION_DEFAULT_METHOD",
            category=cat,
            label="PDF Extraction Method",
            type="str",
            default="auto",
            choices=EXTRACTION_METHOD_CHOICES,
            description="Default extraction method for PDFs.",
        ),
        # --- Advanced PageIndex ---
        SettingDef(
            key="PAGEINDEX_LLM_MAX_CONCURRENT",
            category=cat,
            label="PageIndex Max Concurrent LLM",
            type="int",
            default=7,
            min=1,
            max=50,
            advanced=True,
            description="Max concurrent LLM calls during PageIndex indexing.",
        ),
        SettingDef(
            key="PAGEINDEX_TOC_CHECK_PAGE_NUM",
            category=cat,
            label="ToC Scan Pages",
            type="int",
            default=15,
            min=1,
            max=100,
            advanced=True,
            description="Pages to scan from document start for table of contents.",
        ),
        SettingDef(
            key="PAGEINDEX_TOC_OFFSET_SCAN_PAGES",
            category=cat,
            label="Post-ToC Heading Scan Pages",
            type="int",
            default=50,
            min=1,
            max=200,
            advanced=True,
            description="Pages to scan after ToC for section heading calibration.",
        ),
        SettingDef(
            key="PAGEINDEX_TOC_GAP_TOLERANCE",
            category=cat,
            label="ToC Gap Tolerance",
            type="int",
            default=3,
            min=0,
            max=20,
            advanced=True,
            description="Consecutive non-ToC pages tolerated inside a ToC sequence.",
        ),
        SettingDef(
            key="PAGEINDEX_TOC_MAX_TOKENS_PER_CHUNK",
            category=cat,
            label="ToC Max Tokens/Chunk",
            type="int",
            default=32000,
            min=1000,
            max=128000,
            advanced=True,
            description="Maximum tokens per LLM call during ToC extraction.",
        ),
        SettingDef(
            key="PAGEINDEX_MAX_PAGE_NUM_EACH_NODE",
            category=cat,
            label="Max Pages/Node",
            type="int",
            default=16,
            min=1,
            max=100,
            advanced=True,
            description="Max pages per node before recursive splitting.",
        ),
        SettingDef(
            key="PAGEINDEX_MAX_TOKEN_NUM_EACH_NODE",
            category=cat,
            label="Max Tokens/Node",
            type="int",
            default=16000,
            min=1000,
            max=128000,
            advanced=True,
            description="Max tokens per node before recursive splitting.",
        ),
        SettingDef(
            key="PAGEINDEX_MAX_NODE_TOKENS",
            category=cat,
            label="Leaf Node Max Tokens",
            type="int",
            default=16000,
            min=1000,
            max=128000,
            advanced=True,
            description="Leaf nodes above this are sent to LLM for structure inference.",
        ),
        SettingDef(
            key="PAGEINDEX_MIN_SPLIT_TOKENS",
            category=cat,
            label="Min Split Tokens",
            type="int",
            default=16000,
            min=1000,
            max=128000,
            advanced=True,
            description="Token threshold for paragraph-count-based splitting.",
        ),
        SettingDef(
            key="PAGEINDEX_MIN_PARAGRAPH_COUNT",
            category=cat,
            label="Min Paragraphs for Split",
            type="int",
            default=4,
            min=1,
            max=50,
            advanced=True,
            description="Minimum blank-line-separated paragraphs for tier-2 splitting.",
        ),
        SettingDef(
            key="PAGEINDEX_MIN_TOKEN_THRESHOLD",
            category=cat,
            label="Tree Thinning Threshold",
            type="int",
            default=0,
            min=0,
            max=50000,
            advanced=True,
            description="Subtrees below this token count merge into parent. 0 = disabled.",
        ),
        SettingDef(
            key="PAGEINDEX_SUMMARY_TOKEN_THRESHOLD",
            category=cat,
            label="Summary Token Threshold",
            type="int",
            default=1500,
            min=100,
            max=50000,
            advanced=True,
            description="Nodes below this use raw text as summary instead of LLM.",
        ),
        SettingDef(
            key="PAGEINDEX_DOC_DESCRIPTION_MAX_TOKENS",
            category=cat,
            label="Doc Description Max Tokens",
            type="int",
            default=80000,
            min=1000,
            max=500000,
            advanced=True,
            description="Max input tokens for document description prompt.",
        ),
        # --- Advanced GraphIndex ---
        SettingDef(
            key="GRAPHINDEX_ENRICHMENT_MAX_CONCURRENT",
            category=cat,
            label="GraphIndex Max Concurrent",
            type="int",
            default=7,
            min=1,
            max=50,
            advanced=True,
            description="Max concurrent LLM calls during GraphIndex enrichment.",
        ),
        SettingDef(
            key="GRAPHINDEX_ENRICHMENT_MAX_TOKENS",
            category=cat,
            label="GraphIndex Enrichment Max Tokens",
            type="int",
            default=5000,
            min=500,
            max=50000,
            advanced=True,
            description="Max tokens for LLM response during enrichment.",
        ),
        SettingDef(
            key="GRAPHINDEX_ENRICHMENT_MAX_INPUT_CHARS",
            category=cat,
            label="GraphIndex Max Input Chars",
            type="int",
            default=0,
            min=0,
            max=500000,
            advanced=True,
            description="Max input characters for node text. 0 = no truncation.",
        ),
        SettingDef(
            key="GRAPHINDEX_ENRICHMENT_TOC_INCLUDE_SUMMARIES",
            category=cat,
            label="Include Summaries in Enrichment",
            type="bool",
            default=False,
            advanced=True,
            description="Include node summaries in ToC context for enrichment prompts.",
        ),
        SettingDef(
            key="GRAPHINDEX_ENRICHMENT_MAX_JSON_RETRIES",
            category=cat,
            label="GraphIndex JSON Retries",
            type="int",
            default=3,
            min=0,
            max=10,
            advanced=True,
            description="Retry attempts when LLM returns unparseable JSON.",
        ),
        # --- Advanced Doc2JSON ---
        SettingDef(
            key="DOC2JSON_DEFAULT_WINDOW_SIZE",
            category=cat,
            label="Doc2JSON Window Size (tokens)",
            type="int",
            default=4000,
            min=500,
            max=50000,
            advanced=True,
            description="Sliding window size in tokens for text-based extraction.",
        ),
        SettingDef(
            key="DOC2JSON_DEFAULT_WINDOW_OVERLAP",
            category=cat,
            label="Doc2JSON Window Overlap",
            type="int",
            default=200,
            min=0,
            max=5000,
            advanced=True,
            description="Overlap between sliding windows.",
        ),
        SettingDef(
            key="DOC2JSON_USE_IMAGES",
            category=cat,
            label="Doc2JSON Use Images",
            type="bool",
            default=False,
            advanced=True,
            description="Extract from page images instead of text (multimodal).",
        ),
        SettingDef(
            key="DOC2JSON_DEFAULT_PAGES_PER_WINDOW",
            category=cat,
            label="Doc2JSON Pages/Window",
            type="int",
            default=3,
            min=1,
            max=20,
            advanced=True,
            description="Pages per LLM call when using image-based extraction.",
        ),
        SettingDef(
            key="DOC2JSON_EXTRACTION_MAX_TOKENS",
            category=cat,
            label="Doc2JSON Extraction Max Tokens",
            type="int",
            default=4000,
            min=500,
            max=50000,
            advanced=True,
            description="LLM output token limit for extraction calls.",
        ),
        SettingDef(
            key="DOC2JSON_SUMMARY_MAX_TOKENS",
            category=cat,
            label="Doc2JSON Summary Max Tokens",
            type="int",
            default=2000,
            min=100,
            max=20000,
            advanced=True,
            description="LLM output token limit for summary calls.",
        ),
        SettingDef(
            key="DOC2JSON_MAX_RETRIES",
            category=cat,
            label="Doc2JSON JSON Retries",
            type="int",
            default=3,
            min=0,
            max=10,
            advanced=True,
            description="Retry attempts for JSON parsing failures.",
        ),
        SettingDef(
            key="EMBEDDING_MAX_TOKENS_PER_BATCH",
            category=cat,
            label="Embedding Max Tokens/Batch",
            type="int",
            default=200000,
            min=10000,
            max=500000,
            advanced=True,
            description="Max tokens per single embedding API request.",
        ),
        SettingDef(
            key="FULLDOC_SUMMARY_INPUT_CHARS",
            category=cat,
            label="Full Doc Summary Input Chars",
            type="int",
            default=128000,
            min=1000,
            max=1000000,
            advanced=True,
            description="Max input characters for full-document summary prompt.",
        ),
        SettingDef(
            key="FULLDOC_SUMMARY_MAX_TOKENS",
            category=cat,
            label="Full Doc Summary Max Tokens",
            type="int",
            default=8000,
            min=500,
            max=50000,
            advanced=True,
            description="LLM output token limit for full-document summaries.",
        ),
        SettingDef(
            key="BM25_AUTO_INDEXING",
            category=cat,
            label="Automatic BM25 indexing",
            type="bool",
            default=True,
            advanced=True,
            description=(
                "When enabled (default), the platform keeps the BM25 sparse "
                "index up to date automatically: per-source updates during "
                "indexing, and a one-shot rebuild when a KB's retrieval method "
                "changes to hybrid or full_text. Disable for very large KBs "
                "where the per-source BM25 rebuild dominates indexing time; "
                "you'll then trigger rebuilds manually from the KB detail page."
            ),
        ),
    ]

    # =========================================================================
    # Knowledge Retrieval
    # =========================================================================
    cat = "knowledge-retrieval"

    defs += [
        SettingDef(
            key="KB_DEFAULT_TOP_K",
            category=cat,
            label="Default Top K",
            type="int",
            default=10,
            min=1,
            max=100,
            description="Default number of top results per knowledge base.",
        ),
        SettingDef(
            key="KB_DEFAULT_MAX_CONTEXT_TOKENS",
            category=cat,
            label="Max Context Tokens",
            type="int",
            default=16000,
            min=1000,
            max=128000,
            description="Maximum context tokens for formatted search results.",
        ),
        SettingDef(
            key="DEFAULT_IMAGE_DELIVERY",
            category=cat,
            label="Image Delivery Mode",
            type="str",
            default="base64",
            choices=_IMAGE_DELIVERY_CHOICES,
            description="How multimodal KB content images are delivered.",
        ),
        SettingDef(
            key="HYBRID_DEFAULT_VECTOR_WEIGHT",
            category=cat,
            label="Hybrid Vector Weight",
            type="float",
            default=0.5,
            min=0,
            max=1,
            description="Weight for vector similarity in hybrid RRF fusion.",
        ),
        SettingDef(
            key="PAGEINDEX_RETRIEVAL_MODEL",
            category=cat,
            label="PageIndex Retrieval Model",
            type="str",
            default="gpt-5-mini",
            choices=_LLM_MODEL_CHOICES,
            description="LLM model for PageIndex tree-based retrieval.",
        ),
        SettingDef(
            key="RERANKER_DEFAULT_MODEL",
            category=cat,
            label="Reranker Model",
            type="str",
            default="cohere/rerank-english-v3.0",
            description="Model used for reranking retrieved chunks.",
        ),
        SettingDef(
            key="RERANKER_CANDIDATE_COUNT",
            category=cat,
            label="Reranker Candidate Count",
            type="int",
            default=20,
            min=5,
            max=200,
            description="Candidates fetched in stage 1 before reranking.",
        ),
        SettingDef(
            key="QUERY_ENRICHMENT_DEFAULT_MODEL",
            category=cat,
            label="Query Enrichment Model",
            type="str",
            default="gpt-5-mini",
            choices=_LLM_MODEL_CHOICES,
            description="LLM model for query enrichment.",
        ),
        SettingDef(
            key="METADATA_ENRICHMENT_DEFAULT_MODEL",
            category=cat,
            label="Metadata Enrichment Model",
            type="str",
            default="gpt-5-mini",
            choices=_LLM_MODEL_CHOICES,
            description="LLM model for metadata field extraction.",
        ),
        # --- Advanced ---
        SettingDef(
            key="QUERY_ENRICHMENT_TEMPERATURE",
            category=cat,
            label="Query Enrichment Temperature",
            type="float",
            default=0,
            min=0,
            max=2,
            advanced=True,
            description="Sampling temperature for query enrichment.",
        ),
        SettingDef(
            key="METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS",
            category=cat,
            label="Metadata Enrichment Max Tokens",
            type="int",
            default=2000,
            min=100,
            max=50000,
            advanced=True,
            description="LLM response token budget for metadata enrichment.",
        ),
        SettingDef(
            key="METADATA_ENRICHMENT_MAX_CONCURRENT",
            category=cat,
            label="Metadata Enrichment Concurrency",
            type="int",
            default=10,
            min=1,
            max=50,
            advanced=True,
            description="Max concurrent LLM calls during metadata enrichment.",
        ),
        SettingDef(
            key="METADATA_ENRICHMENT_BATCH_SIZE",
            category=cat,
            label="Metadata Enrichment Batch Size",
            type="int",
            default=50,
            min=1,
            max=500,
            advanced=True,
            description="Items per database commit batch during enrichment.",
        ),
        SettingDef(
            key="METADATA_ENRICHMENT_MAX_RETRIES",
            category=cat,
            label="Metadata Enrichment Retries",
            type="int",
            default=3,
            min=0,
            max=10,
            advanced=True,
            description="Retry attempts for unparseable JSON responses.",
        ),
        SettingDef(
            key="METADATA_ENRICHMENT_MAX_IMAGES",
            category=cat,
            label="Metadata Max Images",
            type="int",
            default=10,
            min=1,
            max=50,
            advanced=True,
            description="Max page images per multimodal enrichment call.",
        ),
        SettingDef(
            key="METADATA_ENRICHMENT_MAX_INPUT_CHARS",
            category=cat,
            label="Metadata Max Input Chars",
            type="int",
            default=0,
            min=0,
            max=500000,
            advanced=True,
            description="Max input characters for text-only enrichment. 0 = no limit.",
        ),
        SettingDef(
            key="METADATA_ENRICHMENT_MAX_INPUT_CHARS_MULTIMODAL",
            category=cat,
            label="Metadata Max Input Chars (Multimodal)",
            type="int",
            default=0,
            min=0,
            max=500000,
            advanced=True,
            description="Max input chars when images are also provided. 0 = no limit.",
        ),
        SettingDef(
            key="MAX_SEARCH_WORKERS",
            category=cat,
            label="Max Concurrent KB Searches",
            type="int",
            default=5,
            min=1,
            max=20,
            advanced=True,
            description="Max concurrent threads when searching multiple KBs.",
        ),
        SettingDef(
            key="DROPPED_ITEM_TEXT_LIMIT",
            category=cat,
            label="Dropped Item Preview Chars",
            type="int",
            default=500,
            min=50,
            max=5000,
            advanced=True,
            description="Max characters for dropped-item preview text in diagnostics.",
        ),
    ]

    # =========================================================================
    # Compaction
    # =========================================================================
    cat = "compaction"

    defs += [
        SettingDef(
            key="DEFAULT_COMPACTION_MODEL",
            category=cat,
            label="Compaction Model",
            type="str",
            default="gpt-5-mini",
            choices=_LLM_MODEL_CHOICES,
            subcategory="llm_model",
            description="LLM model for context compaction.",
        ),
        SettingDef(
            key="COMPACTION_KEEP_LAST_N",
            category=cat,
            label="Turns to Preserve",
            type="int",
            default=2,
            min=0,
            max=20,
            description="Number of recent turns preserved during compaction.",
        ),
        SettingDef(
            key="CHARS_PER_TOKEN",
            category=cat,
            label="Chars per Token Estimate",
            type="int",
            default=4,
            min=1,
            max=10,
            advanced=True,
            description="Rough characters-per-token estimate for token counting.",
        ),
        SettingDef(
            key="COMPACTION_MAX_OUTPUT_TOKENS",
            category=cat,
            label="Max Output Tokens",
            type="int",
            default=8000,
            min=500,
            max=50000,
            advanced=True,
            description="Max tokens for the compaction summary response.",
        ),
        SettingDef(
            key="COMPACTION_BUFFER",
            category=cat,
            label="Compact Buffer (tokens)",
            type="int",
            default=13000,
            min=1000,
            max=100000,
            advanced=True,
            description="Token buffer reserved when deciding whether to compact.",
        ),
    ]

    # =========================================================================
    # Sources
    # =========================================================================
    cat = "sources"

    defs += [
        SettingDef(
            key="URL_IMPORT_MAX_PAGES",
            category=cat,
            label="Max Pages per URL Import",
            type="int",
            default=50,
            min=1,
            max=200,
            description="Maximum number of pages to import per URL import request.",
        ),
        SettingDef(
            key="URL_IMPORT_MAX_IMAGES_PER_PAGE",
            category=cat,
            label="Max Images per Page",
            type="int",
            default=20,
            min=0,
            max=50,
            description="Maximum number of images to download per scraped page.",
        ),
        SettingDef(
            key="URL_IMPORT_CRAWL_MAX_DEPTH",
            category=cat,
            label="Max Crawl Depth",
            type="int",
            default=2,
            min=1,
            max=3,
            description="Maximum link-following depth when crawling from a URL.",
        ),
        SettingDef(
            key="URL_IMPORT_IMAGE_MAX_SIZE_MB",
            category=cat,
            label="Max Image Size (MB)",
            type="int",
            default=10,
            min=1,
            max=50,
            advanced=True,
            description="Maximum size in MB for a single image download.",
        ),
    ]

    return {d.key: d for d in defs}


SETTINGS_REGISTRY: dict[str, SettingDef] = _build_registry()

# Category display order and labels
CATEGORY_META: dict[str, str] = {
    "copilot": "Workflow Copilot",
    "agents": "Agents",
    "tools": "Tools",
    "knowledge-indexing": "Knowledge Indexing",
    "knowledge-retrieval": "Knowledge Retrieval",
    "compaction": "Compaction",
    "sources": "Sources",
}


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def _coerce(value_str: str, setting_type: str) -> Any:
    """Coerce a string value from the DB to the setting's Python type."""
    if setting_type == "int":
        return int(value_str)
    if setting_type == "float":
        return float(value_str)
    if setting_type == "bool":
        return value_str.lower() in ("true", "1", "yes")
    return value_str  # str


def _has_g_context() -> bool:
    """Check if Flask app context is active (flask.g available).

    Returns False when called from a bare thread with no Flask context,
    preventing RuntimeError on flask.g access.
    """
    return has_app_context()


def _load_overrides() -> dict[str, str]:
    """Load all overrides from ai.project_settings into a dict.

    Cached in flask.g for the duration of the request.  When called
    outside a request context (e.g. Celery tasks, background threads)
    the query runs uncached — still a single lightweight SELECT.
    """
    use_cache = _has_g_context()

    if use_cache:
        cache = getattr(g, "_settings_cache", None)
        if cache is not None:
            return cache

    try:
        rows = db.session.execute(
            text(f'SELECT key, value FROM "{AI_SCHEMA}".project_settings')
        ).fetchall()
        result = {row[0]: row[1] for row in rows if row[1] is not None}
    except Exception:
        logger.warning("Failed to load project_settings overrides", exc_info=True)
        result = {}

    if use_cache:
        g._settings_cache = result
    return result


def get_setting(key: str) -> Any:
    """Read a setting value — DB override first, then registry default.

    Uses per-request caching (single SELECT on first call).
    """
    defn = SETTINGS_REGISTRY.get(key)
    if defn is None:
        raise KeyError(f"Unknown setting: {key}")

    overrides = _load_overrides()
    raw = overrides.get(key)
    if raw is not None:
        try:
            return _coerce(raw, defn.type)
        except (ValueError, TypeError):
            logger.warning("Bad stored value for %s=%r, using default", key, raw)

    return defn.default


def get_all_settings() -> dict:
    """Return the full registry merged with DB overrides, grouped by category.

    Returns:
        {
            "categories": {
                "copilot": {
                    "label": "Copilot",
                    "settings": [ { key, label, description, type, default, value, ... }, ... ]
                },
                ...
            }
        }
    """
    overrides = _load_overrides()
    categories: dict[str, dict] = {}

    for key, defn in SETTINGS_REGISTRY.items():
        cat = defn.category
        if cat not in categories:
            categories[cat] = {
                "label": CATEGORY_META.get(cat, cat),
                "settings": [],
            }

        raw = overrides.get(key)
        value = defn.default
        if raw is not None:
            try:
                value = _coerce(raw, defn.type)
            except (ValueError, TypeError):
                pass

        # Mask secret values — only reveal whether one is configured
        if defn.secret:
            display_value = SECRET_MASK if value else ""
        else:
            display_value = value

        setting_dict: dict[str, Any] = {
            "key": defn.key,
            "label": defn.label,
            "description": defn.description,
            "type": defn.type,
            "default": "" if defn.secret else defn.default,
            "value": display_value,
            "advanced": defn.advanced,
            "secret": defn.secret,
        }
        if defn.min is not None:
            setting_dict["min"] = defn.min
        if defn.max is not None:
            setting_dict["max"] = defn.max
        if defn.choices:
            setting_dict["choices"] = defn.choices
        if defn.subcategory:
            setting_dict["subcategory"] = defn.subcategory

        categories[cat]["settings"].append(setting_dict)

    return {"categories": categories}


def validate_setting(key: str, value: Any) -> tuple[bool, str]:
    """Validate a value against the registry definition.

    Returns (ok, error_message).
    """
    defn = SETTINGS_REGISTRY.get(key)
    if defn is None:
        return False, f"Unknown setting: {key}"

    # Type coercion check
    try:
        coerced = _coerce(str(value), defn.type)
    except (ValueError, TypeError):
        return False, f"Invalid {defn.type} value: {value}"

    # Range check for numeric types
    if defn.type in ("int", "float"):
        if defn.min is not None and coerced < defn.min:
            return False, f"Value must be >= {defn.min}"
        if defn.max is not None and coerced > defn.max:
            return False, f"Value must be <= {defn.max}"

    # Choices check
    if defn.choices and str(value) not in defn.choices:
        return False, f"Value must be one of: {', '.join(defn.choices)}"

    return True, ""
