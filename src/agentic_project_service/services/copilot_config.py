"""Copilot configuration — centralized runtime constants.

All tunable parameters for the workflow copilot live here so they're
easy to find, review, and adjust without digging through the 2000+ line
copilot service module.
"""

# ---------------------------------------------------------------------------
# Workflow state sanitization
# ---------------------------------------------------------------------------
# These limits protect against prompt injection and context overflow when
# the current workflow state is serialized and injected into the LLM prompt.

# Maximum length for block display names before truncation.
MAX_BLOCK_NAME_LEN = 100

# Maximum length for any single config value (e.g., a system prompt or
# code snippet inside a block config) before it's truncated with "... [truncated]".
MAX_CONFIG_VALUE_LEN = 2000

# Hard cap on the total serialized workflow state (JSON string length).
# If exceeded, all block configs are replaced with {"_truncated": True}.
MAX_TOTAL_STATE_LEN = 50_000

# Maximum nesting depth when recursively truncating config dicts.
MAX_CONFIG_DEPTH = 10

# ---------------------------------------------------------------------------
# Copilot LLM model configuration
# ---------------------------------------------------------------------------
# The model used by the copilot agent itself (not the models available
# inside workflow agent blocks). Can be overridden per-project via the
# PUT /copilot/settings/model endpoint.

# Fallback model when no project-level override is set.
DEFAULT_COPILOT_MODEL = "claude-opus-4-6"

# Allowed models for the copilot agent. Displayed in the copilot model
# picker UI. Only function-calling-capable models should be listed here.
# Verified against the LiteLLM model registry (1.83.14 as currently resolved
# in uv.lock — a >= floor, not a hard pin, so a lock bump can shift it) — all
# entries return supports_function_calling=True via litellm.get_model_info().
# test_copilot_picker_models.py enforces this (resolves + cost + fcall) per CI,
# so a lock bump that drops an entry fails CI rather than silently shipping.
# Format: list of (display_label, model_id) tuples.
COPILOT_MODEL_OPTIONS = [
    # Anthropic
    ("Claude Opus 4.7", "claude-opus-4-7"),
    ("Claude Opus 4.6", "claude-opus-4-6"),
    ("Claude Sonnet 4.6", "claude-sonnet-4-6"),
    ("Claude Haiku 4.5", "claude-haiku-4-5"),
    # OpenAI
    ("GPT-5.2 Pro", "gpt-5.2-pro"),
    ("GPT-5.2", "gpt-5.2"),
    ("GPT-5", "gpt-5"),
    ("GPT-5 Mini", "gpt-5-mini"),
    ("GPT-4.1", "gpt-4.1"),
    ("GPT-4.1 Mini", "gpt-4.1-mini"),
    # OpenAI — reasoning (o-series). Safe to list again now that the agent
    # reasoning path no longer requests reasoning *summaries* by default
    # (PR #520): the summary request is what required OpenAI org-verification
    # and 400'd unverified orgs ("Your organization must be verified to
    # generate reasoning summaries"). Verified-org deployments can opt
    # summaries back in via OPENAI_REASONING_SUMMARY=1. o3-pro stays out.
    ("o3", "o3"),
    ("o4 Mini", "o4-mini"),
    # Google
    ("Gemini 3.1 Pro (Preview)", "gemini/gemini-3.1-pro-preview"),
    ("Gemini 2.5 Pro", "gemini/gemini-2.5-pro"),
    ("Gemini 2.5 Flash", "gemini/gemini-2.5-flash"),
]

# ---------------------------------------------------------------------------
# Copilot agent runtime parameters
# ---------------------------------------------------------------------------

# LLM sampling temperature for the copilot agent's ReAct loop.
# Lower = more deterministic tool calls; higher = more creative responses.
COPILOT_TEMPERATURE = 0.7

# Maximum number of ReAct reasoning steps (think → tool_call → observe)
# before the agent is forced to produce a final answer.
COPILOT_MAX_STEPS = 25

# ---------------------------------------------------------------------------
# Asset display truncation
# ---------------------------------------------------------------------------

# When the copilot's get_asset_details tool returns an agent's system
# prompt, it's truncated to this many characters to avoid flooding the
# copilot's context window with large prompts.
SYSTEM_PROMPT_TRUNCATE = 1000

# Default model for inline agent blocks when the user doesn't specify one.
AGENT_DEFAULT_MODEL = "gpt-5.2"

# ---------------------------------------------------------------------------
# UI status messages (used by the SSE streaming route)
# ---------------------------------------------------------------------------
# Maps ReAct event types to human-readable status strings shown in the
# copilot chat UI while the agent is working.

STATUS_MESSAGES = {
    "step_started": "Thinking...",
    "tool_call": None,  # handled specially — uses tool name
    "tool_result": "Analyzing result...",
    "step_completed": None,  # silent
    "model_fallback": "Switching models...",
    "proactive_compact": "Managing context...",
    "reactive_compact": "Managing context...",
    "compaction": "Summarizing conversation...",
    "output_recovery": "Recovering output...",
}

# Maps copilot tool names to status messages shown while that tool executes.
TOOL_STATUS = {
    "modify_workflow": "Modifying workflow...",
    "get_block_info": "Looking up block details...",
    "get_db_schema": "Checking database schema...",
    "list_project_assets": "Discovering project assets...",
    "get_asset_details": "Fetching asset details...",
    "execute_public_sql": "Running SQL query...",
    "get_workflow_run_logs": "Reading execution logs...",
    "manage_project_asset": "Managing project asset...",
}
