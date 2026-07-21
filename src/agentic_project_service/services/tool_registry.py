# services/tool_registry.py
"""Load tools for an agent at execution time."""

import json
import logging
import re

from sqlalchemy import text
from sqlalchemy.orm import Session

from agentic.agent.tools import BuiltinTool, CustomTool, KnowledgeSearchTool, ToolDefinition

from ..models.tenant import AgentKnowledgeBase, AgentMcpServer, AgentTool, KnowledgeBase, Tool
from ..services.context_handler import create_and_execute
from ..tools.builtin import BUILTIN_HANDLERS, BUILTIN_TOOL_DEFINITIONS
from . import billing_port as billing
from .run_context import (
    get_run_id,
    new_request_id,
    next_call_seq,
)
from .settings_registry import get_setting


def _derive_tool_idempotency_inputs(action: str, arguments: dict | None) -> tuple[str, str]:
    """Derive (ref_id, idempotency_key) for a tool invocation.

    Spec line 132 calls for a key shaped like sha256(org_id, action, run_id,
    tool_call_id). The framework doesn't expose the LLM-provided tool_call_id
    to the handler signature, so this function approximates it with:

        - run_id from the run_context run_id contextvar (bound by the
          agent_run/orchestration_run/workflow_run entry point);
        - args_hash = first 16 hex of sha256(json.dumps(args, sort_keys=True)) — same
          args on retry produce the same hash so the retry collides on the
          UNIQUE(org_id, idempotency_key) index;
        - call_seq, a per-(run_id, action, args_hash) monotonic counter so
          duplicate same-args calls within one run (parallel tool_use,
          retry-within-loop) each get a distinct key. Without call_seq, the
          UNIQUE index dedupes them and silently under-charges.

    On retry of the same run with identical sequence, every call still
    produces the same key (run_id/args_hash/seq all match) so no double
    charge. Divergent retries (different tool order, different args) get
    different keys, which is correct — they're different ops.

    Falls back to uuid4 when no run_id is bound — preserves the prior
    behavior for non-agent callers (Celery tasks, internal calls) that
    have their own explicit idempotency_key arg.

    JSON serializer note: tool arguments arrive as JSON-decoded primitives
    (str/int/float/bool/list/dict/None) since the LLM API only emits JSON,
    so json.dumps is safe without ``default=`` — using sort_keys for
    deterministic key ordering is sufficient. We pass ``default=repr`` as a
    defense-in-depth fallback in case a future caller passes a Python-native
    object; repr() is stable for the common types where str()'s output
    depends on hash randomization (sets/frozensets).
    """
    import hashlib
    import json

    run_id = get_run_id()
    if run_id is None:
        rid = new_request_id()
        return rid, rid
    try:
        args_blob = json.dumps(arguments or {}, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        # Unserializable args (raw bytes, custom classes with broken repr)
        # — fall back to a short uuid suffix so retries collide on the
        # run_id portion only.
        args_blob = new_request_id()
    args_hash = hashlib.sha256(args_blob.encode()).hexdigest()[:16]
    # next_call_seq only returns None when no run_id is bound, and we've
    # already short-circuited above on that. Assert the contract so any
    # future change that returns None for another reason fails loudly here
    # rather than silently producing colliding keys.
    seq = next_call_seq(action, args_hash)
    assert seq is not None, "next_call_seq returned None despite run_id bound"
    ref_id = f"{run_id}:{action}:{args_hash}:{seq}"
    return ref_id, ref_id


logger = logging.getLogger(__name__)


# Tools with their own dedicated billing action name. Anything not in this
# map bills as the generic ``agent_tool_call``. ``web_search_deep`` has no
# tool of its own — it (and ``web_search_deep_reasoning``) are the pricier
# actions a ``web_search`` call resolves to based on its ``search_type``
# (see _resolve_tool_billing_action).
#
# web_scrape is the builtin counterpart to the URL-extraction Celery path:
# both materialize a remote page locally, so they share the same catalog
# action (and price). Without this entry, every builtin web_scrape call
# bills as agent_tool_call (1 credit) instead of web_scrape (5 credits).
_TOOL_BILLING_ACTION: dict[str, str] = {
    "code_execute": "agent_tool_code_execute",
    "web_search": "web_search",
    # NOTE: no `web_search_deep` entry — there is no BuiltinTool by that name;
    # the deep actions are resolved from a web_search call's search_type via
    # _WEB_SEARCH_DEEP_ACTIONS below, never via a tool_name lookup.
    "web_scrape": "web_scrape",
}

# web_search ``search_type`` values that bill as a pricier deep tier instead
# of the standard ``web_search`` action.
_WEB_SEARCH_DEEP_ACTIONS: dict[str, str] = {
    "deep": "web_search_deep",
    "deep-reasoning": "web_search_deep_reasoning",
}


def _resolve_tool_billing_action(tool_name: str, arguments: dict | None = None) -> str:
    """Map a tool invocation to its billing action.

    ``arguments`` is inspected for tools whose price depends on a call
    parameter: a ``web_search`` with a deep ``search_type`` ('deep' or
    'deep-reasoning') runs Exa's agentic deep search and must bill the pricier
    matching action. For this to be correct when ``search_type`` is forced via
    an agent's config_override, the override must be applied BEFORE this
    resolver runs (see how the handler is composed in
    load_all_tools_for_agent) — otherwise a UI-pinned deep search would run
    deep but bill the standard rate.

    Returns ``agent_tool_call`` for anything not in _TOOL_BILLING_ACTION,
    matching the spec's "charge per generic tool invocation" rule.
    """
    if tool_name == "web_search" and arguments:
        deep_action = _WEB_SEARCH_DEEP_ACTIONS.get(arguments.get("search_type"))
        if deep_action:
            return deep_action
    return _TOOL_BILLING_ACTION.get(tool_name, "agent_tool_call")


def _wrap_handler_with_billing(handler, tool_name: str):
    """Wrap a BuiltinTool handler so each invocation is billed.

    Derives a deterministic idempotency key from the current run_id (bound
    in the run_context run_id contextvar before the agent loop) + a hash of the
    tool's arguments. Retries of the same agent_run that re-invoke the
    same tool with the same arguments collide on the same key, so billing
    rejects the duplicate with HTTP 200 (already-existed) instead of
    minting a second charge. Falls back to uuid4 outside agent runs.

    Billing goes through the billing port (services/billing_port.py) — the
    no-op adapter (OSS build, unit tests, local dev) makes both calls inert;
    the cloud adapter enforces balance/charge against BILLING_ORG_ID.
    """

    from functools import wraps

    @wraps(handler)
    def billing_wrapper(arguments, context):
        action = _resolve_tool_billing_action(tool_name, arguments)
        ref_id, step_id = _derive_tool_idempotency_inputs(action, arguments)
        billing.check_balance(estimated_cost=1)
        result = handler(arguments, context)
        # Platform-misconfig opt-out: a handler can mark its result with
        # `_platform_error: true` (e.g., EXA_API_KEY missing from pod env in
        # web_search_handler) and the wrapper will NOT debit the tenant.
        # The marker is internal — it's stripped from `result` before the
        # result is returned to the agent loop so the LLM never sees the
        # internal signaling key in its tool-result message.
        is_platform_error, cleaned_result = _check_and_strip_platform_error(result)
        if not is_platform_error:
            billing.charge(
                action=action,
                quantity=1,
                ref_type="tool_call",
                ref_id=ref_id,
                idempotency_parts=(step_id,),
            )
        return cleaned_result

    return billing_wrapper


def _check_and_strip_platform_error(result):
    """Detect the `_platform_error: true` marker in a handler's JSON result
    and return ``(is_platform_error, cleaned_result)``.

    When the marker is present, returns the result with the marker key
    removed (re-encoded JSON). When absent, returns the result unchanged.
    Non-string results, non-JSON strings, JSON that isn't a dict, and
    dicts without the marker all default to (False, result) — i.e., bill
    normally and don't mutate the result.
    """
    if not isinstance(result, str):
        return False, result
    try:
        parsed = json.loads(result)
    except (ValueError, TypeError):
        return False, result
    if not (isinstance(parsed, dict) and parsed.get("_platform_error") is True):
        return False, result
    cleaned = {k: v for k, v in parsed.items() if k != "_platform_error"}
    return True, json.dumps(cleaned)


def _wrap_tool_execute_with_billing(tool: ToolDefinition) -> None:
    """Wrap a non-BuiltinTool's .execute() method so each call is billed.

    BuiltinTools are billed via _wrap_handler_with_billing (closer to the
    handler). CustomTools and McpTools own their own execute(), so we mutate
    the bound method here. KnowledgeSearchTool is intentionally NOT wrapped
    — its retrievals are billed inside knowledge_search.py at the actual
    chunk-retrieval step, which is finer-grained than the wrapper layer.
    """
    if isinstance(tool, KnowledgeSearchTool | BuiltinTool):
        return

    original_execute = tool.execute
    tool_name = tool.name

    def billing_execute(arguments, context):
        action = _resolve_tool_billing_action(tool_name, arguments)
        ref_id, step_id = _derive_tool_idempotency_inputs(action, arguments)
        billing.check_balance(estimated_cost=1)
        result = original_execute(arguments, context)
        billing.charge(
            action=action,
            quantity=1,
            ref_type="tool_call",
            ref_id=ref_id,
            idempotency_parts=(step_id,),
        )
        return result

    tool.execute = billing_execute


def _ensure_app_context(func, app):
    """Wrap a function so it pushes Flask app context if not already present.

    When the orchestration supervisor delegates to sub-agents, tool execution
    may run in a ThreadPoolExecutor thread without Flask's application context.
    This wrapper captures the app at tool-load time and pushes context as needed.
    """
    if app is None:
        return func

    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            from flask import current_app as _ca

            _ca._get_current_object()
            return func(*args, **kwargs)
        except RuntimeError:
            with app.app_context():
                return func(*args, **kwargs)

    return wrapper


def _get_flask_app():
    """Capture Flask app reference if available."""
    try:
        from flask import current_app

        return current_app._get_current_object()
    except RuntimeError:
        return None


_BUILTIN_DEFS = {d["name"]: d for d in BUILTIN_TOOL_DEFINITIONS}


def _make_search_handler(db_session):
    """Create a search handler closure that wraps create_and_execute().

    The handler is called by KnowledgeSearchTool.execute() during the ReAct loop.
    It needs access to the db_session for retrieval queries.

    Each invocation runs on its OWN short-lived session bound to the request
    session's engine. The agent loop runs concurrency-safe tools (knowledge_search
    is one) in a ThreadPoolExecutor, submitting each via
    ``contextvars.copy_context().run`` — which copies Flask's app-context
    ContextVar into the workers, so all parallel calls would otherwise resolve
    ``db.session`` to the SAME app-context-scoped Session. SQLAlchemy Sessions
    are not thread-safe, so concurrent ``commit()`` calls collide with
    "method 'commit()' is already in progress". Using a dedicated session per
    call (mirroring context_handler._search_single_kb) isolates them.
    """
    app = _get_flask_app()
    engine = db_session.get_bind()

    def _raw_handler(query, kb_configs, max_tokens, session_history):
        call_session = Session(bind=engine)
        try:
            handler_id, result = create_and_execute(
                db_session=call_session,
                query=query,
                knowledge_base_configs=kb_configs,
                max_context_tokens=max_tokens,
                session_history=session_history,
            )
            # Commit the context_handler immediately so it's visible in the DB
            # even before the ReAct loop completes
            call_session.commit()
        finally:
            call_session.close()

        formatted = result.get("formatted_context", "")
        metadata = {"context_handler_id": handler_id}

        if isinstance(formatted, list):
            block_types = [b.get("type", "unknown") for b in formatted if isinstance(b, dict)]
            image_count = block_types.count("image_url")
            text_count = block_types.count("text")
            logger.info(
                "KB search handler returning multimodal: %d blocks (%d text, %d image_url)",
                len(formatted),
                text_count,
                image_count,
            )
            # Add framing header (mirrors the pre-loaded context path)
            content = [
                {"type": "text", "text": "Retrieved context from knowledge base:"},
                *formatted,
            ]
            return content, metadata
        else:
            logger.info(
                "KB search handler returning text-only: %d chars",
                len(str(formatted)),
            )
        return str(formatted), metadata

    return _ensure_app_context(_raw_handler, app)


def build_kb_tools_for_agent(
    agent_id: str,
    db_session,
) -> dict[str, KnowledgeSearchTool]:
    """Auto-generate knowledge search tools from agent KB assignments."""
    tools: dict[str, KnowledgeSearchTool] = {}

    assignments = AgentKnowledgeBase.query.filter_by(agent_id=agent_id).all()
    if not assignments:
        return tools

    search_handler = _make_search_handler(db_session)

    # Load KB metadata for names/descriptions
    kb_ids = [str(a.knowledge_base_id) for a in assignments]
    kb_rows = KnowledgeBase.query.filter(KnowledgeBase.id.in_(kb_ids)).all()
    kb_map = {str(kb.id): kb for kb in kb_rows}

    all_configs = []
    for assignment in assignments:
        kb = kb_map.get(str(assignment.knowledge_base_id))
        if not kb:
            continue
        config = assignment.config or {}
        kb_retrieval_config = kb.retrieval_config or {}

        # Precedence for top_k (and retrieval_method):
        #   1. Agent assignment override (agent_knowledge_bases.config)
        #   2. KB's own retrieval_config (knowledge_bases.retrieval_config)
        #   3. Project-wide default (KB_DEFAULT_TOP_K setting)
        # Earlier code skipped step 2 and jumped straight from (1) to (3),
        # which silently overrode the KB's configured top_k whenever the
        # agent assignment didn't pin its own value.
        resolved_top_k = (
            config.get("top_k")
            or kb_retrieval_config.get("top_k")
            or get_setting("KB_DEFAULT_TOP_K")
        )
        resolved_method = config.get("retrieval_method") or kb_retrieval_config.get("method")

        all_configs.append(
            {
                "id": str(kb.id),
                "name": kb.name,
                "retrieval_method": resolved_method,
                "top_k": resolved_top_k,
            }
        )

    if not all_configs:
        return tools

    if len(all_configs) == 1:
        single_config = assignments[0].config or {}
        max_tokens = single_config.get(
            "max_context_tokens", get_setting("KB_DEFAULT_MAX_CONTEXT_TOKENS")
        )
        kb_name = all_configs[0]["name"]
        kb_desc = kb_map.get(all_configs[0]["id"])
        desc_suffix = f" {kb_desc.description}" if kb_desc and kb_desc.description else ""
        description = f"Search the '{kb_name}' knowledge base.{desc_suffix}"
        include_filter = False
    else:
        max_tokens = get_setting("KB_DEFAULT_MAX_CONTEXT_TOKENS")
        kb_lines = []
        for c in all_configs:
            kb = kb_map.get(c["id"])
            if kb and kb.description:
                kb_lines.append(f"- {c['name']}: {kb.description}")
            else:
                kb_lines.append(f"- {c['name']}")
        kb_list = "\n".join(kb_lines)
        description = (
            f"Search across knowledge bases:\n{kb_list}\n"
            "Use the knowledge_base_names parameter to target specific ones,"
            " or omit it to search all."
        )
        include_filter = True

    tools["knowledge_search"] = KnowledgeSearchTool(
        name="knowledge_search",
        description=description,
        knowledge_base_configs=all_configs,
        max_context_tokens=max_tokens,
        include_kb_filter=include_filter,
        search_handler=search_handler,
    )

    return tools


_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")


def _introspect_table_metadata(db_session, schemas_config: dict[str, list[str]]) -> str:
    """Query information_schema for column metadata of selected tables.

    Returns a detailed multi-line format with NOT NULL, DEFAULT, and FK info,
    plus a "To insert" hint per table listing required vs auto-generated columns.

    Uses parameterized queries to prevent SQL injection.
    """
    if not schemas_config:
        return ""

    # Validate all identifiers before querying
    schemas_list = []
    tables_list = []
    for schema, tables in schemas_config.items():
        if not _IDENTIFIER_RE.match(schema):
            logger.warning("Invalid schema name in config: %s", schema)
            return ""
        for table in tables:
            if not _IDENTIFIER_RE.match(table):
                logger.warning("Invalid table name in config: %s", table)
                return ""
            schemas_list.append(schema)
            tables_list.append(table)

    if not schemas_list:
        return ""

    # Query columns with PK, nullable, default, and FK info
    result = db_session.execute(
        text("""
            SELECT
                c.table_schema,
                c.table_name,
                c.column_name,
                c.data_type,
                CASE WHEN pk_kcu.column_name IS NOT NULL THEN true ELSE false END AS is_pk,
                c.is_nullable,
                c.column_default,
                fk_ccu.table_schema AS fk_schema,
                fk_ccu.table_name AS fk_table,
                fk_ccu.column_name AS fk_column
            FROM information_schema.columns c
            LEFT JOIN information_schema.table_constraints pk_tc
                ON pk_tc.table_schema = c.table_schema
                AND pk_tc.table_name = c.table_name
                AND pk_tc.constraint_type = 'PRIMARY KEY'
            LEFT JOIN information_schema.key_column_usage pk_kcu
                ON pk_kcu.constraint_name = pk_tc.constraint_name
                AND pk_kcu.table_schema = pk_tc.table_schema
                AND pk_kcu.column_name = c.column_name
            LEFT JOIN information_schema.key_column_usage fk_kcu
                ON fk_kcu.table_schema = c.table_schema
                AND fk_kcu.table_name = c.table_name
                AND fk_kcu.column_name = c.column_name
            LEFT JOIN information_schema.table_constraints fk_tc
                ON fk_tc.constraint_name = fk_kcu.constraint_name
                AND fk_tc.table_schema = fk_kcu.table_schema
                AND fk_tc.constraint_type = 'FOREIGN KEY'
            LEFT JOIN information_schema.constraint_column_usage fk_ccu
                ON fk_ccu.constraint_name = fk_tc.constraint_name
                AND fk_tc.constraint_type = 'FOREIGN KEY'
            WHERE (c.table_schema, c.table_name) IN (
                SELECT unnest(CAST(:schemas AS text[])), unnest(CAST(:tables AS text[]))
            )
            ORDER BY c.table_schema, c.table_name, c.ordinal_position
        """),
        {"schemas": schemas_list, "tables": tables_list},
    )

    # Collect column info per table
    # Each entry: (col_name, data_type, is_pk, is_nullable, column_default, fk_ref)
    table_cols: dict[str, list[tuple]] = {}
    for row in result:
        key = f"{row[0]}.{row[1]}"
        fk_ref = None
        if row[7] and row[8] and row[9]:
            fk_ref = f"{row[7]}.{row[8]}.{row[9]}"
        entry = (row[2], row[3], row[4], row[5], row[6], fk_ref)
        table_cols.setdefault(key, []).append(entry)

    # Deduplicate columns (FK joins can produce duplicates).
    # Prefer entries that have FK info over those without, so we don't
    # lose FK references for columns that are also part of a PK.
    for key in table_cols:
        best: dict[str, tuple] = {}
        for entry in table_cols[key]:
            col_key = entry[0]  # column name
            if col_key not in best or (entry[5] is not None and best[col_key][5] is None):
                best[col_key] = entry
        # Preserve original column order
        seen = set()
        deduped = []
        for entry in table_cols[key]:
            if entry[0] not in seen:
                seen.add(entry[0])
                deduped.append(best[entry[0]])
        table_cols[key] = deduped

    lines = []
    for table_key, cols in table_cols.items():
        lines.append(f"Table: {table_key}")
        lines.append("  Columns:")

        required_cols = []
        auto_cols = []

        for col_name, data_type, is_pk, is_nullable, col_default, fk_ref in cols:
            flags = []
            if is_pk:
                flags.append("PK")
            if col_default:
                # Shorten common defaults for readability
                default_display = col_default
                if "nextval(" in default_display:
                    default_display = "auto-increment"
                flags.append(f"DEFAULT {default_display}")
                auto_cols.append(col_name)
            elif is_nullable == "NO" and not is_pk:
                flags.append("NOT NULL")
                required_cols.append(col_name)
            if is_pk and col_default:
                pass  # already in auto_cols
            elif is_pk and not col_default:
                required_cols.append(col_name)
            if fk_ref:
                flags.append(f"FK -> {fk_ref}")
            suffix = f"  {', '.join(flags)}" if flags else ""
            lines.append(f"    {col_name:<20s} {data_type}{suffix}")

        # Build insert hint
        if required_cols or auto_cols:
            hint_parts = []
            if required_cols:
                hint_parts.append(f"provide {', '.join(required_cols)}")
            if auto_cols:
                hint_parts.append(f"omit {', '.join(auto_cols)} (auto-generated)")
            lines.append(f"  To insert: {'. '.join(hint_parts)}.")

    return "\n".join(lines)


def load_all_tools_for_agent(
    agent_id: str,
    db_session,
    max_tool_output_length: int | None = None,
    default_max_result_chars: int | None = None,
) -> dict[str, ToolDefinition]:
    """Load all tools assigned to an agent: built-in + custom.

    Args:
        max_tool_output_length: Override for CustomTool HTTP response truncation.
        default_max_result_chars: Override for ToolDefinition.max_result_chars.
    """
    tools: dict[str, ToolDefinition] = {}
    app = _get_flask_app()

    assignments = AgentTool.query.filter_by(agent_id=agent_id).all()

    for assignment in assignments:
        if assignment.tool_type == "builtin":
            tool_name = assignment.tool_name
            defn = _BUILTIN_DEFS.get(tool_name)
            handler = BUILTIN_HANDLERS.get(tool_name)
            if not defn or not handler:
                logger.warning("Unknown built-in tool: %s", tool_name)
                continue

            # Database tools: require schema config, build dynamic description
            if tool_name in ("database_query", "database_write"):
                schemas_config = (assignment.config_override or {}).get("schemas", {})
                schemas_config = {s: t for s, t in schemas_config.items() if t}
                if not schemas_config:
                    logger.info(
                        "Skipping %s for agent %s — no tables configured",
                        tool_name,
                        agent_id,
                    )
                    continue

                table_metadata = _introspect_table_metadata(db_session, schemas_config)
                if not table_metadata:
                    continue

                allowed_schemas = list(schemas_config.keys())
                allowed_tables = set()
                for schema_tables in schemas_config.values():
                    allowed_tables.update(schema_tables)

                if tool_name == "database_query":
                    description = (
                        "Run a read-only SQL SELECT query against the project database.\n\n"
                        f"{table_metadata}\n\n"
                        "Usage:\n"
                        "- Only SELECT statements are allowed\n"
                        "- Use the exact table and column names shown above\n"
                        "- Tables can be referenced as 'table_name' or 'schema.table_name'"
                    )
                else:
                    description = (
                        "Insert, update, or delete records in the project database.\n\n"
                        f"{table_metadata}\n\n"
                        "Usage:\n"
                        "- table: bare name ('customers') or schema-qualified ('public.customers')\n"
                        "- insert: set data to an object (one row) or array of objects (batch)\n"
                        "- insert: omit columns marked DEFAULT — they are auto-generated\n"
                        "- update: data = columns to set, where = filter (required)\n"
                        "- delete: where = filter (required, no mass deletes)"
                    )

                def make_restricted_handler(h, schemas, tables, sc):
                    def restricted(arguments, context):
                        arguments["_allowed_schemas"] = schemas
                        arguments["_allowed_tables"] = tables
                        arguments["_schemas_config"] = sc
                        return h(arguments, context)

                    return restricted

                db_builtin = BuiltinTool(
                    name=defn["name"],
                    description=description,
                    input_schema=defn["input_schema"],
                    handler=_wrap_handler_with_billing(
                        _ensure_app_context(
                            make_restricted_handler(
                                handler, allowed_schemas, allowed_tables, schemas_config
                            ),
                            app,
                        ),
                        defn["name"],
                    ),
                )
                if default_max_result_chars is not None:
                    db_builtin.max_result_chars = default_max_result_chars
                tools[tool_name] = db_builtin
                continue

            # All other builtin tools — apply config overrides as forced
            # argument values.  Only inject keys that are declared in the
            # tool's input_schema to avoid leaking metadata (e.g. "rules")
            # into the handler's arguments dict.
            raw_overrides = assignment.config_override or {}
            schema_props = (defn.get("input_schema") or {}).get("properties", {})
            param_overrides = {k: v for k, v in raw_overrides.items() if k in schema_props}

            # Order matters: the override must run OUTSIDE the billing wrapper
            # so a price-affecting forced argument (e.g. web_search search_type
            # = deep) is present in `arguments` when the billing wrapper
            # resolves the action. Wrapping the other way bills the standard
            # rate while running the pricier deep search.
            tool_handler = _wrap_handler_with_billing(
                _ensure_app_context(handler, app), defn["name"]
            )

            if param_overrides:

                def make_overrides_handler(h, overrides):
                    def with_overrides(arguments, context):
                        arguments.update(overrides)
                        return h(arguments, context)

                    return with_overrides

                tool_handler = make_overrides_handler(tool_handler, param_overrides)

            builtin = BuiltinTool(
                name=defn["name"],
                description=defn["description"],
                input_schema=defn["input_schema"],
                handler=tool_handler,
            )
            if default_max_result_chars is not None:
                builtin.max_result_chars = default_max_result_chars
            tools[tool_name] = builtin

        elif assignment.tool_type == "custom" and assignment.tool_id:
            tool_row = db_session.get(Tool, assignment.tool_id)
            if tool_row:
                config = tool_row.config or {}
                custom_tool = CustomTool(
                    name=tool_row.name,
                    description=tool_row.description,
                    input_schema=tool_row.input_schema,
                    endpoint=config.get("endpoint", ""),
                    method=config.get("method", "POST"),
                    headers=config.get("headers", {}),
                    timeout=config.get("timeout_seconds", get_setting("CUSTOM_TOOL_TIMEOUT")),
                )
                if max_tool_output_length is not None:
                    custom_tool.max_output_length = max_tool_output_length
                if default_max_result_chars is not None:
                    custom_tool.max_result_chars = default_max_result_chars
                _wrap_tool_execute_with_billing(custom_tool)
                tools[tool_row.name] = custom_tool

    # 2. Knowledge search tools (auto-generated from ai.agent_knowledge_bases)
    kb_tools = build_kb_tools_for_agent(agent_id, db_session)
    tools.update(kb_tools)

    # 3. MCP tools
    mcp_tools = build_mcp_tools_for_agent(
        agent_id,
        db_session,
        default_max_result_chars=default_max_result_chars,
    )
    tools.update(mcp_tools)

    return tools


def build_mcp_tools_for_agent(
    agent_id: str,
    db_session,
    default_max_result_chars: int | None = None,
) -> dict[str, "ToolDefinition"]:
    """Discover and build tools from MCP servers assigned to an agent."""
    from agentic.agent.tools import McpTool
    from agentic.mcp.client import discover_mcp_tools

    servers = AgentMcpServer.query.filter_by(agent_id=agent_id, enabled=True).all()
    tools: dict[str, ToolDefinition] = {}

    for server in servers:
        try:
            mcp_tools = discover_mcp_tools(server.url, server.headers or {})
        except Exception:
            logger.warning(
                "Failed to discover tools from MCP server %s (%s)", server.name, server.url
            )
            continue

        for mcp_tool in mcp_tools:
            tool_name = f"mcp__{server.name}__{mcp_tool.name}"
            mcp_tool_def = McpTool(
                name=tool_name,
                description=mcp_tool.description,
                input_schema=mcp_tool.input_schema,
                server_name=server.name,
                server_url=server.url,
                server_headers=server.headers or {},
                mcp_tool_name=mcp_tool.name,
                is_concurrency_safe=mcp_tool.read_only_hint,
                is_read_only=mcp_tool.read_only_hint,
                is_destructive=mcp_tool.destructive_hint,
            )
            if default_max_result_chars is not None:
                mcp_tool_def.max_result_chars = default_max_result_chars
            _wrap_tool_execute_with_billing(mcp_tool_def)
            tools[tool_name] = mcp_tool_def

    return tools
