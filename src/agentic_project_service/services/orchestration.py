"""Orchestration execution service."""

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from agentic import Agent
from agentic.orchestration.orchestration import Orchestration

from ..db import AI_SCHEMA, db
from ..models.tenant import (
    Agent as AgentModel,
    OrchestrationEntityModel,
    OrchestrationModel,
    OrchestrationRunModel,
    OrchestrationSessionModel,
)
from ..services.ai_provider_keys_resolver import (
    get_user_provider_keys_with_dropped,
    resolve_api_key_or_raise_for_drop_using,
)
from ..services.settings_registry import get_setting
from ..services.tool_registry import load_all_tools_for_agent

logger = logging.getLogger(__name__)


def build_orchestration(orch_id: str) -> tuple[OrchestrationModel, Orchestration]:
    """Build a core Orchestration object from DB models.

    Returns (orm_model, core_orchestration).
    """
    orch_row = db.session.get(OrchestrationModel, orch_id)
    if not orch_row:
        raise ValueError(f"Orchestration {orch_id} not found")

    entities = (
        OrchestrationEntityModel.query.filter_by(orchestration_id=orch_id)
        .order_by(OrchestrationEntityModel.position)
        .all()
    )

    raw_settings = orch_row.settings or {}

    # Frontend stores orchestrator_config nested inside settings;
    # fall back to the top-level column for API-written data.
    orchestrator_config = (
        raw_settings.get("orchestrator_config") or orch_row.orchestrator_config or {}
    )

    # Pass settings without the nested orchestrator_config to avoid duplication,
    # but use a copy to avoid mutating the ORM object (which could be flushed).
    settings = {k: v for k, v in raw_settings.items() if k != "orchestrator_config"}

    # Resolve provider keys ONCE for the orchestrator + every sub-agent.
    # Fixes issue #246's adjacent bug: the previous _resolve_api_key_for_model
    # read from x-provider-key-* HTTP headers that nothing in CP/FE/PS sets
    # (the org→project key migration in PRs #116/#120/#124 stripped the CP
    # injection). For Anthropic / Google / OpenRouter (no env-var fallback on
    # per-project pods) every orchestration run silently passed api_key=None
    # to litellm. The canonical DB-based resolver in
    # services.ai_provider_keys_resolver is what routes/agents.py uses.
    provider_keys, dropped = get_user_provider_keys_with_dropped()

    orchestrator_config = dict(orchestrator_config)
    orchestrator_model = orchestrator_config.get("model") or settings.get("model")
    if orchestrator_model and not orchestrator_config.get("api_key"):
        # Raises ProviderKeyDecryptDropped if the orchestrator's provider was
        # just self-healed for an undecryptable row — caller's SSE generator
        # surfaces it as a clear "please re-add" error event.
        resolved_key = resolve_api_key_or_raise_for_drop_using(
            orchestrator_model, provider_keys, dropped
        )
        if resolved_key:
            orchestrator_config["api_key"] = resolved_key

    orchestration = Orchestration(
        name=orch_row.name,
        description=orch_row.description or "",
        strategy=orch_row.strategy,
        orchestrator_config=orchestrator_config,
        settings=settings,
    )

    for entity in entities:
        if entity.entity_type == "agent":
            agent_row = db.session.get(AgentModel, entity.entity_ref_id)
            if not agent_row:
                logger.warning(
                    "Agent %s not found for entity %s",
                    entity.entity_ref_id,
                    entity.id,
                )
                continue
            agent = Agent(
                model=agent_row.model,
                system_prompt=agent_row.system_prompt or "",
                name=agent_row.name,
                api_key=resolve_api_key_or_raise_for_drop_using(
                    agent_row.model, provider_keys, dropped
                ),
                reasoning_effort=(agent_row.settings or {}).get("reasoning_effort"),
            )
            # Load tools for this sub-agent
            agent_tools = load_all_tools_for_agent(
                str(agent_row.id),
                db.session,
                max_tool_output_length=get_setting("MAX_TOOL_OUTPUT_LENGTH"),
                default_max_result_chars=get_setting("DEFAULT_MAX_RESULT_CHARS"),
            )

            entity_config = entity.config or {}
            orchestration.add_entity(
                entity_type="agent",
                agent=agent,
                role_description=entity.role_description,
                config=entity_config,
                position=entity.position or 0,
                agent_tools=agent_tools,
                # Pass through the registered agent's UUID so DelegateTool
                # can echo it back via on_run_complete; without this, child
                # agent_runs persist with agent_id NULL, the per-agent
                # dashboard breakdown under-counts, and the
                # idx_agent_runs_agent_created index sits on NULLs.
                agent_id=str(agent_row.id),
            )

    return orch_row, orchestration


def get_or_create_orchestration_session(
    orchestration_id: str,
    session_id: str | None = None,
    user_id: str | None = None,
) -> tuple[str, str, bool]:
    """Get or create an orchestration session.

    Returns (db_session_uuid, session_id, is_new).
    """
    if session_id:
        existing = OrchestrationSessionModel.query.filter_by(session_id=session_id).first()
        if existing:
            return str(existing.id), existing.session_id, False

    new_session_id = session_id or f"orch_sess_{uuid.uuid4().hex[:12]}"
    session = OrchestrationSessionModel(
        session_id=new_session_id,
        orchestration_id=orchestration_id,
        user_id=user_id,
    )
    db.session.add(session)
    db.session.flush()
    return str(session.id), new_session_id, True


def create_orchestration_run(
    session_uuid: str,
    orchestration_id: str,
    message: str,
    reasoning_requested: bool = False,
) -> tuple[str, str]:
    """Create an orchestration run record.

    Returns (db_run_uuid, run_id).
    """
    run_id = f"orch_run_{uuid.uuid4().hex[:12]}"
    run = OrchestrationRunModel(
        session_id=session_uuid,
        run_id=run_id,
        orchestration_id=orchestration_id,
        status="running",
        input_messages=[{"role": "user", "content": message}],
        started_at=datetime.now(UTC),
        reasoning_requested=reasoning_requested,
    )
    db.session.add(run)
    db.session.flush()
    return str(run.id), run_id


def update_orchestration_run(
    run_id: str,
    status: str | None = None,
    content: str | None = None,
    events: list[dict] | None = None,
    usage: dict | None = None,
    error: str | None = None,
    model: str | None = None,
) -> None:
    """Update an orchestration run by run_id.

    `usage` is unpacked into typed columns (the `usage` JSONB column was
    dropped in migration 0019). `model` is optional; orchestrations often
    don't have a single model (supervisor + entities mix), in which case
    the caller may omit it and the column stays NULL.
    """
    from .session import _unpack_usage  # local import avoids cycle

    updates = []
    params: dict[str, Any] = {"run_id": run_id}

    if status is not None:
        updates.append("status = :status")
        params["status"] = status
    if content is not None:
        updates.append("content = :content")
        params["content"] = content
    if events is not None:
        updates.append("events = CAST(:events AS jsonb)")
        params["events"] = json.dumps(events)
    if usage is not None:
        tokens = _unpack_usage(usage)
        updates.append("prompt_tokens = :prompt_tokens")
        updates.append("completion_tokens = :completion_tokens")
        updates.append("reasoning_tokens = :reasoning_tokens")
        updates.append("cached_tokens = :cached_tokens")
        updates.append("total_tokens = :total_tokens")
        params.update(tokens)
    if error is not None:
        updates.append("error = :error")
        params["error"] = error
    if model is not None:
        updates.append("model = :model")
        params["model"] = model

    updates.append("completed_at = NOW()")

    if updates:
        set_clause = ", ".join(updates)
        db.session.execute(
            text(
                f'UPDATE "{AI_SCHEMA}".orchestration_runs SET {set_clause} WHERE run_id = :run_id'
            ),
            params,
        )
