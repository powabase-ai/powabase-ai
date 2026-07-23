# services/hook_loader.py
"""Load hooks and tool rules for agent/orchestration execution."""

import logging
from agentic.agent.hooks import HookConfig
from ..models.tenant import AgentTool

logger = logging.getLogger(__name__)


def load_hooks_for_agent(agent_id: str) -> list[HookConfig]:
    """Load hook configs from ai.hooks for an agent."""
    # Import here to avoid circular imports at module level
    from ..models.tenant import Hook

    rows = (
        Hook.query.filter_by(agent_id=agent_id, enabled=True)
        .order_by(Hook.position, Hook.created_at)
        .all()
    )
    return [
        HookConfig(
            event=row.event,
            type=row.type,
            config=row.config or {},
            matcher=row.matcher,
            enabled=row.enabled,
            id=str(row.id),
            position=row.position,
        )
        for row in rows
    ]


def load_hooks_for_orchestration(orchestration_id: str) -> list[HookConfig]:
    """Load hook configs from ai.hooks for an orchestration."""
    from ..models.tenant import Hook

    # Defense-in-depth: `approval` hooks are rejected at create for
    # orchestrations (no approve endpoint), but filter them at load too so a
    # direct-SQL row can't hang a run on the un-serviceable approval wait.
    rows = (
        Hook.query.filter_by(orchestration_id=orchestration_id, enabled=True)
        .filter(Hook.type != "approval")
        .order_by(Hook.position, Hook.created_at)
        .all()
    )
    return [
        HookConfig(
            event=row.event,
            type=row.type,
            config=row.config or {},
            matcher=row.matcher,
            enabled=row.enabled,
            id=str(row.id),
            position=row.position,
        )
        for row in rows
    ]


def load_tool_rules_for_agent(agent_id: str) -> dict[str, list[dict]]:
    """Load runtime input rules from agent_tools.config_override.

    Returns: dict mapping tool_name -> list of rule dicts.
    Only includes tools that have rules configured.
    """
    assignments = AgentTool.query.filter_by(agent_id=agent_id).all()
    rules = {}
    for a in assignments:
        override = a.config_override or {}
        if "rules" in override and override["rules"]:
            rules[a.tool_name] = override["rules"]
    return rules
