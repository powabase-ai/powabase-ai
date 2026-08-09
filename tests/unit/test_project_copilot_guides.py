"""The backend's play_guide allowlist is the contract with the frontend registry.

If they drift, play_guide either offers a walkthrough the UI cannot run, or the UI
ships a walkthrough the copilot can never launch. Both fail silently at runtime.
"""

from agentic_project_service.services.project_copilot import (
    GUIDE_SEQUENCE_IDS,
    _PLAY_GUIDE_SCHEMA,
)

# NOTE: the schema's property is `sequence_id`, NOT `guide_id` — verified directly
# against the schema definition in services/project_copilot.py. Getting this wrong
# yields a KeyError, not an assertion failure, which reads like a missing module
# rather than a renamed field.

EXPECTED = (
    "connect", "create-table", "add-sources", "create-knowledge-base",
    "create-agent", "create-orchestration", "create-workflow", "sql-query",
    "create-storage-bucket", "add-user", "create-rls-policy",
    "schema-visualizer", "database-functions", "database-triggers",
    "database-indexes", "database-roles", "enable-extension",
    "auth-providers", "realtime-inspector", "llm-provider-keys",
    "manage-compute",
)


def test_guide_ids_are_the_declared_catalogue():
    assert GUIDE_SEQUENCE_IDS == EXPECTED


def test_play_guide_schema_constrains_the_id_to_the_catalogue():
    """An unconstrained enum lets the model invent a sequence id that no-ops in the UI."""
    enum = _PLAY_GUIDE_SCHEMA["properties"]["sequence_id"]["enum"]
    assert sorted(enum) == sorted(EXPECTED)
    assert _PLAY_GUIDE_SCHEMA["required"] == ["sequence_id"]
