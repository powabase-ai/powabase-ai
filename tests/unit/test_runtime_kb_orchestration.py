"""Runtime KBs flow to every orchestration sub-agent's tool build.

Behavioral (not source-inspection) coverage: mocks exactly what
``build_orchestration`` touches at the module level — the ORM lookups, the
``Agent``/``Orchestration`` engine classes, and the provider-key resolver —
then actually executes it end to end for two sub-agent entities, asserting
the resolved ``runtime_kb_configs`` reaches each sub-agent's
``load_all_tools_for_agent`` call.
"""

from unittest.mock import MagicMock

from agentic_project_service.services import orchestration as orch_service


class _FakeOrchRow:
    name = "orch-name"
    description = "orch-desc"
    strategy = "sequential"
    settings: dict = {}
    orchestrator_config: dict = {}


class _FakeAgentRow:
    def __init__(self, agent_id):
        self.id = agent_id
        self.model = "gpt-4o"
        self.system_prompt = "be helpful"
        self.name = f"agent-{agent_id}"
        self.settings: dict = {}


class _FakeEntity:
    def __init__(self, entity_ref_id, position):
        self.entity_type = "agent"
        self.entity_ref_id = entity_ref_id
        self.id = f"entity-{entity_ref_id}"
        self.role_description = "role"
        self.config: dict = {}
        self.position = position


class _FakeEntityQuery:
    def __init__(self, entities):
        self._entities = entities

    def filter_by(self, **_):
        return self

    def order_by(self, *_):
        return self

    def all(self):
        return self._entities


class _FakeOrchestrationEngine:
    """Stand-in for agentic.orchestration.orchestration.Orchestration."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.added_entities: list[dict] = []

    def add_entity(self, **kwargs):
        self.added_entities.append(kwargs)


def _install_fakes(monkeypatch, entities, agent_rows, load_tools_calls):
    fake_db_session = MagicMock()

    def fake_get(model_cls, obj_id):
        if model_cls is orch_service.OrchestrationModel:
            return _FakeOrchRow()
        if model_cls is orch_service.AgentModel:
            return agent_rows[obj_id]
        return None

    fake_db_session.get.side_effect = fake_get
    fake_db = MagicMock()
    fake_db.session = fake_db_session

    class _FakeOrchestrationEntityModel:
        query = _FakeEntityQuery(entities)
        position = None  # only referenced as an order_by() column argument

    def fake_load_all_tools_for_agent(agent_id, db_session, **kwargs):
        load_tools_calls.append((agent_id, kwargs))
        return {}

    monkeypatch.setattr(orch_service, "db", fake_db)
    monkeypatch.setattr(orch_service, "OrchestrationEntityModel", _FakeOrchestrationEntityModel)
    monkeypatch.setattr(orch_service, "Agent", lambda **kwargs: MagicMock(name="fake-agent"))
    monkeypatch.setattr(
        orch_service, "resolve_api_key_or_raise_for_drop_using", lambda *a, **k: "fake-key"
    )
    monkeypatch.setattr(orch_service, "get_user_provider_keys_with_dropped", lambda: ({}, {}))
    monkeypatch.setattr(
        orch_service,
        "get_setting",
        lambda key: {"MAX_TOOL_OUTPUT_LENGTH": 1000, "DEFAULT_MAX_RESULT_CHARS": 500}[key],
    )
    monkeypatch.setattr(orch_service, "load_all_tools_for_agent", fake_load_all_tools_for_agent)
    monkeypatch.setattr(orch_service, "Orchestration", _FakeOrchestrationEngine)


def _two_agent_entities():
    entities = [_FakeEntity("agent-a", 0), _FakeEntity("agent-b", 1)]
    agent_rows = {"agent-a": _FakeAgentRow("agent-a"), "agent-b": _FakeAgentRow("agent-b")}
    return entities, agent_rows


def test_build_orchestration_threads_runtime_kb_configs_to_both_entities(monkeypatch):
    entities, agent_rows = _two_agent_entities()
    load_tools_calls: list = []
    _install_fakes(monkeypatch, entities, agent_rows, load_tools_calls)

    runtime_kb_configs = [{"id": "kb-r"}]
    orch_row, orchestration = orch_service.build_orchestration(
        "orch-1", runtime_kb_configs=runtime_kb_configs
    )

    assert isinstance(orch_row, _FakeOrchRow)
    assert len(load_tools_calls) == 2
    called_agent_ids = {agent_id for agent_id, _kwargs in load_tools_calls}
    assert called_agent_ids == {"agent-a", "agent-b"}
    for _agent_id, kwargs in load_tools_calls:
        assert kwargs["runtime_kb_configs"] == runtime_kb_configs
    assert len(orchestration.added_entities) == 2


def test_build_orchestration_defaults_runtime_kb_configs_to_none(monkeypatch):
    entities, agent_rows = _two_agent_entities()
    load_tools_calls: list = []
    _install_fakes(monkeypatch, entities, agent_rows, load_tools_calls)

    orch_service.build_orchestration("orch-1")

    assert len(load_tools_calls) == 2
    for _agent_id, kwargs in load_tools_calls:
        assert kwargs["runtime_kb_configs"] is None
