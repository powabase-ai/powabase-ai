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
from agentic_project_service.services.tool_caller import ToolCaller


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
        "orch-1", runtime_kb_configs=runtime_kb_configs, caller=ToolCaller.service()
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

    orch_service.build_orchestration("orch-1", caller=ToolCaller.service())

    assert len(load_tools_calls) == 2
    for _agent_id, kwargs in load_tools_calls:
        assert kwargs["runtime_kb_configs"] is None


_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
_KB = "33333333-3333-3333-3333-333333333333"
_ORCH = "4a8b2d3f-6c8e-4f22-8b4d-9e3a7c5f2b81"


def _scoped_setup(monkeypatch):
    entities = [_FakeEntity(_A, 0), _FakeEntity(_B, 1)]
    agent_rows = {_A: _FakeAgentRow(_A), _B: _FakeAgentRow(_B)}
    calls: list = []
    _install_fakes(monkeypatch, entities, agent_rows, calls)
    return calls


def test_agent_scoped_entries_reach_only_the_named_agent(monkeypatch):
    calls = _scoped_setup(monkeypatch)
    runtime = [{"id": "kb-shared"}, {"id": "kb-a-only", "agent_ids": [_A]}]
    orch_service.build_orchestration(
        "orch-1", runtime_kb_configs=runtime, caller=ToolCaller.service()
    )
    by_agent = {agent_id: kwargs["runtime_kb_configs"] for agent_id, kwargs in calls}
    assert by_agent[_A] == [{"id": "kb-shared"}, {"id": "kb-a-only"}]
    assert by_agent[_B] == [{"id": "kb-shared"}]
    assert runtime[1] == {"id": "kb-a-only", "agent_ids": [_A]}  # caller's list untouched


def test_an_agent_with_no_matching_entry_gets_none(monkeypatch):
    calls = _scoped_setup(monkeypatch)
    runtime = [{"id": "kb-a-only", "agent_ids": [_A]}]
    orch_service.build_orchestration(
        "orch-1", runtime_kb_configs=runtime, caller=ToolCaller.service()
    )
    by_agent = {agent_id: kwargs["runtime_kb_configs"] for agent_id, kwargs in calls}
    assert by_agent[_A] == [{"id": "kb-a-only"}]
    assert by_agent[_B] is None


def test_an_entry_naming_both_agents_reaches_both(monkeypatch):
    calls = _scoped_setup(monkeypatch)
    runtime = [{"id": "kb-both", "agent_ids": [_A, _B]}]
    orch_service.build_orchestration(
        "orch-1", runtime_kb_configs=runtime, caller=ToolCaller.service()
    )
    for _agent_id, kwargs in calls:
        assert kwargs["runtime_kb_configs"] == [{"id": "kb-both"}]


def _run_stream(body, **patches):
    from unittest.mock import patch

    from flask import Flask

    from agentic_project_service.routes import orchestrations as orchestrations_route

    app = Flask(__name__)
    app.testing = True  # let the sentinel below propagate
    app.register_blueprint(orchestrations_route.orchestrations_bp)
    fake_session = MagicMock()
    fake_session.execute.return_value = [(_KB,)]
    with (
        patch.object(orchestrations_route.db, "session", fake_session),
        patch.object(orchestrations_route.billing, "check_balance") as check_balance,
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "service", "role": "service_role", "is_service_role": True},
        ),
        patch.object(orchestrations_route, "_orchestration_agent_ids", **patches) as agent_ids,
        app.test_client() as c,
    ):
        check_balance.side_effect = RuntimeError("stop after validation")
        try:
            resp = c.post(
                f"/api/orchestrations/{_ORCH}/run/stream",
                json=body,
                headers={"Authorization": "Bearer fake"},
            )
        except RuntimeError:
            resp = None  # validation passed and the handler went on
    return resp, agent_ids


def test_the_route_rejects_an_agent_outside_the_orchestration():
    body = {"message": "hi", "runtime_knowledge_bases": [{"id": _KB, "agent_ids": [_A]}]}
    resp, agent_ids = _run_stream(body, return_value={_B})
    assert resp.status_code == 400
    assert resp.get_json() == {"error": f"agent_ids not in this orchestration: {_A}"}
    agent_ids.assert_called_once_with(_ORCH)


def test_the_route_accepts_an_agent_of_the_orchestration():
    body = {"message": "hi", "runtime_knowledge_bases": [{"id": _KB, "agent_ids": [_A]}]}
    resp, _ = _run_stream(body, return_value={_A, _B})
    assert resp is None


def test_the_route_reads_the_entities_only_for_scoped_entries():
    body = {"message": "hi", "runtime_knowledge_bases": [{"id": _KB}]}
    resp, agent_ids = _run_stream(body, return_value={_A})
    assert resp is None
    agent_ids.assert_not_called()


def test_orchestration_agent_ids_of_a_malformed_id_is_empty():
    from agentic_project_service.routes import orchestrations as orchestrations_route

    assert orchestrations_route._orchestration_agent_ids("not-a-uuid") == set()
