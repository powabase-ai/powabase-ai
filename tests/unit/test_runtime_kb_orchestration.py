"""Runtime KBs flow to every orchestration sub-agent's tool build."""

from unittest.mock import MagicMock, patch

from agentic_project_service.routes import orchestrations as orch_route
from agentic_project_service.services import orchestration as orch_service


def test_build_orchestration_accepts_and_threads_runtime_kbs():
    """Signature takes runtime_kb_configs and passes it to every
    load_all_tools_for_agent call."""
    import inspect

    sig = inspect.signature(orch_service.build_orchestration)
    assert "runtime_kb_configs" in sig.parameters

    src = inspect.getsource(orch_service.build_orchestration)
    assert "runtime_kb_configs=runtime_kb_configs" in src


def test_stream_route_validates_and_passes_runtime_kbs():
    """The route validates the field and hands the configs to
    build_orchestration."""
    import inspect

    src = inspect.getsource(orch_route.run_orchestration_stream)
    assert "validate_runtime_knowledge_bases" in src
    assert "runtime_kb_configs" in src
