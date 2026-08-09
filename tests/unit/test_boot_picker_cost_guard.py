"""Boot-time guard (main.py) — the pinned Project Copilot model must be
checked against litellm's cost map alongside the workflow-copilot picker.

The guard exists to catch a model whose litellm cost data is missing or
stale before it silently free-rides in production (see main.py's own
comment: "AI-on-us billing will silently drop charges for this model").
PROJECT_COPILOT_MODEL is not one of the COPILOT_MODEL_OPTIONS entries the
guard already iterates, so a guard that only walks the picker list would
never catch a regression on the one model the Project Copilot always calls.

This test observes the actual set of model ids create_app() queries
litellm for at boot — not a separately-maintained list — so it fails if the
pinned model is ever dropped from what the guard checks.
"""

from unittest.mock import patch

import litellm

from agentic_project_service.main import create_app
from agentic_project_service.services.project_copilot import PROJECT_COPILOT_MODEL


def test_boot_guard_checks_the_pinned_project_copilot_model():
    real_get_model_info = litellm.get_model_info

    with patch.object(litellm, "get_model_info", wraps=real_get_model_info) as spy:
        create_app(testing=True)

    checked_models = [call.args[0] for call in spy.call_args_list]
    assert PROJECT_COPILOT_MODEL in checked_models, (
        "the boot-time cost-map guard did not query litellm for "
        f"PROJECT_COPILOT_MODEL ({PROJECT_COPILOT_MODEL}) — a missing/stale "
        "cost entry for it would go undetected until it silently free-rides"
    )
