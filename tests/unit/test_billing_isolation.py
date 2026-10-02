"""powabase-ai core must not import the private billing_cloud package (excluded
from the OSS build). Only the composition root (main.py, guarded) and
billing_cloud itself may. Mirrors agentic/tests/unit/test_import_isolation.py."""

import subprocess
import sys
from pathlib import Path

PS_ROOT = Path(__file__).resolve().parents[2]
PS_SRC = PS_ROOT / "src"

# Ruff from the interpreter running the suite, not `uv run ruff`: that
# re-syncs the venv to uv.lock mid-suite, silently undoing a dev override (an
# editable sibling checkout, a trial dependency) for every later test.
RUFF = [sys.executable, "-m", "ruff"]


def test_no_billing_cloud_imports_in_core():
    result = subprocess.run(
        [*RUFF, "check", "--select=TID251", str(PS_SRC)],
        capture_output=True,
        text=True,
        cwd=PS_ROOT,
    )
    assert result.returncode == 0, f"import violations:\n{result.stdout}\n{result.stderr}"


def test_lint_catches_simulated_core_violation():
    planted = PS_SRC / "agentic_project_service" / "services" / "_iso_check.py"
    # RELATIVE import — the codebase's real style; TID251 resolves + flags it.
    planted.write_text("from .billing_cloud import adapter  # relative — the codebase's style\n")
    try:
        result = subprocess.run(
            [*RUFF, "check", "--select=TID251", str(planted)],
            capture_output=True,
            text=True,
            cwd=PS_ROOT,
        )
        assert result.returncode != 0, f"expected TID251 violation:\n{result.stdout}"
        assert "billing_cloud" in result.stdout
    finally:
        planted.unlink()
