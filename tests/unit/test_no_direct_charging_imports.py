import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "agentic_project_service"

# The billing implementation — allowed to import charging internals.
# billing_cloud/ holds the adapter + the 4 charging modules + identity.py (the
# CUT symbols split out of the old billing_context.py) — all relocated there by
# Task 14, so the directory exemption below is the only exemption they need.
# main.py's single guarded `from .services.billing_cloud import install_billing`
# (inside try/except ImportError, see the call site's own comment) is the
# sanctioned composition-root seam: the one place that must import the real
# adapter to register it behind billing_port. It is not a Phase-2 call-site
# (no charging logic lives in main.py) and there is no port-based alternative —
# something has to wire the adapter in. Task 15's ruff TID251 rule +
# test_billing_isolation.py formalize this precisely; interim-exempted here at
# file granularity, matching this test's existing coarseness.
_COMPOSITION_ROOT_FILES = {"main.py"}

BANNED = re.compile(
    r"from\s+\S*(billing_cloud|credits_client|balance_cache|billing_litellm|jwt_minter)\S*\s+import"
    r"|import\s+\S*(billing_cloud|credits_client|balance_cache|billing_litellm|jwt_minter)\b"
)


def test_no_core_module_imports_a_charging_internal():
    offenders = []
    for p in SRC.rglob("*.py"):
        if "billing_cloud" in p.parts:  # the adapter + charging-impl modules — exempt
            continue
        if p.name in _COMPOSITION_ROOT_FILES:  # the composition-root seam — exempt
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if BANNED.search(line):
                offenders.append(f"{p.relative_to(SRC)}:{i}: {line.strip()}")
    assert not offenders, (
        "charging-internal imports leaked into core (should go through billing_port):\n"
        + "\n".join(offenders)
    )
