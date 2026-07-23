"""Tests for bootstrap — create_app() wires billing via the guarded install_billing seam.

Note: the counterpart test that asserted a private CloudBillingAdapter gets
installed by default has been removed — that only happens when the excluded
services.billing_cloud package is present, which it never is in this OSS
build. What remains here verifies the real OSS condition: create_app() must
boot cleanly with the no-op adapter when billing_cloud is absent.
"""

from unittest.mock import patch

from agentic_project_service.main import create_app
from agentic_project_service.services import billing_port


def test_create_app_survives_absent_billing_cloud(monkeypatch):
    # Simulate the OSS build: billing_cloud import fails -> no-op adapter stands,
    # app still boots.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.endswith("billing_cloud") or ".billing_cloud" in name:
            raise ImportError("simulated OSS build: no billing_cloud")
        return real_import(name, *a, **k)

    billing_port.set_billing_adapter(billing_port.NoopBillingAdapter())
    with patch.object(builtins, "__import__", side_effect=fake_import):
        app = create_app(testing=True)
    assert isinstance(billing_port.get_billing_adapter(), billing_port.NoopBillingAdapter)
    assert app is not None
