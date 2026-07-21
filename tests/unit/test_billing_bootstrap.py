"""Tests for bootstrap — create_app() wires billing via the guarded install_billing seam."""

from unittest.mock import patch

from agentic_project_service.main import create_app
from agentic_project_service.services import billing_port
from agentic_project_service.services.billing_cloud import CloudBillingAdapter


def test_create_app_installs_cloud_billing_adapter():
    app = create_app(testing=True)
    assert isinstance(billing_port.get_billing_adapter(), CloudBillingAdapter)
    # The billing before_request hook is registered (BYOK context).
    hook_names = [f.__name__ for fs in app.before_request_funcs.values() for f in fs]
    assert "_set_billing_byok" in hook_names


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
