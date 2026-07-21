"""Tests for the encryption module's startup behaviour.

The fail-hard semantics (issue #246): if ``API_KEY_ENCRYPTION_KEY`` is
unset, the module refuses to fall back to a process-local Fernet key
unless the operator explicitly opts in via ``ALLOW_TEMP_ENCRYPTION_KEY=1``.
This prevents the failure mode that produced the cookbook-01 incident,
where rows encrypted with a temp key became permanently unreadable after
restart.
"""

import importlib

import pytest


def _reload_encryption():
    """Reset the cached _fernet between tests by reloading the module."""
    import agentic_project_service.services.encryption as mod

    importlib.reload(mod)
    return mod


class TestGetFernetWithEnvVar:
    def test_returns_fernet_when_env_var_set(self, monkeypatch):
        from cryptography.fernet import Fernet

        monkeypatch.setenv("API_KEY_ENCRYPTION_KEY", Fernet.generate_key().decode())
        monkeypatch.delenv("ALLOW_TEMP_ENCRYPTION_KEY", raising=False)

        mod = _reload_encryption()
        plaintext = "sk-test-with-env-var"
        ciphertext = mod.encrypt_api_key(plaintext)
        assert mod.decrypt_api_key(ciphertext) == plaintext


class TestGetFernetWithoutEnvVar:
    def test_raises_runtime_error_by_default(self, monkeypatch):
        """Missing API_KEY_ENCRYPTION_KEY → fail hard.

        Rationale: silently falling back to a process-local Fernet key
        creates rows that become permanently undecryptable on the next
        pod restart. We saw this fail in prod — see issue #246.
        """
        monkeypatch.delenv("API_KEY_ENCRYPTION_KEY", raising=False)
        monkeypatch.delenv("ALLOW_TEMP_ENCRYPTION_KEY", raising=False)

        mod = _reload_encryption()

        with pytest.raises(RuntimeError) as exc_info:
            mod.encrypt_api_key("sk-anything")

        msg = str(exc_info.value)
        assert "API_KEY_ENCRYPTION_KEY" in msg
        assert "ALLOW_TEMP_ENCRYPTION_KEY" in msg or "make gen-keys" in msg

    def test_temp_key_allowed_with_explicit_opt_in(self, monkeypatch):
        """ALLOW_TEMP_ENCRYPTION_KEY=1 unlocks the local-dev temp-key path."""
        monkeypatch.delenv("API_KEY_ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ALLOW_TEMP_ENCRYPTION_KEY", "1")

        mod = _reload_encryption()

        plaintext = "sk-temp-key-roundtrip"
        ciphertext = mod.encrypt_api_key(plaintext)
        assert mod.decrypt_api_key(ciphertext) == plaintext

    def test_temp_key_opt_in_requires_truthy_value(self, monkeypatch):
        """ALLOW_TEMP_ENCRYPTION_KEY=0 / empty / 'false' must NOT bypass."""
        monkeypatch.delenv("API_KEY_ENCRYPTION_KEY", raising=False)

        for falsy in ("0", "", "false", "no"):
            monkeypatch.setenv("ALLOW_TEMP_ENCRYPTION_KEY", falsy)
            mod = _reload_encryption()
            with pytest.raises(RuntimeError):
                mod.encrypt_api_key("sk-anything")


class TestMalformedKey:
    """Critical regression guard (issue #246, review round 3).

    ``Fernet(key)`` raises ``ValueError`` (subclass: ``binascii.Error``) on a
    malformed key — wrong length, non-base64, truncated, etc. Without an
    explicit wrap, that ``ValueError`` propagates up through
    ``encrypt_api_key`` / ``decrypt_api_key`` and the resolver's narrow
    ``except ValueError`` catches it as a per-row decrypt failure — every
    row goes into ``bad_rows`` and gets DELETE'd. That's the exact
    "wipe the table on misconfigured pod" failure mode the narrowing was
    meant to prevent.

    The module MUST wrap the Fernet ctor and re-raise as RuntimeError so the
    resolver's narrow catch lets it propagate.
    """

    @pytest.mark.parametrize(
        "bad_key",
        [
            "not-a-real-key",  # not base64
            "a" * 20,  # too short
            "a" * 100,  # too long
            "aaaa===",  # invalid base64 padding
        ],
    )
    def test_malformed_key_raises_runtime_error_not_value_error(self, monkeypatch, bad_key):
        monkeypatch.setenv("API_KEY_ENCRYPTION_KEY", bad_key)
        monkeypatch.delenv("ALLOW_TEMP_ENCRYPTION_KEY", raising=False)

        mod = _reload_encryption()

        with pytest.raises(RuntimeError) as exc_info:
            mod.encrypt_api_key("sk-anything")

        msg = str(exc_info.value)
        assert "API_KEY_ENCRYPTION_KEY" in msg
        assert "malformed" in msg.lower() or "32" in msg or "base64" in msg.lower()
        # Crucially NOT a bare ValueError (the resolver's narrow catch would
        # treat that as a per-row decrypt failure and wipe the table).
        assert not isinstance(exc_info.value, ValueError)
