"""Tests for the ai_provider_keys_resolver service module."""

from unittest.mock import MagicMock, patch

import pytest

from agentic_project_service.services.ai_provider_keys_resolver import (
    ProviderKeyDecryptDropped,
    get_all_user_provider_keys,
    get_user_provider_keys_with_dropped,
    resolve_api_key_for_model,
    resolve_api_key_or_raise_for_drop,
    resolve_api_key_or_raise_for_drop_using,
)


# ---------------------------------------------------------------------------
# get_all_user_provider_keys
# ---------------------------------------------------------------------------


class TestGetAllUserProviderKeys:
    def test_db_only(self, app):
        """DB rows returned."""
        with app.app_context():
            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver.AIProviderKey"
            ) as MockModel:
                mock_row = MagicMock()
                mock_row.provider = "openai"
                mock_row.api_key_encrypted = "encrypted-value"
                MockModel.query.filter_by.return_value.all.return_value = [mock_row]

                with patch(
                    "agentic_project_service.services.ai_provider_keys_resolver.decrypt_api_key",
                    return_value="sk-from-db",
                ):
                    result = get_all_user_provider_keys()

        assert result == {"openai": "sk-from-db"}

    def test_empty_db_returns_empty(self, app):
        """Empty DB → empty dict."""
        with app.app_context():
            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver.AIProviderKey"
            ) as MockModel:
                MockModel.query.filter_by.return_value.all.return_value = []

                result = get_all_user_provider_keys()

        assert result == {}

    def test_decrypt_failure_logs_skips_and_deletes_via_session(self, app):
        """Decrypt error is logged, row is deleted via conditional query, dict is empty."""
        with app.app_context():
            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver.AIProviderKey"
            ) as MockModel:
                mock_row = MagicMock()
                mock_row.provider = "openai"
                mock_row.api_key_encrypted = "bad-enc"
                MockModel.query.filter_by.return_value.all.return_value = [mock_row]

                with patch(
                    "agentic_project_service.services.ai_provider_keys_resolver.decrypt_api_key",
                    side_effect=ValueError("bad key"),
                ):
                    with patch(
                        "agentic_project_service.services.ai_provider_keys_resolver.db"
                    ) as mock_db:
                        # SAVEPOINT context manager succeeds
                        mock_db.session.begin_nested.return_value.__enter__.return_value = None
                        mock_db.session.begin_nested.return_value.__exit__.return_value = None
                        # Conditional delete returns 1 → row was actually deleted
                        mock_db.session.query.return_value.filter_by.return_value.delete.return_value = 1

                        result = get_all_user_provider_keys()

                        # Conditional delete keyed on (id, ciphertext) so a
                        # concurrently-re-added row would be skipped.
                        mock_db.session.query.return_value.filter_by.assert_called_once_with(
                            id=mock_row.id, api_key_encrypted="bad-enc"
                        )
                        # SAVEPOINT used to isolate delete from caller's txn
                        mock_db.session.begin_nested.assert_called_once()

        assert result == {}


# ---------------------------------------------------------------------------
# get_user_provider_keys_with_dropped
# ---------------------------------------------------------------------------


class TestGetUserProviderKeysWithDropped:
    def test_no_rows_returns_empty_keys_and_dropped(self, app):
        """No rows → ({}, [])."""
        with app.app_context():
            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver.AIProviderKey"
            ) as MockModel:
                MockModel.query.filter_by.return_value.all.return_value = []

                keys, dropped = get_user_provider_keys_with_dropped()

        assert keys == {}
        assert dropped == []

    def test_returns_decryptable_keys_and_dropped_providers(self, app):
        """Decryptable keys go in keys; failed providers go in dropped."""
        with app.app_context():
            good = MagicMock()
            good.provider = "openai"
            good.api_key_encrypted = "good-enc"
            bad = MagicMock()
            bad.provider = "anthropic"
            bad.api_key_encrypted = "bad-enc"

            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver.AIProviderKey"
            ) as MockModel:
                MockModel.query.filter_by.return_value.all.return_value = [good, bad]

                def _decrypt(blob):
                    if blob == "good-enc":
                        return "sk-good-67890"
                    raise ValueError("bad key")

                with patch(
                    "agentic_project_service.services.ai_provider_keys_resolver.decrypt_api_key",
                    side_effect=_decrypt,
                ):
                    with patch(
                        "agentic_project_service.services.ai_provider_keys_resolver.db"
                    ) as mock_db:
                        mock_db.session.begin_nested.return_value.__enter__.return_value = None
                        mock_db.session.begin_nested.return_value.__exit__.return_value = None
                        # Conditional delete returns 1 row affected
                        mock_db.session.query.return_value.filter_by.return_value.delete.return_value = 1

                        keys, dropped = get_user_provider_keys_with_dropped()

                        # Only the bad row's id+ciphertext is filtered for delete
                        mock_db.session.query.return_value.filter_by.assert_called_once_with(
                            id=bad.id, api_key_encrypted="bad-enc"
                        )

        assert keys == {"openai": "sk-good-67890"}
        assert set(dropped) == {"anthropic"}

    def test_delete_failure_still_surfaces_dropped_provider(self, app):
        """CRITICAL regression guard (issue #246, review round 4).

        If the conditional DELETE itself fails (DB blip, lock conflict, etc.)
        the resolver MUST still report the provider as dropped so the route
        emits the actionable ``ProviderKeyDecryptDropped`` error to the user.

        The previous behavior — only adding to ``dropped`` when DELETE
        succeeded — silently restored issue #246's original failure mode:
        decryption failed → row stays in DB AND ``dropped=[]`` → caller
        returns ``api_key=None`` → litellm emits the misleading "Missing
        API Key" error to the user.

        Truth source for ``dropped`` is **decryptability**, not deletion
        success. The DELETE is best-effort cleanup, not a precondition for
        surfacing the error.
        """
        with app.app_context():
            bad = MagicMock()
            bad.provider = "anthropic"
            bad.api_key_encrypted = "bad-enc"

            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver.AIProviderKey"
            ) as MockModel:
                MockModel.query.filter_by.return_value.all.return_value = [bad]

                with patch(
                    "agentic_project_service.services.ai_provider_keys_resolver.decrypt_api_key",
                    side_effect=ValueError("bad key"),
                ):
                    with patch(
                        "agentic_project_service.services.ai_provider_keys_resolver.db"
                    ) as mock_db:
                        # Simulate the conditional-delete path failing.
                        mock_db.session.begin_nested.return_value.__enter__.return_value = None
                        mock_db.session.begin_nested.return_value.__exit__.return_value = None
                        mock_db.session.query.return_value.filter_by.return_value.delete.side_effect = RuntimeError(
                            "db down"
                        )

                        keys, dropped = get_user_provider_keys_with_dropped()

        assert keys == {}
        # CRITICAL: dropped MUST contain the provider even though the delete
        # failed. Otherwise the resolver silently restores issue #246.
        assert dropped == ["anthropic"]

    def test_concurrent_readd_removes_from_dropped(self, app):
        """Concurrent re-add (rowcount=0 from conditional DELETE) means the
        row's ciphertext changed between read and delete — the row is now
        valid with the current encryption key. We must NOT surface the
        dropped error in this case (would be a false positive).
        """
        with app.app_context():
            bad = MagicMock()
            bad.provider = "anthropic"
            bad.api_key_encrypted = "OLD-CIPHERTEXT"

            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver.AIProviderKey"
            ) as MockModel:
                MockModel.query.filter_by.return_value.all.return_value = [bad]

                with patch(
                    "agentic_project_service.services.ai_provider_keys_resolver.decrypt_api_key",
                    side_effect=ValueError("bad key"),
                ):
                    with patch(
                        "agentic_project_service.services.ai_provider_keys_resolver.db"
                    ) as mock_db:
                        mock_db.session.begin_nested.return_value.__enter__.return_value = None
                        mock_db.session.begin_nested.return_value.__exit__.return_value = None
                        # rowcount=0 → row was updated under us; ciphertext
                        # changed; row is now valid → no error.
                        mock_db.session.query.return_value.filter_by.return_value.delete.return_value = 0

                        keys, dropped = get_user_provider_keys_with_dropped()

        assert keys == {}
        assert dropped == []

    def test_runtime_error_from_decrypt_is_not_treated_as_decrypt_failure(self, app):
        """If decrypt_api_key raises a non-ValueError (e.g., RuntimeError from missing
        API_KEY_ENCRYPTION_KEY), the resolver MUST propagate it instead of treating
        every row as undecryptable and wiping the table.

        Regression: under the previous broad ``except Exception``, a misconfigured
        pod (env var unset → ``_get_fernet`` raises RuntimeError on the first
        decrypt call) would have caused get_user_provider_keys_with_dropped to
        delete every row in the database. Catch must be narrow.
        """
        with app.app_context():
            row1 = MagicMock(provider="openai", api_key_encrypted="e1")
            row2 = MagicMock(provider="anthropic", api_key_encrypted="e2")
            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver.AIProviderKey"
            ) as MockModel:
                MockModel.query.filter_by.return_value.all.return_value = [row1, row2]

                with patch(
                    "agentic_project_service.services.ai_provider_keys_resolver.decrypt_api_key",
                    side_effect=RuntimeError("API_KEY_ENCRYPTION_KEY is not set"),
                ):
                    with patch(
                        "agentic_project_service.services.ai_provider_keys_resolver.db"
                    ) as mock_db:
                        with pytest.raises(RuntimeError):
                            get_user_provider_keys_with_dropped()

                        # CRITICAL: must NOT delete any rows when decrypt failure is
                        # not a per-row InvalidToken — the rows are still valid; the
                        # operator's environment is broken.
                        mock_db.session.delete.assert_not_called()
                        mock_db.session.query.return_value.filter_by.return_value.delete.assert_not_called()

    def test_malformed_encryption_key_does_not_wipe_table(self, app, monkeypatch):
        """Critical regression guard (issue #246, review round 3).

        ``Fernet(malformed_key)`` natively raises ``ValueError``. Without an
        explicit wrap in ``encryption.py``, that ValueError would propagate
        through ``decrypt_api_key`` and the resolver's narrow ``except
        ValueError`` would catch it as a per-row failure — every row goes
        into ``bad_rows`` and gets DELETE'd. This test exercises the REAL
        encryption module (no decrypt mock) with a malformed env-var key,
        and asserts NO rows are deleted.
        """
        import agentic_project_service.services.encryption as enc_mod
        import importlib

        # Reset cached Fernet so monkeypatched env var takes effect
        monkeypatch.setattr(enc_mod, "_fernet", None)
        monkeypatch.setenv("API_KEY_ENCRYPTION_KEY", "this-is-not-a-valid-fernet-key")
        monkeypatch.delenv("ALLOW_TEMP_ENCRYPTION_KEY", raising=False)
        importlib.reload(enc_mod)

        with app.app_context():
            row = MagicMock(provider="anthropic", api_key_encrypted="ciphertext")
            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver.AIProviderKey"
            ) as MockModel:
                MockModel.query.filter_by.return_value.all.return_value = [row]

                with patch(
                    "agentic_project_service.services.ai_provider_keys_resolver.db"
                ) as mock_db:
                    # Wire the real (non-mocked) decrypt_api_key into the resolver
                    # module so it actually calls _get_fernet → ValueError from
                    # the malformed key.
                    with patch(
                        "agentic_project_service.services.ai_provider_keys_resolver.decrypt_api_key",
                        wraps=enc_mod.decrypt_api_key,
                    ):
                        with pytest.raises(RuntimeError):
                            get_user_provider_keys_with_dropped()

                    mock_db.session.delete.assert_not_called()
                    mock_db.session.query.return_value.filter_by.return_value.delete.assert_not_called()

        # Restore the cached Fernet for other tests
        monkeypatch.setattr(enc_mod, "_fernet", None)
        importlib.reload(enc_mod)

    def test_self_heal_does_not_delete_freshly_upserted_row(self, app):
        """Race: user re-adds the key (UPDATE on existing row, same id) between
        the resolver's read-loop and delete-loop. The cached row instance still
        resolves to the same id, so a naive ``db.session.delete(row)`` would
        delete the freshly-good row. The conditional delete keyed on
        ``api_key_encrypted`` skips it AND removes the provider from
        ``dropped`` so the user does not see a false-positive error.
        """
        with app.app_context():
            row = MagicMock(provider="anthropic", api_key_encrypted="OLD-ENC")
            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver.AIProviderKey"
            ) as MockModel:
                MockModel.query.filter_by.return_value.all.return_value = [row]

                with patch(
                    "agentic_project_service.services.ai_provider_keys_resolver.decrypt_api_key",
                    side_effect=ValueError("bad key"),
                ):
                    with patch(
                        "agentic_project_service.services.ai_provider_keys_resolver.db"
                    ) as mock_db:
                        mock_db.session.begin_nested.return_value.__enter__.return_value = None
                        mock_db.session.begin_nested.return_value.__exit__.return_value = None
                        # rowcount=0 means the row's api_key_encrypted changed
                        # between read and delete (concurrent re-add).
                        mock_db.session.query.return_value.filter_by.return_value.delete.return_value = 0

                        keys, dropped = get_user_provider_keys_with_dropped()

                        # Conditional delete keyed on (id, ciphertext)
                        mock_db.session.query.return_value.filter_by.assert_called_once_with(
                            id=row.id, api_key_encrypted="OLD-ENC"
                        )

        assert keys == {}
        # Provider was NOT recorded as dropped — the row was changed under us
        assert dropped == []


# ---------------------------------------------------------------------------
# resolve_api_key_for_model
# ---------------------------------------------------------------------------


class TestResolveApiKeyForModel:
    def test_key_found_in_provider_keys(self):
        """Provider key in the dict → returns it."""
        result = resolve_api_key_for_model("gpt-4", {"openai": "sk-x"})
        assert result == "sk-x"

    def test_falls_back_to_env_var(self, monkeypatch):
        """Empty provider_keys + OPENAI_API_KEY set → returns env value."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-value")
        result = resolve_api_key_for_model("gpt-4", {})
        assert result == "sk-env-value"

    def test_unknown_model_returns_none(self):
        """Model that litellm cannot resolve → returns None."""
        result = resolve_api_key_for_model("totally-unknown-model-xyz", {})
        assert result is None

    def test_provider_key_takes_precedence_over_env(self, monkeypatch):
        """provider_keys dict takes precedence over env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-fallback")
        result = resolve_api_key_for_model("gpt-4", {"openai": "sk-from-dict"})
        assert result == "sk-from-dict"

    def test_no_key_anywhere_returns_none(self, monkeypatch):
        """No provider_keys entry and no env var → returns None."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = resolve_api_key_for_model("gpt-4", {})
        assert result is None


# ---------------------------------------------------------------------------
# resolve_api_key_or_raise_for_drop
# ---------------------------------------------------------------------------


class TestResolveApiKeyOrRaiseForDrop:
    def test_returns_key_when_resolvable(self, app, monkeypatch):
        """Happy path: key resolves cleanly, no exception."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with app.app_context():
            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver"
                ".get_user_provider_keys_with_dropped",
                return_value=({"anthropic": "sk-ant-good"}, []),
            ):
                result = resolve_api_key_or_raise_for_drop("claude-opus-4-7")
        assert result == "sk-ant-good"

    def test_raises_when_models_provider_was_dropped(self, app, monkeypatch):
        """If the model's provider was just dropped due to decrypt failure, raise."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with app.app_context():
            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver"
                ".get_user_provider_keys_with_dropped",
                return_value=({}, ["anthropic"]),
            ):
                with pytest.raises(ProviderKeyDecryptDropped) as exc_info:
                    resolve_api_key_or_raise_for_drop("claude-opus-4-7")

        assert exc_info.value.provider == "anthropic"
        msg = str(exc_info.value)
        assert "anthropic" in msg.lower()
        assert "re-add" in msg.lower() or "re add" in msg.lower()

    def test_does_not_raise_when_other_provider_was_dropped(self, app, monkeypatch):
        """Anthropic dropped, but model is openai → no raise (openai unaffected)."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        with app.app_context():
            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver"
                ".get_user_provider_keys_with_dropped",
                return_value=({}, ["anthropic"]),
            ):
                result = resolve_api_key_or_raise_for_drop("gpt-4")
        # Resolved via env-var fallback for openai; anthropic drop is irrelevant
        assert result == "sk-env"

    def test_returns_none_when_no_key_and_no_drop(self, app, monkeypatch):
        """Provider has no key configured anywhere — returns None (not raise)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with app.app_context():
            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver"
                ".get_user_provider_keys_with_dropped",
                return_value=({}, []),
            ):
                result = resolve_api_key_or_raise_for_drop("claude-opus-4-7")
        assert result is None


# ---------------------------------------------------------------------------
# resolve_api_key_or_raise_for_drop_using (pre-computed variant for orchestration)
# ---------------------------------------------------------------------------


class TestResolveApiKeyOrRaiseForDropUsing:
    """The ``_using`` variant accepts pre-computed (provider_keys, dropped) so
    callers resolving N models in one operation (orchestration) only query the
    DB and run the self-heal loop once.
    """

    def test_returns_key_from_provided_dict(self):
        result = resolve_api_key_or_raise_for_drop_using(
            "claude-opus-4-7",
            {"anthropic": "sk-ant-pre-computed"},
            [],
        )
        assert result == "sk-ant-pre-computed"

    def test_raises_when_models_provider_in_dropped(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ProviderKeyDecryptDropped) as exc_info:
            resolve_api_key_or_raise_for_drop_using("claude-opus-4-7", {}, ["anthropic"])
        assert exc_info.value.provider == "anthropic"

    def test_does_not_raise_when_other_provider_dropped(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        result = resolve_api_key_or_raise_for_drop_using("gpt-4", {}, ["anthropic"])
        assert result == "sk-env"

    def test_does_not_query_db(self, app):
        """The ``_using`` variant must not call get_user_provider_keys_with_dropped."""
        with app.app_context():
            with patch(
                "agentic_project_service.services.ai_provider_keys_resolver"
                ".get_user_provider_keys_with_dropped"
            ) as mock_get:
                resolve_api_key_or_raise_for_drop_using(
                    "claude-opus-4-7",
                    {"anthropic": "sk-ant-pre"},
                    [],
                )
                mock_get.assert_not_called()
