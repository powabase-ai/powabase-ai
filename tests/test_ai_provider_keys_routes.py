"""Tests for /api/ai-provider-keys — CRUD, batch PUT, and validate endpoint."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = "/api/ai-provider-keys"


def _post_key(client, auth_headers, provider: str, api_key: str):
    return client.post(BASE, json={"provider": provider, "api_key": api_key}, headers=auth_headers)


# ---------------------------------------------------------------------------
# GET — list
# ---------------------------------------------------------------------------


class TestListKeys:
    def test_empty_list_returns_empty_array(self, client, mock_auth, auth_headers):
        resp = client.get(BASE, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json() == []


# ---------------------------------------------------------------------------
# POST — single upsert
# ---------------------------------------------------------------------------


class TestPostKey:
    def test_post_new_key_returns_201(self, client, mock_auth, auth_headers, mocker):
        mocker.patch(
            "agentic_project_service.routes.ai_provider_keys.validate_provider_key",
            return_value=_ok_result(),
        )
        resp = _post_key(client, auth_headers, "openai", "sk-valid")
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["provider"] == "openai"
        assert data["is_valid"] is True
        assert "masked_key" in data

    def test_post_replace_key_returns_200(self, client, mock_auth, auth_headers, mocker):
        mocker.patch(
            "agentic_project_service.routes.ai_provider_keys.validate_provider_key",
            return_value=_ok_result(),
        )
        _post_key(client, auth_headers, "openai", "sk-first")
        resp = _post_key(client, auth_headers, "openai", "sk-second")
        assert resp.status_code == 200
        # Only one row should exist
        list_resp = client.get(BASE, headers=auth_headers)
        assert len(list_resp.get_json()) == 1

    def test_post_disallowed_provider_returns_400(self, client, mock_auth, auth_headers):
        resp = _post_key(client, auth_headers, "mistral", "some-key")
        assert resp.status_code == 400
        assert "Unknown provider" in resp.get_json()["error"]

    def test_post_provider_401_returns_400_with_field_error(
        self, client, mock_auth, auth_headers, mocker
    ):
        from agentic_project_service.services.provider_validator import ValidationResult

        mocker.patch(
            "agentic_project_service.routes.ai_provider_keys.validate_provider_key",
            return_value=ValidationResult(ok=False, provider_status=401, error="Unauthorized"),
        )
        resp = _post_key(client, auth_headers, "openai", "sk-bad")
        assert resp.status_code == 400
        body = resp.get_json()
        assert "fields" in body
        assert "openai" in body["fields"]

    def test_post_provider_503_stores_row_as_invalid(self, client, mock_auth, auth_headers, mocker):
        from agentic_project_service.services.provider_validator import ValidationResult

        mocker.patch(
            "agentic_project_service.routes.ai_provider_keys.validate_provider_key",
            return_value=ValidationResult(
                ok=False, provider_status=503, error="Service unavailable"
            ),
        )
        resp = _post_key(client, auth_headers, "openai", "sk-flaky")
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["is_valid"] is False
        assert data["last_validated_at"] is None

    def test_post_missing_fields_returns_400(self, client, mock_auth, auth_headers):
        resp = client.post(BASE, json={"provider": "openai"}, headers=auth_headers)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# PUT — batch upsert
# ---------------------------------------------------------------------------


class TestBatchPutKeys:
    def test_put_batch_upserts_multiple_atomically(self, client, mock_auth, auth_headers, mocker):
        mocker.patch(
            "agentic_project_service.routes.ai_provider_keys.validate_provider_key",
            return_value=_ok_result(),
        )
        resp = client.put(
            BASE,
            json={"openai": "sk-oai", "anthropic": "sk-ant"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        providers = {r["provider"] for r in resp.get_json()}
        assert providers == {"openai", "anthropic"}

    def test_put_with_one_bad_provider_rolls_back_all(
        self, client, mock_auth, auth_headers, mocker
    ):
        from agentic_project_service.services.provider_validator import ValidationResult

        call_count = {"n": 0}

        def _side_effect(provider, api_key):
            call_count["n"] += 1
            if provider == "anthropic":
                return ValidationResult(ok=False, provider_status=401, error="Bad key")
            return _ok_result()

        mocker.patch(
            "agentic_project_service.routes.ai_provider_keys.validate_provider_key",
            side_effect=_side_effect,
        )
        resp = client.put(
            BASE,
            json={"openai": "sk-oai", "anthropic": "sk-bad"},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "fields" in body
        assert "anthropic" in body["fields"]

        # No rows should have been committed
        list_resp = client.get(BASE, headers=auth_headers)
        assert list_resp.get_json() == []

    def test_put_does_not_clear_unspecified_providers(
        self, client, mock_auth, auth_headers, mocker
    ):
        mocker.patch(
            "agentic_project_service.routes.ai_provider_keys.validate_provider_key",
            return_value=_ok_result(),
        )
        # Seed openai + anthropic
        client.put(
            BASE,
            json={"openai": "sk-oai", "anthropic": "sk-ant"},
            headers=auth_headers,
        )
        # PUT only google — openai + anthropic must persist
        resp = client.put(BASE, json={"google": "google-key"}, headers=auth_headers)
        assert resp.status_code == 200

        list_resp = client.get(BASE, headers=auth_headers)
        providers = {r["provider"] for r in list_resp.get_json()}
        assert providers == {"openai", "anthropic", "google"}

    def test_put_null_values_are_noop(self, client, mock_auth, auth_headers, mocker):
        mocker.patch(
            "agentic_project_service.routes.ai_provider_keys.validate_provider_key",
            return_value=_ok_result(),
        )
        # Seed openai
        _post_key(client, auth_headers, "openai", "sk-oai")
        # PUT with null anthropic — openai must still be there, anthropic must not appear
        resp = client.put(BASE, json={"anthropic": None}, headers=auth_headers)
        assert resp.status_code == 200
        list_resp = client.get(BASE, headers=auth_headers)
        providers = {r["provider"] for r in list_resp.get_json()}
        assert "openai" in providers
        assert "anthropic" not in providers


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------


class TestDeleteKey:
    def test_delete_removes_row_and_returns_204(self, client, mock_auth, auth_headers, mocker):
        mocker.patch(
            "agentic_project_service.routes.ai_provider_keys.validate_provider_key",
            return_value=_ok_result(),
        )
        _post_key(client, auth_headers, "openai", "sk-oai")
        resp = client.delete(f"{BASE}/openai", headers=auth_headers)
        assert resp.status_code == 204

        list_resp = client.get(BASE, headers=auth_headers)
        assert list_resp.get_json() == []

    def test_delete_nonexistent_provider_is_idempotent(self, client, mock_auth, auth_headers):
        # Provider allowed but row doesn't exist — still 204
        resp = client.delete(f"{BASE}/openai", headers=auth_headers)
        assert resp.status_code == 204

    def test_delete_unknown_provider_returns_400(self, client, mock_auth, auth_headers):
        resp = client.delete(f"{BASE}/mistral", headers=auth_headers)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /validate
# ---------------------------------------------------------------------------


class TestValidateEndpoint:
    def test_validate_valid_key_returns_is_valid_true(
        self, client, mock_auth, auth_headers, mocker
    ):
        mocker.patch(
            "agentic_project_service.routes.ai_provider_keys.validate_provider_key",
            return_value=_ok_result(),
        )
        resp = client.post(
            f"{BASE}/validate",
            json={"provider": "openai", "api_key": "sk-test"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["is_valid"] is True

    def test_validate_bad_key_returns_is_valid_false(self, client, mock_auth, auth_headers, mocker):
        from agentic_project_service.services.provider_validator import ValidationResult

        mocker.patch(
            "agentic_project_service.routes.ai_provider_keys.validate_provider_key",
            return_value=ValidationResult(ok=False, provider_status=401, error="Unauthorized"),
        )
        resp = client.post(
            f"{BASE}/validate",
            json={"provider": "openai", "api_key": "sk-bad"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["is_valid"] is False
        assert "error" in data

    def test_validate_does_not_store_row(self, client, mock_auth, auth_headers, mocker):
        mocker.patch(
            "agentic_project_service.routes.ai_provider_keys.validate_provider_key",
            return_value=_ok_result(),
        )
        client.post(
            f"{BASE}/validate",
            json={"provider": "openai", "api_key": "sk-test"},
            headers=auth_headers,
        )
        list_resp = client.get(BASE, headers=auth_headers)
        assert list_resp.get_json() == []

    def test_validate_unknown_provider_returns_400(self, client, mock_auth, auth_headers):
        resp = client.post(
            f"{BASE}/validate", json={"provider": "mistral", "api_key": "key"}, headers=auth_headers
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_result():
    from agentic_project_service.services.provider_validator import ValidationResult

    return ValidationResult(ok=True, provider_status=200, error=None)
