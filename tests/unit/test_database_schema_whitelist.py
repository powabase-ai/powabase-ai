"""Tests for database route schema whitelist."""

from agentic_project_service.routes.database import _validate_schema


class TestSchemaWhitelist:
    """Only whitelisted schemas should be accessible via database routes."""

    def test_public_schema_allowed(self):
        assert _validate_schema("public") is None

    def test_ai_schema_blocked(self):
        result = _validate_schema("ai")
        assert result is not None
        assert "not accessible" in result

    def test_auth_schema_blocked(self):
        result = _validate_schema("auth")
        assert result is not None
        assert "not accessible" in result

    def test_storage_schema_blocked(self):
        result = _validate_schema("storage")
        assert result is not None
        assert "not accessible" in result

    def test_information_schema_blocked(self):
        result = _validate_schema("information_schema")
        assert result is not None

    def test_pg_catalog_blocked(self):
        result = _validate_schema("pg_catalog")
        assert result is not None

    def test_invalid_schema_name_rejected(self):
        result = _validate_schema("drop; --")
        assert result is not None
        assert "Invalid schema name" in result

    def test_empty_schema_rejected(self):
        result = _validate_schema("")
        assert result is not None

    def test_default_schema_is_public(self):
        """Verify the ALLOWED_SCHEMAS constant includes public."""
        from agentic_project_service.routes.database import ALLOWED_SCHEMAS

        assert "public" in ALLOWED_SCHEMAS
