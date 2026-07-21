"""Unit tests for the _parse_list_params helper."""

import pytest
from unittest.mock import MagicMock

from agentic_project_service.services.list_params import (
    parse_list_params,
    ListParamsError,
    escape_like,
)


def _req(args: dict):
    """Build a fake Flask request object with .args.get(name, default=None)."""
    req = MagicMock()
    req.args.get = lambda name, default=None, type=None: args.get(name, default)
    return req


class TestParseListParams:
    def test_defaults(self):
        result = parse_list_params(_req({}), sort_allowed={"created_at", "name"})
        assert result == (50, 0, None, "created_at", "desc")

    def test_explicit_limit_offset(self):
        result = parse_list_params(
            _req({"limit": "20", "offset": "40"}), sort_allowed={"created_at"}
        )
        assert result[0] == 20
        assert result[1] == 40

    def test_limit_clamped_high(self):
        result = parse_list_params(_req({"limit": "999"}), sort_allowed={"created_at"})
        assert result[0] == 100

    def test_limit_clamped_low(self):
        result = parse_list_params(_req({"limit": "0"}), sort_allowed={"created_at"})
        assert result[0] == 1

    def test_offset_clamped_negative(self):
        result = parse_list_params(_req({"offset": "-5"}), sort_allowed={"created_at"})
        assert result[1] == 0

    def test_invalid_limit_falls_back_to_default(self):
        result = parse_list_params(_req({"limit": "abc"}), sort_allowed={"created_at"})
        assert result[0] == 50

    def test_invalid_offset_falls_back_to_default(self):
        result = parse_list_params(_req({"offset": "abc"}), sort_allowed={"created_at"})
        assert result[1] == 0

    def test_q_trimmed(self):
        result = parse_list_params(_req({"q": "  foo  "}), sort_allowed={"created_at"})
        assert result[2] == "foo"

    def test_q_empty_becomes_none(self):
        result = parse_list_params(_req({"q": "   "}), sort_allowed={"created_at"})
        assert result[2] is None

    def test_q_length_capped(self):
        long = "a" * 300
        result = parse_list_params(_req({"q": long}), sort_allowed={"created_at"})
        assert len(result[2]) == 200

    def test_sort_default(self):
        result = parse_list_params(_req({}), sort_allowed={"created_at", "name"})
        assert result[3] == "created_at"

    def test_sort_explicit(self):
        result = parse_list_params(_req({"sort": "name"}), sort_allowed={"created_at", "name"})
        assert result[3] == "name"

    def test_sort_invalid_raises(self):
        with pytest.raises(ListParamsError) as exc:
            parse_list_params(_req({"sort": "bogus"}), sort_allowed={"created_at", "name"})
        assert "bogus" in str(exc.value)
        assert exc.value.status == 400

    def test_order_default(self):
        result = parse_list_params(_req({}), sort_allowed={"created_at"})
        assert result[4] == "desc"

    def test_order_explicit_asc(self):
        result = parse_list_params(_req({"order": "asc"}), sort_allowed={"created_at"})
        assert result[4] == "asc"

    def test_order_case_insensitive(self):
        result = parse_list_params(_req({"order": "ASC"}), sort_allowed={"created_at"})
        assert result[4] == "asc"

    def test_order_invalid_raises(self):
        with pytest.raises(ListParamsError) as exc:
            parse_list_params(_req({"order": "sideways"}), sort_allowed={"created_at"})
        assert exc.value.status == 400


class TestDefaultUnpaginated:
    def test_no_limit_no_offset_returns_none(self):
        result = parse_list_params(_req({}), sort_allowed={"created_at"}, default_unpaginated=True)
        assert result[0] is None
        assert result[1] is None

    def test_explicit_limit_opts_in(self):
        result = parse_list_params(
            _req({"limit": "10"}), sort_allowed={"created_at"}, default_unpaginated=True
        )
        assert result[0] == 10
        assert result[1] == 0

    def test_explicit_offset_opts_in(self):
        result = parse_list_params(
            _req({"offset": "10"}), sort_allowed={"created_at"}, default_unpaginated=True
        )
        assert result[0] == 50
        assert result[1] == 10


class TestEscapeLike:
    def test_no_wildcards(self):
        assert escape_like("foo") == "foo"

    def test_percent_escaped(self):
        assert escape_like("50%") == "50\\%"

    def test_underscore_escaped(self):
        assert escape_like("a_b") == "a\\_b"

    def test_backslash_escaped(self):
        assert escape_like("a\\b") == "a\\\\b"

    def test_all_three(self):
        # Input runtime: 100%\_done  (10 chars)
        # Expected runtime: 100\%\\\_done  (13 chars)
        # Order: \ → \\, % → \%, _ → \_  (backslash first so \% doesn't re-escape)
        assert escape_like("100%\\_done") == "100\\%\\\\\\_done"
