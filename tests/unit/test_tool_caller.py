"""A ToolCaller is either the service role or one end user with their own token.

Nothing in between: an end user without a token, or a token without claims,
would let Storage fall back to the service key, and claims without a ``sub``
would put a run in the database with no user to hold it to.
"""

import inspect
import json
import time
from unittest.mock import patch

import jwt
import pytest
from flask import Flask, g

from agentic_project_service import auth
from agentic_project_service.services import tool_registry
from agentic_project_service.services.storage import SupabaseStorage, get_storage_for_user
from agentic_project_service.services.tool_caller import ToolCaller
from agentic_project_service.tools import builtin
from agentic_project_service.tools.builtin import (
    BUILTIN_HANDLERS,
    storage_read_handler,
    storage_write_handler,
)

SECRET = "unit-test-jwt-secret-with-enough-length-0123456789"
USER_SUB = "5b0f7a52-3c1e-4f7e-9a51-0d2c6e8b9a11"


# ---------------------------------------------------------------------------
# The type's invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_service(self):
        caller = ToolCaller.service()
        assert not caller.is_end_user
        assert caller.claims is None and caller.token is None

    def test_end_user(self):
        caller = ToolCaller(claims={"sub": USER_SUB, "role": "authenticated"}, token="t")
        assert caller.is_end_user
        assert caller.claims["sub"] == USER_SUB

    def test_no_arguments_is_not_the_service_role(self):
        """Who a run acts for is always said out loud: ToolCaller.service()."""
        with pytest.raises(TypeError):
            ToolCaller()

    @pytest.mark.parametrize(
        "kwargs",
        [{"claims": {"sub": USER_SUB}}, {"token": "t"}],
        ids=["claims-only", "token-only"],
    )
    def test_both_fields_must_be_given(self, kwargs):
        with pytest.raises(TypeError):
            ToolCaller(**kwargs)

    def test_the_fields_are_keyword_only(self):
        with pytest.raises(TypeError):
            ToolCaller({"sub": USER_SUB}, "t")

    def test_claims_without_a_token(self):
        with pytest.raises(ValueError):
            ToolCaller(claims={"sub": USER_SUB}, token=None)

    def test_a_token_without_claims(self):
        with pytest.raises(ValueError):
            ToolCaller(claims=None, token="t")

    @pytest.mark.parametrize("token", ["", None])
    def test_an_empty_token(self, token):
        with pytest.raises(ValueError):
            ToolCaller(claims={"sub": USER_SUB}, token=token)

    @pytest.mark.parametrize(
        "claims",
        [{}, {"sub": ""}, {"sub": None}, {"sub": 7}, {"role": "authenticated"}],
        ids=["empty", "blank-sub", "none-sub", "int-sub", "no-sub"],
    )
    def test_claims_need_a_subject(self, claims):
        with pytest.raises(ValueError):
            ToolCaller(claims=claims, token="t")

    def test_the_claims_are_a_copy(self):
        source = {"sub": USER_SUB, "role": "authenticated"}
        caller = ToolCaller(claims=source, token="t")
        source["sub"] = "someone-else"
        source["role"] = "service_role"
        assert caller.claims == {"sub": USER_SUB, "role": "authenticated"}

    def test_the_claims_are_read_only(self):
        caller = ToolCaller(claims={"sub": USER_SUB, "role": "authenticated"}, token="t")
        with pytest.raises(TypeError):
            caller.claims["sub"] = "someone-else"
        with pytest.raises(TypeError):
            caller.claims["role"] = "service_role"
        assert caller.claims == {"sub": USER_SUB, "role": "authenticated"}

    def test_the_claims_serialize_for_the_database_session(self):
        """agent_sql copies them into a dict and puts them in request.jwt.claims."""
        caller = ToolCaller(claims={"sub": USER_SUB, "aud": "authenticated"}, token="t")
        assert json.loads(json.dumps(dict(caller.claims))) == {
            "sub": USER_SUB,
            "aud": "authenticated",
        }
        assert dict(caller.claims.items())["sub"] == USER_SUB

    def test_the_token_stays_out_of_repr(self):
        caller = ToolCaller(claims={"sub": USER_SUB}, token="secret-bearer")
        assert "secret-bearer" not in repr(caller)
        assert USER_SUB in repr(caller)


class TestSessionExpired:
    """An end user's session is over once its ``exp`` has passed, and a token
    that does not say when it ends is treated as ended."""

    def _user(self, **claims):
        return ToolCaller(claims={"sub": USER_SUB, **claims}, token="t")

    def test_a_future_exp_is_live(self):
        assert not self._user(exp=time.time() + 60).session_expired()

    def test_a_past_exp_has_expired(self):
        assert self._user(exp=int(time.time()) - 1).session_expired()

    @pytest.mark.parametrize(
        "claims",
        [{}, {"exp": None}, {"exp": "4102444800"}, {"exp": True}, {"exp": [4102444800]}],
        ids=["missing", "null", "string", "bool", "list"],
    )
    def test_no_numeric_exp_counts_as_expired(self, claims):
        assert self._user(**claims).session_expired()

    def test_the_service_role_never_expires(self):
        assert not ToolCaller.service().session_expired()


# ---------------------------------------------------------------------------
# from_request, through the real auth decorator
# ---------------------------------------------------------------------------


@pytest.fixture
def signed(monkeypatch):
    """JWT_SECRET and a service key, as a project is configured."""
    now = int(time.time())
    service_key = jwt.encode(
        {"role": "service_role", "iss": "supabase", "iat": now, "exp": now + 3600},
        SECRET,
        algorithm="HS256",
    )
    user_token = jwt.encode(
        {
            "sub": USER_SUB,
            "role": "authenticated",
            "aud": "authenticated",
            "iat": now,
            "exp": now + 3600,
        },
        SECRET,
        algorithm="HS256",
    )
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setenv("SERVICE_ROLE_KEY", service_key)
    return {"service": service_key, "user": user_token}


@pytest.fixture
def app():
    return Flask(__name__)


def _caller_for(app, token):
    """ToolCaller.from_request() inside a route an end user may call."""
    route = auth.require_user_auth(ToolCaller.from_request)
    with app.test_request_context(headers={"Authorization": f"Bearer {token}"}):
        return route()


class TestFromRequest:
    def test_the_service_key_is_the_service_role(self, app, signed):
        caller = _caller_for(app, signed["service"])
        assert isinstance(caller, ToolCaller)
        assert not caller.is_end_user

    def test_an_end_user_carries_their_claims_and_token(self, app, signed):
        caller = _caller_for(app, signed["user"])
        assert caller.is_end_user
        assert caller.claims["sub"] == USER_SUB
        assert caller.token == signed["user"]

    def test_no_authenticated_payload_raises(self, app, signed):
        """A route that forgot its auth decorator must not get a caller."""
        with app.test_request_context(headers={"Authorization": f"Bearer {signed['user']}"}):
            with pytest.raises(RuntimeError):
                ToolCaller.from_request()

    def test_no_bearer_token_raises(self, app, signed):
        with app.test_request_context():
            g.user_id = USER_SUB
            g.jwt_payload = {"sub": USER_SUB, "role": "authenticated"}
            with pytest.raises(RuntimeError):
                ToolCaller.from_request()

    def test_an_empty_payload_raises(self, app, signed):
        with app.test_request_context(headers={"Authorization": f"Bearer {signed['user']}"}):
            g.jwt_payload = {}
            with pytest.raises(RuntimeError):
                ToolCaller.from_request()


# ---------------------------------------------------------------------------
# Storage never falls back to the service key for an end user
# ---------------------------------------------------------------------------


class TestStorageBearer:
    @pytest.mark.parametrize("token", ["", None])
    def test_get_storage_for_user_needs_a_token(self, token):
        with pytest.raises(ValueError):
            get_storage_for_user(token)

    def test_an_empty_bearer_is_refused(self):
        with pytest.raises(ValueError):
            SupabaseStorage(url="http://s.test", service_key="service-key", bearer="")

    def test_no_bearer_is_the_service_role(self):
        storage = SupabaseStorage(url="http://s.test", service_key="service-key")
        assert storage.headers["Authorization"] == "Bearer service-key"

    def test_a_bearer_is_used_as_is(self):
        storage = get_storage_for_user("user-token")
        assert storage.headers["Authorization"] == "Bearer user-token"
        assert storage.headers["apikey"] == storage.service_key

    @pytest.fixture(autouse=True)
    def _service_key(self, monkeypatch):
        monkeypatch.setenv("SERVICE_ROLE_KEY", "service-key")
        monkeypatch.delenv("STORAGE_URL", raising=False)


# ---------------------------------------------------------------------------
# Which builtin tools act as the caller
# ---------------------------------------------------------------------------


def test_every_tool_that_needs_a_caller_gets_one_injected():
    """A builtin that reads ``_caller`` must be one the loader injects it into,
    and the other way round; otherwise a new data tool would always refuse, or
    the loader would hand the caller to a tool that ignores it."""
    reads_caller = {
        name
        for name, handler in BUILTIN_HANDLERS.items()
        if "_pop_caller(" in inspect.getsource(handler)
    }
    injected = {"database_query", "database_write"} | set(tool_registry._CALLER_SCOPED_TOOLS)
    assert reads_caller == injected


@pytest.mark.parametrize(
    "call",
    [
        lambda a: storage_read_handler({"operation": "list", **a}, None),
        lambda a: storage_read_handler({"operation": "download", **a}, None),
        lambda a: storage_read_handler({"operation": "nope", **a}, None),
        lambda a: storage_write_handler({"content": "x", **a}, None),
    ],
    ids=["list", "download", "bad-op", "write"],
)
@pytest.mark.parametrize(
    "args",
    [
        {},
        {"bucket": "", "path": ""},
        {"bucket": "sources", "path": "a.pdf"},
        {"bucket": "x/../sources", "path": "../a"},
        {"bucket": "docs", "path": "a.txt"},
    ],
    ids=["nothing", "blank", "internal", "traversal", "ordinary"],
)
def test_storage_checks_the_caller_before_anything_else(call, args):
    with (
        patch.object(builtin, "get_storage") as service_storage,
        patch.object(builtin, "get_storage_for_user") as user_storage,
    ):
        result = json.loads(call(dict(args)))
    assert result == {"error": builtin._NO_CALLER_MESSAGE}
    service_storage.assert_not_called()
    user_storage.assert_not_called()
