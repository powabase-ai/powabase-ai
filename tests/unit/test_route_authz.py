"""Who may call which /api route: the project's service role key, or an end user.

A project's anon key is public and the gateway only checks that *some* key is
present, so every signed-in end user of a project can reach every /api route
with their own GoTrue JWT. Authorization is therefore this service's job, and
it is deny-by-default:

  * ``require_service_role`` — the default for every route. Management of
    knowledge bases, sources, agents, workflows, settings, keys, tables and
    observability is an administrator's job, done with the service role key
    from a trusted backend or the dashboard.
  * ``require_user_auth`` — a short, explicit list of conversation routes an
    end user may call directly. Each of them scopes what it returns to the
    caller's own sessions and runs.

The inventory test below is the guard: a new route that takes neither
decorator, or a route quietly moved onto the end-user list, fails it.

DB-free, like the rest of tests/unit: blueprints are registered on a bare
Flask app, and the data layer is stubbed where a request gets past auth.
"""

import importlib
import logging
import pkgutil
import time
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
from flask import Blueprint, Flask, g, jsonify

from agentic_project_service import auth
from agentic_project_service import routes as routes_pkg
from agentic_project_service.routes import agents as agents_route
from agentic_project_service.routes import orchestrations as orchestrations_route
from agentic_project_service.routes import sessions as sessions_route
from agentic_project_service.services import run_registry

USER_ID = "11111111-1111-4111-8111-111111111111"
OTHER_USER_ID = "22222222-2222-4222-8222-222222222222"
AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"
ORCH_ID = "4a8b2d3f-6c8e-4f22-8b4d-9e3a7c5f2b81"
KB_ID = "5b9c3e4a-7d9f-4a33-9c5e-af4b8d6a3c92"
# A user id with letters in it, so that its upper-case spelling differs.
LETTERED_USER_ID = "aa1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _authed_as_user(user_id=USER_ID):
    return patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": user_id, "role": "authenticated", "aud": "authenticated"},
    )


def _authed_as_service_role():
    return patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"role": "service_role", "is_service_role": True},
    )


HEADERS = {"Authorization": "Bearer fake"}


def _all_blueprints() -> list[Blueprint]:
    """Every Blueprint defined anywhere under the routes package.

    Walked rather than listed, so a blueprint added in a new module is covered
    by the inventory without anyone remembering to add it here.
    """
    found: dict[int, Blueprint] = {}
    for info in pkgutil.walk_packages(routes_pkg.__path__, routes_pkg.__name__ + "."):
        module = importlib.import_module(info.name)
        for value in vars(module).values():
            if isinstance(value, Blueprint):
                found[id(value)] = value
    return list(found.values())


def _app_with(*blueprints: Blueprint) -> Flask:
    app = Flask(__name__)
    for bp in blueprints:
        app.register_blueprint(bp)
    return app


def _full_app() -> Flask:
    return _app_with(*_all_blueprints())


# ---------------------------------------------------------------------------
# Inventory: deny-by-default, and the end-user list is exactly this
# ---------------------------------------------------------------------------

# Every route an end user's own JWT may call, as (method, rule, endpoint).
# Adding to this set is a security decision: the route must scope everything it
# reads or changes to the caller. Keyed by method and rule, not by endpoint
# alone, so a new URL or method mapped onto an allowed view is caught too.
END_USER_ROUTES = {
    # Agent conversations. Sessions and runs are owned by the calling user.
    ("GET", "/api/agents/<agent_id>/sessions", "agents.list_sessions"),
    ("POST", "/api/agents/<agent_id>/sessions", "agents.create_session_for_agent"),
    ("DELETE", "/api/agents/<agent_id>/sessions/<session_id>", "agents.delete_session_for_agent"),
    ("POST", "/api/agents/<agent_id>/run", "agents.run_agent"),
    ("POST", "/api/agents/<agent_id>/run/stream", "agents.run_agent_stream"),
    ("GET", "/api/agents/runs/<run_id>", "agents.get_agent_run"),
    ("POST", "/api/agents/runs/<run_id>/approve", "agents.approve_run"),
    # Session reads/deletes, owner-only.
    ("GET", "/api/sessions/<session_id>", "sessions.get_session"),
    ("GET", "/api/sessions/<session_id>/messages", "sessions.get_messages"),
    ("GET", "/api/sessions/<session_id>/runs", "sessions.get_runs"),
    (
        "GET",
        "/api/sessions/<session_id>/runs/<run_id>/retrieved-context",
        "sessions.get_run_retrieved_context_route",
    ),
    ("DELETE", "/api/sessions/<session_id>", "sessions.delete_session_route"),
    # Orchestration conversations, owner-only.
    (
        "GET",
        "/api/orchestrations/<orch_id>/sessions",
        "orchestrations.list_orchestration_sessions",
    ),
    (
        "GET",
        "/api/orchestrations/<orch_id>/sessions/<session_id>/messages",
        "orchestrations.get_orchestration_session_messages",
    ),
    (
        "POST",
        "/api/orchestrations/<orch_id>/run/stream",
        "orchestrations.run_orchestration_stream",
    ),
    ("GET", "/api/orchestrations/runs/<run_id>", "orchestrations.get_orchestration_run"),
}

# Routes that take no bearer at all because they authenticate some other way.
UNAUTHENTICATED_ROUTES = {
    # Inbound webhooks: verified by the per-workflow webhook secret.
    ("POST", "/api/webhooks/<webhook_id>", "webhooks.trigger_webhook"),
    # Docs search for the in-project copilot: verified by X-Docs-Search-Token.
    ("POST", "/api/internal/docs/search", "internal_docs.docs_search"),
}

Route = tuple[str, str, str]


def _api_routes(app: Flask) -> dict[Route, object]:
    """Every (method, rule, endpoint) under /api/, with the view it calls."""
    return {
        (method, rule.rule, rule.endpoint): app.view_functions[rule.endpoint]
        for rule in app.url_map.iter_rules()
        if rule.rule.startswith("/api/") and rule.endpoint != "static"
        for method in sorted(rule.methods - {"HEAD", "OPTIONS"})
    }


@pytest.fixture(scope="module")
def served_app() -> Flask:
    """The application exactly as it is served.

    Routes added on the app itself, or blueprints registered with a
    ``url_prefix``, exist only here; a walk of the routes package misses them.
    """
    from agentic_project_service.main import create_app

    return create_app(testing=True)


def _inventory(*apps: Flask) -> dict[Route, object]:
    routes: dict[Route, object] = {}
    for app in apps:
        routes.update(_api_routes(app))
    return routes


def _undeclared(routes: dict[Route, object]) -> list[Route]:
    return sorted(
        route
        for route, view in routes.items()
        if getattr(view, "auth_mode", None) is None and route not in UNAUTHENTICATED_ROUTES
    )


def _end_user(routes: dict[Route, object]) -> set[Route]:
    return {route for route, view in routes.items() if getattr(view, "auth_mode", None) == "user"}


class TestRouteInventory:
    def test_every_api_route_declares_who_may_call_it(self, served_app):
        assert _undeclared(_inventory(served_app, _full_app())) == []

    def test_end_user_routes_are_exactly_the_allowed_list(self, served_app):
        assert _end_user(_inventory(served_app, _full_app())) == END_USER_ROUTES

    def test_unauthenticated_list_names_real_routes(self, served_app):
        """A stale entry here would silently exempt a future route of that name."""
        assert UNAUTHENTICATED_ROUTES <= set(_inventory(served_app))

    def test_served_app_has_every_blueprint_route(self, served_app):
        """A blueprint defined but never registered would pass the walk and never be served."""
        assert set(_api_routes(_full_app())) <= set(_api_routes(served_app))

    def test_catches_a_route_added_on_the_app_itself(self):
        from agentic_project_service.main import create_app

        app = create_app(testing=True)

        @app.route("/api/debug/env")
        def debug_env():
            return jsonify({})

        assert _undeclared(_inventory(app)) == [("GET", "/api/debug/env", "debug_env")]

    def test_catches_a_blueprint_mounted_under_api_by_its_registration(self):
        from agentic_project_service.main import create_app

        app = create_app(testing=True)
        extra = Blueprint("extra", __name__)

        @extra.route("/dump")
        def dump():
            return jsonify({})

        @extra.route("/mine")
        @auth.require_user_auth
        def mine():
            return jsonify({})

        app.register_blueprint(extra, url_prefix="/api/extra")
        routes = _inventory(app)
        assert _undeclared(routes) == [("GET", "/api/extra/dump", "extra.dump")]
        assert _end_user(routes) - END_USER_ROUTES == {("GET", "/api/extra/mine", "extra.mine")}

    def test_catches_a_new_url_mapped_onto_an_end_user_view(self):
        """The view scopes its own URL; a new URL for it is a new decision."""
        from agentic_project_service.main import create_app

        app = create_app(testing=True)
        app.add_url_rule(
            "/api/sessions/<session_id>/export",
            endpoint="sessions.get_session",
            view_func=app.view_functions["sessions.get_session"],
        )
        assert _end_user(_inventory(app)) - END_USER_ROUTES == {
            ("GET", "/api/sessions/<session_id>/export", "sessions.get_session")
        }

    def test_catches_a_new_method_on_an_end_user_url(self):
        from agentic_project_service.main import create_app

        app = create_app(testing=True)
        app.add_url_rule(
            "/api/sessions/<session_id>",
            endpoint="sessions.get_session",
            view_func=app.view_functions["sessions.get_session"],
            methods=["PATCH"],
        )
        assert _end_user(_inventory(app)) - END_USER_ROUTES == {
            ("PATCH", "/api/sessions/<session_id>", "sessions.get_session")
        }

    def test_catches_a_new_url_mapped_onto_an_unauthenticated_view(self):
        from agentic_project_service.main import create_app

        app = create_app(testing=True)
        app.add_url_rule(
            "/api/webhooks/<webhook_id>/replay",
            endpoint="webhooks.trigger_webhook",
            view_func=app.view_functions["webhooks.trigger_webhook"],
            methods=["POST"],
        )
        assert _undeclared(_inventory(app)) == [
            ("POST", "/api/webhooks/<webhook_id>/replay", "webhooks.trigger_webhook")
        ]


# ---------------------------------------------------------------------------
# The decorators themselves
# ---------------------------------------------------------------------------


def _decorator_app() -> Flask:
    app = Flask(__name__)

    @app.route("/admin")
    @auth.require_service_role
    def admin():
        return jsonify({"ok": True})

    @app.route("/mine")
    @auth.require_user_auth
    def mine():
        return jsonify({"user": auth.get_current_user_id()})

    @app.route("/whoami")
    @auth.require_user_auth
    def whoami():
        return jsonify(
            {"user": auth.get_current_user_id(), "service": auth.is_service_role_request()}
        )

    @app.route("/claims-sub")
    @auth.require_user_auth
    def claims_sub():
        return jsonify({"sub": g.jwt_payload.get("sub")})

    return app


class TestRequireServiceRole:
    def test_rejects_end_user_jwt_with_403(self):
        with _authed_as_user(), _decorator_app().test_client() as c:
            resp = c.get("/admin", headers=HEADERS)
        assert resp.status_code == 403
        assert "service role" in resp.get_json()["error"]

    def test_allows_service_role(self):
        with _authed_as_service_role(), _decorator_app().test_client() as c:
            resp = c.get("/admin", headers=HEADERS)
        assert resp.status_code == 200

    def test_missing_bearer_is_401(self):
        with _decorator_app().test_client() as c:
            resp = c.get("/admin")
        assert resp.status_code == 401

    def test_invalid_token_is_401_not_403(self):
        with (
            patch(
                "agentic_project_service.auth.decode_jwt",
                side_effect=auth.AuthError("Invalid token: bad"),
            ),
            _decorator_app().test_client() as c,
        ):
            resp = c.get("/admin", headers=HEADERS)
        assert resp.status_code == 401


class TestRequireUserAuth:
    def test_end_user_token_without_sub_is_401(self):
        """Without an id, every "is it theirs?" check downstream compares None to None."""
        with (
            patch(
                "agentic_project_service.auth.decode_jwt",
                return_value={"role": "authenticated", "aud": "authenticated"},
            ),
            _decorator_app().test_client() as c,
        ):
            resp = c.get("/mine", headers=HEADERS)
        assert resp.status_code == 401

    @pytest.mark.parametrize(
        "sub", ["", "user-1", 42, "{" + USER_ID + "}", USER_ID.replace("-", "")]
    )
    def test_end_user_token_with_a_non_uuid_sub_is_401(self, sub):
        with _authed_as_user(sub), _decorator_app().test_client() as c:
            resp = c.get("/mine", headers=HEADERS)
        assert resp.status_code == 401

    def test_upper_case_sub_is_the_same_user(self):
        """Owners are compared as canonical strings, so the id is normalised once, here."""
        with _authed_as_user(LETTERED_USER_ID.upper()), _decorator_app().test_client() as c:
            whoami = c.get("/whoami", headers=HEADERS)
            claims = c.get("/claims-sub", headers=HEADERS)
        assert whoami.status_code == 200
        assert whoami.get_json()["user"] == LETTERED_USER_ID
        # The tools act with these claims; they must name the same user.
        assert claims.get_json() == {"sub": LETTERED_USER_ID}

    def test_service_role_needs_no_sub(self):
        with _authed_as_service_role(), _decorator_app().test_client() as c:
            resp = c.get("/whoami", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.get_json() == {"user": None, "service": True}

    def test_allows_end_user_and_exposes_their_id(self):
        with _authed_as_user(), _decorator_app().test_client() as c:
            resp = c.get("/mine", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.get_json() == {"user": USER_ID}

    def test_allows_service_role(self):
        with _authed_as_service_role(), _decorator_app().test_client() as c:
            resp = c.get("/mine", headers=HEADERS)
        assert resp.status_code == 200

    def test_missing_bearer_is_401(self):
        with _decorator_app().test_client() as c:
            resp = c.get("/mine")
        assert resp.status_code == 401


class TestRefusalsAreLogged:
    def test_service_only_refusal_logs_method_path_and_sub(self, caplog):
        with (
            caplog.at_level(logging.INFO, logger="agentic_project_service.auth"),
            _authed_as_user(),
            _decorator_app().test_client() as c,
        ):
            c.get("/admin", headers={"Authorization": "Bearer do-not-log-me"})
        refusals = [r for r in caplog.records if r.name == "agentic_project_service.auth"]
        assert len(refusals) == 1
        assert refusals[0].levelno == logging.INFO
        message = refusals[0].getMessage()
        assert "GET" in message and "/admin" in message and USER_ID in message
        assert "do-not-log-me" not in message

    def test_service_role_is_not_logged(self, caplog):
        with (
            caplog.at_level(logging.INFO, logger="agentic_project_service.auth"),
            _authed_as_service_role(),
            _decorator_app().test_client() as c,
        ):
            c.get("/admin", headers=HEADERS)
        assert [r for r in caplog.records if r.name == "agentic_project_service.auth"] == []


# ---------------------------------------------------------------------------
# Real tokens through decode_jwt — nothing mocked below the HTTP request
# ---------------------------------------------------------------------------

JWT_SECRET = "unit-test-jwt-secret-that-is-long-enough-for-hs256"


def _sign(claims: dict, secret: str = JWT_SECRET) -> str:
    return pyjwt.encode(claims, secret, algorithm="HS256")


def _user_claims(**overrides) -> dict:
    claims = {
        "sub": USER_ID,
        "role": "authenticated",
        "aud": "authenticated",
        "exp": int(time.time()) + 3600,
    }
    claims.update(overrides)
    return {k: v for k, v in claims.items() if v is not None}


SERVICE_KEY = _sign({"role": "service_role", "iss": "supabase", "iat": 1700000000})


@pytest.fixture
def real_keys(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
    monkeypatch.setenv("SERVICE_ROLE_KEY", SERVICE_KEY)


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.usefixtures("real_keys")
class TestRealTokens:
    def test_user_token_is_an_end_user(self):
        with _decorator_app().test_client() as c:
            resp = c.get("/whoami", headers=_bearer(_sign(_user_claims())))
        assert resp.status_code == 200
        assert resp.get_json() == {"user": USER_ID, "service": False}

    def test_user_token_is_refused_on_a_service_route_with_the_documented_body(self):
        with _decorator_app().test_client() as c:
            resp = c.get("/admin", headers=_bearer(_sign(_user_claims())))
        assert resp.status_code == 403
        assert resp.get_json() == {"error": "This endpoint requires the project's service role key"}

    def test_the_exact_service_key_is_the_service_role(self):
        with _decorator_app().test_client() as c:
            admin = c.get("/admin", headers=_bearer(SERVICE_KEY))
            whoami = c.get("/whoami", headers=_bearer(SERVICE_KEY))
        assert admin.status_code == 200
        assert whoami.get_json()["service"] is True

    def test_is_service_role_claim_in_a_user_token_is_not_the_service_role(self):
        """A custom access-token hook, or a project minting its own tokens, can add any claim."""
        token = _sign(_user_claims(is_service_role=True))
        with _decorator_app().test_client() as c:
            admin = c.get("/admin", headers=_bearer(token))
            whoami = c.get("/whoami", headers=_bearer(token))
        assert admin.status_code == 403
        assert whoami.get_json() == {"user": USER_ID, "service": False}

    def test_decode_jwt_drops_the_claim_from_user_payloads(self):
        payload = auth.decode_jwt(_sign(_user_claims(is_service_role=True)))
        assert "is_service_role" not in payload
        assert payload["sub"] == USER_ID

    def test_decode_jwt_marks_the_service_key(self):
        assert auth.decode_jwt(SERVICE_KEY)["is_service_role"] is True

    def test_a_different_token_with_role_service_role_is_not_the_service_role(self):
        forged = _sign(_user_claims(role="service_role"))
        with _decorator_app().test_client() as c:
            resp = c.get("/admin", headers=_bearer(forged))
        assert resp.status_code == 403

    def test_token_without_sub_is_401(self):
        with _decorator_app().test_client() as c:
            resp = c.get("/mine", headers=_bearer(_sign(_user_claims(sub=None))))
        assert resp.status_code == 401

    def test_token_with_a_non_uuid_sub_is_401(self):
        with _decorator_app().test_client() as c:
            resp = c.get("/mine", headers=_bearer(_sign(_user_claims(sub="user-1"))))
        assert resp.status_code == 401

    def test_token_without_exp_is_401(self):
        """A token that never expires is not one GoTrue issues."""
        with _decorator_app().test_client() as c:
            resp = c.get("/mine", headers=_bearer(_sign(_user_claims(exp=None))))
        assert resp.status_code == 401
        assert "exp" in resp.get_json()["error"]

    def test_token_with_a_non_numeric_exp_is_401(self):
        with _decorator_app().test_client() as c:
            resp = c.get("/mine", headers=_bearer(_sign(_user_claims(exp="never"))))
        assert resp.status_code == 401

    def test_upper_case_sub_is_the_same_user(self):
        with _decorator_app().test_client() as c:
            token = _sign(_user_claims(sub=LETTERED_USER_ID.upper()))
            resp = c.get("/whoami", headers=_bearer(token))
        assert resp.status_code == 200
        assert resp.get_json() == {"user": LETTERED_USER_ID, "service": False}

    def test_the_service_key_needs_no_exp(self):
        """Project service keys are long-lived; only end-user tokens must expire."""
        assert "exp" not in pyjwt.decode(SERVICE_KEY, options={"verify_signature": False})
        with _decorator_app().test_client() as c:
            resp = c.get("/admin", headers=_bearer(SERVICE_KEY))
        assert resp.status_code == 200

    def test_expired_token_is_401(self):
        expired = _sign(_user_claims(exp=int(time.time()) - 60))
        with _decorator_app().test_client() as c:
            resp = c.get("/mine", headers=_bearer(expired))
        assert resp.status_code == 401
        assert resp.get_json() == {"error": "Token has expired"}

    def test_token_signed_with_another_secret_is_401(self):
        with _decorator_app().test_client() as c:
            resp = c.get("/mine", headers=_bearer(_sign(_user_claims(), secret="x" * 40)))
        assert resp.status_code == 401

    def test_without_a_service_key_configured_the_service_key_is_401(self, monkeypatch):
        """It is decoded as a user token then, and it has no audience."""
        monkeypatch.delenv("SERVICE_ROLE_KEY")
        with _decorator_app().test_client() as c:
            resp = c.get("/admin", headers=_bearer(SERVICE_KEY))
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Real admin routes refuse an end user before touching any data
# ---------------------------------------------------------------------------


class TestAdminRoutesRefuseEndUsers:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/api/knowledge-bases"),
            ("GET", f"/api/knowledge-bases/{KB_ID}"),
            ("DELETE", f"/api/knowledge-bases/{KB_ID}"),
            ("POST", f"/api/knowledge-bases/{KB_ID}/search"),
            ("GET", "/api/sources"),
            ("GET", f"/api/sources/{KB_ID}/download"),
            ("GET", f"/api/sources/{KB_ID}/page-texts"),
            ("POST", "/api/sources/upload"),
            ("GET", "/api/database/tables"),
            ("GET", "/api/database/tables/customers"),
            ("POST", "/api/database/tables/customers"),
            ("GET", "/api/ai-provider-keys"),
            ("PUT", "/api/ai-provider-keys"),
            ("GET", "/api/settings"),
            ("GET", "/api/agents"),
            ("PATCH", f"/api/agents/{AGENT_ID}"),
            ("POST", f"/api/workflows/{AGENT_ID}/execute"),
            ("GET", "/api/observability/health"),
        ],
    )
    def test_end_user_gets_403(self, method, path):
        app = _full_app()
        db_session = MagicMock(side_effect=AssertionError("route body must not run"))
        with (
            _authed_as_user(),
            patch("agentic_project_service.db.db.session", db_session),
            app.test_client() as c,
        ):
            resp = c.open(path, method=method, headers=HEADERS, json={})
        assert resp.status_code == 403, (method, path, resp.get_data(as_text=True))


# ---------------------------------------------------------------------------
# End-user runs cannot point an agent at data the agent was not given
# ---------------------------------------------------------------------------


class TestEndUserRunBodyRestrictions:
    @pytest.mark.parametrize(
        "body",
        [
            {"knowledge_bases": [{"id": KB_ID}]},
            {"runtime_knowledge_bases": [{"id": KB_ID}]},
            {"context_handler_id": "ctx_123"},
            {"context_items": [{"item_id": "chunk-1"}]},
            {"context_items": [{"text": "mine"}, {"item_id": "chunk-1"}]},
        ],
    )
    def test_forbidden_fields_are_named(self, body):
        assert auth.end_user_run_body_error({"message": "hi", **body}) is not None

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"context_override": "text the caller supplies themselves"},
            {"context_items": [{"text": "mine", "meta": {"a": 1}}]},
            {"knowledge_bases": []},
            {"session_id": "sess_abc", "temperature": 0.2},
        ],
    )
    def test_caller_supplied_context_is_allowed(self, body):
        assert auth.end_user_run_body_error({"message": "hi", **body}) is None

    @pytest.mark.parametrize(
        "path",
        [f"/api/agents/{AGENT_ID}/run", f"/api/agents/{AGENT_ID}/run/stream"],
    )
    @pytest.mark.parametrize(
        "body",
        [
            {"knowledge_bases": [{"id": KB_ID}]},
            {"runtime_knowledge_bases": [{"id": KB_ID}]},
            {"context_handler_id": "ctx_123"},
            {"context_items": [{"item_id": "chunk-1"}]},
        ],
    )
    def test_agent_run_routes_reject_them_for_end_users(self, path, body):
        app = _app_with(agents_route.agents_bp)
        with (
            _authed_as_user(),
            patch.object(agents_route.db, "session", MagicMock()),
            patch.object(agents_route.billing, "check_balance") as check_balance,
            app.test_client() as c,
        ):
            resp = c.post(path, headers=HEADERS, json={"message": "hi", **body})
        assert resp.status_code == 403, resp.get_data(as_text=True)
        check_balance.assert_not_called()

    def test_orchestration_run_rejects_runtime_kbs_for_end_users(self):
        app = _app_with(orchestrations_route.orchestrations_bp)
        with (
            _authed_as_user(),
            patch.object(orchestrations_route.db, "session", MagicMock()),
            patch.object(orchestrations_route.billing, "check_balance") as check_balance,
            app.test_client() as c,
        ):
            resp = c.post(
                f"/api/orchestrations/{ORCH_ID}/run/stream",
                headers=HEADERS,
                json={"message": "hi", "runtime_knowledge_bases": [{"id": KB_ID}]},
            )
        assert resp.status_code == 403, resp.get_data(as_text=True)
        check_balance.assert_not_called()

    def test_service_role_may_still_pass_knowledge_bases(self):
        """The trusted-backend path keeps every override; the gate is end-user only."""
        app = _app_with(agents_route.agents_bp)
        app.testing = True  # let the sentinel exception propagate
        with (
            _authed_as_service_role(),
            patch.object(agents_route.db, "session", MagicMock()),
            patch.object(
                agents_route.billing, "check_balance", side_effect=RuntimeError("got past auth")
            ),
            app.test_client() as c,
        ):
            with pytest.raises(RuntimeError, match="got past auth"):
                c.post(
                    f"/api/agents/{AGENT_ID}/run",
                    headers=HEADERS,
                    json={"message": "hi", "knowledge_bases": [{"id": KB_ID}]},
                )


# ---------------------------------------------------------------------------
# An end user may only continue a session they own
# ---------------------------------------------------------------------------


def _session_owner_lookup(row):
    """A db.session whose single execute().fetchone() returns ``row``."""
    session = MagicMock()
    session.execute.return_value.fetchone.return_value = row
    return session


class TestSessionAccessibleTo:
    def test_unknown_session_id_is_accessible(self):
        """Run creates it, owned by the caller."""
        from agentic_project_service.services.session import session_accessible_to

        assert session_accessible_to(_session_owner_lookup(None), "sess_new", USER_ID) is True

    def test_own_session_is_accessible(self):
        from agentic_project_service.services.session import session_accessible_to

        assert session_accessible_to(_session_owner_lookup((USER_ID,)), "sess_a", USER_ID) is True

    def test_someone_elses_session_is_not(self):
        from agentic_project_service.services.session import session_accessible_to

        assert (
            session_accessible_to(_session_owner_lookup((OTHER_USER_ID,)), "sess_b", USER_ID)
            is False
        )

    def test_ownerless_session_is_not(self):
        """Sessions a backend created without a user_id belong to no end user."""
        from agentic_project_service.services.session import session_accessible_to

        assert session_accessible_to(_session_owner_lookup((None,)), "sess_c", USER_ID) is False


class TestRunRoutesUseStrictOwnership:
    @pytest.mark.parametrize(
        "path",
        [f"/api/agents/{AGENT_ID}/run", f"/api/agents/{AGENT_ID}/run/stream"],
    )
    def test_end_user_cannot_run_on_a_session_they_do_not_own(self, path):
        app = _app_with(agents_route.agents_bp)
        with (
            _authed_as_user(),
            patch.object(agents_route.db, "session", MagicMock()),
            patch.object(agents_route.billing, "check_balance"),
            patch.object(agents_route, "session_accessible_to", return_value=False) as check,
            app.test_client() as c,
        ):
            resp = c.post(path, headers=HEADERS, json={"message": "hi", "session_id": "sess_x"})
        assert resp.status_code == 404
        check.assert_called_once()
        assert check.call_args.args[1:] == ("sess_x", USER_ID)

    @pytest.mark.parametrize(
        "path",
        [f"/api/agents/{AGENT_ID}/run", f"/api/agents/{AGENT_ID}/run/stream"],
    )
    def test_owner_gets_past_the_session_check(self, path):
        """Positive control: the next thing the route does is look the agent up."""
        app = _app_with(agents_route.agents_bp)
        session = MagicMock()
        session.execute.return_value.fetchone.return_value = None
        with (
            _authed_as_user(),
            patch.object(agents_route.db, "session", session),
            patch.object(agents_route.billing, "check_balance"),
            patch.object(agents_route, "session_accessible_to", return_value=True),
            app.test_client() as c,
        ):
            resp = c.post(path, headers=HEADERS, json={"message": "hi", "session_id": "sess_x"})
        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Agent not found"}


# ---------------------------------------------------------------------------
# GET /api/agents/runs/<id> — end users see only runs in their own sessions
# ---------------------------------------------------------------------------


def _agent_run(session_id):
    """An AgentRun row the route can serialize: every column None but these."""
    run = MagicMock()
    for column in (
        "id",
        "parent_orchestration_run_id",
        "parent_workflow_execution_id",
        "status",
        "input_messages",
        "output_messages",
        "content",
        "usage",
        "retrieved_context",
        "error",
        "started_at",
        "completed_at",
        "steps",
        "events",
        "tool_calls",
        "reasoning_steps",
        "created_at",
    ):
        setattr(run, column, None)
    run.session_id = session_id
    run.run_id = "run_1"
    return run


class TestGetAgentRunOwnership:
    def _get(self, run, session_owner, *, service=False):
        app = _app_with(agents_route.agents_bp)
        session_row = None if session_owner is ... else MagicMock(user_id=session_owner)
        with (
            _authed_as_service_role() if service else _authed_as_user(),
            patch.object(agents_route, "AgentRun") as agent_run_model,
            patch.object(agents_route, "AgentSession") as agent_session_model,
            app.test_client() as c,
        ):
            agent_run_model.query.filter_by.return_value.first.return_value = run
            agent_session_model.query.filter_by.return_value.first.return_value = session_row
            return c.get("/api/agents/runs/run_1", headers=HEADERS)

    def test_sessionless_run_is_hidden_from_end_users(self):
        """Delegated and workflow-block runs have no owner an end user could be."""
        assert self._get(_agent_run(None), ...).status_code == 404

    def test_ownerless_session_run_is_hidden_from_end_users(self):
        assert self._get(_agent_run("s-uuid"), None).status_code == 404

    def test_other_users_run_is_hidden(self):
        assert self._get(_agent_run("s-uuid"), OTHER_USER_ID).status_code == 404

    def test_owner_sees_their_run(self):
        resp = self._get(_agent_run("s-uuid"), USER_ID)
        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert resp.get_json()["run_id"] == "run_1"

    def test_service_role_sees_any_run(self):
        assert self._get(_agent_run(None), ..., service=True).status_code == 200


# ---------------------------------------------------------------------------
# GET /api/agents/<id>/sessions — an end user lists only their own sessions
# ---------------------------------------------------------------------------


class TestListAgentSessionsScoping:
    def _list(self, *, service=False, user_id=...):
        app = _app_with(agents_route.agents_bp)
        session = MagicMock()
        session.execute.return_value.__iter__.return_value = iter([])
        session.execute.return_value.scalar.return_value = 0
        user_patch = (
            patch.object(agents_route, "get_current_user_id", return_value=user_id)
            if user_id is not ...
            else patch.object(agents_route, "get_current_user_id", wraps=auth.get_current_user_id)
        )
        with (
            _authed_as_service_role() if service else _authed_as_user(),
            patch.object(agents_route.db, "session", session),
            user_patch,
            app.test_client() as c,
        ):
            resp = c.get(f"/api/agents/{AGENT_ID}/sessions", headers=HEADERS)
        assert resp.status_code == 200, resp.get_data(as_text=True)
        # Both the page query and the count query carry the same filter.
        return [(str(call.args[0]), call.args[1]) for call in session.execute.call_args_list]

    def test_end_user_is_filtered_to_their_own_sessions(self):
        queries = self._list()
        assert len(queries) == 2
        for sql, params in queries:
            assert "s.user_id = :scoped_user_id" in sql
            assert params["scoped_user_id"] == USER_ID

    def test_end_user_filter_does_not_depend_on_having_an_id(self):
        """A missing id must filter to nothing, never drop the filter."""
        for sql, params in self._list(user_id=None):
            assert "s.user_id = :scoped_user_id" in sql
            assert params["scoped_user_id"] is None

    def test_service_role_sees_every_session(self):
        for sql, params in self._list(service=True):
            assert "scoped_user_id" not in sql
            assert "scoped_user_id" not in params


# ---------------------------------------------------------------------------
# /api/sessions/<id>/... — owner-only reads and deletes
# ---------------------------------------------------------------------------

SESSION_ROUTES = [
    ("GET", "/api/sessions/sess_a"),
    ("GET", "/api/sessions/sess_a/messages"),
    ("GET", "/api/sessions/sess_a/runs"),
    ("GET", "/api/sessions/sess_a/runs/run_1/retrieved-context"),
    ("DELETE", "/api/sessions/sess_a"),
]


class TestSessionRoutesOwnership:
    def _call(self, method, path, owner, *, service=False):
        app = _app_with(sessions_route.sessions_bp)
        with (
            _authed_as_service_role() if service else _authed_as_user(),
            patch.object(sessions_route.db, "session", MagicMock()),
            patch.object(sessions_route, "get_session_owner", return_value=owner) as lookup,
            patch.object(
                sessions_route, "get_session_by_id", return_value={"session_id": "sess_a"}
            ),
            patch.object(sessions_route, "get_chat_messages", return_value=[]),
            patch.object(sessions_route, "list_runs_for_session", return_value=[]),
            patch.object(sessions_route, "get_run_retrieved_context", return_value=[]),
            patch.object(sessions_route, "delete_session", return_value=True) as delete,
            app.test_client() as c,
        ):
            resp = c.open(path, method=method, headers=HEADERS)
        return resp, lookup, delete

    @pytest.mark.parametrize(("method", "path"), SESSION_ROUTES)
    def test_owner_gets_200(self, method, path):
        resp, lookup, _ = self._call(method, path, USER_ID)
        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert lookup.call_args.args[1] == "sess_a"

    @pytest.mark.parametrize(("method", "path"), SESSION_ROUTES)
    @pytest.mark.parametrize("owner", [OTHER_USER_ID, None])
    def test_other_users_and_ownerless_sessions_are_404(self, method, path, owner):
        resp, _, delete = self._call(method, path, owner)
        assert resp.status_code == 404
        delete.assert_not_called()

    @pytest.mark.parametrize(("method", "path"), SESSION_ROUTES)
    def test_service_role_skips_the_owner_lookup(self, method, path):
        resp, lookup, _ = self._call(method, path, OTHER_USER_ID, service=True)
        assert resp.status_code == 200, resp.get_data(as_text=True)
        lookup.assert_not_called()


# ---------------------------------------------------------------------------
# POST /api/agents/runs/<id>/approve — only the run's owner (or the backend)
# ---------------------------------------------------------------------------


class TestApproveRunOwnership:
    def _approve(self, run_id, *, service=False):
        app = _app_with(agents_route.agents_bp)
        with (
            _authed_as_service_role() if service else _authed_as_user(),
            app.test_client() as c,
        ):
            return c.post(
                f"/api/agents/runs/{run_id}/approve", headers=HEADERS, json={"approved": True}
            )

    def teardown_method(self):
        for run_id in ("run_mine", "run_theirs", "run_backend"):
            run_registry.unregister_run(run_id)

    def test_owner_may_approve(self):
        ctx = MagicMock()
        run_registry.register_run("run_mine", ctx, owner_user_id=USER_ID)
        assert self._approve("run_mine").status_code == 200
        ctx.set_approval_decision.assert_called_once()

    def test_other_user_gets_404_and_nothing_resumes(self):
        ctx = MagicMock()
        run_registry.register_run("run_theirs", ctx, owner_user_id=OTHER_USER_ID)
        assert self._approve("run_theirs").status_code == 404
        ctx.set_approval_decision.assert_not_called()

    def test_backend_started_run_is_not_approvable_by_an_end_user(self):
        ctx = MagicMock()
        run_registry.register_run("run_backend", ctx, owner_user_id=None)
        assert self._approve("run_backend").status_code == 404
        ctx.set_approval_decision.assert_not_called()

    def test_token_without_sub_cannot_approve_a_backend_run(self):
        """None == None must not read as "the owner"."""
        ctx = MagicMock()
        run_registry.register_run("run_backend", ctx, owner_user_id=None)
        app = _app_with(agents_route.agents_bp)
        with (
            patch(
                "agentic_project_service.auth.decode_jwt",
                return_value={"role": "authenticated", "aud": "authenticated"},
            ),
            app.test_client() as c,
        ):
            resp = c.post(
                "/api/agents/runs/run_backend/approve", headers=HEADERS, json={"approved": True}
            )
        assert resp.status_code == 401
        ctx.set_approval_decision.assert_not_called()

    def test_end_user_without_an_id_never_matches_an_ownerless_run(self):
        ctx = MagicMock()
        run_registry.register_run("run_backend", ctx, owner_user_id=None)
        with patch.object(agents_route, "get_current_user_id", return_value=None):
            assert self._approve("run_backend").status_code == 404
        ctx.set_approval_decision.assert_not_called()

    def test_service_role_may_approve_any_run(self):
        ctx = MagicMock()
        run_registry.register_run("run_theirs", ctx, owner_user_id=OTHER_USER_ID)
        assert self._approve("run_theirs", service=True).status_code == 200
        ctx.set_approval_decision.assert_called_once()


# ---------------------------------------------------------------------------
# GET /api/orchestrations/runs/<id> — previously had no ownership check at all
# ---------------------------------------------------------------------------


class TestGetOrchestrationRunOwnership:
    def _get(self, session_id, session_owner):
        app = _app_with(orchestrations_route.orchestrations_bp)
        run = MagicMock(session_id=session_id, run_id="orun_1")
        session_row = None if session_owner is ... else MagicMock(user_id=session_owner)
        with (
            _authed_as_user(),
            patch.object(orchestrations_route, "OrchestrationRunModel") as run_model,
            patch.object(orchestrations_route, "OrchestrationSessionModel") as session_model,
            patch.object(orchestrations_route, "AgentRun") as agent_run_model,
            app.test_client() as c,
        ):
            run_model.query.filter_by.return_value.first.return_value = run
            session_model.query.filter_by.return_value.first.return_value = session_row
            agent_run_model.query.filter_by.side_effect = AssertionError("must not load children")
            return c.get("/api/orchestrations/runs/orun_1", headers=HEADERS)

    def test_sessionless_run_is_hidden_from_end_users(self):
        assert self._get(None, ...).status_code == 404

    def test_ownerless_session_run_is_hidden(self):
        assert self._get("s-uuid", None).status_code == 404

    def test_other_users_run_is_hidden(self):
        assert self._get("s-uuid", OTHER_USER_ID).status_code == 404

    def test_owner_sees_their_run(self):
        app = _app_with(orchestrations_route.orchestrations_bp)
        run = MagicMock(session_id="s-uuid", run_id="orun_1", orchestration_id=None)
        for column in ("status", "content", "events", "usage", "model", "error"):
            setattr(run, column, None)
        run.started_at = run.completed_at = None
        with (
            _authed_as_user(),
            patch.object(orchestrations_route, "OrchestrationRunModel") as run_model,
            patch.object(orchestrations_route, "OrchestrationSessionModel") as session_model,
            patch.object(orchestrations_route, "AgentRun") as agent_run_model,
            patch.object(orchestrations_route, "_load_tool_calls_for_runs", return_value={}),
            app.test_client() as c,
        ):
            run_model.query.filter_by.return_value.first.return_value = run
            session_model.query.filter_by.return_value.first.return_value = MagicMock(
                user_id=USER_ID
            )
            agent_run_model.query.filter_by.return_value.all.return_value = []
            resp = c.get("/api/orchestrations/runs/orun_1", headers=HEADERS)
        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert resp.get_json()["run_id"] == "orun_1"

    def test_service_role_sees_any_run(self):
        """Including one in no session at all, which no end user could own."""
        app = _app_with(orchestrations_route.orchestrations_bp)
        run = MagicMock(session_id=None, run_id="orun_1", orchestration_id=None)
        for column in ("status", "content", "events", "usage", "model", "error"):
            setattr(run, column, None)
        run.started_at = run.completed_at = None
        with (
            _authed_as_service_role(),
            patch.object(orchestrations_route, "OrchestrationRunModel") as run_model,
            patch.object(orchestrations_route, "AgentRun") as agent_run_model,
            patch.object(orchestrations_route, "_load_tool_calls_for_runs", return_value={}),
            app.test_client() as c,
        ):
            run_model.query.filter_by.return_value.first.return_value = run
            agent_run_model.query.filter_by.return_value.all.return_value = []
            resp = c.get("/api/orchestrations/runs/orun_1", headers=HEADERS)
        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert resp.get_json()["run_id"] == "orun_1"


# ---------------------------------------------------------------------------
# GET /api/orchestrations/<id>/sessions/<session_id>/messages — owner-only
# ---------------------------------------------------------------------------


class TestOrchestrationSessionMessagesOwnership:
    def _get(self, session_owner):
        app = _app_with(orchestrations_route.orchestrations_bp)
        with (
            _authed_as_user(),
            patch.object(orchestrations_route, "OrchestrationSessionModel") as session_model,
            patch.object(orchestrations_route, "OrchestrationRunModel") as run_model,
            app.test_client() as c,
        ):
            session_model.query.filter_by.return_value.first.return_value = MagicMock(
                user_id=session_owner, id="s-uuid"
            )
            run_model.query.filter_by.side_effect = AssertionError("must not read the runs")
            return c.get(f"/api/orchestrations/{ORCH_ID}/sessions/sess_x/messages", headers=HEADERS)

    def test_ownerless_session_is_hidden_from_end_users(self):
        """A backend-started session belongs to no end user."""
        resp = self._get(None)
        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Session not found"}

    def test_other_users_session_is_hidden(self):
        assert self._get(OTHER_USER_ID).status_code == 404


# ---------------------------------------------------------------------------
# A session_id is a string of at most 255 characters, checked before any query
# ---------------------------------------------------------------------------

BAD_SESSION_IDS = [
    pytest.param(123, id="int"),
    pytest.param({"a": 1}, id="dict"),
    pytest.param(True, id="bool"),
    pytest.param(["sess_a"], id="list"),
    pytest.param(1.5, id="float"),
    pytest.param("s" * 256, id="too-long"),
]

RUN_PATHS = [
    f"/api/agents/{AGENT_ID}/run",
    f"/api/agents/{AGENT_ID}/run/stream",
    f"/api/orchestrations/{ORCH_ID}/run/stream",
]


def _untouchable_data_layer():
    """Patches that fail the test if a route reads anything."""
    touched = AssertionError("the route queried the database")
    session = MagicMock()
    session.execute.side_effect = touched
    session.get.side_effect = touched
    model = MagicMock()
    model.query.filter_by.side_effect = touched
    return (
        patch.object(agents_route.db, "session", session),
        patch.object(orchestrations_route, "OrchestrationSessionModel", model),
        patch.object(sessions_route, "get_session_owner", side_effect=touched),
    )


class TestSessionIdShape:
    @pytest.mark.parametrize("service", [False, True], ids=["end-user", "service-role"])
    @pytest.mark.parametrize("path", RUN_PATHS)
    @pytest.mark.parametrize("session_id", BAD_SESSION_IDS)
    def test_run_routes_refuse_it_with_400(self, path, session_id, service):
        app = _app_with(agents_route.agents_bp, orchestrations_route.orchestrations_bp)
        db_patch, model_patch, _ = _untouchable_data_layer()
        with (
            _authed_as_service_role() if service else _authed_as_user(),
            db_patch,
            model_patch,
            patch.object(agents_route.billing, "check_balance") as agent_balance,
            patch.object(orchestrations_route.billing, "check_balance") as orch_balance,
            app.test_client() as c,
        ):
            resp = c.post(path, headers=HEADERS, json={"message": "hi", "session_id": session_id})
        assert resp.status_code == 400, resp.get_data(as_text=True)
        assert "session_id" in resp.get_json()["error"]
        agent_balance.assert_not_called()
        orch_balance.assert_not_called()

    @pytest.mark.parametrize("path", RUN_PATHS)
    def test_255_characters_is_still_a_session_id(self, path):
        """Positive control: the longest id the column holds gets past the check."""
        app = _app_with(agents_route.agents_bp, orchestrations_route.orchestrations_bp)
        app.testing = True  # let the sentinel propagate
        with (
            _authed_as_user(),
            patch.object(agents_route.db, "session", MagicMock()),
            patch.object(
                agents_route.billing, "check_balance", side_effect=RuntimeError("past the check")
            ),
            patch.object(
                orchestrations_route.billing,
                "check_balance",
                side_effect=RuntimeError("past the check"),
            ),
            patch.object(orchestrations_route, "_end_user_may_run_in_session", return_value=True),
            app.test_client() as c,
        ):
            with pytest.raises(RuntimeError, match="past the check"):
                c.post(path, headers=HEADERS, json={"message": "hi", "session_id": "s" * 255})

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            *[(m, p.replace("sess_a", "s" * 256)) for m, p in SESSION_ROUTES],
            ("GET", f"/api/orchestrations/{ORCH_ID}/sessions/{'s' * 256}/messages"),
            ("DELETE", f"/api/agents/{AGENT_ID}/sessions/{'s' * 256}"),
        ],
        ids=[
            *[f"{m}-{p.replace(chr(47), '_')}" for m, p in SESSION_ROUTES],
            "GET-orchestration-session-messages",
            "DELETE-agent-session",
        ],
    )
    @pytest.mark.parametrize("service", [False, True], ids=["end-user", "service-role"])
    def test_session_routes_refuse_an_over_long_id_with_400(self, method, path, service):
        app = _app_with(
            agents_route.agents_bp,
            orchestrations_route.orchestrations_bp,
            sessions_route.sessions_bp,
        )
        db_patch, model_patch, owner_patch = _untouchable_data_layer()
        with (
            _authed_as_service_role() if service else _authed_as_user(),
            db_patch,
            model_patch,
            owner_patch,
            app.test_client() as c,
        ):
            resp = c.open(path, method=method, headers=HEADERS)
        assert resp.status_code == 400, (method, path, resp.get_data(as_text=True))
        assert "session_id" in resp.get_json()["error"]
