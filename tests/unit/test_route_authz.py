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
import pkgutil
from unittest.mock import MagicMock, patch

import pytest
from flask import Blueprint, Flask, jsonify

from agentic_project_service import auth
from agentic_project_service import routes as routes_pkg
from agentic_project_service.routes import agents as agents_route
from agentic_project_service.routes import orchestrations as orchestrations_route
from agentic_project_service.services import run_registry

USER_ID = "11111111-1111-4111-8111-111111111111"
OTHER_USER_ID = "22222222-2222-4222-8222-222222222222"
AGENT_ID = "3f9a1c2e-5b7d-4e11-9a3c-8d2f6b4e1a70"
ORCH_ID = "4a8b2d3f-6c8e-4f22-8b4d-9e3a7c5f2b81"
KB_ID = "5b9c3e4a-7d9f-4a33-9c5e-af4b8d6a3c92"


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

# Every route an end user's own JWT may call. Adding to this set is a security
# decision: the route must scope everything it reads or changes to the caller.
END_USER_ENDPOINTS = {
    # Agent conversations. Sessions and runs are owned by the calling user.
    "agents.list_sessions",
    "agents.create_session_for_agent",
    "agents.delete_session_for_agent",
    "agents.run_agent",
    "agents.run_agent_stream",
    "agents.get_agent_run",
    "agents.approve_run",
    # Session reads/deletes, owner-only.
    "sessions.get_session",
    "sessions.get_messages",
    "sessions.get_runs",
    "sessions.get_run_retrieved_context_route",
    "sessions.delete_session_route",
    # Orchestration conversations, owner-only.
    "orchestrations.list_orchestration_sessions",
    "orchestrations.get_orchestration_session_messages",
    "orchestrations.run_orchestration_stream",
    "orchestrations.get_orchestration_run",
}

# Routes that take no bearer at all because they authenticate some other way.
UNAUTHENTICATED_ENDPOINTS = {
    # Inbound webhooks: verified by the per-workflow webhook secret.
    "webhooks.trigger_webhook",
    # Docs search for the in-project copilot: verified by X-Docs-Search-Token.
    "internal_docs.docs_search",
}


def _api_endpoints(app: Flask) -> dict[str, object]:
    return {
        rule.endpoint: app.view_functions[rule.endpoint]
        for rule in app.url_map.iter_rules()
        if rule.rule.startswith("/api/") and rule.endpoint != "static"
    }


class TestRouteInventory:
    def test_every_api_route_declares_who_may_call_it(self):
        endpoints = _api_endpoints(_full_app())
        undeclared = sorted(
            name
            for name, view in endpoints.items()
            if getattr(view, "auth_mode", None) is None and name not in UNAUTHENTICATED_ENDPOINTS
        )
        assert undeclared == []

    def test_end_user_routes_are_exactly_the_reviewed_list(self):
        endpoints = _api_endpoints(_full_app())
        end_user = {
            name for name, view in endpoints.items() if getattr(view, "auth_mode", None) == "user"
        }
        assert end_user == END_USER_ENDPOINTS

    def test_unauthenticated_list_names_real_routes(self):
        """A stale entry here would silently exempt a future route of that name."""
        endpoints = _api_endpoints(_full_app())
        assert UNAUTHENTICATED_ENDPOINTS <= set(endpoints)


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


# ---------------------------------------------------------------------------
# GET /api/agents/runs/<id> — end users see only runs in their own sessions
# ---------------------------------------------------------------------------


def _agent_run(session_id):
    run = MagicMock()
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
