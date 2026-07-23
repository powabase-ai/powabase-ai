"""Tests for hooks CRUD endpoints (agents and orchestrations)."""

import pytest


class TestHookCRUD:
    def test_add_hook(self, client, mock_auth, auth_headers, test_agent):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "rule",
                "matcher": "database_query",
                "config": {
                    "condition": "query CONTAINS 'DROP'",
                    "action": "deny",
                    "message": "No DROP",
                },
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["event"] == "PreToolUse"
        assert data["type"] == "rule"
        assert data["matcher"] == "database_query"
        assert data["config"]["action"] == "deny"
        assert data["enabled"] is True
        assert "id" in data

    def test_list_hooks(self, client, mock_auth, auth_headers, test_agent):
        client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "rule",
                "config": {"condition": "x CONTAINS 'y'"},
            },
            headers=auth_headers,
        )
        resp = client.get(
            f"/api/agents/{test_agent['id']}/hooks",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["hooks"]) == 1
        assert data["hooks"][0]["event"] == "PreToolUse"

    def test_delete_hook(self, client, mock_auth, auth_headers, test_agent):
        create = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "rule",
                "config": {"condition": "x CONTAINS 'y'", "action": "deny"},
            },
            headers=auth_headers,
        )
        hook_id = create.get_json()["id"]
        resp = client.delete(
            f"/api/agents/{test_agent['id']}/hooks/{hook_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["deleted"] is True

        # Confirm it's gone
        list_resp = client.get(
            f"/api/agents/{test_agent['id']}/hooks",
            headers=auth_headers,
        )
        assert len(list_resp.get_json()["hooks"]) == 0

    def test_missing_required_fields(self, client, mock_auth, auth_headers, test_agent):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "type": "rule",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_missing_type(self, client, mock_auth, auth_headers, test_agent):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "config": {},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_missing_config(self, client, mock_auth, auth_headers, test_agent):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "rule",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_list_hooks_empty(self, client, mock_auth, auth_headers, test_agent):
        resp = client.get(
            f"/api/agents/{test_agent['id']}/hooks",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["hooks"] == []

    def test_delete_nonexistent_hook(self, client, mock_auth, auth_headers, test_agent):
        import uuid

        fake_id = str(uuid.uuid4())
        resp = client.delete(
            f"/api/agents/{test_agent['id']}/hooks/{fake_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_hook_position_ordering(self, client, mock_auth, auth_headers, test_agent):
        for pos in [2, 0, 1]:
            resp = client.post(
                f"/api/agents/{test_agent['id']}/hooks",
                json={
                    "event": "PreToolUse",
                    "type": "rule",
                    "config": {"condition": "x CONTAINS 'y'", "action": "deny"},
                    "position": pos,
                },
                headers=auth_headers,
            )
            assert resp.status_code == 201, resp.get_json()
        resp = client.get(
            f"/api/agents/{test_agent['id']}/hooks",
            headers=auth_headers,
        )
        positions = [h["position"] for h in resp.get_json()["hooks"]]
        assert positions == sorted(positions)


class TestOrchestrationHookCRUD:
    @pytest.fixture
    def test_orchestration(self, client, mock_auth, auth_headers):
        resp = client.post(
            "/api/orchestrations",
            json={"name": "Test Orchestration"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        return resp.get_json()

    def test_add_hook_to_orchestration(self, client, mock_auth, auth_headers, test_orchestration):
        resp = client.post(
            f"/api/orchestrations/{test_orchestration['id']}/hooks",
            json={
                "event": "PreResponse",
                "type": "rule",
                "config": {"condition": "x CONTAINS 'y'", "action": "deny"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["event"] == "PreResponse"
        assert data["type"] == "rule"
        assert "id" in data

    def test_list_orchestration_hooks(self, client, mock_auth, auth_headers, test_orchestration):
        client.post(
            f"/api/orchestrations/{test_orchestration['id']}/hooks",
            json={
                "event": "PreResponse",
                "type": "rule",
                "config": {"condition": "x CONTAINS 'y'", "action": "deny"},
            },
            headers=auth_headers,
        )
        resp = client.get(
            f"/api/orchestrations/{test_orchestration['id']}/hooks",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["hooks"]) == 1

    def test_orchestration_hook_missing_fields(
        self, client, mock_auth, auth_headers, test_orchestration
    ):
        resp = client.post(
            f"/api/orchestrations/{test_orchestration['id']}/hooks",
            json={"type": "rule"},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_list_hooks_orchestration_not_found(self, client, mock_auth, auth_headers):
        import uuid

        fake_id = str(uuid.uuid4())
        resp = client.get(
            f"/api/orchestrations/{fake_id}/hooks",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_delete_orchestration_hook(self, client, mock_auth, auth_headers, test_orchestration):
        create = client.post(
            f"/api/orchestrations/{test_orchestration['id']}/hooks",
            json={
                "event": "PreResponse",
                "type": "rule",
                "config": {"condition": "x CONTAINS 'y'", "action": "deny"},
            },
            headers=auth_headers,
        )
        hook_id = create.get_json()["id"]
        resp = client.delete(
            f"/api/orchestrations/{test_orchestration['id']}/hooks/{hook_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["deleted"] is True

        # Confirm it's gone from the list
        list_resp = client.get(
            f"/api/orchestrations/{test_orchestration['id']}/hooks",
            headers=auth_headers,
        )
        assert len(list_resp.get_json()["hooks"]) == 0

    def test_delete_nonexistent_orchestration_hook(
        self, client, mock_auth, auth_headers, test_orchestration
    ):
        import uuid

        fake_id = str(uuid.uuid4())
        resp = client.delete(
            f"/api/orchestrations/{test_orchestration['id']}/hooks/{fake_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_delete_orchestration_hook_bad_uuid(
        self, client, mock_auth, auth_headers, test_orchestration
    ):
        resp = client.delete(
            f"/api/orchestrations/{test_orchestration['id']}/hooks/not-a-uuid",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestHookLoaderIdentifiers:
    def test_loaded_hook_config_carries_id_and_position(
        self, client, mock_auth, auth_headers, test_agent
    ):
        create = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreResponse",
                "type": "http",
                "config": {"url": "https://example.com/hook"},
                "position": 3,
            },
            headers=auth_headers,
        )
        assert create.status_code == 201
        hook_id = create.get_json()["id"]

        from agentic_project_service.services.hook_loader import load_hooks_for_agent

        configs = load_hooks_for_agent(test_agent["id"])
        assert len(configs) == 1
        assert configs[0].id == hook_id
        assert configs[0].position == 3


class TestOrchestrationHookStrategyGate:
    def _make_orch(self, client, auth_headers, strategy):
        resp = client.post(
            "/api/orchestrations",
            json={"name": f"orch-{strategy}", "strategy": strategy},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        return resp.get_json()["id"]

    def test_supervisor_orchestration_accepts_hook(self, client, mock_auth, auth_headers):
        orch_id = self._make_orch(client, auth_headers, "supervisor")
        resp = client.post(
            f"/api/orchestrations/{orch_id}/hooks",
            json={
                "event": "PreResponse",
                "type": "http",
                "config": {"url": "https://example.com/hook"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201

    @pytest.mark.parametrize("strategy", ["sequential", "parallel"])
    def test_non_supervisor_orchestration_rejects_hook(
        self, client, mock_auth, auth_headers, strategy
    ):
        orch_id = self._make_orch(client, auth_headers, strategy)
        resp = client.post(
            f"/api/orchestrations/{orch_id}/hooks",
            json={
                "event": "PreResponse",
                "type": "http",
                "config": {"url": "https://example.com/hook"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "supervisor" in resp.get_json()["error"].lower()

    def test_strategy_change_away_from_supervisor_blocked_while_hooks_exist(
        self, client, mock_auth, auth_headers
    ):
        # C2: the supervisor-only guard must not be bypassable by editing the
        # orchestration's strategy after a (potentially blocking) hook is set.
        orch_id = self._make_orch(client, auth_headers, "supervisor")
        assert (
            client.post(
                f"/api/orchestrations/{orch_id}/hooks",
                json={
                    "event": "PreResponse",
                    "type": "http",
                    "config": {"url": "https://example.com/hook"},
                },
                headers=auth_headers,
            ).status_code
            == 201
        )

        resp = client.put(
            f"/api/orchestrations/{orch_id}",
            json={"strategy": "sequential"},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "supervisor" in resp.get_json()["error"].lower()

    @pytest.mark.parametrize("bad", [{"type": "webhook"}, {"event": "preResponse"}])
    def test_unknown_event_or_type_rejected_at_create(self, client, mock_auth, auth_headers, bad):
        # C3b: typo'd event/type must 400, not land as a silent dead hook.
        orch_id = self._make_orch(client, auth_headers, "supervisor")
        body = {
            "event": "PreResponse",
            "type": "http",
            "config": {"url": "https://example.com/hook"},
        }
        body.update(bad)
        resp = client.post(f"/api/orchestrations/{orch_id}/hooks", json=body, headers=auth_headers)
        assert resp.status_code == 400


class TestHookCrudHardening:
    """Round-2 review: #1 approval-on-orch, #3 config-is-dict, #5 null strategy, #6 tiebreak."""

    def _make_orch(self, client, auth_headers, strategy="supervisor"):
        resp = client.post(
            "/api/orchestrations",
            json={"name": f"orch-{strategy}", "strategy": strategy},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        return resp.get_json()["id"]

    def test_approval_hook_rejected_on_orchestration(self, client, mock_auth, auth_headers):
        # #1: orchestrations have no approve endpoint / run registry, so an
        # approval hook would hang then hard-block. Reject at create.
        orch_id = self._make_orch(client, auth_headers)
        resp = client.post(
            f"/api/orchestrations/{orch_id}/hooks",
            json={"event": "PreToolUse", "type": "approval", "config": {}},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "approval" in resp.get_json()["error"].lower()

    def test_non_dict_config_rejected_orchestration(self, client, mock_auth, auth_headers):
        # #3: a non-dict config (e.g. a string) must 400, not create a hook that
        # raises at dispatch and (for a blocking hook) silently fails open.
        orch_id = self._make_orch(client, auth_headers)
        resp = client.post(
            f"/api/orchestrations/{orch_id}/hooks",
            json={"event": "PreResponse", "type": "rule", "config": "deny-secrets"},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_non_dict_config_rejected_agent(self, client, mock_auth, auth_headers, test_agent):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={"event": "PreToolUse", "type": "rule", "config": ["not", "a", "dict"]},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_put_null_strategy_blocked_while_hooks_exist(self, client, mock_auth, auth_headers):
        # #5: the C2 guard must also catch a null/invalid strategy, not just a
        # different-but-valid one.
        orch_id = self._make_orch(client, auth_headers)
        assert (
            client.post(
                f"/api/orchestrations/{orch_id}/hooks",
                json={
                    "event": "PreResponse",
                    "type": "http",
                    "config": {"url": "https://example.com/hook"},
                },
                headers=auth_headers,
            ).status_code
            == 201
        )
        resp = client.put(
            f"/api/orchestrations/{orch_id}",
            json={"strategy": None},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_loader_tiebreaks_equal_position_by_created_at(
        self, client, mock_auth, auth_headers, app
    ):
        # #6: equal position must fall back to created_at, not DB-physical order.
        from sqlalchemy import text

        from agentic_project_service.db import db
        from agentic_project_service.services.hook_loader import (
            load_hooks_for_orchestration,
        )

        orch_id = self._make_orch(client, auth_headers)
        older = client.post(
            f"/api/orchestrations/{orch_id}/hooks",
            json={
                "event": "PreResponse",
                "type": "http",
                "config": {"url": "https://a.example/hook"},
                "position": 0,
            },
            headers=auth_headers,
        ).get_json()["id"]
        newer = client.post(
            f"/api/orchestrations/{orch_id}/hooks",
            json={
                "event": "PreResponse",
                "type": "http",
                "config": {"url": "https://b.example/hook"},
                "position": 0,
            },
            headers=auth_headers,
        ).get_json()["id"]
        # Back-date the row inserted SECOND so created_at order ([newer, older])
        # disagrees with insertion order ([older, newer]). Only a created_at
        # tiebreak can produce the expected list.
        #
        # Back-dating the second row rather than forward-dating the first is
        # deliberate: an UPDATE can relocate the updated tuple later in a heap
        # scan, so forward-dating the first row yields the expected order even
        # with the ORDER BY removed — the test would pass while broken.
        with app.app_context():
            db.session.execute(
                text(
                    "UPDATE \"ai\".hooks SET created_at = now() - interval '1 hour' WHERE id = :i"
                ),
                {"i": newer},
            )
            db.session.commit()
        configs = load_hooks_for_orchestration(orch_id)
        assert [c.id for c in configs] == [newer, older]


class TestRound3Hardening:
    """Round-3: F1 (approval filtered at load), agent-loader tiebreak (#coverage)."""

    def _make_orch(self, client, auth_headers):
        resp = client.post(
            "/api/orchestrations",
            json={"name": "orch-r3", "strategy": "supervisor"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        return resp.get_json()["id"]

    def test_approval_row_filtered_from_orchestration_loader(
        self, client, mock_auth, auth_headers, app
    ):
        # F1: a direct-SQL approval row (create is blocked, but defend-in-depth)
        # must be excluded at load so it can't hang the run on the approval wait.
        import uuid

        from sqlalchemy import text

        from agentic_project_service.db import db
        from agentic_project_service.services.hook_loader import (
            load_hooks_for_orchestration,
        )

        orch_id = self._make_orch(client, auth_headers)
        # A normal http hook (via API) + a smuggled approval row (via SQL).
        client.post(
            f"/api/orchestrations/{orch_id}/hooks",
            json={
                "event": "PreResponse",
                "type": "http",
                "config": {"url": "https://ok.example/hook"},
            },
            headers=auth_headers,
        )
        with app.app_context():
            db.session.execute(
                text(
                    'INSERT INTO "ai".hooks (id, orchestration_id, event, type, config, enabled, position) '
                    "VALUES (:id, :oid, 'PreToolUse', 'approval', '{}'::jsonb, true, 0)"
                ),
                {"id": str(uuid.uuid4()), "oid": orch_id},
            )
            db.session.commit()
        configs = load_hooks_for_orchestration(orch_id)
        assert all(c.type != "approval" for c in configs)
        assert len(configs) == 1  # only the http hook

    def test_agent_loader_tiebreaks_equal_position_by_created_at(
        self, client, mock_auth, auth_headers, test_agent, app
    ):
        from sqlalchemy import text

        from agentic_project_service.db import db
        from agentic_project_service.services.hook_loader import load_hooks_for_agent

        older = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "http",
                "config": {"url": "https://a.example/hook"},
                "position": 0,
            },
            headers=auth_headers,
        ).get_json()["id"]
        newer = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "http",
                "config": {"url": "https://a.example/hook"},
                "position": 0,
            },
            headers=auth_headers,
        ).get_json()["id"]
        # Back-date the second-inserted row — see the note in the orchestration
        # loader tiebreak test for why the direction matters.
        with app.app_context():
            db.session.execute(
                text(
                    "UPDATE \"ai\".hooks SET created_at = now() - interval '1 hour' WHERE id = :i"
                ),
                {"i": newer},
            )
            db.session.commit()
        configs = load_hooks_for_agent(test_agent["id"])
        assert [c.id for c in configs] == [newer, older]


class TestRound4Hardening:
    """Round-4: N1a (blocking-only type on a non-blocking event), N2 (list
    endpoint tiebreak), N3 (agent-route unknown event/type coverage)."""

    def _make_orch(self, client, auth_headers):
        resp = client.post(
            "/api/orchestrations",
            json={"name": "orch-r4", "strategy": "supervisor"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        return resp.get_json()["id"]

    @pytest.mark.parametrize("event", ["PostToolUse", "OnRunComplete"])
    @pytest.mark.parametrize("hook_type", ["rule", "approval"])
    def test_blocking_only_type_on_non_blocking_event_rejected_agent(
        self, client, mock_auth, auth_headers, test_agent, event, hook_type
    ):
        # N1a: a rule/approval hook can only ever block; PostToolUse/OnRunComplete
        # cannot block per contract → the config is dead. Reject at create.
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": event,
                "type": hook_type,
                "config": {"condition": "x CONTAINS 'y'", "action": "deny"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_blocking_only_type_on_non_blocking_event_rejected_orchestration(
        self, client, mock_auth, auth_headers
    ):
        orch_id = self._make_orch(client, auth_headers)
        resp = client.post(
            f"/api/orchestrations/{orch_id}/hooks",
            json={
                "event": "PostToolUse",
                "type": "rule",
                "config": {"condition": "x CONTAINS 'y'", "action": "deny"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_http_still_allowed_on_non_blocking_events(
        self, client, mock_auth, auth_headers, test_agent
    ):
        # Counterpart: http on PostToolUse (modified_output) and OnRunComplete
        # (fire-and-forget notification) remain legitimate.
        for event in ("PostToolUse", "OnRunComplete"):
            resp = client.post(
                f"/api/agents/{test_agent['id']}/hooks",
                json={
                    "event": event,
                    "type": "http",
                    "config": {"url": "https://example.com/hook"},
                },
                headers=auth_headers,
            )
            assert resp.status_code == 201, event

    @pytest.mark.parametrize("bad", [{"type": "webhook"}, {"event": "preResponse"}])
    def test_unknown_event_or_type_rejected_at_create_agent(
        self, client, mock_auth, auth_headers, test_agent, bad
    ):
        # N3: the agent route's HOOK_EVENTS/HOOK_TYPES guards were untested.
        body = {
            "event": "PreToolUse",
            "type": "rule",
            "config": {"condition": "x CONTAINS 'y'", "action": "deny"},
        }
        body.update(bad)
        resp = client.post(f"/api/agents/{test_agent['id']}/hooks", json=body, headers=auth_headers)
        assert resp.status_code == 400

    def test_list_endpoint_tiebreaks_equal_position_by_created_at(
        self, client, mock_auth, auth_headers, test_agent, app
    ):
        # N2: the list endpoint must order like the loader, or the management UI
        # shows the policy chain in a different order than it fires.
        from sqlalchemy import text

        from agentic_project_service.db import db

        older = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "http",
                "config": {"url": "https://a.example/hook"},
                "position": 0,
            },
            headers=auth_headers,
        ).get_json()["id"]
        newer = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "http",
                "config": {"url": "https://a.example/hook"},
                "position": 0,
            },
            headers=auth_headers,
        ).get_json()["id"]
        # Back-date the second-inserted row — see the note in the orchestration
        # loader tiebreak test for why the direction matters.
        with app.app_context():
            db.session.execute(
                text(
                    "UPDATE \"ai\".hooks SET created_at = now() - interval '1 hour' WHERE id = :i"
                ),
                {"i": newer},
            )
            db.session.commit()
        listed = client.get(
            f"/api/agents/{test_agent['id']}/hooks", headers=auth_headers
        ).get_json()["hooks"]
        assert [h["id"] for h in listed] == [newer, older]


class TestRound5Hardening:
    """Round-5: R5-I2 (dead rule config), R5-I5 (matcher on non-tool event),
    M1 (a disabled hook must not wedge a strategy change)."""

    def _make_orch(self, client, auth_headers, name="orch-r5"):
        resp = client.post(
            "/api/orchestrations",
            json={"name": name, "strategy": "supervisor"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        return resp.get_json()["id"]

    # --- R5-I2: a rule hook that can never deny -------------------------------

    @pytest.mark.parametrize(
        "config",
        [{}, {"rules": []}, {"conditon": "typo CONTAINS 'x'", "action": "deny"}],
        ids=["empty", "empty-rules", "typo'd-condition"],
    )
    def test_dead_rule_config_rejected_agent(
        self, client, mock_auth, auth_headers, test_agent, config
    ):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={"event": "PreToolUse", "type": "rule", "config": config},
            headers=auth_headers,
        )
        assert resp.status_code == 400, (
            "A rule hook with no evaluable rules allows everything while looking "
            "like an active gate; it must not be persisted."
        )

    def test_dead_rule_config_rejected_orchestration(self, client, mock_auth, auth_headers):
        orch_id = self._make_orch(client, auth_headers, "orch-r5-rule")
        resp = client.post(
            f"/api/orchestrations/{orch_id}/hooks",
            json={"event": "PreToolUse", "type": "rule", "config": {}},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_valid_rule_config_still_accepted(self, client, mock_auth, auth_headers, test_agent):
        for config in (
            {"condition": "q CONTAINS 'DROP'", "action": "deny"},
            {"rules": [{"condition": "q CONTAINS 'DROP'", "action": "deny"}]},
        ):
            resp = client.post(
                f"/api/agents/{test_agent['id']}/hooks",
                json={"event": "PreToolUse", "type": "rule", "config": config},
                headers=auth_headers,
            )
            assert resp.status_code == 201, resp.get_json()

    # --- R5-I5: matcher only means something on tool-scoped events ------------

    @pytest.mark.parametrize("event", ["OnRunStart", "PreResponse", "OnRunComplete"])
    def test_matcher_on_non_tool_event_rejected_agent(
        self, client, mock_auth, auth_headers, test_agent, event
    ):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": event,
                "type": "http",
                "config": {"url": "https://a.example/hook"},
                "matcher": "database_query",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400, (
            f"{event} dispatches with tool_name=''; a matcher can never match, "
            "so the hook would never fire and never even be audited."
        )

    def test_matcher_on_non_tool_event_rejected_orchestration(
        self, client, mock_auth, auth_headers
    ):
        orch_id = self._make_orch(client, auth_headers, "orch-r5-matcher")
        resp = client.post(
            f"/api/orchestrations/{orch_id}/hooks",
            json={
                "event": "PreResponse",
                "type": "http",
                "config": {"url": "https://a.example/hook"},
                "matcher": "database_query",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    @pytest.mark.parametrize("event", ["PreToolUse", "PostToolUse", "OnDelegation"])
    def test_matcher_on_tool_event_accepted(
        self, client, mock_auth, auth_headers, test_agent, event
    ):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": event,
                "type": "http",
                "config": {"url": "https://a.example/hook"},
                "matcher": "database_query",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201, resp.get_json()

    # --- M1: a disabled hook must not permanently wedge the strategy ---------

    def test_disabled_hook_does_not_block_strategy_change(self, client, mock_auth, auth_headers):
        orch_id = self._make_orch(client, auth_headers, "orch-r5-disabled")
        resp = client.post(
            f"/api/orchestrations/{orch_id}/hooks",
            json={
                "event": "PreResponse",
                "type": "http",
                "config": {"url": "https://a.example/hook"},
                "enabled": False,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        resp = client.put(
            f"/api/orchestrations/{orch_id}",
            json={"strategy": "sequential"},
            headers=auth_headers,
        )
        assert resp.status_code == 200, (
            "A disabled hook enforces nothing, so it must not permanently block "
            "a strategy change — the error would tell the user to remove hooks "
            "they already consider off."
        )

    def test_enabled_hook_still_blocks_strategy_change(self, client, mock_auth, auth_headers):
        """Control: the round-2 guard must still hold for enabled hooks."""
        orch_id = self._make_orch(client, auth_headers, "orch-r5-enabled")
        resp = client.post(
            f"/api/orchestrations/{orch_id}/hooks",
            json={
                "event": "PreResponse",
                "type": "http",
                "config": {"url": "https://a.example/hook"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        resp = client.put(
            f"/api/orchestrations/{orch_id}",
            json={"strategy": "sequential"},
            headers=auth_headers,
        )
        assert resp.status_code == 400


class TestRound6Hardening:
    """Round-6: R6-I4 (a typo'd condition VALUE allows everything)."""

    def _make_orch(self, client, auth_headers, name="orch-r6"):
        resp = client.post(
            "/api/orchestrations",
            json={"name": name, "strategy": "supervisor"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        return resp.get_json()["id"]

    @pytest.mark.parametrize(
        "condition",
        [
            "input CONTAIN secret",  # operator typo
            "input contains secret",  # wrong case
            "q MATCHES (unclosed",  # uncompilable regex
            "q IN [a, b]",  # not valid JSON
            "no_operator_at_all",  # unparseable
        ],
    )
    def test_unparseable_condition_rejected_agent(
        self, client, mock_auth, auth_headers, test_agent, condition
    ):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "rule",
                "config": {"condition": condition, "action": "deny"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400, (
            f"{condition!r} cannot be evaluated, so the gate denies nothing "
            "while reporting a clean pass — indistinguishable from a rule that "
            "genuinely allowed the input."
        )

    def test_unparseable_condition_rejected_orchestration(self, client, mock_auth, auth_headers):
        orch_id = self._make_orch(client, auth_headers, "orch-r6-cond")
        resp = client.post(
            f"/api/orchestrations/{orch_id}/hooks",
            json={
                "event": "PreToolUse",
                "type": "rule",
                "config": {"condition": "input CONTAIN secret", "action": "deny"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_unparseable_condition_in_rules_list_rejected(
        self, client, mock_auth, auth_headers, test_agent
    ):
        """The nested `rules` form must be validated too, not just the flat one."""
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "rule",
                "config": {
                    "rules": [
                        {"condition": "q CONTAINS 'DROP'", "action": "deny"},
                        {"condition": "q CONTAIN 'DELETE'", "action": "deny"},
                    ]
                },
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    @pytest.mark.parametrize(
        "condition",
        [
            "q CONTAINS 'DROP'",
            "q NOT CONTAINS safe",
            "q STARTS_WITH SELECT",
            "q MATCHES ^SELECT.*",
            "q IN ['a', 'b']",
        ],
    )
    def test_valid_conditions_still_accepted(
        self, client, mock_auth, auth_headers, test_agent, condition
    ):
        resp = client.post(
            f"/api/agents/{test_agent['id']}/hooks",
            json={
                "event": "PreToolUse",
                "type": "rule",
                "config": {"condition": condition, "action": "deny"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201, resp.get_json()

    @pytest.mark.parametrize(
        "raw_body", ["null", "[]", "5", '"a string"'], ids=["null", "list", "number", "string"]
    )
    def test_non_object_update_body_is_400_not_500(self, client, mock_auth, auth_headers, raw_body):
        """M2 (corrected): `get_json()` returns None for a JSON `null` body, and
        non-object JSON yields a non-dict. Either way the strategy guard — the
        first statement to dereference `data` — raised, producing a 500."""
        orch_id = self._make_orch(client, auth_headers, "orch-r6-body")
        resp = client.put(
            f"/api/orchestrations/{orch_id}",
            data=raw_body,
            content_type="application/json",
            headers=auth_headers,
        )
        assert resp.status_code == 400, (
            f"body {raw_body!r} should be a client error, got {resp.status_code}"
        )
