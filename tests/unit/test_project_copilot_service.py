"""Unit tests for the Project Copilot service (search_docs + tools)."""

import json

from agentic_project_service.services import project_copilot as pc


def test_search_docs_unconfigured_returns_graceful_message(monkeypatch):
    monkeypatch.delenv("DOCS_SEARCH_URL", raising=False)
    monkeypatch.delenv("DOCS_SEARCH_TOKEN", raising=False)
    out = pc.search_docs("how do I connect?")
    assert "not configured" in out.lower()


def test_search_docs_formats_results_and_sends_token(monkeypatch, mocker):
    monkeypatch.setenv("DOCS_SEARCH_URL", "http://docs-svc/api/internal/docs/search")
    monkeypatch.setenv("DOCS_SEARCH_TOKEN", "tok")

    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "results": [
                    {
                        "title": "Auth connection",
                        "url": "https://docs.powabase.ai/guides/auth-connection",
                        "text": "Copy your connection string...",
                    }
                ]
            }

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return _Resp()

    mocker.patch.object(pc.httpx, "post", side_effect=_fake_post)

    out = pc.search_docs("connect coding agent", top_k=5)
    assert captured["headers"]["X-Docs-Search-Token"] == "tok"
    assert captured["body"] == {"query": "connect coding agent", "top_k": 5}
    assert "Auth connection" in out
    assert "https://docs.powabase.ai/guides/auth-connection" in out
    assert "connection string" in out


def test_search_docs_handles_non_200(monkeypatch, mocker):
    monkeypatch.setenv("DOCS_SEARCH_URL", "http://docs-svc/x")
    monkeypatch.setenv("DOCS_SEARCH_TOKEN", "tok")

    class _Resp:
        status_code = 503

        def json(self):
            return {}

    mocker.patch.object(pc.httpx, "post", return_value=_Resp())
    out = pc.search_docs("x")
    assert "unavailable" in out.lower()


def test_search_docs_maps_kb_not_ready_to_degraded_sentinel(monkeypatch, mocker):
    """When internal_docs signals kb_not_ready (empty/not-yet-indexed KB), search_docs
    must return the _DOCS_NOT_READY degraded sentinel — not the plain "no relevant
    documentation" message — so the notice-accumulator fires the grounding-degraded
    notice instead of silently answering ungrounded."""
    monkeypatch.setenv("DOCS_SEARCH_URL", "http://docs-svc/x")
    monkeypatch.setenv("DOCS_SEARCH_TOKEN", "tok")

    class _Resp:
        status_code = 200

        def json(self):
            return {"results": [], "kb_not_ready": True}

    mocker.patch.object(pc.httpx, "post", return_value=_Resp())
    out = pc.search_docs("x")
    assert out == pc._DOCS_NOT_READY
    assert pc._DOCS_NOT_READY in pc._DEGRADED_DOCS_RESULTS


def test_search_docs_genuine_no_match_stays_non_degraded(monkeypatch, mocker):
    """A real search that legitimately matched nothing (no kb_not_ready flag) must
    keep returning the plain "no relevant documentation" message — not a degraded
    sentinel — so no false "answered without docs" notice fires."""
    monkeypatch.setenv("DOCS_SEARCH_URL", "http://docs-svc/x")
    monkeypatch.setenv("DOCS_SEARCH_TOKEN", "tok")

    class _Resp:
        status_code = 200

        def json(self):
            return {"results": []}

    mocker.patch.object(pc.httpx, "post", return_value=_Resp())
    out = pc.search_docs("x")
    assert out == "No relevant documentation was found for that query."
    assert out not in pc._DEGRADED_DOCS_RESULTS


def test_play_guide_sets_accumulator():
    acc = [None]
    tools = pc.build_project_copilot_tools(acc)
    assert "play_guide" in tools
    assert "search_docs" in tools
    # reused read-only tools are present
    assert {"get_db_schema", "list_project_assets", "get_asset_details"} <= set(tools)

    result = tools["play_guide"].handler({"sequence_id": "connect"}, None)
    assert json.loads(result)["launched"] == "connect"
    assert acc[0] == "connect"


def test_play_guide_rejects_unknown_sequence():
    acc = [None]
    tools = pc.build_project_copilot_tools(acc)
    result = tools["play_guide"].handler({"sequence_id": "not-real"}, None)
    assert json.loads(result)["status"] == "error"
    assert acc[0] is None


def test_search_docs_tool_delegates(monkeypatch, mocker):
    acc = [None]
    tools = pc.build_project_copilot_tools(acc)
    spy = mocker.patch.object(pc, "search_docs", return_value="DOCS")
    out = tools["search_docs"].handler({"query": "tables"}, None)
    assert out == "DOCS"
    spy.assert_called_once_with("tables")


def test_search_docs_tool_flags_notice_when_degraded(mocker):
    """A degraded search_docs result sets the notice accumulator so the route can
    warn the user the answer wasn't grounded; a healthy result leaves it None."""
    guide_acc, notice_acc = [None], [None]
    tools = pc.build_project_copilot_tools(guide_acc, notice_acc)

    mocker.patch.object(pc, "search_docs", return_value=pc._DOCS_UNAVAILABLE)
    tools["search_docs"].handler({"query": "x"}, None)
    assert notice_acc[0] == "docs_unavailable"

    notice_acc[0] = None
    mocker.patch.object(pc, "search_docs", return_value="# Real doc\nbody")
    tools["search_docs"].handler({"query": "x"}, None)
    assert notice_acc[0] is None


def test_build_input_messages_substitutes_empty_assistant_content():
    """A guide-only turn is persisted with content="" for display. Replaying it
    into the agent must not send an empty assistant text block (Anthropic 400s
    on those) and must keep strict user/assistant alternation intact — even
    when the empty turn sits mid-history, not just at the window's edge."""
    history = [
        {"role": "user", "content": "show me how to connect"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "thanks, now what about auth?"},
        {"role": "assistant", "content": "  "},  # whitespace-only counts as empty
        {"role": "user", "content": "got it"},
    ]
    out = pc._build_input_messages(history)

    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant", "user"]
    for m in out:
        assert m["content"].strip() != ""  # no empty text blocks anywhere

    # Only the empty-content assistant rows were substituted; everything else
    # (including non-empty assistant content) passes through unchanged.
    assert out[0] == history[0]
    assert out[2] == history[2]
    assert out[4] == history[4]
    assert out[1]["content"] == pc._EMPTY_ASSISTANT_PLACEHOLDER
    assert out[3]["content"] == pc._EMPTY_ASSISTANT_PLACEHOLDER


def test_build_input_messages_names_guide_in_placeholder():
    """An empty-content assistant row that HAS a guide_event names the guide in
    its placeholder (so a follow-up "yes, continue" doesn't make the model
    re-launch the same walkthrough), instead of the generic filler text."""
    history = [
        {"role": "user", "content": "show me how to connect"},
        {
            "role": "assistant",
            "content": "",
            "guide_event": {"sequence_id": "connect"},
        },
        {"role": "user", "content": "yes, continue"},
    ]
    out = pc._build_input_messages(history)

    assert out[1]["content"] == "(Launched the connect walkthrough.)"
    assert "connect" in out[1]["content"]

    # No guide_event at all still falls back to the generic placeholder.
    no_guide = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]
    out2 = pc._build_input_messages(no_guide)
    assert out2[1]["content"] == pc._EMPTY_ASSISTANT_PLACEHOLDER


def test_search_docs_tool_flags_notice_when_kb_not_ready(mocker):
    """The _DOCS_NOT_READY sentinel (empty/not-yet-indexed KB) must also flag the
    notice accumulator — same treatment as _DOCS_UNAVAILABLE/_DOCS_NOT_CONFIGURED."""
    guide_acc, notice_acc = [None], [None]
    tools = pc.build_project_copilot_tools(guide_acc, notice_acc)

    mocker.patch.object(pc, "search_docs", return_value=pc._DOCS_NOT_READY)
    tools["search_docs"].handler({"query": "x"}, None)
    assert notice_acc[0] == "docs_unavailable"


# ---------------------------------------------------------------------------
# Injection defense: untrusted-data directive + metadata capping
# ---------------------------------------------------------------------------


def test_system_prompt_contains_untrusted_data_directive():
    """The prompt must tell the model that tool output / project content is
    UNTRUSTED DATA, never to present destructive SQL as a user step, and never
    to relay project-data URLs as links."""
    assert "UNTRUSTED DATA" in pc.SYSTEM_PROMPT
    # Destructive / privilege-changing SQL must never be presented as a step.
    for kw in ("DROP", "TRUNCATE", "DISABLE ROW LEVEL SECURITY", "GRANT"):
        assert kw in pc.SYSTEM_PROMPT, f"missing SQL keyword in directive: {kw}"
    # URLs originating from project data are never relayed as clickable links.
    assert "Never relay" in pc.SYSTEM_PROMPT


def test_truncation_helper_caps_long_metadata():
    """Free-text over the cap is truncated with an ellipsis; short text passes
    through untouched."""
    long = "A" * 5000
    capped = pc._truncate_untrusted(long)
    assert len(capped) == pc._UNTRUSTED_FIELD_MAX_CHARS + 1
    assert capped.endswith("…")
    assert capped.startswith("A" * 10)
    assert pc._truncate_untrusted("short") == "short"


def test_get_asset_details_result_caps_long_untrusted_fields(mocker):
    """A hostile crawled page can stuff kilobytes of injection payload into
    source name / metadata / auto_metadata — the tool result fed to the model
    caps those fields; short fields are untouched."""
    payload = json.dumps(
        {
            "source": {
                "id": "src-1",
                "name": "N" * 3000,
                "metadata": {"note": "ok"},
                "auto_metadata": {"title": "T" * 3000},
            }
        }
    )
    mocker.patch.object(pc, "resolve_get_asset_details", return_value=payload)
    tools = pc.build_project_copilot_tools([None])
    out = json.loads(
        tools["get_asset_details"].handler({"asset_type": "source", "asset_id": "src-1"}, None)
    )

    assert len(out["source"]["name"]) == pc._UNTRUSTED_FIELD_MAX_CHARS + 1
    assert out["source"]["name"].endswith("…")
    assert len(out["source"]["auto_metadata"]["title"]) == pc._UNTRUSTED_FIELD_MAX_CHARS + 1
    assert out["source"]["id"] == "src-1"
    assert out["source"]["metadata"]["note"] == "ok"


def test_list_project_assets_result_caps_long_names(mocker):
    payload = json.dumps(
        {"sources": [{"id": "1", "name": "N" * 2000, "file_type": "url", "status": "ready"}]}
    )
    mocker.patch.object(pc, "resolve_list_project_assets", return_value=payload)
    tools = pc.build_project_copilot_tools([None])
    out = json.loads(tools["list_project_assets"].handler({"asset_type": "sources"}, None))

    name = out["sources"][0]["name"]
    assert len(name) == pc._UNTRUSTED_FIELD_MAX_CHARS + 1
    assert name.endswith("…")
