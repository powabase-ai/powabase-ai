"""Tests for the compaction settings registry — obsolete knobs stay removed."""

from agentic_project_service.services.settings_registry import _build_registry


def test_obsolete_compaction_knobs_removed():
    reg = _build_registry()
    assert "DEFAULT_COMPACTION_MODEL" not in reg
    assert "COMPACTION_KEEP_LAST_N" not in reg
    # COMPACTION_BUFFER is dominated by the window-proportional buffer for
    # every window above ~162,500, so the knob no longer controls anything.
    assert "COMPACTION_BUFFER" not in reg


def test_no_compaction_knobs_remain():
    # CHARS_PER_TOKEN and COMPACTION_MAX_OUTPUT_TOKENS never reach a
    # get_setting() call anywhere in the codebase (verified by `grep -rn`
    # across the whole repo) — the same "dial that does nothing" defect
    # COMPACTION_BUFFER was removed for above. Moving either UI slider
    # changes nothing, so neither is exposed as a setting.
    reg = _build_registry()
    for key in ("CHARS_PER_TOKEN", "COMPACTION_MAX_OUTPUT_TOKENS"):
        assert key not in reg
    assert "compaction" not in {d.category for d in reg.values()}


def test_compaction_category_label_removed():
    # With no settings left in the "compaction" category, its display
    # label is an orphan — nothing groups under it in get_all_settings().
    from agentic_project_service.services.settings_registry import (
        CATEGORY_META,
    )

    assert "compaction" not in CATEGORY_META
