"""The default thresholds have to cover the window where the query change costs time.

Scoping a vector search to its knowledge base makes a knowledge base that has no
partial index of its own SLOWER: it can no longer stop at the project-wide
index's first ``ef_search`` candidates. Measured 3.6 ms to 80 ms where the
knowledge base held 21% of the embeddings table (12.6k rows) and 129 ms at 30%
(18k rows). A build threshold above those row counts leaves precisely the
knowledge bases that regressed with no way out of it, on a project that will
never look big enough to anyone reading the setting.

So the defaults are pinned against the measurement, not against each other.
"""

from __future__ import annotations

from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY

# The two knowledge bases measured as regressing, in rows.
SMALLEST_MEASURED_REGRESSION = 12_600
LARGEST_MEASURED_REGRESSION = 18_000


def test_the_build_default_is_at_or_below_the_measured_regression_window():
    build = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"]
    assert build.default <= SMALLEST_MEASURED_REGRESSION, (
        "a knowledge base that got slower must be able to reach the threshold: "
        f"default {build.default} leaves the measured "
        f"{SMALLEST_MEASURED_REGRESSION}- and {LARGEST_MEASURED_REGRESSION}-row "
        "knowledge bases on the project-wide index for good"
    )


def test_the_drop_default_keeps_the_hysteresis_below_the_new_build_default():
    build = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"]
    drop = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_DROP_ROWS"]
    # Lowering the build threshold without lowering the drop threshold would
    # invert the pair, and `thresholds()` would then silently substitute half.
    assert drop.default < build.default
    assert drop.default == build.default // 2, (
        "the pair is the documented one (drop at half of build), so the stored "
        "value and the substituted fallback agree"
    )


def test_the_drop_threshold_cannot_be_set_to_an_unreachable_zero():
    """``rows < 0`` is never true, so 0 means "never drop, and re-dispatch forever"."""
    drop = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_DROP_ROWS"]
    assert drop.min == 1


def test_the_build_threshold_description_is_honest_about_the_regression():
    """This copy is what an operator reads when deciding to move the threshold.

    It goes stale last of the four places the measurements live, and it is the
    only one a person choosing a value ever sees, so it has to carry both the
    cost (how slow an un-indexed knowledge base now is) and the reason to pay it
    (the old answer was faster and wrong).
    """
    text = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"].description
    assert "80 ms" in text and "21%" in text, "the corrected regression figure"
    assert "129 ms" in text and "30%" in text
    assert "0.65" in text and "0.70" in text, "the recall the old query shape had"
    assert "0.90" in text, "the partial index is itself approximate"
    assert "9.5 MB" in text and "1,000 embeddings" in text, "the disk consequence"
