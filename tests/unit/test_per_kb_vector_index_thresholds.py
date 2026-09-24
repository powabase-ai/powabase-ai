"""The default thresholds encode a decision, so they are pinned where it is written.

Scoping a vector search to its knowledge base can make a knowledge base with no
partial index of its own slower -- it can no longer stop at the project-wide
index's first ``ef_search`` candidates. One fixture measured 3.6 ms to 80 ms at
21% of the embeddings table (12.6k rows) and 129 ms at 30% (18k rows); a later
150,000-embedding one measured 6.2 ms to 50.4 ms at 10k and 4.4 ms to 132.5 ms at
20k; and a third, independently built, measured the new shape 3.6x *faster* at the
same recall with no window at all. So the window is a property of the fixture --
its existence included, because which plan wins is a cost race decided by table
shape rather than by embedding width -- and a default chosen to sit below it would
be chasing a number that does not reproduce.

What decided these defaults is the other side: an index can be built, maintained
on every write, and never scanned, because in some storage layouts the planner
prefers the project-wide index even for a knowledge base that owns one. That was
measured in one layout and could not be reproduced in another, so it is an open
question -- and until it is settled the default stays high enough that almost
nothing crosses it by accident. Lowering it is a per-project setting change, to be
made after confirming with ``pg_stat_all_indexes`` that the index is really
scanned.

The literals are therefore pinned here, with the reasoning, so that moving them is
a deliberate edit against a new measurement rather than a quiet retune.
"""

from __future__ import annotations

import re

from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY

# The knowledge-base sizes measured as regressing, in rows, across two fixtures:
# 12.6k and 18k on the first, 10k and 20k on a later 150,000-embedding one. The
# largest is what this module asserts against, so it has to move when a bigger one
# is measured -- otherwise the assertion below quietly stops meaning anything.
SMALLEST_MEASURED_REGRESSION = 10_000
LARGEST_MEASURED_REGRESSION = 20_000

# The shapes the copy has to carry, whatever the numbers turn out to be and
# however they are worded: a latency, a recall fraction, and an index size.
_LATENCY_FIGURE = re.compile(r"\d[\d,.]* ?ms\b")
_RECALL_FIGURE = re.compile(r"0\.\d+")
_DISK_FIGURE = re.compile(r"\d[\d,.]* ?(?:MB|GB)\b")


def test_the_defaults_are_the_pair_that_was_decided():
    """The literal numbers, pinned once, here.

    Everything else about these two settings is asserted as a relation -- the
    hysteresis, the range -- which is right for properties that hold whatever the
    numbers are. These numbers are a judgement call instead: high enough that a
    project does not start building per-knowledge-base indexes on its first boot
    while it is still unknown whether the planner will scan them. Changing them
    needs the measurement named in the module docstring, and a deliberate edit
    here.
    """
    assert SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"].default == 50_000
    assert SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_DROP_ROWS"].default == 25_000


def test_the_build_default_leaves_the_measured_regression_window_unindexed():
    """The cost of the conservative default, asserted so it cannot be forgotten.

    A knowledge base in the measured window keeps the project-wide index and earns
    no index of its own. That is the accepted trade, not an oversight -- and it is
    the thing to revisit first if the open plan-choice question closes, because
    lowering the threshold is then the whole fix.
    """
    build = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"]
    assert build.default > LARGEST_MEASURED_REGRESSION, (
        "this default is meant to sit above the measured window; if it has been "
        f"lowered to {build.default} on purpose, the module docstring and the "
        "setting's description both need to say why, and this spec should be the "
        "one that made you read them"
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


def test_the_drop_threshold_cannot_be_set_to_zero():
    """At 0 an index is held open until the knowledge base's last embedding is gone.

    Which is not a threshold at all: the write cost of the index is paid on every
    write to the embeddings table, for an index of a handful of rows, and it was
    worse before the drop test included equality -- ``rows < 0`` could never be
    satisfied by any knowledge base, empty or not, while the start-up sweep went
    on re-dispatching it every boot.
    """
    drop = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_DROP_ROWS"]
    assert drop.min == 1


def test_the_build_threshold_description_is_honest_about_the_regression():
    """This copy is what an operator reads when deciding to move the threshold.

    It goes stale last of the four places the measurements live, and it is the
    only one a person choosing a value ever sees, so it has to carry both the
    cost (how slow an un-indexed knowledge base now is) and the reason to pay it
    (the old answer was faster and wrong).

    Pinned by shape rather than by wording. An earlier version of this test named
    nine literal fragments of the copy, so a rephrasing or a fresh measurement
    that changed nothing about what the operator learns broke it -- which trains
    the next person to edit the test rather than to read it. What must not happen
    is one of the three facts quietly going away, and that is what is asserted:
    the two latencies (an un-indexed knowledge base's and an indexed one's), the
    recall the choice trades against, and the disk it costs.
    """
    text = SETTINGS_REGISTRY["VECTOR_PER_KB_INDEX_MIN_ROWS"].description
    assert len(_LATENCY_FIGURE.findall(text)) >= 2, (
        f"the copy has to give both sides of the trade in milliseconds, not one of them: {text!r}"
    )
    assert "slower" in text, (
        "the cost of being below the threshold is that searches got slower, and an "
        f"operator who is not told that cannot weigh it: {text!r}"
    )
    assert _RECALL_FIGURE.search(text), (
        f"the recall figures are the reason to pay that cost: {text!r}"
    )
    assert _DISK_FIGURE.search(text), (
        f"an index of this size per knowledge base is the other cost: {text!r}"
    )
